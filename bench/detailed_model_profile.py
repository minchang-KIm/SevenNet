from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import sevenn._keys as KEY
import sevenn.nn.convolution as conv_mod
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.pair_runtime import prepare_pair_metadata
from sevenn.train.dataload import unlabeled_atoms_to_graph

from local_pair_size_profile import (
    CASE_BASELINE,
    CASE_PAIR,
    DATASETS,
    load_topk_local_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.times_ms: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self.loads: Dict[str, float] = defaultdict(float)

    def reset(self) -> None:
        self.times_ms.clear()
        self.calls.clear()
        self.loads.clear()

    @contextmanager
    def section(self, key: str, load: float | None = None):
        _sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.times_ms[key] += (time.perf_counter() - start) * 1000.0
            self.calls[key] += 1
            if load is not None:
                self.loads[key] += float(load)


def _patch_method(
    obj: Any, attr: str, wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Callable[[], None]:
    original = getattr(obj, attr)
    wrapped = wrapper_factory(original)
    setattr(obj, attr, wrapped)

    def restore():
        setattr(obj, attr, original)

    return restore


def _profile_single_sample(
    atoms,
    *,
    modal: str,
    case: Dict[str, Any],
    repeat: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calc = SevenNetCalculator(
        model='7net-omni',
        modal=modal,
        device=device,
        enable_flash=False,
        enable_pair_execution=case['enable_pair_execution'],
        pair_execution_policy=case['pair_execution_policy'],
    )
    timer = StageTimer(device)
    restore_stack: List[Callable[[], None]] = []

    def timed_call(
        key: str,
        fn: Callable[..., Any],
        *args,
        load: float | None = None,
        **kwargs,
    ):
        with timer.section(key, load=load):
            return fn(*args, **kwargs)

    def wrap_edge_embedding(name: str, module) -> Callable:
        def wrapped(_original):
            def forward(data):
                if (
                    module.pair_execution_policy != 'baseline'
                    and KEY.PAIR_EDGE_VEC in data
                    and KEY.EDGE_PAIR_MAP in data
                    and KEY.EDGE_PAIR_REVERSE in data
                ):
                    pair_rvec = data[KEY.PAIR_EDGE_VEC]
                    if KEY.PAIR_EDGE_FORWARD_INDEX in data and KEY.EDGE_VEC in data:
                        pair_rvec = timed_call(
                            f'{name}.pair_vec_select_ms',
                            data[KEY.EDGE_VEC].index_select,
                            0,
                            data[KEY.PAIR_EDGE_FORWARD_INDEX],
                            load=float(data[KEY.PAIR_EDGE_FORWARD_INDEX].numel()),
                        )
                        data[KEY.PAIR_EDGE_VEC] = pair_rvec
                    pair_r = timed_call(
                        f'{name}.edge_length_norm_ms',
                        torch.linalg.norm,
                        pair_rvec,
                        dim=-1,
                        load=float(pair_rvec.shape[0]),
                    )
                    pair_basis = timed_call(
                        f'{name}.radial_basis_ms',
                        module.basis_function,
                        pair_r,
                        load=float(pair_r.numel()),
                    )
                    cutoff = timed_call(
                        f'{name}.cutoff_ms',
                        module.cutoff_function,
                        pair_r,
                        load=float(pair_r.numel()),
                    )
                    pair_embedding = timed_call(
                        f'{name}.radial_combine_ms',
                        torch.mul,
                        pair_basis,
                        cutoff.unsqueeze(-1),
                        load=float(pair_basis.numel()),
                    )
                    pair_attr = timed_call(
                        f'{name}.spherical_harmonics_ms',
                        module.spherical,
                        pair_rvec,
                        load=float(pair_rvec.shape[0]),
                    )
                    data[KEY.PAIR_EDGE_EMBEDDING] = pair_embedding
                    data[KEY.PAIR_EDGE_ATTR] = pair_attr
                    data[KEY.EDGE_LENGTH] = timed_call(
                        f'{name}.edge_length_expand_ms',
                        pair_r.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    edge_embedding = timed_call(
                        f'{name}.edge_embedding_expand_ms',
                        pair_embedding.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    edge_attr = timed_call(
                        f'{name}.edge_attr_expand_ms',
                        pair_attr.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    reverse_mask = data[KEY.EDGE_PAIR_REVERSE].to(edge_attr.dtype).unsqueeze(-1)
                    sign = timed_call(
                        f'{name}.reverse_sign_ms',
                        lambda: 1.0
                        + reverse_mask
                        * (module.reverse_sh_sign.to(edge_attr.dtype).unsqueeze(0) - 1.0),
                        load=float(edge_attr.numel()),
                    )
                    data[KEY.EDGE_EMBEDDING] = edge_embedding
                    data[KEY.EDGE_ATTR] = timed_call(
                        f'{name}.edge_attr_sign_apply_ms',
                        torch.mul,
                        edge_attr,
                        sign,
                        load=float(edge_attr.numel()),
                    )
                    return data

                rvec = data[KEY.EDGE_VEC]
                if KEY.EDGE_LENGTH in data:
                    r = data[KEY.EDGE_LENGTH]
                else:
                    r = timed_call(
                        f'{name}.edge_length_norm_ms',
                        torch.linalg.norm,
                        rvec,
                        dim=-1,
                        load=float(rvec.shape[0]),
                    )
                    data[KEY.EDGE_LENGTH] = r

                basis = timed_call(
                    f'{name}.radial_basis_ms',
                    module.basis_function,
                    r,
                    load=float(r.numel()),
                )
                cutoff = timed_call(
                    f'{name}.cutoff_ms',
                    module.cutoff_function,
                    r,
                    load=float(r.numel()),
                )
                data[KEY.EDGE_EMBEDDING] = timed_call(
                    f'{name}.radial_combine_ms',
                    torch.mul,
                    basis,
                    cutoff.unsqueeze(-1),
                    load=float(basis.numel()),
                )
                data[KEY.EDGE_ATTR] = timed_call(
                    f'{name}.spherical_harmonics_ms',
                    module.spherical,
                    rvec,
                    load=float(rvec.shape[0]),
                )
                return data

            return forward

        return wrapped

    def wrap_convolution(name: str, module) -> Callable:
        def wrapped(_original):
            def forward(data):
                with timer.section(f'{name}.total_ms'):
                    x = data[module.key_x]

                    if module.is_parallel:
                        x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

                    if module._use_pair_execution(data):
                        pair_input = data[KEY.PAIR_EDGE_EMBEDDING]
                        pair_weight = timed_call(
                            f'{name}.weight_nn_ms',
                            module.weight_nn,
                            pair_input,
                            load=float(pair_input.shape[0]),
                        )
                        if module.pair_execution_policy == 'full' and module.fuse_reduction:
                            edge_index = data[module.key_edge_idx]
                            pair_forward_index = data[KEY.PAIR_EDGE_FORWARD_INDEX]
                            pair_backward_index = data[KEY.PAIR_EDGE_BACKWARD_INDEX]
                            pair_has_reverse = data[KEY.PAIR_EDGE_HAS_REVERSE]
                            edge_src = edge_index[1]
                            edge_dst = edge_index[0]

                            src_forward = timed_call(
                                f'{name}.forward_src_index_ms',
                                edge_src.index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            dst_forward = timed_call(
                                f'{name}.forward_dst_index_ms',
                                edge_dst.index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            x_forward = timed_call(
                                f'{name}.forward_src_gather_ms',
                                x.index_select,
                                0,
                                src_forward,
                                load=float(src_forward.numel()),
                            )
                            filter_forward = timed_call(
                                f'{name}.forward_filter_gather_ms',
                                data[module.key_filter].index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            msg_forward = timed_call(
                                f'{name}.forward_message_tp_ms',
                                module.convolution,
                                x_forward,
                                filter_forward,
                                pair_weight,
                                load=float(pair_forward_index.numel()),
                            )
                            out = timed_call(
                                f'{name}.forward_aggregation_ms',
                                conv_mod.message_gather,
                                x,
                                dst_forward,
                                msg_forward,
                                load=float(dst_forward.numel()),
                            )

                            rev_index = pair_backward_index[pair_has_reverse]
                            if rev_index.numel() > 0:
                                rev_src = timed_call(
                                    f'{name}.reverse_src_index_ms',
                                    edge_src.index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                rev_dst = timed_call(
                                    f'{name}.reverse_dst_index_ms',
                                    edge_dst.index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                x_reverse = timed_call(
                                    f'{name}.reverse_src_gather_ms',
                                    x.index_select,
                                    0,
                                    rev_src,
                                    load=float(rev_src.numel()),
                                )
                                filter_reverse = timed_call(
                                    f'{name}.reverse_filter_gather_ms',
                                    data[module.key_filter].index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                reverse_weight = timed_call(
                                    f'{name}.reverse_weight_select_ms',
                                    pair_weight.index_select,
                                    0,
                                    torch.nonzero(pair_has_reverse, as_tuple=False).flatten(),
                                    load=float(rev_index.numel()),
                                )
                                msg_reverse = timed_call(
                                    f'{name}.reverse_message_tp_ms',
                                    module.convolution,
                                    x_reverse,
                                    filter_reverse,
                                    reverse_weight,
                                    load=float(rev_index.numel()),
                                )
                                out = timed_call(
                                    f'{name}.reverse_aggregation_ms',
                                    lambda current_out: current_out
                                    + conv_mod.message_gather(x, rev_dst, msg_reverse),
                                    out,
                                    load=float(rev_dst.numel()),
                                )
                            x = out
                        else:
                            weight = timed_call(
                                f'{name}.weight_expand_ms',
                                pair_weight.index_select,
                                0,
                                data[KEY.EDGE_PAIR_MAP],
                                load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                            )
                            edge_src = data[module.key_edge_idx][1]
                            edge_dst = data[module.key_edge_idx][0]
                            x_src = timed_call(
                                f'{name}.edge_src_gather_ms',
                                x.index_select,
                                0,
                                edge_src,
                                load=float(edge_src.numel()),
                            )
                            message = timed_call(
                                f'{name}.message_tp_ms',
                                module.convolution,
                                x_src,
                                data[module.key_filter],
                                weight,
                                load=float(edge_src.numel()),
                            )
                            x = timed_call(
                                f'{name}.aggregation_ms',
                                conv_mod.message_gather,
                                x,
                                edge_dst,
                                message,
                                load=float(edge_dst.numel()),
                            )
                    else:
                        weight_input = data[module.key_weight_input]
                        weight = timed_call(
                            f'{name}.weight_nn_ms',
                            module.weight_nn,
                            weight_input,
                            load=float(weight_input.shape[0]),
                        )
                        edge_src = data[module.key_edge_idx][1]
                        edge_dst = data[module.key_edge_idx][0]
                        x_src = timed_call(
                            f'{name}.edge_src_gather_ms',
                            x.index_select,
                            0,
                            edge_src,
                            load=float(edge_src.numel()),
                        )
                        message = timed_call(
                            f'{name}.message_tp_ms',
                            module.convolution,
                            x_src,
                            data[module.key_filter],
                            weight,
                            load=float(edge_src.numel()),
                        )
                        x = timed_call(
                            f'{name}.aggregation_ms',
                            conv_mod.message_gather,
                            x,
                            edge_dst,
                            message,
                            load=float(edge_dst.numel()),
                        )

                    x = timed_call(f'{name}.denominator_ms', x.div, module.denominator)
                    if module.is_parallel:
                        x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
                    data[module.key_x] = x
                    return data

            return forward

        return wrapped

    def wrap_top_level_modules() -> None:
        for name, module in calc.model.named_children():
            if name == 'edge_embedding':
                restore_stack.append(_patch_method(module, 'forward', wrap_edge_embedding(name, module)))
                continue
            if 'convolution' in name:
                restore_stack.append(_patch_method(module, 'forward', wrap_convolution(name, module)))
                continue

            if name in {'onehot_idx_to_onehot', 'one_hot_modality', 'onehot_to_feature_x'}:
                key = 'top_input_embedding_ms'
            elif (
                'self_connection' in name
                or 'self_interaction' in name
                or 'equivariant_gate' in name
            ):
                key = 'top_interaction_other_ms'
            elif name in {
                'reduce_input_to_hidden',
                'reduce_hidden_to_energy',
                'rescale_atomic_energy',
                'reduce_total_enegy',
            }:
                key = 'top_readout_ms'
            elif name == 'force_output':
                key = 'top_force_output_ms'
            else:
                key = 'top_other_ms'

            restore_stack.append(
                _patch_method(
                    module,
                    'forward',
                    lambda original, stage_key=key: (
                        lambda *args, **kwargs: timed_call(stage_key, original, *args, **kwargs)
                    ),
                )
            )

    wrap_top_level_modules()

    def build_data() -> AtomGraphData:
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(
                atoms,
                calc.cutoff,
                with_shift=calc.pair_execution_config['resolved_policy'] != 'baseline',
            )
        )
        if calc.modal:
            data[KEY.DATA_MODALITY] = calc.modal
        data, calc._pair_topology_cache = prepare_pair_metadata(
            data,
            calc.pair_execution_config,
            cache_state=calc._pair_topology_cache,
            num_atoms=len(atoms),
        )
        data.to(device)  # type: ignore[arg-type]
        return data

    try:
        warmup = build_data()
        calc.model(warmup)
        _sync(device)
        timer.reset()

        for _ in range(repeat):
            data = build_data()
            with timer.section('model_total_ms'):
                output = calc.model(data)

        output_energy = float(output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item())
        summary = {
            'resolved_policy': calc.pair_execution_config['resolved_policy'],
            'energy': output_energy,
            'num_edges_runtime': int(output[KEY.EDGE_IDX].shape[1]),
        }
        for key, value in timer.times_ms.items():
            summary[key] = value / repeat
        for key, value in timer.loads.items():
            summary[f'{key}_load'] = value / repeat
        for key, value in timer.calls.items():
            summary[f'{key}_calls'] = value / repeat

        stage_rows: List[Dict[str, Any]] = []
        for key in sorted(timer.times_ms):
            stage_rows.append(
                {
                    'stage': key,
                    'time_ms': timer.times_ms[key] / repeat,
                    'calls': timer.calls[key] / repeat,
                    'load': timer.loads.get(key, 0.0) / repeat,
                }
            )
        return summary, stage_rows
    finally:
        for restore in reversed(restore_stack):
            restore()
        del calc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _aggregate_stage_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped: Dict[str, List[str]] = {
        'edge_length_norm_ms': ['edge_embedding.edge_length_norm_ms'],
        'radial_basis_ms': ['edge_embedding.radial_basis_ms'],
        'cutoff_ms': ['edge_embedding.cutoff_ms'],
        'radial_combine_ms': ['edge_embedding.radial_combine_ms'],
        'spherical_harmonics_ms': ['edge_embedding.spherical_harmonics_ms'],
        'conv_weight_nn_ms': ['weight_nn_ms'],
        'conv_src_gather_ms': ['src_gather_ms'],
        'conv_message_tp_ms': ['message_tp_ms'],
        'conv_aggregation_ms': ['aggregation_ms'],
        'conv_denominator_ms': ['denominator_ms'],
        'top_input_embedding_ms': ['top_input_embedding_ms'],
        'top_interaction_other_ms': ['top_interaction_other_ms'],
        'top_readout_ms': ['top_readout_ms'],
        'top_force_output_ms': ['top_force_output_ms'],
        'model_total_ms': ['model_total_ms'],
    }
    out_rows: List[Dict[str, Any]] = []
    key_cols = ['dataset', 'sample_id', 'case', 'natoms']
    for key, frame in rows.groupby(key_cols):
        record = dict(zip(key_cols, key))
        for out_key, patterns in grouped.items():
            mask = np.zeros(len(frame), dtype=bool)
            for pattern in patterns:
                mask |= frame['stage'].str.contains(pattern, regex=False).to_numpy()
            subset = frame[mask]
            record[out_key] = float(subset['time_ms'].sum())
            record[f'{out_key}_load'] = float(subset['load'].sum())
        out_rows.append(record)
    return pd.DataFrame(out_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=[spec.name for spec in DATASETS],
    )
    parser.add_argument(
        '--cases',
        nargs='*',
        default=['baseline', 'pair_full'],
        choices=['baseline', 'pair_full'],
    )
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args(argv)

    case_map = {
        'baseline': CASE_BASELINE,
        'pair_full': CASE_PAIR,
    }
    selected_specs = [spec for spec in DATASETS if spec.name in set(args.datasets)]
    output_dir = args.output_dir.resolve()
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    for spec in selected_specs:
        sample = load_topk_local_samples(spec, top_k=1)[0]
        for case_name in args.cases:
            case = case_map[case_name]
            summary, stages = _profile_single_sample(
                sample['atoms'],
                modal=spec.modal,
                case=case,
                repeat=args.repeat,
            )
            row = {
                'dataset': spec.name,
                'sample_id': sample['sample_id'],
                'natoms': sample['natoms'],
                'case': case['case'],
                **summary,
            }
            summary_rows.append(row)
            for stage in stages:
                stage_rows.append(
                    {
                        'dataset': spec.name,
                        'sample_id': sample['sample_id'],
                        'natoms': sample['natoms'],
                        'case': case['case'],
                        **stage,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    stage_df = pd.DataFrame(stage_rows)
    aggregate_df = _aggregate_stage_rows(stage_df)
    summary_df.to_csv(metrics_dir / 'summary.csv', index=False)
    stage_df.to_csv(metrics_dir / 'stage_breakdown_long.csv', index=False)
    aggregate_df.to_csv(metrics_dir / 'stage_breakdown_aggregate.csv', index=False)

    report_lines = [
        '# Detailed Model Stage Profiling',
        '',
        f'- Datasets: {", ".join(spec.name for spec in selected_specs)}',
        f'- Cases: {", ".join(args.cases)}',
        f'- Repeat: `{args.repeat}`',
        '',
        '## Outputs',
        '',
        '- `metrics/summary.csv`',
        '- `metrics/stage_breakdown_long.csv`',
        '- `metrics/stage_breakdown_aggregate.csv`',
        '',
        '## Notes',
        '',
        '- This profiler is intrusive: it wraps model internals and synchronizes around each stage.',
        '- Use it for stage decomposition, not for absolute end-to-end latency claims.',
        '- The aggregate table sums across all five convolution blocks for shared stage names.',
    ]
    (output_dir / 'summary.md').write_text('\n'.join(report_lines) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
