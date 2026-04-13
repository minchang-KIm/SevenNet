from __future__ import annotations

import argparse
import gc
import heapq
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from ase import Atoms

import sevenn._keys as KEY
import sevenn.nn.convolution as conv_mod
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.pair_runtime import prepare_pair_metadata
from sevenn.train.dataload import unlabeled_atoms_to_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / 'datasets' / 'raw'


@dataclass(frozen=True)
class LocalDatasetSpec:
    name: str
    modal: str
    loader: str
    root: Path
    size_column: str
    description: str


DATASETS: Sequence[LocalDatasetSpec] = (
    LocalDatasetSpec(
        name='mptrj',
        modal='mpa',
        loader='nimashoghi',
        root=DATASET_ROOT / 'hf' / 'mptrj' / 'data',
        size_column='num_atoms',
        description='MPtrj train parquet mirror',
    ),
    LocalDatasetSpec(
        name='md22_double_walled_nanotube',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'md22_double_walled_nanotube' / 'co',
        size_column='nsites',
        description='MD22 double-walled nanotube',
    ),
    LocalDatasetSpec(
        name='spice_2023',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'spice_2023' / 'co',
        size_column='nsites',
        description='SPICE 2023',
    ),
    LocalDatasetSpec(
        name='md22_stachyose',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'md22_stachyose' / 'co',
        size_column='nsites',
        description='MD22 stachyose',
    ),
    LocalDatasetSpec(
        name='ani1x',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'ani1x' / 'co',
        size_column='nsites',
        description='ANI-1x mirror',
    ),
    LocalDatasetSpec(
        name='rmd17',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'rmd17' / 'co',
        size_column='nsites',
        description='rMD17 mirror',
    ),
    LocalDatasetSpec(
        name='iso17',
        modal='spice',
        loader='colabfit',
        root=DATASET_ROOT / 'hf' / 'iso17' / 'co',
        size_column='nsites',
        description='ISO17 mirror',
    ),
)


CASE_BASELINE = {
    'case': 'e3nn_baseline',
    'enable_pair_execution': False,
    'pair_execution_policy': None,
}
CASE_PAIR = {
    'case': 'e3nn_pair_full',
    'enable_pair_execution': True,
    'pair_execution_policy': 'full',
}


def _as_numeric_array(value: Any, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != object:
        return np.asarray(arr, dtype=dtype)
    return np.asarray([np.asarray(v, dtype=dtype) for v in value], dtype=dtype)


def _as_bool_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray([bool(v) for v in value], dtype=bool)
    return np.asarray(arr, dtype=bool)


def atoms_from_colabfit_row(row: pd.Series) -> Atoms:
    return Atoms(
        numbers=np.asarray(row['atomic_numbers'], dtype=np.int64),
        positions=_as_numeric_array(row['positions']),
        cell=_as_numeric_array(row['cell']),
        pbc=_as_bool_array(row['pbc']),
    )


def atoms_from_nimashoghi_row(row: pd.Series) -> Atoms:
    return Atoms(
        numbers=np.asarray(row['numbers'], dtype=np.int64),
        positions=_as_numeric_array(row['positions']),
        cell=_as_numeric_array(row['cell']),
        pbc=_as_bool_array(row['pbc']),
    )


ROW_ADAPTERS: Dict[str, Callable[[pd.Series], Atoms]] = {
    'colabfit': atoms_from_colabfit_row,
    'nimashoghi': atoms_from_nimashoghi_row,
}


FULL_COLUMNS: Dict[str, Sequence[str]] = {
    'colabfit': (
        'configuration_id',
        'names',
        'nsites',
        'atomic_numbers',
        'positions',
        'cell',
        'pbc',
    ),
    'nimashoghi': (
        'mp_id',
        'filename',
        'num_atoms',
        'numbers',
        'positions',
        'cell',
        'pbc',
    ),
}


def _top_candidates_in_parquet(
    path: Path, size_column: str, top_k: int
) -> List[tuple[int, int, int, Path]]:
    parquet = pq.ParquetFile(path)
    heap: List[tuple[int, int, int, Path]] = []
    for row_group_idx in range(parquet.num_row_groups):
        column = parquet.read_row_group(row_group_idx, columns=[size_column]).column(0)
        for row_idx, value in enumerate(column.to_pylist()):
            size = int(value)
            item = (size, row_group_idx, row_idx, path)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif size > heap[0][0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda item: item[0], reverse=True)


def _rows_from_parquet(
    path: Path,
    candidates: Sequence[tuple[int, int, int, Path]],
    columns: Sequence[str],
) -> List[pd.Series]:
    parquet = pq.ParquetFile(path)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for _, row_group_idx, row_idx, _ in candidates:
        grouped[row_group_idx].append(row_idx)

    rows: List[pd.Series] = []
    for row_group_idx, row_indices in grouped.items():
        table = parquet.read_row_group(row_group_idx, columns=list(columns))
        frame = table.to_pandas()
        for row_idx in row_indices:
            rows.append(frame.iloc[row_idx])
    return rows


def load_topk_local_samples(spec: LocalDatasetSpec, top_k: int) -> List[Dict[str, Any]]:
    paths = sorted(spec.root.glob('*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No parquet files found under {spec.root}')

    candidates: List[tuple[int, int, int, Path]] = []
    for path in paths:
        candidates.extend(_top_candidates_in_parquet(path, spec.size_column, top_k))
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:top_k]

    adapter = ROW_ADAPTERS[spec.loader]
    columns = FULL_COLUMNS[spec.loader]
    by_path: Dict[Path, List[tuple[int, int, int, Path]]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate[3]].append(candidate)

    selected: List[Dict[str, Any]] = []
    for path, path_candidates in by_path.items():
        for row in _rows_from_parquet(path, path_candidates, columns):
            atoms = adapter(row)
            if spec.loader == 'colabfit':
                raw_name = row.get('names')
                if isinstance(raw_name, np.ndarray) and len(raw_name) > 0:
                    sample_name = str(raw_name[0])
                else:
                    sample_name = str(row.get('configuration_id'))
                natoms = int(row['nsites'])
            else:
                sample_name = str(row.get('filename') or row.get('mp_id'))
                natoms = int(row['num_atoms'])
            selected.append(
                {
                    'sample_id': sample_name,
                    'natoms': natoms,
                    'atoms': atoms,
                }
            )

    return sorted(selected, key=lambda item: item['natoms'], reverse=True)[:top_k]


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_sample(
    atoms: Atoms,
    *,
    modal: str,
    case: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calc = SevenNetCalculator(
        model='7net-omni',
        modal=modal,
        device=device,
        enable_flash=False,
        enable_pair_execution=case['enable_pair_execution'],
        pair_execution_policy=case['pair_execution_policy'],
    )
    try:
        _sync(device)
        cold_start = time.perf_counter()
        calc.calculate(atoms)
        _sync(device)
        cold_ms = (time.perf_counter() - cold_start) * 1000.0

        timings: List[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            calc.calculate(atoms)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)

        return {
            'cold_ms': cold_ms,
            'steady_median_ms': float(np.median(timings)),
            'steady_p95_ms': float(np.quantile(np.asarray(timings), 0.95)),
            'energy': float(calc.results['energy']),
            'forces': np.asarray(calc.results['forces']),
            'resolved_policy': calc.pair_execution_config['resolved_policy'],
        }
    finally:
        del calc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def build_graph_and_pair_features(
    atoms: Atoms,
    *,
    cutoff: float,
    pair_enabled: bool,
    policy: str,
) -> Dict[str, Any]:
    graph = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, cutoff, with_shift=pair_enabled)
    )
    pair_cfg = {
        'use': pair_enabled,
        'policy': policy,
        'resolved_policy': policy if pair_enabled else 'baseline',
        'use_topology_cache': True,
        'fuse_reduction': True,
    }
    graph, _ = prepare_pair_metadata(
        graph,
        pair_cfg,
        cache_state={},
        num_atoms=len(atoms),
    )

    num_atoms = int(graph[KEY.NUM_ATOMS].item())
    num_edges = int(graph[KEY.EDGE_IDX].shape[1])
    num_pairs = (
        int(graph[KEY.PAIR_EDGE_FORWARD_INDEX].numel())
        if KEY.PAIR_EDGE_FORWARD_INDEX in graph
        else num_edges
    )
    num_conv_layers = 5
    reverse_fraction = (
        float(graph[KEY.PAIR_EDGE_HAS_REVERSE].float().mean().item())
        if KEY.PAIR_EDGE_HAS_REVERSE in graph
        else float('nan')
    )
    return {
        'natoms': num_atoms,
        'num_edges': num_edges,
        'num_pairs': num_pairs,
        'avg_neighbors_directed': num_edges / max(num_atoms, 1),
        'pair_reuse_factor': num_edges / max(num_pairs, 1),
        'reverse_pair_fraction': reverse_fraction,
        'baseline_edge_embedding_evals': num_edges,
        'pair_edge_embedding_evals': num_pairs,
        'baseline_weight_nn_rows_total': num_edges * num_conv_layers,
        'pair_weight_nn_rows_total': num_pairs * num_conv_layers,
        'baseline_tp_rows_total': num_edges * num_conv_layers,
        'pair_tp_rows_total': num_edges * num_conv_layers,
        'baseline_scatter_rows_total': num_edges * num_conv_layers,
        'pair_scatter_rows_total': num_edges * num_conv_layers,
    }


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.times_ms: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self.loads: Dict[str, float] = defaultdict(float)

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


def _patch_method(obj: Any, attr: str, wrapper_factory: Callable[[Callable], Callable]):
    original = getattr(obj, attr)
    wrapped = wrapper_factory(original)
    setattr(obj, attr, wrapped)

    def restore():
        setattr(obj, attr, original)

    return restore


def profile_sample(
    atoms: Atoms,
    *,
    modal: str,
    case: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
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

    def wrap_top_level_modules() -> None:
        for name, module in calc.model.named_children():
            if name == 'edge_embedding':
                key = 'top_edge_embedding_ms'
            elif 'convolution' in name:
                key = 'top_convolution_blocks_ms'
            elif (
                'self_connection' in name
                or 'self_interaction' in name
                or 'equivariant_gate' in name
            ):
                key = 'top_interaction_other_ms'
            elif name in {
                'onehot_idx_to_onehot',
                'one_hot_modality',
                'onehot_to_feature_x',
            }:
                key = 'top_input_embedding_ms'
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
                        lambda *args, **kwargs: _timed_call(
                            timer, stage_key, original, *args, **kwargs
                        )
                    ),
                )
            )

    def wrap_conv_internals() -> None:
        for name, module in calc.model.named_children():
            if 'convolution' not in name:
                continue

            restore_stack.append(
                _patch_method(
                    module.weight_nn,  # type: ignore[arg-type]
                    'forward',
                    lambda original: (
                        lambda x, *args, **kwargs: _timed_call(
                            timer,
                            'conv_weight_nn_ms',
                            original,
                            x,
                            *args,
                            __load=float(x.shape[0]) if hasattr(x, 'shape') else None,
                            **kwargs,
                        )
                    ),
                )
            )
            restore_stack.append(
                _patch_method(
                    module.convolution,  # type: ignore[arg-type]
                    'forward',
                    lambda original: (
                        lambda x, edge_filter, weight, *args, **kwargs: _timed_call(
                            timer,
                            'conv_tensor_product_ms',
                            original,
                            x,
                            edge_filter,
                            weight,
                            *args,
                            __load=float(x.shape[0]),
                            **kwargs,
                        )
                    ),
                )
            )

        original_message_gather = conv_mod.message_gather

        def wrapped_message_gather(node_features, edge_dst, message):
            with timer.section('conv_scatter_ms', load=float(edge_dst.numel())):
                return original_message_gather(node_features, edge_dst, message)

        conv_mod.message_gather = wrapped_message_gather

        def restore():
            conv_mod.message_gather = original_message_gather

        restore_stack.append(restore)

    def _timed_call(
        timer_obj: StageTimer,
        key: str,
        original: Callable,
        *args,
        __load: float | None = None,
        **kwargs,
    ):
        with timer_obj.section(key, load=__load):
            return original(*args, **kwargs)

    wrap_top_level_modules()
    wrap_conv_internals()

    try:
        with timer.section('graph_build_ms'):
            data = AtomGraphData.from_numpy_dict(
                unlabeled_atoms_to_graph(
                    atoms,
                    calc.cutoff,
                    with_shift=calc.pair_execution_config['resolved_policy'] != 'baseline',
                )
            )
        if calc.modal:
            data[KEY.DATA_MODALITY] = calc.modal

        with timer.section('pair_metadata_ms'):
            data, calc._pair_topology_cache = prepare_pair_metadata(
                data,
                calc.pair_execution_config,
                cache_state=calc._pair_topology_cache,
                num_atoms=len(atoms),
            )

        with timer.section('device_transfer_ms'):
            data.to(device)  # type: ignore[arg-type]

        # warm-up for topology cache + CUDA kernels
        calc.model(data)
        _sync(device)

        # rebuild fresh graph so stage timings reflect the same steady-state path
        with timer.section('graph_build_ms'):
            data = AtomGraphData.from_numpy_dict(
                unlabeled_atoms_to_graph(
                    atoms,
                    calc.cutoff,
                    with_shift=calc.pair_execution_config['resolved_policy'] != 'baseline',
                )
            )
        if calc.modal:
            data[KEY.DATA_MODALITY] = calc.modal
        with timer.section('pair_metadata_ms'):
            data, calc._pair_topology_cache = prepare_pair_metadata(
                data,
                calc.pair_execution_config,
                cache_state=calc._pair_topology_cache,
                num_atoms=len(atoms),
            )
        with timer.section('device_transfer_ms'):
            data.to(device)  # type: ignore[arg-type]

        with timer.section('model_total_ms'):
            output = calc.model(data)

        total_ms = timer.times_ms['model_total_ms']
        stage_rows = {
            key: value
            for key, value in timer.times_ms.items()
            if key != 'model_total_ms'
        }
        summary = {
            'resolved_policy': calc.pair_execution_config['resolved_policy'],
            'energy': float(output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()),
            'num_edges_runtime': int(output[KEY.EDGE_IDX].shape[1]),
            'model_total_ms': total_ms,
        }
        for key, value in timer.times_ms.items():
            summary[key] = value
        for key, value in timer.loads.items():
            summary[f'{key}_load'] = value
        for key, value in timer.calls.items():
            summary[f'{key}_calls'] = value
        return summary, stage_rows
    finally:
        for restore in reversed(restore_stack):
            restore()
        del calc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_speedup_vs_feature(rows: pd.DataFrame, feature: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(rows[feature], rows['steady_speedup_baseline_over_pair'])
    for _, row in rows.iterrows():
        ax.annotate(
            row['dataset'],
            (row[feature], row['steady_speedup_baseline_over_pair']),
            fontsize=7,
            alpha=0.8,
        )
    ax.axhline(1.0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel(feature)
    ax.set_ylabel('baseline / pair_full')
    ax.set_title(f'Speedup vs {feature}')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_profile_stacked(rows: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    stage_cols = [
        'graph_build_ms',
        'pair_metadata_ms',
        'device_transfer_ms',
        'top_input_embedding_ms',
        'top_edge_embedding_ms',
        'top_interaction_other_ms',
        'top_convolution_blocks_ms',
        'top_readout_ms',
        'top_force_output_ms',
    ]
    df = rows[['dataset', 'case'] + stage_cols].copy()
    df['label'] = df['dataset'] + ':' + df['case']
    bottom = np.zeros(len(df), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in stage_cols:
        values = df[col].fillna(0.0).to_numpy(dtype=np.float64)
        ax.bar(df['label'], values, bottom=bottom, label=col)
        bottom += values
    ax.set_ylabel('ms')
    ax.set_title('Steady-State Stage Breakdown')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _analysis_text(
    sample_frame: pd.DataFrame,
    profile_frame: pd.DataFrame,
    corr_nat: float,
    corr_edges: float,
) -> str:
    lines = [
        '# Local Size Effect + Stage Profiling Analysis',
        '',
        f'Run date: `{pd.Timestamp.now().isoformat()}`',
        '',
        '## Goal',
        '',
        'Validate whether larger graphs consistently benefit more from the current pair-execution full path,',
        'using only already-downloaded local datasets, and break down stage-wise runtime/load.',
        '',
        '## Main Findings',
        '',
        f'- Spearman correlation between speedup and `natoms`: `{corr_nat:.3f}`',
        f'- Spearman correlation between speedup and `num_edges`: `{corr_edges:.3f}`',
        '',
    ]

    best = sample_frame.sort_values('steady_speedup_baseline_over_pair', ascending=False)
    worst = sample_frame.sort_values('steady_speedup_baseline_over_pair', ascending=True)
    lines.append('### Largest Wins')
    lines.append('')
    for _, row in best.head(5).iterrows():
        lines.append(
            f"- `{row['dataset']}` / `{row['sample_id']}`: "
            f"`{row['natoms']}` atoms, `{row['num_edges']}` edges, "
            f"speedup `{row['steady_speedup_baseline_over_pair']:.3f}x`."
        )

    lines.extend(['', '### Largest Losses', ''])
    for _, row in worst.head(5).iterrows():
        lines.append(
            f"- `{row['dataset']}` / `{row['sample_id']}`: "
            f"`{row['natoms']}` atoms, `{row['num_edges']}` edges, "
            f"speedup `{row['steady_speedup_baseline_over_pair']:.3f}x`."
        )

    if not profile_frame.empty:
        lines.extend(['', '## Stage Profiling Highlights', ''])
        for dataset in profile_frame['dataset'].drop_duplicates():
            subset = profile_frame[profile_frame['dataset'] == dataset]
            if {'e3nn_baseline', 'e3nn_pair_full'} - set(subset['case']):
                continue
            base = subset[subset['case'] == 'e3nn_baseline'].iloc[0]
            pair = subset[subset['case'] == 'e3nn_pair_full'].iloc[0]
            lines.append(
                f"- `{dataset}`: model_total `{base['model_total_ms']:.2f} -> {pair['model_total_ms']:.2f}` ms, "
                f"edge_embedding `{base['top_edge_embedding_ms']:.2f} -> {pair['top_edge_embedding_ms']:.2f}` ms, "
                f"conv_weight_nn `{base['conv_weight_nn_ms']:.2f} -> {pair['conv_weight_nn_ms']:.2f}` ms, "
                f"conv_tensor_product `{base['conv_tensor_product_ms']:.2f} -> {pair['conv_tensor_product_ms']:.2f}` ms."
            )

    lines.extend(
        [
            '',
            '## Interpretation',
            '',
            '- If speedup tracks `num_edges` more strongly than `natoms`, the current implementation is behaving like an edge-load optimization rather than an atom-count optimization.',
            '- The load metrics explain why: pair execution reduces edge embedding and weight_nn rows from `num_edges` to `num_pairs`, but TP/scatter rows remain `num_edges`.',
            '- Therefore large wins require graphs where saved geometry/weight work dominates indexing and unchanged TP/scatter work.',
        ]
    )
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=[spec.name for spec in DATASETS],
    )
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument(
        '--profile-datasets',
        nargs='*',
        default=['mptrj', 'md22_double_walled_nanotube', 'spice_2023', 'ani1x', 'rmd17'],
    )
    args = parser.parse_args(argv)

    selected_specs = [spec for spec in DATASETS if spec.name in set(args.datasets)]
    if not selected_specs:
        raise ValueError('No datasets selected')

    output_dir = args.output_dir.resolve()
    metrics_dir = output_dir / 'metrics'
    plots_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []

    for spec in selected_specs:
        samples = load_topk_local_samples(spec, top_k=args.top_k)
        dataset_rows.append(
            {
                'dataset': spec.name,
                'description': spec.description,
                'modal': spec.modal,
                'num_samples': len(samples),
                'max_atoms': max(sample['natoms'] for sample in samples),
            }
        )

        feature_calc = SevenNetCalculator(
            model='7net-omni',
            modal=spec.modal,
            device='cpu',
            enable_flash=False,
            enable_pair_execution=True,
            pair_execution_policy='full',
        )
        cutoff = feature_calc.cutoff
        del feature_calc
        gc.collect()

        for sample in samples:
            baseline = benchmark_sample(
                sample['atoms'], modal=spec.modal, case=CASE_BASELINE, repeat=args.repeat
            )
            pair = benchmark_sample(
                sample['atoms'], modal=spec.modal, case=CASE_PAIR, repeat=args.repeat
            )
            feats = build_graph_and_pair_features(
                sample['atoms'],
                cutoff=cutoff,
                pair_enabled=True,
                policy='full',
            )
            row = {
                'dataset': spec.name,
                'description': spec.description,
                'modal': spec.modal,
                'sample_id': sample['sample_id'],
                **feats,
                'baseline_cold_ms': baseline['cold_ms'],
                'pair_cold_ms': pair['cold_ms'],
                'baseline_steady_median_ms': baseline['steady_median_ms'],
                'pair_steady_median_ms': pair['steady_median_ms'],
                'baseline_steady_p95_ms': baseline['steady_p95_ms'],
                'pair_steady_p95_ms': pair['steady_p95_ms'],
                'steady_speedup_baseline_over_pair': (
                    baseline['steady_median_ms'] / pair['steady_median_ms']
                ),
                'cold_ratio_pair_over_baseline': (
                    pair['cold_ms'] / baseline['cold_ms']
                ),
                'max_abs_force_diff_vs_baseline': float(
                    np.max(np.abs(pair['forces'] - baseline['forces']))
                ),
                'abs_energy_diff_vs_baseline': abs(pair['energy'] - baseline['energy']),
            }
            sample_rows.append(row)

        if spec.name in set(args.profile_datasets) and samples:
            sample = samples[0]
            for case in (CASE_BASELINE, CASE_PAIR):
                summary, _ = profile_sample(sample['atoms'], modal=spec.modal, case=case)
                profile_rows.append(
                    {
                        'dataset': spec.name,
                        'sample_id': sample['sample_id'],
                        'case': case['case'],
                        'natoms': sample['natoms'],
                        **summary,
                    }
                )

    sample_frame = pd.DataFrame(sample_rows)
    profile_frame = pd.DataFrame(profile_rows)

    corr_nat = float(sample_frame['natoms'].corr(sample_frame['steady_speedup_baseline_over_pair'], method='spearman'))
    corr_edges = float(sample_frame['num_edges'].corr(sample_frame['steady_speedup_baseline_over_pair'], method='spearman'))

    _write_csv(metrics_dir / 'datasets.csv', dataset_rows)
    _write_csv(metrics_dir / 'samples.csv', sample_rows)
    _write_csv(metrics_dir / 'profiles.csv', profile_rows)

    if not sample_frame.empty:
        _plot_speedup_vs_feature(
            sample_frame, 'natoms', plots_dir / 'speedup_vs_natoms.png'
        )
        _plot_speedup_vs_feature(
            sample_frame, 'num_edges', plots_dir / 'speedup_vs_num_edges.png'
        )
    if not profile_frame.empty:
        _plot_profile_stacked(profile_frame, plots_dir / 'stage_breakdown.png')

    analysis_path = output_dir / 'analysis.md'
    analysis_path.write_text(
        _analysis_text(sample_frame, profile_frame, corr_nat, corr_edges)
    )

    summary_path = output_dir / 'summary.md'
    summary_lines = [
        '# Local Pair Size Validation + Profiling',
        '',
        f'- Datasets: {", ".join(spec.name for spec in selected_specs)}',
        f'- Top-k per dataset: `{args.top_k}`',
        f'- Repeat: `{args.repeat}`',
        f'- Spearman(speedup, natoms): `{corr_nat:.3f}`',
        f'- Spearman(speedup, num_edges): `{corr_edges:.3f}`',
        '',
        '## Outputs',
        '',
        '- `metrics/datasets.csv`',
        '- `metrics/samples.csv`',
        '- `metrics/profiles.csv`',
        '- `plots/speedup_vs_natoms.png`',
        '- `plots/speedup_vs_num_edges.png`',
        '- `plots/stage_breakdown.png`',
        '- `analysis.md`',
    ]
    summary_path.write_text('\n'.join(summary_lines) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
