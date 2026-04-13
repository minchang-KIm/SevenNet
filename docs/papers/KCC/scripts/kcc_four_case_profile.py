from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

import sevenn._keys as KEY
import sevenn.nn.convolution as conv_mod
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.pair_runtime import prepare_pair_metadata
from sevenn.train.dataload import unlabeled_atoms_to_graph

from kcc_common import (
    COLOR_CASES,
    KCC_ROOT,
    aggregate_numeric,
    ensure_dir,
    gpu_info,
    graph_feature_manifest,
    supported_dataset_specs,
    write_json,
)
from all_public_local_pair_bench import load_topk_samples


FOUR_CASES = (
    {
        "case": "e3nn_baseline",
        "enable_flash": False,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "flash_baseline",
        "enable_flash": True,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "e3nn_pair_full",
        "enable_flash": False,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
    },
    {
        "case": "flash_pair_auto",
        "enable_flash": True,
        "enable_pair_execution": True,
        "pair_execution_policy": None,
    },
)

FOUR_CASE_STAGE_GROUP_PATTERNS = {
    "edge_length_norm_ms": ("edge_embedding.edge_length_norm_ms",),
    "radial_basis_ms": ("edge_embedding.radial_basis_ms",),
    "cutoff_ms": ("edge_embedding.cutoff_ms",),
    "radial_combine_ms": ("edge_embedding.radial_combine_ms",),
    "spherical_harmonics_ms": ("edge_embedding.spherical_harmonics_ms",),
    "pair_expand_ms": (
        "edge_embedding.pair_vec_select_ms",
        "edge_embedding.edge_length_expand_ms",
        "edge_embedding.edge_embedding_expand_ms",
        "edge_embedding.edge_attr_expand_ms",
        "edge_embedding.reverse_sign_ms",
        "edge_embedding.edge_attr_sign_apply_ms",
    ),
    "conv_weight_nn_ms": ("weight_nn_ms",),
    "pair_indexing_ms": (
        "forward_src_index_ms",
        "forward_dst_index_ms",
        "reverse_src_index_ms",
        "reverse_dst_index_ms",
        "reverse_weight_select_ms",
        "weight_expand_ms",
    ),
    "conv_src_gather_ms": (
        "edge_src_gather_ms",
        "forward_src_gather_ms",
        "reverse_src_gather_ms",
    ),
    "conv_filter_gather_ms": (
        "forward_filter_gather_ms",
        "reverse_filter_gather_ms",
    ),
    "conv_message_tp_ms": (
        "message_tp_ms",
        "forward_message_tp_ms",
        "reverse_message_tp_ms",
    ),
    "flash_fused_conv_ms": ("fused_convolution_ms",),
    "conv_aggregation_ms": (
        "aggregation_ms",
        "forward_aggregation_ms",
        "reverse_aggregation_ms",
    ),
    "conv_denominator_ms": ("denominator_ms",),
    "top_input_embedding_ms": ("top_input_embedding_ms",),
    "top_interaction_other_ms": ("top_interaction_other_ms",),
    "top_readout_ms": ("top_readout_ms",),
    "top_force_output_ms": ("top_force_output_ms",),
    "model_total_ms": ("model_total_ms",),
}

FOUR_CASE_STAGE_ORDER = [
    "spherical_harmonics_ms",
    "radial_basis_ms",
    "cutoff_ms",
    "conv_weight_nn_ms",
    "pair_expand_ms",
    "pair_indexing_ms",
    "conv_src_gather_ms",
    "conv_filter_gather_ms",
    "conv_message_tp_ms",
    "flash_fused_conv_ms",
    "conv_aggregation_ms",
    "top_force_output_ms",
    "other_ms",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.times_ms: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.loads: dict[str, float] = defaultdict(float)

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
    setattr(obj, attr, wrapper_factory(original))

    def restore() -> None:
        setattr(obj, attr, original)

    return restore


def _aggregate_stage_groups(stage_df: pd.DataFrame, *, key_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in stage_df.groupby(list(key_cols), dropna=False):
        record = dict(zip(key_cols, key))
        used = np.zeros(len(frame), dtype=bool)
        stage_series = frame["stage"].astype(str)
        for stage_name, patterns in FOUR_CASE_STAGE_GROUP_PATTERNS.items():
            mask = np.zeros(len(frame), dtype=bool)
            for pattern in patterns:
                mask |= stage_series.str.contains(pattern, regex=False).to_numpy()
            used |= mask
            subset = frame[mask]
            record[stage_name] = float(subset["time_ms"].sum())
            record[f"{stage_name}_load"] = float(subset["load"].sum())
        other = frame[~used]
        record["other_ms"] = float(other["time_ms"].sum())
        record["other_ms_load"] = float(other["load"].sum())
        rows.append(record)
    return pd.DataFrame(rows)


def _stage_long_from_grouped(grouped_df: pd.DataFrame, *, key_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_names = FOUR_CASE_STAGE_ORDER + ["model_total_ms"]
    for _, row in grouped_df.iterrows():
        base = {key: row[key] for key in key_cols}
        total = float(row.get("model_total_ms_mean", row.get("model_total_ms", np.nan)))
        for stage in stage_names:
            if f"{stage}_mean" in row:
                mean_value = row[f"{stage}_mean"]
                std_value = row.get(f"{stage}_std", np.nan)
            else:
                mean_value = row.get(stage, np.nan)
                std_value = row.get(f"{stage}_std", np.nan)
            rows.append(
                {
                    **base,
                    "stage": stage,
                    "mean_ms": float(mean_value),
                    "std_ms": float(std_value) if pd.notna(std_value) else float("nan"),
                    "share_mean": float(mean_value) / total if total and pd.notna(total) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _is_flash_module(module: Any) -> bool:
    return type(module).__name__ == "IrrepsScatterGatterFusedConvolution"


def _profile_single_sample_flexible(
    atoms: Any,
    *,
    modal: str,
    case: dict[str, Any],
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = SevenNetCalculator(
        model="7net-omni",
        modal=modal,
        device=device,
        enable_flash=case["enable_flash"],
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case.get("pair_execution_policy"),
    )
    timer = StageTimer(device)
    restore_stack: list[Callable[[], None]] = []

    def timed_call(
        key: str,
        fn: Callable[..., Any],
        *args,
        load: float | None = None,
        **kwargs,
    ):
        with timer.section(key, load=load):
            return fn(*args, **kwargs)

    def wrap_edge_embedding(name: str, module: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapped(_original: Callable[..., Any]) -> Callable[..., Any]:
            def forward(data: AtomGraphData) -> AtomGraphData:
                if (
                    module.pair_execution_policy != "baseline"
                    and KEY.PAIR_EDGE_VEC in data
                    and KEY.EDGE_PAIR_MAP in data
                    and KEY.EDGE_PAIR_REVERSE in data
                ):
                    pair_rvec = data[KEY.PAIR_EDGE_VEC]
                    if KEY.PAIR_EDGE_FORWARD_INDEX in data and KEY.EDGE_VEC in data:
                        pair_rvec = timed_call(
                            f"{name}.pair_vec_select_ms",
                            data[KEY.EDGE_VEC].index_select,
                            0,
                            data[KEY.PAIR_EDGE_FORWARD_INDEX],
                            load=float(data[KEY.PAIR_EDGE_FORWARD_INDEX].numel()),
                        )
                        data[KEY.PAIR_EDGE_VEC] = pair_rvec
                    pair_r = timed_call(
                        f"{name}.edge_length_norm_ms",
                        torch.linalg.norm,
                        pair_rvec,
                        dim=-1,
                        load=float(pair_rvec.shape[0]),
                    )
                    pair_basis = timed_call(
                        f"{name}.radial_basis_ms",
                        module.basis_function,
                        pair_r,
                        load=float(pair_r.numel()),
                    )
                    cutoff = timed_call(
                        f"{name}.cutoff_ms",
                        module.cutoff_function,
                        pair_r,
                        load=float(pair_r.numel()),
                    )
                    pair_embedding = timed_call(
                        f"{name}.radial_combine_ms",
                        torch.mul,
                        pair_basis,
                        cutoff.unsqueeze(-1),
                        load=float(pair_basis.numel()),
                    )
                    pair_attr = timed_call(
                        f"{name}.spherical_harmonics_ms",
                        module.spherical,
                        pair_rvec,
                        load=float(pair_rvec.shape[0]),
                    )
                    data[KEY.PAIR_EDGE_EMBEDDING] = pair_embedding
                    data[KEY.PAIR_EDGE_ATTR] = pair_attr
                    data[KEY.EDGE_LENGTH] = timed_call(
                        f"{name}.edge_length_expand_ms",
                        pair_r.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    edge_embedding = timed_call(
                        f"{name}.edge_embedding_expand_ms",
                        pair_embedding.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    edge_attr = timed_call(
                        f"{name}.edge_attr_expand_ms",
                        pair_attr.index_select,
                        0,
                        data[KEY.EDGE_PAIR_MAP],
                        load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                    )
                    reverse_mask = data[KEY.EDGE_PAIR_REVERSE].to(edge_attr.dtype).unsqueeze(-1)
                    sign = timed_call(
                        f"{name}.reverse_sign_ms",
                        lambda: 1.0
                        + reverse_mask
                        * (module.reverse_sh_sign.to(edge_attr.dtype).unsqueeze(0) - 1.0),
                        load=float(edge_attr.numel()),
                    )
                    data[KEY.EDGE_EMBEDDING] = edge_embedding
                    data[KEY.EDGE_ATTR] = timed_call(
                        f"{name}.edge_attr_sign_apply_ms",
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
                        f"{name}.edge_length_norm_ms",
                        torch.linalg.norm,
                        rvec,
                        dim=-1,
                        load=float(rvec.shape[0]),
                    )
                    data[KEY.EDGE_LENGTH] = r

                basis = timed_call(
                    f"{name}.radial_basis_ms",
                    module.basis_function,
                    r,
                    load=float(r.numel()),
                )
                cutoff = timed_call(
                    f"{name}.cutoff_ms",
                    module.cutoff_function,
                    r,
                    load=float(r.numel()),
                )
                data[KEY.EDGE_EMBEDDING] = timed_call(
                    f"{name}.radial_combine_ms",
                    torch.mul,
                    basis,
                    cutoff.unsqueeze(-1),
                    load=float(basis.numel()),
                )
                data[KEY.EDGE_ATTR] = timed_call(
                    f"{name}.spherical_harmonics_ms",
                    module.spherical,
                    rvec,
                    load=float(rvec.shape[0]),
                )
                return data

            return forward

        return wrapped

    def wrap_regular_convolution(name: str, module: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapped(_original: Callable[..., Any]) -> Callable[..., Any]:
            def forward(data: AtomGraphData) -> AtomGraphData:
                with timer.section(f"{name}.total_ms"):
                    x = data[module.key_x]

                    if module.is_parallel:
                        x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

                    if module._use_pair_execution(data):
                        pair_input = data[KEY.PAIR_EDGE_EMBEDDING]
                        pair_weight = timed_call(
                            f"{name}.weight_nn_ms",
                            module.weight_nn,
                            pair_input,
                            load=float(pair_input.shape[0]),
                        )
                        if module.pair_execution_policy == "full" and module.fuse_reduction:
                            edge_index = data[module.key_edge_idx]
                            pair_forward_index = data[KEY.PAIR_EDGE_FORWARD_INDEX]
                            pair_backward_index = data[KEY.PAIR_EDGE_BACKWARD_INDEX]
                            pair_has_reverse = data[KEY.PAIR_EDGE_HAS_REVERSE]
                            edge_src = edge_index[1]
                            edge_dst = edge_index[0]

                            src_forward = timed_call(
                                f"{name}.forward_src_index_ms",
                                edge_src.index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            dst_forward = timed_call(
                                f"{name}.forward_dst_index_ms",
                                edge_dst.index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            x_forward = timed_call(
                                f"{name}.forward_src_gather_ms",
                                x.index_select,
                                0,
                                src_forward,
                                load=float(src_forward.numel()),
                            )
                            filter_forward = timed_call(
                                f"{name}.forward_filter_gather_ms",
                                data[module.key_filter].index_select,
                                0,
                                pair_forward_index,
                                load=float(pair_forward_index.numel()),
                            )
                            msg_forward = timed_call(
                                f"{name}.forward_message_tp_ms",
                                module.convolution,
                                x_forward,
                                filter_forward,
                                pair_weight,
                                load=float(pair_forward_index.numel()),
                            )
                            out = timed_call(
                                f"{name}.forward_aggregation_ms",
                                conv_mod.message_gather,
                                x,
                                dst_forward,
                                msg_forward,
                                load=float(dst_forward.numel()),
                            )

                            rev_index = pair_backward_index[pair_has_reverse]
                            if rev_index.numel() > 0:
                                rev_src = timed_call(
                                    f"{name}.reverse_src_index_ms",
                                    edge_src.index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                rev_dst = timed_call(
                                    f"{name}.reverse_dst_index_ms",
                                    edge_dst.index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                x_reverse = timed_call(
                                    f"{name}.reverse_src_gather_ms",
                                    x.index_select,
                                    0,
                                    rev_src,
                                    load=float(rev_src.numel()),
                                )
                                filter_reverse = timed_call(
                                    f"{name}.reverse_filter_gather_ms",
                                    data[module.key_filter].index_select,
                                    0,
                                    rev_index,
                                    load=float(rev_index.numel()),
                                )
                                reverse_weight = timed_call(
                                    f"{name}.reverse_weight_select_ms",
                                    pair_weight.index_select,
                                    0,
                                    torch.nonzero(pair_has_reverse, as_tuple=False).flatten(),
                                    load=float(rev_index.numel()),
                                )
                                msg_reverse = timed_call(
                                    f"{name}.reverse_message_tp_ms",
                                    module.convolution,
                                    x_reverse,
                                    filter_reverse,
                                    reverse_weight,
                                    load=float(rev_index.numel()),
                                )
                                out = timed_call(
                                    f"{name}.reverse_aggregation_ms",
                                    lambda current_out: current_out
                                    + conv_mod.message_gather(x, rev_dst, msg_reverse),
                                    out,
                                    load=float(rev_dst.numel()),
                                )
                            x = out
                        else:
                            weight = timed_call(
                                f"{name}.weight_expand_ms",
                                pair_weight.index_select,
                                0,
                                data[KEY.EDGE_PAIR_MAP],
                                load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                            )
                            edge_src = data[module.key_edge_idx][1]
                            edge_dst = data[module.key_edge_idx][0]
                            x_src = timed_call(
                                f"{name}.edge_src_gather_ms",
                                x.index_select,
                                0,
                                edge_src,
                                load=float(edge_src.numel()),
                            )
                            message = timed_call(
                                f"{name}.message_tp_ms",
                                module.convolution,
                                x_src,
                                data[module.key_filter],
                                weight,
                                load=float(edge_src.numel()),
                            )
                            x = timed_call(
                                f"{name}.aggregation_ms",
                                conv_mod.message_gather,
                                x,
                                edge_dst,
                                message,
                                load=float(edge_dst.numel()),
                            )
                    else:
                        weight_input = data[module.key_weight_input]
                        weight = timed_call(
                            f"{name}.weight_nn_ms",
                            module.weight_nn,
                            weight_input,
                            load=float(weight_input.shape[0]),
                        )
                        edge_src = data[module.key_edge_idx][1]
                        edge_dst = data[module.key_edge_idx][0]
                        x_src = timed_call(
                            f"{name}.edge_src_gather_ms",
                            x.index_select,
                            0,
                            edge_src,
                            load=float(edge_src.numel()),
                        )
                        message = timed_call(
                            f"{name}.message_tp_ms",
                            module.convolution,
                            x_src,
                            data[module.key_filter],
                            weight,
                            load=float(edge_src.numel()),
                        )
                        x = timed_call(
                            f"{name}.aggregation_ms",
                            conv_mod.message_gather,
                            x,
                            edge_dst,
                            message,
                            load=float(edge_dst.numel()),
                        )

                    x = timed_call(f"{name}.denominator_ms", x.div, module.denominator)
                    if module.is_parallel:
                        x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
                    data[module.key_x] = x
                    return data

            return forward

        return wrapped

    def wrap_flash_convolution(name: str, module: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrapped(_original: Callable[..., Any]) -> Callable[..., Any]:
            def forward(data: AtomGraphData) -> AtomGraphData:
                with timer.section(f"{name}.total_ms"):
                    x = data[module.key_x]
                    if module._use_pair_execution(data):
                        pair_input = data[KEY.PAIR_EDGE_EMBEDDING]
                        pair_weight = timed_call(
                            f"{name}.weight_nn_ms",
                            module.weight_nn,
                            pair_input,
                            load=float(pair_input.shape[0]),
                        )
                        weight = timed_call(
                            f"{name}.weight_expand_ms",
                            pair_weight.index_select,
                            0,
                            data[KEY.EDGE_PAIR_MAP],
                            load=float(data[KEY.EDGE_PAIR_MAP].numel()),
                        )
                    else:
                        weight_input = data[module.key_weight_input]
                        weight = timed_call(
                            f"{name}.weight_nn_ms",
                            module.weight_nn,
                            weight_input,
                            load=float(weight_input.shape[0]),
                        )

                    if module.is_parallel:
                        x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

                    edge_src = data[module.key_edge_idx][1]
                    edge_dst = data[module.key_edge_idx][0]
                    edge_filter = data[module.key_filter]
                    if edge_src.numel() == 0:
                        x = x.new_zeros(x.shape[0], module._out_dim)
                        x = x + (edge_filter.sum() + weight.sum()) * 0
                    else:
                        x = timed_call(
                            f"{name}.fused_convolution_ms",
                            module.convolution,
                            x,
                            edge_filter,
                            weight,
                            edge_src.to(torch.int32),
                            edge_dst.to(torch.int32),
                            load=float(edge_src.numel()),
                        )

                    x = timed_call(f"{name}.denominator_ms", x.div, module.denominator)
                    if module.is_parallel:
                        x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
                    data[module.key_x] = x
                    return data

            return forward

        return wrapped

    def wrap_top_level_modules() -> None:
        for name, module in calc.model.named_children():
            if name == "edge_embedding":
                restore_stack.append(_patch_method(module, "forward", wrap_edge_embedding(name, module)))
                continue
            if "convolution" in name:
                if _is_flash_module(module):
                    restore_stack.append(_patch_method(module, "forward", wrap_flash_convolution(name, module)))
                else:
                    restore_stack.append(_patch_method(module, "forward", wrap_regular_convolution(name, module)))
                continue

            if name in {"onehot_idx_to_onehot", "one_hot_modality", "onehot_to_feature_x"}:
                key = "top_input_embedding_ms"
            elif (
                "self_connection" in name
                or "self_interaction" in name
                or "equivariant_gate" in name
            ):
                key = "top_interaction_other_ms"
            elif name in {
                "reduce_input_to_hidden",
                "reduce_hidden_to_energy",
                "rescale_atomic_energy",
                "reduce_total_enegy",
            }:
                key = "top_readout_ms"
            elif name == "force_output":
                key = "top_force_output_ms"
            else:
                key = "top_other_ms"

            restore_stack.append(
                _patch_method(
                    module,
                    "forward",
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
                with_shift=calc.pair_execution_config["resolved_policy"] != "baseline",
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
        data.to(device)
        return data

    try:
        warmup = build_data()
        calc.model(warmup)
        _sync(device)
        timer.reset()

        output = None
        for _ in range(repeat):
            data = build_data()
            with timer.section("model_total_ms"):
                output = calc.model(data)
        assert output is not None

        output_energy = float(output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item())
        summary = {
            "resolved_policy": calc.pair_execution_config["resolved_policy"],
            "energy": output_energy,
            "num_edges_runtime": int(output[KEY.EDGE_IDX].shape[1]),
        }
        for key, value in timer.times_ms.items():
            summary[key] = value / repeat

        stages = [
            {
                "stage": key,
                "time_ms": value / repeat,
                "calls": timer.calls[key] / repeat,
                "load": timer.loads[key] / repeat,
            }
            for key, value in sorted(timer.times_ms.items())
        ]
        return summary, stages
    finally:
        for restore in reversed(restore_stack):
            restore()
        del calc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _profile_case_runs(
    atoms: Any,
    *,
    modal: str,
    case: dict[str, Any],
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        summary, stages = _profile_single_sample_flexible(
            atoms,
            modal=modal,
            case=case,
            repeat=1,
        )
        summary_rows.append({"repeat_idx": repeat_idx, **summary})
        for stage in stages:
            stage_rows.append({"repeat_idx": repeat_idx, **stage})
    return summary_rows, stage_rows


def _write_per_dataset(
    *,
    dataset: str,
    per_dataset_dir: Path,
    manifest_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    dataset_dir = ensure_dir(per_dataset_dir / dataset)
    manifest_df = pd.DataFrame([row for row in manifest_rows if row["dataset"] == dataset])
    summary_raw_df = pd.DataFrame([row for row in summary_rows if row["dataset"] == dataset])
    stage_raw_df = pd.DataFrame([row for row in stage_rows if row["dataset"] == dataset])
    _write_csv(dataset_dir / "manifest.csv", manifest_df)
    _write_csv(dataset_dir / "four_case_detailed_summary_raw.csv", summary_raw_df)
    _write_csv(dataset_dir / "four_case_detailed_stage_raw.csv", stage_raw_df)
    if not summary_raw_df.empty:
        summary_mean_std_df = aggregate_numeric(
            summary_raw_df,
            keys=("dataset", "sample_id", "modal", "natoms", "case"),
        )
        _write_csv(dataset_dir / "four_case_detailed_summary_mean_std.csv", summary_mean_std_df)
    if not stage_raw_df.empty:
        grouped_df = _aggregate_stage_groups(
            stage_raw_df,
            key_cols=("dataset", "sample_id", "modal", "natoms", "case", "repeat_idx"),
        )
        _write_csv(dataset_dir / "four_case_detailed_stage_grouped_raw.csv", grouped_df)
        stage_mean_std_df = aggregate_numeric(
            grouped_df,
            keys=("dataset", "sample_id", "modal", "natoms", "case"),
        )
        _write_csv(dataset_dir / "four_case_detailed_stage_mean_std.csv", stage_mean_std_df)
        stage_long_df = _stage_long_from_grouped(
            stage_mean_std_df,
            key_cols=("dataset", "sample_id", "modal", "natoms", "case"),
        )
        _write_csv(dataset_dir / "four_case_detailed_stage_long_mean_std.csv", stage_long_df)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    parser.add_argument("--datasets", nargs="*", help="optional subset of dataset names")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_root = ensure_dir(output_root / "metrics" / "four_case")
    global_dir = ensure_dir(metrics_root / "global")
    per_dataset_dir = ensure_dir(metrics_root / "per_dataset")
    run_log = output_root / "run_four_case.log"
    run_log.write_text("")

    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for spec, inventory_row in selected_specs:
        sample_info = load_topk_samples(spec, top_k=1)
        if not sample_info:
            continue
        sample_info = sample_info[0]
        sample = SimpleNamespace(
            dataset=spec.name,
            modal=spec.modal,
            loader=spec.loader,
            sample_id=sample_info["sample_id"],
            natoms=int(sample_info["natoms"]),
            atoms=sample_info["atoms"],
            category=inventory_row.category,
            source_kind=spec.source_kind,
            local_path=str(spec.root),
        )
        manifest = graph_feature_manifest(sample)
        manifest_rows.append(manifest)
        dataset_dir = ensure_dir(per_dataset_dir / sample.dataset)
        write_json(dataset_dir / "manifest.json", manifest)

        line = f"[four-case] {sample.dataset} :: {sample.sample_id} :: natoms={sample.natoms}"
        print(line, flush=True)
        with run_log.open("a") as handle:
            handle.write(line + "\n")

        for case in FOUR_CASES:
            case_summary_rows, case_stage_rows = _profile_case_runs(
                sample.atoms,
                modal=sample.modal,
                case=case,
                repeats=args.repeat,
            )
            for row in case_summary_rows:
                summary_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "n_repeat": args.repeat,
                        **row,
                    }
                )
            for row in case_stage_rows:
                stage_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        **row,
                    }
                )
            _write_per_dataset(
                dataset=sample.dataset,
                per_dataset_dir=per_dataset_dir,
                manifest_rows=manifest_rows,
                summary_rows=summary_rows,
                stage_rows=stage_rows,
            )

    manifest_df = _write_csv(global_dir / "dataset_manifest.csv", manifest_rows)
    summary_raw_df = _write_csv(global_dir / "four_case_detailed_summary_raw.csv", summary_rows)
    stage_raw_df = _write_csv(global_dir / "four_case_detailed_stage_raw.csv", stage_rows)
    summary_mean_std_df = aggregate_numeric(
        summary_raw_df,
        keys=("dataset", "sample_id", "modal", "natoms", "case", "n_repeat"),
    )
    _write_csv(global_dir / "four_case_detailed_summary_mean_std.csv", summary_mean_std_df)
    grouped_df = _aggregate_stage_groups(
        stage_raw_df,
        key_cols=("dataset", "sample_id", "modal", "natoms", "case", "repeat_idx"),
    )
    _write_csv(global_dir / "four_case_detailed_stage_grouped_raw.csv", grouped_df)
    stage_mean_std_df = aggregate_numeric(
        grouped_df,
        keys=("dataset", "sample_id", "modal", "natoms", "case"),
    )
    _write_csv(global_dir / "four_case_detailed_stage_mean_std.csv", stage_mean_std_df)
    stage_long_df = _stage_long_from_grouped(
        stage_mean_std_df,
        key_cols=("dataset", "sample_id", "modal", "natoms", "case"),
    )
    _write_csv(global_dir / "four_case_detailed_stage_long_mean_std.csv", stage_long_df)

    environment = gpu_info()
    environment["repeat"] = args.repeat
    environment["cases"] = [case["case"] for case in FOUR_CASES]
    write_json(global_dir / "environment.json", environment)

    summary_text = "\n".join(
        [
            "# KCC Four-Case Detailed Profile",
            "",
            f"- Datasets benchmarked: `{manifest_df['dataset'].nunique()}`",
            f"- Cases: `{', '.join(case['case'] for case in FOUR_CASES)}`",
            f"- Summary raw rows: `{len(summary_raw_df)}`",
            f"- Stage raw rows: `{len(stage_raw_df)}`",
            "",
            "## Canonical Files",
            "",
            "- `metrics/four_case/global/four_case_detailed_summary_mean_std.csv`",
            "- `metrics/four_case/global/four_case_detailed_stage_mean_std.csv`",
            "- `metrics/four_case/global/four_case_detailed_stage_long_mean_std.csv`",
            "- `metrics/four_case/per_dataset/<dataset>/four_case_detailed_stage_mean_std.csv`",
        ]
    )
    (output_root / "summary_four_case.md").write_text(summary_text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
