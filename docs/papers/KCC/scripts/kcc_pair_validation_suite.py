from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.train.dataload import unlabeled_atoms_to_graph

from kcc_common import KCC_ROOT, ensure_dir, gpu_info, load_topk_samples, supported_dataset_specs, write_json


CASES = (
    {
        "case": "baseline",
        "enable_pair_execution": False,
        "pair_execution_policy": None,
        "expand_full_edges": True,
    },
    {
        "case": "geometry_only",
        "enable_pair_execution": True,
        "pair_execution_policy": "geometry_only",
        "expand_full_edges": True,
    },
    {
        "case": "full_legacy",
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
        "expand_full_edges": True,
    },
    {
        "case": "full_no_expand",
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
        "expand_full_edges": False,
    },
)

SMALL_PROFILER_DATASET = "qm9_hf"
LARGE_PROFILER_DATASET = "mptrj"


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_model_pair_expansion(model, *, expand_full_edges: bool) -> None:
    for module in model.modules():
        if hasattr(module, "expand_full_edges"):
            module.expand_full_edges = expand_full_edges


def _make_energy_only_model(model):
    if hasattr(model, "delete_module_by_key"):
        model.delete_module_by_key("force_output")
    if hasattr(model, "key_grad"):
        model.key_grad = None
    model.eval()
    return model


def _resolve_pair_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return pair_runtime.resolve_pair_execution_config(
        {KEY.PAIR_EXECUTION_CONFIG: {"use": case["enable_pair_execution"], "policy": case["pair_execution_policy"]}},
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case["pair_execution_policy"],
    )


def _build_data(
    atoms: Any,
    *,
    cutoff: float,
    modal: str,
    pair_cfg: dict[str, Any],
    device: torch.device,
) -> AtomGraphData:
    graph = unlabeled_atoms_to_graph(
        atoms,
        cutoff,
        with_shift=pair_cfg["resolved_policy"] != "baseline",
    )
    data = AtomGraphData.from_numpy_dict(graph)
    data[KEY.DATA_MODALITY] = modal
    data, _ = pair_runtime.prepare_pair_metadata(
        data,
        pair_cfg,
        num_atoms=len(atoms),
    )
    data.to(device)  # type: ignore[arg-type]
    return data


def _time_forward(model, data: AtomGraphData, device: torch.device) -> tuple[float, dict[str, Any]]:
    _sync(device)
    start = time.perf_counter()
    out = model(data)
    _sync(device)
    return (time.perf_counter() - start) * 1000.0, out


def _benchmark_step_total(
    atoms: Any,
    *,
    modal: str,
    cutoff: float,
    model,
    case: dict[str, Any],
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_cfg = _resolve_pair_cfg(case)
    rows: list[dict[str, Any]] = []

    def _one() -> dict[str, float]:
        t0 = time.perf_counter()
        graph = unlabeled_atoms_to_graph(
            atoms,
            cutoff,
            with_shift=pair_cfg["resolved_policy"] != "baseline",
        )
        data = AtomGraphData.from_numpy_dict(graph)
        data[KEY.DATA_MODALITY] = modal
        t1 = time.perf_counter()
        data, _ = pair_runtime.prepare_pair_metadata(
            data,
            pair_cfg,
            num_atoms=len(atoms),
        )
        t2 = time.perf_counter()
        data.to(device)  # type: ignore[arg-type]
        _sync(device)
        t3 = time.perf_counter()
        _, out = _time_forward(model, data, device)
        t4 = time.perf_counter()
        return {
            "graph_build_ms": (t1 - t0) * 1000.0,
            "pair_metadata_ms": (t2 - t1) * 1000.0,
            "to_device_ms": (t3 - t2) * 1000.0,
            "model_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
            "energy": float(out[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()),
            "num_edges_runtime": int(out[KEY.EDGE_IDX].shape[1]),
        }

    for _ in range(warmup):
        _one()
    for repeat_idx in range(repeat):
        row = _one()
        row["repeat_idx"] = repeat_idx
        rows.append(row)
    return rows, {"resolved_policy": pair_cfg["resolved_policy"]}


def _benchmark_forward_only(
    atoms: Any,
    *,
    modal: str,
    cutoff: float,
    model,
    case: dict[str, Any],
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_cfg = _resolve_pair_cfg(case)
    base_data = _build_data(atoms, cutoff=cutoff, modal=modal, pair_cfg=pair_cfg, device=device)
    rows: list[dict[str, Any]] = []

    for _ in range(warmup):
        data = base_data.clone()
        _time_forward(model, data, device)

    for repeat_idx in range(repeat):
        data = base_data.clone()
        timing_ms, out = _time_forward(model, data, device)
        rows.append(
            {
                "repeat_idx": repeat_idx,
                "model_ms": timing_ms,
                "energy": float(out[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()),
                "num_edges_runtime": int(out[KEY.EDGE_IDX].shape[1]),
            }
        )
    return rows, {"resolved_policy": pair_cfg["resolved_policy"]}


def _lex_le(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    undecided = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
    result = torch.zeros_like(undecided)
    for col in range(a.shape[1]):
        lt = a[:, col] < b[:, col]
        gt = a[:, col] > b[:, col]
        result = result | (undecided & lt)
        undecided = undecided & ~(lt | gt)
    return result | undecided


def _build_pair_metadata_vectorized(edge_index: torch.Tensor, edge_vec: torch.Tensor, cell_shift: torch.Tensor) -> dict[str, torch.Tensor]:
    device = edge_index.device
    edge_index_i64 = edge_index.to(torch.int64)
    shift = torch.round(cell_shift).to(torch.int64)
    dst = edge_index_i64[0]
    src = edge_index_i64[1]
    forward = torch.stack([dst, src, shift[:, 0], shift[:, 1], shift[:, 2]], dim=1)
    reverse = torch.stack([src, dst, -shift[:, 0], -shift[:, 1], -shift[:, 2]], dim=1)
    choose_forward = _lex_le(forward, reverse)
    canonical = torch.where(choose_forward.unsqueeze(1), forward, reverse)
    _, inverse = torch.unique(canonical, dim=0, sorted=True, return_inverse=True)
    num_pairs = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
    arange_e = torch.arange(edge_index.shape[1], device=device, dtype=torch.int64)
    pair_forward_index = torch.full((num_pairs,), -1, dtype=torch.int64, device=device)
    pair_backward_index = torch.full((num_pairs,), -1, dtype=torch.int64, device=device)
    pair_has_reverse = torch.zeros((num_pairs,), dtype=torch.bool, device=device)
    forward_pairs = inverse[choose_forward]
    pair_forward_index.scatter_(0, forward_pairs, arange_e[choose_forward])
    reverse_mask = ~choose_forward
    if reverse_mask.any():
        reverse_pairs = inverse[reverse_mask]
        pair_backward_index.scatter_(0, reverse_pairs, arange_e[reverse_mask])
        pair_has_reverse[reverse_pairs] = True
    missing_backward = pair_backward_index < 0
    pair_backward_index[missing_backward] = pair_forward_index[missing_backward]
    return {
        KEY.EDGE_PAIR_MAP: inverse,
        KEY.EDGE_PAIR_REVERSE: reverse_mask,
        KEY.PAIR_EDGE_FORWARD_INDEX: pair_forward_index,
        KEY.PAIR_EDGE_BACKWARD_INDEX: pair_backward_index,
        KEY.PAIR_EDGE_HAS_REVERSE: pair_has_reverse,
        KEY.PAIR_EDGE_VEC: edge_vec.index_select(0, pair_forward_index),
    }


def _benchmark_pair_metadata_methods(
    atoms: Any,
    *,
    cutoff: float,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    graph = unlabeled_atoms_to_graph(atoms, cutoff, with_shift=True)
    data = AtomGraphData.from_numpy_dict(graph)
    edge_index = data[KEY.EDGE_IDX]
    edge_vec = data[KEY.EDGE_VEC]
    cell_shift = data[KEY.CELL_SHIFT]
    gpu_edge_index = edge_index.to(device)
    gpu_edge_vec = edge_vec.to(device)
    gpu_cell_shift = cell_shift.to(device)

    def _time_cpu_original() -> float:
        start = time.perf_counter()
        pair_runtime.build_pair_metadata(edge_index, edge_vec, cell_shift=cell_shift, num_atoms=len(atoms))
        return (time.perf_counter() - start) * 1000.0

    def _time_cpu_vectorized() -> float:
        start = time.perf_counter()
        _build_pair_metadata_vectorized(edge_index, edge_vec, cell_shift)
        return (time.perf_counter() - start) * 1000.0

    def _time_gpu_vectorized() -> float:
        _sync(device)
        start = time.perf_counter()
        _build_pair_metadata_vectorized(gpu_edge_index, gpu_edge_vec, gpu_cell_shift)
        _sync(device)
        return (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        _time_cpu_original()
        _time_cpu_vectorized()
        _time_gpu_vectorized()

    rows = []
    for repeat_idx in range(repeat):
        rows.append({"repeat_idx": repeat_idx, "method": "cpu_original", "timing_ms": _time_cpu_original()})
        rows.append({"repeat_idx": repeat_idx, "method": "cpu_vectorized", "timing_ms": _time_cpu_vectorized()})
        rows.append({"repeat_idx": repeat_idx, "method": "gpu_vectorized_kernel_only", "timing_ms": _time_gpu_vectorized()})
    return rows


def _aggregate(df: pd.DataFrame, value_cols: Iterable[str], key_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(key_cols), dropna=False)
    rows: list[dict[str, Any]] = []
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_cols, key))
        for col in value_cols:
            arr = frame[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(arr))
            row[f"{col}_std"] = float(np.std(arr, ddof=0))
            row[f"{col}_median"] = float(np.median(arr))
            row[f"{col}_p95"] = float(np.quantile(arr, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    parser.add_argument("--datasets", nargs="*", help="optional dataset subset")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_root = ensure_dir(output_root / "metrics" / "pair_validation_split")
    global_dir = ensure_dir(metrics_root / "global")
    per_dataset_dir = ensure_dir(metrics_root / "per_dataset")
    run_log = output_root / "run_pair_validation_split.log"
    run_log.write_text("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    model_cache: dict[tuple[str, str, str], tuple[Any, float]] = {}
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metadata_raw_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for spec, inventory_row in selected_specs:
        sample_info = load_topk_samples(spec, top_k=1)
        if not sample_info:
            continue
        sample_info = sample_info[0]
        sample = SimpleNamespace(
            dataset=spec.name,
            modal=spec.modal,
            sample_id=sample_info["sample_id"],
            natoms=int(sample_info["natoms"]),
            atoms=sample_info["atoms"],
            category=inventory_row.category,
            source_kind=spec.source_kind,
            local_path=str(spec.root),
        )
        line = f"[pair-validation] {sample.dataset} :: {sample.sample_id} :: natoms={sample.natoms}"
        print(line, flush=True)
        with run_log.open("a") as handle:
            handle.write(line + "\n")

        manifest_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "modal": sample.modal,
                "natoms": sample.natoms,
                "category": sample.category,
                "source_kind": sample.source_kind,
                "local_path": sample.local_path,
            }
        )

        dataset_dir = ensure_dir(per_dataset_dir / sample.dataset)
        for case in CASES:
            for mode in ("step_force", "forward_energy"):
                cache_key = (sample.modal, case["case"], mode)
                if cache_key not in model_cache:
                    calc = SevenNetCalculator(
                        model="7net-omni",
                        modal=sample.modal,
                        device=device,
                        enable_flash=False,
                        enable_pair_execution=case["enable_pair_execution"],
                        pair_execution_policy=case["pair_execution_policy"],
                    )
                    _configure_model_pair_expansion(calc.model, expand_full_edges=case["expand_full_edges"])
                    if mode == "forward_energy":
                        model = _make_energy_only_model(calc.model)
                    else:
                        model = calc.model
                    model.to(device)
                    model.eval()
                    model_cache[cache_key] = (model, float(calc.cutoff))
                model, cutoff = model_cache[cache_key]

                if mode == "step_force":
                    result_rows, meta = _benchmark_step_total(
                        sample.atoms,
                        modal=sample.modal,
                        cutoff=cutoff,
                        model=model,
                        case=case,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        device=device,
                    )
                else:
                    result_rows, meta = _benchmark_forward_only(
                        sample.atoms,
                        modal=sample.modal,
                        cutoff=cutoff,
                        model=model,
                        case=case,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        device=device,
                    )

                for row in result_rows:
                    raw_rows.append(
                        {
                            "dataset": sample.dataset,
                            "sample_id": sample.sample_id,
                            "modal": sample.modal,
                            "natoms": sample.natoms,
                            "case": case["case"],
                            "mode": mode,
                            "repeat_idx": row["repeat_idx"],
                            "resolved_policy": meta["resolved_policy"],
                            "expand_full_edges": case["expand_full_edges"],
                            **{k: v for k, v in row.items() if k != "repeat_idx"},
                        }
                    )

        model_for_cutoff, cutoff = model_cache[(sample.modal, "baseline", "step_force")]
        _ = model_for_cutoff
        metadata_rows = _benchmark_pair_metadata_methods(
            sample.atoms,
            cutoff=cutoff,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        for row in metadata_rows:
            metadata_raw_rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "modal": sample.modal,
                    "natoms": sample.natoms,
                    **row,
                }
            )

        dataset_raw = pd.DataFrame([row for row in raw_rows if row["dataset"] == sample.dataset])
        dataset_summary = _aggregate(
            dataset_raw,
            value_cols=[col for col in dataset_raw.columns if col.endswith("_ms") or col in ("energy", "num_edges_runtime")],
            key_cols=["dataset", "sample_id", "modal", "natoms", "case", "mode", "resolved_policy", "expand_full_edges"],
        )
        _write_csv(dataset_dir / "pair_validation_raw.csv", dataset_raw)
        _write_csv(dataset_dir / "pair_validation_summary.csv", dataset_summary)
        dataset_meta = pd.DataFrame([row for row in metadata_raw_rows if row["dataset"] == sample.dataset])
        dataset_meta_summary = _aggregate(
            dataset_meta,
            value_cols=["timing_ms"],
            key_cols=["dataset", "sample_id", "modal", "natoms", "method"],
        )
        _write_csv(dataset_dir / "pair_metadata_raw.csv", dataset_meta)
        _write_csv(dataset_dir / "pair_metadata_summary.csv", dataset_meta_summary)
        _write_csv(dataset_dir / "manifest.csv", [manifest_rows[-1]])
        write_json(dataset_dir / "manifest.json", manifest_rows[-1])

    raw_df = _write_csv(global_dir / "pair_validation_raw.csv", raw_rows)
    summary_df = _aggregate(
        raw_df,
        value_cols=[col for col in raw_df.columns if col.endswith("_ms") or col in ("energy", "num_edges_runtime")],
        key_cols=["dataset", "sample_id", "modal", "natoms", "case", "mode", "resolved_policy", "expand_full_edges"],
    )
    _write_csv(global_dir / "pair_validation_summary.csv", summary_df)
    metadata_raw_df = _write_csv(global_dir / "pair_metadata_raw.csv", metadata_raw_rows)
    metadata_summary_df = _aggregate(
        metadata_raw_df,
        value_cols=["timing_ms"],
        key_cols=["dataset", "sample_id", "modal", "natoms", "method"],
    )
    _write_csv(global_dir / "pair_metadata_summary.csv", metadata_summary_df)
    _write_csv(global_dir / "dataset_manifest.csv", manifest_rows)
    write_json(
        global_dir / "environment.json",
        {
            **gpu_info(),
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cases": [case["case"] for case in CASES],
            "small_profiler_dataset": SMALL_PROFILER_DATASET,
            "large_profiler_dataset": LARGE_PROFILER_DATASET,
        },
    )

    summary_text = "\n".join(
        [
            "# Pair Validation Split Suite",
            "",
            f"- datasets: `{summary_df['dataset'].nunique()}`",
            f"- repeats: `{args.repeat}`",
            f"- cases: `{', '.join(case['case'] for case in CASES)}`",
            "",
            "## Canonical files",
            "",
            "- `metrics/pair_validation_split/global/pair_validation_summary.csv`",
            "- `metrics/pair_validation_split/global/pair_validation_raw.csv`",
            "- `metrics/pair_validation_split/global/pair_metadata_summary.csv`",
            "- `metrics/pair_validation_split/per_dataset/<dataset>/pair_validation_summary.csv`",
        ]
    )
    (output_root / "summary_pair_validation_split.md").write_text(summary_text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
