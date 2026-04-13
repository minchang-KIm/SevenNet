from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
BENCH_ROOT = REPO_ROOT / "bench"

for candidate in (REPO_ROOT, BENCH_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from all_public_local_pair_bench import (  # noqa: E402
    SupportedDataset,
    _detect_supported_dataset,
    _load_inventory,
    load_topk_samples,
)
from detailed_model_profile import _profile_single_sample  # noqa: E402
from local_pair_size_profile import build_graph_and_pair_features  # noqa: E402
from sevenn.calculator import SevenNetCalculator  # noqa: E402


KCC_ROOT = REPO_ROOT / "docs" / "papers" / "KCC"

COLOR_CASES = {
    "e3nn_baseline": "#0072B2",
    "e3nn_pair_full": "#D55E00",
    "flash_baseline": "#009E73",
    "flash_pair_auto": "#CC79A7",
}

COLOR_STAGES = {
    "spherical_harmonics_ms": "#E69F00",
    "conv_message_tp_ms": "#4D4D4D",
    "top_force_output_ms": "#3B4C63",
    "conv_weight_nn_ms": "#4C9F70",
    "pair_expand_ms": "#8B6BB8",
    "pair_indexing_ms": "#A84832",
    "other_ms": "#9E9E9E",
}

DETAILED_CASES = (
    {
        "case": "e3nn_baseline",
        "enable_flash": False,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "e3nn_pair_full",
        "enable_flash": False,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
    },
)

FLASH_CASES = (
    {
        "case": "flash_baseline",
        "enable_flash": True,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "flash_pair_auto",
        "enable_flash": True,
        "enable_pair_execution": True,
        "pair_execution_policy": None,
    },
)

REFERENCE_CASE = {
    "case": "e3nn_baseline_reference",
    "enable_flash": False,
    "enable_pair_execution": False,
    "pair_execution_policy": None,
}

REPRESENTATIVE_NSYS_DATASETS = (
    "qm9_hf",
    "iso17",
    "spice_2023",
    "mptrj",
    "omat24_1m_official",
    "oc20_s2ef_train_20m",
)

STAGE_GROUP_PATTERNS = {
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
    "conv_src_gather_ms": ("src_gather_ms",),
    "conv_filter_gather_ms": ("filter_gather_ms",),
    "pair_indexing_ms": (
        "forward_src_index_ms",
        "forward_dst_index_ms",
        "reverse_src_index_ms",
        "reverse_dst_index_ms",
        "reverse_weight_select_ms",
        "weight_expand_ms",
    ),
    "conv_message_tp_ms": ("message_tp_ms",),
    "conv_aggregation_ms": ("aggregation_ms",),
    "conv_denominator_ms": ("denominator_ms",),
    "top_input_embedding_ms": ("top_input_embedding_ms",),
    "top_interaction_other_ms": ("top_interaction_other_ms",),
    "top_readout_ms": ("top_readout_ms",),
    "top_force_output_ms": ("top_force_output_ms",),
    "model_total_ms": ("model_total_ms",),
}

PAPER_STAGE_ORDER = [
    "spherical_harmonics_ms",
    "radial_basis_ms",
    "cutoff_ms",
    "conv_weight_nn_ms",
    "pair_expand_ms",
    "pair_indexing_ms",
    "conv_message_tp_ms",
    "conv_aggregation_ms",
    "top_force_output_ms",
    "other_ms",
]


@dataclass(frozen=True)
class DatasetSample:
    dataset: str
    modal: str
    loader: str
    sample_id: str
    natoms: int
    atoms: Any
    category: str
    source_kind: str
    local_path: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sync_if_needed(device: torch.device | str | None = None) -> None:
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def supported_dataset_samples(dataset_names: Iterable[str] | None = None) -> list[DatasetSample]:
    selected = set(dataset_names or [])
    wanted = bool(selected)
    rows = _load_inventory()
    out: list[DatasetSample] = []
    for row in rows:
        if wanted and row.name not in selected:
            continue
        spec, _reason = _detect_supported_dataset(row)
        if spec is None:
            continue
        samples = load_topk_samples(spec, top_k=1)
        if not samples:
            continue
        sample = samples[0]
        out.append(
            DatasetSample(
                dataset=spec.name,
                modal=spec.modal,
                loader=spec.loader,
                sample_id=sample["sample_id"],
                natoms=int(sample["natoms"]),
                atoms=sample["atoms"],
                category=next(item.category for item in rows if item.name == spec.name),
                source_kind=spec.source_kind,
                local_path=str(spec.root),
            )
        )
    return sorted(out, key=lambda item: item.dataset)


def supported_dataset_specs(dataset_names: Iterable[str] | None = None) -> list[tuple[SupportedDataset, Any]]:
    selected = set(dataset_names or [])
    wanted = bool(selected)
    rows = _load_inventory()
    out: list[tuple[SupportedDataset, Any]] = []
    for row in rows:
        if wanted and row.name not in selected:
            continue
        spec, _reason = _detect_supported_dataset(row)
        if spec is None:
            continue
        out.append((spec, row))
    return sorted(out, key=lambda item: item[0].name)


def calculator_for_case(modal: str, case: dict[str, Any], device: torch.device | None = None) -> SevenNetCalculator:
    device = device or resolve_device()
    return SevenNetCalculator(
        model="7net-omni",
        modal=modal,
        device=device,
        enable_flash=case["enable_flash"],
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case.get("pair_execution_policy"),
    )


def evaluate_case(
    atoms: Any,
    *,
    modal: str,
    case: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = resolve_device()
    calc = calculator_for_case(modal, case, device=device)
    timings_ms: list[float] = []
    warmup_ms: list[float] = []
    try:
        for _ in range(warmup):
            sync_if_needed(device)
            start = time.perf_counter()
            calc.calculate(atoms)
            sync_if_needed(device)
            warmup_ms.append((time.perf_counter() - start) * 1000.0)

        last_energy = float(calc.results["energy"])
        last_forces = np.asarray(calc.results["forces"])
        for _ in range(repeat):
            sync_if_needed(device)
            start = time.perf_counter()
            calc.calculate(atoms)
            sync_if_needed(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
            last_energy = float(calc.results["energy"])
            last_forces = np.asarray(calc.results["forces"])

        timing_arr = np.asarray(timings_ms, dtype=np.float64)
        return {
            "case": case["case"],
            "resolved_policy": calc.pair_execution_config["resolved_policy"],
            "device": str(device),
            "warmup_mean_ms": float(np.mean(warmup_ms)) if warmup_ms else float("nan"),
            "warmup_std_ms": float(np.std(warmup_ms, ddof=0)) if warmup_ms else float("nan"),
            "timings_ms": timings_ms,
            "mean_ms": float(np.mean(timing_arr)),
            "std_ms": float(np.std(timing_arr, ddof=0)),
            "median_ms": float(np.median(timing_arr)),
            "p95_ms": float(np.quantile(timing_arr, 0.95)),
            "energy": last_energy,
            "forces": last_forces,
        }
    finally:
        del calc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def reference_eval(atoms: Any, *, modal: str) -> dict[str, Any]:
    result = evaluate_case(
        atoms,
        modal=modal,
        case=REFERENCE_CASE,
        warmup=1,
        repeat=1,
    )
    return {
        "energy": float(result["energy"]),
        "forces": np.asarray(result["forces"]),
        "resolved_policy": result["resolved_policy"],
    }


def detailed_profile_runs(
    atoms: Any,
    *,
    modal: str,
    case: dict[str, Any],
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        summary, stages = _profile_single_sample(
            atoms,
            modal=modal,
            case={
                "case": case["case"],
                "enable_pair_execution": case["enable_pair_execution"],
                "pair_execution_policy": case["pair_execution_policy"],
            },
            repeat=1,
        )
        summary_rows.append({"repeat_idx": repeat_idx, **summary})
        for stage in stages:
            stage_rows.append({"repeat_idx": repeat_idx, **stage})
    return summary_rows, stage_rows


def graph_feature_manifest(sample: DatasetSample) -> dict[str, Any]:
    calc = SevenNetCalculator(
        model="7net-omni",
        modal=sample.modal,
        device="cpu",
        enable_flash=False,
        enable_pair_execution=True,
        pair_execution_policy="full",
    )
    try:
        features = build_graph_and_pair_features(
            sample.atoms,
            cutoff=calc.cutoff,
            pair_enabled=True,
            policy="full",
        )
    finally:
        del calc
        gc.collect()
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "natoms": sample.natoms,
        "modal": sample.modal,
        "loader": sample.loader,
        "category": sample.category,
        "source_kind": sample.source_kind,
        "local_path": sample.local_path,
        **features,
    }


def aggregate_numeric(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col not in keys and pd.api.types.is_numeric_dtype(df[col])]
    grouped = df.groupby(list(keys), dropna=False, as_index=False)
    mean_df = grouped[numeric_cols].mean()
    std_df = grouped[numeric_cols].std(ddof=0)
    base = mean_df[list(keys)].copy()
    mean_part = mean_df[numeric_cols].rename(columns={col: f"{col}_mean" for col in numeric_cols})
    std_part = std_df[numeric_cols].rename(columns={col: f"{col}_std" for col in numeric_cols})
    return pd.concat([base, mean_part, std_part], axis=1)


def aggregate_stage_groups(stage_df: pd.DataFrame, *, key_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in stage_df.groupby(list(key_cols), dropna=False):
        record = dict(zip(key_cols, key))
        used = np.zeros(len(frame), dtype=bool)
        stage_series = frame["stage"].astype(str)
        for stage_name, patterns in STAGE_GROUP_PATTERNS.items():
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


def stage_long_from_grouped(grouped_df: pd.DataFrame, *, key_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_names = PAPER_STAGE_ORDER + ["model_total_ms"]
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def gpu_info() -> dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        }
    return {
        "cuda_available": False,
        "device_name": "cpu",
        "torch_version": torch.__version__,
    }


def nsight_available() -> bool:
    return any(
        os.access(Path(candidate) / "nsys", os.X_OK)
        for candidate in os.getenv("PATH", "").split(os.pathsep)
        if candidate
    )
