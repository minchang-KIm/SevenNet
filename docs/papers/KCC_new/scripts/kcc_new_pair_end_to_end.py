from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

KCC_SCRIPTS = Path(__file__).resolve().parents[2] / "KCC" / "scripts"
if str(KCC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(KCC_SCRIPTS))

from kcc_common import (
    aggregate_numeric,
    ensure_dir,
    evaluate_case,
    gpu_info,
    graph_feature_manifest,
    load_topk_samples,
    reference_eval,
    supported_dataset_specs,
    write_json,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

PAIR_CASES = (
    {
        "case": "sevennet_baseline",
        "enable_flash": False,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "sevennet_geometry_only",
        "enable_flash": False,
        "enable_pair_execution": True,
        "pair_execution_policy": "geometry_only",
    },
)


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _dataset_bucket(row: pd.Series) -> str:
    large = row["num_edges"] >= 3000
    dense = row["avg_neighbors_directed"] >= 40.0
    if large and dense:
        return "large_dense"
    if large and not dense:
        return "large_sparse"
    if (not large) and dense:
        return "small_dense"
    return "small_sparse"


def _write_per_dataset(
    *,
    dataset: str,
    per_dataset_dir: Path,
    manifest_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> None:
    dataset_dir = ensure_dir(per_dataset_dir / dataset)
    _write_csv(dataset_dir / "manifest.csv", [row for row in manifest_rows if row["dataset"] == dataset])
    _write_csv(dataset_dir / "reference_baseline.csv", [row for row in reference_rows if row["dataset"] == dataset])
    dataset_raw = pd.DataFrame([row for row in raw_rows if row["dataset"] == dataset])
    dataset_summary = pd.DataFrame([row for row in summary_rows if row["dataset"] == dataset])
    _write_csv(dataset_dir / "end_to_end_raw_repeats.csv", dataset_raw)
    _write_csv(dataset_dir / "end_to_end_summary.csv", dataset_summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", help="optional subset of dataset names")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_root = ensure_dir(output_root / "metrics" / "pair_end_to_end")
    global_dir = ensure_dir(metrics_root / "global")
    per_dataset_dir = ensure_dir(metrics_root / "per_dataset")
    run_log = output_root / "run_pair_end_to_end.log"
    run_log.write_text("")

    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    manifest_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

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
        manifest["density_bucket"] = _dataset_bucket(pd.Series(manifest))
        manifest_rows.append(manifest)
        write_json(ensure_dir(per_dataset_dir / sample.dataset) / "manifest.json", manifest)

        line = f"[kcc-new end-to-end] {sample.dataset} :: {sample.sample_id} :: natoms={sample.natoms}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        ref = reference_eval(sample.atoms, modal=sample.modal)
        reference_row = {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "modal": sample.modal,
            "reference_case": "sevennet_baseline",
            "resolved_policy": ref["resolved_policy"],
            "energy": float(ref["energy"]),
        }
        reference_rows.append(reference_row)

        for case in PAIR_CASES:
            result = evaluate_case(
                sample.atoms,
                modal=sample.modal,
                case=case,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            for repeat_idx, timing in enumerate(result["timings_ms"]):
                raw_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "repeat_idx": repeat_idx,
                        "timing_ms": timing,
                        "resolved_policy": result["resolved_policy"],
                        "n_warmup": args.warmup,
                        "n_repeat": args.repeat,
                        "device": result["device"],
                    }
                )
            summary_rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "modal": sample.modal,
                    "natoms": sample.natoms,
                    "case": case["case"],
                    "resolved_policy": result["resolved_policy"],
                    "n_warmup": args.warmup,
                    "n_repeat": args.repeat,
                    "device": result["device"],
                    "mean_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                    "median_ms": result["median_ms"],
                    "p95_ms": result["p95_ms"],
                    "warmup_mean_ms": result["warmup_mean_ms"],
                    "warmup_std_ms": result["warmup_std_ms"],
                    "abs_energy_diff_vs_baseline": abs(float(result["energy"]) - float(ref["energy"])),
                    "max_abs_force_diff_vs_baseline": float(np.max(np.abs(np.asarray(result["forces"]) - ref["forces"]))),
                }
            )

        _write_per_dataset(
            dataset=sample.dataset,
            per_dataset_dir=per_dataset_dir,
            manifest_rows=manifest_rows,
            raw_rows=raw_rows,
            summary_rows=summary_rows,
            reference_rows=reference_rows,
        )

    manifest_df = _write_csv(global_dir / "dataset_manifest.csv", manifest_rows)
    raw_df = _write_csv(global_dir / "pair_end_to_end_raw_repeats.csv", raw_rows)
    summary_df = _write_csv(global_dir / "pair_end_to_end_summary.csv", summary_rows)
    _write_csv(global_dir / "reference_baseline.csv", reference_rows)
    _write_csv(
        global_dir / "pair_end_to_end_raw_mean_std.csv",
        aggregate_numeric(
            raw_df,
            keys=("dataset", "sample_id", "modal", "natoms", "case", "n_warmup", "n_repeat", "device", "resolved_policy"),
        ),
    )

    pivot_mean = summary_df.pivot(index="dataset", columns="case", values="mean_ms")
    pivot_std = summary_df.pivot(index="dataset", columns="case", values="std_ms")
    comparison_rows: list[dict[str, Any]] = []
    for dataset in pivot_mean.index:
        comparison_rows.append(
            {
                "dataset": dataset,
                "baseline_mean_ms": float(pivot_mean.loc[dataset, "sevennet_baseline"]),
                "baseline_std_ms": float(pivot_std.loc[dataset, "sevennet_baseline"]),
                "proposal_mean_ms": float(pivot_mean.loc[dataset, "sevennet_geometry_only"]),
                "proposal_std_ms": float(pivot_std.loc[dataset, "sevennet_geometry_only"]),
                "speedup_baseline_over_proposal": float(
                    pivot_mean.loc[dataset, "sevennet_baseline"] / pivot_mean.loc[dataset, "sevennet_geometry_only"]
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).merge(
        manifest_df[["dataset", "natoms", "num_edges", "avg_neighbors_directed", "density_bucket"]],
        on="dataset",
        how="left",
    )
    _write_csv(global_dir / "pair_end_to_end_comparison.csv", comparison_df)

    env = gpu_info()
    env["warmup"] = args.warmup
    env["repeat"] = args.repeat
    env["cases"] = [case["case"] for case in PAIR_CASES]
    write_json(global_dir / "environment.json", env)
    write_json(output_root / "environment.json", env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
