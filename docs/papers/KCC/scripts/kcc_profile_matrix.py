from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kcc_common import (
    DETAILED_CASES,
    FLASH_CASES,
    KCC_ROOT,
    REPRESENTATIVE_NSYS_DATASETS,
    aggregate_numeric,
    aggregate_stage_groups,
    detailed_profile_runs,
    ensure_dir,
    evaluate_case,
    gpu_info,
    graph_feature_manifest,
    load_topk_samples,
    reference_eval,
    stage_long_from_grouped,
    supported_dataset_specs,
    write_json,
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


def _comparability_rows() -> list[dict[str, str]]:
    return [
        {
            "family": "e3nn baseline detailed",
            "measurement_style": "intrusive synchronized stage timing",
            "directly_comparable_to": "e3nn pair detailed",
            "not_directly_comparable_to": "Flash end-to-end, Nsight kernel time",
            "intended_use": "stage decomposition, load reduction, share analysis",
        },
        {
            "family": "e3nn pair detailed",
            "measurement_style": "intrusive synchronized stage timing",
            "directly_comparable_to": "e3nn baseline detailed",
            "not_directly_comparable_to": "Flash end-to-end, Nsight kernel time",
            "intended_use": "pair-specific overhead and reusable-term interpretation",
        },
        {
            "family": "FlashTP end-to-end",
            "measurement_style": "non-intrusive repeated wall-clock timing",
            "directly_comparable_to": "FlashTP end-to-end counterpart",
            "not_directly_comparable_to": "intrusive detailed stage timing, Nsight kernel time",
            "intended_use": "headline latency and accuracy comparison",
        },
        {
            "family": "Representative Nsight",
            "measurement_style": "kernel-level trace summary",
            "directly_comparable_to": "same traced case and dataset",
            "not_directly_comparable_to": "wall-clock headline latency",
            "intended_use": "kernel mix validation only",
        },
    ]


def _write_per_dataset_snapshot(
    *,
    dataset: str,
    metrics_per_dataset_dir: Path,
    manifest_rows: list[dict[str, Any]],
    flash_raw_rows: list[dict[str, Any]],
    flash_summary_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    detailed_summary_raw_rows: list[dict[str, Any]],
    detailed_stage_raw_rows: list[dict[str, Any]],
) -> None:
    per_dataset_dir = ensure_dir(metrics_per_dataset_dir / dataset)
    dataset_manifest_df = pd.DataFrame([row for row in manifest_rows if row["dataset"] == dataset])
    dataset_flash_raw_df = pd.DataFrame([row for row in flash_raw_rows if row["dataset"] == dataset])
    dataset_flash_summary_df = pd.DataFrame([row for row in flash_summary_rows if row["dataset"] == dataset])
    dataset_reference_df = pd.DataFrame([row for row in reference_rows if row["dataset"] == dataset])
    dataset_detailed_summary_raw_df = pd.DataFrame(
        [row for row in detailed_summary_raw_rows if row["dataset"] == dataset]
    )
    dataset_detailed_stage_raw_df = pd.DataFrame(
        [row for row in detailed_stage_raw_rows if row["dataset"] == dataset]
    )

    _write_csv(per_dataset_dir / "manifest.csv", dataset_manifest_df)
    _write_csv(per_dataset_dir / "reference_e3nn.csv", dataset_reference_df)
    _write_csv(per_dataset_dir / "flash_end_to_end_raw_repeats.csv", dataset_flash_raw_df)
    _write_csv(per_dataset_dir / "flash_end_to_end_summary.csv", dataset_flash_summary_df)
    _write_csv(per_dataset_dir / "detailed_summary_raw_repeats.csv", dataset_detailed_summary_raw_df)
    _write_csv(per_dataset_dir / "detailed_stage_raw_repeats.csv", dataset_detailed_stage_raw_df)

    if not dataset_detailed_stage_raw_df.empty:
        dataset_detailed_stage_grouped_raw_df = aggregate_stage_groups(
            dataset_detailed_stage_raw_df,
            key_cols=("dataset", "sample_id", "modal", "natoms", "case", "repeat_idx"),
        )
        _write_csv(per_dataset_dir / "detailed_stage_grouped_raw.csv", dataset_detailed_stage_grouped_raw_df)
        dataset_detailed_stage_mean_std_df = aggregate_numeric(
            dataset_detailed_stage_grouped_raw_df,
            keys=("dataset", "sample_id", "modal", "natoms", "case"),
        )
        _write_csv(per_dataset_dir / "detailed_stage_mean_std.csv", dataset_detailed_stage_mean_std_df)
        dataset_detailed_stage_long_df = stage_long_from_grouped(
            dataset_detailed_stage_mean_std_df,
            key_cols=("dataset", "sample_id", "modal", "natoms", "case"),
        )
        _write_csv(per_dataset_dir / "detailed_stage_long_mean_std.csv", dataset_detailed_stage_long_df)

    if not dataset_detailed_summary_raw_df.empty:
        dataset_detailed_summary_mean_std_df = aggregate_numeric(
            dataset_detailed_summary_raw_df,
            keys=("dataset", "sample_id", "modal", "natoms", "case", "n_repeat"),
        )
        _write_csv(per_dataset_dir / "detailed_summary_mean_std.csv", dataset_detailed_summary_mean_std_df)

    if not dataset_flash_summary_df.empty:
        flash_pivot = dataset_flash_summary_df.pivot(index="dataset", columns="case", values="mean_ms")
        if {"flash_baseline", "flash_pair_auto"}.issubset(flash_pivot.columns):
            flash_std_pivot = dataset_flash_summary_df.pivot(index="dataset", columns="case", values="std_ms")
            dataset_flash_cmp_df = pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "flash_baseline_mean_ms": float(flash_pivot.loc[dataset, "flash_baseline"]),
                        "flash_baseline_std_ms": float(flash_std_pivot.loc[dataset, "flash_baseline"]),
                        "flash_pair_auto_mean_ms": float(flash_pivot.loc[dataset, "flash_pair_auto"]),
                        "flash_pair_auto_std_ms": float(flash_std_pivot.loc[dataset, "flash_pair_auto"]),
                        "flash_speedup_baseline_over_pair": float(
                            flash_pivot.loc[dataset, "flash_baseline"]
                            / flash_pivot.loc[dataset, "flash_pair_auto"]
                        ),
                    }
                ]
            )
            _write_csv(per_dataset_dir / "flash_comparison.csv", dataset_flash_cmp_df)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    parser.add_argument("--datasets", nargs="*", help="optional subset of dataset names")
    parser.add_argument("--warmup-end-to-end", type=int, default=3)
    parser.add_argument("--repeat-end-to-end", type=int, default=10)
    parser.add_argument("--repeat-detailed", type=int, default=5)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    manuscript_dir = ensure_dir(output_root / "manuscript")
    figures_dir = ensure_dir(output_root / "figures")
    tables_dir = ensure_dir(output_root / "tables")
    metrics_global_dir = ensure_dir(output_root / "metrics" / "global")
    metrics_per_dataset_dir = ensure_dir(output_root / "metrics" / "per_dataset")
    scripts_dir = ensure_dir(output_root / "scripts")
    del manuscript_dir, figures_dir, tables_dir, scripts_dir
    run_log_path = output_root / "run.log"
    run_log_path.write_text("")

    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    manifest_rows: list[dict[str, Any]] = []
    flash_raw_rows: list[dict[str, Any]] = []
    flash_summary_rows: list[dict[str, Any]] = []
    detailed_summary_raw_rows: list[dict[str, Any]] = []
    detailed_stage_raw_rows: list[dict[str, Any]] = []
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
        line = f"[dataset] {sample.dataset} :: {sample.sample_id} :: natoms={sample.natoms}"
        print(line, flush=True)
        with run_log_path.open("a") as handle:
            handle.write(line + "\n")
        manifest = graph_feature_manifest(sample)
        manifest["density_bucket"] = _dataset_bucket(pd.Series(manifest))
        manifest_rows.append(manifest)

        per_dataset_dir = ensure_dir(metrics_per_dataset_dir / sample.dataset)
        write_json(per_dataset_dir / "manifest.json", manifest)

        ref = reference_eval(sample.atoms, modal=sample.modal)
        reference_row = {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "modal": sample.modal,
            "reference_case": "e3nn_baseline",
            "resolved_policy": ref["resolved_policy"],
            "energy": float(ref["energy"]),
        }
        reference_rows.append(reference_row)
        _write_csv(per_dataset_dir / "reference_e3nn.csv", [reference_row])

        for case in FLASH_CASES:
            result = evaluate_case(
                sample.atoms,
                modal=sample.modal,
                case=case,
                warmup=args.warmup_end_to_end,
                repeat=args.repeat_end_to_end,
            )
            for repeat_idx, timing in enumerate(result["timings_ms"]):
                flash_raw_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "repeat_idx": repeat_idx,
                        "timing_ms": timing,
                        "resolved_policy": result["resolved_policy"],
                        "n_warmup": args.warmup_end_to_end,
                        "n_repeat": args.repeat_end_to_end,
                        "device": result["device"],
                    }
                )
            flash_summary_rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "modal": sample.modal,
                    "natoms": sample.natoms,
                    "case": case["case"],
                    "resolved_policy": result["resolved_policy"],
                    "n_warmup": args.warmup_end_to_end,
                    "n_repeat": args.repeat_end_to_end,
                    "device": result["device"],
                    "mean_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                    "median_ms": result["median_ms"],
                    "p95_ms": result["p95_ms"],
                    "warmup_mean_ms": result["warmup_mean_ms"],
                    "warmup_std_ms": result["warmup_std_ms"],
                    "abs_energy_diff_vs_e3nn": abs(float(result["energy"]) - float(ref["energy"])),
                    "max_abs_force_diff_vs_e3nn": float(
                        np.max(np.abs(np.asarray(result["forces"]) - np.asarray(ref["forces"])))
                    ),
                }
            )
            gc.collect()

        for case in DETAILED_CASES:
            summary_rows, stage_rows = detailed_profile_runs(
                sample.atoms,
                modal=sample.modal,
                case=case,
                repeats=args.repeat_detailed,
            )
            for row in summary_rows:
                detailed_summary_raw_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "n_repeat": args.repeat_detailed,
                        **row,
                    }
                )
            for row in stage_rows:
                detailed_stage_raw_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "n_repeat": args.repeat_detailed,
                        **row,
                    }
                )
            gc.collect()

        _write_per_dataset_snapshot(
            dataset=sample.dataset,
            metrics_per_dataset_dir=metrics_per_dataset_dir,
            manifest_rows=manifest_rows,
            flash_raw_rows=flash_raw_rows,
            flash_summary_rows=flash_summary_rows,
            reference_rows=reference_rows,
            detailed_summary_raw_rows=detailed_summary_raw_rows,
            detailed_stage_raw_rows=detailed_stage_raw_rows,
        )
        with run_log_path.open("a") as handle:
            handle.write(f"[done] {sample.dataset}\n")

    manifest_df = _write_csv(metrics_global_dir / "dataset_manifest.csv", manifest_rows)
    flash_raw_df = _write_csv(metrics_global_dir / "flash_end_to_end_raw_repeats.csv", flash_raw_rows)
    flash_summary_df = _write_csv(metrics_global_dir / "flash_end_to_end_summary.csv", flash_summary_rows)
    reference_df = _write_csv(metrics_global_dir / "reference_e3nn.csv", reference_rows)
    detailed_summary_raw_df = _write_csv(
        metrics_global_dir / "detailed_summary_raw_repeats.csv", detailed_summary_raw_rows
    )
    detailed_stage_raw_df = _write_csv(
        metrics_global_dir / "detailed_stage_raw_repeats.csv", detailed_stage_raw_rows
    )

    detailed_stage_grouped_raw_df = aggregate_stage_groups(
        detailed_stage_raw_df,
        key_cols=("dataset", "sample_id", "modal", "natoms", "case", "repeat_idx"),
    )
    _write_csv(metrics_global_dir / "detailed_stage_grouped_raw.csv", detailed_stage_grouped_raw_df)

    detailed_summary_mean_std_df = aggregate_numeric(
        detailed_summary_raw_df,
        keys=("dataset", "sample_id", "modal", "natoms", "case", "n_repeat"),
    )
    detailed_stage_mean_std_df = aggregate_numeric(
        detailed_stage_grouped_raw_df,
        keys=("dataset", "sample_id", "modal", "natoms", "case"),
    )
    _write_csv(metrics_global_dir / "detailed_summary_mean_std.csv", detailed_summary_mean_std_df)
    _write_csv(metrics_global_dir / "detailed_stage_mean_std.csv", detailed_stage_mean_std_df)

    detailed_stage_long_df = stage_long_from_grouped(
        detailed_stage_mean_std_df,
        key_cols=("dataset", "sample_id", "modal", "natoms", "case"),
    )
    _write_csv(metrics_global_dir / "detailed_stage_long_mean_std.csv", detailed_stage_long_df)

    flash_pivot = flash_summary_df.pivot(index="dataset", columns="case", values="mean_ms")
    flash_std_pivot = flash_summary_df.pivot(index="dataset", columns="case", values="std_ms")
    flash_cmp_rows: list[dict[str, Any]] = []
    for dataset in flash_pivot.index:
        if {"flash_baseline", "flash_pair_auto"}.issubset(flash_pivot.columns):
            baseline_mean = float(flash_pivot.loc[dataset, "flash_baseline"])
            pair_mean = float(flash_pivot.loc[dataset, "flash_pair_auto"])
            flash_cmp_rows.append(
                {
                    "dataset": dataset,
                    "flash_baseline_mean_ms": baseline_mean,
                    "flash_baseline_std_ms": float(flash_std_pivot.loc[dataset, "flash_baseline"]),
                    "flash_pair_auto_mean_ms": pair_mean,
                    "flash_pair_auto_std_ms": float(flash_std_pivot.loc[dataset, "flash_pair_auto"]),
                    "flash_speedup_baseline_over_pair": baseline_mean / pair_mean,
                }
            )
    flash_cmp_df = _write_csv(metrics_global_dir / "flash_comparison.csv", flash_cmp_rows)

    comparability_df = _write_csv(metrics_global_dir / "comparability_legend.csv", _comparability_rows())

    for dataset in manifest_df["dataset"].tolist():
        _write_per_dataset_snapshot(
            dataset=dataset,
            metrics_per_dataset_dir=metrics_per_dataset_dir,
            manifest_rows=manifest_rows,
            flash_raw_rows=flash_raw_rows,
            flash_summary_rows=flash_summary_rows,
            reference_rows=reference_rows,
            detailed_summary_raw_rows=detailed_summary_raw_rows,
            detailed_stage_raw_rows=detailed_stage_raw_rows,
        )

    representative_df = manifest_df[manifest_df["dataset"].isin(REPRESENTATIVE_NSYS_DATASETS)].copy()
    _write_csv(metrics_global_dir / "nsys_representative_manifest.csv", representative_df)
    for dataset in representative_df["dataset"].tolist():
        per_dataset_dir = ensure_dir(metrics_per_dataset_dir / dataset)
        _write_csv(
            per_dataset_dir / "nsys_representative_manifest.csv",
            representative_df[representative_df["dataset"] == dataset],
        )

    environment_payload = {
        "repo_root": str(KCC_ROOT.parents[3]),
        "kcc_root": str(output_root),
        "gpu": gpu_info(),
        "selected_datasets": manifest_df["dataset"].tolist(),
        "representative_nsys_datasets": list(REPRESENTATIVE_NSYS_DATASETS),
        "end_to_end_warmup": args.warmup_end_to_end,
        "end_to_end_repeat": args.repeat_end_to_end,
        "detailed_repeat": args.repeat_detailed,
    }
    write_json(metrics_global_dir / "environment.json", environment_payload)
    write_json(output_root / "environment.json", environment_payload)

    summary_lines = [
        "# KCC Profile Matrix",
        "",
        f"- Datasets benchmarked: `{len(manifest_df)}`",
        f"- Flash raw repeats: `{len(flash_raw_df)}`",
        f"- Detailed summary raw repeats: `{len(detailed_summary_raw_df)}`",
        f"- Detailed stage raw repeats: `{len(detailed_stage_raw_df)}`",
        "",
        "## Canonical Files",
        "",
        "- `metrics/global/dataset_manifest.csv`",
        "- `metrics/global/flash_end_to_end_summary.csv`",
        "- `metrics/global/detailed_stage_mean_std.csv`",
        "- `metrics/global/detailed_stage_long_mean_std.csv`",
        "- `metrics/global/flash_comparison.csv`",
        "- `metrics/global/comparability_legend.csv`",
    ]
    (output_root / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(json.dumps({"output_root": str(output_root), "datasets": len(manifest_df)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
