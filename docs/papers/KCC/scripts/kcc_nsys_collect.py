from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from kcc_common import KCC_ROOT, REPRESENTATIVE_NSYS_DATASETS, ensure_dir


def _read_nsys_csv(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("Generated:") or row[0].startswith("Processing"):
                continue
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    header_idx = next((idx for idx, row in enumerate(rows) if "Name" in row or "Operation" in row), None)
    if header_idx is None:
        return pd.DataFrame()
    header = rows[header_idx]
    data = rows[header_idx + 1 :]
    frame = pd.DataFrame(data, columns=header)
    frame = frame.dropna(how="all")
    return frame


def _numeric_value(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            series = frame[candidate].astype(str).str.replace(",", "", regex=False)
            return pd.to_numeric(series, errors="coerce")
    return pd.Series([float("nan")] * len(frame))


def _kernel_group(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("uvu", "sptp", "flash", "fused")):
        return "flash_fused_tp"
    if any(token in lower for token in ("gemm", "cublas", "cutlass", "mma")):
        return "gemm_or_mm"
    if any(token in lower for token in ("scatter", "gather", "index", "reduce")):
        return "indexing_scatter"
    return "other"


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(REPRESENTATIVE_NSYS_DATASETS))
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_global_dir = ensure_dir(output_root / "metrics" / "global")
    nsys_dir = ensure_dir(metrics_global_dir / "nsys")
    per_dataset_root = ensure_dir(output_root / "metrics" / "per_dataset")
    tmp_dir = ensure_dir(output_root / "tmp")
    proc_env = os.environ.copy()
    proc_env["TMPDIR"] = str(tmp_dir)

    raw_kernel_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for case in ("flash_baseline", "flash_pair_auto"):
            for repeat_idx in range(args.repeat):
                stem = nsys_dir / f"{dataset}_{case}_r{repeat_idx}"
                profile_cmd = [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--duration=2",
                    "--stop-on-exit=false",
                    "--kill=sigterm",
                    "--export=sqlite",
                    "--force-overwrite=true",
                    "-o",
                    str(stem),
                    "python",
                    str(output_root / "scripts" / "kcc_nsys_target.py"),
                    "--dataset",
                    dataset,
                    "--case",
                    case,
                    "--warmup",
                    "1",
                    "--loop",
                ]
                _run(profile_cmd, cwd=output_root.parents[3], env=proc_env)

                sqlite_path = stem.with_suffix(".sqlite")
                if not sqlite_path.exists():
                    raise FileNotFoundError(f"Nsight sqlite not found: {sqlite_path}")

                stats_base = stem.parent / f"{stem.name}_stats"
                stats_cmd = [
                    "nsys",
                    "stats",
                    "--report",
                    "gpukernsum,cudaapisum",
                    "--format",
                    "csv,csv",
                    "--output",
                    str(stats_base),
                    "--force-overwrite",
                    str(sqlite_path),
                ]
                _run(stats_cmd, cwd=output_root.parents[3], env=proc_env)

                kernel_csv = Path(f"{stats_base}_gpukernsum.csv")
                cudaapi_csv = Path(f"{stats_base}_cudaapisum.csv")
                kernel_df = _read_nsys_csv(kernel_csv)
                api_df = _read_nsys_csv(cudaapi_csv)

                if not kernel_df.empty:
                    kernel_df["dataset"] = dataset
                    kernel_df["case"] = case
                    kernel_df["repeat_idx"] = repeat_idx
                    kernel_df["kernel_group"] = kernel_df["Name"].astype(str).map(_kernel_group)
                    kernel_df["total_time_ns"] = _numeric_value(
                        kernel_df,
                        ("Total Time (ns)", "Total Time", "Time", "Avg (ns)"),
                    )
                    raw_kernel_rows.extend(kernel_df.to_dict(orient="records"))
                    grouped = (
                        kernel_df.groupby("kernel_group", as_index=False)["total_time_ns"].sum()
                    )
                    total = float(grouped["total_time_ns"].sum()) if not grouped.empty else 0.0
                    for _, row in grouped.iterrows():
                        grouped_rows.append(
                            {
                                "dataset": dataset,
                                "case": case,
                                "repeat_idx": repeat_idx,
                                "kernel_group": row["kernel_group"],
                                "total_time_ns": float(row["total_time_ns"]),
                                "share": float(row["total_time_ns"]) / total if total else float("nan"),
                            }
                        )

                if not api_df.empty:
                    api_out = nsys_dir / f"{dataset}_{case}_r{repeat_idx}_cudaapisum.csv"
                    api_df.to_csv(api_out, index=False)

                # Keep compact CSV summaries and remove large traces after extraction.
                for candidate in [
                    stem.with_suffix(".qdstrm"),
                    stem.with_suffix(".qdrep"),
                    sqlite_path,
                ]:
                    if candidate.exists():
                        candidate.unlink()

    raw_kernel_df = pd.DataFrame(raw_kernel_rows)
    grouped_df = pd.DataFrame(grouped_rows)
    raw_path = metrics_global_dir / "nsys_kernel_raw.csv"
    grouped_path = metrics_global_dir / "nsys_kernel_group_raw.csv"
    raw_kernel_df.to_csv(raw_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)

    if not grouped_df.empty:
        summary_df = (
            grouped_df.groupby(["dataset", "case", "kernel_group"], as_index=False)
            .agg(
                mean_total_time_ns=("total_time_ns", "mean"),
                std_total_time_ns=("total_time_ns", "std"),
                mean_share=("share", "mean"),
                std_share=("share", "std"),
            )
            .fillna(0.0)
        )
    else:
        summary_df = pd.DataFrame(
            columns=[
                "dataset",
                "case",
                "kernel_group",
                "mean_total_time_ns",
                "std_total_time_ns",
                "mean_share",
                "std_share",
            ]
        )
    summary_path = metrics_global_dir / "nsys_kernel_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    for dataset in set(args.datasets):
        per_dataset_dir = ensure_dir(per_dataset_root / dataset)
        raw_subset = raw_kernel_df[raw_kernel_df["dataset"] == dataset]
        summary_subset = summary_df[summary_df["dataset"] == dataset]
        raw_subset.to_csv(per_dataset_dir / "nsys_kernel_raw.csv", index=False)
        summary_subset.to_csv(per_dataset_dir / "nsys_kernel_group_summary.csv", index=False)

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
