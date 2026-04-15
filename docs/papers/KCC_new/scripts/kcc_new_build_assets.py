from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KCC_SCRIPTS = Path(__file__).resolve().parents[2] / "KCC" / "scripts"
if str(KCC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(KCC_SCRIPTS))

from kcc_common import ensure_dir


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_KCC_ROOT = DEFAULT_ROOT.parent / "KCC"

COLOR_CASES = {
    "sevennet_baseline": "#0072B2",
    "sevennet_geometry_only": "#D55E00",
}


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _savefig(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "데이터 없음\n"
    cols = df.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def _format_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _copy_diagnostic_assets(output_root: Path) -> None:
    copy_specs = [
        (
            SOURCE_KCC_ROOT / "geometry_only_breakdown",
            output_root / "geometry_only_breakdown",
        ),
        (
            SOURCE_KCC_ROOT / "lammps_pair_metadata_bench",
            output_root / "lammps_pair_metadata_bench",
        ),
        (
            SOURCE_KCC_ROOT / "reports" / "pair_metadata_force_backward_visual_note.md",
            output_root / "reports" / "pair_metadata_force_backward_visual_note.md",
        ),
        (
            SOURCE_KCC_ROOT / "figures" / "runtime",
            output_root / "figures" / "runtime",
        ),
    ]
    for src, dst in copy_specs:
        if not src.exists():
            continue
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        ensure_dir(dst.parent)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _write_setup_tables(output_root: Path, summary_df: pd.DataFrame) -> None:
    table_dir = ensure_dir(output_root / "tables")
    global_dir = output_root / "metrics" / "pair_end_to_end" / "global"
    environment_path = global_dir / "environment.json"
    env = {}
    if environment_path.exists():
        env = json.loads(environment_path.read_text())
    repeat = int(summary_df["n_repeat"].iloc[0])
    warmup = int(summary_df["n_warmup"].iloc[0])
    dataset_count = int(summary_df["dataset"].nunique())
    rows = pd.DataFrame(
        [
            {"item": "device", "value": env.get("device_name", "unknown")},
            {"item": "torch_version", "value": env.get("torch_version", "unknown")},
            {"item": "datasets", "value": dataset_count},
            {"item": "warmup", "value": warmup},
            {"item": "repeat", "value": repeat},
            {"item": "representative sample per dataset", "value": 1},
            {"item": "headline comparison", "value": "SevenNet baseline vs geometry_only"},
            {"item": "diagnostic profile", "value": "geometry_only breakdown, intrusive, repeat 30"},
            {"item": "LAMMPS diagnostic", "value": "serial pair metadata bench, repeat 30"},
        ]
    )
    rows.to_csv(table_dir / "table_01_experiment_setup.csv", index=False)
    (table_dir / "table_01_experiment_setup.md").write_text(_df_to_md(rows), encoding="utf-8")


def _write_end_to_end_tables(output_root: Path, summary_df: pd.DataFrame, comp_df: pd.DataFrame) -> None:
    table_dir = ensure_dir(output_root / "tables")
    out = pd.DataFrame(
        {
            "dataset": comp_df["dataset"],
            "baseline (ms)": [
                _format_pm(mean, std)
                for mean, std in zip(comp_df["baseline_mean_ms"], comp_df["baseline_std_ms"], strict=True)
            ],
            "geometry_only (ms)": [
                _format_pm(mean, std)
                for mean, std in zip(comp_df["proposal_mean_ms"], comp_df["proposal_std_ms"], strict=True)
            ],
            "speedup (baseline/geometry_only)": comp_df["speedup_baseline_over_proposal"].map(lambda x: f"{x:.3f}x"),
            "natoms": comp_df["natoms"],
            "num_edges": comp_df["num_edges"],
            "avg_neighbors_directed": comp_df["avg_neighbors_directed"].map(lambda x: f"{x:.2f}"),
            "density_bucket": comp_df["density_bucket"],
        }
    ).sort_values("dataset")
    out.to_csv(table_dir / "table_02_end_to_end_summary.csv", index=False)
    (table_dir / "table_02_end_to_end_summary.md").write_text(_df_to_md(out), encoding="utf-8")

    rows = []
    def add_group(name: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "group": name,
                "count": int(len(subset)),
                "wins": int((subset["speedup_baseline_over_proposal"] > 1.0).sum()),
                "win_rate": f"{float((subset['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}",
                "mean_speedup": f"{float(subset['speedup_baseline_over_proposal'].mean()):.3f}",
                "median_speedup": f"{float(subset['speedup_baseline_over_proposal'].median()):.3f}",
                "std_speedup": f"{float(subset['speedup_baseline_over_proposal'].std(ddof=0)):.3f}",
            }
        )

    add_group("num_edges >= 3000", comp_df[comp_df["num_edges"] >= 3000])
    add_group("num_edges < 3000", comp_df[comp_df["num_edges"] < 3000])
    add_group("avg_neighbors_directed >= 40", comp_df[comp_df["avg_neighbors_directed"] >= 40.0])
    add_group("avg_neighbors_directed < 40", comp_df[comp_df["avg_neighbors_directed"] < 40.0])
    for bucket in ["small_sparse", "small_dense", "large_sparse", "large_dense"]:
        add_group(bucket, comp_df[comp_df["density_bucket"] == bucket])
    cond_df = pd.DataFrame(rows)
    cond_df.to_csv(table_dir / "table_03_condition_summary.csv", index=False)
    (table_dir / "table_03_condition_summary.md").write_text(_df_to_md(cond_df), encoding="utf-8")


def _write_accuracy_tables(output_root: Path, summary_df: pd.DataFrame) -> None:
    table_dir = ensure_dir(output_root / "tables")
    pivot = summary_df.pivot(index="dataset", columns="case")
    rows = []
    for dataset in sorted(summary_df["dataset"].unique()):
        rows.append(
            {
                "dataset": dataset,
                "baseline energy diff (eV)": _format_pm(
                    float(pivot.loc[dataset, ("abs_energy_diff_mean", "sevennet_baseline")]),
                    float(pivot.loc[dataset, ("abs_energy_diff_std", "sevennet_baseline")]),
                    3,
                ),
                "geometry_only energy diff (eV)": _format_pm(
                    float(pivot.loc[dataset, ("abs_energy_diff_mean", "sevennet_geometry_only")]),
                    float(pivot.loc[dataset, ("abs_energy_diff_std", "sevennet_geometry_only")]),
                    3,
                ),
                "baseline force diff (eV/A)": _format_pm(
                    float(pivot.loc[dataset, ("max_force_diff_mean", "sevennet_baseline")]),
                    float(pivot.loc[dataset, ("max_force_diff_std", "sevennet_baseline")]),
                    3,
                ),
                "geometry_only force diff (eV/A)": _format_pm(
                    float(pivot.loc[dataset, ("max_force_diff_mean", "sevennet_geometry_only")]),
                    float(pivot.loc[dataset, ("max_force_diff_std", "sevennet_geometry_only")]),
                    3,
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "table_04_accuracy_summary.csv", index=False)
    (table_dir / "table_04_accuracy_summary.md").write_text(_df_to_md(out), encoding="utf-8")


def _write_diagnostic_tables(output_root: Path) -> None:
    table_dir = ensure_dir(output_root / "tables")
    geometry_report = output_root / "geometry_only_breakdown" / "metrics" / "geometry_only_breakdown_summary.csv"
    if geometry_report.exists():
        df = pd.read_csv(geometry_report)
        pivot = df.pivot_table(index=["system", "case"], columns="stage", values="mean_ms", aggfunc="first").reset_index()
        baseline = pivot[pivot["case"] == "baseline"].set_index("system")
        geom = pivot[pivot["case"] == "geometry_only"].set_index("system")
        rows = []
        for system in sorted(set(baseline.index) & set(geom.index)):
            rows.append(
                {
                    "system": system,
                    "baseline_forward_total_ms": float(baseline.loc[system].get("model_total_ms", np.nan)),
                    "geometry_only_forward_total_ms": float(geom.loc[system].get("model_total_ms", np.nan)),
                    "pair_expand_ms": float(
                        sum(
                            float(geom.loc[system].get(stage, 0.0))
                            for stage in (
                                "edge_pair_vec_select_ms",
                                "edge_length_expand_ms",
                                "edge_embedding_expand_ms",
                                "edge_attr_expand_ms",
                                "edge_attr_sign_ms",
                            )
                        )
                    ),
                    "pair_weight_expand_ms": float(geom.loc[system].get("conv_weight_expand_ms", np.nan)),
                    "pair_geometry_ms": float(
                        sum(
                            float(geom.loc[system].get(stage, 0.0))
                            for stage in ("edge_length_norm_ms", "radial_basis_cutoff_ms", "spherical_harmonics_ms")
                        )
                    ),
                }
            )
        keep = pd.DataFrame(rows)
        keep.to_csv(table_dir / "table_05_geometry_breakdown_summary.csv", index=False)
        (table_dir / "table_05_geometry_breakdown_summary.md").write_text(_df_to_md(keep), encoding="utf-8")

    lammps_summary = output_root / "lammps_pair_metadata_bench" / "metrics" / "lammps_pair_metadata_summary.csv"
    if lammps_summary.exists():
        df = pd.read_csv(lammps_summary)
        keep = df[
            [
                "system",
                "case",
                "pair_metadata_total_ms_mean",
                "pair_metadata_total_ms_std",
                "compute_total_ms_mean",
                "compute_total_ms_std",
            ]
        ].copy()
        keep.to_csv(table_dir / "table_06_lammps_pair_metadata_summary.csv", index=False)
        (table_dir / "table_06_lammps_pair_metadata_summary.md").write_text(_df_to_md(keep), encoding="utf-8")


def _plot_latency(summary_df: pd.DataFrame, out_stem: Path) -> None:
    pivot_mean = summary_df.pivot(index="dataset", columns="case", values="mean_ms").sort_index()
    pivot_std = summary_df.pivot(index="dataset", columns="case", values="std_ms").sort_index()
    x = np.arange(len(pivot_mean))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(14, len(pivot_mean) * 0.42), 6.5))
    for offset, case in [(-width / 2, "sevennet_baseline"), (width / 2, "sevennet_geometry_only")]:
        ax.bar(
            x + offset,
            pivot_mean[case],
            width,
            label=case,
            color=COLOR_CASES[case],
            yerr=pivot_std[case],
            capsize=3,
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_mean.index, rotation=45, ha="right")
    ax.set_ylabel("Steady-state latency (ms)")
    ax.set_title("SevenNet baseline vs geometry_only latency")
    ax.legend()
    _savefig(fig, out_stem)


def _plot_speedup(comp_df: pd.DataFrame, out_stem: Path) -> None:
    ordered = comp_df.sort_values("speedup_baseline_over_proposal", ascending=False)
    colors = ["#2F855A" if v >= 1.0 else "#C05621" for v in ordered["speedup_baseline_over_proposal"]]
    fig, ax = plt.subplots(figsize=(max(14, len(ordered) * 0.42), 6.2))
    ax.bar(ordered["dataset"], ordered["speedup_baseline_over_proposal"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / geometry_only")
    ax.set_title("Geometry-only speedup by dataset")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, out_stem)


def _plot_scatter(comp_df: pd.DataFrame, x_col: str, xlabel: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.scatter(comp_df[x_col], comp_df["speedup_baseline_over_proposal"], color=COLOR_CASES["sevennet_geometry_only"], s=70, alpha=0.85)
    for _, row in comp_df.iterrows():
        ax.annotate(row["dataset"], (row[x_col], row["speedup_baseline_over_proposal"]), fontsize=9, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("baseline / geometry_only")
    ax.set_title(f"Geometry-only speedup vs {xlabel}")
    _savefig(fig, out_stem)


def _plot_bucket_summary(comp_df: pd.DataFrame, out_stem: Path) -> None:
    grouped = comp_df.groupby("density_bucket")["speedup_baseline_over_proposal"]
    order = ["small_sparse", "small_dense", "large_sparse", "large_dense"]
    rows = []
    for bucket in order:
        if bucket not in grouped.groups:
            continue
        vals = grouped.get_group(bucket).to_numpy(dtype=np.float64)
        rows.append((bucket, float(np.mean(vals)), float(np.std(vals, ddof=0))))
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    labels = [row[0] for row in rows]
    means = [row[1] for row in rows]
    stds = [row[2] for row in rows]
    ax.bar(labels, means, yerr=stds, color="#4C72B0", capsize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / geometry_only")
    ax.set_title("Geometry-only speedup by size-density bucket")
    _savefig(fig, out_stem)


def _plot_accuracy(summary_df: pd.DataFrame, metric_mean: str, metric_std: str, ylabel: str, out_stem: Path) -> None:
    ordered = summary_df[summary_df["case"] == "sevennet_geometry_only"].sort_values(metric_mean, ascending=False)["dataset"].tolist()
    x = np.arange(len(ordered))
    width = 0.38
    floor = 1e-12
    fig, ax = plt.subplots(figsize=(max(14, len(ordered) * 0.42), 6.5))
    for offset, case in [(-width / 2, "sevennet_baseline"), (width / 2, "sevennet_geometry_only")]:
        frame = summary_df[summary_df["case"] == case].set_index("dataset").loc[ordered]
        ax.bar(
            x + offset,
            np.maximum(frame[metric_mean].to_numpy(dtype=np.float64), floor),
            width,
            yerr=frame[metric_std].to_numpy(dtype=np.float64),
            capsize=3,
            color=COLOR_CASES[case],
            alpha=0.92,
            label=case,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (`mean ± std`, repeat 30)")
    ax.legend()
    _savefig(fig, out_stem)


def _write_reports(output_root: Path, comp_df: pd.DataFrame, accuracy_df: pd.DataFrame) -> None:
    report_dir = ensure_dir(output_root / "reports")
    wins = int((comp_df["speedup_baseline_over_proposal"] > 1.0).sum())
    losses = int((comp_df["speedup_baseline_over_proposal"] <= 1.0).sum())
    median_speedup = float(comp_df["speedup_baseline_over_proposal"].median())
    gmean_speedup = float(np.exp(np.log(comp_df["speedup_baseline_over_proposal"]).mean()))
    large = comp_df[comp_df["num_edges"] >= 3000]
    small = comp_df[comp_df["num_edges"] < 3000]
    dense = comp_df[comp_df["avg_neighbors_directed"] >= 40.0]
    sparse = comp_df[comp_df["avg_neighbors_directed"] < 40.0]
    report = "\n".join(
        [
            "# KCC_new Main Result Summary",
            "",
            f"- datasets: {comp_df['dataset'].nunique()}",
            f"- median speedup: {median_speedup:.6f}",
            f"- geometric mean speedup: {gmean_speedup:.6f}",
            f"- wins: {wins}",
            f"- losses: {losses}",
            "",
            "## Threshold view",
            "",
            f"- num_edges >= 3000: count={len(large)}, win_rate={float((large['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}, median={float(large['speedup_baseline_over_proposal'].median()):.3f}",
            f"- num_edges < 3000: count={len(small)}, win_rate={float((small['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}, median={float(small['speedup_baseline_over_proposal'].median()):.3f}",
            f"- avg_neighbors_directed >= 40: count={len(dense)}, win_rate={float((dense['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}, median={float(dense['speedup_baseline_over_proposal'].median()):.3f}",
            f"- avg_neighbors_directed < 40: count={len(sparse)}, win_rate={float((sparse['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}, median={float(sparse['speedup_baseline_over_proposal'].median()):.3f}",
            "",
            "## Accuracy preservation",
            "",
            f"- baseline median force diff mean: {accuracy_df[accuracy_df['case'] == 'sevennet_baseline']['max_force_diff_mean'].median():.3e} eV/A",
            f"- geometry_only median force diff mean: {accuracy_df[accuracy_df['case'] == 'sevennet_geometry_only']['max_force_diff_mean'].median():.3e} eV/A",
            f"- baseline median energy diff mean: {accuracy_df[accuracy_df['case'] == 'sevennet_baseline']['abs_energy_diff_mean'].median():.3e} eV",
            f"- geometry_only median energy diff mean: {accuracy_df[accuracy_df['case'] == 'sevennet_geometry_only']['abs_energy_diff_mean'].median():.3e} eV",
        ]
    )
    (report_dir / "main_result_summary.md").write_text(report + "\n", encoding="utf-8")


def _plot_lammps_pair_metadata(output_root: Path, out_stem: Path) -> None:
    lammps_summary = output_root / "lammps_pair_metadata_bench" / "metrics" / "lammps_pair_metadata_summary.csv"
    if not lammps_summary.exists():
        return
    df = pd.read_csv(lammps_summary)
    systems = sorted(df["system"].unique())
    keep_cases = ["geometry_only_legacy", "geometry_only_upstream"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
    for ax, metric, title in [
        (axes[0], "pair_metadata_total_ms", "LAMMPS pair metadata"),
        (axes[1], "compute_total_ms", "LAMMPS total compute"),
    ]:
        x = np.arange(len(systems))
        width = 0.34
        for idx, case in enumerate(keep_cases):
            frame = df[df["case"] == case].set_index("system").loc[systems]
            ax.bar(
                x + (-width / 2 if idx == 0 else width / 2),
                frame[f"{metric}_mean"],
                width,
                yerr=frame[f"{metric}_std"],
                capsize=3,
                label=case,
                color=["#A84832", "#2F855A"][idx],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(systems)
        ax.set_title(title)
        ax.set_ylabel("ms")
        ax.legend()
    _savefig(fig, out_stem)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    figures_dir = ensure_dir(output_root / "figures")

    end_summary = _read_csv(output_root / "metrics" / "pair_end_to_end" / "global" / "pair_end_to_end_summary.csv")
    comp_df = _read_csv(output_root / "metrics" / "pair_end_to_end" / "global" / "pair_end_to_end_comparison.csv")
    acc_summary = _read_csv(output_root / "metrics" / "pair_accuracy" / "global" / "pair_accuracy_summary.csv")
    if end_summary.empty or comp_df.empty or acc_summary.empty:
        raise SystemExit("Missing KCC_new metrics; run pair_end_to_end and pair_accuracy first")

    _copy_diagnostic_assets(output_root)
    _write_setup_tables(output_root, end_summary)
    _write_end_to_end_tables(output_root, end_summary, comp_df)
    _write_accuracy_tables(output_root, acc_summary)
    _write_diagnostic_tables(output_root)
    _plot_latency(end_summary, figures_dir / "figure_01_end_to_end_latency")
    _plot_speedup(comp_df, figures_dir / "figure_02_end_to_end_speedup")
    _plot_scatter(comp_df, "num_edges", "Directed edges", figures_dir / "figure_03_speedup_vs_edges")
    _plot_scatter(comp_df, "avg_neighbors_directed", "Average directed neighbors", figures_dir / "figure_04_speedup_vs_neighbors")
    _plot_bucket_summary(comp_df, figures_dir / "figure_05_bucket_speedup")
    _plot_accuracy(acc_summary, "abs_energy_diff_mean", "abs_energy_diff_std", "Absolute energy difference (eV)", figures_dir / "figure_06_accuracy_energy")
    _plot_accuracy(acc_summary, "max_force_diff_mean", "max_force_diff_std", "Maximum absolute force difference (eV/A)", figures_dir / "figure_07_accuracy_force")
    _plot_lammps_pair_metadata(output_root, figures_dir / "figure_08_lammps_pair_metadata")
    _write_reports(output_root, comp_df, acc_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
