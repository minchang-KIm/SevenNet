from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kcc_common import COLOR_CASES, COLOR_STAGES, KCC_ROOT, PAPER_STAGE_ORDER, REPRESENTATIVE_NSYS_DATASETS, ensure_dir


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_md_table(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if df.empty:
        path.write_text("데이터 없음 또는 canonical 결과에서 제외됨.\n")
        return
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def _savefig(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _format_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _figure_manifest_row(name: str, datasets: Iterable[str], purpose: str) -> dict[str, str]:
    return {
        "figure": name,
        "datasets": ", ".join(sorted(set(datasets))),
        "purpose": purpose,
    }


def _plot_flash_latency(summary_df: pd.DataFrame, out_stem: Path) -> None:
    pivot_mean = summary_df.pivot(index="dataset", columns="case", values="mean_ms").sort_index()
    pivot_std = summary_df.pivot(index="dataset", columns="case", values="std_ms").sort_index()
    fig, ax = plt.subplots(figsize=(max(11, len(pivot_mean) * 0.38), 5.4))
    x = np.arange(len(pivot_mean))
    width = 0.38
    for offset, case in [(-width / 2, "flash_baseline"), (width / 2, "flash_pair_auto")]:
        ax.bar(
            x + offset,
            pivot_mean[case],
            width,
            label=case,
            color=COLOR_CASES[case],
            yerr=pivot_std[case],
            capsize=2.5,
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_mean.index, rotation=45, ha="right")
    ax.set_ylabel("Steady-state latency (ms)")
    ax.set_title("FlashTP end-to-end latency across 31 datasets")
    ax.legend()
    _savefig(fig, out_stem)


def _plot_flash_speedup(flash_cmp_df: pd.DataFrame, out_stem: Path) -> None:
    ordered = flash_cmp_df.sort_values("flash_speedup_baseline_over_pair", ascending=False)
    colors = [
        COLOR_CASES["flash_pair_auto"] if value >= 1.0 else "#9C4E63"
        for value in ordered["flash_speedup_baseline_over_pair"]
    ]
    fig, ax = plt.subplots(figsize=(max(11, len(ordered) * 0.38), 5.0))
    ax.bar(ordered["dataset"], ordered["flash_speedup_baseline_over_pair"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("flash_baseline / flash_pair_auto")
    ax.set_title("FlashTP pair reuse speedup by dataset")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, out_stem)


def _plot_dataset_map(manifest_df: pd.DataFrame, out_stem: Path) -> None:
    bucket_colors = {
        "small_sparse": "#4E79A7",
        "small_dense": "#F28E2B",
        "large_sparse": "#59A14F",
        "large_dense": "#E15759",
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for bucket, frame in manifest_df.groupby("density_bucket"):
        ax.scatter(
            frame["num_edges"],
            frame["avg_neighbors_directed"],
            label=bucket,
            color=bucket_colors.get(bucket, "#999999"),
            s=44,
            alpha=0.85,
        )
        for _, row in frame.iterrows():
            ax.annotate(row["dataset"], (row["num_edges"], row["avg_neighbors_directed"]), fontsize=7, alpha=0.85)
    ax.set_xlabel("Directed edges")
    ax.set_ylabel("Average directed neighbors")
    ax.set_title("Dataset size-density map")
    ax.legend()
    _savefig(fig, out_stem)


def _plot_sh_share_vs_speedup(stage_ms_df: pd.DataFrame, out_stem: Path) -> None:
    pivot = stage_ms_df.pivot(index=["dataset", "case"], columns="stage", values="mean_ms").reset_index()
    baseline = pivot[pivot["case"] == "e3nn_baseline"].copy()
    pair = pivot[pivot["case"] == "e3nn_pair_full"].copy()
    merged = baseline.merge(pair[["dataset", "model_total_ms"]], on="dataset", suffixes=("_baseline", "_pair"))
    merged["sh_share_baseline"] = merged["spherical_harmonics_ms"] / merged["model_total_ms_baseline"]
    merged["intrusive_pair_speedup"] = merged["model_total_ms_baseline"] / merged["model_total_ms_pair"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        merged["sh_share_baseline"],
        merged["intrusive_pair_speedup"],
        color=COLOR_STAGES["spherical_harmonics_ms"],
        alpha=0.88,
    )
    for _, row in merged.iterrows():
        ax.annotate(row["dataset"], (row["sh_share_baseline"], row["intrusive_pair_speedup"]), fontsize=7, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Baseline SH share of intrusive model total")
    ax.set_ylabel("Intrusive baseline / pair model_total")
    ax.set_title("Where higher SH share aligns with pair benefit")
    _savefig(fig, out_stem)


def _plot_representative_stages(stage_ms_df: pd.DataFrame, out_stem: Path) -> None:
    rep_df = stage_ms_df[stage_ms_df["dataset"].isin(REPRESENTATIVE_NSYS_DATASETS)].copy()
    rep_df["label"] = rep_df["dataset"] + "\n" + rep_df["case"].str.replace("e3nn_", "", regex=False)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    bottom = np.zeros(len(rep_df), dtype=np.float64)
    for stage in PAPER_STAGE_ORDER:
        col = stage if stage in rep_df.columns else f"{stage}_mean"
        if col not in rep_df.columns:
            values = np.zeros(len(rep_df), dtype=np.float64)
        else:
            values = rep_df[col].to_numpy(dtype=np.float64)
        ax.bar(
            rep_df["label"],
            values,
            bottom=bottom,
            color=COLOR_STAGES.get(stage, "#999999"),
            label=stage,
        )
        bottom += values
    ax.set_ylabel("Intrusive stage time (ms)")
    ax.set_title("Representative detailed stage breakdown")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=3, fontsize=8)
    _savefig(fig, out_stem)


def _plot_nsys_groups(nsys_df: pd.DataFrame, out_stem: Path) -> None:
    if nsys_df.empty:
        return
    rep = nsys_df[nsys_df["dataset"].isin(REPRESENTATIVE_NSYS_DATASETS)].copy()
    rep["label"] = rep["dataset"] + "\n" + rep["case"]
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    bottom = np.zeros(len(rep["label"].drop_duplicates()), dtype=np.float64)
    labels = sorted(rep["label"].drop_duplicates().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for group, frame in rep.groupby("kernel_group"):
        values = np.zeros(len(labels), dtype=np.float64)
        for _, row in frame.iterrows():
            values[label_to_idx[row["label"]]] = float(row["mean_share"])
        ax.bar(labels, values, bottom=bottom, label=group)
        bottom += values
    ax.set_ylabel("Kernel share")
    ax.set_title("Representative Nsight kernel-group mix")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _savefig(fig, out_stem)


def _plot_comparability_diagram(out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, 0.24, 0.25, "e3nn baseline detailed\n(intrusive, stage analysis)", COLOR_CASES["e3nn_baseline"]),
        (0.37, 0.55, 0.24, 0.25, "e3nn pair detailed\n(intrusive, stage analysis)", COLOR_CASES["e3nn_pair_full"]),
        (0.69, 0.55, 0.24, 0.25, "FlashTP end-to-end\n(non-intrusive latency)", COLOR_CASES["flash_baseline"]),
        (0.37, 0.12, 0.24, 0.2, "Representative Nsight\n(kernel validation)", "#888888"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.18, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0.37, 0.67), xytext=(0.29, 0.67), arrowprops={"arrowstyle": "<->", "linewidth": 1.6})
    ax.text(0.33, 0.71, "direct comparison", ha="center", fontsize=9)
    ax.annotate("", xy=(0.81, 0.48), xytext=(0.49, 0.32), arrowprops={"arrowstyle": "-", "linestyle": "--"})
    ax.text(0.78, 0.3, "not direct\ncomparison", ha="center", fontsize=9)
    ax.text(0.5, 0.03, "Use detailed families for stage shares/load. Use Flash family for headline latency.", ha="center", fontsize=9)
    _savefig(fig, out_stem)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    figures_dir = ensure_dir(output_root / "figures")
    tables_dir = ensure_dir(output_root / "tables")
    metrics_global_dir = ensure_dir(output_root / "metrics" / "global")

    manifest_df = _read_csv(metrics_global_dir / "dataset_manifest.csv")
    flash_summary_df = _read_csv(metrics_global_dir / "flash_end_to_end_summary.csv")
    flash_cmp_df = _read_csv(metrics_global_dir / "flash_comparison.csv")
    detailed_stage_df = _read_csv(metrics_global_dir / "detailed_stage_mean_std.csv")
    detailed_stage_long_df = _read_csv(metrics_global_dir / "detailed_stage_long_mean_std.csv")
    comparability_df = _read_csv(metrics_global_dir / "comparability_legend.csv")
    nsys_summary_df = _read_csv(metrics_global_dir / "nsys_kernel_group_summary.csv")

    if detailed_stage_df.empty and not detailed_stage_long_df.empty:
        detailed_stage_df = detailed_stage_long_df.pivot_table(
            index=["dataset", "sample_id", "modal", "natoms", "case"],
            columns="stage",
            values="mean_ms",
            aggfunc="first",
        ).reset_index()

    figure_manifest: list[dict[str, str]] = []

    comparability_table = comparability_df.copy()
    _write_md_table(tables_dir / "table_01_comparability.md", comparability_table)
    comparability_table.to_csv(tables_dir / "table_01_comparability.csv", index=False)

    flash_summary_table = flash_cmp_df.merge(
        manifest_df[["dataset", "natoms", "num_edges", "avg_neighbors_directed", "density_bucket"]],
        on="dataset",
        how="left",
    )
    _write_md_table(tables_dir / "table_s01_flash_summary.md", flash_summary_table)
    flash_summary_table.to_csv(tables_dir / "table_s01_flash_summary.csv", index=False)
    _write_md_table(tables_dir / "table_02_flash_summary.md", flash_summary_table)
    flash_summary_table.to_csv(tables_dir / "table_02_flash_summary.csv", index=False)

    rep_stage_rows = []
    if not detailed_stage_df.empty:
        rep = detailed_stage_df[detailed_stage_df["dataset"].isin(REPRESENTATIVE_NSYS_DATASETS)].copy()
        for _, row in rep.iterrows():
            rep_stage_rows.append(
                {
                    "dataset": row["dataset"],
                    "case": row["case"],
                    "SH (ms)": _format_pm(row.get("spherical_harmonics_ms_mean", np.nan), row.get("spherical_harmonics_ms_std", np.nan)),
                    "WeightNN (ms)": _format_pm(row.get("conv_weight_nn_ms_mean", np.nan), row.get("conv_weight_nn_ms_std", np.nan)),
                    "TP (ms)": _format_pm(row.get("conv_message_tp_ms_mean", np.nan), row.get("conv_message_tp_ms_std", np.nan)),
                    "Force (ms)": _format_pm(row.get("top_force_output_ms_mean", np.nan), row.get("top_force_output_ms_std", np.nan)),
                    "Model total (ms)": _format_pm(row.get("model_total_ms_mean", np.nan), row.get("model_total_ms_std", np.nan)),
                }
            )
    rep_stage_table = pd.DataFrame(rep_stage_rows)
    _write_md_table(tables_dir / "table_05_representative_stage_summary.md", rep_stage_table)
    rep_stage_table.to_csv(tables_dir / "table_05_representative_stage_summary.csv", index=False)
    _write_md_table(tables_dir / "table_03_representative_stage_summary.md", rep_stage_table)
    rep_stage_table.to_csv(tables_dir / "table_03_representative_stage_summary.csv", index=False)

    if not nsys_summary_df.empty:
        nsys_table = nsys_summary_df.copy()
        nsys_table["mean_share"] = nsys_table["mean_share"].map(lambda x: f"{x:.3f}")
        nsys_table["std_share"] = nsys_table["std_share"].map(lambda x: f"{x:.3f}")
    else:
        nsys_table = pd.DataFrame(columns=["dataset", "case", "kernel_group", "mean_share", "std_share"])
    _write_md_table(tables_dir / "table_s02_nsys_kernel_groups.md", nsys_table)
    nsys_table.to_csv(tables_dir / "table_s02_nsys_kernel_groups.csv", index=False)
    _write_md_table(tables_dir / "table_04_nsys_kernel_groups.md", nsys_table)
    nsys_table.to_csv(tables_dir / "table_04_nsys_kernel_groups.csv", index=False)

    if not flash_summary_df.empty:
        _plot_flash_latency(flash_summary_df, figures_dir / "figure_01_flash_latency_all")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_01_flash_latency_all",
                flash_summary_df["dataset"].tolist(),
                "31개 데이터셋 FlashTP end-to-end latency mean±std",
            )
        )
    if not flash_cmp_df.empty:
        _plot_flash_speedup(flash_cmp_df, figures_dir / "figure_02_flash_speedup_all")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_02_flash_speedup_all",
                flash_cmp_df["dataset"].tolist(),
                "Flash baseline 대비 flash pair auto speedup",
            )
        )
    if not manifest_df.empty:
        _plot_dataset_map(manifest_df, figures_dir / "figure_03_dataset_map")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_03_dataset_map",
                manifest_df["dataset"].tolist(),
                "데이터셋 size-density map",
            )
        )
    if not detailed_stage_df.empty:
        _plot_representative_stages(detailed_stage_df, figures_dir / "figure_04_representative_stage_breakdown")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_04_representative_stage_breakdown",
                REPRESENTATIVE_NSYS_DATASETS,
                "대표 데이터셋 detailed stage stacked bar",
            )
        )
    if not detailed_stage_long_df.empty:
        _plot_sh_share_vs_speedup(detailed_stage_long_df, figures_dir / "figure_05_sh_share_vs_intrusive_pair_speedup")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_05_sh_share_vs_intrusive_pair_speedup",
                detailed_stage_long_df["dataset"].tolist(),
                "SH share와 intrusive pair speedup 관계",
            )
        )
    if not nsys_summary_df.empty:
        _plot_nsys_groups(nsys_summary_df, figures_dir / "figure_06_nsys_kernel_groups")
        figure_manifest.append(
            _figure_manifest_row(
                "figure_06_nsys_kernel_groups",
                nsys_summary_df["dataset"].tolist(),
                "대표 데이터셋 Nsight kernel-group mix",
            )
        )
    _plot_comparability_diagram(figures_dir / "figure_00_comparability_diagram")
    figure_manifest.append(
        _figure_manifest_row(
            "figure_00_comparability_diagram",
            (),
            "계측 family 간 직접 비교 가능 범위 설명",
        )
    )

    figure_manifest_df = pd.DataFrame(figure_manifest)
    figure_manifest_df.to_csv(figures_dir / "figure_manifest.csv", index=False)

    notes = [
        "# KCC Tables and Figures",
        "",
        "## Tables",
        "",
        "- `tables/table_01_comparability.md`",
        "- `tables/table_02_flash_summary.md`",
        "- `tables/table_03_representative_stage_summary.md`",
        "- `tables/table_04_nsys_kernel_groups.md`",
        "",
        "## Figures",
        "",
    ]
    for _, row in figure_manifest_df.iterrows():
        notes.append(f"- `{row['figure']}`: {row['purpose']}")
    (output_root / "tables_and_figures.md").write_text("\n".join(notes) + "\n")
    print(figures_dir / "figure_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
