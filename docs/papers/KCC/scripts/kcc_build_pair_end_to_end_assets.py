from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kcc_common import COLOR_CASES, KCC_ROOT, ensure_dir


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _savefig(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _write_md_table(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if df.empty:
        path.write_text("데이터 없음\n")
        return
    path.write_text(_df_to_md(df) + "\n")


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
    return "\n".join(lines)


def _format_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _write_pair_tables(summary_df: pd.DataFrame, comp_df: pd.DataFrame, out_dir: Path) -> None:
    pair_summary = pd.DataFrame(
        {
            "dataset": comp_df["dataset"],
            "baseline (ms)": [
                _format_pm(mean, std)
                for mean, std in zip(comp_df["baseline_mean_ms"], comp_df["baseline_std_ms"], strict=True)
            ],
            "proposal (ms)": [
                _format_pm(mean, std)
                for mean, std in zip(comp_df["proposal_mean_ms"], comp_df["proposal_std_ms"], strict=True)
            ],
            "speedup (baseline/proposal)": comp_df["speedup_baseline_over_proposal"].map(lambda x: f"{x:.3f}x"),
            "natoms": comp_df["natoms"],
            "num_edges": comp_df["num_edges"],
            "avg_neighbors_directed": comp_df["avg_neighbors_directed"].map(lambda x: f"{x:.2f}"),
            "density_bucket": comp_df["density_bucket"],
        }
    ).sort_values("dataset").reset_index(drop=True)
    pair_summary.to_csv(out_dir / "table_02_pair_end_to_end_summary.csv", index=False)
    _write_md_table(out_dir / "table_02_pair_end_to_end_summary.md", pair_summary)

    rows: list[dict[str, object]] = []

    def add_condition_row(group: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "group": group,
                "count": int(len(subset)),
                "wins": int((subset["speedup_baseline_over_proposal"] > 1.0).sum()),
                "win_rate": f"{float((subset['speedup_baseline_over_proposal'] > 1.0).mean()):.3f}",
                "mean_speedup": f"{float(subset['speedup_baseline_over_proposal'].mean()):.3f}",
                "median_speedup": f"{float(subset['speedup_baseline_over_proposal'].median()):.3f}",
                "std_speedup": f"{float(subset['speedup_baseline_over_proposal'].std(ddof=0)):.3f}",
            }
        )

    add_condition_row("num_edges >= 3000", comp_df[comp_df["num_edges"] >= 3000])
    add_condition_row("num_edges < 3000", comp_df[comp_df["num_edges"] < 3000])
    add_condition_row(
        "avg_neighbors_directed >= 40",
        comp_df[comp_df["avg_neighbors_directed"] >= 40.0],
    )
    add_condition_row(
        "avg_neighbors_directed < 40",
        comp_df[comp_df["avg_neighbors_directed"] < 40.0],
    )
    for bucket in ["small_sparse", "small_dense", "large_sparse", "large_dense"]:
        add_condition_row(bucket, comp_df[comp_df["density_bucket"] == bucket])

    condition_df = pd.DataFrame(rows)
    condition_df.to_csv(out_dir / "table_03_pair_condition_summary.csv", index=False)
    _write_md_table(out_dir / "table_03_pair_condition_summary.md", condition_df)

    proposal_acc = summary_df[summary_df["case"] == "e3nn_pair_full"][
        ["dataset", "abs_energy_diff_vs_e3nn", "max_abs_force_diff_vs_e3nn"]
    ].copy()
    accuracy_df = proposal_acc.merge(
        comp_df[["dataset", "natoms", "num_edges", "avg_neighbors_directed", "density_bucket"]],
        on="dataset",
        how="left",
    )
    accuracy_df["abs_energy_diff_vs_e3nn"] = accuracy_df["abs_energy_diff_vs_e3nn"].map(lambda x: f"{x:.6g}")
    accuracy_df["max_abs_force_diff_vs_e3nn"] = accuracy_df["max_abs_force_diff_vs_e3nn"].map(lambda x: f"{x:.6g}")
    accuracy_df = accuracy_df.sort_values(
        by=["max_abs_force_diff_vs_e3nn", "abs_energy_diff_vs_e3nn"],
        ascending=False,
        key=lambda s: s.astype(float),
    ).reset_index(drop=True)
    accuracy_df.to_csv(out_dir / "table_04_pair_accuracy_summary.csv", index=False)
    _write_md_table(out_dir / "table_04_pair_accuracy_summary.md", accuracy_df)


def _plot_latency(summary_df: pd.DataFrame, out_stem: Path) -> None:
    pivot_mean = summary_df.pivot(index="dataset", columns="case", values="mean_ms").sort_index()
    pivot_std = summary_df.pivot(index="dataset", columns="case", values="std_ms").sort_index()
    fig, ax = plt.subplots(figsize=(max(11, len(pivot_mean) * 0.38), 5.4))
    x = np.arange(len(pivot_mean))
    width = 0.38
    for offset, case in [(-width / 2, "e3nn_baseline"), (width / 2, "e3nn_pair_full")]:
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
    ax.set_title("SevenNet baseline vs proposal-only end-to-end latency")
    ax.legend()
    _savefig(fig, out_stem)


def _plot_speedup(comp_df: pd.DataFrame, out_stem: Path) -> None:
    ordered = comp_df.sort_values("speedup_baseline_over_proposal", ascending=False)
    colors = [COLOR_CASES["e3nn_pair_full"] if v >= 1.0 else "#9C4E63" for v in ordered["speedup_baseline_over_proposal"]]
    fig, ax = plt.subplots(figsize=(max(11, len(ordered) * 0.38), 5.0))
    ax.bar(ordered["dataset"], ordered["speedup_baseline_over_proposal"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / proposal")
    ax.set_title("Proposal-only speedup by dataset")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, out_stem)


def _plot_scatter(comp_df: pd.DataFrame, x_col: str, xlabel: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        comp_df[x_col],
        comp_df["speedup_baseline_over_proposal"],
        color=COLOR_CASES["e3nn_pair_full"],
        alpha=0.85,
    )
    for _, row in comp_df.iterrows():
        ax.annotate(row["dataset"], (row[x_col], row["speedup_baseline_over_proposal"]), fontsize=7, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("baseline / proposal")
    ax.set_title(f"Proposal-only speedup vs {xlabel}")
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
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [row[0] for row in rows]
    means = [row[1] for row in rows]
    stds = [row[2] for row in rows]
    ax.bar(labels, means, yerr=stds, color="#4C72B0", capsize=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / proposal")
    ax.set_title("Proposal-only speedup by size-density bucket")
    _savefig(fig, out_stem)


def _plot_size_density_map(comp_df: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    norm = plt.Normalize(comp_df["speedup_baseline_over_proposal"].min(), comp_df["speedup_baseline_over_proposal"].max())
    cmap = plt.cm.RdYlGn
    sc = ax.scatter(
        comp_df["num_edges"],
        comp_df["avg_neighbors_directed"],
        c=comp_df["speedup_baseline_over_proposal"],
        cmap=cmap,
        norm=norm,
        s=48,
        alpha=0.9,
    )
    for _, row in comp_df.iterrows():
        ax.annotate(row["dataset"], (row["num_edges"], row["avg_neighbors_directed"]), fontsize=7, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("baseline / proposal")
    ax.set_xlabel("Directed edges")
    ax.set_ylabel("Average directed neighbors")
    ax.set_title("Where proposal-only helps on the size-density map")
    _savefig(fig, out_stem)


def _write_condition_report(comp_df: pd.DataFrame, out_path: Path) -> None:
    spearman_edges = comp_df["num_edges"].corr(comp_df["speedup_baseline_over_proposal"], method="spearman")
    spearman_neighbors = comp_df["avg_neighbors_directed"].corr(comp_df["speedup_baseline_over_proposal"], method="spearman")
    spearman_natoms = comp_df["natoms"].corr(comp_df["speedup_baseline_over_proposal"], method="spearman")
    bucket = comp_df.groupby("density_bucket")["speedup_baseline_over_proposal"].agg(["count", "mean", "median", "std"])
    bucket["wins"] = comp_df.groupby("density_bucket")["speedup_baseline_over_proposal"].apply(lambda s: int((s > 1.0).sum()))
    bucket["win_rate"] = comp_df.groupby("density_bucket")["speedup_baseline_over_proposal"].apply(lambda s: float((s > 1.0).mean()))
    wins = comp_df[comp_df["speedup_baseline_over_proposal"] > 1.0].copy()
    losses = comp_df[comp_df["speedup_baseline_over_proposal"] <= 1.0].copy()
    gmean = float(np.exp(np.log(comp_df["speedup_baseline_over_proposal"]).mean()))
    edge_large = comp_df[comp_df["num_edges"] >= 3000]
    edge_small = comp_df[comp_df["num_edges"] < 3000]
    dense = comp_df[comp_df["avg_neighbors_directed"] >= 40.0]
    nondense = comp_df[comp_df["avg_neighbors_directed"] < 40.0]
    lines = [
        "# SevenNet Baseline vs Proposal-Only Condition Analysis",
        "",
        f"- datasets: {comp_df['dataset'].nunique()}",
        f"- median speedup: {comp_df['speedup_baseline_over_proposal'].median():.6f}",
        f"- geometric mean speedup: {gmean:.6f}",
        f"- wins: {(comp_df['speedup_baseline_over_proposal'] > 1.0).sum()}",
        f"- losses: {(comp_df['speedup_baseline_over_proposal'] <= 1.0).sum()}",
        "",
        "## Spearman correlations",
        "",
        f"- num_edges vs speedup: {spearman_edges:.6f}",
        f"- avg_neighbors_directed vs speedup: {spearman_neighbors:.6f}",
        f"- natoms vs speedup: {spearman_natoms:.6f}",
        "",
        "## Simple threshold view",
        "",
        f"- num_edges >= 3000: count={len(edge_large)}, win_rate={float((edge_large['speedup_baseline_over_proposal'] > 1.0).mean()):.6f}, median={edge_large['speedup_baseline_over_proposal'].median():.6f}",
        f"- num_edges < 3000: count={len(edge_small)}, win_rate={float((edge_small['speedup_baseline_over_proposal'] > 1.0).mean()):.6f}, median={edge_small['speedup_baseline_over_proposal'].median():.6f}",
        f"- avg_neighbors_directed >= 40: count={len(dense)}, win_rate={float((dense['speedup_baseline_over_proposal'] > 1.0).mean()):.6f}, median={dense['speedup_baseline_over_proposal'].median():.6f}",
        f"- avg_neighbors_directed < 40: count={len(nondense)}, win_rate={float((nondense['speedup_baseline_over_proposal'] > 1.0).mean()):.6f}, median={nondense['speedup_baseline_over_proposal'].median():.6f}",
        "",
        "## Bucket summary",
        "",
        _df_to_md(bucket.reset_index()),
        "",
        "## Top wins",
        "",
        _df_to_md(
            wins.sort_values("speedup_baseline_over_proposal", ascending=False)
            .head(10)
            .reset_index(drop=True)
        ),
        "",
        "## Top losses",
        "",
        _df_to_md(
            losses.sort_values("speedup_baseline_over_proposal", ascending=True)
            .head(10)
            .reset_index(drop=True)
        ),
        "",
        "## Interpretation",
        "",
        "The proposal should be claimed as beneficial under the workload conditions that actually show positive speedup here.",
        "In this result set, the strongest practical separator is graph size: graphs with num_edges < 3000 never win, whereas graphs with num_edges >= 3000 win in most cases.",
        "High-neighbor workloads also align strongly with improvement, but large graph size is the more defensible first condition.",
        "FlashTP synergy remains future work; this report is strictly about SevenNet baseline versus proposal-only.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _write_tables_and_figures_index(output_root: Path) -> None:
    notes = [
        "# KCC Tables and Figures",
        "",
        "## Main Tables",
        "",
        "- `tables/table_01_comparability.md`",
        "- `tables/table_02_pair_end_to_end_summary.md`",
        "- `tables/table_03_pair_condition_summary.md`",
        "- `tables/table_04_pair_accuracy_summary.md`",
        "- `tables/table_05_representative_stage_summary.md`",
        "",
        "## Supplementary Tables",
        "",
        "- `tables/table_s01_flash_summary.md`",
        "- `tables/table_s02_nsys_kernel_groups.md`",
        "",
        "## Main Figures",
        "",
        "- `figure_00_comparability_diagram`: 계측 family 간 직접 비교 가능 범위 설명",
        "- `pair_latency_all`: 31개 데이터셋 SevenNet baseline vs proposal-only latency mean±std",
        "- `pair_speedup_all`: baseline 대비 proposal-only speedup",
        "- `pair_speedup_vs_num_edges`: 그래프 크기와 speedup 관계",
        "- `pair_speedup_vs_avg_neighbors`: 평균 이웃 수와 speedup 관계",
        "- `pair_speedup_by_bucket`: size-density bucket별 speedup",
        "- `pair_speedup_size_density_map`: speedup을 색으로 표현한 size-density map",
        "- `figure_04_representative_stage_breakdown`: representative detailed stage stacked bar",
        "",
        "## Supplementary Figures",
        "",
        "- `figure_01_flash_latency_all`: FlashTP end-to-end latency mean±std",
        "- `figure_02_flash_speedup_all`: Flash baseline 대비 flash pair auto speedup",
        "- `figure_03_dataset_map`: 데이터셋 size-density map",
        "- `figure_05_sh_share_vs_intrusive_pair_speedup`: SH share와 intrusive pair speedup 관계",
        "- `figures/four_case/stage_breakdown_all_option_none.png`",
        "- `figures/four_case/stage_breakdown_all_flashtp_only.png`",
        "- `figures/four_case/stage_breakdown_all_proposal_only.png`",
        "- `figures/four_case/stage_breakdown_all_flashtp_plus_proposal.png`",
        "- `figures/four_case/model_total_heatmap.png`",
        "",
    ]
    (output_root / "tables_and_figures.md").write_text("\n".join(notes))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    global_dir = output_root / "metrics" / "pair_end_to_end" / "global"
    figures_dir = ensure_dir(output_root / "figures" / "pair_end_to_end")
    reports_dir = ensure_dir(output_root / "reports")
    tables_dir = ensure_dir(output_root / "tables")

    summary_df = _read_csv(global_dir / "pair_end_to_end_summary.csv")
    comp_df = _read_csv(global_dir / "pair_end_to_end_comparison.csv")
    if summary_df.empty or comp_df.empty:
        raise SystemExit("pair_end_to_end results not found; run kcc_pair_end_to_end.py first")

    _write_pair_tables(summary_df, comp_df, tables_dir)
    _plot_latency(summary_df, figures_dir / "pair_latency_all")
    _plot_speedup(comp_df, figures_dir / "pair_speedup_all")
    _plot_scatter(comp_df, "num_edges", "Directed edges", figures_dir / "pair_speedup_vs_num_edges")
    _plot_scatter(comp_df, "avg_neighbors_directed", "Average directed neighbors", figures_dir / "pair_speedup_vs_avg_neighbors")
    _plot_bucket_summary(comp_df, figures_dir / "pair_speedup_by_bucket")
    _plot_size_density_map(comp_df, figures_dir / "pair_speedup_size_density_map")
    _write_condition_report(comp_df, reports_dir / "pair_end_to_end_condition_analysis.md")
    _write_tables_and_figures_index(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
