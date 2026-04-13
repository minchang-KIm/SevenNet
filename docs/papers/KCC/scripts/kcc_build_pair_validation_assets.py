from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kcc_common import KCC_ROOT, ensure_dir


CASE_COLORS = {
    "baseline": "#0072B2",
    "geometry_only": "#E69F00",
    "full_legacy": "#D55E00",
    "full_no_expand": "#009E73",
}


def _savefig(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _df_to_md(df: pd.DataFrame) -> str:
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


def _plot_case_bars(summary: pd.DataFrame, *, mode: str, metric: str, title: str, ylabel: str, out_stem: Path) -> None:
    frame = summary[summary["mode"] == mode].copy()
    datasets = sorted(frame["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.42), 5.6))
    for idx, case in enumerate(["baseline", "geometry_only", "full_legacy", "full_no_expand"]):
        sub = frame[frame["case"] == case].set_index("dataset").loc[datasets]
        ax.bar(
            x + (idx - 1.5) * width,
            sub[f"{metric}_mean"],
            width,
            yerr=sub[f"{metric}_std"],
            capsize=2.5,
            color=CASE_COLORS[case],
            label=case,
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncols=2)
    _savefig(fig, out_stem)


def _plot_speedup(summary: pd.DataFrame, *, mode: str, proposal_case: str, out_stem: Path) -> None:
    frame = summary[summary["mode"] == mode]
    pivot = frame.pivot(index="dataset", columns="case", values="total_ms_mean" if mode == "step_force" else "model_ms_mean")
    speedup = (pivot["baseline"] / pivot[proposal_case]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(11, len(speedup) * 0.38), 5.0))
    colors = [CASE_COLORS[proposal_case] if v >= 1.0 else "#9C4E63" for v in speedup.to_numpy()]
    ax.bar(speedup.index, speedup.to_numpy(), color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / proposal")
    ax.set_title(f"{mode}: baseline vs {proposal_case}")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, out_stem)


def _plot_metadata(metadata_summary: pd.DataFrame, out_stem: Path) -> None:
    pivot = metadata_summary.pivot(index="dataset", columns="method", values="timing_ms_mean").sort_index()
    std_pivot = metadata_summary.pivot(index="dataset", columns="method", values="timing_ms_std").sort_index()
    methods = ["cpu_original", "cpu_vectorized", "gpu_vectorized_kernel_only"]
    x = np.arange(len(pivot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.42), 5.6))
    colors = {
        "cpu_original": "#A84832",
        "cpu_vectorized": "#4C72B0",
        "gpu_vectorized_kernel_only": "#55A868",
    }
    for idx, method in enumerate(methods):
        ax.bar(
            x + (idx - 1) * width,
            pivot[method],
            width,
            yerr=std_pivot[method],
            capsize=2.5,
            label=method,
            color=colors[method],
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("pair metadata time (ms)")
    ax.set_title("Pair metadata method comparison (`mean ± std`, 100 repeats)")
    ax.legend()
    _savefig(fig, out_stem)


def _plot_patch_delta(summary: pd.DataFrame, *, mode: str, metric: str, out_stem: Path) -> None:
    frame = summary[summary["mode"] == mode].copy()
    pivot = frame.pivot(index="dataset", columns="case", values=f"{metric}_mean").sort_index()
    ratio = (pivot["full_legacy"] / pivot["full_no_expand"]).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(11, len(ratio) * 0.38), 4.8))
    colors = ["#4C72B0" if abs(v - 1.0) <= 0.01 else "#A84832" for v in ratio.to_numpy()]
    ax.bar(ratio.index, ratio.to_numpy(), color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("full_legacy / full_no_expand")
    ax.set_title(f"{mode}: effect of removing edge expansion")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, out_stem)


def _plot_case_median_speedups(summary: pd.DataFrame, out_stem: Path) -> None:
    rows = []
    for mode, metric in [("forward_energy", "model_ms"), ("step_force", "total_ms")]:
        frame = summary[summary["mode"] == mode]
        pivot = frame.pivot(index="dataset", columns="case", values=f"{metric}_mean")
        for case in ["geometry_only", "full_legacy", "full_no_expand"]:
            speedup = pivot["baseline"] / pivot[case]
            rows.append({"mode": mode, "case": case, "median_speedup": float(speedup.median())})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(2)
    width = 0.22
    case_order = ["geometry_only", "full_legacy", "full_no_expand"]
    for idx, case in enumerate(case_order):
        sub = plot_df[plot_df["case"] == case]
        vals = [
            float(sub[sub["mode"] == "forward_energy"]["median_speedup"].iloc[0]),
            float(sub[sub["mode"] == "step_force"]["median_speedup"].iloc[0]),
        ]
        ax.bar(x + (idx - 1) * width, vals, width, label=case, color=CASE_COLORS[case], alpha=0.92)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["forward_energy", "step_force"])
    ax.set_ylabel("median baseline / case")
    ax.set_title("How force/backward changes the pair-execution outcome")
    ax.legend()
    _savefig(fig, out_stem)


def _write_tables(summary: pd.DataFrame, metadata_summary: pd.DataFrame, out_dir: Path) -> None:
    force = summary[summary["mode"] == "step_force"].copy()
    forward = summary[summary["mode"] == "forward_energy"].copy()

    def _table(frame: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        rows = []
        for dataset in sorted(frame["dataset"].unique()):
            sub = frame[frame["dataset"] == dataset].set_index("case")
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": _format_pm(float(sub.loc["baseline", f"{metric_col}_mean"]), float(sub.loc["baseline", f"{metric_col}_std"])),
                    "geometry_only": _format_pm(float(sub.loc["geometry_only", f"{metric_col}_mean"]), float(sub.loc["geometry_only", f"{metric_col}_std"])),
                    "full_legacy": _format_pm(float(sub.loc["full_legacy", f"{metric_col}_mean"]), float(sub.loc["full_legacy", f"{metric_col}_std"])),
                    "full_no_expand": _format_pm(float(sub.loc["full_no_expand", f"{metric_col}_mean"]), float(sub.loc["full_no_expand", f"{metric_col}_std"])),
                    "baseline/full_no_expand": f"{float(sub.loc['baseline', f'{metric_col}_mean'] / sub.loc['full_no_expand', f'{metric_col}_mean']):.3f}x",
                }
            )
        return pd.DataFrame(rows)

    force_table = _table(force, "total_ms")
    forward_table = _table(forward, "model_ms")
    metadata_table = metadata_summary.copy()
    metadata_table["timing_ms"] = [
        _format_pm(m, s, digits=3)
        for m, s in zip(metadata_table["timing_ms_mean"], metadata_table["timing_ms_std"], strict=True)
    ]
    metadata_table = metadata_table[["dataset", "method", "timing_ms"]]

    force_table.to_csv(out_dir / "table_06_force_step_split_summary.csv", index=False)
    (out_dir / "table_06_force_step_split_summary.md").write_text(_df_to_md(force_table))
    forward_table.to_csv(out_dir / "table_07_forward_only_split_summary.csv", index=False)
    (out_dir / "table_07_forward_only_split_summary.md").write_text(_df_to_md(forward_table))
    metadata_table.to_csv(out_dir / "table_08_pair_metadata_method_summary.csv", index=False)
    (out_dir / "table_08_pair_metadata_method_summary.md").write_text(_df_to_md(metadata_table))

    compact_rows = []
    for mode, frame, metric in [
        ("step_force", force, "total_ms"),
        ("forward_energy", forward, "model_ms"),
    ]:
        pivot = frame.pivot(index="dataset", columns="case", values=f"{metric}_mean")
        compact_rows.append(
            {
                "mode": mode,
                "baseline_vs_geometry_only_median": f"{float((pivot['baseline'] / pivot['geometry_only']).median()):.3f}x",
                "baseline_vs_full_legacy_median": f"{float((pivot['baseline'] / pivot['full_legacy']).median()):.3f}x",
                "baseline_vs_full_no_expand_median": f"{float((pivot['baseline'] / pivot['full_no_expand']).median()):.3f}x",
                "full_legacy_vs_full_no_expand_median": f"{float((pivot['full_legacy'] / pivot['full_no_expand']).median()):.3f}x",
            }
        )
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(out_dir / "table_09_split_compact_summary.csv", index=False)
    (out_dir / "table_09_split_compact_summary.md").write_text(_df_to_md(compact))


def _write_report(summary: pd.DataFrame, metadata_summary: pd.DataFrame, out_path: Path) -> None:
    force = summary[summary["mode"] == "step_force"].copy()
    forward = summary[summary["mode"] == "forward_energy"].copy()
    force_pivot = force.pivot(index="dataset", columns="case", values="total_ms_mean")
    forward_pivot = forward.pivot(index="dataset", columns="case", values="model_ms_mean")
    lines = [
        "# Pair Validation Split Report",
        "",
        "## 각 실험의 의미",
        "",
        "- `baseline vs geometry_only`: 기하 정보 재사용만의 효과를 본다. 메시지 계산 스케줄 변경은 최소화된다.",
        "- `baseline vs full_legacy`: 기존 full pair 실행 전체 효과를 본다. 여기에는 pair 재사용과 forward/reverse 분리, edge 확장이 함께 들어간다.",
        "- `baseline vs full_no_expand`: full 실행에서 불필요한 edge 확장을 줄였을 때 실제 개선이 생기는지 본다.",
        "- `step_force`: 실제 MD 한 step에 가까운 경로다. 그래프 생성, pair metadata, device 이동, 모델 실행, 힘 계산까지 포함한다.",
        "- `forward_energy`: 입력 그래프를 미리 만든 뒤 에너지만 계산한다. 순수 forward 경로에서 geometry 재사용이 얼마나 듣는지 본다.",
        "- `pair metadata method comparison`: 현재 CPU pair metadata가 병목인지, 벡터화/GPU화 여지가 얼마나 있는지 본다.",
        "",
        "## 핵심 비교 지표",
        "",
        f"- step_force median baseline/full_no_expand: {(force_pivot['baseline'] / force_pivot['full_no_expand']).median():.3f}x",
        f"- step_force median baseline/geometry_only: {(force_pivot['baseline'] / force_pivot['geometry_only']).median():.3f}x",
        f"- step_force median full_legacy/full_no_expand: {(force_pivot['full_legacy'] / force_pivot['full_no_expand']).median():.3f}x",
        f"- forward_energy median baseline/full_no_expand: {(forward_pivot['baseline'] / forward_pivot['full_no_expand']).median():.3f}x",
        f"- forward_energy median baseline/geometry_only: {(forward_pivot['baseline'] / forward_pivot['geometry_only']).median():.3f}x",
        f"- forward_energy median full_legacy/full_no_expand: {(forward_pivot['full_legacy'] / forward_pivot['full_no_expand']).median():.3f}x",
        f"- geometry_only: step/forward speedup ratio median = {(((force_pivot['baseline'] / force_pivot['geometry_only']) / (forward_pivot['baseline'] / forward_pivot['geometry_only'])).median()):.3f}",
        f"- full_legacy: step/forward speedup ratio median = {(((force_pivot['baseline'] / force_pivot['full_legacy']) / (forward_pivot['baseline'] / forward_pivot['full_legacy'])).median()):.3f}",
        f"- full_no_expand: step/forward speedup ratio median = {(((force_pivot['baseline'] / force_pivot['full_no_expand']) / (forward_pivot['baseline'] / forward_pivot['full_no_expand'])).median()):.3f}",
        "",
        "## pair metadata 비교",
        "",
    ]
    meta_pivot = metadata_summary.pivot(index="dataset", columns="method", values="timing_ms_mean")
    lines.extend(
        [
            f"- median cpu_original: {meta_pivot['cpu_original'].median():.3f} ms",
            f"- median cpu_vectorized: {meta_pivot['cpu_vectorized'].median():.3f} ms",
            f"- median gpu_vectorized_kernel_only: {meta_pivot['gpu_vectorized_kernel_only'].median():.3f} ms",
            "",
            "## 앞으로의 연구 방향",
            "",
            "- 큰 그래프에서 forward 쪽 이득이 실제 step 전체로 전달되는지와, force backward가 이를 얼마나 상쇄하는지를 함께 봐야 한다.",
            "- full 경로는 pair 상태를 끝까지 유지하지 못하는 구간이 아직 남아 있으므로, pair-major tensor product와 reduction으로 이어져야 한다.",
            "- CPU pair metadata는 현재 구조에서 실사용 경로에 남아 있으므로, 벡터화 또는 GPU 이관의 실익을 별도로 검증할 가치가 있다.",
            "- FlashTP 결합은 이 결과 위에 올리는 후속 단계로 두는 것이 맞다. 먼저 SevenNet 기본 경로에서 어디서 이득이 생기는지 분리해야 한다.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    summary = pd.read_csv(output_root / "metrics" / "pair_validation_split" / "global" / "pair_validation_summary.csv")
    metadata_summary = pd.read_csv(output_root / "metrics" / "pair_validation_split" / "global" / "pair_metadata_summary.csv")

    fig_dir = ensure_dir(output_root / "figures" / "pair_validation_split")
    table_dir = ensure_dir(output_root / "tables")
    report_dir = ensure_dir(output_root / "reports")

    _plot_case_bars(
        summary,
        mode="step_force",
        metric="total_ms",
        title="Force-included step timing (`mean ± std`, 100 repeats)",
        ylabel="Step latency (ms)",
        out_stem=fig_dir / "pair_validation_step_force_latency_all",
    )
    _plot_case_bars(
        summary,
        mode="forward_energy",
        metric="model_ms",
        title="Forward-only energy timing (`mean ± std`, 100 repeats)",
        ylabel="Model forward latency (ms)",
        out_stem=fig_dir / "pair_validation_forward_energy_latency_all",
    )
    _plot_speedup(summary, mode="step_force", proposal_case="geometry_only", out_stem=fig_dir / "pair_validation_step_force_speedup_geometry_only")
    _plot_speedup(summary, mode="step_force", proposal_case="full_no_expand", out_stem=fig_dir / "pair_validation_step_force_speedup_full_no_expand")
    _plot_speedup(summary, mode="forward_energy", proposal_case="geometry_only", out_stem=fig_dir / "pair_validation_forward_energy_speedup_geometry_only")
    _plot_speedup(summary, mode="forward_energy", proposal_case="full_no_expand", out_stem=fig_dir / "pair_validation_forward_energy_speedup_full_no_expand")
    _plot_metadata(metadata_summary, fig_dir / "pair_validation_pair_metadata_methods")
    _plot_patch_delta(summary, mode="step_force", metric="total_ms", out_stem=fig_dir / "pair_validation_step_force_patch_delta")
    _plot_patch_delta(summary, mode="forward_energy", metric="model_ms", out_stem=fig_dir / "pair_validation_forward_energy_patch_delta")
    _plot_case_median_speedups(summary, fig_dir / "pair_validation_case_median_speedups")

    _write_tables(summary, metadata_summary, table_dir)
    _write_report(summary, metadata_summary, report_dir / "pair_validation_split_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
