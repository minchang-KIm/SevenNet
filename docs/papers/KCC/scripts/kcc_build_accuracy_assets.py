from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kcc_common import COLOR_CASES, KCC_ROOT, ensure_dir


EPS_ENERGY = 1e-12
EPS_FORCE = 1e-12


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


def _format_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}e} ± {std:.{digits}e}"


def _write_table(summary_df: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    pivot = summary_df.pivot(index="dataset", columns="case")
    rows = []
    for dataset in sorted(summary_df["dataset"].unique()):
        rows.append(
            {
                "dataset": dataset,
                "baseline energy diff (eV)": _format_pm(
                    float(pivot.loc[dataset, ("abs_energy_diff_mean", "e3nn_baseline")]),
                    float(pivot.loc[dataset, ("abs_energy_diff_std", "e3nn_baseline")]),
                ),
                "proposal energy diff (eV)": _format_pm(
                    float(pivot.loc[dataset, ("abs_energy_diff_mean", "e3nn_pair_full")]),
                    float(pivot.loc[dataset, ("abs_energy_diff_std", "e3nn_pair_full")]),
                ),
                "baseline force diff (eV/A)": _format_pm(
                    float(pivot.loc[dataset, ("max_force_diff_mean", "e3nn_baseline")]),
                    float(pivot.loc[dataset, ("max_force_diff_std", "e3nn_baseline")]),
                ),
                "proposal force diff (eV/A)": _format_pm(
                    float(pivot.loc[dataset, ("max_force_diff_mean", "e3nn_pair_full")]),
                    float(pivot.loc[dataset, ("max_force_diff_std", "e3nn_pair_full")]),
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "table_04_pair_accuracy_summary.csv", index=False)
    (table_dir / "table_04_pair_accuracy_summary.md").write_text(_df_to_md(out))
    return out


def _write_compact_table(summary_df: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    rows = []
    for case in ("e3nn_baseline", "e3nn_pair_full"):
        frame = summary_df[summary_df["case"] == case]
        rows.append(
            {
                "case": case,
                "median energy diff mean (eV)": f"{frame['abs_energy_diff_mean'].median():.3e}",
                "worst energy diff max (eV)": f"{frame['abs_energy_diff_max'].max():.3e}",
                "median force diff mean (eV/A)": f"{frame['max_force_diff_mean'].median():.3e}",
                "worst force diff max (eV/A)": f"{frame['max_force_diff_max'].max():.3e}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "table_04a_pair_accuracy_compact.csv", index=False)
    (table_dir / "table_04a_pair_accuracy_compact.md").write_text(_df_to_md(out))
    return out


def _plot_metric(summary_df: pd.DataFrame, metric_mean: str, metric_std: str, ylabel: str, eps: float, out_stem: Path) -> None:
    ordered = summary_df[summary_df["case"] == "e3nn_pair_full"].sort_values(metric_mean, ascending=False)["dataset"].tolist()
    x = np.arange(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(11, len(ordered) * 0.38), 5.3))
    for offset, case in [(-width / 2, "e3nn_baseline"), (width / 2, "e3nn_pair_full")]:
        frame = summary_df[summary_df["case"] == case].set_index("dataset").loc[ordered]
        means = np.maximum(frame[metric_mean].to_numpy(dtype=np.float64), eps)
        stds = frame[metric_std].to_numpy(dtype=np.float64)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            color=COLOR_CASES[case],
            alpha=0.92,
            label=case,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs baseline reference (`mean ± std`, repeated runs)")
    ax.legend()
    _savefig(fig, out_stem)


def _write_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    base = summary_df[summary_df["case"] == "e3nn_baseline"]
    prop = summary_df[summary_df["case"] == "e3nn_pair_full"]
    lines = [
        "# Pair Accuracy Repeat Report",
        "",
        "이 보고서는 같은 대표 샘플을 여러 번 다시 계산해,",
        "기본 실행 방식의 반복 잡음과 제안기법의 기준 출력 차이를 함께 정리한 결과이다.",
        "",
        f"- 데이터셋 수: {summary_df['dataset'].nunique()}",
        f"- baseline median mean force diff: {base['max_force_diff_mean'].median():.3e} eV/A",
        f"- proposal median mean force diff: {prop['max_force_diff_mean'].median():.3e} eV/A",
        f"- baseline worst force diff max: {base['max_force_diff_max'].max():.3e} eV/A",
        f"- proposal worst force diff max: {prop['max_force_diff_max'].max():.3e} eV/A",
        f"- baseline median mean energy diff: {base['abs_energy_diff_mean'].median():.3e} eV",
        f"- proposal median mean energy diff: {prop['abs_energy_diff_mean'].median():.3e} eV",
        f"- baseline worst energy diff max: {base['abs_energy_diff_max'].max():.3e} eV",
        f"- proposal worst energy diff max: {prop['abs_energy_diff_max'].max():.3e} eV",
        "",
        "핵심 해석:",
        "",
        "- baseline 반복 실행 자체에서도 부동소수점 수준의 아주 작은 차이가 관측된다.",
        "- proposal의 에너지/힘 차이는 이 기준 출력에 대해 매우 작은 절대값을 유지한다.",
        "- 따라서 현재 제안기법은 출력 정확도를 바꾸기보다 실행 시간을 바꾸는 최적화로 해석하는 것이 맞다.",
        "",
        "주의:",
        "",
        "- 이 표와 그림은 baseline 기준 출력 대비 절대 차이이다.",
        "- 표준편차는 같은 샘플을 여러 번 반복 실행해 얻은 값이다.",
        "- 0값도 존재하므로 그림은 로그축 표시를 위해 작은 floor 값을 사용한다.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    output_root = KCC_ROOT
    summary_path = output_root / "metrics" / "pair_accuracy" / "global" / "pair_accuracy_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    figure_dir = ensure_dir(output_root / "figures" / "pair_accuracy")
    table_dir = ensure_dir(output_root / "tables")
    report_dir = ensure_dir(output_root / "reports")

    _write_table(summary_df, table_dir)
    _write_compact_table(summary_df, table_dir)
    _plot_metric(
        summary_df,
        "abs_energy_diff_mean",
        "abs_energy_diff_std",
        "Absolute energy difference (eV)",
        EPS_ENERGY,
        figure_dir / "pair_accuracy_energy_errorbar",
    )
    _plot_metric(
        summary_df,
        "max_force_diff_mean",
        "max_force_diff_std",
        "Maximum absolute force difference (eV/A)",
        EPS_FORCE,
        figure_dir / "pair_accuracy_force_errorbar",
    )
    _write_report(summary_df, report_dir / "pair_accuracy_repeat_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
