from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kcc_common import COLOR_CASES, KCC_ROOT, ensure_dir


FOUR_CASE_STAGE_ORDER = [
    "spherical_harmonics_ms",
    "radial_basis_ms",
    "cutoff_ms",
    "conv_weight_nn_ms",
    "pair_expand_ms",
    "pair_indexing_ms",
    "conv_src_gather_ms",
    "conv_filter_gather_ms",
    "conv_message_tp_ms",
    "flash_fused_conv_ms",
    "conv_aggregation_ms",
    "top_force_output_ms",
    "other_ms",
]

FOUR_CASE_STAGE_COLORS = {
    "spherical_harmonics_ms": "#E69F00",
    "radial_basis_ms": "#56B4E9",
    "cutoff_ms": "#009E73",
    "conv_weight_nn_ms": "#4C9F70",
    "pair_expand_ms": "#8B6BB8",
    "pair_indexing_ms": "#A84832",
    "conv_src_gather_ms": "#A6CEE3",
    "conv_filter_gather_ms": "#1F78B4",
    "conv_message_tp_ms": "#4D4D4D",
    "flash_fused_conv_ms": "#CC79A7",
    "conv_aggregation_ms": "#7F7F7F",
    "top_force_output_ms": "#3B4C63",
    "other_ms": "#BDBDBD",
}

CASE_LABELS = {
    "e3nn_baseline": "No Option",
    "flash_baseline": "FlashTP Only",
    "e3nn_pair_full": "Proposal Only",
    "flash_pair_auto": "FlashTP + Proposal",
}

CASE_FILENAMES = {
    "e3nn_baseline": "option_none",
    "flash_baseline": "flashtp_only",
    "e3nn_pair_full": "proposal_only",
    "flash_pair_auto": "flashtp_plus_proposal",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _savefig(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_case_all_datasets(stage_df: pd.DataFrame, case: str, out_stem: Path) -> None:
    frame = stage_df[stage_df["case"] == case].copy()
    frame = frame.sort_values(["natoms", "dataset"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(max(12, len(frame) * 0.38), 5.6))
    bottom = np.zeros(len(frame), dtype=np.float64)
    x = np.arange(len(frame))
    for stage in FOUR_CASE_STAGE_ORDER:
        subset = frame[frame["stage"] == stage]
        values = subset["mean_ms"].to_numpy(dtype=np.float64) if not subset.empty else np.zeros(len(frame))
        if len(values) != len(frame):
            map_df = subset.set_index("dataset")["mean_ms"]
            values = np.array([float(map_df.get(dataset, 0.0)) for dataset in frame["dataset"]], dtype=np.float64)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=FOUR_CASE_STAGE_COLORS.get(stage, "#999999"),
            label=stage,
            width=0.85,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(frame["dataset"], rotation=45, ha="right")
    ax.set_ylabel("Intrusive stage time (ms)")
    ax.set_title(f"{CASE_LABELS[case]} detailed stage breakdown")
    ax.legend(ncol=3, fontsize=8)
    _savefig(fig, out_stem)


def _plot_dataset_four_cases(stage_df: pd.DataFrame, dataset: str, out_stem: Path) -> None:
    frame = stage_df[stage_df["dataset"] == dataset].copy()
    case_order = ["e3nn_baseline", "flash_baseline", "e3nn_pair_full", "flash_pair_auto"]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    bottom = np.zeros(len(case_order), dtype=np.float64)
    x = np.arange(len(case_order))
    for stage in FOUR_CASE_STAGE_ORDER:
        subset = frame[frame["stage"] == stage].set_index("case")["mean_ms"] if not frame.empty else pd.Series(dtype=float)
        values = np.array([float(subset.get(case, 0.0)) for case in case_order], dtype=np.float64)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=FOUR_CASE_STAGE_COLORS.get(stage, "#999999"),
            label=stage,
            width=0.75,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[case] for case in case_order], rotation=15, ha="right")
    ax.set_ylabel("Intrusive stage time (ms)")
    ax.set_title(f"{dataset}: four-case stage breakdown")
    ax.legend(ncol=3, fontsize=8)
    _savefig(fig, out_stem)


def _plot_model_total_heatmap(summary_df: pd.DataFrame, out_stem: Path) -> None:
    pivot = summary_df.pivot(index="dataset", columns="case", values="model_total_ms_mean")
    pivot = pivot[[case for case in ["e3nn_baseline", "flash_baseline", "e3nn_pair_full", "flash_pair_auto"] if case in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7.8, max(7.0, len(pivot) * 0.28)))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([CASE_LABELS[case] for case in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Intrusive model_total mean by dataset and case")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ms")
    _savefig(fig, out_stem)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=KCC_ROOT)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    global_dir = output_root / "metrics" / "four_case" / "global"
    per_dataset_dir = output_root / "metrics" / "four_case" / "per_dataset"
    figures_root = ensure_dir(output_root / "figures" / "four_case")

    stage_df = _read_csv(global_dir / "four_case_detailed_stage_long_mean_std.csv")
    summary_df = _read_csv(global_dir / "four_case_detailed_stage_mean_std.csv")
    if stage_df.empty or summary_df.empty:
        raise SystemExit("four-case detailed metrics not found; run kcc_four_case_profile.py first")

    for case in ["e3nn_baseline", "flash_baseline", "e3nn_pair_full", "flash_pair_auto"]:
        _plot_case_all_datasets(
            stage_df,
            case,
            figures_root / f"stage_breakdown_all_{CASE_FILENAMES[case]}",
        )

    dataset_fig_dir = ensure_dir(figures_root / "per_dataset")
    for dataset_dir in sorted(per_dataset_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        _plot_dataset_four_cases(
            stage_df,
            dataset,
            dataset_fig_dir / f"{dataset}_four_case_stage_breakdown",
        )

    _plot_model_total_heatmap(summary_df, figures_root / "model_total_heatmap")

    summary_lines = [
        "# KCC Four-Case Stage Figures",
        "",
        "## Global case plots",
        "",
        "- `figures/four_case/stage_breakdown_all_option_none.png`",
        "- `figures/four_case/stage_breakdown_all_flashtp_only.png`",
        "- `figures/four_case/stage_breakdown_all_proposal_only.png`",
        "- `figures/four_case/stage_breakdown_all_flashtp_plus_proposal.png`",
        "- `figures/four_case/model_total_heatmap.png`",
        "",
        "## Per-dataset plots",
        "",
        "- `figures/four_case/per_dataset/<dataset>_four_case_stage_breakdown.png`",
    ]
    (output_root / "four_case_figures.md").write_text("\n".join(summary_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
