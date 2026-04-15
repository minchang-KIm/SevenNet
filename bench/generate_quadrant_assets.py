from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_real_points() -> pd.DataFrame:
    feat = pd.read_csv(REPO_ROOT / 'bench_runs/real_e3nn_pair_k3/derived/sample_features.csv')
    agg = pd.read_csv(REPO_ROOT / 'bench_runs/real_e3nn_pair_k3/metrics/aggregated.csv')
    rep = (
        feat.sort_values(['dataset', 'natoms', 'num_edges'], ascending=[True, False, False])
        .groupby('dataset')
        .head(1)
        .copy()
    )
    pivot = agg.pivot(index='dataset', columns='case', values='steady_median_ms').reset_index()
    pivot['steady_speedup_baseline_over_pair'] = (
        pivot['e3nn_baseline'] / pivot['e3nn_pair_full']
    )
    pivot = pivot.rename(
        columns={
            'e3nn_baseline': 'baseline_steady_median_ms',
            'e3nn_pair_full': 'pair_steady_median_ms',
        }
    )
    out = rep[
        [
            'dataset',
            'sample_id',
            'natoms',
            'num_edges',
            'avg_neighbors_directed',
            'structure_class',
        ]
    ].merge(
        pivot[
            [
                'dataset',
                'baseline_steady_median_ms',
                'pair_steady_median_ms',
                'steady_speedup_baseline_over_pair',
            ]
        ],
        on='dataset',
        how='inner',
    )
    out['source'] = 'real_e3nn_pair_k3'
    return out


def _load_local_points() -> pd.DataFrame:
    samples = pd.read_csv(REPO_ROOT / 'bench_runs/local_pair_size_main/metrics/samples.csv')
    rep = (
        samples.sort_values(['dataset', 'natoms', 'num_edges'], ascending=[True, False, False])
        .groupby('dataset')
        .head(1)
        .copy()
    )
    out = rep[
        [
            'dataset',
            'sample_id',
            'natoms',
            'num_edges',
            'avg_neighbors_directed',
            'baseline_steady_median_ms',
            'pair_steady_median_ms',
            'steady_speedup_baseline_over_pair',
        ]
    ].copy()
    out['structure_class'] = 'molecule_or_bulk_local'
    out['source'] = 'local_pair_size_main'
    return out


def _prepare_dataset_status() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / 'datasets/inventory.csv')
    keep = df[['name', 'approx_size_gib', 'status', 'gated', 'local_path']].copy()
    keep = keep.sort_values('approx_size_gib', ascending=False)
    return keep


def _choose_quadrant_representatives(real_points: pd.DataFrame, local_points: pd.DataFrame) -> pd.DataFrame:
    selected = [
        {
            'quadrant': 'Small Sparse',
            'dataset': 'spice_2023',
            'friendly_name': 'SPICE 2023',
            'source': 'local_pair_size_main',
        },
        {
            'quadrant': 'Small Dense',
            'dataset': 'phonondb_pbe',
            'friendly_name': 'phononDB PBE',
            'source': 'real_e3nn_pair_k3',
        },
        {
            'quadrant': 'Large Sparse',
            'dataset': 'omol25_validation',
            'friendly_name': 'OMol25 validation',
            'source': 'real_e3nn_pair_k3',
        },
        {
            'quadrant': 'Large Dense',
            'dataset': 'mptrj_val',
            'friendly_name': 'MPtrj validation',
            'source': 'real_e3nn_pair_k3',
        },
    ]
    src_map = {
        'real_e3nn_pair_k3': real_points,
        'local_pair_size_main': local_points,
    }
    rows = []
    for item in selected:
        src = src_map[item['source']]
        row = src[src['dataset'] == item['dataset']].iloc[0].to_dict()
        row.update(item)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_all_points(real_points: pd.DataFrame, local_points: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([real_points, local_points], ignore_index=True, sort=False)
    return combined


def _plot_scatter(points: pd.DataFrame, reps: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sc = ax.scatter(
        points['num_edges'],
        points['avg_neighbors_directed'],
        c=points['steady_speedup_baseline_over_pair'],
        cmap='RdYlGn',
        s=90,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.85,
    )
    for _, row in points.iterrows():
        ax.annotate(
            row['dataset'],
            (row['num_edges'], row['avg_neighbors_directed']),
            fontsize=7,
            xytext=(4, 4),
            textcoords='offset points',
            alpha=0.85,
        )
    ax.axvline(3000, color='black', linestyle='--', linewidth=1)
    ax.axhline(40, color='black', linestyle='--', linewidth=1)
    for _, row in reps.iterrows():
        ax.scatter(
            [row['num_edges']],
            [row['avg_neighbors_directed']],
            marker='*',
            s=320,
            c='none',
            edgecolors='black',
            linewidths=1.8,
        )
    ax.set_xscale('log')
    ax.set_xlabel('Directed edges per representative sample (log scale)')
    ax.set_ylabel('Average directed neighbors')
    ax.set_title('Dataset Map: Size, Density, and Pair Speedup')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speedup (baseline / pair_full)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_latency_bars(reps: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = reps['quadrant'].tolist()
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, reps['baseline_steady_median_ms'], width, label='Baseline', color='#4c78a8')
    ax.bar(x + width / 2, reps['pair_steady_median_ms'], width, label='Pair Full', color='#f58518')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Steady median latency (ms)')
    ax.set_title('Representative Latency by Quadrant')
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_speedup_bars(reps: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    colors = ['#2a9d8f' if v >= 1.0 else '#e76f51' for v in reps['steady_speedup_baseline_over_pair']]
    ax.bar(reps['quadrant'], reps['steady_speedup_baseline_over_pair'], color=colors)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    for idx, value in enumerate(reps['steady_speedup_baseline_over_pair']):
        ax.text(idx, value + 0.02, f'{value:.3f}x', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Speedup (baseline / pair_full)')
    ax.set_title('Pair Speedup by Representative Quadrant')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_download_status(status_df: pd.DataFrame, out_path: Path) -> None:
    top = status_df.head(10).copy()
    color_map = {'exists': '#2a9d8f', 'pending': '#e76f51', 'gated': '#9d4edd'}
    colors = [color_map.get(status, '#adb5bd') for status in top['status']]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(top['name'], top['approx_size_gib'], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('Approx size (GiB)')
    ax.set_title('Largest Public MLIP Datasets: Local Download Status')
    for idx, row in top.reset_index(drop=True).iterrows():
        ax.text(
            row['approx_size_gib'] + 0.2,
            idx,
            row['status'],
            va='center',
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_extreme_stage_breakdown(out_path: Path) -> None:
    add = pd.read_csv(
        REPO_ROOT / 'bench_runs/detailed_model_profile_main/metrics/stage_breakdown_additive_long.csv'
    )
    selected = [
        ('spice_2023', 'e3nn_baseline', 'Small Sparse Baseline'),
        ('spice_2023', 'e3nn_pair_full', 'Small Sparse Pair'),
        ('mptrj', 'e3nn_baseline', 'Large Dense Baseline'),
        ('mptrj', 'e3nn_pair_full', 'Large Dense Pair'),
    ]
    rows = []
    for dataset, case, label in selected:
        sub = add[(add['dataset'] == dataset) & (add['case'] == case)]
        rows.append(
            {
                'label': label,
                'spherical_harmonics': sub[sub['stage'].str.contains('spherical_harmonics_ms')]['time_ms'].sum(),
                'weight_nn': sub[sub['stage'].str.contains('weight_nn_ms')]['time_ms'].sum(),
                'message_tp': sub[sub['stage'].str.contains('message_tp_ms')]['time_ms'].sum(),
                'aggregation': sub[sub['stage'].str.contains('aggregation_ms')]['time_ms'].sum(),
                'force_output': sub[sub['stage'] == 'top_force_output_ms']['time_ms'].sum(),
                'other': sub[
                    ~sub['stage'].str.contains('spherical_harmonics_ms|weight_nn_ms|message_tp_ms|aggregation_ms')
                    & (sub['stage'] != 'top_force_output_ms')
                ]['time_ms'].sum(),
            }
        )
    frame = pd.DataFrame(rows)
    stage_cols = ['spherical_harmonics', 'weight_nn', 'message_tp', 'aggregation', 'force_output', 'other']
    colors = ['#2a9d8f', '#8ecae6', '#457b9d', '#e9c46a', '#e76f51', '#adb5bd']
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bottom = np.zeros(len(frame))
    for col, color in zip(stage_cols, colors):
        vals = frame[col].to_numpy(dtype=float)
        ax.bar(frame['label'], vals, bottom=bottom, label=col, color=color)
        bottom += vals
    ax.set_ylabel('Intrusive profiled stage time (ms)')
    ax.set_title('Stage Breakdown for Extreme Quadrants')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mechanism_diagram(reps: pd.DataFrame, out_path_png: Path, out_path_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    quadrants = [
        (0.7, 5.3, 3.7, 3.6, '#fde2e4', 'Small Sparse', 'Few edges\nLow GPU fill\nSaved pair work too small\nOften slower'),
        (5.6, 5.3, 3.7, 3.6, '#d8f3dc', 'Large Sparse', 'More edges\nModerate reuse benefit\nCan win if edge load is large enough'),
        (0.7, 0.9, 3.7, 3.6, '#faedcd', 'Small Dense', 'High neighbors but small graph\nCan still benefit when edge count is high\nphononDB-like case'),
        (5.6, 0.9, 3.7, 3.6, '#dceaf7', 'Large Dense', 'Largest edge load\nBest chance to amortize overhead\nStrongest speedup'),
    ]
    for x, y, w, h, color, title, text in quadrants:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(x + 0.2, y + h - 0.4, title, fontsize=14, weight='bold', va='top')
        ax.text(x + 0.2, y + h - 1.0, text, fontsize=11, va='top')

    ax.text(5.0, 9.55, 'Larger Graph Size', ha='center', fontsize=14, weight='bold')
    ax.text(0.18, 5.0, 'Denser Graph', rotation=90, va='center', fontsize=14, weight='bold')

    ax.add_patch(FancyArrowPatch((1.2, 9.0), (9.1, 9.0), arrowstyle='-|>', mutation_scale=16, linewidth=1.5))
    ax.add_patch(FancyArrowPatch((0.55, 1.2), (0.55, 8.8), arrowstyle='-|>', mutation_scale=16, linewidth=1.5))

    ax.text(5.0, 4.95, 'Observed representatives:\nSmall sparse = SPICE 2023, Small dense = phononDB,\nLarge sparse = OMol25 validation, Large dense = MPtrj validation', ha='center', va='center', fontsize=11)

    plt.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_png, dpi=220)
    plt.savefig(out_path_svg)
    plt.close(fig)


def _write_report(reps: pd.DataFrame, status_df: pd.DataFrame, out_path: Path) -> None:
    pending_large = status_df[status_df['status'] != 'exists'].head(10)
    lines = [
        '# Seminar Asset Pack',
        '',
        '## Representative Quadrants',
        '',
    ]
    for _, row in reps.iterrows():
        lines.append(
            f"- `{row['quadrant']}`: `{row['friendly_name']}` "
            f"(atoms=`{int(row['natoms'])}`, edges=`{int(row['num_edges'])}`, "
            f"avg_neighbors=`{row['avg_neighbors_directed']:.2f}`, "
            f"speedup=`{row['steady_speedup_baseline_over_pair']:.3f}x`, source=`{row['source']}`)"
        )
    lines.extend(
        [
            '',
            '## Largest Datasets Still Missing In `datasets/raw`',
            '',
        ]
    )
    for _, row in pending_large.iterrows():
        lines.append(
            f"- `{row['name']}`: approx `{row['approx_size_gib']:.2f} GiB`, status=`{row['status']}`"
        )
    lines.extend(
        [
            '',
            '## Created Files',
            '',
            '- `quadrant_representatives.csv`',
            '- `all_dataset_points.csv`',
            '- `large_dataset_status.csv`',
            '- `plots/quadrant_dataset_map.png`',
            '- `plots/quadrant_latency_comparison.png`',
            '- `plots/quadrant_speedup_comparison.png`',
            '- `plots/large_dataset_download_status.png`',
            '- `plots/extreme_stage_breakdown.png`',
            '- `diagrams/quadrant_mechanism_diagram.png`',
            '- `diagrams/quadrant_mechanism_diagram.svg`',
        ]
    )
    out_path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'docs' / 'presentations' / 'assets' / 'quadrant_pack',
    )
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    plots_dir = out_dir / 'plots'
    diagrams_dir = out_dir / 'diagrams'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    real_points = _load_real_points()
    local_points = _load_local_points()
    all_points = _build_all_points(real_points, local_points)
    reps = _choose_quadrant_representatives(real_points, local_points)
    status_df = _prepare_dataset_status()

    reps.to_csv(out_dir / 'quadrant_representatives.csv', index=False)
    all_points.to_csv(out_dir / 'all_dataset_points.csv', index=False)
    status_df.to_csv(out_dir / 'large_dataset_status.csv', index=False)

    _plot_scatter(all_points, reps, plots_dir / 'quadrant_dataset_map.png')
    _plot_latency_bars(reps, plots_dir / 'quadrant_latency_comparison.png')
    _plot_speedup_bars(reps, plots_dir / 'quadrant_speedup_comparison.png')
    _plot_download_status(status_df, plots_dir / 'large_dataset_download_status.png')
    _plot_extreme_stage_breakdown(plots_dir / 'extreme_stage_breakdown.png')
    _plot_mechanism_diagram(
        reps,
        diagrams_dir / 'quadrant_mechanism_diagram.png',
        diagrams_dir / 'quadrant_mechanism_diagram.svg',
    )
    _write_report(reps, status_df, out_dir / 'README.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
