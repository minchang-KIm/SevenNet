from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in columns})


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return '(no data)\n'
    if not columns:
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
    header = '| ' + ' | '.join(columns) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = []
    for row in rows:
        body.append(
            '| '
            + ' | '.join(
                '' if row.get(col) is None else str(row.get(col)) for col in columns
            )
            + ' |'
        )
    return '\n'.join([header, sep, *body]) + '\n'


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _plot_bar(
    rows: Sequence[Dict[str, Any]],
    x: str,
    y: str,
    title: str,
    out: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not rows:
        return False
    xs = [row.get(x, '') for row in rows]
    ys = [_safe_float(row.get(y)) for row in rows]
    ys = [yy if yy is not None else 0.0 for yy in ys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def _analysis_lines(perf_rows: Sequence[Dict[str, Any]]) -> List[str]:
    numeric_rows = [
        row
        for row in perf_rows
        if _safe_float(row.get('median_ms')) is not None
    ]
    if not numeric_rows:
        return ['- No performance benchmark data was produced.']

    fastest = min(numeric_rows, key=lambda row: _safe_float(row.get('median_ms')) or 0.0)
    lines = [
        f"- Fastest measured case: `{fastest.get('case', 'unknown')}` "
        f"at `{_safe_float(fastest.get('median_ms')):.3f}` ms median latency."
    ]
    force_vals = [
        _safe_float(row.get('max_abs_force_diff'))
        for row in numeric_rows
        if _safe_float(row.get('max_abs_force_diff')) is not None
    ]
    if force_vals:
        lines.append(
            f"- Worst recorded force delta versus baseline: `{max(force_vals):.3e}`."
        )
    return lines


def generate_report(output_dir: Path) -> Path:
    metrics_dir = output_dir / 'metrics'
    tables_dir = output_dir / 'tables'
    plots_dir = output_dir / 'plots'
    env_dir = output_dir / 'environment'
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    system_info = {}
    system_info_path = env_dir / 'system.json'
    if system_info_path.exists():
        system_info = _read_json(system_info_path)

    steps: List[Dict[str, Any]] = []
    step_path = metrics_dir / 'steps.json'
    if step_path.exists():
        steps = _read_json(step_path)
    _write_csv(tables_dir / 'step_status.csv', steps)

    perf_entries: List[Dict[str, Any]] = []
    perf_path = metrics_dir / 'perf.json'
    if perf_path.exists():
        perf_entries = _read_json(perf_path)
    _write_csv(tables_dir / 'perf.csv', perf_entries)

    baseline = next(
        (
            row
            for row in perf_entries
            if row.get('case') == 'baseline_e3nn_forward'
            and _safe_float(row.get('median_ms')) is not None
        ),
        None,
    )
    speedup_entries: List[Dict[str, Any]] = []
    if baseline and _safe_float(baseline.get('median_ms')) is not None:
        base_ms = float(baseline['median_ms'])
        for row in perf_entries:
            row = dict(row)
            ms = _safe_float(row.get('median_ms'))
            if ms and ms > 0:
                row['speedup_vs_baseline'] = base_ms / ms
            speedup_entries.append(row)
    else:
        speedup_entries = [dict(row) for row in perf_entries]
    _write_csv(tables_dir / 'speedup.csv', speedup_entries)

    plots: List[str] = []
    if _plot_bar(
        perf_entries,
        'case',
        'median_ms',
        'Median latency by benchmark case',
        plots_dir / 'latency_by_case.png',
    ):
        plots.append('plots/latency_by_case.png')
    speedup_rows = [row for row in speedup_entries if row.get('speedup_vs_baseline')]
    if _plot_bar(
        speedup_rows,
        'case',
        'speedup_vs_baseline',
        'Speedup versus baseline e3nn forward',
        plots_dir / 'speedup_vs_baseline.png',
    ):
        plots.append('plots/speedup_vs_baseline.png')

    summary_lines = ['# SevenNet Pair-Execution Test Summary', '']
    if system_info:
        summary_lines.extend(
            [
                '## Environment',
                '',
                f"- Host: `{system_info.get('hostname', 'unknown')}`",
                f"- Platform: `{system_info.get('platform', 'unknown')}`",
                f"- Git branch: `{system_info.get('git', {}).get('branch', 'unknown')}`",
                f"- Git SHA: `{system_info.get('git', {}).get('sha', 'unknown')}`",
                f"- Torch: `{system_info.get('torch', {}).get('torch_version', 'unknown')}`",
                f"- CUDA available: `{system_info.get('torch', {}).get('cuda_available', False)}`",
                '',
            ]
        )

    summary_lines.extend(['## Step Status', '', _markdown_table(steps, [])])

    if perf_entries:
        perf_columns = [
            'case',
            'device',
            'backend',
            'pair_policy',
            'median_ms',
            'p95_ms',
            'max_abs_energy_diff',
            'max_abs_force_diff',
            'max_abs_stress_diff',
            'status',
            'reason',
            'speedup_vs_baseline',
        ]
        summary_lines.extend(
            [
                '## Performance',
                '',
                _markdown_table(speedup_entries, perf_columns),
                '## Analysis',
                '',
                *_analysis_lines(perf_entries),
            ]
        )
        if plots:
            summary_lines.extend(['', '## Plots', ''])
            summary_lines.extend([f'- `{plot}`' for plot in plots])
    else:
        summary_lines.extend(
            ['## Performance', '', '- No performance benchmark data was produced.']
        )

    summary_path = output_dir / 'summary.md'
    summary_path.write_text('\n'.join(summary_lines) + '\n')
    return summary_path
