from __future__ import annotations

import argparse
import gc
import heapq
import json
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from ase import Atoms
from huggingface_hub import hf_hub_download

from sevenn.calculator import SevenNetCalculator


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: str
    modal: str
    description: str
    source: str
    repo_id: str | None = None
    filenames: Sequence[str] = ()
    figshare_url: str | None = None
    size_column: str = ''


DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec(
        name='mptrj_val',
        loader='nimashoghi',
        modal='mpa',
        description='MPtrj validation split',
        source='https://huggingface.co/datasets/nimashoghi/mptrj',
        repo_id='nimashoghi/mptrj',
        filenames=('data/val-00000-of-00001.parquet',),
        size_column='num_atoms',
    ),
    DatasetSpec(
        name='salex_validation',
        loader='colabfit',
        modal='mpa',
        description='sAlex validation',
        source='https://huggingface.co/datasets/colabfit/sAlex_validation',
        repo_id='colabfit/sAlex_validation',
        filenames=('co/co_0.parquet', 'co/co_1.parquet'),
        size_column='nsites',
    ),
    DatasetSpec(
        name='omat24_validation',
        loader='colabfit',
        modal='omat24',
        description='OMat24 rattled-1000-subsampled validation',
        source='https://huggingface.co/datasets/colabfit/OMat24_validation_rattled_1000_subsampled',
        repo_id='colabfit/OMat24_validation_rattled_1000_subsampled',
        filenames=('co/co_0.parquet',),
        size_column='nsites',
    ),
    DatasetSpec(
        name='omol25_validation',
        loader='colabfit',
        modal='omol25_low',
        description='OMol25 neutral validation',
        source='https://huggingface.co/datasets/colabfit/OMol25_neutral_validation',
        repo_id='colabfit/OMol25_neutral_validation',
        filenames=('co/co_0.parquet',),
        size_column='nsites',
    ),
    DatasetSpec(
        name='oc20_val_id',
        loader='colabfit',
        modal='oc20',
        description='OC20 S2EF in-domain validation',
        source='https://huggingface.co/datasets/colabfit/OC20_S2EF_val_id',
        repo_id='colabfit/OC20_S2EF_val_id',
        filenames=('co/co_0.parquet', 'co/co_1.parquet'),
        size_column='nsites',
    ),
    DatasetSpec(
        name='oc22_val_id',
        loader='colabfit',
        modal='oc22',
        description='OC22 S2EF in-domain validation',
        source='https://huggingface.co/datasets/colabfit/OC22-S2EF-Validation-in-domain',
        repo_id='colabfit/OC22-S2EF-Validation-in-domain',
        filenames=('co/co_0.parquet',),
        size_column='nsites',
    ),
    DatasetSpec(
        name='wbm_initial',
        loader='extxyz',
        modal='matpes_pbe',
        description='WBM initial benchmark structures',
        source='https://figshare.com/articles/dataset/Matbench_Discovery_-_Data_Files/22715158',
        figshare_url='https://ndownloader.figshare.com/files/48169597',
    ),
    DatasetSpec(
        name='phonondb_pbe',
        loader='extxyz',
        modal='matpes_r2scan',
        description='phononDB PBE benchmark structures',
        source='https://figshare.com/articles/dataset/Matbench_Discovery_-_Data_Files/22715158',
        figshare_url='https://ndownloader.figshare.com/files/52179965',
    ),
)


CASE_PRESETS: Dict[str, Sequence[Dict[str, Any]]] = {
    'flash_real': (
        {
            'case': 'e3nn_baseline',
            'enable_flash': False,
            'enable_pair_execution': False,
            'pair_execution_policy': None,
        },
        {
            'case': 'flash_baseline',
            'enable_flash': True,
            'enable_pair_execution': False,
            'pair_execution_policy': None,
        },
        {
            'case': 'flash_pair_auto',
            'enable_flash': True,
            'enable_pair_execution': True,
            'pair_execution_policy': None,
        },
    ),
    'e3nn_pair': (
        {
            'case': 'e3nn_baseline',
            'enable_flash': False,
            'enable_pair_execution': False,
            'pair_execution_policy': None,
        },
        {
            'case': 'e3nn_pair_full',
            'enable_flash': False,
            'enable_pair_execution': True,
            'pair_execution_policy': 'full',
        },
    ),
}


def _as_numeric_array(value: Any, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != object:
        return np.asarray(arr, dtype=dtype)
    return np.asarray([np.asarray(v, dtype=dtype) for v in value], dtype=dtype)


def _as_bool_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray([bool(v) for v in value], dtype=bool)
    return np.asarray(arr, dtype=bool)


def atoms_from_colabfit_row(row: pd.Series) -> Atoms:
    return Atoms(
        numbers=np.asarray(row['atomic_numbers'], dtype=np.int64),
        positions=_as_numeric_array(row['positions']),
        cell=_as_numeric_array(row['cell']),
        pbc=_as_bool_array(row['pbc']),
    )


def atoms_from_nimashoghi_row(row: pd.Series) -> Atoms:
    return Atoms(
        numbers=np.asarray(row['numbers'], dtype=np.int64),
        positions=_as_numeric_array(row['positions']),
        cell=_as_numeric_array(row['cell']),
        pbc=_as_bool_array(row['pbc']),
    )


ROW_ADAPTERS: Dict[str, Callable[[pd.Series], Atoms]] = {
    'colabfit': atoms_from_colabfit_row,
    'nimashoghi': atoms_from_nimashoghi_row,
}


FULL_COLUMNS: Dict[str, Sequence[str]] = {
    'colabfit': (
        'configuration_id',
        'names',
        'nsites',
        'atomic_numbers',
        'positions',
        'cell',
        'pbc',
    ),
    'nimashoghi': (
        'mp_id',
        'filename',
        'num_atoms',
        'numbers',
        'positions',
        'cell',
        'pbc',
    ),
}


def _download_figshare(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dst.open('wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


def _prepare_extxyz_from_download(url: str, dst_prefix: Path) -> Path:
    download_path = _download_figshare(url, dst_prefix.with_suffix('.download'))
    if not zipfile.is_zipfile(download_path):
        return download_path

    extracted_path = dst_prefix.with_suffix('.extxyz')
    if extracted_path.exists():
        return extracted_path
    with zipfile.ZipFile(download_path) as archive:
        members = [member for member in archive.namelist() if member.endswith('.extxyz')]
        if not members:
            raise ValueError(f'No extxyz members found in {download_path}')
        with extracted_path.open('wb') as dst:
            for member in members:
                with archive.open(member) as src:
                    dst.write(src.read())
    return extracted_path


def _read_extxyz_topk(path: Path, top_k: int) -> List[Dict[str, Any]]:
    import ase.io

    heap: List[tuple[int, int, Atoms]] = []
    for idx, atoms in enumerate(ase.io.iread(path, index=':', format='extxyz')):
        size = len(atoms)
        item = (size, idx, atoms)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif size > heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    return [
        {
            'sample_id': f'{path.stem}:{idx}',
            'natoms': natoms,
            'atoms': atoms,
        }
        for natoms, idx, atoms in selected
    ]


def _top_candidates_in_parquet(
    path: Path, size_column: str, top_k: int
) -> List[tuple[int, int, int, Path]]:
    parquet = pq.ParquetFile(path)
    heap: List[tuple[int, int, int, Path]] = []
    for row_group_idx in range(parquet.num_row_groups):
        column = parquet.read_row_group(row_group_idx, columns=[size_column]).column(0)
        for row_idx, value in enumerate(column.to_pylist()):
            size = int(value)
            item = (size, row_group_idx, row_idx, path)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif size > heap[0][0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda item: item[0], reverse=True)


def _rows_from_parquet(
    path: Path,
    candidates: Sequence[tuple[int, int, int, Path]],
    columns: Sequence[str],
) -> List[pd.Series]:
    parquet = pq.ParquetFile(path)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for _, row_group_idx, row_idx, _ in candidates:
        grouped[row_group_idx].append(row_idx)

    rows: List[pd.Series] = []
    for row_group_idx, row_indices in grouped.items():
        table = parquet.read_row_group(row_group_idx, columns=list(columns))
        frame = table.to_pandas()
        for row_idx in row_indices:
            rows.append(frame.iloc[row_idx])
    return rows


def _download_hf_dataset_files(spec: DatasetSpec) -> List[Path]:
    assert spec.repo_id is not None
    return [
        Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                repo_type='dataset',
            )
        )
        for filename in spec.filenames
    ]


def load_dataset_samples(
    spec: DatasetSpec,
    *,
    cache_dir: Path,
    top_k: int,
) -> List[Dict[str, Any]]:
    if spec.loader == 'extxyz':
        assert spec.figshare_url is not None
        path = _prepare_extxyz_from_download(spec.figshare_url, cache_dir / spec.name)
        return _read_extxyz_topk(path, top_k)

    if spec.loader not in ROW_ADAPTERS:
        raise ValueError(f'Unknown loader {spec.loader}')

    paths = _download_hf_dataset_files(spec)
    candidates: List[tuple[int, int, int, Path]] = []
    for path in paths:
        candidates.extend(_top_candidates_in_parquet(path, spec.size_column, top_k))
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:top_k]

    selected: List[Dict[str, Any]] = []
    adapter = ROW_ADAPTERS[spec.loader]
    columns = FULL_COLUMNS[spec.loader]
    by_path: Dict[Path, List[tuple[int, int, int, Path]]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate[3]].append(candidate)

    for path, path_candidates in by_path.items():
        for row in _rows_from_parquet(path, path_candidates, columns):
            atoms = adapter(row)
            if spec.loader == 'colabfit':
                raw_name = row.get('names')
                if isinstance(raw_name, np.ndarray) and len(raw_name) > 0:
                    sample_name = str(raw_name[0])
                else:
                    sample_name = str(row.get('configuration_id'))
                natoms = int(row['nsites'])
            else:
                sample_name = str(row.get('filename') or row.get('mp_id'))
                natoms = int(row['num_atoms'])
            selected.append(
                {
                    'sample_id': sample_name,
                    'natoms': natoms,
                    'atoms': atoms,
                }
            )
    return sorted(selected, key=lambda item: item['natoms'], reverse=True)[:top_k]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluate(calc: SevenNetCalculator, atoms: Atoms) -> tuple[float, np.ndarray]:
    calc.calculate(atoms)
    return float(calc.results['energy']), np.asarray(calc.results['forces'])


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float('nan')
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(arr, q))


def benchmark_sample(
    atoms: Atoms,
    *,
    modal: str,
    case: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    calc = SevenNetCalculator(
        model='7net-omni',
        modal=modal,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_flash=case['enable_flash'],
        enable_pair_execution=case['enable_pair_execution'],
        pair_execution_policy=case.get('pair_execution_policy'),
    )
    try:
        _sync_cuda()
        cold_start = time.perf_counter()
        energy, forces = _evaluate(calc, atoms)
        _sync_cuda()
        cold_ms = (time.perf_counter() - cold_start) * 1000.0

        timings: List[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            _evaluate(calc, atoms)
            _sync_cuda()
            timings.append((time.perf_counter() - start) * 1000.0)

        return {
            'cold_ms': cold_ms,
            'steady_median_ms': float(np.median(timings)),
            'steady_p95_ms': _quantile(timings, 0.95),
            'energy': energy,
            'forces': forces,
            'resolved_policy': calc.pair_execution_config['resolved_policy'],
        }
    finally:
        del calc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _plot_grouped_bar(
    rows: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    pivot = rows.pivot(index='dataset', columns='case', values=value_column)
    ax = pivot.plot(kind='bar', figsize=(11, 5))
    ax.set_ylabel(value_column)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_speedup(
    rows: pd.DataFrame,
    *,
    base_case: str,
    target_case: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    pivot = rows.pivot(index='dataset', columns='case', values='steady_median_ms')
    speedup = pivot[base_case] / pivot[target_case]
    ax = speedup.plot(kind='bar', figsize=(10, 4.5))
    ax.set_ylabel(f'{base_case} / {target_case}')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _summary_lines(
    dataset_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
    agg_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    sample_frame = pd.DataFrame(sample_rows)
    case_names = (
        sorted(sample_frame['case'].unique().tolist()) if not sample_frame.empty else []
    )
    lines = [
        '# Real Dataset Benchmark',
        '',
        '## Scope',
        '',
        '- Model: `7net-omni`',
        '- Cases: '
        + (
            ', '.join(f'`{name}`' for name in case_names)
            if case_names
            else '(no cases recorded)'
        ),
        '- Measurement: ASE calculator end-to-end latency with repeated calls on identical topology',
        '',
        '## Dataset Coverage',
        '',
    ]
    for row in dataset_rows:
        lines.append(
            f"- `{row['dataset']}`: modal=`{row['modal']}`, samples={row['num_samples']}, "
            f"max_atoms={row['max_atoms']}, source={row['source']}"
        )

    agg_frame = pd.DataFrame(agg_rows)
    if not agg_frame.empty:
        lines.extend(['', '## Aggregated Results', ''])
        columns = list(agg_frame.columns)
        header = '| ' + ' | '.join(columns) + ' |'
        separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
        lines.append(header)
        lines.append(separator)
        for row in agg_rows:
            lines.append(
                '| ' + ' | '.join(str(row.get(column, '')) for column in columns) + ' |'
            )

        pivot = agg_frame.pivot(index='dataset', columns='case', values='steady_median_ms')
        if {'flash_baseline', 'flash_pair_auto'}.issubset(pivot.columns):
            speedup = (pivot['flash_baseline'] / pivot['flash_pair_auto']).sort_values(
                ascending=False
            )
            if not speedup.empty:
                lines.extend(
                    [
                        '',
                        '## Highlights',
                        '',
                        f"- Best `flash_pair_auto` steady-state speedup over `flash_baseline`: "
                        f"`{speedup.index[0]}` at `{speedup.iloc[0]:.3f}x`.",
                    ]
                )

    if not sample_frame.empty:
        worst_force = float(sample_frame['max_abs_force_diff_vs_e3nn'].max())
        worst_energy = float(sample_frame['abs_energy_diff_vs_e3nn'].max())
        lines.extend(
            [
                f"- Worst absolute energy delta vs e3nn baseline: `{worst_energy:.3e}` eV.",
                f"- Worst absolute force delta vs e3nn baseline: `{worst_force:.3e}` eV/A.",
            ]
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument(
        '--case-preset',
        choices=sorted(CASE_PRESETS.keys()),
        default='flash_real',
        help='named benchmark case set to execute',
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=[spec.name for spec in DATASETS],
        help='subset of dataset names to benchmark',
    )
    args = parser.parse_args(argv)

    selected_specs = [spec for spec in DATASETS if spec.name in set(args.datasets)]
    if not selected_specs:
        raise ValueError('No datasets selected')

    output_dir: Path = args.output_dir.resolve()
    cache_dir = output_dir / 'downloads'
    metrics_dir = output_dir / 'metrics'
    plots_dir = output_dir / 'plots'
    selected_cases = CASE_PRESETS[args.case_preset]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for spec in selected_specs:
        samples = load_dataset_samples(spec, cache_dir=cache_dir, top_k=args.top_k)
        dataset_rows.append(
            {
                'dataset': spec.name,
                'description': spec.description,
                'modal': spec.modal,
                'source': spec.source,
                'num_samples': len(samples),
                'max_atoms': max(sample['natoms'] for sample in samples),
            }
        )

        for sample in samples:
            baseline: Dict[str, Any] | None = None
            for case in selected_cases:
                result = benchmark_sample(
                    sample['atoms'],
                    modal=spec.modal,
                    case=case,
                    repeat=args.repeat,
                )
                row = {
                    'dataset': spec.name,
                    'description': spec.description,
                    'modal': spec.modal,
                    'sample_id': sample['sample_id'],
                    'natoms': sample['natoms'],
                    'case': case['case'],
                    'cold_ms': result['cold_ms'],
                    'steady_median_ms': result['steady_median_ms'],
                    'steady_p95_ms': result['steady_p95_ms'],
                    'resolved_policy': result['resolved_policy'],
                    'abs_energy_diff_vs_e3nn': 0.0,
                    'max_abs_force_diff_vs_e3nn': 0.0,
                }
                if case['case'] == 'e3nn_baseline':
                    baseline = result
                elif baseline is not None:
                    row['abs_energy_diff_vs_e3nn'] = abs(
                        result['energy'] - baseline['energy']
                    )
                    row['max_abs_force_diff_vs_e3nn'] = float(
                        np.max(np.abs(result['forces'] - baseline['forces']))
                    )
                sample_rows.append(row)

    _write_csv(metrics_dir / 'datasets.csv', dataset_rows)
    _write_csv(metrics_dir / 'samples.csv', sample_rows)

    sample_frame = pd.DataFrame(sample_rows)
    agg_frame = (
        sample_frame.groupby(['dataset', 'case'], as_index=False)
        .agg(
            modal=('modal', 'first'),
            mean_atoms=('natoms', 'mean'),
            max_atoms=('natoms', 'max'),
            cold_ms=('cold_ms', 'median'),
            steady_median_ms=('steady_median_ms', 'median'),
            steady_p95_ms=('steady_p95_ms', 'median'),
            max_abs_force_diff_vs_e3nn=('max_abs_force_diff_vs_e3nn', 'max'),
            abs_energy_diff_vs_e3nn=('abs_energy_diff_vs_e3nn', 'max'),
            resolved_policy=('resolved_policy', 'first'),
        )
    )
    agg_rows = agg_frame.to_dict(orient='records')
    _write_csv(metrics_dir / 'aggregated.csv', agg_rows)

    _plot_grouped_bar(
        agg_frame,
        value_column='steady_median_ms',
        title='Steady-state latency by dataset and case',
        out_path=plots_dir / 'steady_state_latency.png',
    )
    _plot_grouped_bar(
        agg_frame,
        value_column='cold_ms',
        title='Cold-call latency by dataset and case',
        out_path=plots_dir / 'cold_latency.png',
    )
    if {'flash_baseline', 'flash_pair_auto'}.issubset(set(agg_frame['case'])):
        _plot_speedup(
            agg_frame,
            base_case='flash_baseline',
            target_case='flash_pair_auto',
            title='FlashTP + pair execution speedup over FlashTP baseline',
            out_path=plots_dir / 'flash_pair_speedup.png',
        )

    summary = '\n'.join(_summary_lines(dataset_rows, sample_rows, agg_rows)) + '\n'
    summary_path = output_dir / 'summary.md'
    summary_path.write_text(summary)

    meta = {
        'repo_root': str(REPO_ROOT),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    (output_dir / 'environment.json').write_text(json.dumps(meta, indent=2))
    print(summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
