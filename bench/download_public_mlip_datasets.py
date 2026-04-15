from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / 'datasets'


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    category: str
    source_kind: str
    source: str
    local_relpath: str
    approx_size_gib: float
    representative_papers: str
    notes: str = ''
    large: bool = False
    gated: bool = False


REGISTRY: Sequence[DatasetEntry] = (
    DatasetEntry(
        name='qm9_hf',
        category='molecular',
        source_kind='hf',
        source='nimashoghi/qm9',
        local_relpath='raw/hf/qm9_hf',
        approx_size_gib=0.05,
        representative_papers='SchNet, DimeNet, PaiNN, GemNet-QM9',
        notes='Equilibrium small-molecule property dataset often used in atomistic GNN pretraining/validation.',
    ),
    DatasetEntry(
        name='md17_aspirin',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-aspirin',
        local_relpath='raw/hf/md17_aspirin',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_benzene',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-benzene',
        local_relpath='raw/hf/md17_benzene',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_ethanol',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-ethanol',
        local_relpath='raw/hf/md17_ethanol',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_malonaldehyde',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-malonaldehyde',
        local_relpath='raw/hf/md17_malonaldehyde',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_naphthalene',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-naphthalene',
        local_relpath='raw/hf/md17_naphthalene',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_salicylic_acid',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-salicylic_acid',
        local_relpath='raw/hf/md17_salicylic_acid',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_toluene',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-toluene',
        local_relpath='raw/hf/md17_toluene',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='md17_uracil',
        category='molecular',
        source_kind='hf',
        source='graphs-datasets/MD17-uracil',
        local_relpath='raw/hf/md17_uracil',
        approx_size_gib=0.01,
        representative_papers='SchNet, DimeNet, PaiNN, TorchMD-NET',
        notes='Original MD17 subset mirror.',
    ),
    DatasetEntry(
        name='rmd17',
        category='molecular',
        source_kind='hf',
        source='colabfit/rMD17',
        local_relpath='raw/hf/rmd17',
        approx_size_gib=0.98,
        representative_papers='NequIP, MACE, Allegro',
        notes='Revised MD17 mirror on ColabFit.',
    ),
    DatasetEntry(
        name='iso17',
        category='molecular',
        source_kind='hf',
        source='colabfit/ISO17_NC_2017',
        local_relpath='raw/hf/iso17',
        approx_size_gib=0.68,
        representative_papers='SchNet, NequIP, MACE',
        notes='ISO17 force-field benchmark mirror.',
    ),
    DatasetEntry(
        name='md22_at_at',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_AT_AT',
        local_relpath='raw/hf/md22_at_at',
        approx_size_gib=0.04,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_at_at_cg_cg',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_AT_AT_CG_CG',
        local_relpath='raw/hf/md22_at_at_cg_cg',
        approx_size_gib=0.04,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_ac_ala3_nhme',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_Ac_Ala3_NHMe',
        local_relpath='raw/hf/md22_ac_ala3_nhme',
        approx_size_gib=0.14,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_dha',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_DHA',
        local_relpath='raw/hf/md22_dha',
        approx_size_gib=0.14,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_double_walled_nanotube',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_double_walled_nanotube',
        local_relpath='raw/hf/md22_double_walled_nanotube',
        approx_size_gib=0.06,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_stachyose',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_stachyose',
        local_relpath='raw/hf/md22_stachyose',
        approx_size_gib=0.08,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='md22_buckyball_catcher',
        category='molecular',
        source_kind='hf',
        source='colabfit/MD22_buckyball_catcher',
        local_relpath='raw/hf/md22_buckyball_catcher',
        approx_size_gib=0.03,
        representative_papers='MD22 benchmark, MACE molecular benchmarks',
    ),
    DatasetEntry(
        name='ani1x',
        category='molecular',
        source_kind='hf',
        source='colabfit/ANI-1x',
        local_relpath='raw/hf/ani1x',
        approx_size_gib=0.30,
        representative_papers='TorchANI, ANI family, MatterSim molecular evaluations',
    ),
    DatasetEntry(
        name='ani1ccx',
        category='molecular',
        source_kind='hf',
        source='roitberg-group/ani1ccx',
        local_relpath='raw/hf/ani1ccx',
        approx_size_gib=0.25,
        representative_papers='ANI-1ccx benchmark, TorchANI',
    ),
    DatasetEntry(
        name='spice_2023',
        category='molecular',
        source_kind='hf',
        source='colabfit/SPICE_2023',
        local_relpath='raw/hf/spice_2023',
        approx_size_gib=2.41,
        representative_papers='SPICE / SPICE2, modern molecular foundation-model evaluations',
    ),
    DatasetEntry(
        name='mptrj',
        category='materials',
        source_kind='hf',
        source='nimashoghi/mptrj',
        local_relpath='raw/hf/mptrj',
        approx_size_gib=1.63,
        representative_papers='CHGNet, M3GNet, MACE-MP',
    ),
    DatasetEntry(
        name='wbm_initial',
        category='materials',
        source_kind='url',
        source='https://ndownloader.figshare.com/files/48169597',
        local_relpath='raw/figshare/wbm_initial.extxyz.zip',
        approx_size_gib=0.0,
        representative_papers='Matbench Discovery, OMat24',
        notes='Matbench Discovery benchmark structures archive.',
    ),
    DatasetEntry(
        name='phonondb_pbe',
        category='materials',
        source_kind='url',
        source='https://ndownloader.figshare.com/files/52179965',
        local_relpath='raw/figshare/phonondb_pbe.extxyz.zip',
        approx_size_gib=0.0,
        representative_papers='Matbench Discovery, phonon benchmark subsets',
        notes='phononDB benchmark extxyz archive.',
    ),
    DatasetEntry(
        name='omat24_1m_official',
        category='materials',
        source_kind='url',
        source='https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/omat24_1M_251210.tar.gz',
        local_relpath='raw/fairchem/omat24_1m/omat24_1M_251210.tar.gz',
        approx_size_gib=2.12,
        representative_papers='OMat24',
        notes='Official 1M OMat24 subsplit tarball.',
    ),
    DatasetEntry(
        name='salex_train_official',
        category='materials',
        source_kind='url',
        source='https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz',
        local_relpath='raw/fairchem/salex/train.tar.gz',
        approx_size_gib=7.52,
        representative_papers='OMat24, UMA inorganic fine-tuning',
        notes='Official sAlex train tarball.',
    ),
    DatasetEntry(
        name='salex_val_official',
        category='materials',
        source_kind='url',
        source='https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz',
        local_relpath='raw/fairchem/salex/val.tar.gz',
        approx_size_gib=0.40,
        representative_papers='OMat24, UMA inorganic fine-tuning',
        notes='Official sAlex validation tarball.',
    ),
    DatasetEntry(
        name='omol25_train_4m',
        category='molecular',
        source_kind='hf',
        source='colabfit/OMol25_train_4M',
        local_relpath='raw/hf/omol25_train_4m',
        approx_size_gib=11.49,
        representative_papers='OMol25',
        notes='Public 4M training subset mirror.',
    ),
    DatasetEntry(
        name='omol25_validation',
        category='molecular',
        source_kind='hf',
        source='colabfit/OMol25_validation',
        local_relpath='raw/hf/omol25_validation',
        approx_size_gib=13.62,
        representative_papers='OMol25',
        notes='Public validation mirror.',
    ),
    DatasetEntry(
        name='oc20_s2ef_train_2m',
        category='catalysis',
        source_kind='hf',
        source='nimashoghi/oc20_s2ef_train_2M',
        local_relpath='raw/hf/oc20_s2ef_train_2m',
        approx_size_gib=4.74,
        representative_papers='GemNet, Allegro, FairChem OC20 baselines',
        notes='Public 2M train subset mirror.',
    ),
    DatasetEntry(
        name='oc20_s2ef_val_id',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC20_S2EF_val_id',
        local_relpath='raw/hf/oc20_s2ef_val_id',
        approx_size_gib=3.00,
        representative_papers='GemNet, Allegro, FairChem OC20 baselines',
    ),
    DatasetEntry(
        name='oc20_s2ef_val_ood_ads',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC20_S2EF_val_ood_ads',
        local_relpath='raw/hf/oc20_s2ef_val_ood_ads',
        approx_size_gib=3.13,
        representative_papers='GemNet, Allegro, FairChem OC20 baselines',
    ),
    DatasetEntry(
        name='oc20_s2ef_val_ood_cat',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC20_S2EF_val_ood_cat',
        local_relpath='raw/hf/oc20_s2ef_val_ood_cat',
        approx_size_gib=3.11,
        representative_papers='GemNet, Allegro, FairChem OC20 baselines',
    ),
    DatasetEntry(
        name='oc20_s2ef_val_ood_both',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC20_S2EF_val_ood_both',
        local_relpath='raw/hf/oc20_s2ef_val_ood_both',
        approx_size_gib=3.31,
        representative_papers='GemNet, Allegro, FairChem OC20 baselines',
    ),
    DatasetEntry(
        name='oc22_s2ef_train',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC22-S2EF-Train',
        local_relpath='raw/hf/oc22_s2ef_train',
        approx_size_gib=30.13,
        representative_papers='SevenNet, UMA catalyst models, OC22 baselines',
    ),
    DatasetEntry(
        name='oc22_s2ef_val_id',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC22-S2EF-Validation-in-domain',
        local_relpath='raw/hf/oc22_s2ef_val_id',
        approx_size_gib=1.39,
        representative_papers='SevenNet, UMA catalyst models, OC22 baselines',
    ),
    DatasetEntry(
        name='oc22_s2ef_val_ood',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC22-S2EF-Validation-out-of-domain',
        local_relpath='raw/hf/oc22_s2ef_val_ood',
        approx_size_gib=1.61,
        representative_papers='SevenNet, UMA catalyst models, OC22 baselines',
    ),
    DatasetEntry(
        name='oc20_s2ef_train_20m',
        category='catalysis',
        source_kind='hf',
        source='colabfit/OC20_S2EF_train_20M',
        local_relpath='raw/hf/oc20_s2ef_train_20m',
        approx_size_gib=76.22,
        representative_papers='GemNet-OC, EquiformerV2, Allegro full OC20-scale training',
        notes='Very large public train subset mirror.',
        large=True,
    ),
    DatasetEntry(
        name='omol25_train_neutral',
        category='molecular',
        source_kind='hf',
        source='colabfit/OMol25_train_neutral',
        local_relpath='raw/hf/omol25_train_neutral',
        approx_size_gib=59.51,
        representative_papers='OMol25',
        notes='Large public neutral-train mirror.',
        large=True,
    ),
    DatasetEntry(
        name='omol25_official_gated',
        category='molecular',
        source_kind='hf',
        source='facebook/OMol25',
        local_relpath='raw/hf/omol25_official_gated',
        approx_size_gib=0.0,
        representative_papers='OMol25',
        notes='Official HF dataset repo is gated; login-required.',
        gated=True,
    ),
)


def iter_selected(names: set[str] | None, include_large: bool) -> Iterable[DatasetEntry]:
    for entry in REGISTRY:
        if names is not None and entry.name not in names:
            continue
        if entry.large and not include_large:
            continue
        yield entry


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob('*'):
        if child.is_file():
            total += child.stat().st_size
    return total


def repo_relative_dataset_path(local_relpath: str) -> str:
    return str(Path('datasets') / local_relpath)


def stream_download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dst.open('wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_entry(entry: DatasetEntry, root: Path, max_workers: int) -> tuple[str, str]:
    dst = root / entry.local_relpath
    if entry.gated:
        return 'gated', 'not downloaded'

    if entry.source_kind == 'hf':
        if dst.exists() and any(dst.iterdir()):
            return 'exists', 'reused existing snapshot'
        dst.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=entry.source,
            repo_type='dataset',
            local_dir=dst,
            max_workers=max_workers,
            resume_download=True,
        )
        return 'downloaded', 'snapshot_download complete'

    if entry.source_kind == 'url':
        stream_download(entry.source, dst)
        return 'downloaded' if dst.exists() else 'missing', 'url download complete'

    raise ValueError(f'Unknown source_kind {entry.source_kind}')


def write_inventory(entries: Sequence[DatasetEntry], root: Path, statuses: dict[str, tuple[str, str]]) -> Path:
    inventory_path = root / 'inventory.csv'
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'name',
                'category',
                'source_kind',
                'source',
                'local_path',
                'approx_size_gib',
                'representative_papers',
                'large',
                'gated',
                'status',
                'message',
                'downloaded_size_bytes',
                'notes',
            ],
        )
        writer.writeheader()
        for entry in entries:
            local_path = root / entry.local_relpath
            repo_local_path = repo_relative_dataset_path(entry.local_relpath)
            if entry.name in statuses:
                status, message = statuses[entry.name]
            elif entry.gated:
                status, message = ('gated', 'not downloaded')
            elif local_path.exists():
                status, message = ('exists', 'found on disk')
            else:
                status, message = ('pending', '')
            writer.writerow(
                {
                    'name': entry.name,
                    'category': entry.category,
                    'source_kind': entry.source_kind,
                    'source': entry.source,
                    'local_path': repo_local_path,
                    'approx_size_gib': entry.approx_size_gib,
                    'representative_papers': entry.representative_papers,
                    'large': entry.large,
                    'gated': entry.gated,
                    'status': status,
                    'message': message,
                    'downloaded_size_bytes': directory_size_bytes(local_path),
                    'notes': entry.notes,
                }
            )
    return inventory_path


def write_summary(entries: Sequence[DatasetEntry], root: Path, statuses: dict[str, tuple[str, str]]) -> Path:
    summary_path = root / 'README.md'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Public MLIP Dataset Cache',
        '',
        'This folder stores public datasets that are repeatedly used in MLIP papers and benchmarks.',
        '',
        'Generation:',
        '',
        '- Script: `python bench/download_public_mlip_datasets.py`',
        f'- Registry entries: `{len(entries)}`',
        '',
        '## Status',
        '',
        '| name | category | approx_size_gib | representative_papers | status | source | local_path |',
        '| --- | --- | ---: | --- | --- | --- | --- |',
    ]
    for entry in entries:
        local_path = root / entry.local_relpath
        repo_local_path = repo_relative_dataset_path(entry.local_relpath)
        if entry.name in statuses:
            status, _ = statuses[entry.name]
        elif entry.gated:
            status = 'gated'
        elif local_path.exists():
            status = 'exists'
        else:
            status = 'pending'
        lines.append(
            f'| `{entry.name}` | `{entry.category}` | {entry.approx_size_gib:.2f} | '
            f'{entry.representative_papers} | `{status}` | '
            f'`{entry.source}` | `{repo_local_path}` |'
        )

    lines.extend(
        [
            '',
            '## Notes',
            '',
            '- Background log path when running long jobs: `datasets/logs/public_download.log`.',
            '- `large=true` entries are skipped unless `--include-large` is passed.',
            '- `gated=true` entries are listed for completeness but require authentication or manual access approval.',
            '- Mirrors on Hugging Face / ColabFit are used when they are easier to automate than the original paper distribution.',
        ]
    )
    summary_path.write_text('\n'.join(lines) + '\n')
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=DATASET_ROOT)
    parser.add_argument('--names', nargs='*', help='subset of registry names to download')
    parser.add_argument('--include-large', action='store_true')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--max-workers', type=int, default=8)
    args = parser.parse_args(argv)

    selected = list(iter_selected(set(args.names) if args.names else None, args.include_large))
    if not selected:
        raise ValueError('No datasets selected')

    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, tuple[str, str]] = {}

    if not args.inventory_only:
        for entry in selected:
            print(
                f'[download] {entry.name} '
                f'({entry.category}, ~{entry.approx_size_gib:.2f} GiB) -> {root / entry.local_relpath}',
                flush=True,
            )
            try:
                statuses[entry.name] = download_entry(entry, root, max_workers=args.max_workers)
            except Exception as exc:
                statuses[entry.name] = ('error', str(exc))
            print(f'[status] {entry.name}: {statuses[entry.name][0]}', flush=True)

    inventory_path = write_inventory(selected, root, statuses)
    summary_path = write_summary(selected, root, statuses)
    meta_path = root / 'inventory.json'
    meta_path.write_text(
        json.dumps(
            {
                'selected_names': [entry.name for entry in selected],
                'include_large': args.include_large,
                'inventory_csv': 'datasets/inventory.csv',
                'summary_md': 'datasets/README.md',
            },
            indent=2,
        )
        + '\n'
    )
    print(f'Wrote inventory: {inventory_path}')
    print(f'Wrote summary: {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
