from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from ase.build import bulk

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
import sevenn.train.dataload as dataload
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import chemical_species_preprocess

from .report import generate_report
from .system_info import write_system_info

try:
    from sevenn.nn.flash_helper import is_flash_available
except Exception:  # pragma: no cover
    def is_flash_available() -> bool:
        return False


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pair_cfg(policy: str, backend_policy: str = 'auto') -> Dict[str, Any]:
    return {
        'use': policy != 'baseline',
        'policy': policy,
        'fuse_reduction': True,
        'use_topology_cache': True,
        'distributed_schedule': 'auto',
        'backend_policy': backend_policy,
    }


def _model_config(use_flash: bool = False, pair_policy: str = 'baseline'):
    config = {
        'cutoff': 4.5,
        'channel': 16,
        'lmax': 1,
        'is_parity': False,
        'num_convolution_layer': 2,
        'self_connection_type': 'nequip',
        'interaction_type': 'nequip',
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [16],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 16.0,
        'train_denominator': False,
        'shift': -1.0,
        'scale': 1.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
        KEY.USE_FLASH_TP: use_flash,
        KEY.PAIR_EXECUTION_CONFIG: _pair_cfg(
            pair_policy,
            backend_policy='prefer_flash' if use_flash else 'prefer_common',
        ),
    }
    config.update(chemical_species_preprocess(['Na', 'Cl']))
    return config


def _graph(device: torch.device, use_pair: bool, pair_policy: str) -> AtomGraphData:
    atoms = bulk('NaCl', 'rocksalt', a=4.0) * (2, 2, 2)
    graph = AtomGraphData.from_numpy_dict(
        dataload.unlabeled_atoms_to_graph(atoms, 4.5, with_shift=True)
    )
    if use_pair:
        graph = pair_runtime.ensure_pair_metadata_graph(graph, _pair_cfg(pair_policy))
    return graph.to(device)


def _clone_graph(graph: AtomGraphData) -> AtomGraphData:
    cloned = graph.clone()
    for key, value in list(cloned.items()):
        if torch.is_tensor(value):
            cloned[key] = value.clone()
    return cloned


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _measure(
    fn,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return {'median_ms': median, 'p95_ms': p95}


def run_perf_benchmarks(
    output_dir: Path,
    *,
    repeat: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_graph = _graph(device, use_pair=False, pair_policy='baseline')
    baseline_cfg = _model_config(use_flash=False, pair_policy='baseline')
    torch.manual_seed(123)
    baseline_model = build_E3_equivariant_model(baseline_cfg)
    baseline_model.to(device)
    baseline_model.eval()
    baseline_model.set_is_batch_data(False)
    baseline_ref = baseline_model(_clone_graph(baseline_graph))

    def _bench_case(case: str, use_flash: bool, pair_policy: str):
        cfg = _model_config(use_flash=use_flash, pair_policy=pair_policy)
        torch.manual_seed(123)
        model = build_E3_equivariant_model(cfg)
        model.to(device)
        model.eval()
        model.set_is_batch_data(False)
        graph = _graph(device, use_pair=pair_policy != 'baseline', pair_policy=pair_policy)

        def _forward():
            model(_clone_graph(graph))

        timing = _measure(_forward, warmup=warmup, repeat=repeat, device=device)
        out = model(_clone_graph(graph))
        metric = {
            'case': case,
            'device': str(device),
            'backend': 'flash' if use_flash else 'e3nn',
            'pair_policy': pair_policy,
            **timing,
            'max_abs_energy_diff': float(
                torch.max(
                    torch.abs(
                        baseline_ref[KEY.PRED_TOTAL_ENERGY].detach().cpu()
                        - out[KEY.PRED_TOTAL_ENERGY].detach().cpu()
                    )
                ).item()
            ),
            'max_abs_force_diff': float(
                torch.max(
                    torch.abs(
                        baseline_ref[KEY.PRED_FORCE].detach().cpu()
                        - out[KEY.PRED_FORCE].detach().cpu()
                    )
                ).item()
            ),
            'max_abs_stress_diff': float(
                torch.max(
                    torch.abs(
                        baseline_ref[KEY.PRED_STRESS].detach().cpu()
                        - out[KEY.PRED_STRESS].detach().cpu()
                    )
                ).item()
            ),
        }
        metrics.append(metric)

    _bench_case('baseline_e3nn_forward', False, 'baseline')
    _bench_case('pair_full_e3nn_forward', False, 'full')
    _bench_case('pair_geometry_only_e3nn_forward', False, 'geometry_only')

    if device.type == 'cuda' and is_flash_available():
        _bench_case('baseline_flash_forward', True, 'baseline')
        _bench_case('pair_auto_flash_forward', True, 'auto')
        _bench_case('pair_geometry_only_flash_forward', True, 'geometry_only')
    else:
        metrics.append(
            {
                'case': 'flash_benchmarks',
                'device': str(device),
                'backend': 'flash',
                'pair_policy': 'auto',
                'status': 'skipped',
                'reason': 'FlashTP or CUDA unavailable',
            }
        )

    perf_path = output_dir / 'metrics' / 'perf.json'
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    perf_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def _env_cmd(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


def _run_pytest_step(
    name: str,
    args: List[str],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    cmd = [sys.executable, '-m', 'pytest', '-q', *args]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f'{name}.stdout.log').write_text(proc.stdout)
    (log_dir / f'{name}.stderr.log').write_text(proc.stderr)
    status = 'passed' if proc.returncode == 0 else 'failed'
    return {
        'name': name,
        'kind': 'pytest',
        'status': status,
        'returncode': proc.returncode,
        'elapsed_sec': round(elapsed, 3),
        'stdout_log': str((log_dir / f'{name}.stdout.log').relative_to(output_dir)),
        'stderr_log': str((log_dir / f'{name}.stderr.log').relative_to(output_dir)),
    }


def _skip_step(name: str, reason: str) -> Dict[str, Any]:
    return {'name': name, 'kind': 'pytest', 'status': 'skipped', 'reason': reason}


def run_mode(mode: str, output_dir: Path) -> List[Dict[str, Any]]:
    lammps_cmd = _env_cmd('SEVENN_LAMMPS_CMD')
    mpirun_cmd = _env_cmd('SEVENN_MPIRUN_CMD')

    steps: List[Dict[str, Any]] = []
    if mode in {'unit', 'all'}:
        steps.append(_run_pytest_step('unit', ['tests/unit_tests'], output_dir=output_dir))
    if mode in {'train', 'all'}:
        steps.append(
            _run_pytest_step(
                'train', ['tests/unit_tests/test_train.py'], output_dir=output_dir
            )
        )
    if mode in {'inference', 'all'}:
        steps.append(
            _run_pytest_step(
                'inference',
                ['tests/unit_tests/test_cli.py', '-k', 'inference'],
                output_dir=output_dir,
            )
        )
    if mode in {'calculator', 'all'}:
        steps.append(
            _run_pytest_step(
                'calculator',
                ['tests/unit_tests/test_calculator.py'],
                output_dir=output_dir,
            )
        )
    if mode in {'deploy', 'all'}:
        steps.append(
            _run_pytest_step(
                'deploy',
                ['tests/unit_tests/test_cli.py', '-k', 'get_model'],
                output_dir=output_dir,
            )
        )
    if mode in {'lammps-serial', 'all'}:
        if not lammps_cmd:
            steps.append(_skip_step('lammps-serial', 'SEVENN_LAMMPS_CMD not set'))
        else:
            steps.append(
                _run_pytest_step(
                    'lammps-serial',
                    ['tests/lammps_tests/test_lammps.py', '--lammps_cmd', lammps_cmd],
                    output_dir=output_dir,
                )
            )
    if mode in {'lammps-parallel', 'all'}:
        if not lammps_cmd or not mpirun_cmd:
            steps.append(
                _skip_step(
                    'lammps-parallel',
                    'SEVENN_LAMMPS_CMD or SEVENN_MPIRUN_CMD not set',
                )
            )
        else:
            steps.append(
                _run_pytest_step(
                    'lammps-parallel',
                    [
                        'tests/lammps_tests/test_lammps.py',
                        '--lammps_cmd',
                        lammps_cmd,
                        '--mpirun_cmd',
                        mpirun_cmd,
                    ],
                    output_dir=output_dir,
                )
            )
    if mode in {'mliap', 'all'}:
        if not lammps_cmd:
            steps.append(_skip_step('mliap', 'SEVENN_LAMMPS_CMD not set'))
        else:
            args = ['tests/lammps_tests/test_mliap.py', '--lammps_cmd', lammps_cmd]
            if mpirun_cmd:
                args.extend(['--mpirun_cmd', mpirun_cmd])
            steps.append(_run_pytest_step('mliap', args, output_dir=output_dir))
    return steps


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        default='all',
        choices=[
            'unit',
            'train',
            'inference',
            'calculator',
            'deploy',
            'lammps-serial',
            'lammps-parallel',
            'mliap',
            'perf',
            'all',
            'report',
        ],
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=int(os.getenv('SEVENN_REPEAT', '7')))
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir.resolve()
    (output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'environment').mkdir(parents=True, exist_ok=True)

    write_system_info(REPO_ROOT, output_dir / 'environment' / 'system.json')

    steps: List[Dict[str, Any]] = []
    if args.mode in {'perf', 'all'}:
        run_perf_benchmarks(output_dir, repeat=args.repeat, warmup=args.warmup)
    if args.mode in {
        'unit',
        'train',
        'inference',
        'calculator',
        'deploy',
        'lammps-serial',
        'lammps-parallel',
        'mliap',
        'all',
    }:
        steps = run_mode(args.mode, output_dir)
        (output_dir / 'metrics' / 'steps.json').write_text(
            json.dumps(steps, indent=2)
        )
    summary_path = generate_report(output_dir)
    print(summary_path)

    failures = [step for step in steps if step.get('status') == 'failed']
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
