import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import ase.io
import torch

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.flash_helper import is_flash_available
from sevenn.nn.pairgeom import build_undirected_pair_mapping
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint


def _load_atoms(structure: str, index: str, target_atoms: int | None):
    atoms = ase.io.read(structure, index=index)
    if target_atoms is None or len(atoms) >= target_atoms:
        return atoms

    repeat = max(1, math.ceil((target_atoms / len(atoms)) ** (1.0 / 3.0)))
    return atoms.repeat((repeat, repeat, repeat))


def _build_graph(
    atoms, cutoff: float, with_pair_metadata: bool = False
) -> AtomGraphData:
    return AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(
            atoms,
            cutoff,
            with_shift=True,
            with_pair_metadata=with_pair_metadata,
        )
    )


@dataclass
class ModuleTimer:
    device: torch.device
    totals_ms: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)
    _pending: List[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )

    def _start(self, key: str) -> None:
        if self.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending.append((key, start, end))
            return
        self._starts[key] = time.perf_counter()

    def _stop(self, key: str) -> None:
        if self.device.type == 'cuda':
            pending_key, _, end = self._pending[-1]
            assert pending_key == key
            end.record()
            return
        elapsed_ms = (time.perf_counter() - self._starts.pop(key)) * 1000.0
        self.totals_ms[key] = self.totals_ms.get(key, 0.0) + elapsed_ms
        self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self) -> None:
        if self.device.type != 'cuda' or not self._pending:
            return
        torch.cuda.synchronize(self.device)
        for key, start, end in self._pending:
            elapsed_ms = start.elapsed_time(end)
            self.totals_ms[key] = self.totals_ms.get(key, 0.0) + elapsed_ms
            self.counts[key] = self.counts.get(key, 0) + 1
        self._pending.clear()

    def make_pre_hook(self, key: str):
        def hook(_module, _inputs):
            self._start(key)

        return hook

    def make_post_hook(self, key: str):
        def hook(_module, _inputs, _output):
            self._stop(key)

        return hook

    def average(self, key: str, denom: int) -> float:
        if denom == 0:
            return 0.0
        return self.totals_ms.get(key, 0.0) / float(denom)


def _register_timers(model, timer: ModuleTimer):
    handles = []
    for name, module in model._modules.items():
        if name == 'edge_embedding':
            key = 'geometry'
        elif name.endswith('_convolution'):
            key = 'tensor_product'
        else:
            continue
        handles.append(module.register_forward_pre_hook(timer.make_pre_hook(key)))
        handles.append(module.register_forward_hook(timer.make_post_hook(key)))
    return handles


def _resolve_modes(mode: str) -> List[dict]:
    modes = {
        'baseline': {'name': 'baseline', 'enable_pairgeom': False, 'enable_flash': False},
        'pairgeom': {'name': 'pairgeom', 'enable_pairgeom': True, 'enable_flash': False},
        'flash': {'name': 'flash', 'enable_pairgeom': False, 'enable_flash': True},
        'pairgeom_flash': {
            'name': 'pairgeom+flash',
            'enable_pairgeom': True,
            'enable_flash': True,
        },
    }
    if mode == 'all':
        return [
            modes['baseline'],
            modes['pairgeom'],
            modes['flash'],
            modes['pairgeom_flash'],
        ]
    if mode not in modes:
        raise ValueError(f'Unknown mode: {mode}')
    return [modes[mode]]


def benchmark_mode(
    checkpoint: str,
    structure: str,
    *,
    index: str = '-1',
    target_atoms: int | None = None,
    warmup: int = 3,
    steps: int = 10,
    device: str = 'cpu',
    enable_pairgeom: bool = False,
    enable_flash: bool = False,
    profile: bool = False,
    profile_dir: str = 'bench/profiles',
) -> dict:
    run_device = torch.device(device)
    if enable_flash and not is_flash_available():
        raise RuntimeError('FlashTP is not available for this benchmark mode')

    model, cfg = model_from_checkpoint(
        checkpoint,
        enable_flash=enable_flash or None,
        enable_pairgeom=enable_pairgeom or None,
    )
    model.to(run_device)
    model.set_is_batch_data(False)
    model.eval()

    atoms = _load_atoms(structure, index, target_atoms)
    graph_template = _build_graph(
        atoms, cfg[KEY.CUTOFF], with_pair_metadata=enable_pairgeom
    )
    graph_template[KEY.BATCH] = torch.zeros([0])
    pair_index, _, _, _ = build_undirected_pair_mapping(
        graph_template[KEY.EDGE_IDX], graph_template.get(KEY.CELL_SHIFT)
    )

    timer = ModuleTimer(run_device)
    handles = _register_timers(model, timer)
    step_times = []
    trace_path = None

    def run_once(with_profile: bool = False):
        graph = graph_template.clone().to(run_device)
        if run_device.type == 'cuda':
            torch.cuda.synchronize(run_device)
        start = time.perf_counter()
        if with_profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if run_device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile_root = Path(profile_dir)
            profile_root.mkdir(parents=True, exist_ok=True)
            nonlocal trace_path
            trace_path = str(
                profile_root / f"{'pairgeom' if enable_pairgeom else 'baseline'}-{int(time.time())}.json"
            )
            with torch.profiler.profile(activities=activities) as prof:
                _ = model(graph)
            prof.export_chrome_trace(trace_path)
        else:
            _ = model(graph)
        timer.flush()
        if run_device.type == 'cuda':
            torch.cuda.synchronize(run_device)
        return (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        run_once()
    for step_idx in range(steps):
        step_times.append(run_once(with_profile=profile and step_idx == steps - 1))

    for handle in handles:
        handle.remove()

    num_edges = int(graph_template[KEY.EDGE_IDX].shape[1])
    num_pairs = int(pair_index.shape[1])
    requested_mode = 'pairgeom' if enable_pairgeom else 'baseline'
    if enable_flash and not enable_pairgeom:
        requested_mode = 'flash'
    elif enable_flash and enable_pairgeom:
        requested_mode = 'pairgeom+flash'

    result = {
        'requested_mode': requested_mode,
        'effective_mode': 'pairgeom+flash'
        if enable_pairgeom and enable_flash
        else 'pairgeom'
        if enable_pairgeom
        else 'flash'
        if enable_flash
        else 'baseline',
        'flags': {
            'use_pairgeom': enable_pairgeom,
            'use_flash_tp': enable_flash,
        },
        'device': str(run_device),
        'natoms': len(atoms),
        'num_edges_directed': num_edges,
        'num_pairs_undirected': num_pairs,
        'geometry_reuse_factor': float(num_edges / num_pairs),
        'avg_step_ms': sum(step_times) / len(step_times),
        'avg_geometry_ms': timer.average('geometry', len(step_times)),
        'avg_tensor_product_ms': timer.average('tensor_product', len(step_times)),
        'throughput_atoms_per_s': len(atoms) / (sum(step_times) / len(step_times) / 1000.0),
        'throughput_edges_per_s': num_edges / (sum(step_times) / len(step_times) / 1000.0),
        'profile_trace': trace_path,
    }
    return result


def run_suite(args) -> List[dict]:
    results = []
    for mode in _resolve_modes(args.mode):
        if mode['enable_flash'] and not is_flash_available():
            continue
        results.append(
            benchmark_mode(
                args.checkpoint,
                args.structure,
                index=args.index,
                target_atoms=args.target_atoms,
                warmup=args.warmup,
                steps=args.steps,
                device=args.device,
                enable_pairgeom=mode['enable_pairgeom'],
                enable_flash=mode['enable_flash'],
                profile=args.profile,
                profile_dir=args.profile_dir,
            )
        )
    return results


def _print_result(result: dict) -> None:
    print(f"requested_mode={result['requested_mode']}")
    print(f"effective_mode={result['effective_mode']}")
    print(
        'flags='
        f"use_pairgeom={result['flags']['use_pairgeom']}, "
        f"use_flash_tp={result['flags']['use_flash_tp']}"
    )
    print(f"device={result['device']}")
    print(f"natoms={result['natoms']}")
    print(f"num_edges_directed={result['num_edges_directed']}")
    print(f"num_pairs_undirected={result['num_pairs_undirected']}")
    print(f"geometry_reuse_factor={result['geometry_reuse_factor']:.3f}")
    print(f"avg_step_ms={result['avg_step_ms']:.3f}")
    print(f"avg_geometry_ms={result['avg_geometry_ms']:.3f}")
    print(f"avg_tensor_product_ms={result['avg_tensor_product_ms']:.3f}")
    print(f"throughput_atoms_per_s={result['throughput_atoms_per_s']:.3f}")
    print(f"throughput_edges_per_s={result['throughput_edges_per_s']:.3f}")
    if result['profile_trace']:
        print(f"profile_trace={result['profile_trace']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Benchmark baseline, pairgeom, flash, and pairgeom+flash.'
    )
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('structure', type=str)
    parser.add_argument('--index', default='-1')
    parser.add_argument('--target-atoms', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--mode',
        choices=['all', 'baseline', 'pairgeom', 'flash', 'pairgeom_flash'],
        default='all',
    )
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-dir', default='bench/profiles')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for result in run_suite(args):
        _print_result(result)


if __name__ == '__main__':
    main()
