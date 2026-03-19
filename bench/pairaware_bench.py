#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import torch
from ase.build import bulk
from ase.io import read as ase_read

import sevenn._keys as KEY
import sevenn.train.dataload as dl
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.profiling import ModuleProfiler, synchronize_device


def infer_device(device: str) -> torch.device:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def generate_atoms(target_atoms: int):
    base = bulk('NaCl', 'rocksalt', a=5.63)
    repeat = max(1, math.ceil((target_atoms / len(base)) ** (1.0 / 3.0)))
    atoms = base * (repeat, repeat, repeat)
    atoms.rattle(stdev=0.01, seed=7)
    return atoms


def load_atoms(args):
    if args.structure is not None:
        return ase_read(args.structure, index=args.index)
    return generate_atoms(args.target_atoms)


def build_graph(atoms, cutoff: float, device: torch.device) -> AtomGraphData:
    graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(atoms, cutoff))
    graph = graph.to(device)  # type: ignore[assignment]
    return graph


def run_step(model, graph_template, device: torch.device):
    graph = graph_template.clone()
    graph = graph.to(device)  # type: ignore[assignment]
    synchronize_device(device)
    start = time.perf_counter()
    output = model(graph)
    synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def validate_effective_mode(requested_runtime, effective_runtime):
    requested_flags = util.get_runtime_mode_flags(requested_runtime)
    effective_flags = util.get_runtime_mode_flags(effective_runtime)
    missing = []
    for key, label in (
        (KEY.USE_PAIRAWARE, 'pairaware'),
        (KEY.USE_FLASH_TP, 'flashtp'),
        ('cuequivariance', 'cueq'),
        (KEY.USE_OEQ, 'oeq'),
    ):
        if requested_flags[key] and not effective_flags[key]:
            missing.append(label)
    if missing:
        raise RuntimeError(
            'Requested accelerator mode not activated: ' + ', '.join(missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark baseline vs pair-aware geometry reuse.'
    )
    parser.add_argument('--checkpoint', default='7net-0', type=str)
    parser.add_argument('--structure', default=None, type=str)
    parser.add_argument('--index', default='0', type=str)
    parser.add_argument('--target-atoms', default=256, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--steps', default=5, type=int)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--enable_pairaware', action='store_true')
    parser.add_argument('--enable_flash', action='store_true')
    parser.add_argument('--enable_cueq', action='store_true')
    parser.add_argument('--enable_oeq', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-dir', default='bench/profiles', type=str)
    args = parser.parse_args()

    device = infer_device(args.device)
    model, config = util.model_from_checkpoint(
        args.checkpoint,
        enable_cueq=args.enable_cueq or None,
        enable_flash=args.enable_flash or None,
        enable_oeq=args.enable_oeq or None,
        enable_pairaware=args.enable_pairaware or None,
    )
    requested_runtime = util.with_runtime_mode(
        config,
        enable_cueq=args.enable_cueq or None,
        enable_flash=args.enable_flash or None,
        enable_oeq=args.enable_oeq or None,
        enable_pairaware=args.enable_pairaware or None,
    )
    effective_runtime = util.runtime_mode_from_model(model, requested_runtime)
    try:
        validate_effective_mode(requested_runtime, effective_runtime)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    atoms = load_atoms(args)
    model = model.to(device)
    model.set_is_batch_data(False)
    model.eval()
    graph_template = build_graph(atoms, model.cutoff, device)

    profiler = ModuleProfiler(device)
    handles = profiler.register(model)

    for _ in range(args.warmup):
        _, _ = run_step(model, graph_template, device)
        profiler.flush()
        profiler.reset()

    profile_path = None
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
        profile_path = (
            Path(args.profile_dir)
            / f'{util.resolve_runtime_mode(effective_runtime)}-{int(time.time())}.json'
        )
        with profile(
            activities=[ProfilerActivity.CPU]
            + (
                [ProfilerActivity.CUDA] if device.type == 'cuda' else []
            ),
        ) as prof:
            _, _ = run_step(model, graph_template, device)
        prof.export_chrome_trace(str(profile_path))

    total_ms = 0.0
    last_output = None
    for _ in range(args.steps):
        output, elapsed_ms = run_step(model, graph_template, device)
        total_ms += elapsed_ms
        last_output = output
        profiler.flush()
        profiler.timings.total_ms += elapsed_ms

    for handle in handles:
        handle.remove()

    assert last_output is not None
    avg_total_ms = total_ms / float(args.steps)
    avg_geometry_ms = profiler.timings.geometry_ms / float(args.steps)
    avg_tp_ms = profiler.timings.tensor_product_ms / float(args.steps)
    natoms = len(atoms)
    num_edges = int(last_output[KEY.EDGE_IDX].shape[1])
    num_pairs = int(
        last_output.get(KEY.PAIRAWARE_NUM_PAIRS, torch.tensor(num_edges)).item()
    )
    reuse_factor = float(
        last_output.get(KEY.PAIRAWARE_REUSE_FACTOR, torch.tensor(1.0)).item()
    )

    print(f'requested_mode={util.resolve_runtime_mode(requested_runtime)}')
    print(f'effective_mode={util.resolve_runtime_mode(effective_runtime)}')
    print(f'flags={util.format_runtime_mode(effective_runtime)}')
    print(f'device={device}')
    print(f'natoms={natoms}')
    print(f'num_edges_directed={num_edges}')
    print(f'num_pairs_undirected={num_pairs}')
    print(f'geometry_reuse_factor={reuse_factor:.3f}')
    print(f'avg_step_ms={avg_total_ms:.3f}')
    print(f'avg_geometry_ms={avg_geometry_ms:.3f}')
    print(f'avg_tensor_product_ms={avg_tp_ms:.3f}')
    print(f'throughput_atoms_per_s={(1000.0 * natoms) / avg_total_ms:.3f}')
    print(f'throughput_edges_per_s={(1000.0 * num_edges) / avg_total_ms:.3f}')
    if profile_path is not None:
        print(f'profile_trace={profile_path}')


if __name__ == '__main__':
    main()
