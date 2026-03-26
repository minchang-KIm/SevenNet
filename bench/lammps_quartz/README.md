# LAMMPS Quartz Benchmark

This directory contains a reproducible LAMMPS benchmark setup for comparing
SevenNet `e3gnn/parallel` export variants on a single-GPU server.

## Files

- `lammps_env.sh`: runtime environment for `lmp`, LibTorch, CUDA libraries,
  and the required unlimited stack size for `e3gnn/parallel`.
- `run_quartz_bench.sh`: generates LAMMPS inputs, runs the benchmark variants,
  and writes logs/results under `runs/`.
- `parse_quartz_bench.py`: parses LAMMPS logs into `summary.csv` and
  `summary.md`.

## Environment

After sourcing the env script, `lmp` points to the patched LAMMPS build at:

```bash
/home/wise/minchang/DenseMLIP/lammps_sevenn/build/lmp
```

Use:

```bash
source bench/lammps_quartz/lammps_env.sh
lmp -help | head
```

## Model Deployment

The benchmark expects four parallel-export model directories:

```bash
sevenn_get_model 7net-0 -p -o bench/lammps_quartz/models/baseline/7net0_parallel
sevenn_get_model 7net-0 -p --enable_pairaware -o bench/lammps_quartz/models/pairaware/7net0_parallel_pairaware
sevenn_get_model 7net-0 -p --enable_flash -o bench/lammps_quartz/models/flash/7net0_parallel_flash
sevenn_get_model 7net-0 -p --enable_flash --enable_pairaware -o bench/lammps_quartz/models/combined/7net0_parallel_flash_pairaware
```

## Run

The paper-style benchmark protocol used here is:

- alpha-quartz input from LAMMPS `data.quartz`
- `units metal`
- `NVT` at `300 K`
- `timestep 0.002`
- `run 110` warmup + `run 100` measurement

For this server, the paper weak-scaling size (`4608 atoms/GPU`, `8x8x8`) did
not fit in `e3gnn/parallel` on RTX 4090 24 GB, so the practical single-GPU
benchmark was reduced to `7x7x7 = 3087 atoms`.

Run:

```bash
source bench/lammps_quartz/lammps_env.sh
DATE_TAG=rerun_quartz_r7 REPLICATE_X=7 REPLICATE_Y=7 REPLICATE_Z=7 bench/lammps_quartz/run_quartz_bench.sh
```

## Reference Result

Measured on 2026-03-25 with `3087` atoms:

| variant | atoms | step_ms | timesteps_per_s | atoms_per_s | ns_per_day | vs_baseline_step |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3087 | 414.415 | 2.413 | 7448.931 | 0.417 | +0.00% |
| pairaware | 3087 | 416.943 | 2.398 | 7402.626 | 0.414 | -0.61% |
| flash | 3087 | 45.364 | 22.044 | 68049.828 | 3.809 | +89.05% |
| combined | 3087 | 49.196 | 20.327 | 62749.449 | 3.513 | +88.13% |

Interpretation:

- `pairaware` alone did not improve runtime on this server.
- `flash` gave the dominant speedup, about `9.1x` over baseline.
- `combined` remained much faster than baseline, but was slightly slower than
  `flash` alone in this run.
