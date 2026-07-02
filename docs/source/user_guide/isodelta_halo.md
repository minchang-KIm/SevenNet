<!--
IsoDelta-Halo researcher notes: this guide documents how to validate and
profile the communication metadata cache without changing model semantics.
-->

# IsoDelta-Halo Runtime Cache

IsoDelta-Halo is a runtime optimization for the parallel `e3gnn/parallel`
LAMMPS pair style. It reuses only communication metadata when the atom graph
topology is stable across MD steps:

- halo pack indexes
- halo unpack indexes
- extra ghost-node index mappings
- CUDA index tensors for the same metadata

It does not cache geometry values, edge vectors, embeddings, messages, energy,
or force. Those values are recomputed every step by the original SevenNet model
path.

## Why This Is Model-Agnostic

The method is placed below the MLIP model and above the LAMMPS brick
communication layer. Any distributed MLIP runtime that follows the common
pipeline below can use the same idea:

```text
positions -> neighbor list -> graph indexes -> halo exchange -> message passing
```

This makes the approach portable to SevenNet-like, NequIP-like, MACE-like, and
Allegro-like runtimes as long as their domain decomposition exposes stable halo
routing metadata between neighbor-list rebuilds.

## Runtime Controls

Set these environment variables before launching LAMMPS:

```bash
export SEVENN_PRINT_INFO=1
export SEVENN_ISODELTA_HALO_PROFILE=1
```

Disable the cache for a fair baseline run:

```bash
export SEVENN_ISODELTA_HALO_DISABLE=1
```

Leave `SEVENN_ISODELTA_HALO_DISABLE` unset for the enabled run.

## Lightweight Validation

Run the dependency-free validation suite before each commit:

```bash
python tools/run_isodelta_validation.py
```

The validation script checks:

- static cache invariants
- metadata-only reuse rules
- cache disable/profile controls
- benchmark log parsing
- Python syntax for validation tools
- whitespace errors through `git diff --check`

This does not replace a full LAMMPS/LibTorch build. It is a fast local guard so
the implementation does not drift while runtime environments are being prepared.

## Build Prerequisite Check

Before patching and compiling LAMMPS, check that the local checkout has the
expected SevenNet pair-style sources and that the LAMMPS source tree matches the
version used by the patch script:

```bash
python tools/check_isodelta_build_prereqs.py \
  --lammps-root /path/to/lammps \
  --require-torch
```

The checker prints JSON so failed checks can be archived with build logs. It
does not modify the LAMMPS tree.

## Paired Benchmark

Use the benchmark runner after building a LAMMPS binary that contains
`e3gnn/parallel`:

```bash
python tools/run_isodelta_lammps_benchmark.py \
  --lammps-command "mpiexec -n 4 lmp" \
  --input /path/to/in.sevenn \
  --repeat 5 \
  --output-dir isodelta_benchmark_runs
```

The runner executes each repeat twice:

- `baseline-disabled`: `SEVENN_ISODELTA_HALO_DISABLE=1`
- `isodelta-enabled`: cache enabled with profiling

LAMMPS runs in the input file directory by default, so relative model and data
paths inside the input script keep working. Use `--work-dir` when the benchmark
must run elsewhere.

## Report Fields

The runner writes `isodelta_benchmark_report.json`. Important fields are:

- `summary.cases.*.mean_loop_time_seconds`
- `summary.speedup_vs_disabled_cache`
- `summary.final_thermo_delta_vs_disabled_cache`
- `results.*.cache_summary.attempts`
- `results.*.cache_summary.hits`
- `results.*.cache_summary.hit_rate_percent`
- `results.*.final_thermo_observables`
- `results.*.cache_summary.miss_neighbor-list-rebuilt`
- `results.*.cache_summary.miss_shape-changed`
- `results.*.cache_summary.miss_tag-order-changed`

For a publishable performance claim, report the mean and variance across
multiple repeats, include cache hit rate, and show that final thermodynamic
scalars match the disabled-cache baseline within the tolerance required by the
simulation. A low hit rate usually means the neighbor list is rebuilt too often,
the graph shape changes often, or atom tag order is not stable enough for reuse.

## Expected Evidence For A Paper

A complete experiment should include:

- wall-clock loop time with cache disabled and enabled
- cache hit rate and miss reason breakdown
- MPI rank count, GPU count, atom count, cutoff, skin, neighbor rebuild period
- energy and force consistency against the disabled-cache baseline
- strong-scaling or weak-scaling curves when claiming parallel benefit

The strongest claim is not that SevenNet alone is faster. The stronger and more
general claim is that topology-stable MD steps contain reusable communication
metadata, and that avoiding repeated halo index construction reduces distributed
MLIP runtime overhead without changing model outputs.
