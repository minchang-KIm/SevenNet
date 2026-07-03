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

Before a cache hit is accepted, the runtime checks graph size, edge count, graph
tag order, neighbor-list age, and the current CommBrick communication topology
signature. The topology signature includes phase count, send/receive counts,
send/receive ranks, and first-receive offsets. The cache also compares the
phase-local sendlist tag order and the contiguous receive-segment tag order so
pass-through halo entries cannot be reused after atom-list reordering.

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

## Model-Agnostic Trace Check

Use `tools/check_isodelta_mlip_trace.py` when evaluating whether the same
runtime idea applies beyond the SevenNet baseline. The checker does not import
another MLIP implementation. Instead, it reads a JSON trace with the fields that
all cutoff-graph, halo-exchange MLIP runtimes can export:

- `model`: label such as `SevenNet`, `NequIP`, `MACE`, or `Allegro`
- `steps.*.neighbor_list_rebuilt`
- `steps.*.graph_node_tags`
- `steps.*.edge_count`
- `steps.*.comm_phases`
- `steps.*.comm_phases.*.send_rank`
- `steps.*.comm_phases.*.recv_rank`
- `steps.*.comm_phases.*.send_count`
- `steps.*.comm_phases.*.recv_count`
- `steps.*.comm_phases.*.first_recv`
- `steps.*.comm_phases.*.send_tags`
- `steps.*.comm_phases.*.recv_tags`
- optional `steps.*.step_time_seconds`
- optional `steps.*.metadata_build_time_seconds`

Print the expected trace schema before writing a new exporter:

```bash
python tools/check_isodelta_mlip_trace.py --print-schema
```

Run the checker on a trace from any distributed MLIP runtime:

```bash
python tools/check_isodelta_mlip_trace.py \
  --trace mace_halo_trace.json \
  --min-hit-rate-percent 50.0 \
  --min-estimated-speedup 1.05 \
  --min-metadata-fraction-percent 5.0 \
  --output mace_isodelta_trace_evidence.json
```

The output includes `hit_rate_percent`, a miss breakdown using the same reason
names as the C++ cache, `estimated_average_speedup`, and
`estimated_worst_case_speedup`. A trace passes only when graph node tags, edge
count, neighbor-list rebuild state, communication topology, and phase-local
send/receive tag order show that reuse would be safe. This lets a paper compare
SevenNet implementation results with NequIP/MACE/Allegro trace evidence without
claiming that another model's kernels were modified.
For precomputed trace evidence, the checker also verifies that `hit_rate_percent`
matches `hits / attempts` and that the miss breakdown counters sum to
`attempts - hits`. It treats `attempts`, `hits`, and each miss counter as
whole nonnegative reuse-decision counts. The evidence must also include
`status` as `evaluated` or `passed`, a non-empty `model` label, and
`model_agnostic_requirements` flags proving that ordered graph node tags, edge
count, neighbor-list rebuilds, communication topology, and communication list
tag order were all used as reuse guards.
Model labels are stripped of surrounding whitespace before required-model
matching, so the label archived in evidence is the normalized value used by the
bundle gate.
When timing fields are present, the checker recomputes
`metadata_fraction_percent` from metadata and baseline seconds, and recomputes
`estimated_average_speedup` and `estimated_worst_case_speedup` from baseline
and enabled seconds. This keeps portability evidence from becoming a standalone
claimed speedup without the timing basis needed to audit it.

### Portable Demo Traces

Before writing a new exporter, generate the dependency-free demo bundle:

```bash
python tools/run_isodelta_mlip_trace_demo.py \
  --output-dir isodelta_mlip_trace_demo
```

The demo writes trace and evidence JSON files for SevenNet, MACE, NequIP, and Allegro.
These examples are not a substitute for traces from real production runs. They
are a reference shape for exporter authors: each file uses the same
ordered `graph_node_tags`, `comm_phases`, send/receive tag arrays, timing
fields, and threshold gates that the publication checker expects. The summary
file `isodelta_mlip_trace_demo_summary.json` records per-model hit rate,
`estimated_average_speedup`, and `estimated_worst_case_speedup`.

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
The same dependency-free gate runs as the `IsoDelta-Halo lightweight validation`
GitHub Actions workflow in `.github/workflows/isodelta-halo.yml`
whenever IsoDelta-Halo sources, validation tools, tests, or this guide change.

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

The patch script copies `pair_e3gnn_oeq_autograd.cpp` together with the serial
and parallel pair styles. Keep that bridge present even for non-oEq builds,
because the pair styles reference its registration symbol and the no-op path
keeps the link step reproducible.

## Binary Smoke Check

After building LAMMPS, verify that the patched pair style is registered before
running long simulations:

```bash
python tools/check_isodelta_lammps_binary.py \
  --lammps-command "mpiexec -n 1 lmp"
```

This command runs LAMMPS help output and checks for `e3gnn/parallel`. It is a
fast registration check, not a numerical correctness test.
The smoke checker rejects an empty `--lammps-command`, and
`--timeout-seconds` must be positive so a failed launch cannot masquerade as a
valid registration check.

## End-to-End Experiment Driver

For paper experiments, run the full gate as one command after building LAMMPS:

```bash
python tools/run_isodelta_experiment.py \
  --lammps-command "mpiexec -n 4 lmp" \
  --input /path/to/in.sevenn \
  --lammps-root /path/to/lammps \
  --require-torch \
  --repeat 5 \
  --benchmark-timeout-seconds 3600 \
  --output-dir isodelta_experiment_runs \
  --max-abs-thermo-delta 1.0e-8 \
  --min-paired-thermo-count 5 \
  --min-speedup 1.05 \
  --min-hit-rate-percent 50.0 \
  --min-enabled-cache-attempts 5 \
  --min-enabled-cache-hits 1
```

The driver runs the prerequisite checker, binary smoke check, paired benchmark,
and report correctness gate in that order. It writes stage logs and
`isodelta_experiment_report.json`; archive that report with the benchmark JSON
when using the numbers in a manuscript.
The driver rejects empty or impossible evidence settings before launching
external commands. `--repeat` must be at least 1.
`--binary-timeout-seconds` must be positive.
`--benchmark-timeout-seconds` must be positive. Percentage gates must stay
between 0 and 100, and minimum cache hits cannot exceed minimum cache attempts.
Both the experiment report and benchmark report include a `provenance` object
with schema version, Git commit, Git branch, dirty-worktree state, Python
runtime, platform string, and benchmark environment overrides. Treat a dirty
worktree as a signal to archive the exact diff beside the raw logs.
The report checker requires `provenance.report_schema_version`,
`provenance.git_commit`, `provenance.git_branch`, `provenance.git_dirty`,
runtime fields, and case environment overrides before accepting benchmark
evidence.

To append the publication evidence bundle gate to the same driver run, add one
or more trace evidence files:

```bash
python tools/run_isodelta_experiment.py \
  --lammps-command "mpiexec -n 4 lmp" \
  --input /path/to/in.sevenn \
  --trace-evidence mace_isodelta_trace_evidence.json \
  --require-trace-model MACE \
  --min-trace-hit-rate-percent 50.0 \
  --min-trace-estimated-speedup 1.05 \
  --min-trace-metadata-fraction-percent 5.0
```

When trace evidence is provided, the driver adds an `evidence-bundle` stage and
writes `bundle_evidence.json` under the experiment output directory.
Trace-specific gates such as `--min-distinct-trace-models`,
`--min-trace-hit-rate-percent`, and `--min-trace-estimated-speedup` also request
that bundle stage. The driver rejects those settings before launching LAMMPS if
no trace evidence files are supplied, if a trace evidence path is duplicated, or
if a required trace model label is empty or duplicated.

## 8-GPU Cluster Paper Suite

Use `tools/run_isodelta_cluster_paper_suite.py` when the paper experiment needs
one command to cover an 8-GPU cluster run, artifact downloads, multi-model
evidence collection, tables, correlations, and figures. The suite reads a
TOML manifest instead of hard-coding dataset URLs or MACE/NequIP launch syntax.
That is intentional: SevenNet, MACE, and NequIP foundation-model runs often use
different checkpoint formats, data loaders, and launchers, while the paper
still needs one auditable result bundle.

Start by writing a commented template:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --write-template isodelta_cluster_suite.toml
```

Fill in the real dataset and checkpoint artifacts, command lines, and trace
paths. Artifact entries may use `https://` or `file://` URLs, and the runner
checks SHA-256 digests when a `sha256` value is present. The default manifest
gate requires cases for `SevenNet`, `MACE`, and `NequIP`, so a portability
experiment cannot accidentally omit one model family. Artifact `required_by`
entries are also checked against the manifest's case model names, which catches
misspelled model labels before a cluster job starts.

Run the complete suite on the cluster:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml
```

The runner checks the visible GPU count against `expected_gpus = 8`, downloads
missing artifacts, prints terminal progress as `[suite] [stage/total] ...`,
then executes each case. A `sevennet_lammps` case calls
`run_isodelta_experiment.py` and therefore runs the disabled/enabled LAMMPS
benchmark plus report gates. An `external_pair` case is for MACE, NequIP, or
another runtime whose disabled and enabled commands are supplied in the
manifest. A `trace_only` case validates portable MLIP trace evidence when a
model has applicability evidence but no paired runtime benchmark yet.

After successful collection, the suite writes:

- `isodelta_cluster_paper_summary.json`
- `tables/case_summary.csv`
- `tables/case_summary.md`
- `tables/correlation.csv`
- `figures/speedup_by_case.svg`
- `figures/hit_rate_vs_speedup.svg`
- `figures/trace_metadata_fraction_vs_speedup.svg`

Use `--collect-only` to regenerate tables, correlations, and figures from
existing benchmark reports, external timing reports, and trace evidence without
rerunning the cluster jobs. Use `--skip-gpu-check` only for local dry runs or
CI tests; for paper runs, keep the GPU check enabled and archive the summary
JSON with the raw logs. The suite-level evidence gate recounts all passed cases
before writing the summary: `min_trace_count` must be satisfied by distinct
trace evidence files, and `min_distinct_trace_models` must be satisfied by the
`model` labels inside those trace files. This prevents a three-model claim from
passing with duplicated MACE evidence mislabeled in the manifest.

## Paired Benchmark

Use the benchmark runner after building a LAMMPS binary that contains
`e3gnn/parallel`:

```bash
python tools/run_isodelta_lammps_benchmark.py \
  --lammps-command "mpiexec -n 4 lmp" \
  --input /path/to/in.sevenn \
  --repeat 5 \
  --run-timeout-seconds 3600 \
  --output-dir isodelta_benchmark_runs
```

The runner executes each repeat twice:

- `baseline-disabled`: `SEVENN_PRINT_INFO=1`,
  `SEVENN_ISODELTA_HALO_DISABLE=1`, and
  `SEVENN_ISODELTA_HALO_PROFILE=1`
- `isodelta-enabled`: `SEVENN_PRINT_INFO=1` and
  `SEVENN_ISODELTA_HALO_PROFILE=1`, with no disable flag

LAMMPS runs in the input file directory by default, so relative model and data
paths inside the input script keep working. Use `--work-dir` when the benchmark
must run elsewhere.
The runner rejects `--repeat` values below 1 because a zero-repeat report has
no paired timing, cache-hit, or final-thermo evidence to audit.
It rejects an empty `--lammps-command` before launching any external process.
It also applies `--run-timeout-seconds` to each individual LAMMPS invocation so
a hung MPI launch becomes a failed run with raw stdout/stderr logs instead of an
unbounded experiment.

## Report Fields

The runner writes `isodelta_benchmark_report.json`. Important fields are:

- `provenance.report_schema_version`
- `provenance.git_commit`
- `provenance.git_dirty`
- `provenance.case_environment_overrides`
- `run_timeout_seconds`
- `summary.runs`
- `summary.cases.*.mean_loop_time_seconds`
- `summary.cases.*.sample_variance_loop_time_seconds`
- `summary.cases.*.sample_stddev_loop_time_seconds`
- `summary.cases.*.min_loop_time_seconds`
- `summary.cases.*.max_loop_time_seconds`
- `summary.speedup_vs_disabled_cache`
- `summary.final_thermo_delta_vs_disabled_cache`
- `results.*.repeat_index`
- `results.*.loop_time_seconds`
- `results.*.cache_summary.attempts`
- `results.*.cache_summary.hits`
- `results.*.cache_summary.hit_rate_percent`
- `results.*.cache_summary.summary_rank_count`
- `results.*.final_thermo_observables`
- `results.*.cache_summary.miss_disabled`
- `results.*.cache_summary.miss_no-cache`
- `results.*.cache_summary.miss_neighbor-list-rebuilt`
- `results.*.cache_summary.miss_shape-changed`
- `results.*.cache_summary.miss_index-tensor-shape-changed`
- `results.*.cache_summary.miss_tag-count-changed`
- `results.*.cache_summary.miss_tag-order-changed`
- `results.*.cache_summary.miss_comm-topology-changed`
- `results.*.cache_summary.miss_comm-list-tag-order-changed`

## Report Correctness Gate

After a paired benchmark finishes, validate the JSON report before using the
numbers in a paper table:

```bash
python tools/check_isodelta_benchmark_report.py \
  --report isodelta_benchmark_runs/isodelta_benchmark_report.json \
  --max-abs-thermo-delta 1.0e-8 \
  --min-paired-thermo-count 5 \
  --min-speedup 1.05 \
  --min-hit-rate-percent 50.0 \
  --min-enabled-cache-attempts 5 \
  --min-enabled-cache-hits 1
```

`--max-abs-thermo-delta` should match the precision and observable scale of the
target simulation. `--min-speedup`, `--min-hit-rate-percent`,
`--min-enabled-cache-attempts`, and `--min-enabled-cache-hits` are effect gates:
use them when making a performance claim, and archive the command with the
benchmark report so the acceptance rule is reproducible.
The checker also requires every enabled run to include all IsoDelta-Halo miss
reason counters, which keeps failed reuse diagnosable instead of reducing the
experiment to a single speedup number.
The report checker validates `run_timeout_seconds` as a positive finite value,
so accepted benchmark evidence proves that each LAMMPS invocation was bounded.
It also checks `provenance.case_environment_overrides` to prove that the
`baseline-disabled` case recorded `SEVENN_PRINT_INFO=1`,
`SEVENN_ISODELTA_HALO_DISABLE=1`, and `SEVENN_ISODELTA_HALO_PROFILE=1`, while
the `isodelta-enabled` case recorded `SEVENN_PRINT_INFO=1` and
`SEVENN_ISODELTA_HALO_PROFILE=1` with no disable flag.
For the `baseline-disabled` cache summary, the checker requires zero hits,
zero hit rate, and `miss_disabled` equal to `attempts`; this proves the baseline
run did not accidentally reuse IsoDelta-Halo metadata.
It also validates that every `repeat_index` has exactly one `baseline-disabled`
run and one `isodelta-enabled` run, then reports `paired_repeat_count` in the
evidence summary.
For thermo deltas, `max_abs_delta` must be nonnegative and `paired_count` must
be a whole positive count that does not exceed `paired_repeat_count`.
For timing evidence, the checker recomputes case mean loop times and
min/max and sample variance/stddev from `results.*.loop_time_seconds`, then
rejects reports whose `summary.speedup_vs_disabled_cache` does not match those
means.
It also requires `summary.runs` to match the number of `results` rows.
When multiple MPI ranks print profiling summaries, the benchmark parser sums
rank-local attempts, hits, and miss counters, then recomputes
`hit_rate_percent` from the aggregated hits and attempts. The report checker
requires `summary_rank_count` to be a positive integer and rejects cache
summaries whose hit rate is inconsistent with the aggregated totals.
The parser accepts ordinary decimal values and scientific notation in loop-time
and IsoDelta-Halo summary lines, so very short smoke runs and large counters do
not lose timing or cache evidence.
It also rejects reports where the required miss reason counters do not sum to
`attempts - hits`, so a miss breakdown table cannot silently drift away from
the measured cache activity. The checker treats `attempts`, `hits`, and every
miss reason counter as whole nonnegative profiling counts, and each cache
summary must record at least one attempt.
On CUDA-aware MPI runs, the runtime also checks that cached index tensors still
match their cached CPU index-vector lengths before reuse; a mismatch is treated
as `miss_index-tensor-shape-changed` and the metadata cache is rebuilt.

For a publishable performance claim, report the mean and variance across
multiple repeats, include cache hit rate, and show that final thermodynamic
scalars match the disabled-cache baseline within the tolerance required by the
simulation. A low hit rate usually means the neighbor list is rebuilt too often,
the graph shape changes often, or atom tag order is not stable enough for reuse.

## Evidence Bundle Gate

For a manuscript claim that combines the implemented SevenNet/LAMMPS speedup
with model-agnostic MLIP portability evidence, validate the benchmark report and
trace evidence files together:

```bash
python tools/check_isodelta_evidence_bundle.py \
  --benchmark-report isodelta_benchmark_runs/isodelta_benchmark_report.json \
  --trace-evidence mace_isodelta_trace_evidence.json \
  --trace-evidence nequip_isodelta_trace_evidence.json \
  --require-trace-model MACE \
  --require-trace-model NequIP \
  --min-distinct-trace-models 2 \
  --max-abs-thermo-delta 1.0e-8 \
  --min-paired-thermo-count 5 \
  --min-speedup 1.05 \
  --min-hit-rate-percent 50.0 \
  --min-enabled-cache-attempts 5 \
  --min-enabled-cache-hits 1 \
  --min-trace-hit-rate-percent 50.0 \
  --min-trace-estimated-speedup 1.05 \
  --min-trace-metadata-fraction-percent 5.0 \
  --output bundle_evidence.json
```

Use one `--trace-evidence` argument per model trace and one
`--require-trace-model` argument for every model label that must appear in the
claim. The bundle checker rejects duplicate trace evidence paths and duplicate
or empty required model labels, so `min_trace_count` cannot be satisfied by
reusing the same artifact. Set `--min-distinct-trace-models` above one when the
claim needs multi-model portability evidence; this separate gate rejects two
trace files from the same MLIP label as insufficient for a cross-model claim.
The resulting `bundle_evidence.json` records the path, SHA-256 digest, and byte
size for the benchmark report and every trace evidence file. It also includes
`bundle_schema_version` and a `provenance` object with Git, Python, and platform
metadata for the bundle checker run, so archive it beside raw LAMMPS logs,
trace JSON, and plotting scripts.

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
