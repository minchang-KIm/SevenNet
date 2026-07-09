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
During graph construction the runtime keeps tag lookup, graph-index, edge-index,
and edge-vector work buffers in heap-backed `std::vector` storage instead of
runtime-sized stack arrays, which keeps large 8-GPU paper jobs from depending on
compiler-specific variable-length arrays or small per-rank stack limits. LAMMPS
neighbor-list special bits are stripped with `NEIGHMASK` before the runtime reads
atom tags or types, so graph construction and the cached halo guards see the same
canonical atom index.
Graph-node capacity, local and ghost atom-array indexes, neighbor-count-derived
edge capacity, and every edge write index are checked before graph buffers are
written.
LAMMPS atom types are mapped to SevenNet species only after the type map is
initialized, the atom type is confirmed to be in range, and the mapped model
type is confirmed to exist.
The atom-tag lookup table size and every local or ghost atom-tag index are also
checked before the graph builder writes `tag_to_graph_idx`.
The communication preprocessing path reuses the same checked atom-tag helper
before reading through the active `tag_to_graph_idx` pointer for pack and unpack
index maps.
The per-step graph-index pointer used by `CommBrick` pack/unpack initialization
is reset after each compute step. If that communication path is entered without
an active graph-index map, the runtime aborts with a named IsoDelta-Halo error
instead of dereferencing an inactive lookup.
Every pair-style communication helper also validates the `comm_phase` argument
before indexing the six-phase communication arrays, which turns an invalid
phase route into a named runtime error instead of silent metadata corruption.
The communication preprocessing init path also checks pack/unpack counts,
nonempty send-list pointers, and atom-array ranges before reserving vectors or
reading atom tags, so a malformed halo setup cannot turn into an oversized
allocation or an out-of-bounds tag lookup.
Extra graph indexes, the trash slot, and the extra communication tensor size
are computed through checked integer helpers, so large halo maps fail with a
named error instead of silently wrapping an index or allocating the wrong tensor.
The node feature tensor produced by each model stage is checked for rank and
feature width before `x_dim` is updated, so malformed model outputs cannot feed
an invalid hidden-state width into halo buffer sizing.
Cache tag-signature reuse and storage also validate graph-index capacity,
graph-to-atom indexes, sendlist atom ids, and contiguous receive ranges before
reading `atom->tag`.
The `CommBrick` read-only topology accessors used by the cache apply the same
phase guard and also check sendlist indexes before returning atom ids.
For the actual halo payload, the e3gnn communication path allocates host and
CUDA/MPI float buffers by `feature_width * atom_capacity`, not by the scalar
LAMMPS atom buffer length, so wide hidden-state tensors have explicit capacity.
CUDA buffer element counts, byte counts, buffer releases, allocations, device
selection, and device-to-device copy failures are also checked and reported
through named LAMMPS errors instead of being ignored by the halo path.
The pair pack/unpack helpers compute payload element and byte counts through a
checked helper before MPI or CUDA copy calls, so `x_dim * atom_count` overflow
cannot silently truncate a communication message.
The `pair_coeff` parser checks the minimum argument count, positive model-file
count, model path existence, explicit model-file argument range, nonempty
species mapping, positive finite `cutoff`, and positive integer `comm_size`
before loading TorchScript modules. Invalid deployment input therefore fails as
a named LAMMPS error instead of escaping as a C++ parsing exception or an empty
model list.
The deployed `chemical_symbols_to_index` string is tokenized without mutating
the metadata buffer, then cross-checked against the deployed `num_species`
field and the number of LAMMPS atom types supplied to `pair_coeff`.
The deployment script writes that species list as a single space-delimited
string without leading or trailing whitespace, so serial and parallel
TorchScript artifacts expose the same metadata contract.
Cached CUDA index tensors are reused only when their length, integer dtype, and
device still match the current communication path, preventing a stale tensor
layout from crossing a later `index_select` or `scatter_` call.

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

The output includes a top-level `report_comment`, `hit_rate_percent`, a miss
breakdown using the same reason names as the C++ cache,
`estimated_average_speedup`, and `estimated_worst_case_speedup`. A trace passes
only when graph node tags, edge count, neighbor-list rebuild state,
communication topology, and phase-local send/receive tag order show that reuse
would be safe. This lets a paper compare SevenNet implementation results with
NequIP/MACE/Allegro trace evidence without claiming that another model's
kernels were modified.
For precomputed trace evidence, the checker also verifies that `hit_rate_percent`
matches `hits / attempts` and that the miss breakdown counters sum to
`attempts - hits`. It treats `attempts`, `hits`, and each miss counter as
whole nonnegative reuse-decision counts. The evidence must also include
the expected trace evidence `report_comment`, `status` as `evaluated` or `passed`, a non-empty `model` label, and
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
`estimated_average_speedup`, and `estimated_worst_case_speedup`. Demo raw trace files carry `artifact_comment`,
while validated trace evidence and the demo summary carry `report_comment` so
generated portability artifacts remain self-describing outside the repository.

## Runtime Controls

Set these environment variables before launching LAMMPS:

```bash
export SEVENN_PRINT_INFO=1
export SEVENN_ISODELTA_HALO_PROFILE=1
```

Both IsoDelta-Halo flags use boolean-style parsing. Unset, empty, `0`,
`false`, `no`, and `off` values mean off, case-insensitively. Surrounding
whitespace is ignored before parsing; any other set value means on. For
example, `SEVENN_ISODELTA_HALO_DISABLE=0` or
`SEVENN_ISODELTA_HALO_DISABLE=" off "` keeps the cache enabled, and
`SEVENN_ISODELTA_HALO_PROFILE=0` keeps profiling off.

Disable the cache for a fair baseline run:

```bash
export SEVENN_ISODELTA_HALO_DISABLE=1
```

Leave `SEVENN_ISODELTA_HALO_DISABLE` unset for the enabled run.

## Lightweight Validation

Run the dependency-free validation suite before each commit:

```bash
python tools/run_isodelta_validation.py \
  --report-path isodelta_validation_report.json
```

The validation script checks:

- static cache invariants
- metadata-only reuse rules
- cache disable/profile controls
- benchmark log parsing
- Python syntax for validation tools
- whitespace errors through `git diff --check`

With `--report-path`, the runner writes `isodelta_validation_report.json` with
the schema version, `report_comment`, Git branch/commit/status metadata, every
validation command, return code, elapsed time, and bounded stdout/stderr tails.
Keep that report with the sync attempt when validating a commit before pushing.

To make the validation-and-push sequence itself auditable, run the sync gate
after committing:

```bash
python tools/run_isodelta_sync_gate.py \
  --remote fork \
  --branch codex/isodelta-halo-runtime \
  --report-path isodelta_sync_report.json \
  --validation-report-path isodelta_validation_report.json
```

The sync gate runs `run_isodelta_validation.py` first and only attempts
non-interactive `git push -u` when validation passes. It writes
`isodelta_sync_report.json` with the validation command record, push command
record, `git ls-remote --heads` remote-ref verification record, Git status,
target remote/branch, the push stderr tail, and a
`validation_report_fingerprint` object with the validation report SHA-256
digest and byte size. The sync report also carries its own `report_comment`,
and `validation_report_summary` records the validation report `report_comment`
so reviewers can see that both generated JSON files describe their evidence purpose.
If the validation command exits successfully but the
validation report file is missing, the sync gate reports
`validation_report_missing` and does not push. It also parses the validation
report JSON into `validation_report_summary`; if that summary does not show
the expected schema version, required validation command fields,
`status = "passed"` for the same expected branch and current HEAD commit, and
zero failed validation command return codes, the gate reports
`validation_report_invalid` and does not push. The required command fields
must use valid types and values. The report also stores `sync_command_summary`,
which counts sync command records, failed return codes, missing fields, and
invalid field values for fast audit. If the target branch cannot be determined,
the report records a replayable `push_branch_precondition` command instead of
an empty push command. The report also stores `worktree_status`, a parsed
`git status --short` snapshot with entry count,
per-entry index/worktree status, paths, and a clean flag. For final paper
syncs, add `--require-clean-worktree` to fail with `dirty_worktree` before
validation or push when tracked or untracked files are present. It also stores
`git_provenance` fields for the current branch, validated HEAD commit, local
target branch commit, remote URL, and locally known remote-tracking commit. It
also writes `remote_ref_verification` and only reports `synced` when the remote
branch resolves to the same commit that was pushed. A mismatch is reported as
`remote_verification_failed`. The report writes a `push_failure` object when
the push fails. That object classifies common failures as `auth-prompt-disabled`,
`network-unreachable`, or `unknown`, and adds a `suggested_action` so a
non-interactive GitHub credential failure is actionable from the report itself.
This makes credential or network failures explicit instead of losing the
evidence after a failed sync. It also forwards the same target branch to the
goal-readiness audit as `--expected-branch`, so the gate must validate one checkout branch before pushing that same branch and cannot mix targets by mistake.
When the cluster login node cannot authenticate to GitHub, add
the `--push-failure-bundle` option with a path such as
`isodelta_push_failure.bundle`; after validation passes and push fails, the gate
writes a portable `git bundle` for the validated branch and records the bundle
creation command, `git bundle verify` `verify_returncode`, SHA-256 digest,
and byte size under `push_failure_bundle`.

For a local completion-readiness audit, generate
`isodelta_goal_readiness_report.json` as a source-tree report:

```bash
python tools/check_isodelta_goal_readiness.py \
  --expected-branch codex/isodelta-halo-runtime \
  --report-path isodelta_goal_readiness_report.json
```

This audit checks that the runtime cache files, cluster paper-suite script,
validation runner, sync gate, CI workflow, tests, and this guide all contain the
required IsoDelta-Halo feature markers and explanatory file headers. It also
scans every `tools/*isodelta*.py` script and
`tests/unit_tests/test_isodelta*.py` test so a newly generated IsoDelta helper
cannot enter the workflow without a file-level comment. It records the current
Git branch, commit, and status but does not treat unrelated local workspace
files as proof that the implementation itself is missing.

This does not replace a full LAMMPS/LibTorch build. It is a fast local guard so
the implementation does not drift while runtime environments are being prepared.
The same dependency-free gate runs as the `IsoDelta-Halo lightweight validation`
GitHub Actions workflow in `.github/workflows/isodelta-halo.yml`
whenever IsoDelta-Halo sources, validation tools, tests, or this guide change,
and the workflow uploads the same JSON validation report as an artifact.

## Build Prerequisite Check

Before patching and compiling LAMMPS, check that the local checkout has the
expected SevenNet pair-style sources and that the LAMMPS source tree matches the
version used by the patch script:

```bash
python tools/check_isodelta_build_prereqs.py \
  --lammps-root /path/to/lammps \
  --require-torch
```

The checker prints JSON so failed checks can be archived with build logs. That
JSON includes top-level `report_schema_version` and `report_comment` fields
identifying it as prerequisite evidence for source-file, LAMMPS tree, version,
and optional torch checks. It does not modify the LAMMPS tree.

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
Its JSON output includes top-level `report_schema_version` and `report_comment`
fields so the archived stdout log remains recognizable as binary-registration
evidence.
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
The experiment report carries a top-level `report_comment` that identifies it
as the driver-level audit record for launched commands, paths, return codes,
and run provenance.
Each command record also carries `stdout_fingerprint` and `stderr_fingerprint`
objects with SHA-256 digests and byte sizes for the generated log files, so a
log edit after the run is visible from the archived experiment report.
After moving or archiving the run directory, re-open the report and logs with:

```bash
python tools/check_isodelta_experiment_report.py \
  --report isodelta_experiment_runs/isodelta_experiment_report.json \
  --output isodelta_experiment_runs/experiment_report_check.json
```

The checker exits nonzero if the driver report comment, provenance schema, or
any command log fingerprint no longer matches the files on disk. Its optional
`--output` JSON carries its own `report_comment`, schema version, status, and
checked log count so the verification step can be archived beside the run.
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

For quick ablation timing, add `--ablation-mode baseline-disabled` or
`--ablation-mode isodelta-enabled` to the experiment driver. The default
`--ablation-mode paired` keeps the publishable disabled/enabled gate. A
one-sided ablation run launches only the requested benchmark case and writes the
raw benchmark JSON, but it intentionally skips the paired report and evidence
bundle gates because speedup and final-thermo deltas require both modes.

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
For final paper runs, keep `require_artifact_sha256 = true` in the generated
template. With that gate enabled, every required dataset, checkpoint, input
deck, or runtime bundle must carry a full 64-character SHA-256 digest before the
suite will launch, which prevents a result table from being built from mutable
or placeholder inputs.
Each case may also define a `preflight_command` for short import or module
checks:

```toml
preflight_command = 'python -c "import mace"'
```

The suite runs that command before trace generation or paired timing loops,
captures `preflight.stdout.log` and `preflight.stderr.log`, and stops the case
early if the environment is not ready. Use `preflight_env` and
`preflight_timeout_seconds` when a model family needs module, container, or
license-server variables that differ from the timed disabled/enabled commands.
For `external_pair` MACE/NequIP-style cases, keep the disabled and enabled
modes auditable with `disabled_env` and `enabled_env`. The runner always starts
the disabled command with `SEVENN_ISODELTA_HALO_DISABLE` set and the enabled
command with that variable unset, then records those external_pair mode
controls in the run plan, generated timing report, and summary
`case_mode_controls`. If `enabled_env` sets `SEVENN_ISODELTA_HALO_DISABLE` to a
truthy value, the manifest is rejected because the "applied" run would actually
be cache-off too. Explicit false values such as an empty string, `0`, `false`,
`no`, and `off` are treated as cache-enabled evidence, matching the C++
runtime parser. The recorded mode-control JSON also carries
`env_flag_false_values`, the false-value parser list used for that decision, so
an archived report can be interpreted without reopening the source code.
These external_pair mode controls are part of the paper audit trail, not only a
runtime convenience.
For `sevennet_lammps` and `external_pair` cases, the manifest can set
`ablation_mode = "paired"`, `"baseline-disabled"`, or `"isodelta-enabled"`.
Keep `ablation_mode = "paired"` for final paper timing because only paired runs
produce speedup and final-thermo delta evidence. A one-sided SevenNet or
external_pair ablation is useful for quick smoke timing inside the cluster
suite, but the readiness gate treats it as ablation-only evidence rather than a
publishable disabled/enabled comparison.
For one-sided `external_pair` ablation manifests, only the command side named by
`ablation_mode` is required: `baseline-disabled` needs `disabled_command`, and
`isodelta-enabled` needs `enabled_command`. Paired `external_pair` timing still
requires both commands because it is the only mode that can support a speedup
claim.
For temporary sweeps, keep the TOML manifest at `ablation_mode = "paired"` and
use `--ablation-mode-override` to pass
`--ablation-mode-override baseline-disabled` or
`--ablation-mode-override isodelta-enabled` at execution time. The override is a
runtime convenience for `sevennet_lammps` and `external_pair` cases; `trace_only`
cases remain unchanged because they do not launch disabled/enabled timing
commands. The generated readiness report, artifact-preparation report, plan,
preflight report, pipeline report, and final summary keep the original
manifest fingerprint and also record the effective CLI change in
`suite.runtime_overrides`, so reviewers can distinguish the archived TOML from
the actual ablation mode used for that run.
Required artifacts must already exist or be downloadable; if `--skip-downloads`
is active and a required artifact is missing, the suite fails before launching
any case. Optional artifacts with `required = false` may be absent, but the
download record marks them as skipped rather than silently treating them as
present.

Before reserving GPUs, prepare the immutable inputs on a login or CPU node:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --prepare-artifacts
```

The `--prepare-artifacts` mode downloads missing datasets, checkpoints, input decks, and runtime bundles, verifies every declared SHA-256 digest, and writes `artifact_preparation_report.json` under the suite output directory. Each download record includes `download_progress` with total bytes, written bytes, percent, and completion status; the terminal also prints byte-level download progress for long transfers. It does not probe GPUs or launch SevenNet/MACE/NequIP cases, so failed URLs or checksum mismatches are caught before the 8-GPU allocation starts.

Before occupying a long allocation, run the cluster preflight gate:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --preflight-only \
  --preflight-output preflight_report.json
```

The `--preflight-only` mode verifies the manifest schema, probes the requested
GPU count unless `--skip-gpu-check` is set, downloads or verifies declared
artifacts, runs each case's `preflight_command`, writes command stdout/stderr
logs, fingerprints those logs, and stores package/GPU environment details in
`preflight_environment_snapshot.json`. It exits before any LAMMPS, MACE, or
NequIP timing loop starts, so import failures, missing modules, broken URLs,
wrong SHA-256 digests, and unavailable GPUs are visible in
`preflight_report.json` while the cluster job is still cheap to rerun.

For a one-command final-paper path, run the full pipeline:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --pipeline \
  --pipeline-report pipeline_report.json
```

The `--pipeline` mode executes the final-paper readiness gate, prepares
artifacts, runs the cluster preflight gate, writes the plan JSON, executes the
full suite, and then verifies the output bundle fingerprints. Its
`pipeline_report.json` records every stage, stage report path, selected modes,
manifest fingerprint, suite artifact SHA gate, stage report fingerprints,
final bundle verification counts, and final status. If the readiness gate fails, the pipeline writes
`readiness_report.json` and stops before downloads, preflight commands, timing runs, or summary generation. Use `--reuse-passed` with
`--pipeline` after an interrupted run to reuse already validated case outputs
while still rerunning readiness, preflight, summary generation, and bundle
verification.
If the preflight gate fails after artifact preparation, the pipeline keeps
`preflight_report.json` and `artifact_preparation_report.json`, marks the
pipeline failed, and stops before plan, timing runs, or summary generation.
If output bundle verification fails after the suite writes tables and figures,
the pipeline records the fingerprint mismatch in `pipeline_report.json` and
does not report the run as passed.
After archiving a pipeline run, verify the top-level pipeline evidence with
`--verify-pipeline-report`:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --verify-pipeline-report pipeline_report.json
```

This rechecks the pipeline report schema, every recorded stage report
fingerprint, re-runs the current output-bundle verifier, and compares every
embedded final output-bundle `summary_json` path and verification count, including
SLURM Python provenance and experiment-report-check evidence counts. It is a
publication-success gate: a failed or planned `pipeline_report.json` exits
nonzero even when its diagnostic stage fingerprints are internally consistent.
It also requires the full final-paper stage sequence, non-skipped stage
fingerprints, and a passed `output_bundle_verification` object, so changing only
the top-level status field cannot turn an incomplete run into publication
evidence. Non-skipped stage fingerprints must point to present report files;
an absent-file fingerprint is accepted only for a skipped artifact-preparation
stage. Each `stages[*].report_path` entry must also match the paired
`stage_report_fingerprints[*].report.path`, so the human-facing stage index
cannot drift away from the protected fingerprint record. The `run_suite` and
`verify_output_bundle` stage reports must both resolve to the verified output bundle
summary, so a copied or stale summary cannot masquerade as the final pipeline
result. The summary's suite metadata must also match the pipeline suite metadata,
including manifest SHA-256, output directory, requested GPU count,
SevenNet/MACE/NequIP model scope, runtime overrides, and the artifact SHA gate.
The pipeline suite record itself stores `require_artifact_sha256`, and the
verifier requires the artifact preparation report, run plan, and final summary
to keep the same value.
The verifier also
checks pipeline `modes` as booleans, rejects
`dry_run = true`, `skip_gpu_check = true`, or `allow_gpu_mismatch = true` for a
passed report, and requires suite metadata to keep `expected_gpus >= 8`, the
SevenNet/MACE/NequIP required-model scope, and a well-formed manifest fingerprint.
Runtime overrides are limited to known keys, and a passed publication report may
only record `ablation_mode = "paired"`; a one-sided ablation override must stay
in the ablation-only path instead of being accepted as final paper evidence. It
also reopens the fingerprinted readiness report and requires every final-paper readiness check to be present and passed.
It reopens the fingerprinted artifact preparation report, requires `dry_run = false`,
and checks `missing_required_artifacts` plus each required artifact SHA-256 digest.
It also reopens the fingerprinted run plan report, requires final-paper execution modes,
checks `gpu_check_planned`, verifies required model case plans, and compares
`run_plan.paper_outputs` against the summary artifact index. That comparison
includes the summary JSON path, case/correlation tables, command/repeat timing
tables, uncertainty figures, and the manifest snapshot, so a run plan cannot
promise one publication artifact while the verified bundle archives another.
It then reopens the fingerprinted preflight report and checks `gpu_check` so the
archive proves the requested GPU count was actually detected.

On a SLURM cluster, generate a commented submission script from the same
manifest:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --write-slurm-script run_isodelta_cluster_suite.sbatch \
  --slurm-repo-root /scratch/icpp/SevenNet-main \
  --slurm-manifest-path /scratch/icpp/SevenNet-main/isodelta_cluster_suite.toml \
  --slurm-output-dir /scratch/icpp/isodelta_cluster_paper_runs
```

The generated `sbatch` file requests `expected_gpus`, writes scheduler logs
under `slurm_logs/`, runs `--preflight-only`, creates the preflight plan with
the exact same `COMMON_ARGS` used by the final run, and then executes the full
`--pipeline` path with `--pipeline-report`. Before preflight it writes
`python_runtime_provenance.txt` into the output bundle with a comment header,
recording `PYTHON_BIN`, `SUITE_RUNNER`, `python --version`, and `sys.executable` so archived runs show
which interpreter actually executed the experiment. The launcher then requires
that file to be non-empty and passes marker checks for each recorded field before
starting preflight, and runs the same checks again after the final verification command.
After the pipeline exits, the
launcher immediately runs `--verify-pipeline-report "$PIPELINE_OUTPUT"` so the
SLURM job only succeeds after stage fingerprints and final bundle verification
counts are rechecked. The
script records the repository root in `REPO_ROOT`, using `--slurm-repo-root` as
the embedded cluster-checkout default. `--slurm-manifest-path` and
`--slurm-output-dir` do the same for the manifest and output directory, so a
launcher generated on a login node or copied from another machine does not
accidentally preserve that machine's absolute paths. The script changes into `REPO_ROOT` before launching Python and defines `PYTHON_BIN` and `SUITE_RUNNER`
as overridable shell variables so cluster module systems can select the intended
environment without editing the recorded experiment command. `--collect-only` is not accepted when generating
this SLURM launcher because the launcher is reserved for the full 8-GPU
pipeline; use the direct `--collect-only` command after jobs finish.
The generated header also includes a `CLI runtime overrides` comment, so the
standalone `sbatch` file records whether it used the manifest as-is or applied
an execution-time ablation override.
Before submitting or archiving the file, verify the launcher itself with
`--verify-slurm-script`:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --verify-slurm-script run_isodelta_cluster_suite.sbatch
```

This checks the `#SBATCH` GPU request, strict Bash mode, `REPO_ROOT`,
`MANIFEST_PATH`, `ISODELTA_OUTPUT_DIR`, Python provenance capture, shared
`COMMON_ARGS`, preflight/plan stages, and the final `--verify-pipeline-report` or `--verify-output-bundle` gate.
If the launcher is generated with `--ablation-mode-override baseline-disabled`
or `--ablation-mode-override isodelta-enabled`, it runs the one-sided ablation suite
and reopens the result with `--verify-output-bundle` instead of using the
final-paper `--pipeline` path. That keeps quick ablation timing usable on the
cluster without weakening the paired readiness gate used for paper numbers.
For ablation batches, use `--write-slurm-ablation-sweep-dir` with
`--slurm-ablation-sweep-modes` to generate both one-sided launchers and their
verification index in one command:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --write-slurm-ablation-sweep-dir slurm_ablation_sweep \
  --slurm-ablation-sweep-modes baseline-disabled isodelta-enabled \
  --slurm-repo-root /scratch/icpp/SevenNet-main \
  --slurm-manifest-path /scratch/icpp/SevenNet-main/isodelta_cluster_suite.toml \
  --slurm-output-dir /scratch/icpp/isodelta_ablation_sweep
```

The sweep writer creates `run_isodelta_baseline-disabled.sbatch`,
`run_isodelta_isodelta-enabled.sbatch`, and
`slurm_ablation_sweep_index.json`. Each generated script gets a mode-specific
`ISODELTA_OUTPUT_DIR`, records the matching `--ablation-mode-override`, and is
immediately reopened with `--verify-slurm-script`; the index stores that
verification evidence plus each launcher SHA-256 fingerprint, so ablation jobs
can be submitted or archived without guessing which launcher passed the
publication-safety checks.
After copying or archiving the sweep directory, reopen the index and every
referenced launcher with `--verify-slurm-ablation-sweep-index`:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --verify-slurm-ablation-sweep-index slurm_ablation_sweep/slurm_ablation_sweep_index.json
```

This verifier checks the sweep schema, mode list, script count,
mode-specific `ISODELTA_OUTPUT_DIR`, recorded `--ablation-mode-override`,
launcher SHA-256 fingerprint, and the nested `--verify-slurm-script` safety
checks for each generated launcher.

Before submitting a long job, write a preflight plan:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --plan-only \
  --plan-output isodelta_cluster_paper_plan.json
```

The plan JSON records artifact existence and download intent, per-case commands,
resolved inputs, expected benchmark/trace/timing outputs, thresholds, and final
paper artifact paths. It also records the manifest SHA-256 digest, so reviewers
can confirm the submitted job used the same manifest they inspected. Review
this file before occupying the 8-GPU queue.

For the final paper run, add a readiness gate before submitting the job:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --readiness-check
```

The `--readiness-check` mode fails if the manifest is still in template form, requests fewer than 8 GPUs, omits SevenNet/MACE/NequIP, uses `trace_only` instead of paired enabled/disabled cases for the required model families, uses one-sided SevenNet or external_pair ablation for a final-paper timing case, leaves required artifacts without SHA-256 protection, omits case preflight commands, or lacks `min_speedup_95ci_lower_bound` gates for paired timing claims. The JSON output lists every passed and failed readiness item so the cluster job is not submitted until the paper claim is auditable.

Run the complete suite on the cluster:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml
```

The runner checks the visible GPU count against `expected_gpus = 8`, downloads
missing artifacts, prints terminal progress as `[suite] [stage/total] ...`,
then executes each case. A `sevennet_lammps` case calls
`run_isodelta_experiment.py`, passes the manifest `ablation_mode`, and then
runs `check_isodelta_experiment_report.py --output` against the generated
driver report before command-log fingerprints are recorded. The suite summary
keeps both the driver report and the report-check evidence as fingerprinted
source evidence, so a reviewer can reopen the bundle and verify the checker
status, schema, command count, and log-fingerprint count; the default `paired`
mode runs the disabled/enabled LAMMPS benchmark plus report gates, while
one-sided modes record raw timing without speedup claims. An
`external_pair` case is for MACE, NequIP, or
another runtime whose disabled and enabled commands are supplied in the
manifest; it uses the same `ablation_mode` field to run both commands or only
the requested external timing side. A `trace_only` case validates portable MLIP
trace evidence when a model has applicability evidence but no paired runtime
benchmark yet.
Unless `--dry-run` is active, the normal suite run also reopens the completed
output bundle immediately after writing tables, correlations, figures, logs,
and fingerprints; any bundle verification failure returns a nonzero exit code.
For `external_pair` results, the suite validates the external timing report
schema, manifest case/model labels, repeat success counts, positive
`baseline_mean_seconds` and `enabled_mean_seconds`, and requires
`speedup_vs_disabled_cache` to match baseline divided by enabled seconds. The
report also keeps raw `baseline_times_seconds` and `enabled_times_seconds`, and
the suite recomputes sample variance/stddev before writing the summary table.
The external timing report also carries `mode_controls`; those controls must
match the manifest disabled/enabled commands and cache-off/cache-on environment
before collect-only tables are accepted. The same `mode_controls` object records
`env_flag_false_values`, so empty string, `0`, `false`, `no`, and `off`
interpretation is auditable in the timing artifact itself. Its `commands` array
must also contain one successful `case:disabled:N` and `case:enabled:N` command record for every repeat, with command text and tracked environment matching the manifest mode controls.
The companion `command_log_fingerprints` array fingerprints every external
stdout/stderr log, and suite validation rechecks those hashes before accepting
the timing report.
In short, external timing report `mode_controls` must match the manifest.
The generated table also reports baseline/enabled timing sample counts and
normal-approximation 95% CI half-widths for the mean timings, so paper tables
can present variability next to the speedup number instead of only reporting a
single average. It also derives `speedup_95ci_lower_bound` and
`speedup_95ci_upper_bound` from the conservative combination of baseline and
enabled timing intervals, which makes it clear whether the measured improvement
survives uncertainty in the repeat timings. Set
`min_speedup_95ci_lower_bound` in the manifest when the final paper run should
fail any case whose conservative speedup bound does not exceed the required
claim threshold.

After successful collection, the suite writes:

- `isodelta_cluster_paper_summary.json`
- `environment_snapshot.json`
- `tables/case_summary.csv`
- `tables/case_summary.md`
- `tables/correlation.csv`
- `tables/command_timing.csv`
- `tables/command_timing.md`
- `tables/repeat_timing.csv`
- `tables/repeat_timing.md`
- `tables/speedup_uncertainty.csv`
- `tables/speedup_uncertainty.md`
- `figures/speedup_by_case.svg`
- `figures/speedup_uncertainty.svg`
- `figures/hit_rate_vs_speedup.svg`
- `figures/trace_metadata_fraction_vs_speedup.svg`
- `isodelta_cluster_suite_manifest.toml`

Generated paper artifacts are self-describing. CSV tables start with a
`# IsoDelta-Halo ...` comment, Markdown tables start with an HTML comment, SVG
figures include a `<desc>` element, and JSON artifacts carry an
`artifact_comment` or `report_comment` field. The stage JSON reports
(`readiness_report.json`, `artifact_preparation_report.json`,
`preflight_report.json`, `isodelta_cluster_paper_plan.json`, and
`pipeline_report.json`) must also keep their `report_comment`; the pipeline
verifier rejects a passed publication report when any required stage report no
longer describes its evidence purpose. These comments are part of the evidence
contract, not decoration: they tell a reviewer what the file means before the
numbers are interpreted.

Use `--collect-only` to regenerate tables, correlations, and figures from
existing benchmark reports, external timing reports, and trace evidence without
rerunning the cluster jobs. Use `--skip-gpu-check` only for local dry runs or
CI tests; for paper runs, keep the GPU check enabled and archive the summary
JSON with the raw logs. The summary JSON stores the manifest SHA-256 digest and
the copied `isodelta_cluster_suite_manifest.toml` snapshot path. It also stores
`case_mode_controls`, so each disabled/enabled pair can be audited for the
cache-off/cache-on environment used to produce the timing rows, including the
`env_flag_false_values` parser list that explains explicit false environment
values. It also stores
`artifact_fingerprints` with the SHA-256 digest and byte size of each generated
environment snapshot, table, SVG figure, and manifest snapshot. When
`preflight_report.json`, `preflight_environment_snapshot.json`,
`isodelta_cluster_paper_plan.json`, or the SLURM-generated
`python_runtime_provenance.txt` already exists in the output directory, those
auxiliary run artifacts are fingerprinted too; when they are absent, the
summary records that absence explicitly so the bundle verifier can still
distinguish a direct run from a missing archived file. If the Python provenance
file is present, `--verify-output-bundle` also reopens it and requires the
comment header plus the `PYTHON_BIN`, `SUITE_RUNNER`, `Python`, and
`sys.executable` markers, so an archived cluster result shows which interpreter
actually executed the experiment. This lets reviewers verify that the submitted
paper artifacts match the archived run.
The sibling `artifacts` index must carry the same artifact names and paths as
`artifact_fingerprints`; `--verify-output-bundle` rejects the bundle if the
human-facing path index points to a different file than the protected hash
record.
The `figures/speedup_uncertainty.svg` plot shows measured speedup with 95% CI
lower/upper error bars for every case that has bounded repeat timing evidence.
The bundle verifier requires one plotted point and one `speedup-ci` error bar
per eligible summary case, so the paper figure cannot silently omit a weak or
wide-confidence run while the table still reports it.
The summary also stores `evidence_fingerprints` for each case's benchmark
report, bundle evidence, SevenNet experiment driver report,
`experiment_report_check.json`, external timing report, and trace evidence
files. Benchmark report, bundle evidence, external timing report, and trace evidence files are all protected source evidence.
The bundle verifier checks those source-evidence SHA-256 digests too, then
reopens any SevenNet experiment report-check evidence to confirm it passed
against the fingerprinted driver report with matching command and log counts.
This means a paper table cannot be verified after its input evidence was edited or lost.
`environment_snapshot.json` records Git/Python provenance, GPU check results,
selected CUDA/SLURM environment variables, package versions for SevenNet, torch,
e3nn, ASE, MACE, and NequIP when installed, and lightweight `nvidia-smi` GPU
identity rows when that tool is available. The suite-level evidence gate
recounts all passed cases before writing the summary:
`min_trace_count` must be satisfied by distinct trace evidence files, and
`min_distinct_trace_models` must be satisfied by the `model` labels inside
those trace files. This prevents a three-model claim from passing with
duplicated MACE evidence mislabeled in the manifest.
The summary also stores `command_log_fingerprints` for every launched command,
including each stdout/stderr log path, existence flag, SHA-256 digest, and byte
size. This lets reviewers confirm that the archived logs match the command
records used to build the paper tables. The generated `command_timing.csv` and
`command_timing.md` tables mirror those command records with command name,
return code, elapsed seconds, stdout/stderr paths, and working directory so
repeat-level timing provenance is inspectable without opening the JSON first.
The generated `repeat_timing.csv` and `repeat_timing.md` tables mirror the raw
per-repeat source timing evidence itself: SevenNet rows come from benchmark
`results[*].loop_time_seconds`, while external MACE/NequIP rows come from
`baseline_times_seconds` and `enabled_times_seconds` in the external timing
report. This gives the paper appendix the disabled/enabled timing samples
behind the means, confidence intervals, speedup bounds, and correlation tables.
The generated `speedup_uncertainty.csv` and `speedup_uncertainty.md` tables
pull the repeat counts, baseline/enabled means, 95% CI half-widths, and
conservative speedup lower/upper bounds into a compact appendix table for
checking the numerical claim without scanning the wider case summary.
Each command record also stores the
working directory and a focused `tracked_env` snapshot for cache mode, CUDA,
SLURM, and CPU thread variables. That makes a disabled/enabled MACE, NequIP, or
SevenNet timing row auditable without dumping unrelated environment variables.
The bundle verifier also checks that command records and log fingerprints carry
the same command names, return codes, and stdout/stderr paths before it accepts
the archived log hashes.
After archiving or moving a result directory, verify the bundle fingerprints:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --verify-output-bundle isodelta_cluster_paper_runs
```

The verifier accepts either the output directory or the
`isodelta_cluster_paper_summary.json` path and fails if any recorded artifact or
command log fingerprint no longer matches the filesystem.
It also performs semantic paper-artifact checks after the SHA-256 pass:
`case_summary.csv` and `case_summary.md` must contain exactly one row per
summary case and the same cell values as `summary["cases"]`.
`correlation.csv` must contain the configured metric-pair rows and the same values as
`summary["correlations"]`, and `command_timing.csv`/`command_timing.md` must
contain the same command rows as `summary["commands"]`.
`repeat_timing.csv`/`repeat_timing.md` must match the raw source timing evidence
protected by `evidence_fingerprints`, so changing a benchmark report or external
timing report without regenerating the repeat table is rejected.
`speedup_uncertainty.csv`/`speedup_uncertainty.md` must match the uncertainty
fields in `summary["cases"]`, so the paper-ready CI table cannot drift from the
JSON evidence.
Every generated table must keep its explanatory comment.
Each SVG figure must parse as an SVG document with width, height, viewBox, and the expected `<desc>` description;
the speedup chart must include every measured-speedup case label from
`summary["cases"]`, and scatter plots must contain the same number of plotted
points as the summary data pairs they visualize.
`environment_snapshot.json` must carry the expected snapshot schema and `artifact_comment`, stage reports
must carry the expected `report_comment`, and the manifest snapshot must contain both the generated-file comment and the `[suite]` table.
When the summary records `suite.manifest`, the verifier also hashes the manifest snapshot body after the generated comment and requires that SHA-256 digest and byte size to match the recorded manifest provenance.
This catches a corrupted table or graph even when the summary JSON was
regenerated with a matching hash.
For `external_pair` cases, `--verify-output-bundle` also reopens the archived
external timing report and rechecks its nested `command_log_fingerprints`, so
MACE/NequIP stdout/stderr logs cannot drift silently after collection.

For an interrupted cluster job, rerun with `--reuse-passed` instead of starting
from zero:

```bash
python tools/run_isodelta_cluster_paper_suite.py \
  --manifest isodelta_cluster_suite.toml \
  --reuse-passed
```

The runner only reuses a case when the planned benchmark report, external timing
report, trace evidence, or bundle evidence already exists and still passes the
current validation gates. Reused rows are marked as `reused` in the summary
table, and suite-level evidence treats them as passed evidence. If a threshold
changed or a file is missing/corrupt, that case falls back to the normal command
execution path.

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
For one-sided ablation smoke runs, pass `--ablation-mode baseline-disabled` or
`--ablation-mode isodelta-enabled`. The report records both `ablation_mode` and
`benchmark_cases`, and `summary.speedup_vs_disabled_cache` is left empty unless
both modes were run. Use the default `--ablation-mode paired` for paper numbers
and for `check_isodelta_benchmark_report.py`.
The runner rejects `--repeat` values below 1 because a zero-repeat report has
no paired timing, cache-hit, or final-thermo evidence to audit.
It rejects an empty `--lammps-command` before launching any external process.
It also applies `--run-timeout-seconds` to each individual LAMMPS invocation so
a hung MPI launch becomes a failed run with raw stdout/stderr logs instead of an
unbounded experiment.

## Report Fields

The runner writes `isodelta_benchmark_report.json`. Important fields are:

- `report_comment`
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
The checker also requires `report_comment`, so a copied benchmark JSON still
describes that it contains disabled-cache baseline and enabled-cache timing,
thermo, cache-summary, and provenance evidence.
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
match their cached CPU index-vector lengths, integer dtype, and target device
before reuse; a mismatch is treated as `miss_index-tensor-shape-changed` and the
metadata cache is rebuilt.

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
The resulting `bundle_evidence.json` records a top-level `report_comment`, the
path, SHA-256 digest, and byte size for the benchmark report and every trace evidence file.
It also includes `bundle_schema_version` and a `provenance` object with Git, Python, and platform
metadata for the bundle checker run, so archive it beside raw LAMMPS logs, trace
JSON, and plotting scripts. The cluster suite reopens an existing bundle
evidence file and rejects it when the `report_comment` is missing or does not
describe the bundle evidence purpose.

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
