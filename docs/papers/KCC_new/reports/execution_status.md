# KCC_new Execution Status

## Executed in current pass

### 1. main end-to-end
- script: `scripts/kcc_new_pair_end_to_end.py`
- status: executed
- config: warmup 3, repeat 30
- scope: 31 datasets, representative sample 1개

### 2. accuracy repeat
- script: `scripts/kcc_new_pair_accuracy_repeats.py`
- status: executed
- config: warmup 2, repeat 30
- scope: 31 datasets, representative sample 1개

### 3. representative torch profiler
- script: `scripts/kcc_new_pair_profiler_representatives.py`
- status: executed
- scope: `qm9_hf`, `mptrj`
- modes: `forward_energy`, `force_model`
- cases: `sevennet_baseline`, `sevennet_geometry_only`

### 4. geometry_only intrusive breakdown
- script: `docs/papers/KCC/scripts/kcc_geometry_only_breakdown.py`
- status: executed and copied into `KCC_new`
- config: repeat 30

### 5. LAMMPS pair metadata bench
- script: `docs/papers/KCC/scripts/kcc_lammps_pair_metadata_bench.py`
- status: executed and copied into `KCC_new`
- config: repeat 30
- scope: serial `pair_style e3gnn`

## Not executed in current canonical pass

### 1. FlashTP / four-case intrusive profile
- reason: current paper mainline excludes FlashTP

### 2. pair_validation_split (100 repeat)
- reason: old KCC diagnostic retained as reference, but current canonical paper uses `baseline vs geometry_only`

### 3. lmax sweep / quadrant runtime
- reason: useful background analysis but outside current canonical runtime paper scope

### 4. Nsight
- reason: optional diagnostic, not required for current canonical argument

## Canonical paper evidence

1. accuracy preservation
2. main end-to-end latency
3. geometry_only internal breakdown
4. LAMMPS upstream pair-metadata reduction
5. representative profiler interpretation
