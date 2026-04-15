# Revised Plan Toward an ICPP Submission

## Objective

The goal should no longer be:

- "submit the current geometry-only reuse implementation as an acceleration paper"

The goal should be:

- "turn the current diagnosis and partial reuse implementation into a real pair-major execution system with sound end-to-end evaluation"

This revised objective is necessary because the present implementation does not yet support a strong systems paper claim.

## Revised Research Claim

The paper should aim to claim the following, and nothing stronger:

- short-range equivariant GNN-IP inference contains exact pair-symmetric reuse opportunities before tensor product,
- a runtime that preserves the original model formulation can exploit those opportunities,
- the main obstacle is not spherical harmonics reuse itself, but the fact that tensor product, aggregation, and force backward remain edge-major,
- pair-major tensor-product execution and topology-aware runtime design are required for robust end-to-end gains.

This claim is narrower than the seminar ambition, but it is technically defensible.

## Phase 0: Benchmark Sanitation

This phase is mandatory before any new claim.

Tasks:

- revise the benchmark scripts so warm-up iterations are explicitly excluded from steady-state reporting
- separate cold-start, post-warm-up steady-state, and long-run median
- re-run all-public local, real e3nn, and size-density benchmarks with corrected methodology
- flag previous outliers such as `qm9_hf` as invalidated headline results
- produce one authoritative benchmark table for future writing

Deliverables:

- corrected CSVs
- corrected plots
- corrected summary report

Exit criterion:

- no flagship number depends on transient warm-up behavior

## Phase 1: Scope and Claim Repair

Tasks:

- rewrite all seminar and paper language so that current implementation is described as geometry-side reuse, not pair-major TP
- separate current implementation from future work in every document
- define a clear go/no-go rule for submission

Deliverables:

- updated slide deck
- updated reports
- updated manuscript framing

Exit criterion:

- all claims in text are directly backed by current code or clearly marked as future work

## Phase 2: Runtime Redesign

This is the actual technical phase that can create an ICPP-level contribution.

### 2.1 Pair-Major Tensor Product

Tasks:

- design a pair-major TP interface that consumes one canonical pair and emits both directions
- keep pair layout alive through message generation instead of expanding back to edge-major form early
- avoid re-materializing per-edge pair weights wherever possible
- add a reference implementation first, then optimize

Success criterion:

- the dominant message-generation path is no longer split into separate forward and reverse directed-edge TP passes

### 2.2 Topology-Epoch Cache

Tasks:

- tie pair metadata reuse to actual neighbor-topology epochs rather than per-call checks only
- integrate topology cache logic with LAMMPS neighbor rebuild semantics
- minimize metadata and index regeneration when topology is unchanged

Success criterion:

- pair metadata construction is no longer paid on every MD step when topology is unchanged

### 2.3 LAMMPS-Aware Systems Path

Tasks:

- benchmark serial LAMMPS end-to-end
- benchmark parallel LAMMPS end-to-end
- profile where time goes in deployed serial and parallel execution
- quantify how much of the remaining overhead is metadata, TP, force backward, and communication

Success criterion:

- at least one end-to-end LAMMPS path shows stable, reproducible gain on realistic workloads

### 2.4 Distributed Backward Pruning

Tasks:

- analyze whether only scalar energy-contributing paths need to participate in some communication steps
- test whether `self_conn_grads` and related paths can be structurally pruned in deployed inference

Success criterion:

- either a measured communication reduction or a defensible proof that this path is not worth pursuing

## Phase 3: Evaluation Plan

### Workloads

Use four workload groups:

- small sparse molecular
- small dense periodic
- large sparse
- large dense periodic

Representative datasets:

- `SPICE 2023`, `ANI-1x`, `ANI-1ccx`
- `phononDB`, `WBM`
- `OMol25`, `OC20`, `OC22`
- `MPtrj`, `OMat24`, `sAlex`, `MD22 nanotube`

### Baselines

- SevenNet e3nn baseline
- SevenNet current pair-execution implementation
- revised pair-major implementation
- FlashTP baseline where applicable
- revised pair-major plus accelerator path if implemented

### Metrics

- energy, force, stress exactness
- cold-start latency
- post-warm-up steady-state latency
- long-run median latency
- model-stage breakdown
- memory footprint and traffic proxy
- GPU utilization
- serial LAMMPS end-to-end runtime
- parallel LAMMPS strong scaling and weak scaling

### Required Plots

- speedup by dataset
- speedup versus atoms
- speedup versus edges
- speedup versus average neighbors
- stage breakdown on representative cases
- single-GPU to multi-GPU scaling
- exactness summary

## Go/No-Go Criteria for ICPP

Do **not** submit if any of the following is still true:

- the benchmark methodology is still not fully corrected
- the main implementation is still geometry-only reuse without pair-major TP
- there is no serial LAMMPS end-to-end section
- there is no parallel LAMMPS section
- realistic large periodic datasets only show noise-level gains

Recommended minimum evidence before submission:

- stable end-to-end gain on realistic periodic workloads
- at least one convincing LAMMPS end-to-end speedup section
- clear explanation of where the gain comes from
- exactness validated against baseline
- performance scaling story that looks like a systems contribution, not a local micro-optimization

Practical threshold:

- if the revised system cannot show a robust win beyond a few percent on the target workloads, the work should be deferred or redirected to a workshop, internal report, or methodology paper rather than ICPP

## Writing Plan

The paper should be written only after Phase 0 and Phase 2.1 are complete.

Recommended section structure:

1. Problem: exact pair-symmetric reuse opportunity in equivariant GNN-IP inference
2. Background: current edge-major execution in SevenNet/NequIP-style models
3. Current limitation: geometry-only reuse is insufficient because TP and backward remain edge-major
4. Revised design: pair-major TP, topology-epoch cache, and deployed runtime integration
5. Evaluation methodology: corrected warm-up policy, ASE and LAMMPS, single and multi-GPU
6. Results: exactness, performance, scaling, ablation
7. Discussion: when pair-major execution helps and when it does not

## Final Recommendation

The work should proceed in the following order:

1. benchmark sanitation
2. claim repair
3. pair-major TP implementation
4. LAMMPS end-to-end evaluation
5. distributed evaluation
6. final manuscript assembly

Submitting before that sequence is complete would be risky and likely premature.
