# Current Status Report for Pair Execution

## Executive Summary

The current repository state is **not ready for an ICPP submission** as-is.

The main reasons are:

- the implemented method is weaker than the strongest seminar claim,
- the benchmark methodology used in some summary tables has a warm-up artifact,
- the end-to-end gains on realistic periodic workloads are currently modest,
- there is no LAMMPS end-to-end serial/parallel performance section that would support a systems paper claim,
- there is no pair-major tensor-product kernel yet, so the dominant edge-major work remains.

The work is still valuable, but its current value is closer to:

- a validated runtime diagnosis,
- an exactness-preserving geometry-side reuse implementation,
- and a concrete roadmap toward pair-major execution.

That is not yet the same thing as a publishable high-impact performance result.

## What Is Actually Implemented

The current implementation does the following:

- pair metadata construction and cacheable topology signature generation in [pair_runtime.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_runtime.py)
- pair-wise geometry reuse in [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)
- reverse spherical harmonics reconstruction via parity sign in [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py#L217)
- pair-wise `weight_nn` reuse in [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py#L120)
- forward and reverse message evaluation in separate directed-edge passes in [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py#L124)
- ASE and local public-dataset benchmarking infrastructure in [all_public_local_pair_bench.py](/home/wise/minchang/DenseMLIP/SevenNet/bench/all_public_local_pair_bench.py)
- dataset download and local inventory management in [download_public_mlip_datasets.py](/home/wise/minchang/DenseMLIP/SevenNet/bench/download_public_mlip_datasets.py)

What is **not** implemented:

- pair-major tensor-product execution
- pair-major fused online reduction
- FlashTP plus pair-major fused backend
- topology-epoch reuse tied to real MD neighbor rebuild epochs
- distributed backward/communication pruning

## Seminar Claim Audit

The following seminar claims are currently supported by code:

- reusable quantities can be classified into distance, radial basis, cutoff, spherical harmonics, weight/filter, and directed message terms
- the current implementation preserves the original model formulation and only changes execution layout
- spherical harmonics reuse is mathematically justified by parity sign reconstruction
- the current implementation reuses geometry-side work but leaves tensor product and aggregation effectively directed-edge based

The following seminar directions are **future work**, not current implementation:

- FlashTP-tight pair-major integration
- graph-specific JIT compilation
- pair-major tensor-product kernel
- distributed pair-aware execution redesign

## Correctness and Exactness Status

Baseline comparison has been performed.

Relevant files:

- [all_public_local_pair_main aggregated.csv](/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/all_public_local_pair_main/metrics/aggregated.csv)
- [real_e3nn_pair aggregated.csv](/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/real_e3nn_pair/metrics/aggregated.csv)
- [real_main aggregated.csv](/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/real_main/metrics/aggregated.csv)

Observed worst deltas:

- all-public local run: worst force delta `2.441e-04 eV/A`, worst energy delta `6.104e-05 eV`
- real e3nn run: worst force delta `9.155e-05 eV/A`, worst energy delta `0`
- FlashTP real run: worst force delta `7.63e-05 eV/A`, worst energy delta `0`

Interpretation:

- energy agreement is effectively exact within floating-point tolerance,
- force differences are small but not literally zero,
- the current method is still acceptable as an exact-execution reformulation up to floating-point ordering effects,
- the `OMat24` representative case should be rechecked with tighter numerical logging because it is the worst public-local case.

## Benchmark Status

### What Has Been Measured

Measured artifacts exist for:

- real public datasets
- detailed model profiling
- size-density analysis
- all-public local dataset sweep

Representative result files:

- [gnn_ip_pair_execution_all_public_local_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/gnn_ip_pair_execution_all_public_local_report.md)
- [gnn_ip_pair_execution_size_profiling_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/gnn_ip_pair_execution_size_profiling_report.md)
- [gnn_ip_pair_execution_baseline_detailed_profile_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/gnn_ip_pair_execution_baseline_detailed_profile_report.md)

### What Is Wrong With the Current Headline Summary

The `all_public_local_pair_main` summary contains a methodological issue:

- the steady-state number uses a short repeated evaluation window,
- small graphs still show strong warm-up and autotuning transients in the first one or two repeated calls,
- therefore some headline values are overstated or even qualitatively wrong.

The clearest example is `qm9_hf`.

The summary file reports:

- `qm9_hf`: `3.737x`

However, when the first two repeated calls are discarded and only the stable region is inspected, the behavior flips:

- baseline stable median: about `28.6 ms`
- pair stable median: about `47.1 ms`

This means the previous `qm9_hf` headline is a benchmark artifact, not a real steady-state result.

The same recheck confirms that the current method is only modestly positive on representative periodic workloads:

- `salex_train_official`: about `1.03x`
- `oc20_s2ef_train_20m`: about `1.06x`
- `mptrj`: about `1.01x`
- `omat24_1m_official`: about `1.01x`

Therefore the correct interpretation is:

- the current method is **not** a universal acceleration,
- it is **not** strongly positive on small molecular workloads,
- and even on favorable periodic workloads the gain is currently only a few percent.

## Why the Speedups Are Modest

The central reason is simple: the implementation reduces geometry-side duplication, but it does not remove the dominant edge-major work.

What gets reduced:

- pair geometry construction
- radial basis and cutoff evaluation
- spherical harmonics evaluation
- `weight_nn` rows

What remains effectively unchanged:

- tensor-product message generation for both directions
- edge-to-node aggregation
- force/stress backward path through the energy graph

Code evidence:

- geometry-side reuse: [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py#L217)
- pair `weight_nn`: [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py#L120)
- directed forward and reverse TP execution: [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py#L124)

This is why the current implementation behaves like a **partial reuse optimization**, not a full pair-major execution engine.

## LAMMPS Clarification

LAMMPS already provides the neighbor list, but that does **not** eliminate all pair-execution overhead.

What LAMMPS avoids:

- rebuilding the neighbor list in Python/ASE style

What still remains:

- cell-shift handling
- edge tensor materialization
- pair metadata construction or topology-cache validation
- `pair_edge_vec` materialization
- backward-derived force/stress path
- in parallel, ghost communication and reverse communication

Code evidence:

- serial LAMMPS path still builds pair metadata and `pair_edge_vec`: [pair_e3gnn.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn.cpp#L303)
- parallel path still builds pair metadata and then runs segmented backward with communication: [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp#L412), [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp#L499)

So the correct statement is:

- LAMMPS removes some graph-build overhead,
- but it does **not** remove pair metadata overhead, control-path overhead, or backward/communication cost.

## Where Backward Appears in Inference

Inference backward is not limited to a final MLP readout.

In the standard model build:

- `ForceStressOutputFromEdge` is appended as the final module in [model_build.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/model_build.py#L625)
- the model marks `EDGE_VEC` as requiring grad through [sequential.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/sequential.py#L166)
- the final force/stress module calls `torch.autograd.grad(E, edge_vec)` in [force_output.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/force_output.py#L173)

That means the gradient traverses the whole upstream graph:

- edge embedding
- convolution
- tensor product
- self interaction
- readout

In serial ASE and serial torchscript deployment, this backward is hidden inside the model forward.

In parallel LAMMPS deployment, the backward is performed explicitly across model segments in C++:

- [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp#L499)

This is why any ICPP submission must treat backward and communication as first-class runtime costs, not as a negligible epilogue.

## ICPP Readiness Assessment

### What ICPP Will Likely Expect

- a clear systems contribution beyond a local code cleanup
- stable and defensible benchmarking methodology
- end-to-end evidence in the target runtime, especially LAMMPS
- meaningful speedups on realistic workloads
- single-GPU and multi-GPU scaling analysis
- strong ablation and exactness validation

### What Is Missing Today

- corrected steady-state methodology across the full benchmark matrix
- end-to-end LAMMPS serial benchmark section
- end-to-end LAMMPS parallel benchmark section
- pair-major TP or another stronger runtime contribution
- distributed communication analysis
- hardware-counter or memory-traffic evidence
- a convincing performance win over strong baselines on realistic large workloads

### Critical Verdict

The current implementation is **not submission-ready** for ICPP.

If submitted now, the paper would be vulnerable on all of the following fronts:

- overclaim: implementation is weaker than the strongest narrative
- methodology: warm-up artifact exists in a flagship summary
- novelty: no pair-major kernel yet
- systems depth: LAMMPS and distributed sections are incomplete
- performance impact: current gains are too small and workload-dependent

## Recommended Immediate Action

- treat the current repository state as a diagnosis and baseline-construction phase
- repair the benchmark methodology first
- narrow the claim to what is already true
- do not use `qm9_hf 3.7x` or similar values in any external-facing narrative
- only move toward submission after pair-major TP or another comparably strong runtime redesign is implemented and validated
