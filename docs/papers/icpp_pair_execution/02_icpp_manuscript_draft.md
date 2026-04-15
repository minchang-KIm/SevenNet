# Toward Pair-Major Exact Execution for Equivariant GNN Interatomic Potentials

## Note

This manuscript is a **pre-submission draft** aligned with the revised plan.

It is intentionally honest about the current repository state:

- current code implements geometry-side pair reuse,
- current code does not yet implement pair-major tensor product,
- some earlier benchmark headlines were affected by warm-up methodology,
- therefore this draft should be treated as a target manuscript scaffold for ICPP, not as a final submission-ready paper.

## Abstract

Short-range equivariant graph neural network interatomic potentials process the same physical interaction twice as two directed edges. This creates an exact runtime reuse opportunity: distance, radial basis, cutoff, and spherical-harmonic geometry for a reverse edge can be derived from the corresponding forward edge without changing the learned model or its mathematical formulation. We implement this idea in SevenNet as a pair-aware execution path that constructs pair metadata, computes geometry-side quantities once per undirected pair, reconstructs reverse spherical harmonics through parity, and reuses pair-level filter-network inputs. The implementation preserves the original model output up to floating-point ordering effects and matches baseline energies and forces closely across public workloads. However, our current results also show that geometry-only reuse is not sufficient for strong end-to-end gains. Tensor-product message generation, edge-to-node aggregation, and force/stress backward remain effectively edge-major, so observed benefits on realistic periodic workloads are modest and workload-dependent. This draft therefore argues for a stronger systems design centered on pair-major tensor product execution, topology-epoch metadata reuse, and deployed LAMMPS evaluation. The current implementation and measurements provide the exactness foundation and the runtime diagnosis needed to motivate that next step.

## 1. Introduction

Equivariant GNN interatomic potentials such as NequIP-style models operate on local atomic environments represented as graphs. For short-range models, edges are built from neighbors inside a cutoff radius and messages are exchanged over directed edges. This design is natural from a message-passing perspective, but it introduces a systems-level inefficiency: the physical interaction between atoms `i` and `j` is represented twice, once as `i -> j` and once as `j -> i`.

The two directed edges are not independent from a geometric point of view. Their distances are identical, their radial basis and cutoff factors are identical, and their spherical harmonics are related by parity. This creates an opportunity to restructure the runtime without changing the model formulation or retraining any parameters.

The central question of this work is whether an equivariant GNN-IP runtime can exploit this pair symmetry in a way that yields measurable inference gains while preserving exactness. The current repository work answers this question partially. It shows that geometry-side reuse is straightforward and exact up to floating-point effects. At the same time, it shows that this step alone is not enough to produce a compelling systems result, because the expensive downstream path remains edge-major.

This observation is important for two reasons. First, it prevents overclaiming a partial optimization as a general acceleration method. Second, it identifies the true systems target: pair-major tensor-product execution and pair-aware deployed runtime integration.

## 2. Background

### 2.1 Equivariant GNN-IP Inference Pipeline

In a NequIP-style equivariant model, inference proceeds through the following stages:

- atom types are embedded into node features,
- neighbor edges are built from positions and cutoff,
- edge geometry is encoded through radial basis, cutoff, and spherical harmonics,
- interaction blocks compute filter weights and tensor-product messages,
- messages are aggregated to node features,
- scalar atomic energies are read out,
- forces and stress are obtained by differentiation of the total energy.

In SevenNet, this structure is visible directly in the model construction path. The model appends a `ForceStressOutputFromEdge` module after energy prediction and marks `EDGE_VEC` as requiring gradient, so force/stress inference is part of the deployed inference graph rather than a separate post-processing step.

### 2.2 Reusable and Non-Reusable Terms

For a reverse pair of directed edges, some quantities are exactly reusable:

- distance
- radial basis
- cutoff
- pair-level filter-network input

Some quantities are reconstructable:

- spherical harmonics, via parity sign for the reverse direction

Some quantities remain directed:

- source-node-dependent tensor-product message
- edge-to-node aggregation destination
- force/stress backward through the energy graph

This distinction is the key systems boundary in the current work.

## 3. Current Implementation

The current repository implements pair-aware geometry reuse in SevenNet.

First, runtime pair metadata is built from the directed edge list. The implementation stores:

- edge-to-pair map
- reverse-direction mask
- canonical forward edge index
- reverse edge index
- pair-has-reverse mask

Second, geometry-side edge embedding is changed from edge-major to pair-aware execution. For each canonical pair direction, the runtime computes:

- pair distance
- radial basis
- cutoff-applied edge embedding
- spherical harmonics

The reverse spherical harmonics are then reconstructed using the parity sign associated with each irreducible representation degree.

Third, the filter network is evaluated once per pair on the pair-level embedding rather than once per directed edge.

What the current implementation does **not** change is the tensor-product execution model. In the non-accelerated e3nn path, forward and reverse messages are still computed separately. In the FlashTP path, the pair-level weights are expanded back to edge-major layout before entering the fused kernel. Therefore the current implementation is best described as **geometry-side pair reuse**, not pair-major execution.

## 4. Exactness

The implementation preserves the original model outputs closely.

Across the public-local benchmark suite, the worst observed energy difference versus baseline is `6.104e-05 eV`, and the worst observed force difference is `2.441e-04 eV/A`. On the real-dataset e3nn benchmark, the worst observed force delta is `9.155e-05 eV/A`. These differences are consistent with floating-point ordering effects rather than a semantic change in the model.

This exactness result matters because it confirms the central premise of the work: the runtime can be restructured without modifying the trained parameters or the model formulation.

## 5. Current Performance Evidence

### 5.1 What the Current Results Show

The current evidence supports three main observations.

First, geometry-side reuse is real and measurable. On some larger periodic workloads, the pair-aware path is faster than the baseline.

Second, the end-to-end speedup is currently modest on the workloads that matter most for a systems paper. Representative rechecks with stable post-warm-up timing show approximately:

- `salex_train_official`: `1.03x`
- `oc20_s2ef_train_20m`: `1.06x`
- `mptrj`: `1.01x`
- `omat24_1m_official`: `1.01x`

Third, many smaller molecular workloads are still slower under the current implementation.

### 5.2 What the Current Results Do Not Yet Support

The current results do not support the following stronger claims:

- universal acceleration
- large end-to-end gains on realistic workloads
- strong benefit on molecular workloads
- superiority over accelerator-aware baselines such as FlashTP in their strongest path

### 5.3 Benchmark Caveat

One important methodological issue was discovered during review of the current benchmark suite. The earlier all-public local summary used a short repeated evaluation window that still included heavy warm-up effects on small graphs. As a result, some headline numbers, especially the previously reported `qm9_hf 3.7x`, should not be treated as final steady-state results.

This does not invalidate the correctness of the implementation. It does invalidate any strong performance claim derived from those transient-heavy measurements. A corrected benchmark pass must precede any submission.

## 6. Why Geometry-Only Reuse Is Not Enough

The performance diagnosis from the current implementation is clear.

The runtime saves work in:

- geometry construction
- spherical harmonics evaluation
- filter-network input rows

But it leaves the following costs largely intact:

- tensor-product message generation
- edge-to-node aggregation
- force/stress backward through the whole energy graph

This explains why speedups are small. The optimization reduces one important slice of the workload, but not the dominant slice.

The diagnosis is even stronger in deployed settings. In serial ASE-style inference, the force module performs `autograd.grad(E, edge_vec)` inside the model. In parallel LAMMPS deployment, the code performs explicit segmented backward and communication over intermediate activations and ghost features. Therefore force inference is not a negligible epilogue; it is a first-class runtime path that must be co-optimized.

## 7. LAMMPS and Distributed Implications

It would be incorrect to assume that LAMMPS eliminates the relevant overhead simply because it provides the neighbor list.

LAMMPS does remove some graph-build work, but the deployed runtime still pays for:

- edge tensor construction
- cell-shift handling
- pair metadata construction or validation
- pair-edge vector extraction
- force/stress backward
- and, in parallel, communication and reverse communication

This is particularly important for an ICPP paper. A strong submission must demonstrate end-to-end deployed behavior, not just Python-side calculator timing.

## 8. Revised System Design

The current implementation suggests the correct next design.

### 8.1 Pair-Major Tensor Product

The runtime should keep pair layout alive through message generation. Instead of computing a canonical pair and then re-expanding to edge-major layout before tensor product, the runtime should consume:

- pair source and destination indices
- pair geometry
- pair filter weights
- source-node features for both directions

and emit both directed messages from one pair-major execution unit.

### 8.2 Topology-Epoch Metadata Reuse

Pair metadata should not be regenerated or fully revalidated on every step when the neighbor topology is unchanged. In MD, topology is often stable over a short epoch. The runtime should exploit this directly.

### 8.3 Deployed Runtime Evaluation

Any final paper must include:

- corrected ASE-side evaluation
- serial LAMMPS evaluation
- parallel LAMMPS evaluation
- and a communication-aware distributed analysis

## 9. Limitations of the Current Draft

This draft has three major limitations.

First, the strongest performance contribution is not yet implemented. Pair-major TP remains future work.

Second, the benchmark methodology still needs one corrected, authoritative re-run across the full dataset matrix.

Third, the systems story is incomplete without deployed LAMMPS end-to-end measurements and distributed scaling.

These limitations are serious enough that the work should not be submitted in its current form.

## 10. Conclusion

The current repository work establishes an exactness-preserving pair-aware geometry reuse path for equivariant GNN interatomic potentials. That is a meaningful result, but it is only the first stage of the intended systems contribution. The current evidence shows that geometry-only reuse is insufficient for a strong end-to-end acceleration claim because tensor-product execution, aggregation, and force backward remain edge-major. The correct path forward is therefore not to oversell the existing implementation, but to treat it as the foundation for a stronger pair-major runtime that integrates tensor-product execution, topology-aware reuse, and deployed LAMMPS evaluation. That revised direction is the one that can plausibly support an ICPP submission.

## Submission Readiness Statement

If this project is to be submitted to ICPP, the following should be completed first:

- corrected benchmark methodology
- pair-major TP or an equally strong runtime redesign
- serial and parallel LAMMPS evaluation
- exactness section with authoritative tables
- realistic large-workload speedups that remain after warm-up correction

Without those items, the work is better positioned as an internal report or a pre-submission design study than as a conference paper.
