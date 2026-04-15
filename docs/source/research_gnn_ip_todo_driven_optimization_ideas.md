# TODO-Driven Optimization Ideas for SevenNet Pair Execution

This note records optimization and research ideas that emerged from existing
`TODO` comments and adjacent runtime code. Nothing in this document is
implemented here. The goal is to capture promising directions before they are
lost inside scattered comments.

## Scope

The ideas below are driven by TODOs and current bottlenecks in:

- `sevenn/pair_runtime.py`
- `sevenn/nn/convolution.py`
- `sevenn/nn/node_embedding.py`
- `sevenn/scripts/deploy.py`
- `sevenn/scripts/inference.py`
- `sevenn/train/graph_dataset.py`
- `sevenn/model_build.py`
- `sevenn/pair_e3gnn/pair_e3gnn.cpp`
- `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`

The focus is not on generic cleanup. The focus is on ideas that could plausibly
improve runtime, memory traffic, backend utilization, or distributed scaling.

## Current pressure points inferred from TODOs and runtime structure

### 1. Pair metadata generation still sits on the hot path

The current pair execution path removes duplicated geometry work, but reverse
edge matching is still built dynamically in Python and C++ using general-purpose
lookup structures. This means pair execution can save tensor work while still
paying a nontrivial control-path cost.

Relevant code and TODO context:

- `sevenn/pair_runtime.py`
- `sevenn/pair_e3gnn/pair_e3gnn.cpp`
- `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`

### 2. Backend selection is still static or heuristic

`model_build.py` already contains a TODO noting that some cuEquivariance choices
were benchmark-driven on a specific A100 setup. The pair runtime currently uses
policy heuristics rather than measured backend choice.

Relevant code and TODO context:

- `sevenn/model_build.py`
- `sevenn/nn/flash_helper.py`
- `sevenn/nn/cue_helper.py`
- `sevenn/nn/oeq_helper.py`

### 3. Node feature preparation still expands one-hot explicitly

`node_embedding.py` already notes that one-hot preprocessing should ideally move
out of the current path. With pair execution enabled, one-hot expansion becomes
even more obviously memory-bound because it happens before all interaction
blocks and is reused repeatedly.

Relevant code and TODO context:

- `sevenn/nn/node_embedding.py`

### 4. Deploy and dataset flows still duplicate work

`deploy.py` explicitly notes that the parallel deploy path should build the
model only once. `graph_dataset.py` also notes missing dataset metadata
serialization. These are not only maintenance issues: they hint at avoidable
runtime and preprocessing duplication.

Relevant code and TODO context:

- `sevenn/scripts/deploy.py`
- `sevenn/train/graph_dataset.py`

### 5. Distributed backward and communication remain over-general

`pair_e3gnn_parallel.cpp` already notes that most values of
`self_conn_grads` are zero because only scalar channels contribute to energy.
This suggests the current backward/communication path may still carry data that
the deployed inference pipeline does not need.

Relevant code and TODO context:

- `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`

## Proposed ideas

## 1. Neighbor-Epoch Pair Plan Caching

### Idea

Instead of rebuilding reverse-edge pairing from raw edge lists whenever pair
execution is enabled, cache a compact "pair plan" for each topology epoch.
The pair plan should include:

- forward edge index per pair
- reverse edge index per pair, or a singleton marker
- reverse parity flag
- optional packed pair key for debugging

The key change is that the cache should be tied to the neighbor builder's own
topology version or rebuild event, not inferred by rebuilding hashable metadata
from scratch.

### Why this is promising

The current implementation already caches pair metadata after first build, but
the cache check itself still reconstructs and compares topology descriptors.
A topology-version-driven plan would move pair construction out of the steady
state almost completely.

### Novelty potential

Moderate. On its own this is a systems optimization. Combined with pair
execution, it becomes part of a larger "topology-epoch execution" story.

## 2. Packed-Key Reverse Matching for Pair Construction

### Idea

Replace string or tuple based reverse-edge matching with packed integer keys.
For periodic paths, the key can be built from:

- local/global source index
- local/global destination index
- integer cell shift

For non-periodic Python graphs, the key can be built from:

- source index
- destination index

This keeps pair matching exact while removing repeated string formatting,
allocation, and hashing overhead.

### Why this is promising

Current pair construction uses general-purpose keys. This is flexible but not
cheap. Pair matching is deterministic and structurally simple enough to justify
a packed representation.

### Novelty potential

Low as a standalone idea. High practical value. Best positioned as an enabling
mechanism for faster pair execution rather than as the main claim.

## 3. Pair-Plan Serialization in Graph Datasets

### Idea

Extend `SevenNetGraphDataset` save/load so that preprocessed `.pt` graphs can
store:

- cutoff metadata
- pair plan metadata
- topology signature metadata

At load time, the runtime would validate that the graph already contains a
compatible pair plan and skip rebuilding it.

### Why this is promising

This directly follows from the dataset TODOs and removes repeated preprocessing
from training and offline inference workflows. It also makes pair execution
much more realistic for large graph corpora.

### Novelty potential

Low as a paper idea, but useful as infrastructure. It also strengthens any
experimental story because it reduces measurement noise from preprocessing.

## 4. Backend-Autotuned Pair Policy Selection

### Idea

Replace the current heuristic `auto` policy with a microbenchmark-guided
selection layer keyed by:

- backend (`e3nn`, `FlashTP`, `cuEq`, `OEQ`)
- irreps layout
- average degree
- node count
- device capability

The autotuner would choose among:

- baseline
- pair `geometry_only`
- pair `full`

and cache the result.

### Why this is promising

The current code already hints that backend choices were benchmarked manually on
one GPU generation. A shape-aware autotuner would turn that ad hoc tuning into a
reusable runtime policy.

### Novelty potential

Moderate. It is not enough for a paper alone, but it becomes interesting if the
main paper claim is that pair execution must be co-designed with backend
specialization rather than exposed as a single global switch.

## 5. Pair-Aware FlashTP Layout Reordering

### Idea

Instead of computing pair geometry once and then expanding back to directed edge
tensors immediately, keep pair-major layout as long as possible and feed the
FlashTP path through a pair-aware gather/scatter adapter.

This means:

- pair-level geometry and filter preparation remain unique
- reverse-direction reuse happens before directed expansion
- expansion is delayed until the last interface point that strictly requires it

### Why this is promising

Current FlashTP-compatible pair execution still expands directed weights and
attributes before entering the fused backend path. This limits how much of the
pair-execution benefit survives on accelerated backends.

### Novelty potential

High relative to other ideas in this note. This is one of the most plausible
places to find a publishable "pair execution + backend co-design" contribution.

## 6. Scalar-Only Distributed Backward/Communication Pruning

### Idea

Use the observation in `pair_e3gnn_parallel.cpp` that many `self_conn_grads`
are effectively zero to introduce a scalar-aware backward plan.

For deployed energy/force/stress inference, the runtime can analyze which
channels actually contribute to the energy scalar and avoid carrying or
communicating known-zero gradient paths across segment boundaries.

### Why this is promising

The current parallel path is structurally correct but conservative. If the
deployed readout only consumes a restricted subset of channels, then the
communication graph for backward-derived force computation may be reducible.

### Novelty potential

High if it can be shown exact. This could become a distinct contribution for
distributed equivariant MLIP inference, especially if combined with pair-aware
segment scheduling.

## 7. Direct Type-Index Embedding Without Explicit One-Hot Materialization

### Idea

Replace explicit one-hot expansion with a direct learned lookup or fused gather
into the first linear projection.

The goal is to avoid writing and reading a dense `(N, num_species)` tensor when
the model only needs the projected hidden feature.

### Why this is promising

The existing TODO in `node_embedding.py` already points toward this. It is even
more attractive in pair-execution settings because the rest of the runtime is
trying to remove redundant memory traffic.

### Novelty potential

Low to moderate. It is probably not a paper claim by itself, but it is a clean
systems optimization that fits the overall direction.

## 8. Deploy-Once Segmented Serialization

### Idea

Refactor parallel deployment so the model is built once, segmented once, and
then serialized into multiple artifacts without rebuilding the original graph of
modules repeatedly.

### Why this is promising

The TODO in `deploy.py` is a strong hint that deployment still pays avoidable
construction cost. This matters more once pair execution and backend policy add
more runtime choices.

### Novelty potential

Low for research, high for usability and repeatability.

## 9. Pair-Aware Inference Streaming for Large Target Sets

### Idea

Build a streamed offline inference pipeline where:

- graph loading and pair-plan preparation happen in a bounded producer queue
- GPU compute consumes prepared batches asynchronously
- results are written by a separate sink

### Why this is promising

`inference.py` still mixes graph loading, optional graph saving, and evaluation
in a straightforward serial flow. For large offline studies, pair metadata
preparation can become a visible preprocessing stage. This is a good place to
hide latency behind compute.

### Novelty potential

Low alone. Useful supporting optimization and useful for benchmarking at scale.

## 10. Pair-Plan Compression for Persistent Storage

### Idea

Do not store every pair-related tensor at full runtime form. Instead, store a
minimal persistent plan:

- pair count
- forward index
- reverse index or singleton bit
- optional packed topology signature

and reconstruct derived tensors lazily.

### Why this is promising

The current runtime is naturally drifting toward more metadata. Persistent pair
plans should be compact enough to live in datasets and deployment artifacts
without inflating file size.

### Novelty potential

Low by itself, but valuable infrastructure if dataset-scale pair execution is
part of future work.

## Priority ranking

The most promising next steps, balancing performance value and research
potential, are:

1. Pair-aware FlashTP layout reordering
2. Scalar-only distributed backward/communication pruning
3. Neighbor-epoch pair plan caching
4. Backend-autotuned pair policy selection
5. Direct type-index embedding without explicit one-hot materialization

The most practical infrastructure improvements are:

1. Packed-key reverse matching
2. Pair-plan serialization in graph datasets
3. Deploy-once segmented serialization
4. Pair-plan compression for persistent storage
5. Pair-aware inference streaming

## Recommended paper-facing directions

If the goal is a publishable systems contribution rather than only engineering
cleanup, the strongest combinations appear to be:

- **Pair execution + backend co-design**
  - Pair-aware FlashTP layout reordering
  - Backend-autotuned policy selection

- **Pair execution + distributed exactness**
  - Neighbor-epoch pair plan caching
  - Scalar-only distributed backward/communication pruning

- **Pair execution + dataflow redesign**
  - Direct type-index embedding
  - Pair-plan serialization and compression

## Recommendation

The next document-worthy step should not be "more TODO cleanup". It should be a
small set of explicit hypotheses:

1. Which metadata can be moved completely out of the hot path
2. Which backend still destroys pair-execution gains by forcing early directed
   expansion
3. Which distributed gradients or messages are structurally unnecessary for the
   deployed energy-only readout path

Those three questions are much more likely to produce meaningful performance
improvements than isolated micro-optimizations.
