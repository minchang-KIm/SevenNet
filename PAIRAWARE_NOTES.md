# Pair-Aware Geometry Reuse Notes

## System Map
- Training entrypoint: `sevenn/main/sevenn.py` -> `sevenn/scripts/train.py::train_v2` -> `sevenn/model_build.py::build_E3_equivariant_model`
- Inference entrypoint: `sevenn/main/sevenn_inference.py` -> `sevenn/scripts/inference.py::inference` -> `sevenn/util.py::model_from_checkpoint` -> `sevenn/checkpoint.py::SevenNetCheckpoint.build_model`
- Export entrypoint: `sevenn/main/sevenn_get_model.py` -> `sevenn/scripts/deploy.py::{deploy,deploy_parallel}`
- Geometry call chain: `AtomGraphSequential.forward` -> `edge_embedding` -> `EdgeEmbedding.forward` -> `EDGE_LENGTH`, `EDGE_EMBEDDING`, and `EDGE_ATTR` -> `sevenn/nn/convolution.py::IrrepsConvolution.forward`

## Pair Mapping Note
- Pair-aware reuse is implemented in `sevenn/nn/edge_embedding.py`.
- The mapping uses canonical node indices plus a canonicalized edge vector, not just `(min(i, j), max(i, j))`.
- This avoids collapsing distinct periodic images or self-image edges that share the same node pair but differ in displacement.
- The tradeoff is that the pair construction path uses `torch.unique(..., dim=0, return_inverse=True)` for correctness on periodic graphs; the benchmark harness reports geometry time so the overhead can be measured directly.

## Benchmark
- Python harness: `bench/pairaware_bench.py`
- Shell runner: `bench/run_bench.sh`
- Modes exercised by the shell runner:
  - baseline
  - pairaware
  - flashtp
  - flashtp + pairaware
