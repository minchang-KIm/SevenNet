# Pair-Aware Geometry Reuse Notes

## System Map
- Training entrypoint: `sevenn/main/sevenn.py` -> `sevenn/scripts/train.py::train_v2` -> `sevenn/model_build.py::build_E3_equivariant_model`
- Inference entrypoint: `sevenn/main/sevenn_inference.py` -> `sevenn/scripts/inference.py::inference` -> `sevenn/util.py::model_from_checkpoint` -> `sevenn/checkpoint.py::SevenNetCheckpoint.build_model`
- Export entrypoint: `sevenn/main/sevenn_get_model.py` -> `sevenn/scripts/deploy.py::{deploy,deploy_parallel}`
- Geometry call chain: `AtomGraphSequential.forward` -> `edge_embedding` -> `EdgeEmbedding.forward` -> `EDGE_LENGTH`, `EDGE_EMBEDDING`, and `EDGE_ATTR` -> `sevenn/nn/convolution.py::IrrepsConvolution.forward`

## Spec Traceability
| Task section | Implementation | Tests / docs | Status | Notes |
| --- | --- | --- | --- | --- |
| 0. Non-negotiable constraints | `sevenn/nn/edge_embedding.py`, `sevenn/nn/convolution.py` | `tests/unit_tests/test_pairaware.py` | Implemented | Convolution / TP math is unchanged; only geometry/filter preparation is patched. |
| 1. Pair-aware definition | `build_undirected_pair_mapping`, SH parity helpers in `sevenn/nn/edge_embedding.py` | `tests/unit_tests/test_pairaware.py` | Implemented | Uses canonicalized edge vectors in the pair key to keep periodic/self-image edges distinct. |
| 2. Repo recon / system map | This note + runtime mode plumbing in `sevenn/model_build.py` / `sevenn/util.py` | This note | Implemented | Exact call chain is recorded above. |
| 3. `use_pairaware` user option | `_keys.py`, `_const.py`, CLI entrypoints, calculator / checkpoint / deploy plumbing | `tests/unit_tests/test_cli.py` | Implemented | Default is `False`; `--enable_pairaware` is wired for train / inference / get_model. |
| 4. Integration point | `EdgeEmbedding.forward`, `init_edge_embedding`, `build_E3_equivariant_model` | `tests/unit_tests/test_pairaware.py` | Implemented | Output tensor shapes remain baseline-compatible. |
| 5. Memory / performance design | Pair mapping note below, benchmark harness in `bench/pairaware_bench.py` | Bench runs below | Partial | Correctness path is implemented, but CPU performance target is currently not met. |
| 6. Validation plan | Pair-aware unit tests, CLI/export/profile tests, fallback tests | `tests/unit_tests/test_pairaware.py`, `tests/unit_tests/test_cli.py`, `tests/unit_tests/test_model.py` | Implemented | CPU correctness path passed in this workspace. |
| 7. Benchmarking | `bench/pairaware_bench.py`, `bench/run_bench.sh` | Bench runs below | Implemented with findings | Profile path bug fixed; flash/combined runs skip cleanly when unavailable. |
| 8. Self-verification loop | Existing test / bench commands | This note + `CODEX_TASK_Version2.md` known issues | Partial | The loop is executable, but there is no single orchestrator script yet. |
| 9. Docs / usage note | `docs/source/user_guide/cli.md`, `docs/source/user_guide/accelerator.md` | CLI tests + docs | Implemented | Pair-aware composeability with accelerators is documented. |

## Pair Mapping Note
- Pair-aware reuse is implemented in `sevenn/nn/edge_embedding.py`.
- The mapping uses canonical node indices plus a canonicalized edge vector, not just `(min(i, j), max(i, j))`.
- This avoids collapsing distinct periodic images or self-image edges that share the same node pair but differ in displacement.
- The tradeoff is that the pair construction path uses `torch.unique(..., dim=0, return_inverse=True)` for correctness on periodic graphs; the benchmark harness reports geometry time so the overhead can be measured directly.

## Validation Environment
- Validation venv: `/tmp/sevenn-pairaware-venv`
- Install command used: `/tmp/sevenn-pairaware-venv/bin/pip install -e '.[test]'`
- Runtime facts on this workspace:
  - `torch.cuda.is_available() == False`
  - `sevenn.nn.flash_helper.is_flash_available() == False`
  - `sevenn.nn.cue_helper.is_cue_available() == False`
  - `sevenn.nn.oeq_helper.is_oeq_available() == False`

## Executed Validation
- CPU correctness:
  - `python -m pytest -q tests/unit_tests/test_pairaware.py` -> `5 passed`
  - `python -m pytest -q tests/unit_tests/test_model.py -k 'fallback_when_unavailable'` -> `3 passed`
- CLI / export / profile:
  - `python -m pytest -q tests/unit_tests/test_cli.py -k 'test_inference or test_sevenn_preset_pairaware or test_get_model_serial_pairaware_runtime'` -> `9 passed`
  - `bench/pairaware_bench.py --enable_pairaware --profile` generated `/tmp/pairaware-profiles/pairaware-1773898625.json`
  - `sevenn_inference ... --enable_pairaware --profile --output /tmp/sevenn-inference-profile-run3` generated `inference_profile.json` plus CSV outputs
- Accelerator matrix:
  - `tests/unit_tests/test_flash.py` -> `10 skipped`
  - `tests/unit_tests/test_cueq.py` -> `13 skipped`
  - `tests/unit_tests/test_oeq.py` -> `10 skipped`
  - Skip reason in all cases: accelerator unavailable on this machine

## Benchmark Results
- Harness files:
- Python harness: `bench/pairaware_bench.py`
- Shell runner: `bench/run_bench.sh`
- Modes exercised by the shell runner:
  - baseline
  - pairaware
  - flashtp
  - flashtp + pairaware
- CPU benchmark snapshot on this workspace:

| target_atoms | mode | avg_step_ms | avg_geometry_ms | avg_tensor_product_ms | reuse_factor | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 256 | baseline | 670.279 | 67.588 | 113.800 | 1.000 | shell runner |
| 256 | pairaware | 688.501 | 78.585 | 121.407 | 2.000 | shell runner |
| 2000 | baseline | 3150.195 | 65.571 | 596.047 | 1.000 | shell runner |
| 2000 | pairaware | 3181.347 | 123.951 | 593.622 | 2.000 | shell runner |
| 21296 | baseline | 40861.381 | 53.144 | 7891.892 | 1.000 | direct run |
| 21296 | pairaware | 40213.184 | 773.454 | 6496.268 | 2.000 | direct run |

## Findings
- Pair-aware correctness is validated on CPU: mapping, SH parity, full-model equivalence, CLI flag propagation, TorchScript export metadata, and inference/profile output all passed.
- The benchmark harness had a real bug before validation: profile trace naming referenced `runtime_config`, which did not exist. This was fixed in `bench/pairaware_bench.py`, and the harness now also validates requested `pairaware` mode activation explicitly.
- CPU performance is currently the main blocker. On this machine, pair-aware reduces the pair count exactly as expected, but the measured geometry phase is slower than baseline at every tested size.
- The most likely cause is the correctness-oriented pair construction path built on `torch.unique(..., dim=0, return_inverse=True)` in `sevenn/nn/edge_embedding.py`. The correctness story is strong; the current bottleneck is the pair construction overhead.
