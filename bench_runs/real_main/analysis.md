# FlashTP + Pair Execution Real-Dataset Analysis

Run date: `2026-03-27`

## Goal

Evaluate whether the current `pair_execution` implementation improves real-data inference when combined with `FlashTP`, using the current branch and the recommended `7net-omni` checkpoint family.

Compared cases:

- `e3nn_baseline`
- `flash_baseline`
- `flash_pair_auto`

All measurements use the ASE calculator path and repeated calls on fixed topology so that the current topology-cache behavior is reflected in the numbers.

## Dataset Set

The run covers public real datasets drawn from the current SevenNet-Omni paper ecosystem:

- `MPtrj` validation
- `sAlex` validation
- `OMat24` validation
- `OMol25` neutral validation
- `OC20` S2EF validation (in-domain)
- `OC22` S2EF validation (in-domain)
- `phononDB` benchmark structures

For each dataset, the largest structure among the sampled shard(s) was selected to emphasize the slow-path regime.

## Main Findings

`FlashTP` alone is consistently beneficial. Against `e3nn_baseline`, steady-state speedups range from about `1.41x` to `12.79x`.

`FlashTP + pair execution(auto)` does not currently deliver a meaningful steady-state win over `FlashTP` on these real datasets.

- Best case: `OC22`, `1.006x` over `flash_baseline`
- Near break-even: `phononDB`, `0.996x`
- Regressions: `MPtrj 0.926x`, `OMat24 0.920x`, `OMol25 0.894x`

Cold-call latency is much worse with pair execution than with FlashTP alone, because pair metadata construction and cache setup remain on the critical path.

Representative cold-call overhead:

- `MPtrj`: `78.2 ms -> 279.4 ms`
- `OC20`: `66.8 ms -> 172.4 ms`
- `OC22`: `65.5 ms -> 147.7 ms`

Numerical agreement remains good. Worst observed deviations versus e3nn were:

- Energy: `0.0 eV` in this run
- Force: `7.63e-05 eV/A`

## Interpretation

The current branch resolves `flash_pair_auto` to `geometry_only` on FlashTP, which is the correct policy under the present heuristic. Real-data results indicate that this policy is still not enough to beat FlashTP itself in steady state.

This matches the branch research notes:

- Pair geometry reuse is present.
- Pair metadata generation is still expensive.
- FlashTP still prefers directed-edge layout deeply enough that pair-level reuse does not survive as an end-to-end speedup.

In short, the branch is now functionally correct for `FlashTP + pair execution`, but it is not yet a positive performance result on real datasets.

## Code Fixes Required For Correctness

Two correctness bugs were fixed before running the benchmarks:

1. `sevenn/checkpoint.py`

Backend override (`enable_flash=True`) used to resolve pair policy before the backend flags were updated in config. Old checkpoints therefore kept `auto -> full` instead of the intended `auto -> geometry_only`.

2. `sevenn/calculator.py`

The checkpoint calculator path reused the original checkpoint config after model conversion and re-resolved pair policy from stale backend flags. This caused the calculator path to disagree with the built model path on old checkpoints.

Both fixes are covered by regression tests in `tests/unit_tests/test_pair_execution.py`.

## What The Data Suggests Next

If the goal is a publishable performance result, the next work should focus on:

- Pair-aware FlashTP layout retention instead of early expansion back to directed-edge tensors
- Lower-overhead pair-plan construction and cache validation
- Separating cold-start and steady-state stories explicitly in the paper
- Backend autotuning instead of assuming pair execution should always help FlashTP

At the moment, the strongest paper-safe claim is:

`FlashTP + current pair_execution is numerically correct on real datasets, but the present implementation does not yet produce a consistent end-to-end speedup over FlashTP alone.`
