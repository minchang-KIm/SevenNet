# E3NN Baseline vs Pair Execution Full Analysis

Run date: `2026-03-30`

## Goal

Evaluate the current pair-execution implementation without FlashTP, comparing:

- `e3nn_baseline`
- `e3nn_pair_full`

The question is whether the current implementation shows a real benefit when the backend remains the original SevenNet e3nn path and the pair policy resolves to `full`.

## Setup

- Model: `7net-omni`
- Path: ASE calculator end-to-end latency
- Datasets:
  - `MPtrj`
  - `sAlex`
  - `OMat24`
  - `OMol25`
  - `OC20`
  - `OC22`
  - `phononDB`
- Sample selection: largest available structure among sampled shards
- Repeat: `3`
- Pair case: explicit `pair_execution_policy='full'`

## Main Findings

The result is mixed, unlike the FlashTP case.

`e3nn_pair_full` can help on larger real graphs, but it is not universally faster than the e3nn baseline.

Steady-state speedup `e3nn_baseline / e3nn_pair_full`:

- `MPtrj`: `1.433x`
- `OC20`: `1.011x`
- `sAlex`: `1.011x`
- `OC22`: `1.007x`
- `OMat24`: `0.876x`
- `OMol25`: `0.781x`
- `phononDB`: `0.774x`

Summary statistics:

- Median speedup: `1.007x`
- Geometric-mean speedup: `0.965x`

This means the current full pair path is not a universal runtime win, but it does show a meaningful gain on the largest tested structure and near break-even behavior on several medium-sized real structures.

## Cold-Call Behavior

Cold-call latency is consistently worse with pair execution.

Representative ratios `pair_full / baseline`:

- `phononDB`: `1.62x`
- `OMol25`: `1.60x`
- `OMat24`: `1.56x`
- `OC20`: `1.56x`
- `OC22`: `1.55x`
- `sAlex`: `1.55x`
- `MPtrj`: `1.09x`

The pair metadata path remains visible on the cold path even without FlashTP.

## Interpretation

The no-Flash result is more favorable to pair execution than the FlashTP result because the current e3nn `full` path actually keeps pair reuse alive through the message path more effectively than the FlashTP `geometry_only` path.

What this means in practice:

- Pair geometry reuse is real.
- Pair-level `weight_nn` reuse is real.
- In the e3nn `full` path, forward and reverse messages are still computed separately, but pair weights are not expanded back into a FlashTP-directed edge path.
- This allows the current implementation to show a genuine benefit on some large graphs.

At the same time, the current implementation still pays:

- pair metadata construction cost
- control-path overhead
- extra indexing and gather logic

So the benefit depends strongly on graph size and workload shape.

## Numerical Agreement

Agreement with the baseline remains good.

- Worst absolute energy delta: `0.0 eV`
- Worst absolute force delta: `9.155e-05 eV/A`

This is consistent with the earlier FlashTP-side experiments: the current pair execution path is numerically stable even when performance benefit is mixed.

## Practical Conclusion

Without FlashTP, the current pair-execution implementation is not just a preprocessing optimization. In the e3nn `full` path it can produce a real steady-state speedup, but only on part of the workload distribution.

The strongest safe claim from this run is:

`Current pair_execution(full) can improve pure e3nn inference on larger real graphs, but the benefit is shape-dependent and not universal across real datasets.`

## Next Step Suggested by These Results

The data suggests a two-part story:

- For pure e3nn, the current full pair path is already worth studying as a size-dependent optimization.
- For FlashTP, the current implementation still needs pair-major backend co-design before it can inherit the same benefit.
