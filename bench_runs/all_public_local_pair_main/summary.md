# All Public Local Pair Benchmark

- Total inventory entries: `40`
- Supported and benchmarked: `31`
- Skipped: `9`

- Best steady-state speedup: `qm9_hf` at `3.737x`.
- Worst steady-state speedup: `iso17` at `0.730x`.
- Worst force delta vs baseline: `2.441e-04` eV/A.
- Worst energy delta vs baseline: `6.104e-05` eV.

## Skip Reasons

- `graph-only parquet; atomic positions unavailable`: `7` datasets
- `no parquet files found`: `1` datasets
- `gated dataset`: `1` datasets

## Additional Outputs

- `metrics/profiles.csv`
- `plots/stage_breakdown_top.png`

## Files

- `metrics/datasets.csv`
- `metrics/samples.csv`
- `metrics/aggregated.csv`
- `metrics/skipped.csv`
- `plots/steady_state_latency_all.png`
- `plots/cold_latency_all.png`
- `plots/speedup_by_dataset.png`
- `plots/speedup_vs_natoms_all.png`
- `plots/speedup_vs_num_edges_all.png`
- `plots/support_status.png`
