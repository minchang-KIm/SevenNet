# Detailed Model Stage Profiling

- Datasets: mptrj, md22_double_walled_nanotube, spice_2023, md22_stachyose, ani1x, rmd17, iso17
- Cases: baseline, pair_full
- Repeat: `1`

## Outputs

- `metrics/summary.csv`
- `metrics/stage_breakdown_long.csv`
- `metrics/stage_breakdown_aggregate.csv`

## Notes

- This profiler is intrusive: it wraps model internals and synchronizes around each stage.
- Use it for stage decomposition, not for absolute end-to-end latency claims.
- The aggregate table sums across all five convolution blocks for shared stage names.
