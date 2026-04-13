# Local Pair Size Validation + Profiling

- Datasets: mptrj, md22_double_walled_nanotube, spice_2023, md22_stachyose, ani1x, rmd17, iso17
- Top-k per dataset: `3`
- Repeat: `3`
- Spearman(speedup, natoms): `0.541`
- Spearman(speedup, num_edges): `0.487`

## Outputs

- `metrics/datasets.csv`
- `metrics/samples.csv`
- `metrics/profiles.csv`
- `plots/speedup_vs_natoms.png`
- `plots/speedup_vs_num_edges.png`
- `plots/stage_breakdown.png`
- `analysis.md`
