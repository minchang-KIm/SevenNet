# Pair Execution All-Public Local Benchmark Report

Run root: `bench_runs/all_public_local_pair_main`

## 2026-04-03 Correction Note

- 이 문서는 원래 `all_public_local_pair_main` raw sweep 결과를 정리한 내부 보고서다.
- 이후 stable recheck에서 small-graph case 일부에 warm-up artifact가 있었음이 확인되었다.
- 따라서 이 문서의 headline raw speedup은 외부 발표나 논문용 canonical 수치로 직접 사용하면 안 된다.
- 현재 canonical 해석은 아래 문서를 따른다.
  - `docs/papers/icpp_pair_execution/00_current_status_report.md`
  - `docs/papers/icpp_pair_execution/03_final_manuscript.md`
- 특히 여기 기록된 `qm9_hf 3.737x`는 stable recheck로 invalidated되었다.

## Download Status

- Public downloadable datasets were fetched into `datasets/raw`.
- The only remaining non-local entry is `omol25_official_gated`, which requires gated Hugging Face access.

## Coverage

- Inventory entries: `40`
- Benchmarked: `31`
- Skipped: `9`

Benchmarked set includes:

- `qm9_hf`
- `rmd17`
- `iso17`
- `md22_at_at`
- `md22_at_at_cg_cg`
- `md22_ac_ala3_nhme`
- `md22_dha`
- `md22_double_walled_nanotube`
- `md22_stachyose`
- `md22_buckyball_catcher`
- `ani1x`
- `ani1ccx`
- `spice_2023`
- `mptrj`
- `wbm_initial`
- `phonondb_pbe`
- `omat24_1m_official`
- `salex_train_official`
- `salex_val_official`
- `omol25_train_4m`
- `omol25_validation`
- `oc20_s2ef_train_2m`
- `oc20_s2ef_val_id`
- `oc20_s2ef_val_ood_ads`
- `oc20_s2ef_val_ood_cat`
- `oc20_s2ef_val_ood_both`
- `oc22_s2ef_train`
- `oc22_s2ef_val_id`
- `oc22_s2ef_val_ood`
- `oc20_s2ef_train_20m`
- `omol25_train_neutral`

Skipped set:

- `md17_aspirin`
- `md17_benzene`
- `md17_ethanol`
- `md17_malonaldehyde`
- `md17_naphthalene`
- `md17_salicylic_acid`
- `md17_toluene`
- `md17_uracil`
- `omol25_official_gated`

## Raw Sweep Summary

- Raw repeated-window best speedup: `qm9_hf` at `3.737x`
- Worst steady-state speedup: `iso17` at `0.730x`
- Geometric-mean speedup: `0.942x`
- Median speedup: `1.006x`
- Wins over baseline: `19 / 31`
- Worst force delta: `2.441e-04 eV/A`
- Worst energy delta: `6.104e-05 eV`

Raw top wins in the original sweep:

- `qm9_hf`: `3.737x`
- `oc20_s2ef_train_20m`: `1.034x`
- `oc20_s2ef_val_ood_ads`: `1.028x`
- `md22_buckyball_catcher`: `1.023x`
- `salex_train_official`: `1.018x`
- `omol25_train_neutral`: `1.017x`

Largest losses:

- `iso17`: `0.730x`
- `wbm_initial`: `0.748x`
- `md22_dha`: `0.751x`
- `ani1ccx`: `0.751x`
- `ani1x`: `0.753x`
- `md22_ac_ala3_nhme`: `0.754x`

## Interpretation

- The current implementation still leaves tensor-product and scatter work edge-major.
- Gains therefore come from reduced geometry-side work and are strongest when edge-side reuse is large enough to offset extra indexing/control overhead.
- After warm-up correction, the strongest trustworthy gains are on larger periodic workloads such as `OC20`, `sAlex`, `OMat24`, and `MPtrj`.
- Most `spice`-family molecular datasets remain slower after stable recheck.
- `OMat24` and `sAlex` official tarball datasets should be interpreted as modest wins, not dramatic gains.

Representative stable recheck cases:

- `qm9_hf`: baseline `28.6 ms`, pair `47.1 ms`
- `iso17`: baseline `29.0 ms`, pair `47.2 ms`
- `salex_train_official`: baseline `151.6 ms`, pair `147.4 ms`
- `oc20_s2ef_train_20m`: baseline `125.3 ms`, pair `118.6 ms`
- `omat24_1m_official`: baseline `232.1 ms`, pair `229.0 ms`
- `mptrj`: baseline `424.7 ms`, pair `419.9 ms`

Modal summary:

- `qcml`: raw sweep shows a strong win, but this is invalidated by stable recheck
- `oc20`: consistent small wins
- `oc22`: consistent small wins
- `omat24`: consistent small wins
- `omol25_low`: near break-even to small wins
- `spice`: mostly losses
- `matpes_*`: losses on the sampled representatives

## Artifacts

Logs and tables:

- `bench_runs/all_public_local_pair_main/run.log`
- `bench_runs/all_public_local_pair_main/summary.md`
- `bench_runs/all_public_local_pair_main/analysis.md`
- `bench_runs/all_public_local_pair_main/metrics/datasets.csv`
- `bench_runs/all_public_local_pair_main/metrics/samples.csv`
- `bench_runs/all_public_local_pair_main/metrics/aggregated.csv`
- `bench_runs/all_public_local_pair_main/metrics/skipped.csv`
- `bench_runs/all_public_local_pair_main/metrics/profiles.csv`

Plots:

- `bench_runs/all_public_local_pair_main/plots/steady_state_latency_all.png`
- `bench_runs/all_public_local_pair_main/plots/cold_latency_all.png`
- `bench_runs/all_public_local_pair_main/plots/speedup_by_dataset.png`
- `bench_runs/all_public_local_pair_main/plots/speedup_vs_natoms_all.png`
- `bench_runs/all_public_local_pair_main/plots/speedup_vs_num_edges_all.png`
- `bench_runs/all_public_local_pair_main/plots/support_status.png`
- `bench_runs/all_public_local_pair_main/plots/stage_breakdown_top.png`

For final writing, use this document together with the corrected manuscript package in `docs/papers/icpp_pair_execution/`.

## Notes

- The `MD17` mirrors skipped here are graph-only parquet mirrors or missing raw parquet, so they cannot be reconstructed into ASE `Atoms` with the local cache alone.
- `omol25_official_gated` remains inaccessible without external authentication.
