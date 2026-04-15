# Restored Full Two-Pass Win Recheck

## Configuration

- comparison: `SevenNet baseline` vs `pair full restored two-pass`
- warmup: `3`
- repeat: `30`
- datasets: `31`
- note: this restores the pre-`7be5a2b` two-pass forward/reverse full path and avoids the `torch.cat` single-pass regression.

## Headline

- median speedup: `1.009916x`
- geometric mean speedup: `0.858294x`
- wins: `18`
- losses: `13`

## Condition Summary

| group | count | wins | win_rate | mean_speedup | median_speedup | std_speedup |
| --- | --- | --- | --- | --- | --- | --- |
| num_edges >= 3000 | 20 | 18 | 0.900 | 1.021 | 1.014 | 0.021 |
| num_edges < 3000 | 11 | 0 | 0.000 | 0.627 | 0.613 | 0.030 |
| avg_neighbors_directed >= 40 | 17 | 16 | 0.941 | 1.017 | 1.013 | 0.016 |
| avg_neighbors_directed < 40 | 14 | 2 | 0.143 | 0.716 | 0.620 | 0.173 |
| large_dense | 17 | 16 | 0.941 | 1.017 | 1.013 | 0.016 |
| large_sparse | 3 | 2 | 0.667 | 1.042 | 1.053 | 0.031 |
| small_sparse | 11 | 0 | 0.000 | 0.627 | 0.613 | 0.030 |

## Top Wins

| dataset | baseline_mean_ms | proposal_mean_ms | speedup_baseline_over_proposal | num_edges | avg_neighbors_directed | density_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| oc20_s2ef_val_ood_ads | 102.78 | 95.76 | 1.073x | 6686 | 29.71555555555556 | large_sparse |
| md22_buckyball_catcher | 104.09 | 97.18 | 1.071x | 6872 | 46.43243243243244 | large_dense |
| oc20_s2ef_train_20m | 125.60 | 119.29 | 1.053x | 8260 | 36.71111111111111 | large_sparse |
| salex_train_official | 151.94 | 147.88 | 1.027x | 9932 | 75.24242424242425 | large_dense |
| omol25_train_neutral | 150.32 | 146.39 | 1.027x | 9926 | 54.83977900552486 | large_dense |
| oc20_s2ef_val_ood_cat | 156.13 | 152.63 | 1.023x | 10270 | 45.644444444444446 | large_dense |
| oc20_s2ef_train_2m | 163.55 | 160.44 | 1.019x | 10738 | 47.72444444444445 | large_dense |
| oc20_s2ef_val_ood_both | 174.43 | 171.75 | 1.016x | 11486 | 51.506726457399104 | large_dense |
| salex_val_official | 201.41 | 198.56 | 1.014x | 13088 | 102.25 | large_dense |
| oc22_s2ef_val_id | 179.19 | 176.72 | 1.014x | 11704 | 58.52 | large_dense |

## Top Losses

| dataset | baseline_mean_ms | proposal_mean_ms | speedup_baseline_over_proposal | num_edges | avg_neighbors_directed | density_bucket |
| --- | --- | --- | --- | --- | --- | --- |
| md22_at_at | 29.44 | 48.90 | 0.602x | 1312 | 21.866666666666667 | small_sparse |
| md22_ac_ala3_nhme | 29.39 | 48.55 | 0.605x | 1142 | 27.19047619047619 | small_sparse |
| md22_dha | 29.42 | 48.50 | 0.607x | 1406 | 25.107142857142858 | small_sparse |
| qm9_hf | 29.59 | 48.50 | 0.610x | 570 | 19.655172413793103 | small_sparse |
| iso17 | 29.44 | 48.12 | 0.612x | 340 | 17.894736842105264 | small_sparse |
| rmd17 | 29.55 | 48.17 | 0.613x | 366 | 15.25 | small_sparse |
| phonondb_pbe | 29.52 | 47.98 | 0.615x | 208 | 26.0 | small_sparse |
| spice_2023 | 30.96 | 49.49 | 0.626x | 728 | 8.179775280898877 | small_sparse |
| ani1x | 31.09 | 48.30 | 0.644x | 2008 | 31.873015873015877 | small_sparse |
| ani1ccx | 31.47 | 47.75 | 0.659x | 2042 | 37.127272727272725 | small_sparse |

## Interpretation

- The win is recovered only when the full path uses the old two-pass forward/reverse schedule.
- The previous single-pass `torch.cat` full path is a regression for the large/dense cases that previously carried the paper claim.
- This result should be described as a pair-aware full execution/scheduling result, not as pure `geometry_only` reuse.
- The clean paper split is: `geometry_only` proves exact geometry reuse and accuracy preservation; `restored full two-pass` demonstrates the performance condition for large/dense workloads.
