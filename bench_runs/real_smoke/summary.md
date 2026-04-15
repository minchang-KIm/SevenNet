# Real Dataset FlashTP + Pair-Execution Benchmark

## Scope

- Model: `7net-omni`
- Cases: `e3nn_baseline`, `flash_baseline`, `flash_pair_auto`
- Measurement: ASE calculator end-to-end latency with repeated calls on identical topology

## Dataset Coverage

- `mptrj_val`: modal=`mpa`, samples=1, max_atoms=426, source=https://huggingface.co/datasets/nimashoghi/mptrj
- `salex_validation`: modal=`mpa`, samples=1, max_atoms=168, source=https://huggingface.co/datasets/colabfit/sAlex_validation
- `omat24_validation`: modal=`omat24`, samples=1, max_atoms=112, source=https://huggingface.co/datasets/colabfit/OMat24_validation_rattled_1000_subsampled
- `omol25_validation`: modal=`omol25_low`, samples=1, max_atoms=110, source=https://huggingface.co/datasets/colabfit/OMol25_neutral_validation
- `phonondb_pbe`: modal=`matpes_r2scan`, samples=1, max_atoms=8, source=https://figshare.com/articles/dataset/Matbench_Discovery_-_Data_Files/22715158

## Aggregated Results

| dataset | case | modal | mean_atoms | max_atoms | cold_ms | steady_median_ms | steady_p95_ms | max_abs_force_diff_vs_e3nn | abs_energy_diff_vs_e3nn | resolved_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mptrj_val | e3nn_baseline | mpa | 426.0 | 426 | 694.0630670287646 | 695.051066490123 | 736.8291299295379 | 0.0 | 0.0 | baseline |
| mptrj_val | flash_baseline | mpa | 426.0 | 426 | 78.91314703738317 | 72.3988029640168 | 91.98956016916782 | 1.214444637298584e-06 | 0.0 | baseline |
| mptrj_val | flash_pair_auto | mpa | 426.0 | 426 | 285.95778497401625 | 213.6395894922316 | 357.3531712463591 | 2.9802322387695312e-06 | 0.0 | geometry_only |
| omat24_validation | e3nn_baseline | omat24 | 112.0 | 112 | 158.28342601889744 | 131.15048749023117 | 180.3409392159665 | 0.0 | 0.0 | baseline |
| omat24_validation | flash_baseline | omat24 | 112.0 | 112 | 60.34025002736598 | 63.52818102459423 | 80.50378661428113 | 3.0517578125e-05 | 0.0 | baseline |
| omat24_validation | flash_pair_auto | omat24 | 112.0 | 112 | 93.90304295811802 | 66.13286098581739 | 82.29970952088479 | 4.57763671875e-05 | 3.0517578125e-05 | geometry_only |
| omol25_validation | e3nn_baseline | omol25_low | 110.0 | 110 | 147.71132299210876 | 120.56403551832773 | 170.40635544981342 | 0.0 | 0.0 | baseline |
| omol25_validation | flash_baseline | omol25_low | 110.0 | 110 | 60.06487796548754 | 62.69206551951356 | 80.05676497414242 | 1.9073486328125e-06 | 0.0 | baseline |
| omol25_validation | flash_pair_auto | omol25_low | 110.0 | 110 | 88.26830104226246 | 65.40828451397829 | 81.00448574696202 | 2.384185791015625e-06 | 0.0 | geometry_only |
| phonondb_pbe | e3nn_baseline | matpes_r2scan | 8.0 | 8 | 121.5712430421263 | 107.51832701498643 | 151.33259589783847 | 0.0 | 0.0 | baseline |
| phonondb_pbe | flash_baseline | matpes_r2scan | 8.0 | 8 | 58.26765799429268 | 62.23670899635181 | 79.44504470215179 | 7.561307313608268e-08 | 0.0 | baseline |
| phonondb_pbe | flash_pair_auto | matpes_r2scan | 8.0 | 8 | 60.64770795637742 | 62.4795829935465 | 79.63313748768996 | 1.0989606380462646e-07 | 0.0 | geometry_only |
| salex_validation | e3nn_baseline | mpa | 168.0 | 168 | 298.83005999727175 | 462.2376959596295 | 669.0079493651865 | 0.0 | 0.0 | baseline |
| salex_validation | flash_baseline | mpa | 168.0 | 168 | 66.43909704871476 | 67.18292948789895 | 81.82821865775622 | 8.344650268554688e-07 | 0.0 | baseline |
| salex_validation | flash_pair_auto | mpa | 168.0 | 168 | 172.9237930267118 | 68.84431251091883 | 86.180928372778 | 1.3746321201324463e-06 | 0.0 | geometry_only |

## Highlights

- Best `flash_pair_auto` steady-state speedup over `flash_baseline`: `phonondb_pbe` at `0.996x`.
- Worst absolute energy delta vs e3nn baseline: `3.052e-05` eV.
- Worst absolute force delta vs e3nn baseline: `4.578e-05` eV/A.
