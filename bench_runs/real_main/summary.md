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
- `oc20_val_id`: modal=`oc20`, samples=1, max_atoms=225, source=https://huggingface.co/datasets/colabfit/OC20_S2EF_val_id
- `oc22_val_id`: modal=`oc22`, samples=1, max_atoms=200, source=https://huggingface.co/datasets/colabfit/OC22-S2EF-Validation-in-domain
- `phonondb_pbe`: modal=`matpes_r2scan`, samples=1, max_atoms=8, source=https://figshare.com/articles/dataset/Matbench_Discovery_-_Data_Files/22715158

## Aggregated Results

| dataset | case | modal | mean_atoms | max_atoms | cold_ms | steady_median_ms | steady_p95_ms | max_abs_force_diff_vs_e3nn | abs_energy_diff_vs_e3nn | resolved_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mptrj_val | e3nn_baseline | mpa | 426.0 | 426 | 706.8961780169047 | 643.3057899703272 | 730.5327045498416 | 0.0 | 0.0 | baseline |
| mptrj_val | flash_baseline | mpa | 426.0 | 426 | 78.19780497811735 | 50.31322204740718 | 89.15731199667789 | 1.2367963790893555e-06 | 0.0 | baseline |
| mptrj_val | flash_pair_auto | mpa | 426.0 | 426 | 279.4098330195993 | 54.30835398146883 | 341.5229213074781 | 2.682209014892578e-06 | 0.0 | geometry_only |
| oc20_val_id | e3nn_baseline | oc20 | 225.0 | 225 | 303.89614595333114 | 238.60375297954306 | 320.2725159819238 | 0.0 | 0.0 | baseline |
| oc20_val_id | flash_baseline | oc20 | 225.0 | 225 | 66.80112396134064 | 49.031724978704005 | 87.58150420035236 | 7.748603820800781e-07 | 0.0 | baseline |
| oc20_val_id | flash_pair_auto | oc20 | 225.0 | 225 | 172.4311810103245 | 50.282051030080765 | 81.26479396596551 | 2.0563602447509766e-06 | 0.0 | geometry_only |
| oc22_val_id | e3nn_baseline | oc22 | 200.0 | 200 | 258.48595099523664 | 187.39757494768128 | 274.6452138875611 | 0.0 | 0.0 | baseline |
| oc22_val_id | flash_baseline | oc22 | 200.0 | 200 | 65.49682904733345 | 51.702452008612454 | 81.39159078709781 | 1.1444091796875e-05 | 0.0 | baseline |
| oc22_val_id | flash_pair_auto | oc22 | 200.0 | 200 | 147.6954209501855 | 51.4173400006257 | 83.29385140095837 | 2.6702880859375e-05 | 0.0 | geometry_only |
| omat24_validation | e3nn_baseline | omat24 | 112.0 | 112 | 157.06930600572377 | 76.46614097757265 | 173.2313536980655 | 0.0 | 0.0 | baseline |
| omat24_validation | flash_baseline | omat24 | 112.0 | 112 | 60.01113500678912 | 44.289599056355655 | 78.00841222633608 | 3.0517578125e-05 | 0.0 | baseline |
| omat24_validation | flash_pair_auto | omat24 | 112.0 | 112 | 92.39653497934341 | 48.1418400304392 | 79.23312662169337 | 7.62939453125e-05 | 0.0 | geometry_only |
| omol25_validation | e3nn_baseline | omol25_low | 110.0 | 110 | 147.01855299063027 | 65.10662398068234 | 163.17724524997175 | 0.0 | 0.0 | baseline |
| omol25_validation | flash_baseline | omol25_low | 110.0 | 110 | 59.46823500562459 | 43.129233992658556 | 77.16051366878673 | 1.9073486328125e-06 | 0.0 | baseline |
| omol25_validation | flash_pair_auto | omol25_low | 110.0 | 110 | 86.88228798564523 | 48.2299430295825 | 78.60289821401238 | 2.1457672119140625e-06 | 0.0 | geometry_only |
| phonondb_pbe | e3nn_baseline | matpes_r2scan | 8.0 | 8 | 124.53128199558705 | 62.24905594717711 | 148.7440770957619 | 0.0 | 0.0 | baseline |
| phonondb_pbe | flash_baseline | matpes_r2scan | 8.0 | 8 | 60.430155019275844 | 44.1650579450652 | 79.20356368413195 | 1.0337680578231812e-07 | 0.0 | baseline |
| phonondb_pbe | flash_pair_auto | matpes_r2scan | 8.0 | 8 | 61.39709200942889 | 44.33786001754925 | 78.15445908927359 | 1.0291114449501038e-07 | 0.0 | geometry_only |
| salex_validation | e3nn_baseline | mpa | 168.0 | 168 | 297.79288900317624 | 232.2453489759937 | 642.9500071157236 | 0.0 | 0.0 | baseline |
| salex_validation | flash_baseline | mpa | 168.0 | 168 | 65.79855305608362 | 48.63392800325528 | 79.03051627799869 | 1.253560185432434e-06 | 0.0 | baseline |
| salex_validation | flash_pair_auto | mpa | 168.0 | 168 | 169.7886959882453 | 49.52805500943214 | 82.40431696758606 | 1.3187527656555176e-06 | 0.0 | geometry_only |

## Highlights

- Best `flash_pair_auto` steady-state speedup over `flash_baseline`: `oc22_val_id` at `1.006x`.
- Worst absolute energy delta vs e3nn baseline: `0.000e+00` eV.
- Worst absolute force delta vs e3nn baseline: `7.629e-05` eV/A.
