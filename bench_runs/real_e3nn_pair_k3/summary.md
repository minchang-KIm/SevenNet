# Real Dataset Benchmark

## Scope

- Model: `7net-omni`
- Cases: `e3nn_baseline`, `e3nn_pair_full`
- Measurement: ASE calculator end-to-end latency with repeated calls on identical topology

## Dataset Coverage

- `mptrj_val`: modal=`mpa`, samples=3, max_atoms=426, source=https://huggingface.co/datasets/nimashoghi/mptrj
- `salex_validation`: modal=`mpa`, samples=3, max_atoms=168, source=https://huggingface.co/datasets/colabfit/sAlex_validation
- `omat24_validation`: modal=`omat24`, samples=3, max_atoms=112, source=https://huggingface.co/datasets/colabfit/OMat24_validation_rattled_1000_subsampled
- `omol25_validation`: modal=`omol25_low`, samples=3, max_atoms=110, source=https://huggingface.co/datasets/colabfit/OMol25_neutral_validation
- `oc20_val_id`: modal=`oc20`, samples=3, max_atoms=225, source=https://huggingface.co/datasets/colabfit/OC20_S2EF_val_id
- `oc22_val_id`: modal=`oc22`, samples=3, max_atoms=200, source=https://huggingface.co/datasets/colabfit/OC22-S2EF-Validation-in-domain
- `phonondb_pbe`: modal=`matpes_r2scan`, samples=3, max_atoms=8, source=https://figshare.com/articles/dataset/Matbench_Discovery_-_Data_Files/22715158

## Aggregated Results

| dataset | case | modal | mean_atoms | max_atoms | cold_ms | steady_median_ms | steady_p95_ms | max_abs_force_diff_vs_e3nn | abs_energy_diff_vs_e3nn | resolved_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mptrj_val | e3nn_baseline | mpa | 348.6666666666667 | 426 | 349.41646800143644 | 515.3394140070304 | 719.3062385020312 | 0.0 | 0.0 | baseline |
| mptrj_val | e3nn_pair_full | mpa | 348.6666666666667 | 426 | 543.8376889796928 | 308.75415552873164 | 328.6940492806025 | 3.6656856536865234e-06 | 0.0 | full |
| oc20_val_id | e3nn_baseline | oc20 | 225.0 | 225 | 303.73923399019986 | 284.92171550169587 | 327.1683118218789 | 0.0 | 0.0 | baseline |
| oc20_val_id | e3nn_pair_full | oc20 | 225.0 | 225 | 476.80337098427117 | 258.7775445135776 | 280.19059881044086 | 1.9222497940063477e-06 | 6.103515625e-05 | full |
| oc22_val_id | e3nn_baseline | oc22 | 200.0 | 200 | 255.29785599792376 | 233.41891201562248 | 278.3755419164663 | 0.0 | 0.0 | baseline |
| oc22_val_id | e3nn_pair_full | oc22 | 200.0 | 200 | 399.09968100255355 | 206.78188200690784 | 229.48913582076784 | 0.00018310546875 | 0.0 | full |
| omat24_validation | e3nn_baseline | omat24 | 110.0 | 112 | 158.03992102155462 | 131.5071654971689 | 181.02823492372409 | 0.0 | 0.0 | baseline |
| omat24_validation | e3nn_pair_full | omat24 | 110.0 | 112 | 248.31924500176683 | 110.05258598015644 | 129.24117546353955 | 9.1552734375e-05 | 3.0517578125e-05 | full |
| omol25_validation | e3nn_baseline | omol25_low | 110.0 | 110 | 142.13691296754405 | 114.53582299873233 | 164.45164821343496 | 0.0 | 0.0 | baseline |
| omol25_validation | e3nn_pair_full | omol25_low | 110.0 | 110 | 229.16284098755568 | 101.67551849735901 | 119.56202985020354 | 2.1457672119140625e-06 | 0.0 | full |
| phonondb_pbe | e3nn_baseline | matpes_r2scan | 8.0 | 8 | 122.42903199512511 | 108.29748050309718 | 152.46547596761957 | 0.0 | 0.0 | baseline |
| phonondb_pbe | e3nn_pair_full | matpes_r2scan | 8.0 | 8 | 199.46617598179728 | 98.0376125080511 | 116.07482383551542 | 2.150142250911813e-07 | 0.0 | full |
| salex_validation | e3nn_baseline | mpa | 164.0 | 168 | 299.12276001414284 | 279.9887180153746 | 321.51811102812644 | 0.0 | 0.0 | baseline |
| salex_validation | e3nn_pair_full | mpa | 164.0 | 168 | 467.54201501607895 | 255.16323451302014 | 276.85315934650134 | 3.6116689443588257e-06 | 0.0 | full |
- Worst absolute energy delta vs e3nn baseline: `6.104e-05` eV.
- Worst absolute force delta vs e3nn baseline: `1.831e-04` eV/A.
