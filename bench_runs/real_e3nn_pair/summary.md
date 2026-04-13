# Real Dataset Benchmark

## Scope

- Model: `7net-omni`
- Cases: `e3nn_baseline`, `e3nn_pair_full`
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
| mptrj_val | e3nn_baseline | mpa | 426.0 | 426 | 706.7059029941447 | 646.2135030305944 | 732.7476308913901 | 0.0 | 0.0 | baseline |
| mptrj_val | e3nn_pair_full | mpa | 426.0 | 426 | 769.9405699968338 | 450.8934349869378 | 742.6138783979695 | 3.039836883544922e-06 | 0.0 | full |
| oc20_val_id | e3nn_baseline | oc20 | 225.0 | 225 | 303.65604296093807 | 238.19083702983335 | 320.12763438397087 | 0.0 | 0.0 | baseline |
| oc20_val_id | e3nn_pair_full | oc20 | 225.0 | 225 | 472.56828600075096 | 235.64847902162 | 278.26867470867 | 1.8030405044555664e-06 | 0.0 | full |
| oc22_val_id | e3nn_baseline | oc22 | 200.0 | 200 | 259.03229997493327 | 187.17044399818406 | 276.58509481698275 | 0.0 | 0.0 | baseline |
| oc22_val_id | e3nn_pair_full | oc22 | 200.0 | 200 | 402.2694800514728 | 185.82315900130197 | 229.9365239974577 | 7.62939453125e-06 | 0.0 | full |
| omat24_validation | e3nn_baseline | omat24 | 112.0 | 112 | 157.38854795927182 | 76.70656201662496 | 174.1433726856485 | 0.0 | 0.0 | baseline |
| omat24_validation | e3nn_pair_full | omat24 | 112.0 | 112 | 245.95615902217105 | 87.55792601732537 | 125.0438989198301 | 9.1552734375e-05 | 0.0 | full |
| omol25_validation | e3nn_baseline | omol25_low | 110.0 | 110 | 147.45986502384767 | 65.29819499701262 | 163.45628462149762 | 0.0 | 0.0 | baseline |
| omol25_validation | e3nn_pair_full | omol25_low | 110.0 | 110 | 235.33572000451386 | 83.58427201164886 | 119.07523649861105 | 1.7881393432617188e-06 | 0.0 | full |
| phonondb_pbe | e3nn_baseline | matpes_r2scan | 8.0 | 8 | 122.31929402332753 | 59.92185400100425 | 147.89055218570866 | 0.0 | 0.0 | baseline |
| phonondb_pbe | e3nn_pair_full | matpes_r2scan | 8.0 | 8 | 198.00697499886155 | 77.37443299265578 | 112.76947639416903 | 8.638016879558563e-08 | 0.0 | full |
| salex_validation | e3nn_baseline | mpa | 168.0 | 168 | 298.9741599885747 | 232.56391001632437 | 646.462395391427 | 0.0 | 0.0 | baseline |
| salex_validation | e3nn_pair_full | mpa | 168.0 | 168 | 463.376106985379 | 230.1219169748947 | 272.866346873343 | 1.3760291039943695e-06 | 0.0 | full |
- Worst absolute energy delta vs e3nn baseline: `0.000e+00` eV.
- Worst absolute force delta vs e3nn baseline: `9.155e-05` eV/A.
