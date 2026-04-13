# Public MLIP Dataset Cache

This folder stores public datasets that are repeatedly used in MLIP papers and benchmarks.

Generation:

- Script: `python bench/download_public_mlip_datasets.py`
- Registry entries: `40`

## Status

| name | category | approx_size_gib | representative_papers | status | source | local_path |
| --- | --- | ---: | --- | --- | --- | --- |
| `qm9_hf` | `molecular` | 0.05 | SchNet, DimeNet, PaiNN, GemNet-QM9 | `exists` | `nimashoghi/qm9` | `datasets/raw/hf/qm9_hf` |
| `md17_aspirin` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-aspirin` | `datasets/raw/hf/md17_aspirin` |
| `md17_benzene` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-benzene` | `datasets/raw/hf/md17_benzene` |
| `md17_ethanol` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-ethanol` | `datasets/raw/hf/md17_ethanol` |
| `md17_malonaldehyde` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-malonaldehyde` | `datasets/raw/hf/md17_malonaldehyde` |
| `md17_naphthalene` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-naphthalene` | `datasets/raw/hf/md17_naphthalene` |
| `md17_salicylic_acid` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-salicylic_acid` | `datasets/raw/hf/md17_salicylic_acid` |
| `md17_toluene` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-toluene` | `datasets/raw/hf/md17_toluene` |
| `md17_uracil` | `molecular` | 0.01 | SchNet, DimeNet, PaiNN, TorchMD-NET | `exists` | `graphs-datasets/MD17-uracil` | `datasets/raw/hf/md17_uracil` |
| `rmd17` | `molecular` | 0.98 | NequIP, MACE, Allegro | `exists` | `colabfit/rMD17` | `datasets/raw/hf/rmd17` |
| `iso17` | `molecular` | 0.68 | SchNet, NequIP, MACE | `exists` | `colabfit/ISO17_NC_2017` | `datasets/raw/hf/iso17` |
| `md22_at_at` | `molecular` | 0.04 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_AT_AT` | `datasets/raw/hf/md22_at_at` |
| `md22_at_at_cg_cg` | `molecular` | 0.04 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_AT_AT_CG_CG` | `datasets/raw/hf/md22_at_at_cg_cg` |
| `md22_ac_ala3_nhme` | `molecular` | 0.14 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_Ac_Ala3_NHMe` | `datasets/raw/hf/md22_ac_ala3_nhme` |
| `md22_dha` | `molecular` | 0.14 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_DHA` | `datasets/raw/hf/md22_dha` |
| `md22_double_walled_nanotube` | `molecular` | 0.06 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_double_walled_nanotube` | `datasets/raw/hf/md22_double_walled_nanotube` |
| `md22_stachyose` | `molecular` | 0.08 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_stachyose` | `datasets/raw/hf/md22_stachyose` |
| `md22_buckyball_catcher` | `molecular` | 0.03 | MD22 benchmark, MACE molecular benchmarks | `exists` | `colabfit/MD22_buckyball_catcher` | `datasets/raw/hf/md22_buckyball_catcher` |
| `ani1x` | `molecular` | 0.30 | TorchANI, ANI family, MatterSim molecular evaluations | `exists` | `colabfit/ANI-1x` | `datasets/raw/hf/ani1x` |
| `ani1ccx` | `molecular` | 0.25 | ANI-1ccx benchmark, TorchANI | `exists` | `roitberg-group/ani1ccx` | `datasets/raw/hf/ani1ccx` |
| `spice_2023` | `molecular` | 2.41 | SPICE / SPICE2, modern molecular foundation-model evaluations | `exists` | `colabfit/SPICE_2023` | `datasets/raw/hf/spice_2023` |
| `mptrj` | `materials` | 1.63 | CHGNet, M3GNet, MACE-MP | `exists` | `nimashoghi/mptrj` | `datasets/raw/hf/mptrj` |
| `wbm_initial` | `materials` | 0.00 | Matbench Discovery, OMat24 | `exists` | `https://ndownloader.figshare.com/files/48169597` | `datasets/raw/figshare/wbm_initial.extxyz.zip` |
| `phonondb_pbe` | `materials` | 0.00 | Matbench Discovery, phonon benchmark subsets | `exists` | `https://ndownloader.figshare.com/files/52179965` | `datasets/raw/figshare/phonondb_pbe.extxyz.zip` |
| `omat24_1m_official` | `materials` | 2.12 | OMat24 | `exists` | `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/omat24_1M_251210.tar.gz` | `datasets/raw/fairchem/omat24_1m/omat24_1M_251210.tar.gz` |
| `salex_train_official` | `materials` | 7.52 | OMat24, UMA inorganic fine-tuning | `exists` | `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz` | `datasets/raw/fairchem/salex/train.tar.gz` |
| `salex_val_official` | `materials` | 0.40 | OMat24, UMA inorganic fine-tuning | `exists` | `https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz` | `datasets/raw/fairchem/salex/val.tar.gz` |
| `omol25_train_4m` | `molecular` | 11.49 | OMol25 | `exists` | `colabfit/OMol25_train_4M` | `datasets/raw/hf/omol25_train_4m` |
| `omol25_validation` | `molecular` | 13.62 | OMol25 | `exists` | `colabfit/OMol25_validation` | `datasets/raw/hf/omol25_validation` |
| `oc20_s2ef_train_2m` | `catalysis` | 4.74 | GemNet, Allegro, FairChem OC20 baselines | `exists` | `nimashoghi/oc20_s2ef_train_2M` | `datasets/raw/hf/oc20_s2ef_train_2m` |
| `oc20_s2ef_val_id` | `catalysis` | 3.00 | GemNet, Allegro, FairChem OC20 baselines | `exists` | `colabfit/OC20_S2EF_val_id` | `datasets/raw/hf/oc20_s2ef_val_id` |
| `oc20_s2ef_val_ood_ads` | `catalysis` | 3.13 | GemNet, Allegro, FairChem OC20 baselines | `exists` | `colabfit/OC20_S2EF_val_ood_ads` | `datasets/raw/hf/oc20_s2ef_val_ood_ads` |
| `oc20_s2ef_val_ood_cat` | `catalysis` | 3.11 | GemNet, Allegro, FairChem OC20 baselines | `exists` | `colabfit/OC20_S2EF_val_ood_cat` | `datasets/raw/hf/oc20_s2ef_val_ood_cat` |
| `oc20_s2ef_val_ood_both` | `catalysis` | 3.31 | GemNet, Allegro, FairChem OC20 baselines | `exists` | `colabfit/OC20_S2EF_val_ood_both` | `datasets/raw/hf/oc20_s2ef_val_ood_both` |
| `oc22_s2ef_train` | `catalysis` | 30.13 | SevenNet, UMA catalyst models, OC22 baselines | `exists` | `colabfit/OC22-S2EF-Train` | `datasets/raw/hf/oc22_s2ef_train` |
| `oc22_s2ef_val_id` | `catalysis` | 1.39 | SevenNet, UMA catalyst models, OC22 baselines | `exists` | `colabfit/OC22-S2EF-Validation-in-domain` | `datasets/raw/hf/oc22_s2ef_val_id` |
| `oc22_s2ef_val_ood` | `catalysis` | 1.61 | SevenNet, UMA catalyst models, OC22 baselines | `exists` | `colabfit/OC22-S2EF-Validation-out-of-domain` | `datasets/raw/hf/oc22_s2ef_val_ood` |
| `oc20_s2ef_train_20m` | `catalysis` | 76.22 | GemNet-OC, EquiformerV2, Allegro full OC20-scale training | `exists` | `colabfit/OC20_S2EF_train_20M` | `datasets/raw/hf/oc20_s2ef_train_20m` |
| `omol25_train_neutral` | `molecular` | 59.51 | OMol25 | `exists` | `colabfit/OMol25_train_neutral` | `datasets/raw/hf/omol25_train_neutral` |
| `omol25_official_gated` | `molecular` | 0.00 | OMol25 | `gated` | `facebook/OMol25` | `datasets/raw/hf/omol25_official_gated` |

## Notes

- Background log path when running long jobs: `datasets/logs/public_download.log`.
- `large=true` entries are skipped unless `--include-large` is passed.
- `gated=true` entries are listed for completeness but require authentication or manual access approval.
- Mirrors on Hugging Face / ColabFit are used when they are easier to automate than the original paper distribution.
