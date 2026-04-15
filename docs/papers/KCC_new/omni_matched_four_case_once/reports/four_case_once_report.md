# FlashTP Four-Case One-Shot Result

## Purpose

성능이 잘 나온 대표 데이터셋에서 기본 SevenNet, 제안기법, FlashTP, FlashTP+제안기법을 한 번씩 실행해 결합 가능성과 대략적인 latency를 확인한다.

## Run Configuration

- datasets: `mptrj, oc20_s2ef_val_ood_ads, oc22_s2ef_val_ood, omat24_1m_official, omol25_validation, phonondb_pbe, qm9_hf, spice_2023, wbm_initial`
- warmup: `1`
- repeat: `1`
- note: repeat=1 결과이므로 논문 수치가 아니라 smoke benchmark다.

## Results

| dataset | case | status | timing_ms | resolved_policy | baseline/case or error |
| --- | --- | --- | ---: | --- | --- |
| mptrj | sevennet_baseline | ok | 501.455 | baseline | 1.000x |
| mptrj | sevennet_pair_full | ok | 756.477 | full | 0.663x |
| mptrj | flashtp_baseline | ok | 91.470 | baseline | 5.482x |
| mptrj | flashtp_pair_full | ok | 95.126 | full | 5.271x |
| oc20_s2ef_val_ood_ads | sevennet_baseline | ok | 217.465 | baseline | 1.000x |
| oc20_s2ef_val_ood_ads | sevennet_pair_full | ok | 159.183 | full | 1.366x |
| oc20_s2ef_val_ood_ads | flashtp_baseline | ok | 82.233 | baseline | 2.644x |
| oc20_s2ef_val_ood_ads | flashtp_pair_full | ok | 83.048 | full | 2.619x |
| oc22_s2ef_val_ood | sevennet_baseline | ok | 374.639 | baseline | 1.000x |
| oc22_s2ef_val_ood | sevennet_pair_full | ok | 330.551 | full | 1.133x |
| oc22_s2ef_val_ood | flashtp_baseline | ok | 86.628 | baseline | 4.325x |
| oc22_s2ef_val_ood | flashtp_pair_full | ok | 88.991 | full | 4.210x |
| omat24_1m_official | sevennet_baseline | ok | 332.236 | baseline | 1.000x |
| omat24_1m_official | sevennet_pair_full | ok | 286.178 | full | 1.161x |
| omat24_1m_official | flashtp_baseline | ok | 88.303 | baseline | 3.762x |
| omat24_1m_official | flashtp_pair_full | ok | 86.497 | full | 3.841x |
| omol25_validation | sevennet_baseline | ok | 367.796 | baseline | 1.000x |
| omol25_validation | sevennet_pair_full | ok | 324.264 | full | 1.134x |
| omol25_validation | flashtp_baseline | ok | 84.330 | baseline | 4.361x |
| omol25_validation | flashtp_pair_full | ok | 86.805 | full | 4.237x |
| phonondb_pbe | sevennet_baseline | ok | 155.760 | baseline | 1.000x |
| phonondb_pbe | sevennet_pair_full | ok | 117.802 | full | 1.322x |
| phonondb_pbe | flashtp_baseline | ok | 82.890 | baseline | 1.879x |
| phonondb_pbe | flashtp_pair_full | ok | 81.564 | full | 1.910x |
| qm9_hf | sevennet_baseline | ok | 157.482 | baseline | 1.000x |
| qm9_hf | sevennet_pair_full | ok | 118.770 | full | 1.326x |
| qm9_hf | flashtp_baseline | ok | 81.771 | baseline | 1.926x |
| qm9_hf | flashtp_pair_full | ok | 81.836 | full | 1.924x |
| spice_2023 | sevennet_baseline | ok | 157.983 | baseline | 1.000x |
| spice_2023 | sevennet_pair_full | ok | 119.865 | full | 1.318x |
| spice_2023 | flashtp_baseline | ok | 82.486 | baseline | 1.915x |
| spice_2023 | flashtp_pair_full | ok | 83.142 | full | 1.900x |
| wbm_initial | sevennet_baseline | ok | 157.551 | baseline | 1.000x |
| wbm_initial | sevennet_pair_full | ok | 118.529 | full | 1.329x |
| wbm_initial | flashtp_baseline | ok | 81.586 | baseline | 1.931x |
| wbm_initial | flashtp_pair_full | ok | 82.218 | full | 1.916x |

## Interpretation Guardrail

- `sevennet_pair_full`은 최근 win을 복구한 two-pass 쌍 단위 실행 경로다.
- `flashtp_pair_full`은 FlashTP fused tensor-product backend 위에서 pair geometry/weight 재사용 입력을 붙인 결합 smoke test다.
- 현재 FlashTP convolution class는 일반 e3nn convolution의 two-pass gather 경로와 동일하지 않으므로, 이 결과를 최종 결합 성능 claim으로 쓰면 안 된다.
