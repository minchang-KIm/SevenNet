# 7net-omni 맞춤 데이터셋 빠른 재실험 정리

## 목적

이 실험은 논문에 넣을 최종 평균 성능 실험이 아니다. `7net-omni`가 어떤 데이터 분포와 task를 가정하는지 확인하고, 그 가정에 직접 대응되는 로컬 데이터셋에서 `SevenNet baseline`, `제안기법 full`, `FlashTP`, `FlashTP+제안기법` 경로가 모두 정상 동작하는지 빠르게 확인하기 위한 smoke benchmark다.

설정은 `warmup=1`, `repeat=1`이다. 따라서 수치의 방향성은 참고할 수 있지만, 논문 claim에는 `repeat=30` 재측정이 필요하다.

## 7net-omni가 가정하는 모델 성격

`7net-omni`는 단일 도메인 모델이 아니라 multi-task pretrained MLIP다. SevenNet 문서 기준으로 crystals, molecules, surfaces를 포함하는 15개 공개 ab initio 데이터셋을 함께 사용하고, 입력 데이터의 DFT 수준과 데이터 출처에 맞춰 modal을 선택한다.

핵심 가정은 다음과 같다.

- `lmax=3`, `N_layer=5`, full parity 계열의 더 무거운 등변 message passing 모델이다.
- `mpa`, `omat24`, `matpes_pbe`, `matpes_r2scan`, `oc20`, `oc22`, `omol25_low`, `omol25_high`, `spice`, `qcml`, `pet_mad` 등의 task/modal을 지원한다.
- 같은 checkpoint 안에서 modal만 바꾸어 materials, catalysis, molecular 데이터셋을 처리할 수 있다.
- 따라서 실행 최적화 논문에서는 데이터셋마다 다른 pretrained model을 고르는 것보다, `7net-omni` 하나로 고정하고 modal만 데이터셋에 맞추는 것이 더 통제된 비교다.

## 선택한 데이터셋과 선택 이유

| dataset | modal | category | 이유 |
| --- | --- | --- | --- |
| `mptrj` | `mpa` | materials | Omni의 대표 PBE(+U) materials task |
| `omat24_1m_official` | `omat24` | materials | OMat24 task에 직접 대응되는 dense materials |
| `oc20_s2ef_val_ood_ads` | `oc20` | catalysis | OC20 surface/catalysis task, 이전 full 경로 win 대표 |
| `oc22_s2ef_val_ood` | `oc22` | catalysis | OC22 surface/catalysis task, large dense 대표 |
| `omol25_validation` | `omol25_low` | molecular | OMol25 molecular task에 직접 대응 |
| `spice_2023` | `spice` | molecular | SPICE task에 직접 대응 |
| `qm9_hf` | `qcml` | molecular | QCML task에 대응되는 small molecular 대표 |
| `wbm_initial` | `matpes_pbe` | materials | Matbench/WBM 계열 PBE materials 대표 |
| `phonondb_pbe` | `matpes_r2scan` | materials | phonon/r2SCAN 계열 small materials 대표 |

## 실행 명령

```bash
python docs/papers/KCC_new/scripts/kcc_new_flash_four_case_once.py \
  --datasets mptrj omat24_1m_official oc20_s2ef_val_ood_ads oc22_s2ef_val_ood omol25_validation spice_2023 qm9_hf wbm_initial phonondb_pbe \
  --warmup 1 --repeat 1 \
  --output-root docs/papers/KCC_new/omni_matched_four_case_once
```

## 결과 요약

| dataset | modal | edges | neighbors | proposal / SevenNet | FlashTP / SevenNet | FlashTP / FlashTP+proposal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `oc20_s2ef_val_ood_ads` | `oc20` | 6686 | 29.72 | 1.366x | 2.644x | 0.990x |
| `wbm_initial` | `matpes_pbe` | 2260 | 22.60 | 1.329x | 1.931x | 0.992x |
| `qm9_hf` | `qcml` | 570 | 19.66 | 1.326x | 1.926x | 0.999x |
| `phonondb_pbe` | `matpes_r2scan` | 208 | 26.00 | 1.322x | 1.879x | 1.016x |
| `spice_2023` | `spice` | 728 | 8.18 | 1.318x | 1.915x | 0.992x |
| `omat24_1m_official` | `omat24` | 15224 | 95.15 | 1.161x | 3.762x | 1.021x |
| `omol25_validation` | `omol25_low` | 17972 | 51.35 | 1.134x | 4.361x | 0.971x |
| `oc22_s2ef_val_ood` | `oc22` | 18264 | 84.56 | 1.133x | 4.325x | 0.973x |
| `mptrj` | `mpa` | 27612 | 62.19 | 0.663x | 5.482x | 0.962x |

모든 케이스는 실패 없이 실행됐다.

## 해석

`SevenNet baseline` 대비 `제안기법 full`은 9개 중 8개 데이터셋에서 빠르게 나왔다. 다만 `repeat=1`이므로 이 결과를 그대로 논문 수치로 쓰면 안 된다. 특히 small sparse 데이터셋에서 1.3x 수준의 이득이 나온 것은 topology cache, CUDA kernel warmup, repeat=1 변동의 영향을 받을 수 있다. 최종 주장은 같은 데이터셋 subset을 `repeat=30`으로 다시 돌린 뒤에 해야 한다.

`mptrj`는 예외적으로 `제안기법 full`이 baseline보다 느렸다. 이 데이터셋은 edge 수가 가장 크고 dense하다. 단순히 edge 수가 크다고 항상 유리하지 않다는 기존 관찰과 일치한다. 가능한 원인은 full 경로의 pair indexing, edge expansion, two-pass message path가 TP 절감분보다 커졌기 때문이다. 즉 유리 조건은 "edge 수가 크다" 하나가 아니라, "재사용 가능한 geometry/weight 비용이 충분히 크면서 pair metadata와 expansion 비용이 과도하게 커지지 않는 구조"로 정의해야 한다.

`FlashTP baseline`은 모든 데이터셋에서 큰 이득을 보였다. 이것은 FlashTP가 tensor product/backend 쪽을 강하게 줄이기 때문이다. 반면 현재 `FlashTP+제안기법`은 FlashTP보다 항상 좋아지지는 않았다. 현재 결합은 FlashTP 내부 kernel을 pair-major로 바꾼 것이 아니라, FlashTP 앞단에 pair geometry/weight reuse를 붙이고 다시 edge-major 입력으로 펼치는 방식이다. 따라서 pair-to-edge expansion과 추가 indexing이 남아 있다.

흥미로운 점은 `omat24_1m_official`과 `phonondb_pbe`에서는 `FlashTP+제안기법`이 FlashTP 단독보다 약간 빠르게 나왔다는 점이다. 그러나 차이가 작고 `repeat=1`이므로, 현재 단계에서는 "결합 가능성 후보"로만 본다. 논문에는 이 결과를 결론으로 쓰지 말고 후속 구현 방향의 근거로 쓰는 것이 안전하다.

## 다음 실험 우선순위

1. `oc20_s2ef_val_ood_ads`, `omat24_1m_official`, `omol25_validation`, `oc22_s2ef_val_ood`, `mptrj`를 `repeat=30`으로 재측정한다.
2. 같은 subset에서 detailed profiling을 돌려 `spherical harmonics`, `radial/cutoff`, `weight_nn`, `pair indexing`, `pair expansion`, `TP`, `force_output`의 증감을 분리한다.
3. `mptrj`만 따로 원인 분석한다. 큰 dense graph에서 왜 손해가 나는지 확인해야 최종 논리의 반례 처리가 가능하다.
4. `FlashTP+제안기법`은 현재 구현을 성능 claim으로 두지 말고, pair-major FlashTP kernel 또는 최소한 weight-only reuse의 추가 실험 대상으로 둔다.
5. 논문 주장은 `7net-omni`를 공통 backbone으로 고정하고, "Omni modal과 직접 대응되는 데이터셋에서 정확도 보존 및 조건부 속도 개선"으로 쓰는 것이 가장 방어 가능하다.

## 산출물

- `metrics/four_case_once_summary.csv`: 4개 케이스 요약 결과
- `metrics/four_case_once_raw.csv`: repeat별 raw timing
- `metrics/dataset_manifest.csv`: 데이터셋별 atom/edge/density 정보
- `metrics/derived_speedups.csv`: 해석용 speedup 표
- `figures/omni_matched_four_case_latency.png`
- `figures/omni_matched_proposal_speedup.png`
- `figures/omni_matched_flash_combination_speedup.png`
