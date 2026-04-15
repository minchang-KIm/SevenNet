# lmax와 데이터셋 크기/밀도 메모

이 문서는 `lmax`를 올릴 때 비용이 어떤 그래프에서 더 급격히 늘어나는지 보기 위한 보조 실험 메모다.

## 전제

- 정확도 실험은 현재 `rMD17 azobenzene` baseline-only sweep에서 직접 검증했다.
- 시간 실험은 대표 그래프 4개에서, 같은 baseline 구조를 랜덤 초기화 상태로 만들어 측정했다.
- 이 시간 실험은 학습된 가중치의 차이가 아니라 그래프 크기와 밀도, 그리고 `lmax` 증가가 만드는 구조적 비용을 보기 위한 것이다.

## 대표 데이터셋

- small_sparse: `spice_2023`
- small_dense: `salex_val_official`
- large_sparse: `oc20_s2ef_train_20m`
- large_dense: `mptrj`

선정 기준은 현재 public-local benchmark 샘플 분포의 `natoms`와 `avg_neighbors_directed` 중앙값을 기준으로 한 사분면이다.

## 정확도 쪽에서 이미 확인된 것

- `rMD17 azobenzene`에서는 `lmax=7`이 가장 좋은 힘 정확도(`0.136777 eV/A`)를 보였다.
- `lmax=8`은 더 비싸지만 정확도가 다시 약간 나빠졌다.
- 따라서 `lmax`가 높을수록 정확도가 무한정 좋아진다고 쓰면 안 된다.

## 시간/프로파일 쪽 핵심 관찰

- small_sparse (spice_2023): latency 4.816 -> 51.391 ms (10.67x), SH share 2.13% -> 4.51%, TP share 7.24% -> 16.66%, force share 52.52% -> 69.04%
- small_dense (salex_val_official): latency 4.855 -> 427.153 ms (87.97x), SH share 1.98% -> 0.57%, TP share 9.31% -> 8.78%, force share 52.63% -> 88.93%
- large_sparse (oc20_s2ef_train_20m): latency 4.873 -> 298.129 ms (61.18x), SH share 2.00% -> 0.82%, TP share 8.46% -> 8.88%, force share 53.91% -> 88.04%
- large_dense (mptrj): latency 4.880 -> 958.058 ms (196.31x), SH share 1.94% -> 0.26%, TP share 11.96% -> 8.43%, force share 50.53% -> 90.15%

## 해석

- `lmax`는 spherical harmonics의 최대 차수이고, edge에서 다루는 각도 표현 차원을 직접 늘린다.
- 그러나 실제 비용은 SH 하나만의 문제가 아니다.
- `lmax`가 올라가면 SH 차원뿐 아니라 hidden irreps, edge irreps, output irreps 사이의 tensor product 경로 수가 함께 증가한다.
- 그래서 large/dense 그래프에서는 `TP`와 `force backward` 쪽 시간이 더 빠르게 커진다.
- small/sparse 그래프에서는 절대 시간이 작아서 증가 폭이 상대적으로 덜 커 보일 수 있다.

## 현재 단계에서 방어 가능한 메시지

- `lmax`는 모델 설계자가 정하는 값이며, 데이터셋이 자동으로 정해주지 않는다.
- 높은 `lmax`는 더 풍부한 각도 표현을 가능하게 하지만, 실제 최적값은 데이터셋과 학습 조건에 따라 달라진다.
- 큰 graph, 특히 dense graph일수록 `lmax` 증가가 시간 비용으로 더 크게 나타난다.
- 따라서 geometry-side 비용을 줄이는 현재 제안기법의 가치는 높은 `lmax`와 large/dense graph에서 더 커질 가능성이 있다.
- 다만 이 문장 자체는 시간 구조에 대한 해석이고, 정확도 이득까지 일반화하려면 각 데이터셋별 재학습이 추가로 필요하다.
