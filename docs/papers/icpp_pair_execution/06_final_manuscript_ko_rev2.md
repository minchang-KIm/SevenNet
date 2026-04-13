# 등변 GNN 원자간 퍼텐셜 추론에서 Pair-Aware 구면조화함수 계산 재사용

**Pair-Aware Reuse of Spherical-Harmonic Computation in Equivariant GNN Interatomic Potential Inference**

김민창  
아주대학교 분산병렬컴퓨팅 연구소 WiseLab  
minchang111@ajou.ac.kr

## 요 약

NequIP 계열의 short-range 등변 그래프 신경망 원자간 퍼텐셜은 동일한 물리적 상호작용을 `i -> j`와 `j -> i`의 두 directed edge로 중복 표현한다. 이때 거리, radial basis, cutoff는 양방향에서 동일하며, spherical harmonics는 parity 관계를 이용하여 역방향 값을 복원할 수 있으므로 exact runtime reuse 기회가 존재한다. 본 논문에서는 SevenNet에 pair-aware 실행 경로를 구현하여 undirected pair 단위로 metadata를 구성하고, geometry-side 계산을 pair 기준으로 1회만 수행한 뒤 역방향 spherical harmonics를 복원하는 방법을 제안한다. 구현은 학습된 모델의 수식과 파라미터를 바꾸지 않으며 baseline과 거의 동일한 에너지와 힘을 유지한다. 실험 결과 현재 구현은 small molecular workload에서는 대체로 손해를 보지만, 일부 large periodic workload에서는 소폭의 성능 개선을 보였다. 그러나 tensor product 메시지 생성, aggregation, force/stress backward는 여전히 edge-major로 남아 있으므로 전체 성능 향상은 제한적이다. 본 연구는 국내 학회 선행 연구로서 현재 구현의 정확도와 한계를 분석하고, 구면조화함수 계산값 재사용만을 위해 오버헤드를 감수한 현재 구조에서도 일부 workload에서 양의 성능 이득이 관찰되었다는 점을 바탕으로, 후속 pair-major 실행 재설계의 필요성과 가능성을 논의한다.

## 1. 서 론

등변 그래프 신경망 기반 원자간 퍼텐셜은 지역 원자 환경을 그래프로 표현하고, 회전 및 병진 대칭성을 보존하면서 에너지와 힘을 예측한다. NequIP 계열의 모델은 높은 정확도와 우수한 데이터 효율성으로 인해 재료 및 분자 시뮬레이션에서 널리 사용되고 있다. 그러나 이러한 모델의 추론 경로를 시스템 관점에서 보면, 동일한 원자 쌍 상호작용이 두 개의 directed edge로 중복 계산된다는 구조적 비효율이 존재한다.

원자 `i`와 `j` 사이의 short-range 상호작용은 물리적으로 하나의 pair로 볼 수 있으나, 메시지 패싱 구현에서는 `i -> j`와 `j -> i`가 서로 다른 edge로 취급된다. 이때 두 edge의 거리와 radial basis, cutoff는 동일하며, spherical harmonics는 parity 부호에 의해 역방향 값을 정확히 복원할 수 있다. 따라서 모델을 다시 학습하거나 수학적 정의를 바꾸지 않고도 geometry-side 계산을 pair 단위로 재사용할 수 있는 가능성이 생긴다.

본 논문은 이러한 pair-symmetric 재사용 가능성을 SevenNet 런타임에 실제로 반영한 결과를 정리한다. 다만 본 연구가 다루는 구현은 pair-major tensor-product 실행 전체를 완성한 결과가 아니라, 구면조화함수와 geometry-side 계산의 exact reuse에 초점을 둔 선행 연구이다. 다시 말해, 추론 경로 전체를 pair 단위로 재설계한 것이 아니라, 메시지 생성 직전까지의 중복 geometry 계산을 줄이는 수준의 구현을 대상으로 한다.

본 연구의 목적은 세 가지이다. 첫째, NequIP/SevenNet 계열 추론 파이프라인에서 pair 단위로 재사용 가능한 항과 재사용 불가능한 항을 구분한다. 둘째, SevenNet에 pair-aware geometry-side 실행 경로를 구현하고 baseline 대비 정확도를 검증한다. 셋째, 성능이 일부 workload에서만 제한적으로 개선되는 이유를 분석하고, 후속 pair-major 실행 재설계가 필요한 이유를 논의한다.

![그림 1](./assets/pair_execution_mechanism.png)

그림 1은 현재 pair-aware 실행 아이디어의 개념을 보여준다. geometry-side 항은 공유 또는 복원할 수 있지만, 메시지 생성과 후속 reduction은 현재 구현에서 여전히 directed edge 기준으로 남아 있다.

## 2. 배 경

### 2.1 등변 GNN-IP의 추론 파이프라인

NequIP 계열의 등변 모델에서 추론은 원자종을 node feature로 임베딩한 뒤, cutoff 이내의 이웃으로부터 directed edge를 구성하고, 각 edge에 대해 distance, radial basis, cutoff, spherical harmonics를 계산하는 과정으로 시작한다. 이후 interaction block에서 filter network와 tensor product가 결합되어 메시지를 생성하고, 이 메시지가 node 수준으로 aggregation된다. 마지막으로 atomic energy를 readout한 후, 총 에너지에 대한 미분을 통해 힘과 stress를 구한다.

여기서 중요한 점은 spherical harmonics를 계산하는 단계가 곧바로 메시지 생성 그 자체는 아니라는 사실이다. 이 단계는 메시지 생성을 위한 edge attribute를 준비하는 과정이며, 실제 메시지 생성은 그 다음 tensor product에서 일어난다. 따라서 pair 단위 재사용 가능성은 바로 이 geometry encoding 경계에서 발생한다.

### 2.2 pair 단위에서 재사용 가능한 항

역방향 pair에 대해 distance, radial basis, cutoff는 완전히 동일하다. 또한 spherical harmonics는 parity 관계에 의해 역방향 값을 정확히 복원할 수 있다. 따라서 geometry-side 값들은 pair 기준으로 1회만 계산한 뒤 양방향 edge에 공유하거나 복원하는 것이 가능하다. 반면, source node feature에 의존하는 tensor-product message와 destination node로의 aggregation은 directed edge의 방향성을 그대로 가진다. 총 에너지에 대한 force/stress backward 역시 이러한 directed 계산 경로 전체를 다시 지나므로 현재 형태로는 pair 간 완전 공유가 어렵다.

이 구분은 본 논문의 핵심이다. 즉, 본 연구는 메시지 생성 전체를 줄이는 방법이 아니라, 메시지 생성 이전 단계에서 중복되는 geometry-side 계산을 줄이는 방법이다.

### 2.3 관련 연구

NequIP [1]는 등변 메시지 패싱 기반 원자간 퍼텐셜의 대표적인 출발점이며, SevenNet [2]은 이러한 계열의 모델을 실제 대규모 분자동역학 환경에 배포할 수 있도록 한 시스템이다. FlashTP [3]는 tensor product와 gather/scatter를 fused kernel로 최적화하여 directed edge 경로의 효율을 높였으나, geometry-side 중복 자체를 제거하는 접근은 아니다. 반면 NEMP [4]와 같은 연구는 메시지 패싱 표현을 보다 크게 재설계하는 방향에 가깝다.

본 연구는 이들 사이의 다른 축을 다룬다. 즉, 모델의 수식과 학습된 파라미터는 그대로 유지하면서 실행 표현만 pair-aware하게 바꾸어 exact reuse를 얻고자 한다. 따라서 본 논문은 tensor product kernel 자체를 재설계한 연구도 아니고, 모델 정의를 근본적으로 바꾼 연구도 아니다.

## 3. 현재 구현

현재 SevenNet 구현은 pair-major execution이 아니라 pair-aware geometry-side reuse이다. 먼저 directed edge 목록으로부터 edge-to-pair map, canonical forward edge index, reverse edge index, reverse mask, pair-has-reverse mask, topology signature와 같은 metadata를 구성한다. 이어서 canonical 방향 기준으로 각 pair에 대해 distance, radial basis, cutoff가 적용된 edge embedding, spherical harmonics를 한 번만 계산한다. 역방향 spherical harmonics는 parity 부호를 이용하여 복원하고, filter network 역시 pair마다 한 번만 수행하여 pair-level `weight_nn` 결과를 만든다.

그러나 tensor product 메시지 생성은 여전히 directed edge 기준으로 수행된다. 즉, 현재 구현은 geometry-side 중복 계산은 줄이지만, 메시지 생성과 aggregation, force/stress backward는 pair-major로 통합하지 못한다. 이러한 구현 상태를 표 1에 정리하였다.

표 1. 현재 구현 상태

| 구성 요소 | 구현 여부 |
| --- | --- |
| Pair metadata 및 topology signature | 구현 |
| Pair-wise radial / cutoff 재사용 | 구현 |
| Parity 기반 reverse SH 복원 | 구현 |
| Pair-wise `weight_nn` 재사용 | 구현 |
| Pair-major tensor-product kernel | 미구현 |
| Pair-major fused reduction | 미구현 |
| FlashTP와의 pair-major 결합 | 미구현 |
| LAMMPS topology-epoch cache 통합 | 미구현 |
| Distributed backward pruning | 미구현 |

따라서 현재 코드의 정확한 표현은 pair-major 전체 실행 엔진이 아니라, geometry-side exact reuse를 구현한 부분 최적화이다.

## 4. 실험 방법

실험에는 local public inventory에 포함된 데이터셋들을 사용하였다. 공식 `OMat24`, 공식 `sAlex`, `ANI-1ccx` 로더를 포함하여 총 31개 데이터셋을 직접 벤치마킹할 수 있었고, 나머지 9개는 graph-only mirror, raw 파일 부재, gated access로 인해 제외하였다.

![그림 2](./assets/quadrant_dataset_map.png)

그림 2는 본 연구에서 사용한 대표 workload map을 보여준다. 단순히 그래프 크기만이 아니라 평균 directed neighbors로 표현되는 밀도도 함께 고려하였다.

![그림 3](./assets/support_status.png)

그림 3은 all-public local benchmark inventory의 지원 상태를 요약한 것이다.

측정은 cold latency, repeated latency, stage breakdown을 포함하여 수행하였으나, 최종 성능 주장은 warm-up 이후 stable region에 기반하여 해석하였다. 이는 small graph에서 반복 초반의 autotuning 및 warm-up 영향이 headline 수치를 왜곡할 수 있기 때문이다. 또한 intrusive profiling은 단계별 비중과 load 해석을 위한 도구로만 사용하고, 절대 지연 시간 비교는 non-intrusive timing을 기준으로 하였다.

정확도 평가는 baseline 대비 에너지 차이와 힘 차이를 사용하였다. 성능 평가는 stable latency, 그래프 크기, edge 수, 평균 directed neighbors, 그리고 단계별 profile을 함께 사용하였다.

## 5. 실험 결과

### 5.1 정확도 검증

표 2는 주요 benchmark suite에서 baseline과 비교한 최대 차이를 보여준다.

표 2. baseline 대비 최대 오차

| 벤치마크 | 최대 에너지 차이 | 최대 힘 차이 |
| --- | ---: | ---: |
| All-public local | `6.104e-05 eV` | `2.441e-04 eV/A` |
| Real e3nn pair | `0` | `9.155e-05 eV/A` |
| FlashTP real | `0` | `7.63e-05 eV/A` |

worst all-public local case는 `OMat24 official`이었으나, 대부분의 workload에서 힘 차이는 `1e-6 ~ 1e-5` 수준에 머물렀다. 이는 모델의 의미가 바뀌었다기보다 부동소수점 합산 순서 차이에 가까운 결과로 해석할 수 있다. 따라서 현재 구현은 baseline의 수학적 의미를 바꾸는 근사화가 아니라, exact execution reformulation으로 보는 것이 타당하다.

### 5.2 warm-up artifact와 representative stable recheck

초기 broad sweep에서는 `qm9_hf`가 약 `3.7x`의 speedup을 보이는 것처럼 나타났다. 그러나 반복 초반의 warm-up이 강하게 섞인 짧은 repeated window를 그대로 steady-state로 해석하면 이러한 outlier가 생길 수 있다.

![그림 4](./assets/qm9_warmup_artifact.png)

그림 4는 `qm9_hf`의 warm-up artifact를 보여준다. warm-up이 지배적인 앞선 반복을 제외하면 `qm9_hf`는 오히려 손해 사례로 바뀐다.

이에 따라 대표 데이터셋들에 대해 stable region만을 다시 측정한 결과를 표 3에 정리하였다.

표 3. representative stable recheck 결과

| 데이터셋 | workload 유형 | Baseline stable median (ms) | Pair stable median (ms) | Baseline / Pair |
| --- | --- | ---: | ---: | ---: |
| `qm9_hf` | small molecular | 28.59 | 47.14 | 0.61 |
| `iso17` | small molecular | 29.04 | 47.23 | 0.61 |
| `salex_train_official` | periodic medium | 151.56 | 147.37 | 1.03 |
| `oc20_s2ef_train_20m` | periodic medium | 125.32 | 118.60 | 1.06 |
| `omat24_1m_official` | periodic large | 232.09 | 228.99 | 1.01 |
| `mptrj` | periodic large | 424.73 | 419.95 | 1.01 |

![그림 5](./assets/rechecked_representative_latency.png)

그림 5는 representative stable latency를, 그림 6은 그에 대응하는 speedup을 보여준다.

![그림 6](./assets/rechecked_representative_speedup.png)

이 결과는 현재 구현이 범용 가속기가 아니라는 점을 분명히 한다. small molecular workload에서는 일관되게 손해가 발생하였고, larger periodic workload에서는 본전에서 소폭의 이득만이 남았다. 따라서 현재 구현을 전체 추론 엔진의 일반적인 성능 향상으로 주장하는 것은 적절하지 않다.

### 5.3 구면조화함수 재사용의 효과와 한계

현재 구현의 강점은 tensor product 자체를 줄인 것이 아니라 geometry-side exact reuse를 실현했다는 점에 있다. 이를 보여주기 위해 representative profile에서 baseline model total 중 spherical harmonics가 차지하는 비중을 살펴보면, `mptrj`에서는 `18.1%`, `md22_double_walled_nanotube`에서는 `26.4%`, 반면 `spice_2023`에서는 `0.19%`에 불과하였다. 즉, large periodic graph에서는 spherical harmonics가 무시할 수 없는 비중을 차지하지만, small molecular graph에서는 그 비중이 매우 작다.

load 관점에서 보면 이 차이는 더 분명해진다. `mptrj` 대표 샘플에서 SH, radial, cutoff 관련 edge 수는 `27612`에서 `13806`으로 정확히 절반으로 줄었고, `weight_nn` 입력 row 수 역시 `138060`에서 `69030`으로 절반으로 줄었다. 그러나 message tensor product load는 `138060`에서 `138060`으로 변하지 않았다. 즉, 현재 구현은 geometry-side 값의 중복을 제거하였지만, 메시지 생성의 directed workload는 그대로 남겨 두고 있다.

이 점은 현재 구현의 성능이 제한적일 수밖에 없는 이유를 설명한다. SH 비중이 작은 workload에서는 재사용의 이득이 거의 없고, SH 비중이 큰 workload에서도 tensor product, aggregation, force/stress backward가 그대로 남아 있기 때문에 전체 성능 향상이 몇 퍼센트 수준에 머물게 된다.

### 5.4 force/stress backward가 가지는 의미

현재 추론 경로에서 force/stress 계산은 최종 MLP readout 뒤에 붙는 사소한 후처리가 아니다. 총 에너지에 대해 `edge_vec`로 미분을 수행하므로 gradient는 edge embedding, convolution, tensor product, self interaction, readout을 모두 다시 통과한다. 따라서 inference에서의 backward는 실제로 주요 비용 경로이며, 현재 구현은 이 부분을 줄이지 못한다. 다시 말해, 구면조화함수 계산값 재사용만으로는 end-to-end 실행 시간의 주요 부분을 해결할 수 없다.

## 6. 논 의

본 연구는 두 가지 사실을 동시에 보여준다. 첫째, pair-aware geometry reuse는 모델 수식과 학습된 파라미터를 바꾸지 않고 구현 가능하며, baseline 대비 정확도도 실용적으로 유지할 수 있다. 둘째, 이러한 geometry-side 재사용만으로는 전체 추론 성능을 크게 개선하기 어렵다. 현재 병목은 tensor product, aggregation, force/stress backward에 남아 있으며, 이들은 모두 여전히 edge-major 실행 경로를 따른다.

그럼에도 본 연구의 결과는 후속 pair-major 재설계를 정당화하는 근거가 된다. 현재 구현은 구면조화함수 계산값 재사용과 geometry-side 중복 제거만을 위해 pair metadata 생성, reverse-edge bookkeeping, topology cache 검사와 같은 추가 오버헤드를 감수하였다. 그럼에도 `OC20`, `sAlex`, `OMat24`, `MPtrj`와 같은 일부 large periodic workload에서는 baseline 대비 양의 성능 이득이 남았다. 이는 geometry-side reuse만으로도 favorable workload에서 최소한의 효과가 관찰된다는 뜻이다.

따라서 후속 연구에서 동일한 pair 단위를 tensor product와 aggregation까지 유지하는 pair-major runtime을 구현한다면, 현재 남아 있는 directed message 비용과 force/stress backward 경로의 중복 일부까지 더 줄일 수 있을 것으로 기대된다. 물론 이 기대는 본 논문이 직접 증명한 결론은 아니다. 다만 현재와 같은 오버헤드를 이미 부담하는 구조에서도 일부 workload에서 성능 향상이 관찰되었다는 사실은, 더 강한 pair-major 재설계가 실질적인 성능 향상으로 이어질 가능성을 시사한다.

이러한 이유로 본 논문은 pair-major 전체 실행 엔진의 완성본이 아니라, 그 방향으로 가기 위한 국내 학회 선행 연구로 위치짓는 것이 적절하다. 현재 구현의 범위와 한계를 엄격히 기술하고, pair-major tensor-product 실행, accelerator co-design, distributed runtime 개선은 후속 연구 과제로 남겨 두는 것이 현재 결과에 가장 부합하는 정리라고 판단한다.

## 7. 결론 및 향후 연구

본 논문에서는 SevenNet에서 등변 GNN 원자간 퍼텐셜 추론의 pair-symmetric 재사용 기회를 구현하고 분석하였다. 현재 구현은 pair-major execution이 아니라 pair-aware geometry-side reuse이며, distance, radial basis, cutoff, spherical harmonics, pair-level filter input을 재사용하거나 복원한다. baseline 대비 에너지와 힘은 거의 동일하게 유지되었고, 일부 large periodic workload에서는 소폭의 성능 향상이 나타났다. 반면 small molecular workload에서는 대체로 손해가 발생하였다.

결국 본 연구가 보여주는 핵심은 다음과 같다. 구면조화함수와 geometry-side exact reuse 자체는 실현 가능하며, 이것만으로도 일부 favorable workload에서는 양의 효과가 나타날 수 있다. 그러나 tensor product, aggregation, force/stress backward가 여전히 edge-major로 남아 있으므로, geometry-side 최적화만으로는 충분하지 않다. 따라서 후속 연구에서는 pair-major tensor-product 실행, topology-epoch 기반 metadata 재사용, LAMMPS end-to-end 경로에서의 성능 검증을 포함한 전체 runtime 재설계가 필요하다.

## 참 고 문 헌

[1] S. Batzner, A. Musaelian, L. Sun, et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, vol. 13, 2453, 2022.  
[2] Y. Park, et al., “SevenNet: a graph neural network interatomic potential package supporting efficient multi-GPU parallel molecular dynamics simulations,” *Journal of Chemical Theory and Computation*, 2024.  
[3] J. Lee, et al., “FlashTP: fused, sparsity-aware tensor product for machine learning interatomic potentials,” *Proceedings of the International Conference on Machine Learning*, 2024.  
[4] Y. Zhang and H. Guo, “Node-Equivariant Message Passing for Efficient and Accurate Machine Learning Interatomic Potentials,” *Chemical Science*, 2025.
