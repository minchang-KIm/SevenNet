# 등변 GNN 원자간 퍼텐셜 추론에서 Pair-Aware 구면조화함수 계산값 재사용

**Pair-Aware Reuse of Spherical-Harmonic Computation in Equivariant GNN Interatomic Potential Inference**

**김민창**  
아주대학교 분산병렬컴퓨팅 연구소 WiseLab  
minchang111@ajou.ac.kr

## 요 약

NequIP 계열의 short-range 등변 그래프 신경망 원자간 퍼텐셜(GNN-IP)은 동일한 물리적 상호작용을 `i -> j`와 `j -> i`의 두 directed edge로 중복 표현한다. 이때 거리, radial basis, cutoff는 양방향에서 동일하고, spherical harmonics는 parity 관계를 이용해 역방향 값을 복원할 수 있으므로 정확한 런타임 재사용 기회가 존재한다. 본 연구는 SevenNet에 pair-aware 실행 경로를 구현하여 undirected pair 단위의 metadata를 구성하고, geometry-side 계산을 pair 기준으로 1회만 수행한 뒤 역방향 spherical harmonics를 복원하는 방법을 제안한다. 구현은 학습된 모델 수식과 파라미터를 바꾸지 않으며 baseline과 거의 동일한 에너지 및 힘을 유지한다. 실험 결과 현재 구현은 small molecular workload에서는 대체로 손해를 보지만, 일부 large periodic workload에서는 소폭의 성능 개선을 보였다. 그러나 tensor product 메시지 생성, aggregation, force/stress backward는 여전히 edge-major이므로 성능 향상은 제한적이다. 본 논문은 이러한 구현과 결과를 선행 연구로 정리하고, 현재의 metadata 오버헤드를 감수하면서도 일부 workload에서 성능 이득이 나타났다는 점을 근거로, 후속 연구인 pair-major tensor-product 실행 재설계가 더 큰 성능 향상을 가져올 가능성을 논의한다.

**주제어:** 등변 그래프 신경망, 원자간 퍼텐셜, 구면조화함수, 런타임 최적화, SevenNet, pair execution

## 1. 서론

등변 그래프 신경망 기반 원자간 퍼텐셜은 주변 원자 환경을 그래프로 표현하고, 회전 및 병진 대칭성을 보존하면서 에너지와 힘을 예측한다. NequIP 계열의 모델은 높은 정확도로 인해 다양한 재료 및 분자 시뮬레이션에 활용되고 있으나, 추론 시 동일한 원자 쌍 상호작용을 두 개의 directed edge로 중복 계산한다는 시스템 관점의 비효율이 존재한다.

원자 `i`와 `j` 사이의 short-range 상호작용은 물리적으로 하나의 pair이지만, 메시지 패싱 구현에서는 `i -> j`와 `j -> i`가 별개의 edge로 취급된다. 이때 두 edge의 거리, radial basis, cutoff는 동일하며, spherical harmonics는 parity 부호를 통해 역방향 값을 정확히 복원할 수 있다. 따라서 모델의 수학적 정의를 바꾸지 않고도 geometry-side 계산을 pair 단위로 재사용할 여지가 있다.

본 연구의 목적은 이러한 pair-symmetric 재사용 기회를 SevenNet 런타임에서 실제로 구현하고, 정확도 및 성능 특성을 정량적으로 검증하는 것이다. 다만 본 논문이 다루는 현재 구현은 **pair-major tensor-product 실행 전체**가 아니라, **구면조화함수와 geometry-side 계산의 exact reuse**에 초점을 둔 선행 연구이다. 즉, 추론 경로 전체를 pair 단위로 재설계한 것이 아니라, geometry 준비 단계의 중복 계산을 제거하는 수준까지 구현하였다.

본 논문의 기여는 다음과 같다.

1. NequIP/SevenNet 계열 추론 파이프라인에서 pair 단위로 정확히 재사용 가능한 항과 재사용 불가능한 항을 구분한다.
2. SevenNet에 pair-aware geometry-side 실행 경로를 구현하여 distance, radial basis, cutoff, spherical harmonics, pair-level filter input의 재사용을 가능하게 한다.
3. public workload 전반에서 baseline 대비 정확도를 검증하고, warm-up artifact를 교정한 후 representative stable latency를 다시 측정한다.
4. 현재 구현의 성능이 제한적인 이유를 분석하고, 후속 pair-major tensor-product 재설계가 필요한 이유를 논리적으로 제시한다.

![그림 1](./assets/pair_execution_mechanism.png)

*그림 1. 현재 pair-aware 실행 아이디어의 개념도. geometry-side 항은 공유 또는 복원 가능하지만, 메시지 생성과 후속 reduction은 현재 구현에서 여전히 directed edge 기준으로 남아 있다.*

## 2. 배경

### 2.1 등변 GNN-IP 추론 파이프라인

NequIP 계열의 등변 모델에서 추론은 대체로 다음 순서로 이루어진다.

1. 원자종을 node embedding으로 변환한다.
2. cutoff 이내의 이웃으로부터 directed edge를 구성한다.
3. edge geometry로부터 distance, radial basis, cutoff, spherical harmonics를 계산한다.
4. interaction block에서 filter network와 tensor product를 통해 메시지를 생성한다.
5. 메시지를 node로 aggregation한다.
6. atomic energy를 readout하고, 총 에너지로부터 힘과 stress를 미분한다.

여기서 중요한 점은 spherical harmonics를 계산하는 단계가 곧바로 메시지 생성 자체는 아니라는 것이다. 이 단계는 **메시지 생성을 위한 edge attribute를 준비하는 단계**이며, 실제 메시지 생성은 그 다음 tensor product에서 일어난다. 따라서 재사용 가능성은 정확히 이 경계에서 발생한다.

### 2.2 재사용 가능한 항과 재사용 불가능한 항

역방향 pair에 대해 정확히 재사용 가능한 항은 다음과 같다.

- distance
- radial basis
- cutoff
- pair-level filter-network input

정확히 복원 가능한 항은 다음과 같다.

- spherical harmonics: parity 부호를 이용해 역방향 값을 복원 가능

반면, 다음 항은 source node feature 또는 destination node에 의존하므로 현재 형태로는 pair 간 완전 공유가 어렵다.

- source-node-dependent tensor-product message
- edge-to-node aggregation 목적지
- 총 에너지에 대한 force/stress backward 경로

### 2.3 관련 연구

NequIP [1]는 등변 메시지 패싱 기반 원자간 퍼텐셜의 대표적인 출발점이다. SevenNet [2]은 이러한 계열의 모델을 실제 대규모 시뮬레이션에서 사용할 수 있도록 배포 및 병렬 추론 경로를 제공한다. FlashTP [3]는 tensor product와 gather/scatter를 fused kernel로 최적화하지만, directed edge 자체의 geometry-side 중복을 없애지는 않는다. 반면 NEMP [4]와 같은 연구는 메시지 패싱 표현 자체를 더 크게 바꾸는 방향에 가깝다.

본 연구는 이들 중간에 위치한다. 모델 수식과 학습된 파라미터를 유지한 채, runtime execution layout만 pair-aware하게 바꾸어 exact reuse를 얻고자 한다. 따라서 본 논문은 FlashTP처럼 TP kernel을 직접 최적화한 연구도 아니고, NEMP처럼 모델 정의 자체를 바꾼 연구도 아니다.

### 2.4 LAMMPS가 이웃 리스트를 제공해도 오버헤드가 남는 이유

LAMMPS는 neighbor list를 제공하므로 Python/ASE 경로에서의 그래프 구성 일부는 줄어들 수 있다. 그러나 이것이 pair execution 오버헤드를 완전히 제거하지는 않는다. 실제 배포 경로에서는 여전히 다음 비용이 남는다.

- cell shift 처리
- edge tensor materialization
- pair metadata 생성 또는 topology cache 검증
- `pair_edge_vec` 구성
- 총 에너지에 대한 autograd 기반 force/stress backward
- 병렬 환경에서 ghost communication 및 reverse communication

따라서 LAMMPS가 이웃 리스트를 준다는 사실만으로 “pair execution 오버헤드가 없어야 한다”고 보기 어렵다. 본 연구의 성능 해석은 이 점을 전제로 한다.

## 3. 현재 구현

현재 레포지토리 구현은 **pair-major execution**이 아니라 **pair-aware geometry-side reuse**이다.

먼저 directed edge 목록으로부터 다음과 같은 pair metadata를 구성한다.

- edge-to-pair map
- reverse direction mask
- canonical forward edge index
- reverse edge index
- pair-has-reverse mask
- topology signature

그 다음 canonical 방향 기준으로 pair마다 다음 항을 한 번만 계산한다.

- pair distance
- radial basis
- cutoff 적용 edge embedding
- spherical harmonics

역방향 spherical harmonics는 parity 부호를 곱해 복원한다. 또한 filter network는 pair마다 한 번만 수행되어 pair-level `weight_nn` 결과를 만든다.

그러나 tensor product 메시지 생성은 여전히 directed edge 기준이다. 즉 현재 구현은 geometry-side 중복은 줄이지만, 메시지 생성과 aggregation, force/stress backward는 pair-major로 통합하지 못한다.

표 1은 현재 구현 상태를 요약한 것이다.

| 구성 요소 | 현재 상태 | 구현 여부 |
| --- | --- | --- |
| Pair metadata / topology signature | 사용 가능 | 예 |
| Pair-wise radial / cutoff 재사용 | 사용 가능 | 예 |
| Parity 기반 reverse SH 복원 | 사용 가능 | 예 |
| Pair-wise `weight_nn` 재사용 | 사용 가능 | 예 |
| Pair-major tensor-product kernel | 미구현 | 아니오 |
| Pair-major fused reduction | 미구현 | 아니오 |
| FlashTP + pair-major fused 경로 | 미구현 | 아니오 |
| LAMMPS topology-epoch cache 통합 | 미구현 | 아니오 |
| Distributed backward pruning | 미구현 | 아니오 |

## 4. 실험 설정

### 4.1 데이터셋과 coverage

현재 local public inventory에는 총 40개 항목이 있으며, 공식 `OMat24`, 공식 `sAlex`, `ANI-1ccx` 로더 추가 후 31개 데이터셋을 로컬 캐시로 직접 벤치마킹할 수 있다. 나머지 9개는 graph-only mirror, raw 파일 부재, gated access로 인해 제외되었다.

![그림 2](./assets/quadrant_dataset_map.png)

*그림 2. 대표 workload map. 본 연구에서는 크기뿐 아니라 평균 directed neighbors로 나타나는 밀도도 함께 고려한다.*

![그림 3](./assets/support_status.png)

*그림 3. all-public local benchmark inventory의 지원 상태. 공식 tarball 및 `ANI-1ccx` 로더 추가 후 31개 데이터셋을 직접 벤치마킹하였다.*

### 4.2 측정 정책

초기 all-public sweep은 cold 1회와 짧은 repeated window를 이용해 steady-state를 추정하였다. 그러나 small graph에서는 반복 초반의 warm-up 및 autotuning 영향이 강하게 남아 headline 수치가 과대평가될 수 있었다. 따라서 본 논문에서는 다음 원칙을 따른다.

1. broad sweep은 coverage 확인용으로 사용한다.
2. 최종 성능 주장은 warm-up 이후 stable region만을 사용한 representative recheck를 기준으로 한다.
3. intrusive profiling은 절대 성능이 아니라 단계별 해석용으로만 사용한다.

### 4.3 평가 지표

본 논문에서 사용하는 지표는 다음과 같다.

- baseline 대비 에너지 차이
- baseline 대비 힘 차이
- cold latency
- post-warm-up stable latency
- 단계별 profile 및 stage share
- 그래프 크기, edge 수, 평균 directed neighbors

## 5. 실험 결과

### 5.1 정확도 검증

표 2는 주요 benchmark suite에서 baseline과 비교한 최대 차이를 요약한 것이다.

| 벤치마크 | 에너지 지표 | 힘 지표 | 최대 에너지 차이 | 최대 힘 차이 |
| --- | --- | --- | ---: | ---: |
| All-public local | baseline 대비 차이 | baseline 대비 차이 | `6.104e-05 eV` | `2.441e-04 eV/A` |
| Real e3nn pair | e3nn baseline 대비 차이 | e3nn baseline 대비 차이 | `0` | `9.155e-05 eV/A` |
| FlashTP real | e3nn baseline 대비 차이 | e3nn baseline 대비 차이 | `0` | `7.63e-05 eV/A` |

worst all-public local case는 `OMat24 official`이었으며, 대부분의 workload에서는 힘 차이가 `1e-6 ~ 1e-5` 수준에 머물렀다. 이는 모델 의미가 달라졌다기보다 부동소수점 합산 순서 차이에 가까운 결과로 해석하는 것이 타당하다.

### 5.2 warm-up artifact 교정

초기 broad sweep에서 가장 강한 outlier는 `qm9_hf`였다. 짧은 repeated window에서는 약 `3.7x`의 speedup처럼 보였으나, 이는 warm-up artifact였다.

![그림 4](./assets/qm9_warmup_artifact.png)

*그림 4. `qm9_hf`의 warm-up artifact. 반복 초반의 transient가 짧은 repeated window에 섞이면 steady-state 결과가 정반대로 해석될 수 있다.*

warm-up이 지배적인 앞선 반복을 제외하면 `qm9_hf`는 오히려 손해 사례이다. 따라서 본 논문은 broad sweep headline 숫자를 그대로 주장에 사용하지 않는다.

### 5.3 representative stable recheck

표 3은 warm-up 이후 stable region만을 사용해 다시 측정한 representative 결과이다.

| 데이터셋 | workload 유형 | 원자 수 | Baseline stable median (ms) | Pair stable median (ms) | Baseline / Pair |
| --- | --- | ---: | ---: | ---: | ---: |
| `qm9_hf` | small molecular | 29 | 28.59 | 47.14 | 0.61 |
| `iso17` | small molecular | 19 | 29.04 | 47.23 | 0.61 |
| `salex_train_official` | periodic medium | 132 | 151.56 | 147.37 | 1.03 |
| `oc20_s2ef_train_20m` | periodic medium | 225 | 125.32 | 118.60 | 1.06 |
| `omat24_1m_official` | periodic large | 160 | 232.09 | 228.99 | 1.01 |
| `mptrj` | periodic large | 444 | 424.73 | 419.95 | 1.01 |

![그림 5](./assets/rechecked_representative_latency.png)

*그림 5. warm-up 이후 stable representative latency. small molecular workload에서는 손해가 크고, larger periodic workload에서는 본전에서 소폭 이득이 나타난다.*

![그림 6](./assets/rechecked_representative_speedup.png)

*그림 6. warm-up correction 이후의 representative speedup. 현재 구현은 범용 가속기가 아니라 workload-dependent한 부분 최적화임을 보여준다.*

교정된 대표 결과는 다음과 같은 해석을 뒷받침한다.

1. 현재 구현은 small molecular workload에서 대체로 손해이다.
2. large periodic workload에서는 break-even에서 소폭의 양의 이득이 남는다.
3. 따라서 현재 구현을 “범용 가속기”로 주장하는 것은 과장이다.

### 5.4 우리가 실제로 줄인 부분과 줄이지 못한 부분

현재 구현의 강점은 tensor product 자체의 감소가 아니라, **geometry-side exact reuse**에 있다. representative profile run에서 baseline의 spherical harmonics 비중을 보면 이 점이 분명해진다.

표 4. Baseline representative profile에서의 SH 비중

| 데이터셋 | 대표 workload | Baseline model total 중 SH 비중 |
| --- | --- | ---: |
| `mptrj` | large periodic | `18.1%` |
| `md22_double_walled_nanotube` | large dense periodic | `26.4%` |
| `spice_2023` | small molecular | `0.19%` |

즉, large periodic graph에서는 SH 자체가 결코 무시할 수 없는 비중을 차지하지만, small molecular graph에서는 SH 비중이 거의 없다. 이 차이가 workload-dependent 결과의 중요한 원인이다.

또한 load 관점에서 보면 현재 구현은 다음을 정확히 줄인다.

표 5. `mptrj` 대표 샘플에서의 load 변화

| 항목 | Baseline load | Pair load | 변화 |
| --- | ---: | ---: | --- |
| SH / radial / cutoff 관련 edge 수 | `27612` | `13806` | 절반 감소 |
| `weight_nn` 입력 row 수 | `138060` | `69030` | 절반 감소 |
| Message TP load | `138060` | `138060` | 변화 없음 |

표 5는 현재 구현의 본질을 잘 보여준다. 구면조화함수와 geometry-side 입력은 exact reuse로 줄었지만, 실제 메시지 생성을 담당하는 tensor product의 directed workload는 그대로 남아 있다. 따라서 end-to-end 향상이 제한적일 수밖에 없다.

### 5.5 왜 성능이 제한적으로만 좋아지는가

현재 구현의 런타임 비용은 개략적으로 다음과 같이 볼 수 있다.

- pair metadata 생성 또는 cache 검증
- geometry-side reuse를 위한 추가 bookkeeping
- 줄어든 SH / radial / cutoff / `weight_nn`
- 그대로 남아 있는 tensor product
- 그대로 남아 있는 aggregation
- 총 에너지에 대한 force/stress backward

여기서 force/stress 계산은 최종 MLP readout 뒤의 사소한 후처리가 아니다. 총 에너지에 대해 `edge_vec`로 미분을 수행하므로, gradient는 edge embedding, convolution, tensor product, self interaction, readout을 모두 다시 지난다. 즉 추론에서의 backward는 실제로 주요 비용 경로이며, 현재 구현은 이 경로를 줄이지 못한다.

정리하면, 현재 구현은 “SH 재사용 하나만으로도 다 빨라진다”는 주장을 뒷받침하지 않는다. 대신 “SH 및 geometry-side exact reuse는 가능하며, 그것만으로는 충분하지 않다”는 점을 보여준다.

## 6. 논의

### 6.1 본 연구가 실제로 달성한 것

현재 구현은 다음 네 가지를 명확히 입증한다.

1. pair-aware geometry reuse는 모델 수식을 바꾸지 않고 구현 가능하다.
2. reverse spherical harmonics는 parity 부호로 안정적으로 복원 가능하다.
3. baseline 대비 정확도는 실용적으로 유지된다.
4. 현재 병목은 tensor product, aggregation, force/stress backward에 남아 있다는 점이 명확히 드러난다.

### 6.2 본 연구가 아직 달성하지 못한 것

현재 구현은 다음을 아직 제공하지 못한다.

- pair-major tensor-product execution
- pair-major online reduction
- FlashTP와의 tight pair-major 결합
- LAMMPS neighbor rebuild epoch와 연계된 topology cache
- distributed backward / communication pruning

따라서 본 연구를 pair-major 전체 실행 엔진으로 표현하는 것은 부정확하다. 현재 결과는 pair-major 설계로 가기 위한 **정확한 진단과 선행 구현**으로 보는 것이 맞다.

### 6.3 왜 후속 pair-major 재설계가 필요한가

본 연구는 구면조화함수 계산값 재사용과 geometry-side 중복 제거만을 위해 pair metadata 생성, reverse-edge bookkeeping, topology cache 검사와 같은 추가 오버헤드를 감수하였다. 그럼에도 `OC20`, `sAlex`, `OMat24`, `MPtrj`와 같은 일부 large periodic workload에서는 baseline 대비 소폭의 성능 향상이 관측되었다.

이 사실은 중요한 해석을 가능하게 한다. 현재와 같은 metadata/topology 계열의 오버헤드를 이미 부담하는 상황에서도 양의 이득이 남는다면, 후속 연구에서 동일한 pair 단위를 tensor product와 aggregation까지 유지하는 **pair-major runtime**을 구현할 경우, 현재 남아 있는 directed message 비용과 backward 경로의 일부 중복까지 더 줄일 수 있다. 즉, 본 연구의 결과는 pair-major 전체 재설계가 더 **유의미한 성능 향상**으로 이어질 가능성을 시사한다.

물론 이는 현재 논문이 직접 증명한 결과는 아니다. 본 논문이 보여주는 것은 “geometry-only reuse만으로도 일부 favorable workload에서 양의 효과가 남는다”는 점이며, 그 위에 pair-major tensor-product 실행을 올렸을 때 더 큰 효과가 날 것이라는 가설은 후속 연구에서 검증되어야 한다.

### 6.4 국내 학회 선행 연구로서의 위치

본 논문은 전체 실행 계층을 pair-major로 재설계한 최종 결과가 아니라, 그 방향으로 가기 위한 **국내 학회 선행 연구**의 성격을 가진다. 따라서 본 논문에서는 현재 구현의 범위를 엄격히 geometry-side reuse로 한정하고, pair-major tensor-product, accelerator co-design, distributed runtime 개선은 후속 연구로 명시한다.

이와 같은 구분은 과장된 성능 주장을 피하고, 현재 코드와 실험 결과에 의해 직접 뒷받침되는 범위만을 논문 본문에 남긴다는 점에서 중요하다.

## 7. 결론

본 연구는 SevenNet에서 등변 GNN 원자간 퍼텐셜 추론의 pair-symmetric 재사용 기회를 구현하고 분석하였다. 현재 구현은 pair-major execution이 아니라 pair-aware geometry-side reuse이며, distance, radial basis, cutoff, spherical harmonics, pair-level filter input을 재사용하거나 복원한다. baseline 대비 에너지와 힘은 거의 동일하게 유지되었고, 일부 large periodic workload에서는 소폭의 성능 개선이 확인되었다. 반면 small molecular workload에서는 대체로 손해를 보았다.

핵심 결론은 다음과 같다. 첫째, 구면조화함수와 geometry-side exact reuse 자체는 실현 가능하다. 둘째, 그것만으로는 충분하지 않으며 tensor product, aggregation, force/stress backward가 여전히 주요 병목이다. 셋째, 현재 구현이 추가 오버헤드를 감수하고도 일부 favorable workload에서 양의 이득을 보였다는 사실은, 향후 pair-major tensor-product 실행 재설계가 더 큰 성능 향상을 가져올 가능성을 뒷받침한다.

따라서 본 논문은 최종 pair-major 런타임의 완성본이 아니라, 그 필요성과 방향을 데이터로 보여주는 선행 연구로 위치짓는 것이 적절하다.

## 참고문헌

1. S. Batzner, A. Musaelian, L. Sun, et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, vol. 13, 2453, 2022.
2. Y. Park, et al., “SevenNet: a graph neural network interatomic potential package supporting efficient multi-GPU parallel molecular dynamics simulations,” *Journal of Chemical Theory and Computation*, 2024.
3. J. Lee, et al., “FlashTP: fused, sparsity-aware tensor product for machine learning interatomic potentials,” *Proceedings of the International Conference on Machine Learning*, 2024.
4. Y. Zhang and H. Guo, “Node-Equivariant Message Passing for Efficient and Accurate Machine Learning Interatomic Potentials,” *Chemical Science*, 2025.
