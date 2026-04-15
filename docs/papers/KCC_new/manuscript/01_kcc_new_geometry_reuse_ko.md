# 등변 그래프 신경망 원자간 퍼텐셜 추론을 위한 원자쌍 기반 기하 정보 재사용의 구현과 병목 분석

**Implementation and Bottleneck Analysis of Pair-Based Geometric Reuse for Equivariant GNN Interatomic Potential Inference**

김민창  
아주대학교 분산병렬컴퓨팅 연구소 WiseLab  
minchang111@ajou.ac.kr

## 요 약

본 논문은 NequIP 계열 등변 그래프 신경망 원자간 퍼텐셜에서 하나의 원자쌍이 두 개의 방향 연결(`i -> j`, `j -> i`)로 반복 표현된다는 점에 주목한다. 기존 SevenNet 실행 방식에서는 이 두 연결을 서로 독립적으로 처리하므로 거리, radial basis, cutoff, spherical harmonics와 같은 기하 정보가 중복 계산된다. 우리는 이 중복을 줄이기 위해 reverse edge pair를 묶고, 재사용 가능한 geometry-side 값을 pair 단위로 한 번만 계산하는 `geometry_only` 실행 방식을 구현하였다. 이 방식은 pair-major tensor product 엔진이 아니며, 메시지 생성과 힘 계산 경로는 기존 edge-major 구조를 유지한다.

실험은 단일 `NVIDIA GeForce RTX 4090` 환경에서 로컬에서 바로 벤치 가능한 공개 데이터셋 31개 전체를 대상으로 수행하였다. 메인 비교는 `SevenNet baseline`과 `SevenNet + geometry_only`의 end-to-end latency이며, 각 대표 샘플에 대해 warmup 3회, 반복 30회로 평균과 표준편차를 측정하였다. 또한 정확도 보존 여부를 확인하기 위해 같은 샘플을 다시 계산하여 baseline 기준 에너지/힘 차이를 warmup 2회, 반복 30회로 측정하였다. 그 결과 에너지 차이의 중앙값은 두 경우 모두 `0 eV`였고, 힘 차이의 중앙값은 baseline `1.189e-06 eV/A`, geometry_only `1.809e-06 eV/A`로 매우 작았다.

반면 generic SevenNet calculator 경로에서의 성능은 아직 baseline을 넘지 못했다. 31개 전체 기준 median speedup은 `0.9877배`, geometric mean은 `0.9864배`였으며, 모든 데이터셋에서 geometry_only가 baseline보다 약간 느렸다. 그러나 intrusive forward-only 진단에서는 large/dense representative system에서 `3.164 ms -> 3.092 ms`로 약한 이득이 관측되었고, LAMMPS serial 경로에서는 upstream pair-metadata fast path를 적용했을 때 pair metadata 시간이 `4.612 ms -> 0.304 ms`로 `15.19배` 감소하였다. 이는 현재 손해의 주원인이 exact reuse 수식 자체가 아니라, pair를 다시 edge로 펼치는 구조와 pair metadata 생성 방식에 있음을 뜻한다.

따라서 본 논문의 기여는 “이미 빨라졌다”는 선언보다, 정확도 보존형 geometry-side exact reuse를 SevenNet에 실제로 구현하고, 왜 아직 end-to-end 승리로 이어지지 않는지를 실험적으로 규명했다는 데 있다. 이 결과는 이후 pair-major message path와 pair-aware reduction, upstream neighbor/pair integration으로 넘어갈 명확한 우선순위를 제공한다.

**주제어**: 등변 그래프 신경망, 원자간 퍼텐셜, 구면조화함수, 실행 시간 최적화, SevenNet, 병목 분석

## 1. 서 론

등변 그래프 신경망 기반 원자간 퍼텐셜은 분자와 재료 시뮬레이션에서 높은 정확도를 제공하는 대표적인 방법이다. NequIP와 SevenNet 같은 모델은 원자 사이의 상대적 거리와 방향을 직접 다루면서 회전 대칭을 보존하고, 복잡한 다체 상호작용을 잘 표현한다. 그러나 계산 관점에서 보면 이 계열 모델은 방향이 있는 연결을 기본 단위로 사용하기 때문에, 물리적으로 같은 원자쌍이 두 번 처리되는 구조를 가진다.

가장 단순한 예가 `i -> j`와 `j -> i`이다. 이 두 연결은 출발 원자 feature가 다르므로 최종 메시지는 서로 같지 않다. 그러나 거리, radial basis, cutoff, spherical harmonics처럼 순수하게 pair geometry에서 나오는 값은 양방향에서 같거나 간단한 parity 규칙으로 정확히 복원할 수 있다. 따라서 모델 수식이나 학습 파라미터를 바꾸지 않고도, 실행 순서만 바꾸어 geometry-side 중복 계산을 줄일 수 있다.

문제는 이 아이디어를 실제 runtime에 넣었을 때의 효과가 명확하지 않다는 점이다. 단순히 “SH를 한 번 덜 계산하니 빨라질 것”이라고 말하는 것은 부족하다. pair를 찾는 오버헤드, pair를 다시 edge로 펼치는 비용, 힘 계산을 위한 backward 경로가 남아 있기 때문이다. 즉 정확도 보존형 exact reuse를 실제 코드에 넣고, 어떤 부분이 줄고 어떤 부분이 남는지를 같이 분석해야 한다.

본 논문은 SevenNet에 `geometry_only` 실행 방식을 구현하고, 이를 31개 공개 데이터셋 전체에서 repeat 30 기준으로 다시 평가한다. 또한 메인 end-to-end 비교 외에 intrusive geometry breakdown과 LAMMPS upstream pair-metadata fast path를 함께 측정하여, 현재 구조에서 실제 병목이 어디에 있는지 정리한다. 본 논문의 핵심 기여는 다음과 같다.

1. reverse edge pair를 이용한 geometry-side exact reuse runtime을 SevenNet에 구현하였다.
2. 같은 representative sample을 반복 계산해, 제안 방식이 energy/force 정확도를 사실상 바꾸지 않음을 보였다.
3. 현재 generic calculator path에서 why-not-fast를 정량적으로 분석하고, upstream pair integration이 실제로 큰 오버헤드를 줄인다는 점을 보였다.

## 2. 배 경

### 2.1 SevenNet 추론 흐름

SevenNet 추론은 크게 네 단계로 나눌 수 있다.

1. 원자 그래프 생성
2. edge geometry 준비
3. interaction block을 통한 메시지 생성과 feature 갱신
4. readout과 total energy 계산, 그리고 force 계산

여기서 edge geometry 준비 단계에는 거리, radial basis, cutoff, spherical harmonics가 들어간다. interaction block은 이 geometry 정보와 출발 원자 feature를 결합해 메시지를 만들고, readout은 최종 node feature를 원자별 에너지로 바꾼 뒤 전체 에너지로 합친다. force는 total energy를 좌표 또는 `EDGE_VEC`에 대해 미분해 얻는다.

### 2.2 재사용 가능한 값과 재사용이 어려운 값

현재 구조에서 pair 단위로 재사용 가능한 값은 다음과 같다.

- distance
- radial basis
- cutoff
- spherical harmonics
- pair 단위 `weight_nn` 입력

반면 현재 구현에서 재사용이 어려운 값은 다음과 같다.

- 출발 원자 feature에 의존하는 최종 메시지
- 목적지 원자별 aggregation
- force 계산을 위한 generic backward 경로

즉 본 논문은 메시지 생성 전체를 pair-major로 다시 설계한 것이 아니라, 메시지 생성 앞단의 geometry-side 중복을 먼저 줄이는 단계라고 보는 것이 맞다.

## 3. 제안 방법

### 3.1 geometry_only 실행 방식

본 논문의 `geometry_only` 실행 방식은 reverse 관계인 directed edge 두 개를 하나의 pair로 묶는다. 이후 대표 방향에서 pair geometry를 한 번만 계산하고, 역방향 spherical harmonics는 parity 부호를 이용해 복원한다. pair 단위 `weight_nn` 입력도 한 번만 계산한다.

개념적으로는 다음과 같다.

```text
directed edges
-> reverse pair grouping
-> pair distance / radial / cutoff / SH 계산 1회
-> 필요한 경우 edge-major 경로로 다시 확장
-> 기존 convolution / readout / force path
```

이 방식은 모델 출력 정의를 바꾸지 않으며, 따라서 정확도 변화 없이 실행 시간만 건드리는 최적화로 해석할 수 있다.

### 3.2 현재 구현의 한계

현재 구현은 pair-major 전체 엔진이 아니다. 즉, pair geometry를 계산한 뒤에도 다시 edge-major convolution과 force path를 사용한다. 이 때문에 다음 비용이 그대로 남는다.

- pair를 다시 edge tensor로 펼치는 비용
- edge-major weight expansion
- force 계산 시 generic autograd 경로

따라서 현재 결과는 pair-major 전체 구현의 성능 상한이 아니라, exact geometry reuse만 먼저 넣었을 때의 결과로 해석해야 한다.

## 4. 실험 설정

### 4.1 환경

실험은 단일 `NVIDIA GeForce RTX 4090`에서 수행하였다. `PyTorch 2.7.1+cu126`를 사용했으며, 메인 headline latency는 warmup 3회, 반복 30회로 측정하였다. 정확도 반복은 warmup 2회, 반복 30회로 측정하였다.

| 항목 | 값 |
| --- | --- |
| device | NVIDIA GeForce RTX 4090 |
| framework | PyTorch 2.7.1+cu126 |
| dataset 수 | 31 |
| headline latency | warmup 3, repeat 30 |
| accuracy repeat | warmup 2, repeat 30 |

### 4.2 데이터셋과 메트릭

현재 로컬에서 바로 벤치 가능한 공개 데이터셋 31개 전체를 사용하였다. 각 데이터셋에서 representative sample 1개를 선택해 반복 측정하였다. 각 sample에 대해 `natoms`, `num_edges`, `avg_neighbors_directed`를 기록하고, size-density bucket도 함께 저장하였다.

메인 메트릭은 다음과 같다.

- steady-state latency mean / std / median / p95
- speedup = `baseline / geometry_only`
- absolute energy difference vs baseline
- maximum absolute force difference vs baseline

## 5. 실험 결과

### 5.1 정확도 보존

먼저 제안 방식이 출력을 바꾸지 않는지 확인하였다. 같은 representative sample을 baseline 기준으로 다시 계산한 결과, 에너지 차이의 중앙값은 두 경우 모두 `0 eV`였고, 힘 차이의 중앙값은 baseline `1.189e-06 eV/A`, geometry_only `1.809e-06 eV/A`였다. 최악의 경우에도 baseline은 에너지 `1.831e-04 eV`, 힘 `3.662e-04 eV/A`, geometry_only는 에너지 `2.441e-04 eV`, 힘 `4.883e-04 eV/A` 수준이었다.

즉 geometry_only는 출력 정확도를 사실상 바꾸지 않는 exact reuse runtime으로 볼 수 있다.

![그림 1. 에너지 차이](../figures/figure_06_accuracy_energy.png)

![그림 2. 힘 차이](../figures/figure_07_accuracy_force.png)

### 5.2 메인 end-to-end latency

메인 결과는 기대와 다르게 나왔다. 31개 전체 기준으로 geometry_only는 모든 데이터셋에서 baseline보다 약간 느렸다. median speedup은 `0.9877배`, geometric mean은 `0.9864배`였고, 31개 중 win은 `0`, loss는 `31`이었다.

| 지표 | 값 |
| --- | --- |
| datasets | 31 |
| median speedup | 0.9877x |
| geometric mean speedup | 0.9864x |
| wins | 0 |
| losses | 31 |

이 결과는 large/dense 쪽으로 갈수록 좋아진다는 약한 경향은 남아 있지만, 실제로 1.0을 넘지는 못한다는 뜻이다. 예를 들어 `num_edges >= 3000`인 20개 데이터셋의 median speedup은 `0.988`, `num_edges < 3000`인 11개 데이터셋의 median speedup은 `0.983`이었다. `avg_neighbors_directed >= 40`인 경우도 median `0.988`로, small_sparse보다 손해 폭이 작지만 아직 baseline을 넘지는 못한다.

![그림 3. 전체 latency](../figures/figure_01_end_to_end_latency.png)

![그림 4. 데이터셋별 speedup](../figures/figure_02_end_to_end_speedup.png)

![그림 5. speedup vs edge 수](../figures/figure_03_speedup_vs_edges.png)

![그림 6. speedup vs 평균 이웃 수](../figures/figure_04_speedup_vs_neighbors.png)

![그림 7. bucket별 speedup](../figures/figure_05_bucket_speedup.png)

### 5.3 왜 아직 느린가: geometry_only 내부 진단

메인 end-to-end 결과만 보면 제안 방식이 실패한 것처럼 보일 수 있다. 그러나 intrusive forward-only 진단은 더 구체적인 그림을 보여준다. large/dense representative system인 `bulk_large`에서는 baseline forward total이 `3.164 ms`, geometry_only forward total이 `3.092 ms`로, intrusive 기준이지만 geometry-side reuse 자체는 이미 약한 이득을 보였다. 반면 `bulk_small`과 `dimer_small`에서는 각각 `0.974x`, `0.949x`로 손해가 남아 있었다.

| system | baseline forward total (ms) | geometry_only forward total (ms) | pair expand (ms) | pair weight expand (ms) | pair geometry (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bulk_large | 3.164 | 3.092 | 0.039 | 0.037 | 0.192 |
| bulk_small | 2.980 | 3.059 | 0.041 | 0.033 | 0.192 |
| dimer_small | 2.812 | 2.962 | 0.038 | 0.033 | 0.179 |

이 표는 현재 손해의 핵심이 SH 수식 자체가 아니라, pair를 다시 edge로 펼치는 구조와 weight expansion에 있음을 보여준다. 즉 geometry reuse의 수학적 방향은 맞지만, runtime이 아직 pair 상태를 충분히 오래 유지하지 못한다.

### 5.4 upstream pair metadata는 실제로 효과가 있다

LAMMPS serial 경로에서 upstream pair-metadata fast path를 넣은 결과는 더 분명하다. `bulk_large`에서 legacy pair metadata 시간은 `4.612 ± 0.052 ms`, upstream pair metadata 시간은 `0.304 ± 0.014 ms`로 `15.19배` 감소하였다. `bulk_small`에서도 `0.439 ± 0.013 ms -> 0.101 ± 0.008 ms`로 `4.35배` 감소하였다. total compute도 각각 `1.018배`, `1.037배` 줄었다.

![그림 8. LAMMPS pair metadata](../figures/figure_08_lammps_pair_metadata.png)

| system | case | pair metadata mean (ms) | total compute mean (ms) |
| --- | --- | ---: | ---: |
| bulk_large | geometry_only_legacy | 4.612 | 170.320 |
| bulk_large | geometry_only_upstream | 0.304 | 167.296 |
| bulk_small | geometry_only_legacy | 0.439 | 168.856 |
| bulk_small | geometry_only_upstream | 0.101 | 162.871 |

즉 “upstream에서 이미 아는 neighbor/pair 정보를 직접 넘기게 한다”는 방향은 실제로 효과가 확인된 개선이다. 이는 현재 generic calculator path에서도 pair metadata와 pair reconstruction을 더 upstream으로 밀어야 한다는 근거가 된다.

## 6. 논의

현재 결과를 가장 정확하게 해석하면 다음과 같다.

첫째, geometry_only exact reuse는 정확도를 사실상 바꾸지 않는다. 이는 제안 방식이 수학적으로 틀렸거나 출력 일관성을 깨는 최적화가 아니라는 뜻이다.

둘째, generic SevenNet calculator path에서 geometry_only가 아직 baseline보다 느린 이유는 exact reuse 아이디어 자체보다 runtime 구조에 있다. intrusive forward-only 진단에서 large/dense case가 거의 본전 또는 약한 이득을 보이고, LAMMPS upstream pair-metadata fast path가 분명한 효과를 보인다는 사실이 이를 뒷받침한다.

셋째, 따라서 현재 논문의 novelty는 “이미 대규모 속도 향상을 달성했다”가 아니라, 정확도 보존형 pair-aware geometry reuse를 실제 코드에 넣고, 어디서부터 무엇을 먼저 고쳐야 하는지를 정량적으로 밝힌 데 있다. 특히 upstream pair metadata는 구현이 비교적 쉽고 결과가 확실한 개선으로 확인되었다.

본 논문의 결과는 다음 구현 우선순위를 분명히 제시한다.

1. pair metadata를 가능한 한 upstream에서 직접 넘기게 하기
2. pair를 다시 edge로 펼치는 확장 비용 줄이기
3. edge-major force path를 더 pair-aware하게 바꾸기
4. 이후 pair-major message path와 reduction으로 확장하기

## 7. 결 론

본 논문은 SevenNet에서 reverse edge pair를 이용한 geometry-side exact reuse를 구현하고, 이를 31개 공개 데이터셋 전체에서 repeat 30 기준으로 다시 검증하였다. geometry_only는 에너지와 힘의 출력 정확도를 사실상 바꾸지 않았지만, generic calculator path의 end-to-end latency에서는 아직 baseline을 이기지 못했다. 전체 median speedup은 `0.9877배`, geometric mean은 `0.9864배`였다.

그러나 이 결과는 연구 방향 자체가 틀렸다는 의미가 아니다. intrusive forward-only 진단은 large/dense system에서 geometry-side reuse가 거의 본전 또는 약한 이득을 만들 수 있음을 보였고, LAMMPS upstream pair-metadata fast path는 pair metadata 시간을 `4.35배`에서 `15.19배`까지 줄였다. 즉 현재 남은 문제는 exact reuse 수식보다 runtime 구조에 더 가깝다.

따라서 본 논문의 기여는 정확도 보존형 pair-aware geometry reuse의 구현, generic runtime에서의 실패 양상 규명, 그리고 이후 pair-major runtime으로 가기 위한 우선순위 제시에 있다. 이는 이후 upstream neighbor integration, pair-major message path, pair-aware reduction을 설계하는 직접적인 근거가 된다.

## 참 고 문 헌

[1] S. Batzner, A. Musaelian, L. Sun, et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, vol. 13, 2453, 2022.  
[2] Y. Park, et al., “SevenNet: a graph neural network interatomic potential package supporting efficient multi-GPU parallel molecular dynamics simulations,” *Journal of Chemical Theory and Computation*, 2024.  
[3] J. Lee, et al., “FlashTP: fused, sparsity-aware tensor product for machine learning interatomic potentials,” 2024.  
[4] A. Musaelian, et al., “Learning local equivariant representations for large-scale atomistic dynamics,” 2023.

