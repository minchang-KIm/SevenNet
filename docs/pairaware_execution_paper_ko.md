# Pair-aware Geometry Reuse와 호환형 Execution Layer를 이용한 Equivariant GNN 기반 Interatomic Potential의 실행 최적화

## 초록
Equivariant graph neural network(GNN) 기반 interatomic potential(IP)은 높은 정확도를 제공하지만, 원자쌍 상호작용을 양방향 directed edge로 전개하는 실행 구조 때문에 동일한 기하 정보를 반복 계산하는 비효율을 가진다. 특히 하나의 원자쌍에 대해 거리, radial basis, cutoff, spherical harmonics가 각 방향 edge에서 중복 계산되며, 이는 수식 자체의 요구라기보다 실행 단위가 원자쌍 수준의 공통성과 방향 의존성을 분리하지 못하기 때문에 발생한다. 본 논문에서는 기존 모델의 수식 구조와 학습된 파라미터를 유지한 채 이러한 중복을 제거하기 위한 pair-aware geometry reuse 방법과 baseline-compatible execution layer를 제안한다. 제안 방법은 directed edge 집합으로부터 주기적 이미지까지 보존하는 canonical pair mapping을 구성하고, 원자쌍 수준에서 거리, radial basis, cutoff를 1회 계산한 뒤 방향 반전에 대해서는 구면조화함수의 parity 관계 $Y_l(-\mathbf{r}) = (-1)^l Y_l(\mathbf{r})$를 적용하여 edge 표현을 복원한다. 이 구조는 기존 Equivariant MLIP의 convolution 및 tensor product 수식을 변경하지 않으면서 `edge_embedding` 단계만 치환하도록 설계되었다. SevenNet 코드베이스와 LAMMPS `e3gnn/parallel` 경로에 통합한 결과, 에너지, 힘, 응력 출력은 baseline과 수치적으로 일치하였다. 또한 quartz 3087-atom MD 벤치마크에서 제안 구조는 기존 실행 경로와 호환되며 FlashTP와 조합 가능함을 확인하였다. 다만 현재 CPU 구현에서는 pair construction이 `torch.unique(..., dim=0)`에 의해 지배되어 pair-aware만으로 즉각적인 속도 향상은 나타나지 않았다. 이러한 결과는 공통 기하 정보 재사용 자체의 타당성과 함께, 향후 최적화의 핵심 병목이 기하 계산이 아니라 pair mapping 단계에 있음을 보여준다.

**주요어:** equivariant GNN, interatomic potential, geometry reuse, pair-aware execution, SevenNet, LAMMPS

## 1. 서론
최근 equivariant GNN 기반 interatomic potential은 원자 수준 시뮬레이션에서 높은 정확도와 데이터 효율성을 보이며 널리 사용되고 있다 [1]. 이러한 모델은 일반적으로 원자 간 상호작용을 directed edge 기반 message passing으로 표현하며, 상대 위치 벡터, 거리, radial basis, cutoff, spherical harmonics 등 기하 특징을 edge 단위에서 생성한다. 그러나 실제 물리적 상호작용은 원자쌍 단위로 정의되는 경우가 많고, directed edge 전개는 같은 원자쌍 $(i,j)$에 대해 $(i \rightarrow j)$와 $(j \rightarrow i)$를 모두 유지한다. 이 때문에 동일한 거리와 radial embedding, 그리고 부호 반전만으로 연결되는 방향별 구면조화 표현이 반복 계산된다.

기존 연구는 equivariant message passing 구조의 정립 [1], 새로운 상호작용 표현의 도입 [2], node-equivariant 재구성 [3], tensor product kernel의 가속 [4] 등 다양한 방향에서 효율 향상을 시도하였다. 그러나 이미 학습된 기존 GNN-IP의 수식과 파라미터를 유지하면서, directed edge 수준의 중복 기하 계산을 원자쌍 수준의 공통 계산으로 끌어올리는 실행 계층은 충분히 정립되지 않았다. 특히 어떤 항이 정확히 공유 가능하며, 어떤 항이 방향 반전 parity만으로 복원 가능한지에 대한 정식화가 부족했다.

본 논문의 목표는 다음과 같다. 첫째, periodic image까지 보존하는 pair interaction unit을 정의하고, pair 수준에서 정확히 재사용 가능한 기하 항을 정식화한다. 둘째, 이를 기존 SevenNet 계열 모델에 직접 적용 가능한 pair-aware execution layer로 구현하여, convolution과 tensor product 수식을 바꾸지 않고도 기하 계산 경로를 대체한다. 셋째, CPU 및 LAMMPS 경로 실험을 통해 정확성, 호환성, 그리고 실제 병목 지점을 분석한다.

본 논문의 기여는 다음과 같다.

1. directed edge 기반 equivariant MLIP에서 공유 가능한 기하 항과 parity 기반 복원 항을 분리하는 pair-aware geometry reuse 공식을 제시한다.
2. 기존 모델의 수식 구조와 학습 파라미터를 유지한 채 적용 가능한 baseline-compatible execution layer를 SevenNet에 구현한다.
3. Quartz LAMMPS 벤치마크와 모듈 프로파일링을 통해, 제안 방법의 수치적 동등성과 현재 CPU 병목이 pair mapping 단계에 있음을 실험적으로 보인다.

## 2. 배경 및 문제 정의
### 2.1 Directed edge 기반 기하 계산의 중복
일반적인 equivariant MLIP에서는 이웃 원자쌍 $(i,j)$마다 directed edge $e_{ij}$와 $e_{ji}$를 모두 구성한다. 각 edge에 대해 상대 위치 벡터 $\mathbf{r}_{ij}$, 거리 $r_{ij} = ||\mathbf{r}_{ij}||$, radial basis $g(r_{ij})$, cutoff $c(r_{ij})$, spherical harmonics $Y_l(\hat{\mathbf{r}}_{ij})$가 계산된다. 그러나 다음 성질이 즉시 성립한다.

- $r_{ij} = r_{ji}$
- $g(r_{ij}) = g(r_{ji})$
- $c(r_{ij}) = c(r_{ji})$
- $Y_l(\hat{\mathbf{r}}_{ji}) = Y_l(-\hat{\mathbf{r}}_{ij}) = (-1)^l Y_l(\hat{\mathbf{r}}_{ij})$

즉, 거리 관련 스칼라 항은 완전히 동일하고, 구면조화 항 역시 parity 부호만 다르다. 그럼에도 불구하고 기존 실행 경로는 이를 edge 단위로 매번 계산한다.

### 2.2 기존 가속화 연구와 공백
기존 가속화 연구는 주로 세 가지 방향을 따른다. 첫째, message passing 표현 자체를 재설계한다 [1,2,3]. 둘째, tensor product나 kernel 수준을 최적화한다 [4]. 셋째, 특정 하드웨어나 런타임 백엔드에 맞춘 저수준 최적화를 수행한다. 반면, 이미 구축된 GNN-IP에서 directed edge의 공통성을 pair 수준으로 승격시키는 실행 계층은 상대적으로 덜 다루어졌다. 본 논문은 이 공백을 채우기 위해 모델 수식은 유지하고 실행 단위만 재구성하는 호환형 접근을 취한다.

## 3. 제안 방법
### 3.1 Pair interaction unit 정의
본 논문은 directed edge 집합으로부터 원자쌍 기반의 canonical pair를 구성한다. 단순한 $(\min(i,j), \max(i,j))$ 인덱스만으로는 periodic image가 다른 edge를 구분할 수 없으므로, canonicalized edge vector를 함께 pair descriptor에 포함한다. 즉 pair key는 다음으로 정의된다.

\[
\mathbf{k}_{ij} = \left[\min(i,j), \max(i,j), \tilde{\mathbf{r}}_{ij}\right]
\]

여기서 $\tilde{\mathbf{r}}_{ij}$는 canonical orientation으로 정렬된 edge vector이다. 이 정의는 다음 두 조건을 만족한다.

1. 같은 원자쌍의 양방향 edge는 동일한 pair에 매핑된다.
2. periodic image나 self-image가 다른 경우는 서로 다른 pair로 유지된다.

이를 통해 pair-aware mapping은 directed edge를 무손실로 undirected pair 집합으로 축약하되, 주기경계 조건에서 필요한 구분은 그대로 보존한다.

### 3.2 Pair-aware geometry reuse
Pair-aware 경로에서는 먼저 각 directed edge를 canonical pair에 매핑하고, pair 단위 벡터 $\mathbf{r}_p$를 구성한다. 이후 다음 기하 항을 pair 수준에서 1회만 계산한다.

- pair length: $r_p = ||\mathbf{r}_p||$
- radial embedding: $\phi_p = g(r_p)c(r_p)$
- pair spherical attribute: $\mathbf{y}_p = Y(\mathbf{r}_p)$

그 다음 각 directed edge $e$에 대해 pair id를 사용해 $(r_p, \phi_p, \mathbf{y}_p)$를 gather하고, reversed edge에 한해 구면조화 parity를 적용한다.

\[
\mathbf{y}_{e} =
\begin{cases}
\mathbf{y}_p, & \text{if } e \text{ is canonical direction} \\
\mathbf{s} \odot \mathbf{y}_p, & \text{if } e \text{ is reversed}
\end{cases}
\]

여기서 $\mathbf{s}$는 각 irreducible representation 채널의 $(-1)^l$ 부호 벡터이다. 이 과정은 baseline과 동일한 `EDGE_LENGTH`, `EDGE_EMBEDDING`, `EDGE_ATTR`를 반환하므로 이후 convolution 블록은 수정 없이 그대로 사용된다.

### 3.3 Baseline-compatible execution layer
제안 방법의 핵심은 새로운 GNN 수식을 도입하지 않는다는 점이다. 구현은 `EdgeEmbedding.forward` 내부에서 baseline 경로와 pair-aware 경로를 분기하는 방식으로 이루어진다. Pair-aware 경로는 다음 절차를 따른다.

1. directed `edge_index`와 `edge_vec`로부터 canonical pair mapping을 생성한다.
2. pair 수준에서 `pair_length`, `pair_embedding`, `pair_attr`를 계산한다.
3. `edge_to_pair`를 이용해 pair 결과를 directed edge로 gather한다.
4. `edge_is_reversed`에 대해 spherical harmonic parity를 적용한다.
5. 결과를 기존 키 이름으로 저장하여 후속 연산과 완전 호환되도록 한다.

이 구조는 convolution, gate, tensor product, readout 수식을 변경하지 않으며, 기존 체크포인트와 export 경로를 유지한다. 또한 runtime flag로 제어되어 baseline, pair-aware, FlashTP, FlashTP+pair-aware 조합을 동일한 인터페이스로 다룰 수 있다.

### 3.4 Pair-fused operator로의 확장 가능성
현재 구현은 geometry/filter preparation 단계의 pair-aware 치환에 집중한다. 그러나 동일한 pair metadata는 이후 filter 생성과 양방향 message 생성까지 확장 가능하다. 즉 pair mapping, reversal 정보, pair representative tensor를 기반으로 filter와 message를 pair 단위에서 생성한 뒤 edge 방향으로 재구성하는 pair-fused operator 설계가 가능하다. 본 논문에서는 이 방향을 후속 연구 과제로 남기며, 현재 결과는 그 전단계인 pair-aware execution layer의 타당성과 병목 위치를 분명히 보여준다.

## 4. 구현
본 구현은 SevenNet 코드베이스의 `edge_embedding` 경로에 통합되었다. `build_undirected_pair_mapping`은 canonical node pair와 canonical edge vector를 이용해 periodic graph에서도 안전한 pair id를 생성한다. `build_spherical_harmonic_parity_sign`은 spherical harmonics 출력 irreps로부터 채널별 부호 벡터를 구성하고, `apply_spherical_harmonic_parity`는 reversed edge에 대해 $(-1)^l$ 부호를 적용한다.

이 구현의 중요한 특징은 다음과 같다.

- 기존 `IrrepsConvolution.forward` 및 tensor product 수식은 변경하지 않는다.
- 출력 텐서 shape와 의미를 baseline과 동일하게 유지한다.
- 학습, 추론, 체크포인트 로딩, TorchScript export, LAMMPS deploy 경로에서 동일한 runtime mode 플래그 체계를 사용한다.
- pair-aware는 geometry/filter preparation만 바꾸므로 FlashTP 같은 backend 가속과 조합 가능하다.

즉 본 구현은 정확성 보존과 호환성을 우선하는 execution-layer 최적화이며, 모델 재학습 없이 기존 체크포인트에 직접 적용할 수 있다.

## 5. 실험 설정
### 5.1 정확성 검증
수치적 정확성은 세 단계에서 검증하였다.

1. toy edge graph에서 pair mapping의 정확성과 양방향 edge의 pair collapse 여부를 검증하였다.
2. 구면조화 parity 적용 결과가 baseline의 직접 계산과 일치하는지 확인하였다.
3. HfO$_2$ 그래프에 대해 동일한 랜덤 시드로 생성한 baseline 모델과 pair-aware 모델의 총에너지, 힘, 응력을 비교하였다.

검증 허용오차는 에너지에 대해 $10^{-6}$, 힘과 응력에 대해 $10^{-5}$ 수준으로 설정하였다.

### 5.2 CPU 마이크로벤치마크
마이크로벤치마크는 NaCl rocksalt 구조를 기반으로 원자 수를 증가시키며 수행하였다. 목표 원자 수는 256, 2000, 20000으로 설정했고, 실제 생성 구조는 각각 432, 2000, 21296 atoms였다. 비교 항목은 다음과 같다.

- 전체 모델 추론 시간
- `edge_embedding` 단계 시간
- pair-aware 내부 세부 시간: mapping, pair reduction, pair geometry, gather
- baseline geometry 세부 시간: norm, radial, spherical

### 5.3 LAMMPS quartz 벤치마크
실제 시뮬레이션 경로 검증을 위해 LAMMPS `e3gnn/parallel` pair style을 사용한 quartz 벤치마크를 수행하였다. 입력 구조는 `data.quartz`를 `7 x 7 x 7`로 replicate한 3087-atom 시스템이다. 각 실험은 다음 네 가지 variant로 수행하였다.

- `baseline`
- `pairaware`
- `flash`
- `combined` = FlashTP + pair-aware

MD 조건은 `metal` 단위계, `NVT`, 300 K, timestep 0.002 ps이며, `run 110` 이후 `run 100`의 마지막 loop summary를 성능 지표로 사용하였다.

## 6. 실험 결과 및 분석
### 6.1 수치적 동등성
제안 방법은 baseline의 기하 표현을 수치적으로 정확하게 복원하였다. Toy graph 테스트에서 pair mapping은 각 양방향 edge를 동일 pair로 정확히 축약했으며, parity 기반 spherical harmonic 복원은 baseline 직접 계산과 일치했다. 전체 모델 수준에서도 총에너지, 힘, 응력이 설정한 허용오차 내에서 baseline과 동일했다. 이는 제안 방법이 근사나 재학습 없이 기존 모델의 수학적 동작을 유지함을 의미한다.

### 6.2 CPU 프로파일링: 병목은 geometry가 아니라 mapping
CPU 마이크로벤치마크에서 pair-aware는 directed edge 수를 정확히 절반 수준의 undirected pair로 축약하여 재사용 계수 2.0을 달성했다. 그러나 현재 구현에서는 pair construction 비용이 커서 end-to-end 이득이 제한되었다.

다음은 21296-atom case에서의 pair-aware 세부 시간이다.

| 항목 | 시간 (ms) |
|---|---:|
| mapping | 606.233 |
| pair reduction | 4.578 |
| pair geometry | 2.665 |
| gather | 3.126 |

동일 조건의 baseline geometry 시간은 다음과 같다.

| 항목 | 시간 (ms) |
|---|---:|
| norm | 0.226 |
| radial | 2.491 |
| spherical | 2.548 |

즉 실제로 재사용하려는 pair geometry 자체는 매우 작고, 전체 병목은 canonical pair를 생성하기 위한 `torch.unique(..., dim=0)` 기반 mapping 단계에 집중되어 있다. 2000-atom case의 CPU profiler에서도 pair-aware 경로는 `aten::unique_dim`이 약 137.096 ms로 지배적인 반면, baseline의 상위 연산은 norm, spherical harmonics, basis math 등 기대한 geometry kernel이었다. 이 결과는 현재 pair-aware 구조의 한계가 reuse 개념의 실패가 아니라, pair construction 구현 방식의 비용에 있음을 보여준다.

### 6.3 Quartz LAMMPS benchmark
Quartz 3087-atom LAMMPS benchmark의 결과는 표 1과 같다.

| Variant | Step time (ms) | Timesteps/s | Atoms/s | ns/day | Baseline 대비 |
|---|---:|---:|---:|---:|---:|
| baseline | 414.415 | 2.413 | 7448.931 | 0.417 | +0.00% |
| pairaware | 416.943 | 2.398 | 7402.626 | 0.414 | -0.61% |
| flash | 45.364 | 22.044 | 68049.828 | 3.809 | +89.05% |
| combined | 49.196 | 20.327 | 62749.449 | 3.513 | +88.13% |

Pair-aware 단독 경로는 baseline과 사실상 동일한 수준의 step time을 보였으나, 현재 CPU 구현에서는 유의미한 추가 가속을 제공하지 못했다. 반면 FlashTP는 큰 폭의 속도 향상을 보였고, pair-aware는 FlashTP와 조합 가능한 실행 구조임을 확인하였다. 다만 현재 `combined`는 `flash` 단독보다 빠르지 않았으며, 이는 pair-aware의 추가 geometry 재사용 이득보다 pair mapping 오버헤드가 더 크기 때문이다.

별도의 반복 실행(`probe_r7_all`)에서도 결과 순서는 동일했다.

| Variant | Step time (ms) |
|---|---:|
| baseline | 433.585 |
| pairaware | 440.913 |
| flash | 50.827 |
| combined | 51.798 |

즉 quartz 경로에서는 "FlashTP의 고속 경로는 즉시 효과적이며, pair-aware는 아직 CPU에서 mapping 최적화가 선행되어야 한다"는 결론이 반복적으로 재현되었다.

## 7. 논의
본 연구의 핵심 성과는 두 가지다. 첫째, directed edge 기반 equivariant MLIP에서 공통 기하 항과 parity 복원 항을 명확히 분리하여 pair-aware execution을 수식적으로 정당화했다. 둘째, 이 구조를 기존 SevenNet/LAMMPS 경로에 호환적으로 통합해 numerical equivalence를 유지했다.

반면 성능 측면에서는 중요한 제한도 확인되었다. 현재 구현은 periodic image 안전성을 위해 floating-point descriptor에 대한 `torch.unique(..., dim=0, return_inverse=True)`를 사용한다. 이 방식은 correctness에는 유리하지만 CPU에서는 지나치게 비싸다. 따라서 다음 단계의 핵심은 spherical harmonics나 radial basis 계산 자체를 더 줄이는 것이 아니라, pair construction을 더 저렴한 integer-key 또는 sort-segment 기반 방식으로 대체하는 것이다.

이 점은 오히려 향후 연구 방향을 분명히 한다. Pair-aware geometry reuse는 정확히 작동하며, 이제 필요한 것은 그 실행 경로를 pair-fused operator 수준으로 확장하고, pair mapping 자체를 하드웨어 친화적으로 재설계하는 것이다. 특히 GPU나 전용 커널 환경에서는 현재 CPU와 다른 비용 구조를 보일 가능성이 높으므로, 후속 연구에서는 GPU-native pair mapping과 filter/message 단계의 pair fusion을 함께 다룰 필요가 있다.

## 8. 결론
본 논문은 equivariant GNN 기반 interatomic potential에서 양방향 directed edge가 반복 계산하는 공통 기하 항을 pair 수준으로 재구성하는 pair-aware geometry reuse 방법과 baseline-compatible execution layer를 제안했다. 제안 방법은 거리, radial basis, cutoff를 pair 단위에서 1회만 계산하고, spherical harmonics는 parity 관계를 이용해 edge 방향으로 복원함으로써 기존 모델의 수식 구조를 바꾸지 않고 재사용을 가능하게 한다. SevenNet과 LAMMPS 경로에의 통합 결과, 에너지, 힘, 응력의 수치적 동등성과 FlashTP와의 조합 가능성을 확인했다. 현재 CPU 구현에서 즉각적인 속도 향상은 pair mapping 오버헤드로 제한되었지만, 그 병목을 명확히 분리해냈다는 점에서 본 연구는 pair-fused 실행 계층과 커널 수준 최적화로 이어지는 실질적 기반을 제공한다.

## 참고문헌
[1] Batzner, S. et al., “E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials,” *Nature Communications*, 2022.

[2] Musaelian, A. et al., “Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics,” *Nature Communications*, 2023.

[3] Zhang, Y.; Guo, H., “Node-equivariant message passing for efficient and accurate machine learning interatomic potentials,” *Chemical Science*, 17, 3793-3803, 2026.

[4] “FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials,” *ICML*, 2025.

## 부록: 현재 초안에서 바로 보완할 항목
- 영문 제목 및 영문 초록 추가
- 그림 1: directed edge와 pair interaction unit 분해도 삽입
- 표 번호와 그림 번호를 LaTeX 양식에 맞게 재정리
- pair-fused operator를 실제 구현 결과가 아닌 설계 확장으로 유지할지, 후속 연구로 완전히 분리할지 최종 결정
