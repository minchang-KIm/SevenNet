# KCC Paper Structure Plan

이 문서는 KCC 제출용 논문을 어떤 논리 구조로 작성할지 정리한 설계안이다. 목표는 단순히 빠른 결과를 나열하는 것이 아니라, 문제정의, 제안기법의 정확한 범위, 조건부 성능 이득, 남은 병목, 후속 연구 방향이 한 흐름으로 읽히게 만드는 것이다.

## 1. 논문 위치와 핵심 주장

### 1.1 한 줄 포지셔닝

본 논문은 NequIP/SevenNet 계열 등변 GNN-MLIP에서 양방향 edge가 반복 계산하는 geometry-side 항목을 원자쌍 단위로 정확히 재사용하고, 이 재사용이 정확도를 바꾸지 않으면서 large/dense workload에서 성능 이득으로 이어질 수 있는 조건을 실험적으로 규명한다.

### 1.2 중심 문제

NequIP/SevenNet은 원자쌍 `(i, j)`를 두 방향 edge `i -> j`, `j -> i`로 처리한다. 이때 다음 항목은 양방향에서 중복된다.

- distance
- cutoff
- radial basis
- spherical harmonics, 단 방향 반전 parity 필요
- pair 단위 `weight_nn` 입력

하지만 다음 항목은 단순 재사용하면 안 된다.

- source node feature에 의존하는 message
- tensor product 결과 전체
- aggregation 결과
- force backward 전체

따라서 이 논문의 핵심은 “모든 것을 절반으로 줄였다”가 아니라, “정확히 재사용 가능한 geometry-side와 재사용하면 안 되는 node-conditioned computation을 분리했다”이다.

### 1.3 최종 주장

논문에서 방어 가능한 최종 주장은 다음이다.

1. reverse edge pair에서 geometry-side reusable terms는 정확히 재사용할 수 있다.
2. `geometry_only`는 energy/force 정확도를 사실상 바꾸지 않는다.
3. pure `geometry_only`는 현재 edge-major runtime에서는 아직 end-to-end win을 만들지 못한다.
4. restored `full` two-pass는 pair-aware scheduling까지 포함했을 때 large/dense 및 일부 large/sparse workload에서 성능 이득을 보인다.
5. small/sparse workload에서는 pair-aware overhead가 이득보다 크다.
6. 따라서 novelty는 단순 speedup claim이 아니라, exact reuse formulation, 정확도 보존 검증, workload condition 발견, 병목 분석, pair-major runtime으로 가는 설계 근거다.

### 1.4 쓰면 안 되는 주장

- “현재 구현은 완전한 pair-major fused kernel이다.”
- “message 전체를 parity로 재사용한다.”
- “force backward 전체를 절반으로 줄인다.”
- “geometry_only만으로 large/dense에서 이미 빨라진다.”
- “large/dense면 항상 빨라진다.”
- “FlashTP 결합 효과를 이미 증명했다.”

## 2. KCC 형식 기준 전체 구성

KCC 스타일의 짧은 학술 논문 구조로 다음 순서를 권장한다.

1. 제목
2. 저자/소속
3. 요약
4. 주제어
5. 서론
6. 배경 및 문제정의
7. 제안 방법
8. 실험 설정
9. 실험 결과
10. 논의
11. 결론
12. 참고문헌

페이지 제한이 빡빡하면 `배경 및 문제정의`와 `제안 방법`을 합치고, `논의`를 결과 마지막 subsection으로 줄인다.

권장 분량 배분:

| 절 | 분량 | 역할 |
| --- | ---: | --- |
| 요약 | 150~220단어 | 문제, 방법, 핵심 수치, 한계까지 압축 |
| 서론 | 0.5쪽 | 왜 이 문제가 중요한지와 논문 기여 |
| 배경/문제정의 | 0.5쪽 | reusable vs non-reusable 계산 구분 |
| 제안 방법 | 0.8~1.0쪽 | geometry_only, restored full two-pass, metadata |
| 실험 설정 | 0.4쪽 | 환경, 데이터셋, 반복, metric |
| 결과 | 1.2~1.5쪽 | 정확도, 성능 조건, 병목 분석 |
| 논의/결론 | 0.5쪽 | novelty, 한계, 후속 방향 |

## 3. 제목 후보

### 3.1 보수적 제목

등변 그래프 신경망 원자간 퍼텐셜 추론을 위한 원자쌍 기반 기하 정보 재사용의 구현과 병목 분석

장점:

- 현재 구현과 실험을 과장하지 않는다.
- `geometry_only`와 restored `full`의 차이를 설명하기 쉽다.

### 3.2 성능 조건을 강조하는 제목

등변 그래프 신경망 원자간 퍼텐셜에서 원자쌍 기반 기하 정보 재사용과 대규모 그래프 조건부 가속

장점:

- large/dense 조건부 win을 제목에서 보여준다.

주의:

- “가속”이 제목에 들어가면 small/sparse loss와 `geometry_only` loss를 본문에서 명확히 설명해야 한다.

### 3.3 최종 추천

KCC 본문에는 3.2가 더 매력적이지만, 리뷰 방어력은 3.1이 높다. 현재 결과의 안정성을 고려하면 3.1을 유지하고, 부제 또는 요약에서 “large/dense workload 조건부 가속”을 강조하는 것이 좋다.

## 4. 초록 구조

초록은 다음 5문장 구조가 가장 안전하다.

1. 문제: NequIP/SevenNet은 양방향 edge 구조 때문에 geometry-side 계산을 중복 수행한다.
2. 제안: reverse edge pair를 묶어 distance/radial/cutoff/SH/weight input을 pair 단위로 재사용한다.
3. 정확도: `geometry_only`는 energy/force 차이가 부동소수점 수준임을 보인다.
4. 성능: pure geometry_only는 아직 end-to-end win이 아니지만, restored full two-pass는 31개 중 18개에서 win, `num_edges >= 3000`에서 18/20 win, median `1.014x`를 보인다.
5. 의미: pair-aware 실행은 small/sparse에서는 불리하지만 large/dense에서 유리하며, pair-major runtime과 upstream neighbor integration으로 확장할 근거를 제공한다.

초록에 넣을 핵심 수치:

- `geometry_only` force diff median: `1.809e-06 eV/A`
- `geometry_only` energy diff median: `0 eV`
- restored `full` median speedup: `1.0099x`
- restored `full` wins: `18/31`
- `num_edges >= 3000`: `18/20` win, median `1.014x`
- `large_dense`: `16/17` win, median `1.013x`

## 5. 1장 서론

### 5.1 작성 목적

서론은 “왜 지금 이 최적화가 필요한가”를 설득해야 한다.

### 5.2 핵심 흐름

1. MLIP는 DFT보다 빠르면서 높은 정확도를 제공하기 때문에 MD/재료/촉매 시뮬레이션에서 중요하다.
2. NequIP, SevenNet, MACE, Allegro 같은 등변 GNN은 방향과 회전 대칭을 다루기 위해 spherical harmonics와 tensor product를 사용한다.
3. 이 구조는 정확도에는 유리하지만, directed edge 기반이라 같은 원자쌍이 양방향으로 반복 처리된다.
4. 물리적으로 같은 pair에서 distance/radial/cutoff/SH 같은 geometry-side 항목은 중복된다.
5. 그러나 message 전체는 source node feature가 달라서 단순 재사용할 수 없다.
6. 본 논문은 이 경계를 정확히 나누고, 재사용 가능한 부분을 구현하며, 어떤 workload에서 실제 성능 이득으로 이어지는지 보인다.

### 5.3 서론에 넣을 그림/표

권장 그림:

- 신규 개념 그림 1개 필요: `directed edges -> reverse pair -> geometry reuse -> two directional messages`
- 이미 있는 런타임 그림 후보:
  - `docs/papers/KCC_new/figures/runtime/pair_metadata_pipeline.svg`
  - `docs/papers/KCC_new/figures/runtime/force_backward_pipeline.svg`

서론에서는 너무 복잡한 그래프보다 개념도 하나가 좋다.

### 5.4 서론 마지막 contribution 문장

권장 문장:

> 본 논문의 기여는 다음과 같다. (1) reverse edge pair에서 정확히 재사용 가능한 geometry-side 항목과 재사용하면 안 되는 node-conditioned 항목을 구분한다. (2) SevenNet runtime에 geometry-side exact reuse를 구현하고 energy/force 정확도가 유지됨을 보인다. (3) restored full two-pass 실행에서 large/dense workload가 유리해지는 조건을 31개 공개 데이터셋으로 규명한다. (4) pair->edge expansion, pair metadata, force path가 남은 병목임을 분석하고 pair-major runtime 설계 방향을 제시한다.

### 5.5 서론 참고문헌

- NequIP: 등변 GNN-MLIP의 대표 출발점 [1]
- SevenNet: 본 연구의 구현 기반 [2]
- MACE/Allegro: 등변/고차 message passing 계열 관련 연구 [3], [4]
- OC20/MD22 등: MLIP 벤치마크 중요성 [6], [7]

## 6. 2장 배경 및 문제정의

### 6.1 작성 목적

이 장은 전문 용어를 줄이고, “무엇을 줄일 수 있고 무엇은 못 줄이는지”를 독자가 납득하게 해야 한다.

### 6.2 넣을 내용

#### 6.2.1 SevenNet/NequIP 추론 파이프라인

다음 흐름을 짧게 설명한다.

```text
원자 좌표
-> neighbor graph / directed edges
-> edge vector, distance
-> radial basis, cutoff, spherical harmonics
-> convolution / tensor product message
-> aggregation
-> readout energy
-> force = -dE/dR
```

#### 6.2.2 pair symmetry와 정확한 재사용 조건

수식으로 짧게 쓴다.

```text
r_ij = x_j - x_i
r_ji = -r_ij
|r_ij| = |r_ji|
RBF(|r_ij|) = RBF(|r_ji|)
cutoff(|r_ij|) = cutoff(|r_ji|)
Y_lm(-r) = (-1)^l Y_lm(r)
```

따라서 geometry-side는 재사용 가능하다.

#### 6.2.3 message는 왜 단순 재사용 불가인가

반드시 써야 하는 부분이다.

```text
m_{j -> i} = TP(h_j, Y(r_ij), w_ij)
m_{i -> j} = TP(h_i, Y(r_ji), w_ij)
```

일반적으로 `h_i != h_j`이므로 `m_{i -> j}`는 `m_{j -> i}`의 부호 반전이 아니다. 이것을 명확히 쓰면 논문이 과장되지 않는다.

#### 6.2.4 문제정의

문제정의는 다음처럼 적는다.

> 본 연구의 문제는 등변 GNN-MLIP의 directed edge 표현에서 출력 정확도를 바꾸지 않고 재사용 가능한 pair geometry 계산을 줄이는 것이다. 동시에, 이 재사용이 실제 end-to-end latency 개선으로 이어지는 workload 조건과, 개선을 막는 runtime 병목을 밝히는 것을 목표로 한다.

### 6.3 이 장에 넣을 표

표 1 추천: 재사용 가능/불가능 항목 구분

| 항목 | pair reuse 가능 여부 | 이유 |
| --- | --- | --- |
| distance | 가능 | 방향 반전에도 크기 동일 |
| cutoff | 가능 | distance의 함수 |
| radial basis | 가능 | distance의 함수 |
| spherical harmonics | 가능 | parity로 복원 |
| weight input | 가능 | radial embedding 공유 |
| message / TP output | 일반적으로 불가 | source node feature가 다름 |
| aggregation | 직접 재사용 불가 | destination node가 다름 |
| force backward 전체 | 직접 절반화 불가 | multi-layer graph dependency |

### 6.4 참고문헌

- NequIP [1]
- SevenNet [2]
- MACE [3]
- Allegro [4]
- FlashTP/TP 최적화 연구 [5]

## 7. 3장 제안 방법

### 7.1 작성 목적

방법 장에서는 `geometry_only`와 restored `full`을 섞지 말고 역할을 분리한다.

### 7.2 3.1 Geometry-Side Exact Reuse

작성할 내용:

1. reverse edge pair를 만든다.
2. pair representative edge에서 geometry를 계산한다.
3. reverse SH는 parity sign으로 복원한다.
4. pair embedding / pair weight input을 공유한다.
5. 이 과정은 모델 파라미터와 수식을 바꾸지 않는다.

넣을 코드/구현 기준:

- [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)
- `PAIR_EDGE_VEC`
- `PAIR_EDGE_EMBEDDING`
- `PAIR_EDGE_ATTR`
- `PAIR_EDGE_REVERSE_ATTR`
- `EDGE_PAIR_MAP`

권장 문장:

> geometry_only는 모델의 의미를 바꾸는 approximation이 아니라, directed edge 두 개가 공유하는 pair geometry를 한 번만 계산하는 exact reuse runtime이다.

### 7.3 3.2 Pair-Aware Full Two-Pass Execution

작성할 내용:

1. `geometry_only`만으로는 pair->edge expansion과 edge-major force path 비용이 남는다.
2. restored `full`은 pair weight를 공유하고, forward/reverse message를 two-pass로 처리한다.
3. 두 방향 message는 각각 계산한다. 한쪽을 parity로 복사하지 않는다.
4. single-pass `torch.cat` 버전은 호출 수는 줄였지만 GPU에서 느려졌다.
5. 따라서 현재 성능 경로는 restored two-pass이다.

넣을 구현 기준:

- [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)
- `_pair_forward`
- `msg_forward`
- `msg_reverse`
- `message_gather`

권장 문장:

> restored full two-pass는 pair geometry와 pair weight를 공유하지만, node feature가 다른 두 방향 message는 각각 계산한다. 따라서 이 방식은 수학적으로 안전하며, 동시에 large graph에서 pair-aware scheduling의 이득을 활용한다.

### 7.4 3.3 Upstream Pair Metadata

작성할 내용:

1. ASE/generic path에서는 pair metadata를 뒤에서 재구성한다.
2. LAMMPS는 neighbor loop에서 이미 edge 정보를 알고 있다.
3. 따라서 pair id / reverse map / forward-backward index를 upstream에서 직접 넘기면 metadata overhead가 줄어든다.

넣을 구현 기준:

- [pair_e3gnn.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn.cpp)
- LAMMPS upstream pair metadata fast path

### 7.5 이 장에 넣을 그림

그림 1: proposed runtime overview

```text
Baseline:
edge i->j: geometry + weight + message
edge j->i: geometry + weight + message

Geometry only:
pair(i,j): geometry once
edge i->j / j->i: expanded message path

Restored full:
pair(i,j): geometry + weight once
forward message pass
reverse message pass
aggregation
```

그림 2: single-pass regression explanation

```text
single-pass:
cat(src, dst, filter, weight) -> one TP -> one gather

problem:
extra materialization + changed kernel shape + scatter pattern
```

KCC 지면이 부족하면 그림 2는 논의 절의 짧은 문장으로 대체한다.

## 8. 4장 실험 설정

### 8.1 작성 목적

실험 설정은 비교 가능성을 명확히 해야 한다. 특히 `geometry_only`와 restored `full`의 목적이 다르다는 점을 표로 고정한다.

### 8.2 실험 환경

넣을 내용:

- GPU: `NVIDIA GeForce RTX 4090`
- PyTorch: `2.7.1+cu126`
- dataset: local benchmarkable public datasets 31개
- repeat: 30
- warmup: 3
- accuracy repeat: warmup 2, repeat 30
- headline timing: synchronized repeated wall-clock timing

### 8.3 비교 case

표 2 추천:

| case | 목적 | 직접 비교 대상 | 논문 역할 |
| --- | --- | --- | --- |
| baseline | SevenNet 기본 실행 | 모든 case | 기준 |
| geometry_only | exact geometry reuse | baseline | 정확도 보존/순수 reuse 검증 |
| restored full two-pass | pair-aware scheduling | baseline | 조건부 성능 이득 검증 |
| LAMMPS upstream metadata | metadata 병목 진단 | legacy metadata | 후속 최적화 근거 |

### 8.4 데이터셋 구성

전체 31개는 부록/전체표로 유지한다. 본문 대표 6개는 다음을 권장한다.

| dataset | 역할 | bucket | 이유 |
| --- | --- | --- | --- |
| `md22_buckyball_catcher` | large_dense positive | large_dense | MD22, restored full top win |
| `oc20_s2ef_val_ood_ads` | large_sparse positive | large_sparse | OC20, top win |
| `mptrj` | large material representative | large_dense | Materials Project 계열 |
| `qm9_hf` | small_sparse negative | small_sparse | QM9, small graph 대표 |
| `rmd17` | small_sparse negative | small_sparse | MD force benchmark |
| `md22_stachyose` | boundary/counterexample | large_dense | threshold 근처지만 loss |

### 8.5 사분면 처리

현재 31개 데이터셋에는 `small_dense`가 없다. 이를 억지로 만들지 않는다.

권장 문장:

> 본 연구의 local benchmarkable dataset 31개에서 canonical threshold를 적용하면 `small_dense` 영역은 비어 있다. 따라서 본문 대표 실험은 `large_dense`, `large_sparse`, `small_sparse`, boundary case로 구성한다.

## 9. 5장 실험 결과

결과 장은 RQ 형태로 쓰면 논리가 깔끔하다.

### 9.1 RQ1: 정확도는 유지되는가?

주장:

`geometry_only`는 energy/force 출력을 사실상 바꾸지 않는다.

넣을 데이터:

- [table_04_accuracy_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_04_accuracy_summary.md)
- [figure_06_accuracy_energy.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_06_accuracy_energy.png)
- [figure_07_accuracy_force.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_07_accuracy_force.png)

본문 수치:

- energy diff median: `0 eV`
- geometry_only force diff median: `1.809e-06 eV/A`

해석:

> 이 결과는 제안 방식이 모델을 근사하거나 학습 파라미터를 바꾸는 것이 아니라, 동일한 계산을 다른 실행 순서로 수행함을 보여준다.

### 9.2 RQ2: pure geometry_only는 왜 아직 빠르지 않은가?

주장:

`geometry_only`는 정확하지만 현재 edge-major runtime에서는 pair->edge expansion과 index/select overhead 때문에 baseline보다 약간 느리다.

넣을 데이터:

- [figure_02_end_to_end_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_02_end_to_end_speedup.png)
- [table_02_end_to_end_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_02_end_to_end_summary.md)
- [pair_profiler_interpretation.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/reports/pair_profiler_interpretation.md)

본문 수치:

- median speedup: `0.9877x`
- wins: `0/31`

해석:

> pure geometry reuse만으로는 현재 generic calculator path의 overhead를 넘지 못한다. 이는 아이디어 실패가 아니라, runtime이 pair 상태를 충분히 오래 유지하지 못한다는 증거다.

### 9.3 RQ3: 어떤 조건에서 성능 이득이 생기는가?

주장:

restored `full` two-pass는 large/dense 및 일부 large/sparse workload에서 성능 이득을 만든다.

넣을 데이터:

- [figure_09_full_restored_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_09_full_restored_speedup.png)
- [figure_10_full_restored_condition_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_10_full_restored_condition_speedup.png)
- [table_07_full_restored_end_to_end_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_07_full_restored_end_to_end_summary.md)
- [table_08_full_restored_condition_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_08_full_restored_condition_summary.md)

본문 수치:

- restored full median: `1.0099x`
- wins: `18/31`
- `num_edges >= 3000`: `18/20` win, median `1.014x`
- `large_dense`: `16/17` win, median `1.013x`
- `small_sparse`: `0/11` win, median `0.613x`

해석:

> pair-aware 실행은 고정 오버헤드가 있으므로 작은 그래프에서는 불리하다. 그러나 edge 수와 평균 이웃 수가 커지면 pair geometry와 pair weight 공유, forward/reverse scheduling 이득이 오버헤드를 넘어선다.

### 9.4 RQ4: 병목은 어디에 남아 있는가?

주장:

남은 병목은 SH 수식 자체보다 pair->edge expansion, pair weight expansion, metadata, force path다.

넣을 데이터:

- [geometry_only_breakdown_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/geometry_only_breakdown/reports/geometry_only_breakdown_report.md)
- [table_05_geometry_breakdown_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_05_geometry_breakdown_summary.md)
- [figure_08_lammps_pair_metadata.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_08_lammps_pair_metadata.png)
- [table_06_lammps_pair_metadata_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_06_lammps_pair_metadata_summary.md)

본문 수치:

- intrusive geometry_only representative speedup: 약 `0.965x ~ 0.970x`
- LAMMPS pair metadata:
  - `bulk_large`: `4.636 ms -> 0.322 ms`, `14.40x`
  - `bulk_small`: `0.441 ms -> 0.100 ms`, `4.41x`

해석:

> upstream에서 이미 알고 있는 neighbor/pair 정보를 직접 넘기면 metadata 병목은 크게 줄어든다. 하지만 total compute는 force path와 다른 runtime 병목이 남아 있어 같은 비율로 줄지 않는다.

### 9.5 결과 장에서 피해야 할 구성

결과 장을 다음 순서로 쓰면 안 된다.

1. restored full speedup부터 제시
2. geometry_only 정확도는 뒤에 짧게 언급
3. small sparse loss는 숨김

이렇게 쓰면 “성능을 위해 임의로 경로를 바꿨다”는 인상을 준다.

권장 순서:

1. 정확도 보존
2. pure geometry_only 한계
3. restored full 조건부 win
4. 병목 분석
5. 후속 설계

## 10. 6장 논의

### 10.1 논의에서 강조할 점

1. 정확도 보존과 성능 이득은 다른 축이다.
2. `geometry_only`는 정확도 보존과 수식 정당성을 증명한다.
3. restored `full`은 성능 조건을 보여준다.
4. small/sparse에서는 이득이 없다는 것도 기여다. 어떤 조건에서 쓰면 안 되는지 밝힌 것이기 때문이다.
5. single-pass `torch.cat` regression은 중요한 실험적 교훈이다. GPU에서는 호출 수 감소가 항상 성능 개선이 아니다.

### 10.2 한계

반드시 써야 할 한계:

- 현재 구현은 완전한 pair-major fused kernel이 아니다.
- force backward를 custom pair backward로 줄이지 않았다.
- FlashTP와의 결합은 후속 연구다.
- 2-GPU LAMMPS parallel canonical 결과는 아직 본문 headline으로 쓰지 않는다.
- local 31개에는 `small_dense` 사분면이 없다.

### 10.3 후속 연구

후속 연구는 다음 순서로 쓴다.

1. upstream neighbor/pair metadata integration 확대
2. pair 상태를 edge-major로 너무 빨리 펼치지 않는 pair-major message path
3. pair-aware aggregation/reduction
4. force path의 pair-aware custom backward
5. FlashTP 또는 fused TP backend와의 상보적 결합

## 11. 결론

결론은 세 문단이면 충분하다.

1. 문제와 방법 요약
   - reverse edge pair의 geometry-side exact reuse 구현
2. 결과 요약
   - 정확도 보존
   - restored full 조건부 speedup
   - small/sparse loss
   - metadata 병목 감소
3. 의미
   - 현재 구현은 최종 답이 아니라 pair-major runtime으로 가는 정확한 중간 단계

권장 결론 문장:

> 본 연구는 등변 GNN-MLIP의 양방향 edge 구조에서 정확히 재사용 가능한 geometry-side 항목을 분리하고, 이를 SevenNet runtime에 구현하였다. 순수 geometry reuse는 출력 정확도를 보존하지만 현재 edge-major runtime에서는 아직 성능 이득을 만들지 못한다. 반면 pair-aware scheduling을 포함한 restored full two-pass 경로는 large/dense workload에서 일관된 이득을 보였으며, 이는 pair-major execution으로 확장할 실험적 근거를 제공한다.

## 12. 그림과 표 배치 계획

### 12.1 본문 우선순위

| 우선순위 | 그림/표 | 파일 | 넣을 위치 | 역할 |
| ---: | --- | --- | --- | --- |
| 1 | 재사용 가능/불가능 표 | 새로 작성 | 2장 | 논문 논리의 핵심 |
| 2 | 정확도 그래프 | `figure_06`, `figure_07` | 5.1 | exact reuse 검증 |
| 3 | restored full speedup | `figure_09` | 5.3 | win 복구 |
| 4 | 조건별 speedup | `figure_10` | 5.3 | large/dense 조건 |
| 5 | 대표 데이터셋 표 | `dataset_selection_strategy` 기반 새 표 | 4장 또는 5장 | cherry-pick 방지 |
| 6 | LAMMPS metadata | `figure_08` | 5.4 | 후속 최적화 근거 |

### 12.2 부록/보조

| 자료 | 파일 | 역할 |
| --- | --- | --- |
| 31개 geometry_only 전체표 | `table_02_end_to_end_summary.md` | pure exact reuse 한계 |
| 31개 restored full 전체표 | `table_07_full_restored_end_to_end_summary.md` | 전체 결과 투명성 |
| 조건별 표 | `table_08_full_restored_condition_summary.md` | claim 근거 |
| profiler 해석 | `pair_profiler_interpretation.md` | 병목 설명 |

## 13. 대표 데이터셋 배치

본문 대표 데이터셋은 다음 6개를 추천한다.

| dataset | 절 | 역할 |
| --- | --- | --- |
| `md22_buckyball_catcher` | 결과/프로파일링 | large_dense strong win |
| `oc20_s2ef_val_ood_ads` | 결과/프로파일링 | large_sparse top win |
| `mptrj` | 결과/프로파일링 | large material representative |
| `qm9_hf` | 결과/프로파일링 | small_sparse negative |
| `rmd17` | 결과 표 | small molecular dynamics benchmark |
| `md22_stachyose` | 논의 | boundary/counterexample |

본문에서 이 6개를 쓰면 다음 균형이 맞는다.

- 성능 win만 고르지 않음
- MLIP 인지도 있는 데이터셋 포함
- large/dense 논리와 small/sparse overhead 논리 모두 설명
- boundary case로 과장 방지

## 14. 참고문헌 후보와 위치

KCC 본문에서는 10~14개 정도면 충분하다.

### 14.1 필수 참고문헌

1. NequIP
   - S. Batzner et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, 2022.
   - 위치: 서론, 배경
   - 역할: 등변 GNN-MLIP 대표 모델

2. SevenNet
   - Y. Park et al., “Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations,” 2024.
   - 위치: 서론, 실험 설정
   - 역할: 구현 기반, parallel MD context

3. MACE
   - I. Batatia et al., “MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields,” NeurIPS 2022.
   - 위치: 관련연구
   - 역할: 고차 등변 message passing 관련 모델

4. Allegro
   - A. Musaelian et al., “Learning local equivariant representations for large-scale atomistic dynamics,” 2023.
   - 위치: 관련연구
   - 역할: local equivariant representation과 대규모 atomistic dynamics

5. FlashTP 또는 fused TP 계열
   - J. Lee et al., “FlashTP: fused, sparsity-aware tensor product for machine learning interatomic potentials,” 2024.
   - 위치: 관련연구, 논의
   - 역할: TP backend 최적화와 본 연구의 geometry-side 최적화가 다른 축임을 설명

### 14.2 데이터셋 참고문헌

6. OC20
   - L. Chanussot et al., “The Open Catalyst 2020 (OC20) Dataset and Community Challenges,” *ACS Catalysis*, 2021.
   - 위치: 실험 설정
   - 역할: 촉매/표면 large workload 대표

7. MD22
   - S. Chmiela et al. 또는 MD22 원 논문, “Accurate global machine learning force fields for molecules with hundreds of atoms.”
   - 위치: 실험 설정
   - 역할: 큰 분자 동역학 workload 대표
   - 주의: 최종 bibliography 작성 시 MD22 원 논문의 venue/year를 확인할 것

8. QM9
   - R. Ramakrishnan et al., “Quantum chemistry structures and properties of 134 kilo molecules,” *Scientific Data*, 2014.
   - 위치: 실험 설정
   - 역할: small molecular benchmark 대표

9. rMD17 / MD17
   - S. Chmiela et al., “Machine learning of accurate energy-conserving molecular force fields,” *Science Advances*, 2017.
   - rMD17 revised reference는 최종 bibliography에서 추가 확인
   - 위치: 실험 설정
   - 역할: molecular dynamics benchmark

10. SPICE
   - P. Eastman et al., “SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials,” *Scientific Data*, 2023.
   - 위치: 실험 설정 또는 부록
   - 역할: modern molecular dataset

11. Materials Project / MPtrj 관련
   - CHGNet 또는 M3GNet reference를 사용 가능
   - 위치: 실험 설정
   - 역할: materials workload 대표

### 14.3 선택 참고문헌

12. SchNet
   - 초기 continuous-filter neural network baseline
   - QM9/MD17 context에서 사용

13. DimeNet/DimeNet++
   - directional message passing과 OC20 baseline context

14. GemNet-OC / EquiformerV2
   - OC20 large-scale baseline context

### 14.4 참고문헌 배치 원칙

본문에서 레퍼런스는 다음처럼 붙인다.

- “등변 GNN-MLIP는 NequIP 이후 분자 및 재료 시뮬레이션에서 널리 사용된다” → [1], [3], [4]
- “SevenNet은 NequIP 계열 구조를 기반으로 multi-GPU MD를 지원한다” → [2]
- “OC20, MD22, QM9/rMD17은 각각 촉매, 대형 분자, 소형 분자 동역학을 대표한다” → [6], [7], [8], [9]
- “TP backend 최적화와 geometry-side reuse는 서로 다른 축이다” → [5]

## 15. KCC 최종 원고 작성 순서

실제 원고는 다음 순서로 수정하는 것이 효율적이다.

1. 초록을 먼저 다시 쓴다.
2. 서론 contribution을 위 4개로 고정한다.
3. 배경에 reusable/non-reusable 표를 넣는다.
4. 방법에서 `geometry_only`와 restored `full`을 분리한다.
5. 실험 설정에 selected dataset 6개와 31개 전체 사용 원칙을 쓴다.
6. 결과를 RQ1~RQ4로 재배치한다.
7. 논의에서 single-pass regression과 limitations를 숨기지 않는다.
8. 결론에서 조건부 성능 이득과 후속 pair-major 방향을 강조한다.

## 16. 최종 논리 골격

최종 논리 골격은 아래 한 문단으로 요약된다.

> 등변 GNN-MLIP는 directed edge를 사용하기 때문에 하나의 원자쌍이 두 방향으로 처리된다. 이때 distance, radial basis, cutoff, spherical harmonics처럼 pair geometry에만 의존하는 항목은 정확히 재사용 가능하지만, source node feature에 의존하는 message와 force backward 전체는 단순 재사용할 수 없다. 본 연구는 이 경계를 명확히 나누고 SevenNet에 geometry-side exact reuse를 구현했다. `geometry_only`는 energy/force 정확도를 보존했지만 현재 edge-major runtime에서는 아직 성능 이득을 만들지 못했다. 반면 pair weight 공유와 forward/reverse two-pass scheduling을 포함한 restored `full` 경로는 large/dense workload에서 일관된 이득을 보였고, small/sparse에서는 overhead 때문에 불리했다. 이 결과는 pair-aware 실행의 적용 조건과 pair-major runtime으로 확장해야 할 병목을 동시에 제시한다.
