# GNN-IP Pair Execution 데이터셋 특성 분석 보고서

세미나 발표용 정리 문서  
기준 브랜치: `pair-major`  
기준 커밋: `583e0c1`

## 2026-04-03 Note

- 이 문서는 size-density 및 dataset feature correlation을 설명하는 내부 해석 문서다.
- 이후 all-public benchmark sanitation에서 warm-up artifact가 확인되었으므로, 여기 있는 성능 숫자는 정량 headline이 아니라 정성 해석용으로 사용하는 것이 안전하다.
- 현재 canonical performance summary는 `docs/papers/icpp_pair_execution/03_final_manuscript.md`를 따른다.

## 1. 질문

이 문서는 다음 질문에 답하기 위한 보고서다.

> 현재 구현한 pair execution은 어떤 데이터셋, 어떤 그래프, 어떤 병렬 환경에서 효과가 좋은가?

여기서 중요한 전제는 다음과 같다.

- 이 보고서는 `FlashTP`를 제외한 순수 SevenNet `e3nn` 경로를 기준으로 본다.
- 현재 구현은 아직 TP 자체를 pair-major로 fused한 것이 아니다.
- 현재 구현이 실제로 줄이는 것은 주로 `geometry + spherical harmonics + pair-level weight_nn`의 중복 계산이다.
- 최종 message TP는 여전히 양방향에 대해 따로 계산된다.

즉 이 보고서는 "내 제안기법이 정확히 어디서 이득을 만드는가"를 dataset feature와 연결해서 설명하는 문서다.

## 2. 검증 범위와 근거

### 2.1 실험으로 직접 검증한 것

아래 산출물을 사용했다.

- `bench_runs/real_e3nn_pair/metrics/aggregated.csv`
- `bench_runs/real_e3nn_pair/metrics/samples.csv`
- `bench_runs/real_e3nn_pair/analysis.md`
- `bench_runs/real_e3nn_pair_k3/derived/dataset_inventory.csv`
- `bench_runs/real_e3nn_pair_k3/derived/sample_features.csv`

실험 설정:

- 비교: `e3nn_baseline` vs `e3nn_pair_full`
- 모델: `7net-omni`
- 측정: ASE calculator end-to-end latency
- 샘플: 각 공개셋에서 가장 큰 구조 1개
- 반복: cold-call 1회 + steady-state 3회

### 2.2 코드에서만 추론한 것

분산 병렬 환경에 대한 해석은 아래 경로를 읽고 정리했다.

- `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`
- `sevenn/nn/edge_embedding.py`
- `sevenn/nn/convolution.py`

즉 분산 병렬 부분은 "실험으로 입증"이 아니라 "현재 코드 경로를 읽고 추론"한 결론이다.

## 3. 현재 구현이 실제로 줄이는 계산

현재 구현의 핵심은 아래 두 부분이다.

1. `pair` 기준으로 geometry를 한 번만 만든다.
2. `pair` 기준으로 `weight_nn`을 한 번만 만든다.

구체적으로는:

- `edge_embedding.py`에서 `pair_edge_vec` 기준으로 `basis`, `cutoff`, `spherical harmonics`를 1회 계산한다.
- reverse 방향 SH는 재계산하지 않고 sign flip으로 복원한다.
- `convolution.py`에서 `pair_weight = weight_nn(pair_edge_embedding)`을 1회 계산한다.

하지만 최종 TP message는 아직 아래처럼 남아 있다.

- forward 방향 message 계산
- reverse 방향 message 계산
- node reduction

즉 지금 구현은 `TP 자체의 fused pair execution`이 아니라, **TP 앞단과 weight path의 중복 제거**라고 보는 것이 정확하다.

## 4. 데이터셋 전체 규모

아래 표는 공개 validation/benchmark split 전체의 규모다.

| dataset | total configs | total atoms | avg atoms/config | max atoms/config |
| --- | ---: | ---: | ---: | ---: |
| `MPtrj` | 10,206 | 320,367 | 31.39 | 426 |
| `sAlex` | 553,111 | 5,716,263 | 10.33 | 168 |
| `OMat24` | 38,271 | 549,832 | 14.37 | 112 |
| `OMol25` | 27,697 | 1,238,644 | 44.72 | 110 |
| `OC20` | 999,866 | 73,147,343 | 73.16 | 225 |
| `OC22` | 406,249 | 31,934,356 | 78.61 | 200 |
| `phononDB` | 103 | 692 | 6.72 | 8 |

이 표는 "어떤 데이터셋이 운영상 중요한가"를 보여준다.  
다만 **per-sample speedup을 dataset 전체 평균 speedup으로 바로 일반화할 수는 없다.**

## 5. 실제로 측정한 동일 샘플의 그래프 특징과 결과

아래 표는 안정적으로 측정한 동일 샘플에 대해, graph feature와 latency 결과를 직접 연결한 것이다.

| dataset | sample atoms | directed edges | unique pairs | avg neighbors | class | PBC dims | steady speedup `baseline/pair` | cold ratio `pair/baseline` |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `MPtrj` | 426 | 28,964 | 14,482 | 67.99 | bulk-like | 3 | 1.433x | 1.089x |
| `sAlex` | 168 | 14,816 | 7,408 | 88.19 | bulk-like | 3 | 1.011x | 1.550x |
| `OC20` | 225 | 15,230 | 7,615 | 67.69 | surface-like | 3 | 1.011x | 1.556x |
| `OC22` | 200 | 11,704 | 5,852 | 58.52 | surface-like | 3 | 1.007x | 1.553x |
| `OMat24` | 112 | 4,470 | 2,235 | 39.91 | bulk-like | 3 | 0.876x | 1.563x |
| `OMol25` | 110 | 3,826 | 1,913 | 34.78 | molecule-like | 0 | 0.781x | 1.596x |
| `phononDB` | 8 | 208 | 104 | 26.00 | bulk-like | 3 | 0.774x | 1.619x |

추가로, 이 동일 샘플들에서는 다음 두 값이 모두 거의 상수였다.

- `pair_reuse_factor = 2.0`
- `reverse_pair_fraction = 1.0`

즉 **모든 샘플이 reverse pair를 매우 잘 형성하고 있었기 때문에, pair의 존재 여부 자체는 성능 차이를 설명하지 못한다.**

## 6. 가장 중요한 관찰

### 6.1 성능을 가장 잘 설명하는 1차 feature는 절대 그래프 크기다

이번 실험에서 가장 강한 분기점은 `pair` 존재 여부가 아니라, **그래프의 절대 크기**, 특히 `num_edges`였다.

관찰된 패턴:

- `208 ~ 4k` edges 구간: 명확한 손해
- `11k ~ 15k` edges 구간: 거의 break-even
- `29k` edges 구간: 뚜렷한 이득

즉 현재 구현은 "pair를 만들 수 있느냐"보다, **pair reuse로 줄어드는 계산이 metadata/control overhead를 압도할 만큼 충분히 크냐**가 더 중요하다.

### 6.2 평균 이웃 수만으로는 설명되지 않는다

`sAlex`는 평균 이웃 수가 `88.19`로 가장 높았지만 speedup은 `1.011x`에 그쳤다.  
반대로 `MPtrj`는 평균 이웃 수가 `67.99`로 더 낮지만 `1.433x`였다.

이 말은:

- 평균 degree가 높다고 무조건 유리한 것은 아니다.
- degree는 도움이 되지만, 절대 atom 수와 edge 수가 같이 커져야 한다.

즉 현재 구현은 **dense graph**에 유리하다기보다, **dense하면서도 충분히 큰 graph**에 유리하다고 보는 것이 맞다.

### 6.3 dense할 때와 sparse할 때 어떻게 읽어야 하는가

이번 결과를 dense/sparse 관점으로만 단순화하면 다음처럼 정리할 수 있다.

#### dense-large

- atom 수가 크고
- atom당 이웃 수가 많고
- 따라서 전체 edge 수가 매우 큰 경우

현재 구현에 가장 유리하다.

이 경우에는 pair가 줄여주는

- radial/cutoff 계산
- SH 계산
- pair-level weight_nn 계산

의 총량이 매우 커진다.  
`MPtrj`가 정확히 이 경우이고, 이번 실험에서 가장 큰 speedup이 여기서 나왔다.

#### dense-medium

- 이웃 수는 많지만
- atom 수가 아주 크지는 않은 경우

현재 구현은 대체로 break-even 근처다.

`sAlex`, `OC20`, `OC22`가 이 구간에 가깝다.  
즉 density 자체는 높지만, 절감되는 양이 metadata/indexing 오버헤드를 압도할 정도로 크지는 않았다.

#### sparse-small

- atom 수가 작고
- edge 수가 작거나
- atom당 unique pair 수가 낮은 경우

현재 구현은 불리하다.

`OMat24`, `OMol25`, `phononDB`가 이 경향을 보였다.  
이 구간에서는 baseline 자체가 이미 가벼워서, pair metadata를 만들고 관리하는 비용이 더 크게 보인다.

따라서 발표에서 dense/sparse를 한 줄로 말하면 아래가 가장 정확하다.

> 현재 구현은 dense하다는 사실 하나만으로 좋아지는 것이 아니라, dense해서 절감되는 계산량이 control overhead를 확실히 넘을 정도로 graph가 충분히 클 때 좋아진다.

### 6.4 구조 클래스 자체는 단독 설명변수가 아니다

구조 클래스별 관찰:

- `bulk-like`: `MPtrj`는 크게 이득, `OMat24`와 `phononDB`는 손해
- `surface-like`: `OC20`, `OC22`는 break-even 근처
- `molecule-like`: `OMol25`는 손해

즉 `bulk/surface/molecule` 레이블만으로 speedup을 예측할 수는 없다.

다만 현재 데이터에서는 다음 해석이 가장 합리적이다.

- `molecule-like`는 절대 graph가 작고 PBC가 없어 현재 오버헤드가 더 두드러진다.
- `surface-like`는 degree와 atom 수가 중간 이상이라 손익분기점 근처에 걸린다.
- `bulk-like`도 작으면 손해, 크면 큰 이득이다.

결론적으로 구조 클래스보다 중요한 것은:

- 절대 atom 수
- 절대 edge 수
- 반복 호출에서 topology가 재사용되는지

이다.

### 6.5 원자 종류 수와 PBC 여부도 단독 설명변수는 아니다

이번 샘플들에서 `num_elements`는 `2`에서 `5` 사이였지만 speedup과 뚜렷한 단조 관계는 없었다.

예를 들면:

- `MPtrj`와 `OMol25`는 둘 다 원소 종류 수가 `5`지만 결과는 크게 달랐다.
- `OC20`, `OC22`, `OMat24`는 모두 `4`종 원소를 가지지만 하나는 break-even, 하나는 손해였다.

PBC 여부도 마찬가지다.

- `OMol25`는 `PBC dims = 0`이고 손해였다.
- 하지만 `OMat24`, `phononDB`는 `PBC dims = 3`이어도 손해였다.
- 반대로 `MPtrj`는 `PBC dims = 3`에서 큰 이득을 보였다.

즉 현재 증거만으로는:

- 원소 종류 수가 많아서 유리하다
- periodic라서 무조건 유리하다

라고 말할 수 없다.

더 정확한 해석은 다음이다.

- PBC와 reverse pair는 pair 재사용이 가능해지는 전제 조건에 가깝다.
- 하지만 실제 speedup을 결정하는 것은 결국 그 위에 올라가는 절대 graph 크기와 steady-state 반복성이다.

## 7. 내 제안기법의 어느 부분이 어떤 상황에서 듣는가

현재 구현에서 실제 이득을 만드는 부분은 아래 세 가지다.

1. `pair` 기준 radial/cutoff 재사용
2. reverse SH 재계산 제거
3. `pair_weight` 1회 계산

이 세 가지가 특히 잘 듣는 상황은 아래와 같다.

### 7.1 큰 periodic graph

`MPtrj`가 가장 대표적이다.

- atoms: `426`
- edges: `28,964`
- avg neighbors: `67.99`
- steady-state speedup: `1.433x`

이 경우에는 pair reuse로 아끼는 양이 충분히 커서, pair metadata와 indexing 오버헤드를 이긴다.

### 7.2 중간 이상 크기의 surface/bulk graph

`OC20`, `OC22`, `sAlex`는 모두 완전한 reverse pair를 가지지만 speedup은 `1.0x` 근처다.

이 구간은 다음이 거의 상쇄되는 영역이다.

- 절감되는 geometry/SH/weight_nn 중복
- pair metadata, index select, forward/reverse 분리 실행 오버헤드

따라서 현재 구현은 이 구간에서 "조금 이득이 있거나 거의 본전"이다.

### 7.3 작은 graph와 molecule-like 구조

`OMat24`, `OMol25`, `phononDB`는 모두 손해였다.

이 데이터에서 읽을 수 있는 메시지는 명확하다.

- 현재 구현은 small graph 최적화가 아니다.
- 특히 `3k ~ 4k` edges 이하 구간에서는 현재 제안기법의 절감량이 아직 충분하지 않다.
- `molecule-like` 구조는 pair는 잘 형성되더라도 절대 edge 수가 적어 이득이 작다.

즉 지금 구현은 **"쌍 재사용이 가능한가"보다 "그 재사용으로 줄어드는 일이 얼마나 큰가"**에 민감하다.

## 8. 데이터셋 총 크기 관점의 해석

dataset 전체 규모는 "이득이 작더라도 운영상 의미가 있는가"를 판단할 때 중요하다.

### 8.1 운영상 의미가 큰 데이터셋

- `OC20`: 약 `100만` config
- `OC22`: 약 `40만` config
- `sAlex`: 약 `55만` config

이 셋은 이번 top-1 실험에서 speedup이 `1.0x` 근처였지만, 만약 비슷한 steady-state 경향이 dataset 전반에서도 유지된다면 작은 개선도 누적 효과는 커질 수 있다.

단, 현재 증거는 **각 데이터셋의 가장 큰 샘플 1개**에 한정된다.  
따라서 "dataset 전체 평균 throughput도 좋아진다"는 주장은 아직 할 수 없다.

### 8.2 연구적으로 의미가 큰 데이터셋

`MPtrj`는 전체 config 수는 상대적으로 적지만, 큰 graph에서 현재 구현의 순수 효과를 가장 명확하게 보여준다.

즉 논문 메시지 관점에서는:

- `MPtrj`: 메커니즘을 증명하는 strongest positive example
- `OC20`, `OC22`, `sAlex`: 손익분기점 근처 workload
- `OMat24`, `OMol25`, `phononDB`: 현재 구현의 한계를 보여주는 negative example

으로 쓰는 것이 가장 정직하다.

## 9. 분산 병렬 환경에서의 해석

이 절의 내용은 직접 벤치가 아니라 코드 기반 추론이다.

`pair_e3gnn_parallel.cpp`를 보면:

- 각 rank에서 pair metadata를 만들고 cache한다.
- 하지만 layer 진행 중 `forward_comm`으로 ghost feature를 계속 교환한다.
- backward에서도 `reverse_comm`으로 gradient를 다시 교환한다.

즉 현재 구현은 분산 병렬에서 다음을 **하지 않는다**.

- ghost 통신 제거
- pair-major communication fusion
- rank 간 message 방향 쌍 동시 처리

따라서 분산 환경에서 현재 구현이 유리할 가능성이 큰 조건은 아래와 같다.

1. rank당 `nlocal`이 충분히 크다.
2. rank당 local edge 수가 많다.
3. ghost 비율이 낮다.
4. topology cache가 오래 유지된다.
5. 한 번 만든 pair metadata를 반복 호출에서 재사용할 수 있다.

반대로 불리할 가능성이 큰 조건은 아래와 같다.

1. rank를 너무 잘게 쪼개서 local graph가 작다.
2. ghost node 비율이 높다.
3. 통신 지배 workload다.
4. neighbor topology가 자주 바뀌어 pair cache 효율이 낮다.

즉 분산 환경에서도 현재 구현은 **communication optimization**이 아니라 **local compute reuse optimization**으로 보는 것이 맞다.

## 10. 가장 안전한 결론

검증된 사실과 코드 추론을 합치면, 현재 가장 안전한 결론은 다음과 같다.

### 10.1 검증된 결론

- 현재 구현의 효과는 TP 자체가 아니라 `geometry/SH/weight_nn` 재사용에서 나온다.
- reverse pair 형성 자체는 이번 샘플들에서 모두 충분히 좋았다.
- 따라서 성능 차이를 가르는 주된 변수는 reverse pair 존재 여부가 아니라 절대 graph 크기다.
- 관측상 `4k edges 이하`는 손해, `11k~15k edges`는 break-even, `29k edges`는 뚜렷한 이득이었다.
- small molecule/small crystal은 현재 구현에 불리하다.
- large periodic bulk graph는 현재 구현에 유리하다.

### 10.2 코드 기반 추론

- 분산 병렬에서도 ghost communication은 그대로 남아 있다.
- 따라서 rank당 local graph가 크고 communication보다 local compute가 큰 환경에서만 효과가 커질 가능성이 있다.

## 11. 세미나 발표용 한 장 요약

발표에서는 아래 네 줄로 요약하면 된다.

> 현재 pair execution은 TP fused kernel이 아니라 geometry, SH, weight_nn 중복 제거에 가까운 구현이다.  
> 이번 실험에서는 reverse pair 형성률은 모든 샘플에서 거의 완전했기 때문에, 성능 차이는 pair 존재 여부가 아니라 그래프 절대 크기로 갈렸다.  
> 약 `4k` edges 이하에서는 손해였고, `11k~15k` edges에서는 본전, `29k` edges에서는 `1.433x` speedup이 나왔다.  
> 따라서 현재 구현은 small graph 최적화가 아니라 large periodic graph용 steady-state 최적화로 해석하는 것이 가장 정확하다.

## 12. 다음 실험 제안

이 보고서 다음 단계는 세 가지다.

1. 각 dataset에서 `top-1`이 아니라 atom/edge 구간별 stratified sampling을 수행한다.
2. 분산 병렬에서 rank 수와 ghost ratio를 바꾸며 실제 scaling benchmark를 수행한다.
3. FlashTP 없이도 현재 e3nn full path 내부 profiler를 찍어, 절감되는 시간이 `SH`, `weight_nn`, `indexing` 중 어디서 나는지 layer별로 분해한다.

이 세 가지가 있어야 "어떤 dataset에 유리한가"를 넘어, "왜 정확히 그 dataset에서 유리한가"를 더 강하게 주장할 수 있다.
