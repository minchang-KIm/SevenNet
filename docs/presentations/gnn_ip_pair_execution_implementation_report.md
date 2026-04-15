# GNN-IP Pair Execution 구현 보고서

세미나 발표용 정리 문서  
기준 브랜치: `pair-major`  
기준 커밋: `583e0c1`

## 2026-04-03 Note

- 이 문서는 현재 branch의 구현 구조를 설명하는 보고서로 유지한다.
- 현재 구현 상태에 대한 canonical 평가는 `docs/papers/icpp_pair_execution/00_current_status_report.md`에 정리되어 있다.
- 외부 문서에서는 현재 구현을 `pair-major execution`이 아니라 `pair-aware geometry-side reuse`라고 부르는 것이 맞다.

## 1. 문서 목적

이 문서는 지금까지 SevenNet 레포지토리에서 구현된 `pair_execution` 구조를 세미나 발표용으로 정리한 보고서다. 목표는 다음 세 가지다.

- 현재 구현이 baseline 대비 무엇을 바꾸었는지 명확히 설명한다.
- `FlashTP`와 결합했을 때 실제로 어떤 부분이 공유되고 어떤 부분은 여전히 중복되는지 설명한다.
- 실제 공개 데이터셋 측정 결과를 바탕으로, 현재 구현의 성과와 한계를 분리해서 보고한다.

이 문서는 “제안 아이디어” 중심이 아니라 “현재 브랜치에 실제로 구현된 것” 중심으로 쓴다.

## 2. 핵심 요약

- baseline은 `i -> j`, `j -> i`를 완전히 별개의 directed edge로 처리한다.
- 현재 `pair_execution`은 reverse edge를 pair로 묶고, geometry와 `weight_nn` 일부를 pair 기준으로 재사용한다.
- 그러나 최종 message tensor product는 source node feature가 방향별로 다르기 때문에 완전히 pair 단위로 합쳐지지 않는다.
- `FlashTP`와 결합한 현재 경로는 `pair-major` 실행이 아니라, pair 기준으로 만든 결과를 다시 directed edge layout으로 펴서 fused kernel에 넣는 구조다.
- 그래서 `FlashTP + pair_execution`은 수치적으로는 맞게 동작하지만, 현재 실측 기준으로는 `FlashTP` 단독 대비 일관된 성능 향상을 만들지 못한다.

## 3. 문제 배경

short-range equivariant GNN 기반 MLIP는 물리적으로는 하나의 상호작용 pair를 다루지만, 구현상으로는 보통 다음 두 directed edge를 각각 처리한다.

- `i -> j`
- `j -> i`

이 방식의 장점은 구현이 단순하다는 점이다. 하지만 다음 중복이 생긴다.

- 거리 기반 embedding 중복
- spherical harmonics 중복
- weight network 입력 준비 중복
- edge별 message 생성 뒤 다시 node 기준으로 reduction해야 하는 메모리 이동

현재 branch의 `pair_execution`은 이 중복을 줄이기 위해 추가된 runtime 최적화다.

## 4. 구현 범위

현재 구현은 다음 수준까지 들어가 있다.

- Python inference 경로 pair metadata 생성
- ASE calculator 경로 pair metadata 및 topology cache 적용
- LAMMPS 및 MLIAP 경로 pair metadata 연동
- backend별 pair policy 선택
- `FlashTP`, `OEQ`, `e3nn` 경로와의 호환
- 실제 공개 데이터셋 기준 정확성 및 latency 벤치마크

현재 구현은 다음 수준까지는 아직 아니다.

- pair-major FlashTP fused kernel
- pair-major backward kernel
- end-to-end pair layout 유지
- backend autotuning 기반 pair policy 선택

## 5. 코드 구조 개요

현재 구조를 이해할 때 핵심 파일은 아래와 같다.

- `sevenn/pair_runtime.py`
- `sevenn/nn/edge_embedding.py`
- `sevenn/nn/convolution.py`
- `sevenn/model_build.py`
- `sevenn/nn/flash_helper.py`
- `sevenn/checkpoint.py`
- `sevenn/calculator.py`

역할은 다음과 같다.

- `pair_runtime.py`: reverse edge를 pair로 묶는 metadata 생성과 cache 관리
- `edge_embedding.py`: pair 기준 geometry, radial, SH 재사용
- `convolution.py`: pair 기준 `weight_nn` 재사용과 backend별 message path 선택
- `model_build.py`: backend patch와 pair policy 전달
- `flash_helper.py`: 기존 `IrrepsConvolution`을 FlashTP fused convolution으로 치환
- `checkpoint.py`, `calculator.py`: backend override와 pair policy 해석을 일관되게 유지

## 6. Baseline 실행 흐름

baseline은 directed edge 중심 실행이다.

```text
neighbor list
-> directed edge 생성
-> edge별 |r|, radial, cutoff 계산
-> edge별 spherical harmonics 계산
-> edge별 weight_nn(edge_embedding)
-> edge별 tensor product message
-> node 기준 scatter_reduce(sum)
```

한 pair `(i, j)`를 보면 내부적으로는 아래처럼 동작한다.

```text
i -> j: geometry, SH, weight_nn, TP, scatter
j -> i: geometry, SH, weight_nn, TP, scatter
```

즉 양방향에서 공유되는 기하 정보도 두 번 계산된다.

## 7. FlashTP만 사용했을 때의 흐름

`FlashTP`를 켜면 앞단은 거의 baseline과 같고, 마지막 convolution 부분만 fused backend로 바뀐다.

```text
neighbor list
-> directed edge 생성
-> edge별 |r|, radial, cutoff 계산
-> edge별 spherical harmonics 계산
-> edge별 weight_nn(edge_embedding)
-> FlashTP fused kernel
   (gather x[edge_src] + TP + scatter to edge_dst)
```

즉 `FlashTP`는 다음을 가속한다.

- gather
- tensor product
- scatter/reduction

반면 아래는 그대로 남는다.

- geometry 생성
- spherical harmonics 계산
- weight network 계산
- directed edge layout 자체

따라서 `FlashTP`는 “edge 기반 실행을 유지한 채 convolution 구간을 빠르게 만드는 최적화”라고 보는 것이 정확하다.

## 8. 현재 pair_execution 흐름

현재 구현은 먼저 directed edge를 pair로 묶는다.

```text
neighbor list
-> directed edge 생성
-> reverse directed edge 매칭
-> pair metadata 생성
```

이때 생성되는 핵심 metadata는 다음과 같다.

- `edge_pair_map`
- `edge_pair_reverse`
- `pair_edge_forward_index`
- `pair_edge_backward_index`
- `pair_edge_has_reverse`
- `pair_edge_vec`

그 다음 geometry 경로는 pair 기준으로 계산한다.

```text
pair 기준 canonical 방향 하나 선택
-> pair_r = |r|
-> pair_embedding = basis(pair_r) * cutoff(pair_r)
-> pair_attr = spherical(pair_rvec)
-> reverse 방향 SH는 (-1)^l sign flip으로 복원
```

즉 현재 구현에서 공유되는 것은 아래와 같다.

- 거리
- radial basis
- cutoff embedding
- spherical harmonics의 canonical 방향 계산
- reverse SH 복원을 위한 parity sign

그리고 convolution 앞단에서 `weight_nn`도 pair 기준으로 한 번만 계산한다.

```text
pair_weight = weight_nn(pair_edge_embedding)
```

## 9. 현재 구현에서 실제로 공유되는 것과 공유되지 않는 것

현재 구현을 이해할 때 가장 중요한 구분은 아래 두 줄이다.

- geometry는 pair 기준으로 공유된다.
- final message는 pair 기준으로 완전히 공유되지 않는다.

공유되는 항목:

- `|r|`
- radial embedding
- cutoff embedding
- canonical 방향 SH
- reverse 방향 SH 복원용 sign rule
- `weight_nn(pair_embedding)`

공유되지 않는 항목:

- `i -> j` message에서 쓰는 source feature `x_i`
- `j -> i` message에서 쓰는 source feature `x_j`
- destination node로의 최종 누산

즉 현재 구현은 “pair-aware reuse”이지, “완전한 pair-major execution”은 아니다.

## 10. e3nn backend에서의 현재 pair execution

`e3nn` 경로에서 `full` policy를 쓰면 현재 구조는 대략 이렇게 된다.

```text
pair_weight 1회 계산
-> forward 방향 edge 묶음에 대해 TP 1회
-> reverse 방향 edge 묶음에 대해 TP 1회
-> 두 결과를 각각 node로 gather
```

이 구조의 의미는 다음과 같다.

- geometry와 weight는 pair 기준으로 재사용한다.
- 하지만 forward message와 reverse message는 source feature가 다르므로 둘 다 계산한다.
- pair 하나를 두 thread가 나눠 계산하는 구조는 아니다.
- 실제로는 forward edge batch와 reverse edge batch를 나눠 커널을 두 번 실행하는 구조에 가깝다.

## 11. FlashTP와 결합했을 때의 현재 구조

현재 branch에서 `FlashTP + pair_execution(auto)`는 대부분 `geometry_only`로 해석된다.

이 경로는 아래처럼 동작한다.

```text
pair 기준 geometry 계산
-> pair 기준 weight_nn 계산
-> pair_weight를 edge_pair_map으로 다시 directed edge weight로 expansion
-> FlashTP fused kernel에 directed edge 전체 투입
```

이 점이 매우 중요하다.

현재 `FlashTP + pair_execution`은 다음 구조가 아니다.

- pair 하나를 읽고 양방향 message를 한 번에 계산하는 커널

현재 구조는 다음 구조다.

- pair 기준으로 앞단 일부를 재사용
- 하지만 fused convolution 직전에는 다시 directed edge layout으로 복귀

즉 현재 FlashTP 결합 경로는 “pair-major FlashTP”가 아니라 “pair-aware preprocessing + 기존 FlashTP 재사용”이다.

## 12. backend policy 해석

현재 pair runtime은 설정을 통해 policy를 결정한다.

- `baseline`
- `geometry_only`
- `full`
- `auto`

현재 heuristic의 핵심은 다음과 같다.

- accelerator가 없으면 `auto`는 `full` 또는 `geometry_only`로 갈 수 있다.
- `FlashTP`, `OEQ`, `cuEquivariance` 같은 accelerator가 켜져 있으면 `auto`는 기본적으로 `geometry_only`로 내려간다.

이 설계는 현재 accelerated backend가 pair-major layout을 끝까지 유지하지 못한다는 현실을 반영한다.

## 13. correctness 관련 수정 사항

실제 공개 데이터셋 실험 전에 두 개의 correctness bug를 수정했다.

### 13.1 checkpoint backend override 문제

구형 checkpoint에서 `enable_flash=True` 같은 override를 줄 때, backend flag 반영보다 pair policy 해석이 먼저 일어나서 `auto -> geometry_only`가 아니라 `auto -> full`로 남는 문제가 있었다.

이 문제는 `sevenn/checkpoint.py`에서 수정했다.

### 13.2 calculator 경로 설정 불일치 문제

ASE calculator가 checkpoint를 읽은 뒤, 변환된 model 설정이 아니라 원래 checkpoint config를 다시 참조해 pair policy를 잘못 해석하는 문제가 있었다.

이 문제는 `sevenn/calculator.py`에서 수정했다.

두 수정 모두 regression test로 커버했다.

## 14. 실제 데이터셋 벤치마크 범위

현재 실험은 ASE calculator 경로 기준으로 진행했다. 모델은 `7net-omni`를 사용했고, 각 데이터셋에 맞는 modal을 붙였다.

측정 데이터셋:

- `MPtrj` validation
- `sAlex` validation
- `OMat24` validation
- `OMol25` neutral validation
- `OC20` S2EF in-domain validation
- `OC22` S2EF in-domain validation
- `phononDB`

비교 case:

- `e3nn_baseline`
- `flash_baseline`
- `flash_pair_auto`

측정 방식:

- 동일 topology에서 cold-call 1회
- steady-state 반복 호출 latency
- energy/force deviation 측정

## 15. 실험 결과 요약

### 15.1 FlashTP 단독 효과

`FlashTP`는 `e3nn_baseline` 대비 일관되게 유리했다.

- `MPtrj`: `12.79x`
- `OC20`: `4.87x`
- `sAlex`: `4.78x`
- `OC22`: `3.62x`
- `OMat24`: `1.73x`
- `OMol25`: `1.51x`
- `phononDB`: `1.41x`

즉 FlashTP 자체는 현재 branch에서 실효성이 명확하다.

### 15.2 FlashTP + pair_execution(auto) 효과

`flash_pair_auto`는 `flash_baseline` 대비 steady-state에서 거의 이득이 없거나 약간 느렸다.

- `OC22`: `1.006x`
- `phononDB`: `0.996x`
- `sAlex`: `0.982x`
- `OC20`: `0.975x`
- `MPtrj`: `0.926x`
- `OMat24`: `0.920x`
- `OMol25`: `0.894x`

즉 현재 구현은 “정확히 동작한다”까지는 달성했지만, “FlashTP 위에서 end-to-end 성능 이득이 난다”는 결과는 아직 아니다.

### 15.3 Cold-call latency

pair execution은 cold-call latency를 악화시켰다.

대표 예시:

- `MPtrj`: `78.2 ms -> 279.4 ms`
- `OC20`: `66.8 ms -> 172.4 ms`
- `OC22`: `65.5 ms -> 147.7 ms`

원인은 pair metadata 생성과 topology cache 준비가 cold path에 남아 있기 때문이다.

### 15.4 수치 정확성

수치적으로는 안정적이었다.

- worst energy delta: `0.0 eV`
- worst force delta: `7.63e-05 eV/A`

즉 현재 pair runtime은 성능 이득은 미약하지만, 수치 정확성 측면에서는 실사용 가능한 수준으로 보인다.

## 16. 왜 현재 FlashTP + pair_execution이 안 빨라지는가

현재 결과를 가장 잘 설명하는 문장은 다음과 같다.

> pair 정보를 만들었지만, backend 내부까지 pair layout을 유지하지 못해서 end-to-end 이득으로 이어지지 않는다.

세부적으로는 다음 세 가지가 문제다.

### 16.1 pair metadata 생성 비용이 여전히 hot path에 있다

- reverse edge matching
- topology signature 확인
- cache hit/miss 판정

이 경로가 특히 cold-call에서 크게 드러난다.

### 16.2 pair_weight를 다시 directed edge로 펴는 비용이 있다

현재 FlashTP 경로는 `pair_weight`를 만든 뒤 `index_select`로 다시 edge weight로 펼친다. 따라서 pair-aware reuse로 아낀 메모리 트래픽 일부를 다시 잃는다.

### 16.3 최종 FlashTP kernel은 pair-major가 아니다

현재 FlashTP kernel은 여전히 다음 입력을 기대한다.

- `x`
- `edge_filter`
- `weight`
- `edge_src`
- `edge_dst`

즉 backend 인터페이스가 directed edge 기준이므로, pair-aware 결과를 backend 내부까지 유지할 수 없다.

## 17. 프로파일 관점 해석

steady-state 프로파일 기준으로 보면 `FlashTP` baseline은 이미 convolution의 핵심 구간을 잘 fused하고 있다.

실제 큰 샘플 기준으로 관찰된 특징:

- FlashTP forward kernel과 backward kernel이 CUDA 시간의 큰 비중을 차지한다.
- 특히 force 계산에서는 backward 비중이 매우 크다.
- 현재 `flash_pair_auto`는 `weight_nn` 일부는 줄이지만, `index_select`, `index_add_` 같은 재배치 비용이 추가된다.

즉 현재 구조에서 절약되는 부분:

- geometry 일부
- `weight_nn` 일부

현재 구조에서 새로 생기는 비용:

- pair metadata 준비
- pair to edge expansion
- backend 진입 전 재배치

결론적으로 현재 branch는 pair reuse의 잠재 이득을 가지고 있지만, backend 구조 제약 때문에 그 이득을 유지하지 못한다.

## 18. 현재 구현의 의미

현재 구현의 의미를 과장 없이 정리하면 다음과 같다.

### 성과

- exact pair reuse 개념을 코드에 실제로 도입했다.
- geometry, SH, weight_nn 재사용이 동작한다.
- ASE, checkpoint, inference 경로에서 설정 일관성을 맞췄다.
- 실제 공개 데이터셋 기준으로 수치 정확성을 검증했다.
- FlashTP와 결합해도 수치적 mismatch 없이 동작한다.

### 한계

- pair-major backend가 아니다.
- FlashTP에서 pair layout 유지가 끊긴다.
- cold-path pair metadata 비용이 크다.
- steady-state에서도 FlashTP 대비 일관된 성능 이득이 없다.

따라서 현재 branch의 가장 안전한 해석은 다음이다.

> 현재 구현은 pair execution의 correctness와 부분 재사용 구조를 확보한 첫 단계 구현이다. 그러나 accelerated backend와 공동 설계된 최종 형태는 아직 아니다.

## 19. 세미나 발표용 메시지

발표에서 강조해야 할 포인트는 아래 순서가 적절하다.

### 메시지 1

우리는 reverse directed edge 중복을 runtime 수준의 first-class inefficiency로 정의했고, 이를 줄이기 위한 pair execution runtime을 구현했다.

### 메시지 2

현재 구현은 geometry와 weight reuse까지는 성공했지만, backend 내부까지 pair layout을 유지하는 단계는 아니다.

### 메시지 3

실제 데이터셋에서 수치 정확성은 확인되었고, FlashTP 단독 성능 이득도 재현되었다.

### 메시지 4

하지만 `FlashTP + current pair_execution`은 아직 end-to-end 속도 개선으로 이어지지 않는다.

### 메시지 5

따라서 다음 연구 단계는 pair-aware FlashTP kernel과 pair-major backward를 포함한 backend co-design이어야 한다.

## 20. 예상 질문과 답변 포인트

### 질문 1. 지금 구현이 pair-major execution인가

아니다. 현재 구현은 pair-aware reuse다. 특히 FlashTP 경로에서는 pair 결과를 다시 directed edge layout으로 펴기 때문에 진짜 pair-major는 아니다.

### 질문 2. 왜 message를 pair 기준으로 한 번만 계산하지 못하나

`i -> j`와 `j -> i`는 source node feature가 다르다. 따라서 geometry는 공유할 수 있어도 최종 message는 양방향 모두 계산해야 한다.

### 질문 3. 그러면 pair execution으로 절약되는 것은 무엇인가

거리, radial, cutoff, canonical SH, reverse SH 복원, `weight_nn` 일부다.

### 질문 4. 왜 FlashTP 위에서 안 빨라지나

현재 FlashTP backend가 directed edge 기준 인터페이스를 사용하기 때문에 pair layout을 끝까지 유지하지 못하고, 중간에 pair to edge expansion 비용이 다시 들어가기 때문이다.

### 질문 5. 지금 결과로 논문 claim을 어떻게 잡아야 하나

현재 safest claim은 “FlashTP + current pair_execution is numerically correct on real datasets, but does not yet deliver a consistent end-to-end speedup over FlashTP alone”이다.

## 21. 다음 단계

다음 단계는 아래 순서가 적절하다.

- pair-major FlashTP forward kernel 설계
- pair-major backward kernel 설계
- pair metadata 생성 hot path 축소
- topology epoch 기반 pair plan cache 강화
- backend autotuning 기반 policy 선택

이 단계가 들어가야 비로소 “현재 구현”에서 “논문용 시스템 최적화 결과”로 넘어갈 수 있다.

## 22. 결론

현재 SevenNet branch의 `pair_execution`은 제안 수준을 넘어 실제 코드로 구현되었고, geometry 및 weight 재사용, backend policy, calculator 경로, 공개 데이터셋 정확성 검증까지 마친 상태다.

하지만 현재 구현의 본질은 “pair-aware reuse를 기존 directed-edge backend 위에 얹은 구조”다. 따라서 pair 실행의 잠재 이득이 backend 내부까지 살아남지 못한다. 이 점이 현재 실험에서 `FlashTP + pair_execution`이 `FlashTP` 단독보다 거의 안 빠르거나 약간 느린 이유다.

정리하면 현재 단계의 성과는 다음과 같다.

- correctness 확보
- runtime 구조 재정의의 첫 구현 확보
- 실데이터 검증 완료

그리고 남은 핵심 과제는 다음 한 줄로 요약된다.

> pair 정보를 만든 뒤 다시 edge로 풀지 않는 backend를 만들어야 한다.
