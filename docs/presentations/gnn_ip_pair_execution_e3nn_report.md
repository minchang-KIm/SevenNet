# GNN-IP Pair Execution e3nn 기준 구현 보고서

세미나 발표용 정리 문서  
기준 브랜치: `pair-major`  
기준 커밋: `583e0c1`

## 2026-04-03 Note

- 이 문서는 pure `e3nn` 경로에서 현재 구현의 구조적 의미를 설명하는 내부 보고서로 유지한다.
- 외부 발표나 논문용 최종 claim은 `docs/papers/icpp_pair_execution/03_final_manuscript.md`를 따른다.
- 현재 canonical 표현은 `pair-major TP`가 아니라 `pair-aware geometry-side reuse`다.

## 1. 이 문서의 질문

이 문서는 다음 질문에 답하기 위한 세미나용 보고서다.

> FlashTP를 빼고, 순수 SevenNet e3nn baseline과 현재 pair execution 구현을 비교하면 실제 성능 이득이 있는가?

이 질문이 중요한 이유는, 현재 `FlashTP + pair_execution` 경로는 backend 내부에서 pair-major 실행을 하지 않기 때문에, 사용자의 구현 기여가 최종 TP/backend 최적화로 직접 이어지지 않는다는 해석이 가능하기 때문이다.

따라서 이번 보고서는 FlashTP를 제거한 상태에서, 현재 구현이 원래 SevenNet baseline 위에서 어떤 의미를 가지는지 별도로 평가한다.

## 2. 핵심 결론

- FlashTP를 제거하면 현재 구현은 `geometry_only`가 아니라 `full` pair path로 동작한다.
- 이 경로에서는 현재 구현이 실제로 의미 있는 성능 이득을 보이는 경우가 있다.
- 가장 큰 실데이터 샘플인 `MPtrj`에서는 steady-state 기준 `1.433x` 속도 향상이 관측되었다.
- `OC20`, `OC22`, `sAlex`에서는 근소한 이득 또는 사실상 break-even이었다.
- `OMat24`, `OMol25`, `phononDB`에서는 오히려 느려졌다.
- 즉 현재 구현은 pure e3nn 경로에서는 “부분적으로 효과 있는 구현”이지만, 아직 universal win은 아니다.

## 3. 왜 이 비교가 필요한가

이전 FlashTP 중심 보고서의 결론은 명확했다.

- `FlashTP` 자체는 빠르다.
- 그러나 `FlashTP + current pair_execution`은 거의 안 빨라지거나 약간 느리다.

이 결과만 보면 “현재 구현은 별 의미가 없다”는 오해가 생길 수 있다. 하지만 그 결론은 정확히 말하면 FlashTP backend에 대한 결론이다.

현재 pair execution 구현의 본질은 아래 두 줄로 요약된다.

- pair 기준으로 geometry와 weight를 재사용한다.
- backend가 pair layout을 끝까지 유지하지 못하면 그 이득이 상쇄된다.

따라서 accelerator를 제거한 e3nn path에서 다시 보면, 구현의 순수 효과를 더 직접적으로 볼 수 있다.

## 4. 이번 실험 설정

비교 case:

- `e3nn_baseline`
- `e3nn_pair_full`

설정 의미:

- `e3nn_baseline`: 원래 SevenNet directed-edge 경로
- `e3nn_pair_full`: pair execution을 켜고, policy를 명시적으로 `full`로 고정

모델과 경로:

- model: `7net-omni`
- inference path: ASE calculator end-to-end
- device: CUDA

데이터셋:

- `MPtrj`
- `sAlex`
- `OMat24`
- `OMol25`
- `OC20`
- `OC22`
- `phononDB`

샘플 선택:

- 각 공개셋에서 가장 큰 구조 1개

측정:

- cold-call 1회
- steady-state 반복 `3`회
- energy/force 차이 기록

산출물은 아래 경로에 저장했다.

- `bench_runs/real_e3nn_pair/summary.md`
- `bench_runs/real_e3nn_pair/analysis.md`
- `bench_runs/real_e3nn_pair/metrics/aggregated.csv`
- `bench_runs/real_e3nn_pair/plots/steady_state_latency.png`
- `bench_runs/real_e3nn_pair/plots/cold_latency.png`
- `bench_runs/real_e3nn_pair/plots/e3nn_pair_speedup.png`

## 5. baseline과 현재 구현의 차이

### 5.1 baseline

baseline은 directed edge 중심이다.

```text
directed edge 생성
-> edge별 geometry/SH 계산
-> edge별 weight_nn
-> edge별 TP message
-> node 기준 sum reduction
```

### 5.2 현재 구현의 e3nn full path

현재 구현은 먼저 reverse edge를 pair로 묶는다.

```text
directed edge 생성
-> pair metadata 생성
-> pair 기준 geometry 계산
-> pair 기준 weight_nn 계산
-> forward edge 묶음 TP
-> reverse edge 묶음 TP
-> node 기준 reduction
```

여기서 중요한 차이는 다음과 같다.

- geometry는 pair 기준 1회
- `weight_nn`도 pair 기준 1회
- final message는 양방향 모두 계산

즉 FlashTP와 달리, e3nn `full` 경로에서는 현재 구현이 단순 preprocessing이 아니라 실제 message path까지 들어간 최적화다.

## 6. 결과 요약

steady-state speedup `e3nn_baseline / e3nn_pair_full`:

- `MPtrj`: `1.433x`
- `OC20`: `1.011x`
- `sAlex`: `1.011x`
- `OC22`: `1.007x`
- `OMat24`: `0.876x`
- `OMol25`: `0.781x`
- `phononDB`: `0.774x`

요약 통계:

- median speedup: `1.007x`
- geometric-mean speedup: `0.965x`

이 결과는 다음처럼 읽는 것이 맞다.

- 큰 그래프에서는 pair path가 실제 이득을 낼 수 있다.
- 중간 크기 그래프에서는 거의 본전이다.
- 작은 그래프나 특정 workload에서는 control overhead가 더 커질 수 있다.

## 7. 결과 해석

### 7.1 가장 중요한 positive result

`MPtrj`에서 `1.433x` 향상이 나왔다는 점은 중요하다.

이건 현재 구현이 pure e3nn path에서는 단순한 부수 최적화가 아니라, 실제로 end-to-end steady-state latency를 줄일 수 있다는 뜻이다.

즉 “지금까지의 구현이 TP 부분은 아니다”라는 평가는 FlashTP 경로에서는 맞지만, e3nn `full` 경로까지 포함하면 완전히 맞는 말은 아니다. e3nn path에서는 현재 구현이 실제 message path 최적화로 작동한다.

### 7.2 왜 중간 크기에서는 거의 본전인가

`OC20`, `OC22`, `sAlex`는 소폭 개선 또는 사실상 break-even이다.

이 구간에서는 다음 두 요소가 서로 비슷해진다.

- pair reuse로 줄어드는 geometry/weight 중복
- pair metadata 및 control-path 오버헤드

즉 이 구간은 현재 구현이 “될 수도 있고 안 될 수도 있는 경계”에 해당한다.

### 7.3 왜 작은 쪽은 손해가 나는가

`OMat24`, `OMol25`, `phononDB`는 오히려 느려졌다.

이건 current pair path가 여전히 아래 비용을 갖고 있기 때문이다.

- pair metadata 생성
- reverse edge matching
- 추가 인덱싱과 branch
- cache validation

구조가 작아지면 원래 baseline 계산량이 충분히 작아서, pair reuse가 줄여주는 양보다 control overhead가 더 크게 보인다.

## 8. cold-call 결과

cold-call은 전 구간에서 나빠졌다.

대표 ratio `pair_full / baseline`:

- `phononDB`: `1.62x`
- `OMol25`: `1.60x`
- `OMat24`: `1.56x`
- `OC20`: `1.56x`
- `OC22`: `1.55x`
- `sAlex`: `1.55x`
- `MPtrj`: `1.09x`

즉 현재 pair execution은 steady-state 최적화 성격이 강하고, cold path에서는 명확한 비용을 지불한다.

세미나에서는 이 점을 분명히 말하는 것이 좋다.

- steady-state와 cold-call을 분리해서 봐야 한다.
- topology cache가 있어도 첫 호출 비용은 여전히 크다.

## 9. 수치 정확성

정확성은 안정적이었다.

- worst energy delta: `0.0 eV`
- worst force delta: `9.155e-05 eV/A`

즉 성능이 mixed라는 점과 별개로, 현재 full pair path는 수치적으로는 baseline과 잘 맞는다.

## 10. 이 결과가 의미하는 것

이번 결과는 FlashTP 보고서와 합쳐서 보면 다음 메시지로 정리된다.

### 메시지 1

현재 구현은 “pair reuse runtime”으로서 실제로 동작한다.

### 메시지 2

pure e3nn 경로에서는 큰 그래프에서 유의미한 speedup이 가능하다.

### 메시지 3

하지만 이득은 graph size와 workload shape에 따라 달라진다.

### 메시지 4

FlashTP 경로에서는 backend가 pair-major가 아니어서 같은 이득이 유지되지 않는다.

즉 지금까지의 구현은 “의미 없는 preprocessing”은 아니고, “backend에 따라 성과가 달라지는 1단계 구현”이라고 보는 것이 정확하다.

## 11. 세미나 발표용 핵심 문장

발표에서는 아래 문장을 그대로 써도 된다.

> FlashTP를 제거하고 원래 SevenNet e3nn 경로만 보면, 현재 pair execution full path는 실제로 large real graph에서 성능 이득을 만든다. 다만 그 이득은 universal하지 않고, graph size와 control-path overhead의 균형에 따라 달라진다.

그리고 FlashTP와의 차이는 아래처럼 정리하면 된다.

> 따라서 현재 구현의 문제는 pair idea 자체가 아니라, 그 이득을 accelerated backend 내부까지 유지하지 못하는 integration 구조에 있다.

## 12. 발표용 Q&A 포인트

### 질문 1. 현재 구현이 정말 TP path 최적화라고 볼 수 있나

FlashTP 경로만 보면 아니다. 하지만 e3nn `full` 경로에서는 pair weight 재사용과 forward/reverse 분리 실행이 실제 message path에 걸쳐 들어가므로, pure e3nn 기준으로는 맞다.

### 질문 2. 왜 universal speedup이 아니냐

pair reuse가 줄여주는 계산량과 pair metadata/control-path 오버헤드가 graph size에 따라 상대적 비중이 달라지기 때문이다.

### 질문 3. 그럼 지금 가장 강한 claim은 무엇인가

현재 pair execution full path는 pure e3nn inference에서 large real graph에 대해 의미 있는 steady-state speedup을 만들 수 있지만, 전체 real workload에서 보편적으로 빠르다고 주장할 단계는 아니다.

### 질문 4. 다음 단계는 무엇인가

FlashTP에서도 같은 이득을 내고 싶다면 pair-major backend를 설계해야 한다. 즉 pair를 만든 뒤 다시 edge로 풀지 않는 커널이 필요하다.

## 13. 최종 정리

이번 e3nn 기준 실험은 현재 구현을 다시 평가하게 만든다.

- FlashTP 기준으로는 현재 구현이 backend integration 한계 때문에 약했다.
- 하지만 pure e3nn 기준으로는 large graph에서 분명한 positive result가 나온다.
- 따라서 지금까지의 구현은 연구 가치가 있으며, 특히 “exact pair reuse runtime”이라는 관점에서는 이미 실질적인 결과를 낸 상태다.

세미나 마무리 문장은 아래가 가장 안전하다.

> 현재 pair execution 구현은 pure e3nn 경로에서는 효과가 있고, FlashTP 경로에서는 아직 backend co-design이 부족하다. 즉 아이디어가 틀린 것이 아니라, backend별 완성도가 다른 상태다.
