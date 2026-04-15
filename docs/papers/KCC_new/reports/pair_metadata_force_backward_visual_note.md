# Pair Metadata / Force Backward Visual Note

## 1. 최근 `full` 경로가 정말 개선됐는가

결론부터 쓰면, **baseline 대비로는 아직 개선됐다고 말할 수 없다. 여전히 느리다.**  
최근 수정의 의미는 `full` 경로 내부 구조를 덜 비효율적으로 만든 것이지, baseline을 이겼다는 뜻이 아니다.

대표 재측정 결과는 [pre_fused_singlepass_patch_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pre_fused_singlepass_patch_report.md)에 정리돼 있다.

- `mptrj`, `step_force`: `424.70 ms -> 464.04 ms`
- `oc20_s2ef_train_20m`, `step_force`: `124.66 ms -> 139.81 ms`
- `qm9_hf`, `step_force`: `29.26 ms -> 31.33 ms`

즉 지금은 이렇게 써야 정확하다.

- `full` 경로는 예전보다 덜 나쁘게 만들 수는 있었다.
- 하지만 현재 코드 기준으로는 여전히 baseline보다 느리다.
- 논문 메인 비교는 `baseline vs geometry_only`가 더 정직하다.

## 2. `lmax`가 4가 최대인가

아니다. **3차원이라는 사실이 `lmax <= 4`를 강제하지 않는다.**

`lmax`는 구면조화함수 `Y_l^m`의 최대 차수이고, 3차원에서도 `l = 0, 1, 2, ...`처럼 계속 정의된다.  
즉 이론적으로는 `4`가 상한이 아니다. 상한은 수학이 아니라 **모델 설계와 계산 비용**이 사실상 정한다.

현재 SevenNet 코드도 이를 그대로 반영한다.

- edge SH 차수는 [model_build.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/model_build.py#L93)에서 `lmax_edge`로 설정된다.
- 실제 SH irreps는 [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py#L176)에서 `Irreps.spherical_harmonics(lmax, parity)`로 만들어진다.
- node 쪽도 기본 설정에서는 [model_build.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/model_build.py#L479)처럼 `lmax_node`가 전역 `lmax`를 따른다.

즉 `lmax=5,6,7,8`도 코드상 완전히 유효하다.

실무적으로는 `1~4`를 많이 쓰지만, 그건

- 표현력이 충분한 경우가 많고
- 비용이 급격히 커지기 때문

이지, 3차원이어서 `4`가 최대이기 때문은 아니다.

## 3. 그럼 `lmax` 실험을 4로 잘라야 하나

현재 단계에서는 **잘라서 없애는 것보다, 본문과 보조자료를 나누는 것이 맞다.**

권장 해석:

- 본문 메인 비교: `lmax = 1..4`
  - 실무적으로 익숙한 범위
  - 독자가 받아들이기 쉬움
- 보조 분석/부록: `lmax = 5..8`
  - 높은 `lmax`에서 비용이 어떻게 폭증하는지 보여주는 구조 분석
  - pair-aware geometry reuse의 잠재력을 설명하는 근거

즉 `5~8`을 “틀린 실험”으로 버릴 필요는 없다.  
다만 본문에서 어떻게 배치할지는 더 보수적으로 가져가는 게 좋다.

## 4. pair metadata 파이프라인의 실제 동작

그림: [pair_metadata_pipeline.svg](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/runtime/pair_metadata_pipeline.svg)

예시 구조를 직관적으로 보여주는 그림: [pair_metadata_example.svg](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/runtime/pair_metadata_example.svg)

핵심:

1. ASE 경로에서는 먼저 그래프를 만든다.
2. 그 뒤 `data.to(device)`를 수행한다.
3. `prepare_pair_metadata(...)`가 호출된다.
4. pair canonicalization은 현재 텐서 연산으로 device에서 진행될 수 있다.
5. 하지만 topology signature는 여전히 CPU 해시를 쓴다.

즉 pair metadata는 완전한 device-only 경로가 아니다.

- pair mapping 본체: device/vectorized
- topology signature: CPU hash

또, 현재 코드에는 명시적인 CUDA stream 병렬화가 없다.

- `torch.cuda.Stream`
- `wait_stream`
- `record_stream`

같은 코드는 없다.  
즉 pair metadata와 이후 model forward는 **기본 stream에서 사실상 직렬**로 이어진다.

## 5. force backward 파이프라인의 실제 동작

그림: [force_backward_pipeline.svg](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/runtime/force_backward_pipeline.svg)

핵심:

1. `AtomGraphSequential._preprocess()`가 `EDGE_VEC.requires_grad_(True)`를 켠다.  
   근거: [sequential.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/sequential.py#L167)
2. model forward가 total energy를 만든다.
3. 마지막 `ForceStressOutputFromEdge`가 `torch.autograd.grad(energy, [rij])`를 호출한다.  
   근거: [force_output.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/force_output.py#L173)
4. 이 backward는 마지막 readout만 다시 도는 게 아니라, energy를 만드는 상류 전체를 다시 지난다.
5. gradient로 얻은 `fij`를 다시 scatter해서 원자 힘과 stress를 만든다.

즉 force 경로는

- readout의 부록이 아니라
- 전체 energy graph의 backward

다.

이 점 때문에 `forward_energy`와 `step_force` 결과를 같은 의미로 해석하면 안 된다.

## 6. 동시성 관점에서 현재 코드가 말해주는 것

현재 코드 기준으로는 다음 해석이 맞다.

- pair metadata: 부분적으로 device화됐지만 topology signature는 CPU 해시
- edge embedding, weight NN, convolution, readout, force backward: 기본 stream 직렬 큐잉
- force backward: explicit overlap 없음
- pair metadata와 force backward가 서로 겹쳐 도는 구조도 없음

즉 지금 병목은 “복잡한 비동기 overlap 실패”보다, **구조 자체가 직렬로 연결돼 있는 점**에 더 가깝다.

## 7. 지금 단계에서 가장 정직한 논문 메시지

- 현재 `full` 경로는 아직 baseline보다 느리다.
- `geometry_only`는 pure forward에서는 거의 본전이다.
- 실제 step에서 느려지는 큰 이유는 force backward와 pair execution 구조다.
- `lmax`는 4가 이론적 최대가 아니며, 높은 `lmax`는 가능하지만 비용이 빠르게 커진다.
- 따라서 mainline 논문은 `lmax=1..4` 중심으로 보수적으로 쓰고, `5..8`은 구조 분석과 잠재력 근거로 쓰는 것이 적절하다.

## 8. `h`와 “민감도”를 직관적으로 이해하는 방법

여기서 `h`는 **노드 피처**, 즉 각 원자가 현재까지 모은 정보를 담고 있는 내부 상태라고 생각하면 된다.

예를 들면 매우 단순하게:

```text
좌표 r
-> 거리 d(r)
-> 메시지 m(d)
-> 노드 피처 h(m)
-> 에너지 E(h)
```

라고 두자.

이때 힘은 결국

```text
F = - dE / dr
```

이다.  
그런데 `E`는 직접 `r`에서 나온 것이 아니라, `h`를 거쳐 나온다. 따라서 실제로는

```text
dE/dr = dE/dh * dh/dm * dm/dd * dd/dr
```

가 된다.

이 식에서 각 항이 뜻하는 바는 다음과 같다.

- `dE/dh`: **에너지가 노드 피처 변화에 얼마나 민감한가**
- `dh/dm`: **노드 피처가 메시지 변화에 얼마나 민감한가**
- `dm/dd`: **메시지가 거리/방향 변화에 얼마나 민감한가**
- `dd/dr`: **거리와 방향이 실제 좌표 변화에 얼마나 민감한가**

여기서 말하는 “민감도”는 그냥 **어느 입력을 조금 바꿨을 때 출력이 얼마나 크게 변하는가**라는 뜻이다.

아주 간단한 예를 보면 더 직관적이다.

```text
h = 2d
E = h^2
```

이면

```text
dE/dh = 2h
dh/dd = 2
```

이므로 `E`는 `d`에 대해 직접 미분하지 않아도,
중간 변수 `h`를 거쳐서 미분을 이어 붙일 수 있다.

GNN의 backward도 똑같다.  
즉 “최종 에너지를 좌표로 미분한다”는 말은 결국 **중간에 있는 모든 함수 연결을 타고 내려가며 민감도를 곱한다**는 뜻이다.

그래서 autograd가 필요한 것이다.  
autograd는 쓸데없는 추가 연산이 아니라, **원래 필요한 chain rule 계산을 자동화하는 도구**다.

## 9. 현재 병목을 한 장으로 요약하면

그림: [bottleneck_root_fix.svg](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/runtime/bottleneck_root_fix.svg)

현재 최신 결과를 가장 정확히 요약하면 다음과 같다.

### step 전체에서 큰 병목

- `pair metadata`
  - 최신 분리 실험에서 가장 큰 차이를 만든다
  - 대표값:
    - `mptrj`: `0.01 ms -> 189.58 ms`
    - `oc20_s2ef_train_20m`: `0.01 ms -> 57.89 ms`
    - `qm9_hf`: `0.01 ms -> 4.03 ms`

### model 내부의 추가 손해

- pair 값에서 edge 값으로 펼치는 `index_select`
- reverse edge sign 적용
- shared pair tensor로 gradient가 다시 모이는 backward fan-in
- `full` 경로의 pair-aware message path 오버헤드

즉 현재 시점에서는

- “backward가 느려서 전체가 망한다”
보다는
- “**pair metadata가 step 전체를 크게 악화시키고, 그 위에 backward/구조 오버헤드가 조금 더 얹힌다**”

가 더 정확하다.

## 10. 이 문제를 근본적으로 해결하려면

해결 순서는 명확하다.

### 1. pair metadata를 downstream에서 다시 만들지 말기

지금은 neighbor list가 있더라도 SevenNet 쪽에서 다시

- reverse pair 찾기
- canonical direction 정하기
- pair map 만들기

를 한다.

이건 불필요한 복원 작업이다.

근본 해결은

- upstream neighbor builder가
  - reverse edge index
  - canonical pair id
  - representative forward edge
를 직접 넘겨주는 것이다.

그러면 SevenNet은 pair metadata를 “복원”하지 않고 바로 “사용”할 수 있다.

### 2. pair 상태를 끝까지 유지하는 pair-major path

지금 `geometry_only`는

```text
pair에서 줄였다
-> 다시 edge로 펼쳤다
-> 원래 edge-major 경로로 계산했다
```

이다.

근본 해결은

```text
pair geometry
-> pair weight
-> pair-major message
-> pair-aware reduction
-> pair-aware backward
```

로 가는 것이다.

즉 중간에 edge로 다시 풀지 않아야 한다.

### 3. force backward도 pair-aware하게 설계

지금은 `autograd.grad(E, EDGE_VEC)`로 일반 계산 그래프를 그대로 타고 간다.

근본 해결은

- pair-major forward에 맞는 custom backward
- 혹은 pair 공유 구조를 이해하는 gradient 경로

를 두는 것이다.

그러면 shared pair tensor로 gradient를 모을 때의 추가 fan-in 비용을 줄일 수 있다.

## 11. LAMMPS / TorchSim에서 neighbor list를 쓰려던 시도는 어디까지 와 있나

### LAMMPS

현재 LAMMPS 경로는 neighbor list를 **직접** 받는다.

- serial: [pair_e3gnn.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn.cpp)
- parallel: [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp)

또 두 경로 모두 `REQ_FULL`을 요청한다.  
즉 full neighbor list는 이미 upstream에 있다.

하지만 pair 정보는 아직 **최대한 활용하지 못하고 있다.**

현재 구현은 neighbor list를 받은 뒤에도 다시

- string key/hash lookup
- reverse edge 찾기
- pair map 생성

을 한다.

즉:

- **neighbor list는 활용**
- **pair metadata는 다시 복원**

하는 구조다.

따라서 LAMMPS 쪽의 진짜 다음 단계는

- neighbor loop 안에서 바로 pair id 생성
- string key 제거
- reverse map을 바로 넘기기

다.

### TorchSim

TorchSim은 optional dependency가 맞다.  
현재 wrapper는 [torchsim.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/torchsim.py)에 있다.

지금 TorchSim wrapper는:

- `torchsim_nl`로 neighbor list를 받는다
- `edge_index`, `unit_shifts`, `edge_vec`를 만든다
- model forward를 호출한다

하지만 아직

- `prepare_pair_metadata()`
- `enable_pair_execution`
- `pair_execution_policy`

를 pair-aware하게 연결하지 않았다.

즉 TorchSim도

- **neighbor list는 upstream에서 이미 있음**
- **pair execution 연결은 아직 미완성**

인 상태다.

### 의미

LAMMPS와 TorchSim 둘 다 공통적으로,

- upstream neighbor 정보는 이미 있는데
- pair metadata는 downstream에서 다시 만들고 있거나
- 아예 pair path와 연결하지 못했다

는 게 문제다.

즉 근본 해결의 핵심은 **upstream neighbor builder와 pair runtime을 붙이는 것**이다.

## 12. 지금 당장 가장 현실적인 연구 방향

우선순위는 다음이 맞다.

1. pair metadata를 upstream neighbor output과 직접 연결
2. LAMMPS full-list fast path 추가
3. TorchSim pair-execution 연결
4. pair-major message/reduction path 구현
5. pair-aware backward
6. 그 다음에 FlashTP와 결합

즉 지금은 “FlashTP를 붙일까”보다
“**pair metadata를 다시 만들지 않는 구조로 갈 수 있나**”가 더 중요한 단계다.

## 13. 현재 구현된 upstream fast path 상태

위 방향 중에서 가장 먼저 손댈 수 있는 부분은 **LAMMPS 경로에서 neighbor loop를 도는 순간 pair metadata를 바로 구성하는 것**이었다.  
이 부분은 현재 코드에 1차 구현이 들어가 있다.

수정 파일:

- [pair_e3gnn.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn.cpp)
- [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp)

### 바뀐 점

기존에는 LAMMPS가 full neighbor list를 이미 준 뒤에도,
SevenNet 쪽에서 edge를 전부 만든 다음 다시

- reverse edge 찾기
- pair map 생성
- forward/backward pair index 생성

을 별도 함수에서 수행했다.

serial:

- `build_pair_metadata_serial(...)`

parallel:

- `build_pair_metadata_parallel(...)`

이 함수들은 결국 **두 번째 패스**로 edge 전체를 다시 돌며 reverse pair를 복원하는 구조였다.

현재 패치에서는 이 복원 단계를 neighbor loop 안으로 당겼다.

- serial 경로는 `SerialPairBuilder`
- parallel 경로는 `ParallelPairBuilder`

를 사용한다.

동작은 다음과 같다.

1. edge를 하나 만들 때 바로 canonical key를 만든다.
2. 그 key의 reverse가 이미 기다리고 있으면 현재 edge를 backward edge로 붙인다.
3. 없으면 현재 edge를 새 pair의 forward edge로 기록하고 reverse key를 waiting table에 등록한다.
4. neighbor loop가 끝나면 이미 채워진 배열을 tensor로 바꾼다.

즉 구조가

```text
neighbor loop
-> edge 생성
-> pair metadata 즉시 생성
```

으로 바뀌었다.

이 방식은 기존처럼

```text
neighbor loop
-> edge 생성
-> 두 번째 패스
-> string/hash 기반 reverse 복원
```

을 하지 않는다.

### 의미

이 패치가 줄이는 것은 SH나 TP가 아니라,
**LAMMPS 경로에서 pair metadata를 만들기 위해 edge를 다시 훑는 전처리 비용**이다.

따라서 기대 효과는 다음과 같다.

- string key/hash 기반 reverse lookup 제거
- second pass 제거
- pair metadata 생성 시점 단축
- full neighbor list 정보를 더 직접적으로 활용

### 아직 남아 있는 한계

이 구현이 곧바로 “근본 해결 완료”를 의미하지는 않는다.

남은 한계는 다음과 같다.

- 아직 LAMMPS build 환경에서 실제 컴파일 검증을 하지 못했다.
- 현재 세션에는 LAMMPS 빌드가 없어서 정적 코드 수정까지만 수행했다.
- topology cache와 model 내부 pair-aware 실행 구조는 그대로다.
- TorchSim 경로는 아직 같은 방식으로 연결하지 않았다.

즉 지금 단계는

- **LAMMPS upstream pair metadata fast path의 1차 구현**

까지 들어간 상태라고 보는 것이 정확하다.

### 다음 검증 항목

이 패치 이후 가장 먼저 확인해야 할 것은 아래 네 가지다.

1. LAMMPS serial 빌드가 정상적으로 되는지
2. LAMMPS parallel 빌드가 정상적으로 되는지
3. baseline 대비 `geometry_only`의 pair metadata 시간이 실제로 내려가는지
4. step 전체 시간에서 `pair_metadata` 항목 감소가 얼마인지

즉 이 패치는 방향이 분명하고 구조적으로도 맞지만,
현재는 **구현 완료, 실측 검증 대기** 상태다.
