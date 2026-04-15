# full 경로 pair-major화 가능성 메모

이 메모는 현재 `full` 경로를 더 pair-major하게 바꾸는 작업이 지금 당장 쉽고 확실한 개선인지 판단하기 위한 메모다.

## 1. 현재 상태

현재 `IrrepsConvolution._pair_forward()`는 이미 다음 개선이 들어간 상태다.

- forward branch와 reverse branch를 따로 convolution 두 번 돌리지 않음
- forward/reverse `src/dst/filter/weight`를 concat해서
- convolution 1회 + gather 1회로 줄였음

파일:

- [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)

핵심 구조:

```text
pair_weight 1회
-> forward edge 정보 준비
-> reverse edge 정보 준비
-> concat
-> convolution 1회
-> gather 1회
```

즉 현재 `full`은 예전보다 이미 한 단계 pair-major에 가까워진 상태다.

## 2. 남은 쉬운 패치가 있는가

코드상 남아 있는 가장 쉬운 후보는 `expand_full_edges=False`를 기본으로 두는 것이다.

왜냐하면:

- [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)에는 `PAIR_EDGE_ATTR`, `PAIR_EDGE_REVERSE_ATTR`, `PAIR_EDGE_EMBEDDING`이 이미 있고
- [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)의 `_pair_forward()`도 `expand_full_edges=False`면 pair attr를 직접 쓸 수 있게 되어 있다

즉 구현 난이도만 보면 어렵지 않다.

## 3. 그런데 왜 바로 안 넣는가

이미 실험으로 거의 확인했다.

- `full_legacy`
- `full_no_expand`

를 비교했을 때, single-pass 이후에도 `no_expand`가 거의 개선을 주지 못했다.

대표값:

- `mptrj` step-force
  - `full_legacy`: `464.04 ms`
  - `full_no_expand`: `465.71 ms`
- `qm9_hf` step-force
  - `31.33 ms`
  - `31.52 ms`
- `oc20_s2ef_train_20m` step-force
  - `139.81 ms`
  - `140.76 ms`

파일:

- [pre_fused_recheck_singlepass.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pre_fused_recheck_singlepass.csv)

즉 현재 병목은 `EDGE_ATTR/EDGE_EMBEDDING` 확장 하나가 아니다.

## 4. 그래서 남은 작업은 어떤 종류인가

진짜 다음 단계는 아래 셋 중 하나다.

1. pair-major message path 재설계
2. pair-aware backward
3. custom fused kernel

이 셋은 모두 “쉽고 확실한 미세 패치” 범주가 아니다.

## 5. 현재 판단

- **쉽다**: `expand_full_edges` 같은 소규모 플래그 패치
- **효과가 확실하다**: 아님
- **지금 결과로 바로 진행할 만하다**: 아님

즉 현재 코드와 실험 결과를 같이 보면,
`full`을 더 pair-major하게 만드는 진짜 다음 단계는 이미 작은 패치 수준을 넘어섰다.

## 6. 결론

지금 단계에서 더 진행해야 할 우선순위는:

1. `geometry_only`를 더 깔끔하게 만드는 것
2. LAMMPS 병렬 경로에서 실제 pair metadata 오버헤드를 다시 보는 것
3. 그 다음에 pair-major/custom kernel 설계로 넘어가는 것

따라서 현재는 `full` 추가 패치를 더 밀기보다, pair-major 재설계를 별도 과제로 두는 판단이 맞다.
