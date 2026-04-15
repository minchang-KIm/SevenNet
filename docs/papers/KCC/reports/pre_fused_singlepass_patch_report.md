# Pre-Fused Single-Pass Full Path Recheck

## 목적

완전한 pair-major fused kernel 전 단계에서, 현재 `full` 경로의 구조적 손해를 줄일 수 있는지 확인했다.  
이번 수정은 `forward`와 `reverse`를 따로 두 번 돌리던 convolution 경로를, 한 번의 convolution 호출과 한 번의 gather로 합치는 것이다.

## 코드 변경

- 대상 파일: [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)
- 변경 내용:
  - 기존 `full` 경로는
    - forward convolution 1회
    - reverse convolution 1회
    - gather 2회
    구조였다.
  - 현재는
    - forward/reverse edge를 먼저 합친 뒤
    - convolution 1회
    - gather 1회
    로 바꾸었다.

## 실험 조건

- 실행 환경: 현재 세션에서 보이는 GPU는 `1 x RTX 4090`
- 실험 경로: `ASE calculator` 기반 단일 GPU 재측정
- 반복 수: `warmup 3`, `repeat 10`
- 비교 대상:
  - `baseline`
  - `geometry_only`
  - `full_legacy`
  - `full_no_expand`

주의:
- 이 결과는 **단일 GPU, 현재 코드 트리** 기준의 대표셋 재측정이다.
- SevenNet 실행 가이드의 **2-GPU 병렬 MD 실험**은 별도로 `LAMMPS e3gnn/parallel` 경로에서 수행해야 한다.

## 대표 결과

원본 CSV: [pre_fused_recheck_singlepass.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pre_fused_recheck_singlepass.csv)

### Step Force

| dataset | baseline total ms | geometry_only total ms | full_legacy total ms | full_no_expand total ms |
| --- | ---: | ---: | ---: | ---: |
| `mptrj` | 424.70 | 459.94 | 464.04 | 465.71 |
| `oc20_s2ef_train_20m` | 124.66 | 136.54 | 139.81 | 140.76 |
| `qm9_hf` | 29.26 | 30.68 | 31.33 | 31.52 |

### Forward Energy

| dataset | baseline model ms | geometry_only model ms | full_legacy model ms | full_no_expand model ms |
| --- | ---: | ---: | ---: | ---: |
| `mptrj` | 54.69 | 56.06 | 58.05 | 58.10 |
| `oc20_s2ef_train_20m` | 18.28 | 18.64 | 19.75 | 19.92 |
| `qm9_hf` | 9.23 | 9.45 | 9.91 | 9.94 |

## 해석

1. 이번 single-pass 변경은 `full` 경로의 구조를 단순하게 만들었다.
2. 대표셋 기준으로는 `full`이 여전히 baseline보다 느리다.
3. 다만 이전처럼 큰 폭으로 손해가 나는 구조에서, baseline 근처까지 손해를 줄이는 방향으로는 개선되었다.
4. `full_no_expand`는 이번 재측정에서도 `full_legacy`보다 거의 낫지 않았다. 즉 지금 병목은 edge expansion 한 항목보다, 전체 pair 실행 구조와 backward 경로 쪽에 더 가깝다.
5. 지금 단계의 결론은 다음과 같다.
   - `pair metadata`를 줄이고
   - `full`의 convolution 분할을 줄이는 것만으로는
   - large graph에서 손해폭은 줄일 수 있지만, baseline을 안정적으로 넘기기엔 부족하다.

## 다음 단계

우선순위는 다음과 같다.

1. `pair metadata`의 남은 CPU signature 비용과 캐시 경로를 더 줄인다.
2. `full` 경로를 더 pair-major하게 바꿔 source gather와 message 계산을 계속 pair 상태로 유지한다.
3. force 계산에서 generic autograd backward가 차지하는 비용을 다시 분리 측정한다.
4. 그 다음에야 FlashTP 같은 TP 계열 최적화와의 결합을 보는 것이 맞다.
