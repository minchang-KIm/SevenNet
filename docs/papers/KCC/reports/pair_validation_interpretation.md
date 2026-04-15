# Pair Validation Interpretation

## 목적

이 문서는 `baseline / geometry_only / full_legacy / full_no_expand` 분리 검증의 의미를 정리한다.  
핵심 질문은 세 가지다.

1. 순수 forward에서 기하 정보 재사용만 떼어 놓고 보면 실제로 얼마나 손해인지
2. force 계산이 들어가면 왜 결과가 더 나빠지는지
3. 현재 구현의 구조적 문제와 다음 단계 수정 우선순위가 무엇인지

## 각 실험이 뜻하는 것

- `baseline`
  - 현재 SevenNet 기본 실행 방식
- `geometry_only`
  - pair metadata를 만들고, 거리, radial basis, cutoff, 구면조화함수 같은 기하 정보만 pair 기준으로 재사용
  - convolution을 full pair 방식으로 나누지 않으므로, 제안기법의 가장 좁은 의미를 본다
- `full_legacy`
  - 현재 full pair 실행 전체
  - pair 재사용, forward/reverse 분리, edge 확장까지 모두 포함
- `full_no_expand`
  - `full_legacy`와 같지만 `EDGE_ATTR/EDGE_EMBEDDING`의 불필요한 full-edge 확장을 줄인 실험 버전

측정 모드는 두 가지다.

- `forward_energy`
  - 입력 그래프를 미리 만든 뒤 에너지만 계산
  - 순수 forward 경로에서 제안기법 자체가 얼마나 듣는지 본다
- `step_force`
  - 그래프 생성, pair metadata, device 이동, 모델 실행, 힘 계산까지 포함
  - 실제 MD 한 step과 가까운 추론 경로를 본다

## 핵심 수치

31개 데이터셋 전체, 100회 반복 기준 중앙값:

- `baseline / geometry_only`
  - `forward_energy`: `0.977x`
  - `step_force`: `0.684x`
- `baseline / full_legacy`
  - `forward_energy`: `0.865x`
  - `step_force`: `0.696x`
- `baseline / full_no_expand`
  - `forward_energy`: `0.862x`
  - `step_force`: `0.694x`
- `full_legacy / full_no_expand`
  - `forward_energy`: `0.996x`
  - `step_force`: `0.998x`

즉, 현재 관측은 다음처럼 읽는 것이 맞다.

- `geometry_only`는 pure forward에서는 거의 본전이다.
- `full`은 pure forward에서도 이미 느리다.
- `full_no_expand` 패치는 거의 효과가 없다.
- force 계산이 들어가면 `geometry_only`도 크게 느려진다.

## 왜 forward-only를 따로 봐야 하는가

이 질문은 타당하다. MD 한 step 전체를 한 번에 재면, 제안기법 자체의 효과와 force 계산, pair metadata, 그래프 준비 비용이 섞인다. 그러면 “재사용 아이디어가 안 듣는 것인지”, 아니면 “뒤쪽 경로가 더 무거운 것인지”를 분리할 수 없다.

`forward_energy`는 바로 이 문제를 푸는 실험이다. 입력 그래프를 미리 device에 올리고, `force_output`을 제거한 에너지 전용 모델만 호출한다. 이렇게 하면 기하 정보 재사용이 순수 forward 경로에서 어느 정도까지 갈 수 있는지 볼 수 있다.

결과는 명확하다.

- `geometry_only`가 `forward_energy`에서 `0.977x`까지 올라온다.
- 같은 case가 `step_force`에서는 `0.684x`로 내려간다.

이 차이는 “기하 정보 재사용 자체는 거의 본전인데, force 경로와 step 바깥 오버헤드가 이를 크게 상쇄한다”는 뜻이다.

## backward는 어디에서 왜 생기는가

SevenNet의 force 계산은 마지막 출력층만 다시 계산하는 구조가 아니다.  
에너지를 예측한 뒤, 그 에너지를 `EDGE_VEC`에 대해 미분해 힘과 stress를 얻는다.

코드 경로는 다음과 같다.

- `model_build.py`
  - 모델 마지막에 `ForceStressOutputFromEdge`를 붙인다.
- `force_output.py`
  - `torch.autograd.grad(energy, [rij])`를 호출한다.

즉 backward는 readout MLP만 통과하는 것이 아니라, 에너지 계산에 참여한 상류 전체를 다시 지난다.

- readout
- interaction block
- convolution
- `weight_nn`
- geometric path

그래서 pure forward와 force 포함 결과가 크게 다르게 나오는 것이 자연스럽다.

대표 `torch.profiler` 결과도 이를 뒷받침한다.

- `qm9_hf`, `forward_energy`
  - `aten::tensordot`, `aten::mm`, `aten::einsum`이 중심
- `mptrj`, `force_model`
  - `autograd::engine::evaluate_function`
  - `SliceBackward0`
  - `aten::slice_backward`
  - `aten::add_`
  - `aten::mul`
  가 상위에 나타난다.

즉 force 경로의 문제는 “마지막 MLP가 비싸다”가 아니라, 전체 autograd graph가 다시 도는 데 있다.

## current full이 왜 느린가

분리 검증 결과를 보면 `full_legacy`는 pure forward에서도 이미 `0.865x`다.  
따라서 backward만이 문제는 아니다.

현재 full 구현이 느린 주된 이유는 다음 순서로 해석하는 것이 가장 자연스럽다.

1. pair-major 전체 실행이 아니라 forward/reverse 분리 실행이다.
2. pair 상태를 끝까지 유지하지 못하고 중간에 edge 중심 흐름으로 돌아간다.
3. `EDGE_ATTR/EDGE_EMBEDDING` 확장 제거 패치 하나로는 이 구조적 손해를 줄이지 못한다.

`full_legacy / full_no_expand`가 거의 `1.0`인 것은, 지금 병목이 “확장 한 번”이 아니라 훨씬 더 큰 실행 구조에 있음을 말해 준다.

## pair metadata는 실제 병목인가

그렇다. 특히 큰 그래프에서는 분명히 병목이다.

중앙값 기준:

- `cpu_original`: `55.866 ms`
- `cpu_vectorized`: `10.019 ms`
- `gpu_vectorized_kernel_only`: `0.730 ms`

따라서 현재 Python/CPU 기반 `prepare_pair_metadata`는 단순한 주변 비용이 아니라, 실제로 줄여야 할 비용이다.  
다만 이것만 고쳐도 full path 전체가 좋아지는 것은 아니다. 현재 결과는 pair metadata와 full path 구조 문제가 동시에 존재함을 보여 준다.

## 이 결과가 뜻하는 연구 방향

이번 분리 검증으로 추측이 아니라 사실로 확인된 것은 다음이다.

1. `geometry_only`는 순수 forward에서는 거의 본전이다.
2. force 포함 경로가 되면 이득이 크게 사라진다.
3. current full path는 pure forward에서도 구조적으로 느리다.
4. `EDGE_ATTR/EDGE_EMBEDDING` expansion 제거만으로는 부족하다.
5. pair metadata는 벡터화/GPU화 여지가 매우 크다.

따라서 다음 연구 우선순위는 아래가 맞다.

1. pair metadata를 CPU 파이썬 경로에서 벡터화 또는 device 경로로 옮기기
2. current full path를 patch 수준으로 고치는 것이 아니라, pair 상태를 끝까지 유지하는 pair-major message path로 다시 설계하기
3. force backward를 pair-aware하게 다루는 방법을 별도로 설계하기
4. 그 다음 단계에서 FlashTP와 결합하기

즉, FlashTP 결합은 지금 당장 메인 해법이 아니라, SevenNet 기본 경로에서 구조적 손해를 먼저 줄인 뒤에 얹어야 할 후속 단계다.
