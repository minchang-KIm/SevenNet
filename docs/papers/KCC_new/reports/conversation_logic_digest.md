# Conversation Logic Digest

이 문서는 지금까지 사용자와의 대화에서 나온 핵심 문제의식, 논리 구조, 구현 우선순위, 그리고 이에 대한 간략한 답변을 정리한 메모다.

## 1. 사용자 핵심 문제의식

### A. 논문에서 데이터보다 논리가 중요하다

- 사용자는 단순한 숫자 나열보다, 왜 이 구조가 의미가 있는지와 어떤 조건에서 통하는지가 더 중요하다고 지속적으로 강조했다.

간략한 응답 요약:
- 실험은 논리를 검증하는 근거로 배치해야 하며, direct claim은 현재 코드와 데이터가 지지하는 범위로 제한해야 한다.

### B. 정확성을 바꾸지 않고 속도만 바뀌는 것이 중요하다

- 사용자는 proposal이 정확도를 바꾸지 않는다는 점을 실험으로 분명히 보여야 한다고 요구했다.

간략한 응답 요약:
- baseline 기준 output difference를 반복 측정하고 mean/std를 표와 그림으로 정리해야 한다.

### C. FlashTP는 기대효과이지 현재 메인 비교축이 아니다

- 사용자는 현재 논문 메인 비교를 SevenNet baseline vs proposal로 잡아야 한다고 명확히 말했다.

간략한 응답 요약:
- 메인 실험은 baseline vs proposal-only 또는 baseline vs geometry_only로 고정하고, FlashTP는 후속 연구 또는 보조 진단으로 내려야 한다.

### D. LAMMPS / TorchSim upstream neighbor 정보를 제대로 활용해야 한다

- 사용자는 pair metadata를 downstream에서 다시 만드는 것이 비효율적이라고 지적했다.

간략한 응답 요약:
- upstream에서 pair id / reverse map을 직접 넘기는 것이 가장 쉽고 확실한 개선이며, 실제로 LAMMPS fast path는 유의미한 reduction을 보였다.

### E. geometry_only를 메인 타깃으로 삼아야 한다

- full path보다 geometry-only가 해석이 깨끗하고 현재 논문 범위에 맞는다는 문제의식이 제시되었다.

간략한 응답 요약:
- full은 아직 구조적으로 무겁고 pair-major가 아니므로, 지금 메인 축은 geometry_only가 맞다.

## 2. 구현/실험 관련 핵심 논리

### A. 왜 force inference에서 backward가 필요한가

- 사용자는 “추론인데 왜 autograd로 거슬러 올라가야 하느냐”를 여러 번 물었다.

간략한 응답 요약:
- force는 `F = -dE/dr`이므로, energy를 좌표에 대해 미분해야 한다.
- 최종 node feature는 현재 값만 가지고 있고 좌표 민감도까지 들고 있지 않으므로, chain rule을 따라 backward가 필요하다.

### B. 현재 느려지는 이유는 backward 하나가 아니라 구조 전체다

- 사용자는 pair reuse를 넣었는데 왜 더 느린지, 어디가 실제 병목인지 명확히 알고 싶어 했다.

간략한 응답 요약:
- 현재 손해는 geometry-side exact reuse 아이디어 자체보다는
  - pair metadata
  - pair->edge expansion
  - edge-major force path
  같은 runtime 구조에서 온다.

### C. 지금 구조는 pair-major가 아니라 pair-aware geometry reuse다

- 사용자는 현재 구현 범위를 정확히 규정해야 한다고 요구했다.

간략한 응답 요약:
- 현재 구현은 pair-major tensor product 전체가 아니라, geometry-side exact reuse만 먼저 구현한 단계다.

## 3. 사용자가 제시한 중요한 판단과 현재 반영 상태

### A. lmax 관련

- 사용자는 SH의 `l`과 `lmax` 의미를 계속 물었고, practical 범위를 어떻게 해석해야 하는지 검증을 요구했다.

간략한 응답 요약:
- `lmax`는 SH의 최대 차수이며, practical preset은 1~4가 많지만 이론적 최대가 4는 아니다.
- 다만 현재 논문 메인 축은 runtime exact reuse이므로, `lmax` sweep은 이번 canonical 본문에 넣지 않고 보조 자료로 남기는 것이 맞다.

### B. parallel / TorchSim / FlashTP

- 사용자는 이 세 축을 계속 물었지만, 현재 논문에는 무엇을 넣고 무엇을 빼야 하는지 정리를 원했다.

간략한 응답 요약:
- parallel은 현재 환경과 논문 범위상 제외
- TorchSim은 separate runtime이므로 현재는 제외
- FlashTP도 현재 메인 비교축에서는 제외

## 4. 지금 논문에서 쓰는 최종 논리 구조

1. SevenNet은 directed edge 기반이라 geometry-side 중복 계산이 존재한다.
2. reverse edge pair를 이용하면 distance / radial / cutoff / SH / pair weight input은 exact reuse가 가능하다.
3. 이를 geometry_only runtime으로 구현했다.
4. accuracy repeat 결과는 energy/force 차이가 매우 작아 정확도 보존형 runtime optimization으로 볼 수 있다.
5. 그러나 generic calculator path에서는 아직 baseline보다 느리다.
6. intrusive breakdown과 LAMMPS upstream fast path는, 문제의 핵심이 exact reuse 수식이 아니라 runtime 구조에 있음을 보여준다.
7. 따라서 현재 논문의 novelty는 speedup claim보다, exact reuse 구현과 병목 규명, 그리고 다음 pair-major 우선순위 제시에 있다.

## 5. 앞으로의 직접적인 구현 우선순위

1. upstream pair metadata
2. pair->edge expand 최소화
3. pair weight expand 최소화
4. edge-major force path 구조 축소
5. 그 다음 pair-major message path

## 6. 지금 논문에서 피해야 하는 과장

- “현재 geometry_only가 이미 large/dense에서 빠르다”
- “FlashTP와 결합하면 확실히 이긴다”
- “현재 구현이 이미 pair-major runtime이다”

## 7. 지금 논문에서 적극적으로 써도 되는 주장

- exact reuse formulation은 정확도 보존형 runtime optimization이다
- 현재 generic path에서는 아직 승리하지 못했다
- 하지만 intrusive breakdown과 LAMMPS upstream fast path는 runtime 병목이 어디인지 분명히 보여준다
- upstream pair integration은 쉽고 결과가 확실한 개선으로 이미 확인되었다
