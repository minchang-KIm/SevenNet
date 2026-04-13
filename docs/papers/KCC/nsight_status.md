# Nsight Status

KCC 패키지에는 representative Nsight 수집 스크립트를 포함한다.

- `scripts/kcc_nsys_target.py`
- `scripts/kcc_nsys_collect.py`

그러나 현재 canonical KCC 결과는 Nsight 추적으로부터 직접 생성하지 않는다. 이유는 다음과 같다.

1. 본 논문의 주 결과는 31개 데이터셋 전수에 대한 repeated timing과 stage decomposition이다.
2. 이 전수 통계에는 lightweight하고 반복 가능한 synchronized wall-clock 계측이 더 적합하다.
3. 현재 호스트의 `nsys 2020.3` 환경에서는 FlashTP Python 경로에 대한 per-run export와 post-processing 안정성이 충분하지 않았다.
4. 따라서 Nsight는 representative validation을 위한 보조 도구로만 위치시키고, canonical 본문 수치와 그림은 다음 세 family로 제한한다.

- `e3nn baseline detailed`
- `e3nn pair detailed`
- `FlashTP end-to-end`

즉, 본 패키지의 논문 본문에서 headline latency와 stage 해석은 위 세 family를 기준으로 읽어야 하며, Nsight 스크립트는 후속 kernel mix 검증을 위한 보조 자산으로 간주한다.
