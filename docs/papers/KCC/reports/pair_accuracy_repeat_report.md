# Pair Accuracy Repeat Report

이 보고서는 같은 대표 샘플을 여러 번 다시 계산해,
기본 실행 방식의 반복 잡음과 제안기법의 기준 출력 차이를 함께 정리한 결과이다.

- 데이터셋 수: 31
- baseline median mean force diff: 1.204e-06 eV/A
- proposal median mean force diff: 1.867e-06 eV/A
- baseline worst force diff max: 1.526e-04 eV/A
- proposal worst force diff max: 2.441e-04 eV/A
- baseline median mean energy diff: 0.000e+00 eV
- proposal median mean energy diff: 0.000e+00 eV
- baseline worst energy diff max: 1.221e-04 eV
- proposal worst energy diff max: 1.221e-04 eV

핵심 해석:

- baseline 반복 실행 자체에서도 부동소수점 수준의 아주 작은 차이가 관측된다.
- proposal의 에너지/힘 차이는 이 기준 출력에 대해 매우 작은 절대값을 유지한다.
- 따라서 현재 제안기법은 출력 정확도를 바꾸기보다 실행 시간을 바꾸는 최적화로 해석하는 것이 맞다.

주의:

- 이 표와 그림은 baseline 기준 출력 대비 절대 차이이다.
- 표준편차는 같은 샘플을 여러 번 반복 실행해 얻은 값이다.
- 0값도 존재하므로 그림은 로그축 표시를 위해 작은 floor 값을 사용한다.
