# Pair Validation Split Report

## 각 실험의 의미

- `baseline vs geometry_only`: 기하 정보 재사용만의 효과를 본다. 메시지 계산 스케줄 변경은 최소화된다.
- `baseline vs full_legacy`: 기존 full pair 실행 전체 효과를 본다. 여기에는 pair 재사용과 forward/reverse 분리, edge 확장이 함께 들어간다.
- `baseline vs full_no_expand`: full 실행에서 불필요한 edge 확장을 줄였을 때 실제 개선이 생기는지 본다.
- `step_force`: 실제 MD 한 step에 가까운 경로다. 그래프 생성, pair metadata, device 이동, 모델 실행, 힘 계산까지 포함한다.
- `forward_energy`: 입력 그래프를 미리 만든 뒤 에너지만 계산한다. 순수 forward 경로에서 geometry 재사용이 얼마나 듣는지 본다.
- `pair metadata method comparison`: 현재 CPU pair metadata가 병목인지, 벡터화/GPU화 여지가 얼마나 있는지 본다.

## 핵심 비교 지표

- step_force median baseline/full_no_expand: 0.694x
- step_force median baseline/geometry_only: 0.684x
- step_force median full_legacy/full_no_expand: 0.998x
- forward_energy median baseline/full_no_expand: 0.862x
- forward_energy median baseline/geometry_only: 0.977x
- forward_energy median full_legacy/full_no_expand: 0.996x
- geometry_only: step/forward speedup ratio median = 0.701
- full_legacy: step/forward speedup ratio median = 0.780
- full_no_expand: step/forward speedup ratio median = 0.780

## pair metadata 비교

- median cpu_original: 55.866 ms
- median cpu_vectorized: 10.019 ms
- median gpu_vectorized_kernel_only: 0.730 ms

## 앞으로의 연구 방향

- 큰 그래프에서 forward 쪽 이득이 실제 step 전체로 전달되는지와, force backward가 이를 얼마나 상쇄하는지를 함께 봐야 한다.
- full 경로는 pair 상태를 끝까지 유지하지 못하는 구간이 아직 남아 있으므로, pair-major tensor product와 reduction으로 이어져야 한다.
- CPU pair metadata는 현재 구조에서 실사용 경로에 남아 있으므로, 벡터화 또는 GPU 이관의 실익을 별도로 검증할 가치가 있다.
- FlashTP 결합은 이 결과 위에 올리는 후속 단계로 두는 것이 맞다. 먼저 SevenNet 기본 경로에서 어디서 이득이 생기는지 분리해야 한다.
