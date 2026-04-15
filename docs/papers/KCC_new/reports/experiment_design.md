# KCC_new Experiment Design

## Main Question

같은 SevenNet 모델과 같은 출력 정확도를 유지하면서, `geometry_only` 실행 방식이 실제 추론 시간을 줄일 수 있는가.

## Main Comparison

- `SevenNet baseline`
- `SevenNet + geometry_only`

## Main Metrics

- steady-state latency mean / std / median / p95
- speedup = `baseline / geometry_only`
- absolute energy difference vs baseline
- maximum absolute force difference vs baseline

## Diagnostic Metrics

- geometry-only intrusive stage breakdown
- LAMMPS pair metadata total / build time
- pair metadata reduction factor

## Dataset Policy

- benchmarkable public-local datasets 전체
- 각 데이터셋에서 representative sample 1개 사용
- dataset size / density 메타 정보 저장

## Repeat Policy

- main end-to-end latency: warmup 3, repeat 30
- accuracy repeat: warmup 2, repeat 30
- geometry-only breakdown: existing repeat 30 diagnostic
- LAMMPS pair-metadata bench: existing repeat 30 diagnostic

## Expected Paper Structure

1. 문제 정의: directed edge 중복으로 인한 geometry-side 중복 계산
2. 제안: pair-aware geometry reuse
3. 정확도 보존
4. 메인 성능 결과
5. 어떤 조건에서 유리한지
6. 왜 아직 모든 경우에 빠르지 않은지
7. LAMMPS upstream pair-metadata 최적화와 향후 pair-major 방향
