# Existing KCC Experiment Inventory

이 문서는 기존 `docs/papers/KCC/` 폴더에서 어떤 실험을 했는지, 각각 무엇을 확인하려는 실험인지, 현재 논문에서 어떻게 써야 하는지를 정리한다.

## 1. end-to-end 메인 비교

### script
- `docs/papers/KCC/scripts/kcc_pair_end_to_end.py`
- `docs/papers/KCC/scripts/kcc_build_pair_end_to_end_assets.py`

### purpose
- `SevenNet baseline` vs `proposal-only(full)` non-intrusive steady-state latency 비교

### repeat policy in old KCC
- warmup `3`
- repeat `10`

### key outputs
- `metrics/pair_end_to_end/global/pair_end_to_end_summary.csv`
- `metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv`
- `reports/pair_end_to_end_condition_analysis.md`
- `figures/pair_end_to_end/*`

### status in KCC_new
- **재실행함**
- 다만 메인 비교축을 `baseline vs geometry_only`로 변경했고 repeat를 `30`으로 다시 맞춤

## 2. 정확도 반복

### script
- `docs/papers/KCC/scripts/kcc_pair_accuracy_repeats.py`
- `docs/papers/KCC/scripts/kcc_build_accuracy_assets.py`

### purpose
- baseline 기준 출력 대비 proposal의 energy / force 차이 반복 측정

### repeat policy in old KCC
- warmup `2`
- repeat `10`

### key outputs
- `metrics/pair_accuracy/global/pair_accuracy_summary.csv`
- `reports/pair_accuracy_repeat_report.md`
- `figures/pair_accuracy/*`

### status in KCC_new
- **재실행함**
- `baseline vs geometry_only`, repeat `30`

## 3. split validation

### script
- `docs/papers/KCC/scripts/kcc_pair_validation_suite.py`
- `docs/papers/KCC/scripts/kcc_build_pair_validation_assets.py`

### purpose
- `baseline / geometry_only / full_legacy / full_no_expand`
- `forward_energy / step_force`
- current full path가 왜 느린지 원인 분리

### repeat policy in old KCC
- warmup `5`
- repeat `100`

### key outputs
- `metrics/pair_validation_split/global/pair_validation_summary.csv`
- `metrics/pair_validation_split/global/pair_metadata_summary.csv`
- `reports/pair_validation_split_report.md`
- `reports/pair_validation_interpretation.md`
- `tables/table_06~09_*`

### status in KCC_new
- **직접 재실행하지 않음**
- 현재 논문 메인 비교축이 `geometry_only`이므로, old KCC 결과를 참고 진단으로만 사용

## 4. geometry_only breakdown

### script
- `docs/papers/KCC/scripts/kcc_geometry_only_breakdown.py`

### purpose
- `baseline` vs `geometry_only`
- intrusive forward-only stage 분해
- pair->edge expansion, pair weight expand, pair geometry 비용 확인

### repeat policy
- warmup `5`
- repeat `30`

### key outputs
- `geometry_only_breakdown/metrics/geometry_only_breakdown_summary.csv`
- `geometry_only_breakdown/reports/geometry_only_breakdown_report.md`

### status in KCC_new
- **기존 현재 코드 기준 결과를 canonical diagnostic으로 복사**
- main 논문의 핵심 진단 근거로 사용

## 5. LAMMPS pair metadata bench

### script
- `docs/papers/KCC/scripts/kcc_lammps_pair_metadata_bench.py`

### purpose
- serial `pair_style e3gnn`
- `baseline`, `geometry_only_legacy`, `geometry_only_upstream`
- upstream pair-metadata fast path가 실제로 pair metadata 시간을 얼마나 줄이는지 확인

### repeat policy
- repeat `30`

### key outputs
- `lammps_pair_metadata_bench/metrics/lammps_pair_metadata_summary.csv`
- `lammps_pair_metadata_bench/reports/lammps_pair_metadata_report.md`

### status in KCC_new
- **기존 현재 코드 기준 결과를 canonical diagnostic으로 복사**
- 현재 가장 강한 positive diagnostic evidence

## 6. four-case profile

### script
- `docs/papers/KCC/scripts/kcc_four_case_profile.py`
- `docs/papers/KCC/scripts/kcc_build_four_case_assets.py`

### purpose
- 옵션 없음 / FlashTP / 제안기법 / 둘 다
- intrusive stage profiling

### repeat policy
- repeat `5`

### status in KCC_new
- **현재 논문 canonical 범위에서는 제외**
- 이유: 메인 비교축이 `baseline vs geometry_only`이고, FlashTP는 후속 확장 방향임

## 7. representative profiler

### old script
- `docs/papers/KCC/scripts/kcc_pair_profiler_representatives.py`

### purpose
- old KCC에서는 `baseline / full_legacy / full_no_expand`를 representative dataset에서 `torch.profiler`로 확인

### status in KCC_new
- **새로 baseline vs geometry_only representative profiler를 추가 실행**
- script:
  - `docs/papers/KCC_new/scripts/kcc_new_pair_profiler_representatives.py`

## 8. lmax 계열 실험

### scripts
- `docs/papers/KCC/scripts/kcc_lmax_baseline_sweep.py`
- `docs/papers/KCC/scripts/kcc_lmax_quadrant_runtime.py`
- `docs/papers/KCC/scripts/kcc_lmax_strength_analysis.py`

### purpose
- `lmax` 하이퍼파라미터가 baseline 정확도와 비용에 어떤 영향을 주는지 분석
- large/dense graph에서 lmax 비용이 어떻게 커지는지 확인

### status in KCC_new
- **현재 메인 논문 축에서는 제외**
- 이유: 이번 논문 메인 비교축은 geometry_only runtime 자체이기 때문
- 단, 보조 해석 자료로는 유지 가능

## 9. Nsight / FlashTP 계열

### scripts
- `docs/papers/KCC/scripts/kcc_nsys_collect.py`
- `docs/papers/KCC/scripts/kcc_nsys_target.py`

### purpose
- representative Nsight validation
- FlashTP kernel composition 참고

### status in KCC_new
- **현재 canonical 논문 범위에서는 제외**

## 10. KCC_new에서 지금 canonical로 쓰는 실험 종류

1. main end-to-end
   - baseline vs geometry_only
   - repeat 30
2. accuracy repeat
   - baseline vs geometry_only
   - repeat 30
3. geometry_only intrusive breakdown
   - repeat 30
4. LAMMPS upstream pair-metadata bench
   - repeat 30
5. representative torch profiler
   - baseline vs geometry_only

즉 현재 논문은 성능 승리보다,
- exact reuse 구현
- 정확도 보존
- generic runtime 병목 규명
- upstream pair integration의 실효성
을 메인 축으로 삼는 것이 맞다.

