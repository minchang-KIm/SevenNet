# KCC_new Package

이 디렉터리는 `baseline vs geometry_only`를 메인 비교축으로 다시 정리한 KCC 제출용 canonical 패키지다.

핵심 원칙:

1. 메인 실험은 `SevenNet baseline`과 `SevenNet + geometry_only` 비교만 사용한다.
2. 모든 메인 반복 실험은 `repeat 30` 기준으로 정리한다.
3. `intrusive diagnostic`과 `non-intrusive end-to-end latency`는 같은 축에서 직접 비교하지 않는다.
4. 현재 구현은 `pair-major full runtime`이 아니라 `pair-aware geometry reuse`로 기술한다.

## 폴더 구조

- `manuscript/`
  - 제출용 한국어 원고와 무기명 원고
- `scripts/`
  - `KCC_new` 실험 러너와 자산 생성기
- `metrics/`
  - 30회 반복 메인 성능/정확도 결과
- `figures/`
  - 논문 삽입용 PNG/SVG
- `tables/`
  - 논문 삽입용 CSV/Markdown 표
- `reports/`
  - 실험 환경, 결과 해석, 진단 결과 메모
- `geometry_only_breakdown/`
  - intrusive geometry-only stage breakdown
- `lammps_pair_metadata_bench/`
  - serial LAMMPS upstream pair-metadata fast path 진단

## 실행 순서

1. 메인 성능:
   - `python docs/papers/KCC_new/scripts/kcc_new_pair_end_to_end.py`
2. 정확도 반복:
   - `python docs/papers/KCC_new/scripts/kcc_new_pair_accuracy_repeats.py`
3. 자산 생성:
   - `python docs/papers/KCC_new/scripts/kcc_new_build_assets.py`
4. representative profiler:
   - `python docs/papers/KCC_new/scripts/kcc_new_pair_profiler_representatives.py`

## 현재 canonical 문서

- `reports/main_result_summary.md`
  - 메인 수치 요약
- `reports/kcc_experiment_inventory.md`
  - 기존 `KCC` 폴더의 실험 종류와 현재 canonical 실험 매트릭스 정리
- `reports/conversation_logic_digest.md`
  - 사용자 문제의식과 현재 논문 논리 구조 요약
- `reports/kcc_new_context_full.md`
  - 논문/실험/진단을 한 번에 설명하는 장문 컨텍스트
- `reports/execution_status.md`
  - 이번 패스에서 실제로 실행한 실험과 제외한 실험 정리
- `tables/table_01_experiment_setup.md`
  - 환경/반복/비교축
- `tables/table_02_end_to_end_summary.md`
  - 31개 데이터셋 메인 성능 요약
- `tables/table_03_condition_summary.md`
  - 어떤 조건에서 geometry_only가 유리한지
- `tables/table_04_accuracy_summary.md`
  - 정확도 보존 결과
- `tables/table_05_geometry_breakdown_summary.md`
  - geometry_only 내부 확장 비용 요약
- `tables/table_06_lammps_pair_metadata_summary.md`
  - LAMMPS upstream fast path 진단
- `reports/pair_profiler_representative_report.md`
  - representative `torch.profiler` 진단
- `reports/pair_profiler_interpretation.md`
  - profiler 결과 해석

## 주의

- `geometry_only_breakdown/`은 intrusive timing이므로 headline latency로 쓰면 안 된다.
- `lammps_pair_metadata_bench/`는 현재 single-GPU serial `pair_style e3gnn` 진단이다.
- 메인 논문 주장에는 `baseline vs geometry_only` 결과만 사용하고, 나머지는 원인 분석과 후속 설계 근거로 사용한다.
