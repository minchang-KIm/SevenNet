# KCC Pair-Aware Geometry Reuse Package

이 디렉터리는 KCC 제출용 원고, 표, 그림, 메트릭, 생성 스크립트를 한곳에 모은 canonical 작업 공간이다.  
핵심 원칙은 다음 세 가지다.

1. 현재 구현은 `pair-major TP`가 아니라 `pair-aware geometry-side exact reuse`로 기술한다.
2. `intrusive detailed profile`과 `non-intrusive end-to-end latency`는 같은 축에서 직접 비교하지 않는다.
3. 실험 결과는 논리를 보조하는 근거로 사용하고, 본문 주장은 구현 범위와 상보성 설명에 맞춘다.

이 README의 목적은 단순한 파일 목록이 아니라, **상위 모델이나 다른 연구자가 KCC 폴더 전체를 읽고 바로 실험 설계와 논문 정리를 이어갈 수 있도록 각 문서의 역할을 설명하는 것**이다.

## 이 패키지를 어떻게 읽어야 하나

가장 먼저 봐야 할 문서:

- `reports/chatgpt_pro_prompt_context_full.md`
  - 현재 구현, 실험, 해석, 열린 가설을 한 번에 설명한 가장 큰 컨텍스트 문서
  - ChatGPT Pro 같은 상위 모델에 넣을 용도로 만든 파일
  - 무엇이 사실이고 무엇이 가설인지 최대한 분리해 둠
- `manuscript/01_kcc_pair_geometry_reuse_ko.md`
  - 현재 한국어 논문 본문 초안
  - 지금까지의 결과를 바탕으로 실제 논문 구조를 어떻게 잡았는지 보여줌
- `reports/pair_end_to_end_condition_analysis.md`
  - 메인 성능 결과가 어떤 조건에서 유리한지 정리한 핵심 보고서
- `reports/pair_validation_interpretation.md`
  - 현재 구현이 왜 small graph에서 느리고 large graph에서만 이득이 나는지 분리 검증한 해석 문서
- `reports/lmax_baseline_sweep_report.md`
  - `lmax` 실험의 핵심 결과를 요약한 문서
- `reports/lmax_quadrant_runtime_report.md`
  - `lmax` 비용이 어떤 데이터셋 구조에서 더 빨리 커지는지 정리한 문서

논문에 바로 쓰기보다 **원인 분석이나 후속 설계를 위한 진단 문서**:

- `reports/pair_validation_split_report.md`
- `reports/pair_validation_interpretation.md`
- `reports/full_pair_major_feasibility_note.md`
- `reports/lammps_2gpu_runbook.md`
- `reports/nsight_status.md`
- `reports/lmax_spherical_harmonics_strength_note.md`
- `reports/pair_profiler_representative_report.md`

논문에 직접 넣을 표/그림 자산:

- `tables/`
- `figures/`
- `summary*.md`

## Canonical / Diagnostic / Planning 구분

이 폴더의 문서는 용도가 서로 다르다. ChatGPT나 다른 상위 모델이 분석할 때 이 구분을 먼저 이해해야 한다.

- `Canonical`
  - 현재 논문에서 가장 우선적으로 믿고 인용해야 하는 결과
  - main baseline-vs-proposal end-to-end, accuracy preservation, `lmax` baseline sweep 등
- `Diagnostic`
  - 원인 분석용 실험
  - intrusive profile, split validation, pair metadata microbenchmark, profiler/Nsight 메모
  - headline 성능 숫자로 직접 쓰면 안 되지만, “왜 이런 결과가 나왔는지” 설명하는 데 중요
- `Planning`
  - 후속 구현과 실험을 어떻게 할지 정리한 메모
  - 아직 실험으로 입증되지 않은 아이디어도 포함됨

대략적으로 보면:

- `manuscript/`, `tables/`, `figures/`, `summary_pair_end_to_end.md`, `summary_pair_accuracy.md`, `lmax_baseline_sweep/`는 `Canonical`
- `pair_validation_split/`, `pair_profiler/`, `four_case/`, `nsight_status.md`는 `Diagnostic`
- `full_pair_major_feasibility_note.md`, `lammps_2gpu_runbook.md`, `chatgpt_pro_prompt_context_full.md`는 `Planning + synthesis`

## 디렉터리 구조

- `manuscript/`
  - KCC 제출용 한국어 원고
  - 저자본 / 무기명본
- `figures/`
  - 논문 삽입용 PNG, SVG
  - `figure_manifest.csv`
- `tables/`
  - 논문 삽입용 CSV, Markdown 표
- `metrics/global/`
  - 전체 31개 데이터셋 통합 메트릭
- `metrics/pair_end_to_end/`
  - SevenNet baseline vs proposal-only non-intrusive repeated timing
- `metrics/pair_accuracy/`
  - SevenNet baseline 기준 출력 대비 반복 정확도 측정
- `metrics/pair_validation_split/`
  - `baseline / geometry_only / full_legacy / full_no_expand`
  - `forward_energy`와 `step_force`를 분리한 100회 반복 검증
- `metrics/pair_profiler/`
  - small/large representative `torch.profiler` 결과
- `metrics/four_case/`
  - `옵션 없음 / FlashTP / 제안기법만 / 둘 다` intrusive stage profiling
- `metrics/per_dataset/<dataset>/`
  - 데이터셋별 raw repeats, summary, stage breakdown
- `scripts/`
  - KCC 전용 실험 러너, Nsight 수집 스크립트, figure/table builder
- `nsight_status.md`
  - representative Nsight를 canonical headline 결과에서 제외한 이유와 현재 상태

## 주요 문서 설명

### `manuscript/`

- `manuscript/01_kcc_pair_geometry_reuse_ko.md`
  - 현재 저자 정보가 포함된 KCC용 한국어 본문
  - baseline vs proposal을 메인 축으로 설명
  - 결과, 제한점, 후속 방향까지 포함
- `manuscript/02_kcc_pair_geometry_reuse_ko_anonymous.md`
  - 무기명 심사용 버전
  - 본문 내용은 거의 같고 저자 정보만 제거

### `reports/`

- `reports/chatgpt_pro_prompt_context_full.md`
  - KCC 폴더 전체를 설명하는 가장 큰 컨텍스트 문서
  - 지금까지의 구현, 코드 경로, 실험 결과, 해석, 열린 가설, 다음 실험 제안을 모두 담음
  - 상위 모델에게 넘길 1차 입력으로 가장 적합
- `reports/pair_end_to_end_condition_analysis.md`
  - 메인 결과 해석 문서
  - 어떤 데이터셋 조건에서 proposal이 유리한지 정리
  - 큰 그래프 / 높은 연결 수 조건이 핵심인지 설명
- `reports/pair_accuracy_repeat_report.md`
  - 정확도 보존 검증 문서
  - baseline과 proposal이 기준 출력에 비해 어느 정도 차이가 나는지 반복 측정으로 정리
- `reports/pair_validation_split_report.md`
  - `baseline / geometry_only / full_legacy / full_no_expand`
  - `forward_energy / step_force`
  - 각 실험이 무엇을 의미하는지 설명하는 요약 문서
- `reports/pair_validation_interpretation.md`
  - 위 split validation 결과가 의미하는 구조적 해석
  - full path가 왜 느린지, backward가 왜 큰지, 다음 수정 우선순위가 무엇인지 정리
- `reports/full_pair_major_feasibility_note.md`
  - current full path를 더 pair-major하게 바꾸는 것이 얼마나 쉬운지/어려운지, 어디까지가 작은 패치이고 어디부터가 큰 재설계인지 정리
- `reports/lammps_2gpu_runbook.md`
  - LAMMPS `e3gnn/parallel` 2-GPU 실험 절차 정리
  - 현재 canonical 결과는 single GPU이므로, 이 문서는 실행 계획서에 가깝다
- `reports/lmax_spherical_harmonics_strength_note.md`
  - `lmax`와 geometry-side 비용의 관계를 설명하는 이론+구조 메모
  - “높은 `lmax`일수록 geometry-side 절감 가치가 커질 수 있다”는 방향을 정리
- `reports/pair_profiler_representative_report.md`
  - representative `torch.profiler` 결과 해석
  - forward와 force-including 경로에서 어떤 연산이 상위에 오는지 설명
- `reports/pre_fused_singlepass_patch_report.md`
  - current full path를 덜 쪼개지게 바꾼 pre-fused 패치 실험 결과
  - 작은 구조 패치로 얼마나 나아지는지 보는 보조 문서

### `summary*.md`

- `summary.md`
  - KCC 패키지의 전반적 요약
- `summary_pair_end_to_end.md`
  - main baseline-vs-proposal 실험의 핵심 결과만 짧게 요약
- `summary_pair_accuracy.md`
  - 정확도 반복 실험 요약
- `summary_pair_validation_split.md`
  - split validation 요약
- `summary_four_case.md`
  - `옵션 없음 / FlashTP / 제안기법 / 둘 다` intrusive 4-case 프로파일링 요약

### `lmax_baseline_sweep/`

- 목적:
  - `lmax`가 데이터셋이 정해주는 값인지, 높을수록 정확도가 항상 좋아지는지, 비용은 얼마나 커지는지를 직접 baseline-only로 검증
- 핵심 문서:
  - `lmax_baseline_sweep/reports/lmax_baseline_sweep_report.md`
- 핵심 메트릭:
  - `lmax_sweep_summary.csv`
  - `lmax_training_history.csv`
  - `lmax_latency_summary.csv`
  - `lmax_stage_profile_summary.csv`
  - `lmax_inference_errors.csv`
- 핵심 그림:
  - `lmax_accuracy_rmse.png`
  - `lmax_step_latency.png`
  - `lmax_accuracy_latency_frontier.png`
  - `lmax_trainable_params.png`
  - `lmax_training_wall_time.png`
  - `lmax_stage_profile.png`
- 해석 용도:
  - `lmax`가 무한정 좋지 않음을 직접 보이는 canonical 결과

### `lmax_quadrant_runtime/`

- 목적:
  - `lmax`를 올릴 때 비용이 어떤 데이터셋 구조에서 더 급격히 커지는지 보기 위한 보조 실험
  - `small_sparse`, `small_dense`, `large_sparse`, `large_dense` 대표 데이터셋을 사용
- 핵심 문서:
  - `lmax_quadrant_runtime/reports/lmax_quadrant_runtime_report.md`
- 핵심 메트릭:
  - `quadrant_lmax_latency_summary.csv`
  - `quadrant_lmax_stage_summary.csv`
  - `quadrant_lmax_growth_summary.csv`
- 핵심 그림:
  - `quadrant_lmax_latency.png`
  - `quadrant_lmax_latency_growth.png`
  - `quadrant_lmax_stage_shares.png`
  - `azobenzene_accuracy_vs_large_dense_cost.png`
- 해석 용도:
  - `lmax` 비용이 large/dense graph에서 왜 더 빨리 커지는지 설명하는 보조 결과
  - 정확도 실험은 아니고 구조적 시간 분석이다

### `metrics/global/`

- 31개 benchmarkable dataset 전체에 대한 통합 메트릭 저장
- 중요한 파일:
  - `dataset_manifest.csv`
    - 각 dataset representative sample의 크기, edge 수, 평균 이웃 수 등
  - `reference_e3nn.csv`
    - 기준 baseline 설정 관련 메타 정보
  - `flash_end_to_end_summary.csv`, `flash_comparison.csv`
    - FlashTP 관련 end-to-end 결과
  - `detailed_summary_mean_std.csv`, `detailed_stage_mean_std.csv`, `detailed_stage_long_mean_std.csv`
    - intrusive detailed profile 결과
  - `comparability_legend.csv`
    - 어떤 수치끼리 직접 비교 가능한지 정리

### `metrics/pair_end_to_end/`

- 목적:
  - 논문 메인 비교 실험
  - `SevenNet baseline` vs `SevenNet + proposal`
- 핵심 파일:
  - `pair_end_to_end/global/pair_end_to_end_comparison.csv`
  - `pair_end_to_end/global/pair_end_to_end_raw_repeats.csv`
- 해석 용도:
  - 실제 headline latency claim의 기준

### `metrics/pair_accuracy/`

- 목적:
  - proposal이 정확도를 바꾸지 않는다는 점을 반복 측정으로 확인
- 핵심 파일:
  - `pair_accuracy/global/pair_accuracy_summary.csv`
  - `pair_accuracy/global/pair_accuracy_raw_repeats.csv`

### `metrics/pair_validation_split/`

- 목적:
  - “왜 지금 결과가 이렇게 나오는지” 원인 분리
- 구성:
  - `baseline`, `geometry_only`, `full_legacy`, `full_no_expand`
  - `forward_energy`, `step_force`
- 핵심 파일:
  - `pair_validation_summary.csv`
  - `pair_validation_raw.csv`
  - `pair_metadata_summary.csv`
- 해석 용도:
  - current full path가 왜 느린지
  - geometry-only가 순수 forward에서는 거의 본전인지
  - force/backward와 pair metadata가 실제로 얼마나 큰지

### `metrics/four_case/`

- 목적:
  - `옵션 없음 / FlashTP / 제안기법만 / 둘 다`의 intrusive stage profiling 비교
- 해석 용도:
  - FlashTP와 proposal이 각각 어떤 stage를 건드리는지 시각적으로 비교
- 주의:
  - headline latency용이 아니라 stage decomposition용이다

### `metrics/pair_profiler/`

- 목적:
  - representative `torch.profiler` 결과 저장
- 대표 dataset:
  - `qm9_hf`, `mptrj`
- 해석 용도:
  - forward-only와 force-including 경로에서 상위 연산이 무엇인지 확인

### `figures/`

- 논문 본문과 보고서에 들어갈 그림을 모은 폴더
- 하위 폴더별 의미:
  - `pair_end_to_end/`: 메인 성능 결과
  - `pair_accuracy/`: 정확도 보존
  - `pair_validation_split/`: 원인 분리
  - `four_case/`: FlashTP/제안기법 4-case intrusive profile
  - `lmax_strength/`: SH/geometry-side microbenchmark
- `figure_manifest.csv`
  - 각 그림이 어디서 생성되었고 무슨 의미인지 추적하는 인덱스

### `tables/`

- 논문 삽입용 표의 Markdown/CSV 버전
- 메인 paper용 표와 보조 부록용 표가 함께 들어 있음

### `scripts/`

- 이 폴더의 스크립트는 실험 runner와 그림/표 생성기로 나뉜다.

실험 runner:
- `kcc_pair_end_to_end.py`
- `kcc_pair_accuracy_repeats.py`
- `kcc_pair_validation_suite.py`
- `kcc_four_case_profile.py`
- `kcc_lmax_strength_analysis.py`
- `kcc_lmax_baseline_sweep.py`
- `kcc_lmax_quadrant_runtime.py`
- `kcc_pair_profiler_representatives.py`
- `kcc_nsys_collect.py`
- `kcc_nsys_target.py`

그림/표 생성기:
- `kcc_build_assets.py`
- `kcc_build_pair_end_to_end_assets.py`
- `kcc_build_accuracy_assets.py`
- `kcc_build_pair_validation_assets.py`
- `kcc_build_four_case_assets.py`

공통 유틸:
- `kcc_common.py`

## ChatGPT나 상위 모델이 이 폴더를 분석할 때 주의할 점

1. `manuscript/`만 읽으면 현재 해석의 근거가 부족하다. 반드시 `reports/`와 `metrics/`를 같이 봐야 한다.
2. `figures/`만 보면 왜 그런 그림이 나왔는지 알 수 없다. 대응되는 `reports/`와 `scripts/`를 확인해야 한다.
3. `intrusive detailed profile`과 `end-to-end latency`는 같은 축에서 비교하면 안 된다.
4. `FlashTP` 관련 문서는 보조/후속 방향이지, 현재 KCC 본선 비교의 중심은 아니다.
5. `lmax` 관련 문서는
   - baseline-only accuracy/latency sweep
   - representative graph runtime scaling
   두 층으로 나뉜다.
6. 현재 canonical 결과는 single GPU 기준이다. 2-GPU LAMMPS 실험은 runbook 수준이다.

## ChatGPT용 추천 읽기 순서

1. `reports/chatgpt_pro_prompt_context_full.md`
2. `manuscript/01_kcc_pair_geometry_reuse_ko.md`
3. `reports/pair_end_to_end_condition_analysis.md`
4. `reports/pair_validation_interpretation.md`
5. `reports/lmax_baseline_sweep_report.md`
6. `reports/lmax_quadrant_runtime_report.md`
7. 필요 시 `metrics/...`와 `figures/...` 원본 확인

## Canonical Results

주 결과 파일:

- `metrics/pair_end_to_end/global/pair_end_to_end_summary.csv`
- `metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv`
- `metrics/pair_accuracy/global/pair_accuracy_summary.csv`
- `metrics/pair_accuracy/global/pair_accuracy_raw_repeats.csv`
- `reports/pair_end_to_end_condition_analysis.md`
- `reports/pair_accuracy_repeat_report.md`
- `reports/pair_validation_split_report.md`
- `reports/pair_validation_interpretation.md`
- `reports/lmax_spherical_harmonics_strength_note.md`
- `reports/lammps_2gpu_runbook.md`
- `reports/full_pair_major_feasibility_note.md`
- `metrics/global/dataset_manifest.csv`
- `metrics/global/reference_e3nn.csv`
- `metrics/global/flash_end_to_end_summary.csv`
- `metrics/global/flash_comparison.csv`
- `metrics/global/detailed_summary_mean_std.csv`
- `metrics/global/detailed_stage_mean_std.csv`
- `metrics/global/detailed_stage_long_mean_std.csv`
- `metrics/global/comparability_legend.csv`
- `metrics/four_case/global/four_case_detailed_summary_mean_std.csv`
- `metrics/four_case/global/four_case_detailed_stage_mean_std.csv`
- `metrics/four_case/global/four_case_detailed_stage_long_mean_std.csv`

데이터셋별 산출물:

- `metrics/per_dataset/<dataset>/manifest.json`
- `metrics/per_dataset/<dataset>/flash_end_to_end_summary.csv`
- `metrics/per_dataset/<dataset>/detailed_stage_mean_std.csv`
- `metrics/per_dataset/<dataset>/detailed_stage_long_mean_std.csv`
- `metrics/per_dataset/<dataset>/flash_comparison.csv`

## Regeneration

전수 canonical 런:

```bash
python -u docs/papers/KCC/scripts/kcc_profile_matrix.py --output-root docs/papers/KCC
```

대표 Nsight 수집:

```bash
python docs/papers/KCC/scripts/kcc_nsys_collect.py --output-root docs/papers/KCC
```

대표 Nsight는 스크립트와 수집 경로를 패키지에 포함하지만, 현재 canonical 본문 수치와 그림은 primary family 세 가지에만 의존한다. 이유는 `nsight_status.md`에 정리한다.

표/그림 생성:

```bash
python docs/papers/KCC/scripts/kcc_build_assets.py --output-root docs/papers/KCC
```

4-case intrusive profiling:

```bash
python -u docs/papers/KCC/scripts/kcc_four_case_profile.py --output-root docs/papers/KCC --repeat 5
python docs/papers/KCC/scripts/kcc_build_four_case_assets.py --output-root docs/papers/KCC
```

SevenNet baseline vs proposal-only end-to-end:

```bash
python -u docs/papers/KCC/scripts/kcc_pair_end_to_end.py --output-root docs/papers/KCC --warmup 3 --repeat 10
python docs/papers/KCC/scripts/kcc_build_pair_end_to_end_assets.py --output-root docs/papers/KCC
```

정확도 반복 검증:

```bash
python -u docs/papers/KCC/scripts/kcc_pair_accuracy_repeats.py --warmup 2 --repeat 10
python docs/papers/KCC/scripts/kcc_build_accuracy_assets.py
```

분리 검증:

```bash
python -u docs/papers/KCC/scripts/kcc_pair_validation_suite.py --output-root docs/papers/KCC --warmup 5 --repeat 30
python docs/papers/KCC/scripts/kcc_build_pair_validation_assets.py --output-root docs/papers/KCC
python docs/papers/KCC/scripts/kcc_pair_profiler_representatives.py --output-root docs/papers/KCC
```

`lmax`와 SH 재사용 강점 메모:

```bash
python -u docs/papers/KCC/scripts/kcc_lmax_strength_analysis.py
```

## Measurement Families

- `e3nn baseline detailed`
  - intrusive, stage decomposition
- `e3nn pair detailed`
  - intrusive, pair overhead / reusable-term 해석
- `FlashTP end-to-end`
  - non-intrusive, headline latency
- `SevenNet baseline vs proposal-only end-to-end`
  - non-intrusive, main paper comparison
- `Representative Nsight`
  - kernel mix validation only
- `Four-case intrusive stage profiling`
  - `e3nn_baseline`, `flash_baseline`, `e3nn_pair_full`, `flash_pair_auto`
  - step-by-step stage timing for each representative dataset
- `Pair validation split`
  - `baseline`, `geometry_only`, `full_legacy`, `full_no_expand`
  - `forward_energy`: 입력 그래프를 미리 만든 뒤 순수 에너지 forward만 측정
  - `step_force`: 그래프 생성, pair metadata, device 이동, 힘 계산을 포함한 실제 추론 경로
- `Representative torch.profiler`
  - `qm9_hf`, `mptrj`
  - `force_model`, `forward_energy`

직접 비교 규칙은 `metrics/global/comparability_legend.csv`와 `tables/table_01_comparability.md`를 기준으로 한다.

메인 논문용 표:

- `tables/table_02_pair_end_to_end_summary.md`
- `tables/table_03_pair_condition_summary.md`
- `tables/table_04_pair_accuracy_summary.md`
- `tables/table_04a_pair_accuracy_compact.md`
- `tables/table_05_representative_stage_summary.md`

보조 표:

- `tables/table_s01_flash_summary.md`
- `tables/table_s02_nsys_kernel_groups.md`
- `tables/table_06_force_step_split_summary.md`
- `tables/table_07_forward_only_split_summary.md`
- `tables/table_08_pair_metadata_method_summary.md`
- `tables/table_09_split_compact_summary.md`

4-case profiling 산출물:

- `summary_four_case.md`
- `metrics/four_case/global/four_case_detailed_stage_mean_std.csv`
- `figures/four_case/stage_breakdown_all_option_none.png`
- `figures/four_case/stage_breakdown_all_flashtp_only.png`
- `figures/four_case/stage_breakdown_all_proposal_only.png`
- `figures/four_case/stage_breakdown_all_flashtp_plus_proposal.png`
- `figures/four_case/per_dataset/<dataset>_four_case_stage_breakdown.png`

baseline-vs-proposal end-to-end 산출물:

- `summary_pair_end_to_end.md`
- `metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv`
- `tables/table_02_pair_end_to_end_summary.md`
- `tables/table_03_pair_condition_summary.md`
- `tables/table_04_pair_accuracy_summary.md`
- `figures/pair_end_to_end/pair_latency_all.png`
- `figures/pair_end_to_end/pair_speedup_all.png`
- `figures/pair_end_to_end/pair_speedup_vs_num_edges.png`
- `figures/pair_end_to_end/pair_speedup_vs_avg_neighbors.png`
- `figures/pair_end_to_end/pair_speedup_by_bucket.png`
- `figures/pair_end_to_end/pair_speedup_size_density_map.png`
- `figures/pair_accuracy/pair_accuracy_energy_errorbar.png`
- `figures/pair_accuracy/pair_accuracy_force_errorbar.png`
- `reports/pair_end_to_end_condition_analysis.md`
- `reports/pair_accuracy_repeat_report.md`

분리 검증 산출물:

- `summary_pair_validation_split.md`
- `metrics/pair_validation_split/global/pair_validation_summary.csv`
- `metrics/pair_validation_split/global/pair_validation_raw.csv`
- `metrics/pair_validation_split/global/pair_metadata_summary.csv`
- `metrics/pair_profiler/pair_profiler_summary.csv`
- `tables/table_06_force_step_split_summary.md`
- `tables/table_07_forward_only_split_summary.md`
- `tables/table_08_pair_metadata_method_summary.md`
- `tables/table_09_split_compact_summary.md`
- `figures/pair_validation_split/pair_validation_step_force_latency_all.png`
- `figures/pair_validation_split/pair_validation_forward_energy_latency_all.png`
- `figures/pair_validation_split/pair_validation_pair_metadata_methods.png`
- `figures/pair_validation_split/pair_validation_case_median_speedups.png`
- `reports/pair_validation_split_report.md`
- `reports/pair_validation_interpretation.md`

## Writing Rule

원고에서는 다음 표현을 고정한다.

- 현재 구현은 `pair-aware geometry-side reuse runtime`
- 재사용 항은 `distance / radial basis / cutoff / spherical harmonics / pair-level weight_nn input`
- 미구현 항은 `pair-major tensor product / pair-major fused reduction / pair-major force backward`
- FlashTP와의 관계는 `중복 없는 상보적 최적화`

강한 claim은 피한다.  
성능 결과는 “현재 구현의 범위 안에서 어떤 비용이 줄고 어떤 비용이 남는가”를 설명하는 근거로만 사용한다.
