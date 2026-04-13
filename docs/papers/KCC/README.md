# KCC Pair-Aware Geometry Reuse Package

이 디렉터리는 KCC 제출용 원고, 표, 그림, 메트릭, 생성 스크립트를 한곳에 모은 canonical 작업 공간이다.  
핵심 원칙은 다음 세 가지다.

1. 현재 구현은 `pair-major TP`가 아니라 `pair-aware geometry-side exact reuse`로 기술한다.
2. `intrusive detailed profile`과 `non-intrusive end-to-end latency`는 같은 축에서 직접 비교하지 않는다.
3. 실험 결과는 논리를 보조하는 근거로 사용하고, 본문 주장은 구현 범위와 상보성 설명에 맞춘다.

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
- `metrics/four_case/`
  - `옵션 없음 / FlashTP / 제안기법만 / 둘 다` intrusive stage profiling
- `metrics/per_dataset/<dataset>/`
  - 데이터셋별 raw repeats, summary, stage breakdown
- `scripts/`
  - KCC 전용 실험 러너, Nsight 수집 스크립트, figure/table builder
- `nsight_status.md`
  - representative Nsight를 canonical headline 결과에서 제외한 이유와 현재 상태

## Canonical Results

주 결과 파일:

- `metrics/pair_end_to_end/global/pair_end_to_end_summary.csv`
- `metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv`
- `reports/pair_end_to_end_condition_analysis.md`
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

직접 비교 규칙은 `metrics/global/comparability_legend.csv`와 `tables/table_01_comparability.md`를 기준으로 한다.

메인 논문용 표:

- `tables/table_02_pair_end_to_end_summary.md`
- `tables/table_03_pair_condition_summary.md`
- `tables/table_04_pair_accuracy_summary.md`
- `tables/table_05_representative_stage_summary.md`

보조 표:

- `tables/table_s01_flash_summary.md`
- `tables/table_s02_nsys_kernel_groups.md`

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
- `reports/pair_end_to_end_condition_analysis.md`

## Writing Rule

원고에서는 다음 표현을 고정한다.

- 현재 구현은 `pair-aware geometry-side reuse runtime`
- 재사용 항은 `distance / radial basis / cutoff / spherical harmonics / pair-level weight_nn input`
- 미구현 항은 `pair-major tensor product / pair-major fused reduction / pair-major force backward`
- FlashTP와의 관계는 `중복 없는 상보적 최적화`

강한 claim은 피한다.  
성능 결과는 “현재 구현의 범위 안에서 어떤 비용이 줄고 어떤 비용이 남는가”를 설명하는 근거로만 사용한다.
