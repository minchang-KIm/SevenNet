# KCC Tables and Figures

## Main Tables

- `tables/table_01_comparability.md`
- `tables/table_02_pair_end_to_end_summary.md`
- `tables/table_03_pair_condition_summary.md`
- `tables/table_04_pair_accuracy_summary.md`
- `tables/table_04a_pair_accuracy_compact.md`
- `tables/table_05_representative_stage_summary.md`

## Supplementary Tables

- `tables/table_s01_flash_summary.md`
- `tables/table_s02_nsys_kernel_groups.md`
- `tables/table_06_force_step_split_summary.md`
- `tables/table_07_forward_only_split_summary.md`
- `tables/table_08_pair_metadata_method_summary.md`
- `tables/table_09_split_compact_summary.md`

## Main Figures

- `figure_00_comparability_diagram`: 계측 family 간 직접 비교 가능 범위 설명
- `pair_latency_all`: 31개 데이터셋 SevenNet baseline vs proposal-only latency mean±std
- `pair_speedup_all`: baseline 대비 proposal-only speedup
- `pair_accuracy_energy_errorbar`: baseline 기준 절대 에너지 차이 mean±std
- `pair_accuracy_force_errorbar`: baseline 기준 최대 절대 힘 차이 mean±std
- `pair_speedup_vs_num_edges`: 그래프 크기와 speedup 관계
- `pair_speedup_vs_avg_neighbors`: 평균 이웃 수와 speedup 관계
- `pair_speedup_by_bucket`: size-density bucket별 speedup
- `pair_speedup_size_density_map`: speedup을 색으로 표현한 size-density map
- `figure_04_representative_stage_breakdown`: representative detailed stage stacked bar

## Supplementary Figures

- `figure_01_flash_latency_all`: FlashTP end-to-end latency mean±std
- `figure_02_flash_speedup_all`: Flash baseline 대비 flash pair auto speedup
- `figure_03_dataset_map`: 데이터셋 size-density map
- `figure_05_sh_share_vs_intrusive_pair_speedup`: SH share와 intrusive pair speedup 관계
- `pair_validation_step_force_latency_all`: `baseline / geometry_only / full_legacy / full_no_expand`의 force 포함 추론 시간 mean±std
- `pair_validation_forward_energy_latency_all`: 같은 네 case의 순수 forward 에너지 시간 mean±std
- `pair_validation_step_force_speedup_geometry_only`: force 포함 경로에서 `baseline / geometry_only` speedup
- `pair_validation_forward_energy_speedup_geometry_only`: pure forward에서 `baseline / geometry_only` speedup
- `pair_validation_step_force_patch_delta`: `full_legacy / full_no_expand` 비교
- `pair_validation_forward_energy_patch_delta`: pure forward에서의 patch 효과 비교
- `pair_validation_pair_metadata_methods`: 기존 CPU pair metadata와 벡터화/GPU 실험 비교
- `pair_validation_case_median_speedups`: case별 중앙값 speedup 요약
- `figures/four_case/stage_breakdown_all_option_none.png`
- `figures/four_case/stage_breakdown_all_flashtp_only.png`
- `figures/four_case/stage_breakdown_all_proposal_only.png`
- `figures/four_case/stage_breakdown_all_flashtp_plus_proposal.png`
- `figures/four_case/model_total_heatmap.png`
