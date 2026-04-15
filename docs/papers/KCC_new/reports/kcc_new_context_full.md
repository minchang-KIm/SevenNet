# KCC_new Full Context

## 1. 목적

`KCC_new`는 SevenNet의 `geometry_only` 실행 방식을 메인 비교축으로 다시 정리한 canonical 패키지다.  
이 패키지의 목적은 다음 세 가지다.

1. 같은 모델 출력 정확도를 유지한 채, 원자쌍 기반 geometry reuse가 실제 추론 시간에 어떤 영향을 주는지 repeat 30 기준으로 다시 측정한다.
2. 단순히 빠르다/느리다를 말하는 데서 끝나지 않고, 현재 구조에서 어디가 실제 병목인지 진단 실험으로 분리한다.
3. 논문을 “성능 승리 선언”이 아니라 “정확도 보존형 exact reuse 구현 + 병목 규명 + 다음 구현 우선순위 제시”로 완성한다.

## 2. 현재 구현 정의

현재 논문에서 다루는 구현은 `pair-major full runtime`이 아니라 `pair-aware geometry reuse`다.

- `baseline`
  - directed edge마다 거리, radial basis, cutoff, spherical harmonics, weight input을 각각 계산
- `geometry_only`
  - reverse edge pair를 기준으로 재사용 가능한 geometry-side 값만 한 번 계산
  - 이후에는 다시 edge-major convolution과 generic force path를 사용

즉 현재 구현은 다음까지는 하지 않는다.

- pair-major tensor product
- pair-major aggregation
- pair-aware backward
- upstream neighbor builder와 완전 통합된 pair-major end-to-end runtime

## 3. 실험 환경

- device: `NVIDIA GeForce RTX 4090`
- framework: `PyTorch 2.7.1+cu126`
- main headline comparison:
  - warmup `3`
  - repeat `30`
- accuracy repeat:
  - warmup `2`
  - repeat `30`
- diagnostic:
  - geometry-only intrusive breakdown: repeat `30`
  - LAMMPS serial pair-metadata bench: repeat `30`

환경 요약 표:
- [table_01_experiment_setup.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_01_experiment_setup.md)

## 4. 데이터셋 정책

- benchmarkable public-local dataset 전체 `31개`
- 각 데이터셋에서 representative sample `1개`
- 크기와 밀도를 나타내는 메타데이터 저장:
  - `natoms`
  - `num_edges`
  - `avg_neighbors_directed`
  - `density_bucket`

대표적인 분포:

- `small_sparse`: 11개
- `large_sparse`: 3개
- `large_dense`: 17개

dataset manifest:
- [dataset_manifest.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_end_to_end/global/dataset_manifest.csv)

## 5. 메인 실험

메인 비교:

- `SevenNet baseline`
- `SevenNet + geometry_only`

측정값:

- steady-state latency mean / std / median / p95
- speedup = `baseline / geometry_only`
- absolute energy difference vs baseline
- maximum absolute force difference vs baseline

메인 raw / summary:

- [pair_end_to_end_raw_repeats.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_end_to_end/global/pair_end_to_end_raw_repeats.csv)
- [pair_end_to_end_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_end_to_end/global/pair_end_to_end_summary.csv)
- [pair_end_to_end_comparison.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv)

## 6. 메인 결과

### 6.1 headline latency

- datasets: `31`
- median speedup: `0.987725x`
- geometric mean speedup: `0.986435x`
- wins: `0`
- losses: `31`

즉 현재 generic calculator path 기준으로는 `geometry_only`가 아직 baseline을 이기지 못한다.

### 6.2 조건별 해석

- `num_edges >= 3000`
  - count `20`
  - win rate `0.000`
  - median `0.988`
- `num_edges < 3000`
  - count `11`
  - win rate `0.000`
  - median `0.983`
- `avg_neighbors_directed >= 40`
  - count `17`
  - win rate `0.000`
  - median `0.988`
- `avg_neighbors_directed < 40`
  - count `14`
  - win rate `0.000`
  - median `0.985`

즉 과거 `full` 경로에서 보였던 “large/high-neighbor일수록 유리” 패턴이 `geometry_only` 메인 비교에서는 아직 end-to-end win으로 나타나지 않는다.  
다만 small_sparse 쪽이 더 크게 손해이고 large/dense 쪽이 거의 본전에 가깝다는 gradient는 남아 있다.

bucket summary:

- `large_dense`: mean `0.987865`, median `0.987846`
- `large_sparse`: mean `0.987797`, median `0.988187`
- `small_sparse`: mean `0.983874`, median `0.983418`

### 6.3 대표 예시

가장 덜 느린 쪽:

- `phonondb_pbe`: `0.991x`
- `md22_dha`: `0.991x`
- `md22_buckyball_catcher`: `0.990x`

가장 느린 쪽:

- `iso17`: `0.973x`
- `wbm_initial`: `0.979x`
- `md22_ac_ala3_nhme`: `0.981x`

그림:

- [figure_01_end_to_end_latency.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_01_end_to_end_latency.png)
- [figure_02_end_to_end_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_02_end_to_end_speedup.png)
- [figure_03_speedup_vs_edges.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_03_speedup_vs_edges.png)
- [figure_04_speedup_vs_neighbors.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_04_speedup_vs_neighbors.png)
- [figure_05_bucket_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_05_bucket_speedup.png)

표:

- [table_02_end_to_end_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_02_end_to_end_summary.md)
- [table_03_condition_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_03_condition_summary.md)

## 7. 정확도 보존

정확도 repeat 30 결과는 다음과 같다.

- baseline median force diff mean: `1.189e-06 eV/A`
- geometry_only median force diff mean: `1.809e-06 eV/A`
- baseline median energy diff mean: `0 eV`
- geometry_only median energy diff mean: `0 eV`
- baseline worst energy diff max: `1.831e-04 eV`
- geometry_only worst energy diff max: `2.441e-04 eV`
- baseline worst force diff max: `3.662e-04 eV/A`
- geometry_only worst force diff max: `4.883e-04 eV/A`

즉 현재 `geometry_only`는 출력을 바꾸기보다 실행 순서를 바꾸는 최적화로 해석하는 것이 맞다.  
difference는 baseline 반복 잡음과 같은 부동소수점 수준에 머무른다.

파일:

- [pair_accuracy_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_accuracy/global/pair_accuracy_summary.csv)
- [pair_accuracy_raw_repeats.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/metrics/pair_accuracy/global/pair_accuracy_raw_repeats.csv)
- [figure_06_accuracy_energy.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_06_accuracy_energy.png)
- [figure_07_accuracy_force.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_07_accuracy_force.png)
- [table_04_accuracy_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_04_accuracy_summary.md)

## 8. geometry_only 내부 진단

intrusive forward-only 진단은 headline latency가 아니라 남은 구조 손해를 보기 위한 stage comparison이다.

대표값:

- `bulk_large`
  - baseline forward total: `2.997 ms`
  - geometry_only forward total: `3.107 ms`
  - baseline / geometry_only: `0.965x`
- `bulk_small`
  - baseline forward total: `2.940 ms`
  - geometry_only forward total: `3.047 ms`
  - baseline / geometry_only: `0.965x`
- `dimer_small`
  - baseline forward total: `2.830 ms`
  - geometry_only forward total: `2.918 ms`
  - baseline / geometry_only: `0.970x`

남아 있는 대표 비용:

- pair expand: 약 `0.037 ~ 0.039 ms`
- pair weight expand: 약 `0.032 ~ 0.036 ms`
- pair geometry: 약 `0.173 ~ 0.194 ms`

이 결과는 다음을 뜻한다.

1. geometry_only는 representative forward diagnostic에서도 아직 baseline을 넘지 못한다.
2. 손해 폭은 약 `3.0 ~ 3.5%` 수준이며, pair->edge expansion과 pair weight expansion이 직접 관측된다.
3. end-to-end generic calculator path에서 손해를 만드는 큰 축은 geometry 계산 그 자체보다 pair->edge 확장 구조와 generic force path다.

파일:

- [geometry_only_breakdown_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/geometry_only_breakdown/reports/geometry_only_breakdown_report.md)
- [table_05_geometry_breakdown_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_05_geometry_breakdown_summary.md)

## 9. LAMMPS upstream pair-metadata fast path

이 진단은 “upstream neighbor 정보를 직접 넘기게 하면 무엇이 좋아지는가”를 가장 잘 보여준다.

### 9.1 구현

LAMMPS pair style 안에서:

- neighbor loop 안에서 바로 pair id 생성
- second-pass string/hash reverse lookup 제거
- `edge_pair_map`, `pair_forward_index`, `pair_backward_index`를 즉시 채움

### 9.2 결과

`bulk_large`

- legacy pair_metadata: `4.636 ± 0.054 ms`
- upstream pair_metadata: `0.322 ± 0.029 ms`
- reduction: `14.40x`
- geometry_only total: `175.248 ± 5.417 ms -> 173.153 ± 9.628 ms`
- total reduction: `1.012x`

`bulk_small`

- legacy pair_metadata: `0.441 ± 0.011 ms`
- upstream pair_metadata: `0.100 ± 0.005 ms`
- reduction: `4.41x`
- geometry_only total: `167.673 ± 6.979 ms -> 169.793 ± 10.672 ms`
- total reduction: `0.988x`

이 결과는 두 가지를 말해준다.

1. pair metadata는 실제로 줄일 가치가 큰 구조 병목이었다.
2. upstream에서 pair 정보를 직접 쓰게 만들면 pair metadata 병목은 꽤 확실하게 줄어든다.
3. total compute는 측정 분산과 다른 병목의 영향이 남으므로, pair metadata 감소가 항상 같은 비율의 end-to-end 감소로 이어지지는 않는다.

즉 “neighbor list를 이미 아는 곳에서 pair를 같이 넘기는 것”은 실험적으로 효과가 확인된 개선이다.

파일:

- [lammps_pair_metadata_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/lammps_pair_metadata_bench/reports/lammps_pair_metadata_report.md)
- [figure_08_lammps_pair_metadata.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_08_lammps_pair_metadata.png)
- [table_06_lammps_pair_metadata_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_06_lammps_pair_metadata_summary.md)

## 10. 현재까지 방어 가능한 결론

현재 결과로 방어 가능한 문장은 다음과 같다.

1. `geometry_only` exact reuse는 baseline과 거의 같은 출력 정확도를 유지한다.
2. generic SevenNet calculator path에서 `geometry_only`는 아직 end-to-end latency win을 만들지 못한다.
3. intrusive forward diagnostic과 LAMMPS upstream pair-metadata bench는, 병목이 exact reuse 수식 하나가 아니라 runtime 구조에 있음을 보여준다.
4. 특히 upstream pair metadata 제거는 쉽고 효과가 분명한 개선이다.
5. 남은 핵심 병목은 pair를 다시 edge로 펼치는 비용과 edge-major force path이다.

## 10.1 representative profiler가 보여주는 것

대표 profiler(`qm9_hf`, `mptrj`)는 geometry_only가 현재 dominant kernel class를 아직 바꾸지 못하고 있음을 보여준다.

- 상위 device op는 여전히 `aten::mul`, `aten::fill_`, `aten::copy_`, `aten::clone` 계열이다.
- 반면 `index_select` 계열 device time은 geometry_only에서 분명히 증가한다.
  - `mptrj forward_energy`: `1.600 us -> 2017.683 us`
  - `mptrj force_model`: `1.632 us -> 2027.508 us`
  - `qm9_hf forward_energy`: `1.920 us -> 42.402 us`
  - `qm9_hf force_model`: `1.921 us -> 43.233 us`

즉 geometry_only는 pair->edge 확장 구조를 실제로 추가하고 있으며, force path에서는 generic backward 성격의 큰 비용 구조가 여전히 남아 있다.

## 10.2 restored full two-pass가 보여주는 것

`7be5a2b`에서 `full` 경로를 single-pass `torch.cat` 구조로 바꾼 뒤 large/dense win이 사라졌다. 이를 확인하기 위해 `full` 경로를 pre-`7be5a2b`의 forward/reverse two-pass 구조로 되돌리고, 31개 데이터셋 전체를 repeat 30으로 다시 측정했다.

관련 파일:

- [full_restored_win_recheck.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/reports/full_restored_win_recheck.md)
- [pair_end_to_end_comparison.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/full_restored_pair_end_to_end_merged/metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv)
- [table_07_full_restored_end_to_end_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_07_full_restored_end_to_end_summary.md)
- [table_08_full_restored_condition_summary.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/tables/table_08_full_restored_condition_summary.md)
- [figure_09_full_restored_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_09_full_restored_speedup.png)
- [figure_10_full_restored_condition_speedup.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC_new/figures/figure_10_full_restored_condition_speedup.png)

headline:

- median speedup: `1.009916x`
- geometric mean speedup: `0.858294x`
- wins: `18`
- losses: `13`

조건별 결과:

- `num_edges >= 3000`: `20`개 중 `18`개 win, median `1.014x`
- `num_edges < 3000`: `11`개 중 `0`개 win, median `0.613x`
- `avg_neighbors_directed >= 40`: `17`개 중 `16`개 win, median `1.013x`
- `large_dense`: `17`개 중 `16`개 win, median `1.013x`
- `large_sparse`: `3`개 중 `2`개 win, median `1.053x`
- `small_sparse`: `11`개 중 `0`개 win, median `0.613x`

상위 win:

- `oc20_s2ef_val_ood_ads`: `1.073x`
- `md22_buckyball_catcher`: `1.071x`
- `oc20_s2ef_train_20m`: `1.053x`
- `salex_train_official`: `1.027x`
- `omol25_train_neutral`: `1.027x`

해석:

1. win은 `geometry_only`가 아니라 restored `full` two-pass에서 복구된다.
2. restored `full`은 pure geometry-only 효과가 아니라 pair-aware full execution/scheduling 효과를 포함한다.
3. `single-pass torch.cat`은 convolution/gather 호출 수를 줄였지만, 중간 텐서 materialization과 kernel shape 변화 때문에 large/dense win을 없앤 regression이다.
4. 논문에서는 `geometry_only`를 정확도 보존형 exact reuse 증거로 쓰고, restored `full` two-pass를 large/dense workload에서의 성능 증거로 분리해서 쓰는 것이 가장 방어 가능하다.

## 11. 지금 논문에서 피해야 할 주장

- “geometry_only가 이미 대부분의 데이터셋에서 빨라졌다”
- “순수 geometry_only만으로 큰 그래프에서 확실히 이긴다”
- “FlashTP와 결합하면 반드시 빨라진다”
- “lmax가 높을수록 현재 geometry_only가 바로 더 유리하다”

이건 현재 `KCC_new` main result로는 입증되지 않았다.

## 12. 지금 논문에서 적극적으로 쓸 수 있는 노블티

현재 시점에서 가장 타당한 novelty는 다음 네 축이다.

1. **exact reuse formulation**
   - reverse pair에서 재사용 가능한 geometry-side 값을 정확도 변화 없이 공유하는 실행 방식
2. **accuracy-preserving runtime implementation**
   - 실제 SevenNet runtime에 넣어 에너지/힘 정확도를 반복 실험으로 검증
3. **systematic bottleneck diagnosis**
   - main end-to-end, intrusive forward diagnostic, LAMMPS upstream fast path를 분리해서 어디가 실제 병목인지 규명
4. **conditioned performance evidence**
   - restored `full` two-pass에서 large/dense workload가 실제로 유리해지는 조건을 31개 데이터셋 repeat 30으로 확인
5. **actionable next-step evidence**
   - pair metadata upstream integration은 쉽고 효과가 분명함을 실험으로 확인

## 13. 다음 구현 우선순위

1. `geometry_only`에서 pair->edge expand 최소화
2. pair weight expand 최소화
3. force path에서 edge-major fan-in 구조 축소
4. 그 다음 pair-major message path

`TorchSim`, `parallel`, `FlashTP 결합`은 지금 논문의 canonical 범위 밖에 두는 것이 맞다.
