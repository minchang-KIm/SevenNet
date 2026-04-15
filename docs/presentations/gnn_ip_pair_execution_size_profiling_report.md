# Pair Execution Size Validation And Pipeline Profiling Report

## 2026-04-03 Note

- 이 문서는 stage decomposition과 size-density tendency를 설명하는 내부 보고서다.
- 현재 public-facing interpretation은 `docs/papers/icpp_pair_execution/03_final_manuscript.md`를 따른다.
- 여기 포함된 speedup 수치는 submission headline이 아니라 qualitative trend 설명용으로 사용하는 것이 맞다.

## 1. Goal

이번 보고서의 목적은 두 가지다.

1. 이미 다운로드해 둔 공개 데이터셋들 중 상대적으로 큰 셋을 여러 개 골라서, `SevenNet e3nn baseline` 대비 `e3nn + pair_execution(full)`이 실제로 "그래프가 클수록" 유리해지는지 검증한다.
2. steady-state 기준으로 파이프라인 단계별 시간과, 각 단계가 실제로 처리하는 load를 같이 측정해서 왜 어떤 데이터셋에서 이득이 나고 어떤 데이터셋에서 손해가 나는지 설명한다.

중요한 전제는 이번 실험이 `FlashTP`가 아니라 `SevenNet 기본 e3nn 경로`만 대상으로 한다는 점이다. 즉, 이번 보고서는 "현재 구현의 pair execution 자체"를 본다.

## 2. Experimental Setup

- Model: `7net-omni`
- Backend: `e3nn baseline` vs `e3nn + pair_execution(full)`
- Device: `NVIDIA GeForce RTX 4090`
- Steady-state benchmark: 각 샘플당 cold 1회 후 `repeat=3` median
- Large-sample selection: 각 로컬 데이터셋에서 atom 수 기준 top-3 샘플 선택
- Representative pipeline profiling: 각 데이터셋의 최대 샘플 1개에 대해 steady-state 파이프라인 단위로 재측정

사용한 데이터셋:

- `mptrj` 최대 `444` atoms
- `md22_double_walled_nanotube` 최대 `370` atoms
- `spice_2023` 최대 `89` atoms
- `md22_stachyose` 최대 `87` atoms
- `ani1x` 최대 `63` atoms
- `rmd17` 최대 `24` atoms
- `iso17` 최대 `19` atoms

원본 산출물:

- Summary: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/summary.md`
- Sample metrics: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/metrics/samples.csv`
- Internal load profile: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/metrics/profiles.csv`
- Steady pipeline profile: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/metrics/pipeline_profiles.csv`
- Derived summary: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/derived/dataset_summary.csv`
- Size-bin summary: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/derived/size_bin_summary.csv`
- Pipeline deltas: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/derived/pipeline_deltas.csv`
- Plot: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/plots/speedup_vs_natoms.png`
- Plot: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/plots/speedup_vs_num_edges.png`
- Plot: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/plots/size_bin_speedup.png`
- Plot: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/plots/pipeline_breakdown_steady.png`
- Plot: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/local_pair_size_main/plots/representative_speedup_by_dataset.png`

## 3. Headline Result

결론부터 말하면, "클수록 무조건 좋다"는 과장이고, "충분히 큰 그래프에서는 좋아질 가능성이 확실히 커진다"가 현재 데이터로 방어 가능한 표현이다.

샘플 단위 상관:

- Spearman(speedup, `natoms`) = `0.541`
- Spearman(speedup, `num_edges`) = `0.487`
- Pearson(speedup, `natoms`) = `0.866`
- Pearson(speedup, `num_edges`) = `0.882`

Representative max-sample 기준 상관:

- Spearman(speedup, `natoms`) = `0.679`
- Pearson(speedup, `natoms`) = `0.911`

즉, 현재 구현의 효과는 확실히 크기와 양의 상관을 보인다. 다만 이 상관은 "그래프가 조금만 커져도 항상 이득"이라는 뜻은 아니다.

## 4. Size-Bin Validation

edge 수 기준으로 묶으면 패턴이 더 명확하다.

| Edge bin | Sample count | Mean speedup (`baseline / pair_full`) | Interpretation |
| --- | ---: | ---: | --- |
| `<=1k` | 9 | `0.759x` | pair가 꾸준히 손해 |
| `1k-5k` | 6 | `0.755x` | 여전히 손해 |
| `5k-15k` | 0 | N/A | 이번 로컬 대형 샘플 셋에는 없음 |
| `>15k` | 6 | `1.079x` | 평균적으로 pair가 유리 |

이 결과는 이번 질문의 핵심 검증으로 충분하다.

- 작은/중간 그래프에서는 pair execution이 일관되게 손해였다.
- 아주 큰 그래프 묶음에서는 pair execution이 평균적으로 이득이었다.

즉, 현재 구현은 **small-graph optimization이 아니라 large-graph optimization**이다.

## 5. Dataset-Level Result

데이터셋별 top-3 요약:

| Dataset | Atom range | Edge range | Median speedup | Mean speedup | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `mptrj` | 444 | 24,168 to 28,508 | `1.007x` | `1.152x` | 한 샘플은 큰 승리, 나머지는 near break-even |
| `md22_double_walled_nanotube` | 370 | 24,024 to 24,546 | `1.005x` | `1.005x` | 사실상 본전 |
| `md22_stachyose` | 87 | 3,522 to 3,718 | `0.756x` | `0.764x` | 손해 |
| `ani1x` | 62 to 63 | 1,684 to 2,086 | `0.753x` | `0.746x` | 손해 |
| `spice_2023` | 89 | 692 to 772 | `0.760x` | `0.755x` | 손해 |
| `rmd17` | 24 | 366 to 370 | `0.761x` | `0.761x` | 손해 |
| `iso17` | 19 | 340 to 342 | `0.761x` | `0.761x` | 손해 |

핵심 해석:

- `mptrj`, `md22_double_walled_nanotube`처럼 매우 큰 그래프에서는 최소 break-even까지는 간다.
- `spice_2023`, `ani1x`, `rmd17`, `iso17`처럼 작은 분자 그래프는 명확히 손해다.
- `md22_stachyose`는 분자 계열 중에서는 edge가 큰 편이지만 여전히 `3.5k` 수준이라 pair 이득이 아직 overhead를 못 이긴다.

## 6. Dense Vs Sparse

이번 결과로 보면 "dense면 좋다"도 절반만 맞다.

맞는 부분:

- 현재 구현은 reverse pair가 잘 형성되는 graph에서 geometry/SH/weight 재사용이 가능하므로, directed edge 수가 커질수록 절약 가능한 절대 연산량도 커진다.
- 큰 periodic graph는 보통 edge 수와 평균 이웃 수가 모두 커서 pair가 아낄 수 있는 절대량이 커진다.

틀린 부분:

- dense alone is not enough.
- `md22_double_walled_nanotube`는 평균 directed neighbor가 약 `65`로 충분히 dense하지만 speedup은 `1.005x`에 머문다.
- 즉 density는 필요조건에 가깝고, 충분조건은 아니다.

가장 안전한 정리는 이렇다.

- `sparse + small`: 거의 확실히 손해
- `dense + small/medium`: 여전히 손해 또는 near break-even
- `dense + large`: 이득 가능성이 높아짐
- 하지만 `dense + large`에서도 이득 크기는 그래프별로 크게 달라진다

## 7. What Pair Execution Actually Saves

현재 `pair_execution(full)`이 줄이는 것은 모든 연산이 아니다.

줄어드는 것:

- edge embedding 계산 횟수: `num_edges -> num_pairs`
- weight NN row 수: `num_edges * num_conv_layers -> num_pairs * num_conv_layers`

줄지 않는 것:

- final TP row 수
- scatter row 수

대표 예시는 아래와 같다.

### 7.1 `mptrj` largest sample

- Atoms: `444`
- Directed edges: `28,508`
- Pairs: `14,254`
- Edge embedding evals: `28,508 -> 14,254`
- Weight NN rows total: `142,540 -> 71,270`
- TP rows total: `142,540 -> 142,540`
- Steady speedup: `1.444x`

### 7.2 `spice_2023` largest sample

- Atoms: `89`
- Directed edges: `772`
- Pairs: `386`
- Edge embedding evals: `772 -> 386`
- Weight NN rows total: `3,860 -> 1,930`
- TP rows total: `3,860 -> 3,860`
- Steady speedup: `0.743x`

이 비교가 말해주는 것은 명확하다. 현재 pair execution은 **모든 계산을 반으로 줄이는 기법이 아니다.**  
정확히는 `edge_embedding + weight_nn` 계열만 절반으로 줄이고, 최종 message TP와 scatter는 그대로 둔다.

## 8. Steady-State Pipeline Profiling

이 절에서는 intrusive 내부 타이머가 아니라, steady-state 경로를 그대로 따라가면서 아래 네 단계를 별도로 잰 값을 사용한다.

1. `graph_build`
2. `pair_metadata`
3. `device_transfer`
4. `model`

대표 샘플 결과:

| Dataset | Baseline pipeline | Pair pipeline | Pipeline speedup | Baseline model | Pair model | Model speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mptrj` | `627.33 ms` | `428.95 ms` | `1.462x` | `622.04 ms` | `421.54 ms` | `1.476x` |
| `md22_double_walled_nanotube` | `373.77 ms` | `371.88 ms` | `1.005x` | `370.06 ms` | `366.73 ms` | `1.009x` |
| `spice_2023` | `60.49 ms` | `79.20 ms` | `0.764x` | `58.47 ms` | `76.30 ms` | `0.766x` |
| `md22_stachyose` | `61.64 ms` | `81.22 ms` | `0.759x` | `60.38 ms` | `79.05 ms` | `0.764x` |
| `ani1x` | `58.60 ms` | `78.36 ms` | `0.748x` | `57.59 ms` | `76.46 ms` | `0.753x` |
| `rmd17` | `58.91 ms` | `77.26 ms` | `0.762x` | `57.88 ms` | `75.71 ms` | `0.764x` |
| `iso17` | `58.96 ms` | `77.24 ms` | `0.763x` | `57.96 ms` | `75.53 ms` | `0.767x` |

여기서 중요한 사실은 steady-state에서는 `pair_metadata`가 생각보다 매우 작다는 점이다.

pair 경로에서 `pair_metadata` 추가비용:

- `mptrj`: `+1.42 ms`
- `md22_double_walled_nanotube`: `+1.11 ms`
- `spice_2023`: `+0.35 ms`
- `ani1x`: `+0.37 ms`
- `iso17`: `+0.20 ms`

즉 steady-state에서 병목은 metadata가 아니라 여전히 `model`이다.  
실제 승패도 거의 전부 `model` 단계의 변화로 결정된다.

### 8.1 Why Large Graphs Win

`mptrj`:

- `pair_metadata`는 `+1.42 ms` 느려지지만
- `model`은 `-200.50 ms` 빨라진다
- 그래서 최종적으로 `1.462x` speedup이 난다

`md22_double_walled_nanotube`:

- `pair_metadata`는 `+1.11 ms`
- `model`은 `-3.33 ms`
- 결국 거의 break-even이다

즉 큰 그래프라고 해서 항상 큰 승리를 주는 게 아니라, 큰 그래프에서 **saved model work**가 충분히 커질 때만 실제 speedup으로 이어진다.

### 8.2 Why Small Graphs Lose

`spice_2023`:

- `pair_metadata`는 `+0.35 ms`
- `model`은 오히려 `+17.83 ms`
- 최종적으로 `0.764x`

`ani1x`:

- `pair_metadata`는 `+0.37 ms`
- `model`은 `+18.88 ms`
- 최종적으로 `0.748x`

즉 작은 그래프에서 손해가 나는 이유는 metadata 때문이 아니라, 현재 pair path가 model 내부에서 TP/scatter를 줄이지 못한 채 additional indexing/control overhead를 안고 들어가면서 model 자체가 느려지기 때문이다.

## 9. Load-Based Interpretation

이번 실험에서 load는 아래처럼 해석했다.

- `edge_embedding load`: edge embedding이 실제로 계산되는 row 수
- `weight_nn load`: 모든 convolution layer에서 weight MLP가 처리하는 총 row 수
- `TP load`: 모든 convolution layer에서 tensor product가 처리하는 총 row 수
- `scatter load`: 모든 convolution layer에서 reduce/scatter가 처리하는 총 row 수

현재 구현의 load 변화는 dataset와 무관하게 구조적으로 동일하다.

- `edge_embedding load`: 항상 `1/2`
- `weight_nn load`: 항상 `1/2`
- `TP load`: 그대로 `1.0`
- `scatter load`: 그대로 `1.0`

즉 현재 기법은 **model 내부 workload의 일부만 줄인다.**

이게 곧 관측 결과를 설명한다.

- 작은 graph에서는 줄어드는 load 절대량이 너무 작아서 overhead를 못 이긴다.
- 큰 graph에서는 줄어드는 load 절대량이 커져서 break-even을 넘기기 시작한다.
- 하지만 TP/scatter가 그대로라서 large graph에서도 무조건 큰 speedup이 나오지는 않는다.

## 10. What This Says About The Current Proposal

이번 결과를 가장 정확하게 표현하면 아래와 같다.

1. 현재 구현은 `FlashTP` 같은 TP-level fusion이 아니라, `geometry/SH/weight reuse` 중심의 최적화다.
2. 따라서 speedup 조건은 "그래프가 크고, 절약되는 edge/weight work가 충분히 클 것"이다.
3. 현재 구현은 `small molecule` regime에서는 적합하지 않다.
4. `large periodic graph` regime에서는 실제로 이득이 날 수 있다.
5. 다만 현재 구현은 TP/scatter를 줄이지 않으므로 speedup ceiling이 제한적이다.

## 11. Practical Message For Seminar

세미나에서는 아래처럼 정리하는 것이 가장 안전하다.

- "우리가 구현한 현재 버전은 pair-major TP kernel이 아니라, edge geometry와 weight 계산을 pair 단위로 재사용하는 단계까지 구현된 버전이다."
- "그래서 모든 상황에서 빨라지지 않는다."
- "실제 로컬 대형 공개 데이터셋 검증 결과, `<=5k edges` 영역에서는 pair path가 일관되게 손해였고, `>15k edges` 영역에서는 평균적으로 이득이었다."
- "steady-state profiling 결과, 승패는 거의 전부 model 단계에서 갈렸고, pair metadata overhead는 cache가 켜진 상태에서는 매우 작았다."
- "결국 현재 구현은 dense-large graph에서는 유효하지만, universal optimization은 아니다."

## 12. Limitation

- 샘플 선택은 각 데이터셋의 atom 수 top-3이므로 dataset 전체 분포를 완전히 대표하지는 않는다.
- `5k-15k` edge 구간 샘플이 이번 로컬 셋에서는 비어 있어, 전이 구간을 더 촘촘히 보지는 못했다.
- `profiles.csv`의 model-internal 상세 타이머는 함수 wrapping 기반이라 absolute runtime 해석에는 부적절하고, load 변화 방향을 확인하는 용도로만 써야 한다.

## 13. Bottom Line

현재 데이터가 지지하는 가장 강한 결론은 다음이다.

- **크기가 클수록 pair execution이 유리해질 가능성은 분명히 커진다.**
- **하지만 size alone으로는 충분하지 않고, dense-large regime에서만 본격적으로 듣는다.**
- **현재 speedup의 원인은 TP 자체 감소가 아니라 edge_embedding/weight_nn 재사용이다.**
- **따라서 다음 단계에서 더 큰 개선을 원하면 TP/scatter까지 pair-major로 바꾸는 새 커널이 필요하다.**
