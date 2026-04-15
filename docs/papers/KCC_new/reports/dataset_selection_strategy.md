# Dataset Selection Strategy for KCC_new

이 문서는 KCC 논문 본문/부록에 넣을 데이터셋을 고르는 기준과 추천 세트를 정리한다. 목표는 결과를 유리하게만 고르는 것이 아니라, 논문 논리와 실제 측정 결과가 서로 어긋나지 않도록 대표성을 확보하는 것이다.

## 1. 선별 원칙

데이터셋은 다음 네 조건을 동시에 만족하도록 고른다.

1. 논문 주장과 어긋나지 않아야 한다.
   - `geometry_only`는 정확도 보존형 exact reuse 검증용이다.
   - restored `full` two-pass는 pair-aware 실행 스케줄까지 포함한 성능 경로이다.
   - 따라서 `geometry_only` 결과를 성능 win으로 포장하지 않는다.
2. size/density 논리와 어긋나지 않아야 한다.
   - 현재 canonical bucket 기준은 `num_edges >= 3000`이면 large, `avg_neighbors_directed >= 40`이면 dense이다.
   - 현재 31개 로컬 데이터셋에는 `small_dense` bucket이 없다.
   - 따라서 본문에서는 존재하지 않는 사분면을 억지로 만들지 않고, `large_dense`, `large_sparse`, `small_sparse`, 그리고 boundary case로 구성한다.
3. profiler 해석과 어긋나지 않아야 한다.
   - `geometry_only` profiler는 `index_select`와 pair->edge expansion이 늘어남을 보여준다.
   - restored `full`의 win은 pure geometry-only가 아니라 pair-aware scheduling 효과를 포함한다.
   - 따라서 대표 profiler는 small loss와 large win을 모두 포함해야 한다.
4. MLIP 문헌에서 인지도가 있어야 한다.
   - 우선순위는 OC20, MD22, rMD17/ISO17, QM9, Materials Project/MPtrj, SPICE 계열이다.
   - OMat/OMol/OC22/SALE X는 최신·대형 재료/분자 데이터셋으로 보조 근거로 둔다.

## 2. 현재 결과 요약

### 2.1 geometry_only

`geometry_only`는 정확도 보존 검증에는 적합하지만, 현재 end-to-end 성능 주장은 어렵다.

| bucket | count | median speedup | 해석 |
| --- | ---: | ---: | --- |
| large_dense | 17 | 0.988x | 손해 폭은 작지만 win은 아님 |
| large_sparse | 3 | 0.988x | 손해 폭은 작지만 win은 아님 |
| small_sparse | 11 | 0.983x | small 쪽 손해가 더 큼 |

### 2.2 restored full two-pass

restored `full` two-pass는 large/dense 또는 large/sparse workload에서 성능 win을 복구한다.

| bucket/condition | count | wins | median speedup | 해석 |
| --- | ---: | ---: | ---: | --- |
| num_edges >= 3000 | 20 | 18 | 1.014x | large graph에서 대부분 win |
| num_edges < 3000 | 11 | 0 | 0.613x | small graph에서는 전부 loss |
| avg_neighbors_directed >= 40 | 17 | 16 | 1.013x | dense graph에서 대부분 win |
| avg_neighbors_directed < 40 | 14 | 2 | 0.620x | sparse에서는 large 여부가 중요 |
| large_dense | 17 | 16 | 1.013x | 본문 positive claim의 핵심 |
| large_sparse | 3 | 2 | 1.053x | edge 수가 충분하면 sparse도 win 가능 |
| small_sparse | 11 | 0 | 0.613x | negative/control group |

## 3. 본문 추천 세트

본문에는 6개를 추천한다. positive 3개, negative/control 2개, boundary 1개 구성이다.

| 역할 | dataset | bucket | restored full speedup | geometry_only speedup | 선택 이유 |
| --- | --- | --- | ---: | ---: | --- |
| large_dense positive | `md22_buckyball_catcher` | large_dense | 1.071x | 0.990x | MD22 계열, top win, 분자 MLIP 인지도 높음 |
| large_sparse positive | `oc20_s2ef_val_ood_ads` | large_sparse | 1.073x | 0.988x | OC20 S2EF, top win, 촉매/표면 MLIP 대표 |
| large_dense canonical material | `mptrj` | large_dense | 1.011x | 0.988x | Materials Project 계열, SevenNet/재료 MLIP와 연결성이 큼 |
| small_sparse negative | `qm9_hf` | small_sparse | 0.610x | 0.984x | QM9 계열, 소형 분자 대표, small loss 설명에 적합 |
| small_sparse negative | `rmd17` 또는 `iso17` | small_sparse | 0.613x / 0.612x | 0.983x / 0.973x | 분자 동역학 벤치마크로 인지도 높음 |
| boundary/counterexample | `md22_stachyose` | large_dense | 0.987x | 0.984x | threshold 근처 large_dense지만 loss, 단순 dense claim 방지 |

### 본문에서 쓰는 방식

1. `md22_buckyball_catcher`, `oc20_s2ef_val_ood_ads`, `mptrj`로 large workload에서 win이 복구됨을 보인다.
2. `qm9_hf`, `rmd17` 또는 `iso17`로 small graph에서는 pair-aware overhead가 더 크다는 점을 보인다.
3. `md22_stachyose`로 `num_edges >= 3000`과 `avg_neighbors >= 40`만으로 무조건 빨라지는 것이 아니라, 충분한 계산량과 workload shape가 필요하다는 점을 보인다.

이 구성이 가장 안전하다. positive만 고르면 cherry-pick으로 보이고, small loss만 빼면 논리 방어가 약해진다.

## 4. 부록/전체표 추천 세트

부록에는 전체 31개 표를 유지한다. 본문에는 6개만 상세 그래프/프로파일링 대상으로 쓰고, 전체 31개 결과는 조건별 표로 제시한다.

본문 외 보조로 넣기 좋은 데이터셋은 다음과 같다.

| 역할 | dataset | bucket | restored full speedup | 이유 |
| --- | --- | --- | ---: | --- |
| OC20 추가 positive | `oc20_s2ef_train_20m` | large_sparse | 1.053x | OC20 train split, top win |
| OC20 dense positive | `oc20_s2ef_val_ood_cat` | large_dense | 1.023x | OC20 dense split |
| 최신 대형 분자 | `omol25_train_neutral` | large_dense | 1.027x | 대형 분자 workload |
| 재료 dense | `omat24_1m_official` | large_dense | 1.013x | 최신 재료 데이터셋 |
| 대형 molecular dynamics | `md22_double_walled_nanotube` | large_dense | 1.010x | 매우 큰 MD22 계열 |
| small molecular control | `spice_2023` | small_sparse | 0.626x | SPICE 계열, small sparse loss |

## 5. 사분면 문제

현재 threshold 기준으로는 `small_dense` 데이터셋이 없다.

| quadrant | 현재 존재 여부 | 추천 처리 |
| --- | --- | --- |
| large_dense | 있음 | `md22_buckyball_catcher`, `mptrj` |
| large_sparse | 있음 | `oc20_s2ef_val_ood_ads`, `oc20_s2ef_train_20m` |
| small_sparse | 있음 | `qm9_hf`, `rmd17`, `iso17` |
| small_dense | 없음 | 억지로 만들지 말고 “현재 로컬 31개에는 없음”으로 명시 |

만약 반드시 4사분면 그림이 필요하면 두 선택지가 있다.

1. canonical threshold를 유지한다.
   - 장점: 논리적으로 안전하다.
   - 단점: small_dense quadrant가 비어 있다.
2. threshold를 바꾸거나 별도 synthetic/constructed dense small system을 만든다.
   - 장점: 그림은 예쁘다.
   - 단점: 논문 신뢰도가 떨어질 수 있다.

권장안은 1번이다. KCC 논문에서는 비어 있는 quadrant를 그대로 보이는 것이 더 정직하다.

## 6. 프로파일링 추천 세트

현재 profiler는 `qm9_hf`와 `mptrj`만 있다. 논문 논리를 강화하려면 아래 4개를 추가 profiler 대상으로 잡는 것이 좋다.

| dataset | 목적 | 기대되는 해석 |
| --- | --- | --- |
| `qm9_hf` | small_sparse loss | pair overhead와 index/select 비용이 이득보다 큼 |
| `md22_buckyball_catcher` | large_dense strong win | restored full two-pass가 large molecular workload에서 유리 |
| `oc20_s2ef_val_ood_ads` | large_sparse top win | dense가 아니어도 edge 수가 충분하면 win 가능 |
| `mptrj` | canonical large material | 재료 MLIP 대표에서 modest but stable win |
| `md22_stachyose` | boundary counterexample | threshold 근처에서는 win이 보장되지 않음 |

프로파일링 결과가 논문 논리와 맞으려면 다음 패턴을 확인해야 한다.

1. small_sparse에서는 pair-aware overhead, metadata, gather/indexing이 상대적으로 커야 한다.
2. large_dense/large_sparse winner에서는 baseline 대비 restored full의 model path가 줄어야 한다.
3. boundary case에서는 pair-aware 이득과 overhead가 거의 상쇄되어야 한다.
4. force path는 여전히 남은 병목으로 설명해야 한다.

## 7. 최종 권장

본문 핵심 6개:

1. `md22_buckyball_catcher`
2. `oc20_s2ef_val_ood_ads`
3. `mptrj`
4. `qm9_hf`
5. `rmd17`
6. `md22_stachyose`

본문 표에는 위 6개를 넣고, 전체 31개는 부록 또는 supplementary 표로 유지한다.  
본문 그래프는 `speedup by dataset`, `condition summary`, `selected dataset latency`, `selected profiler breakdown` 네 개면 충분하다.

주의할 문장:

- “large/dense에서는 항상 빨라진다”라고 쓰면 안 된다.
- “restored full의 win이 pure geometry_only 효과다”라고 쓰면 안 된다.
- “현재 구현이 pair-major fused kernel이다”라고 쓰면 안 된다.

권장 문장:

> restored full two-pass 실행은 pair-aware geometry reuse와 pair-level weight sharing, forward/reverse 분리 스케줄을 함께 사용한다. 이 경로는 small sparse graph에서는 고정 오버헤드 때문에 불리하지만, edge 수가 충분한 large/dense 및 일부 large/sparse workload에서는 baseline보다 빠르다.
