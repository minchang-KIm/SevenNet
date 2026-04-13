# 등변 GNN 원자간 퍼텐셜 추론에서 Pair-Aware Geometry-Side Exact Reuse

**Pair-Aware Geometry-Side Exact Reuse for Equivariant GNN Interatomic Potential Inference**

김민창  
아주대학교 분산병렬컴퓨팅 연구소 WiseLab  
minchang111@ajou.ac.kr

## 요 약

본 논문은 NequIP 계열 등변 그래프 신경망 원자간 퍼텐셜 추론에서 동일한 물리적 pair가 두 개의 directed edge로 중복 표현된다는 점에 주목하고, SevenNet 런타임에 pair-aware geometry-side exact reuse를 구현한 결과를 정리한다. 현재 구현은 pair-major tensor-product 실행 엔진이 아니라, reverse edge pair를 이용하여 distance, radial basis, cutoff, spherical harmonics, pair-level `weight_nn` 입력을 pair 기준으로 1회만 계산하거나 복원하는 runtime reformulation이다. 모델의 수식과 학습된 파라미터는 바꾸지 않으며, baseline 대비 에너지와 힘을 실용적으로 동일한 수준으로 유지한다. 본 논문의 메인 비교는 `SevenNet baseline`과 `SevenNet + 제안기법`의 non-intrusive repeated end-to-end latency이며, intrusive synchronized detailed profile은 어떤 비용이 줄고 어떤 오버헤드가 추가되는지를 해석하기 위한 보조 실험으로 사용한다. local public inventory에서 직접 벤치 가능한 31개 데이터셋 전체에 대해 측정한 결과, `SevenNet + 제안기법`은 전체 median 기준 `1.010x`의 speedup을 보였고 31개 중 18개에서 baseline보다 빨랐다. 그러나 이 이득은 모든 workload에서 균일하지 않았다. `num_edges >= 3000`인 그래프의 승률은 `90%`였고, `avg_neighbors_directed >= 40`인 경우 승률은 `94.1%`였다. 반대로 `num_edges < 3000`인 그래프에서는 승률이 `0%`였다. bucket 기준으로도 `large_dense`는 17개 중 16개에서, `large_sparse`는 3개 중 2개에서 이득을 보였지만, `small_sparse` 11개는 모두 손해였다. 즉 본 연구의 핵심 기여는 단순히 최적화를 구현한 것에 그치지 않고, 어떠한 MLIP workload 조건에서 geometry-side exact reuse가 실질적인 이득으로 이어지는지를 규정한 데 있다. intrusive detailed profile에서는 `e3nn_pair_full`이 `e3nn_baseline` 대비 model-total 기준 median `1.318x`의 감소를 보였고, measured `conv_message_tp_ms`가 median `55.7 ms` 줄어든 반면 pair bookkeeping 오버헤드는 `pair_indexing_ms` 약 `0.51 ms`, `pair_expand_ms` 약 `0.084 ms` 수준이었다. 이 결과는 large, high-neighbor workload에서 오버헤드를 넘어서는 성능 향상이 이미 관측된다는 점을 보여주며, pair-major execution과 FlashTP 결합은 후속 연구로 다룬다.

**주제어**: 등변 그래프 신경망, 원자간 퍼텐셜, 구면조화함수, 추론 최적화, SevenNet, workload characterization

## 1. 서 론

등변 그래프 신경망 기반 원자간 퍼텐셜은 회전 대칭성과 물리적 귀납 bias를 직접 반영할 수 있기 때문에 재료 및 분자 시뮬레이션에서 중요한 계산 모델로 자리잡고 있다. 특히 NequIP 계열의 모델은 local atomic environment를 directed edge 집합으로 표현하고, 각 edge에 대해 기하 기반 attribute를 구성한 뒤 tensor product 메시지 생성과 aggregation을 반복하는 구조를 취한다. 이 접근은 높은 정확도를 제공하지만, 시스템 관점에서 보면 동일한 pair 상호작용을 `i -> j`와 `j -> i`의 두 edge로 중복 표현한다는 구조적 비효율을 안고 있다.

이 중복 표현이 항상 낭비를 의미하는 것은 아니다. source node feature에 의존하는 메시지 생성은 방향성을 그대로 가지므로 두 edge를 완전히 하나로 합칠 수 없다. 반면, edge geometry에서 출발하는 distance, radial basis, cutoff, spherical harmonics는 pair 관점에서 exact reuse 기회를 가진다. distance와 radial/cutoff는 양방향에서 동일하고, spherical harmonics는 parity 관계를 이용해 역방향 값을 정확히 복원할 수 있기 때문이다. 따라서 모델 수식과 학습된 파라미터를 바꾸지 않고도, 실행 표현만 pair-aware하게 바꾸어 geometry-side 중복 계산을 줄일 수 있다.

본 논문은 이 아이디어를 SevenNet 런타임에 실제로 구현하고, 그 의미와 한계를 정리한 선행 연구이다. 여기서 강조해야 할 점은 현재 구현이 pair-major tensor-product 실행 전체를 완성한 것이 아니라는 사실이다. 본 연구가 다루는 것은 `pair-aware geometry-side exact reuse runtime`이며, 최종 tensor product, aggregation, force/stress backward는 여전히 edge-major로 남아 있다. 따라서 본 논문의 목표는 범용적인 대규모 성능 향상을 선언하는 것이 아니라, SevenNet baseline 대비 어떤 조건에서 성능 이득이 나타나는지, 그리고 현재 구현 범위 안에서 무엇이 줄고 무엇이 남는지를 정확히 밝히는 데 있다. FlashTP와의 결합은 본 논문의 핵심 성능 주장 대상이 아니라, 후속 runtime 결합 연구의 방향으로만 다룬다.

## 2. 배 경

### 2.1 등변 GNN-IP 추론 파이프라인

NequIP/SevenNet 계열에서 추론은 크게 네 단계로 나뉜다. 먼저 원자종을 node feature로 임베딩하고, cutoff 이내 이웃으로부터 directed edge를 구성한다. 다음으로 각 edge에 대해 거리 기반 scalar embedding과 spherical harmonics 기반 tensor attribute를 계산한다. 그 후 interaction block에서 `weight_nn`과 tensor product를 통해 메시지를 생성하고, 이를 destination node에 aggregation한다. 마지막으로 atomic energy를 readout한 뒤 total energy에 대한 미분을 통해 힘과 stress를 계산한다.

여기서 중요한 사실은 spherical harmonics 계산이 메시지 생성 그 자체가 아니라는 점이다. spherical harmonics는 tensor product에 들어갈 edge-side attribute를 준비하는 단계이며, 실제 메시지 생성은 이후 source node feature와 edge attribute가 tensor product에서 결합될 때 일어난다. 따라서 재사용 가능성과 불가능성의 경계는 바로 이 geometry encoding 단계와 directed message generation 단계 사이에 존재한다.

### 2.2 pair 단위에서 재사용 가능한 항과 불가능한 항

pair 단위에서 exact reuse가 가능한 항은 distance, radial basis, cutoff, spherical harmonics, 그리고 pair-wise `weight_nn` 입력이다. 반대로 source node feature에 직접 의존하는 tensor-product message, destination node 방향의 aggregation, 그리고 total energy에서 force/stress를 구하기 위한 backward 경로는 현재 구조에서 directed edge 의미를 그대로 유지한다. 이 구분이 본 논문의 핵심이다. 즉, 본 연구의 제안기법은 메시지 생성 전체를 줄이는 방법이 아니라, 메시지 생성 이전 geometry-side 계산의 중복을 제거하는 방법이다.

### 2.3 FlashTP와의 관계

FlashTP는 tensor product, gather, scatter를 fused kernel로 실행함으로써 directed edge 기반 convolution backend를 최적화한다. 그러나 FlashTP는 edge embedding 단계의 spherical harmonics 계산 자체를 바꾸지 않는다. SevenNet 코드에서도 FlashTP 활성화는 convolution backend를 `IrrepsScatterGatterFusedConvolution`과 `uvu_TP`로 교체할 뿐, `edge_embedding.py`에서 수행되는 `self.spherical(...)` 호출은 그대로 남아 있다. 따라서 FlashTP와 본 연구의 geometry-side reuse는 같은 부분을 두 번 최적화하는 관계가 아니라, 서로 다른 층위를 겨냥하는 상보적 관계이다.

## 3. Pair-Aware Geometry-Side Reuse Runtime

### 3.1 현재 구현의 범위

현재 SevenNet 구현은 pair metadata를 구성한 뒤 canonical forward edge 기준으로 geometry-side 항을 한 번만 계산한다. 구체적으로는 undirected pair에 대해 edge-to-pair map, forward/backward index, reverse mask, has-reverse mask를 만들고, canonical 방향의 distance, radial basis, cutoff, spherical harmonics를 1회 계산한다. 역방향 spherical harmonics는 parity sign을 이용해 복원하고, `weight_nn` 역시 pair-level 입력에 대해 1회만 수행한다.

그러나 이 구현은 pair-major tensor-product kernel이 아니다. non-Flash 경로에서는 pair-wise `weight_nn`과 geometry reuse 뒤에 forward와 reverse 메시지를 여전히 directed edge 기준으로 생성하며, Flash 경로에서는 pair-wise `weight_nn` 이후 다시 edge-major weight로 펼쳐 fused convolution kernel에 넣는다. 다시 말해, 현재 구현은 geometry-side exact reuse는 실현하지만, message TP와 reduction을 pair-major 상태로 끝까지 유지하지 못한다.

### 3.2 논문에서의 정확한 주장

따라서 본 논문이 주장하는 공헌은 다음 네 가지로 제한한다. 첫째, reverse edge pair를 이용해 geometry-side reusable terms를 exact reuse 또는 exact reconstruction하는 runtime을 구현했다. 둘째, pair-wise `weight_nn` 입력 재사용까지 포함해 execution reformulation을 실제 SevenNet 코드에 반영했다. 셋째, 31개 public-local 데이터셋 전수 실험을 통해 제안기법이 유리해지는 workload 조건을 정량적으로 규정했다. 넷째, 현재 구현이 pair-major TP가 아님을 명시적으로 드러내고, 후속 연구에서 pair-major fused execution 및 FlashTP 결합으로 확장할 논리적 근거를 제시한다.

이 범위를 벗어나는 주장은 본 논문에서 하지 않는다. 즉, 현재 구현을 pair-major 전체 엔진으로 부르지 않으며, FlashTP 위에서 항상 큰 폭의 추가 가속을 달성했다고 주장하지 않는다.

## 4. 실험 방법

### 4.1 데이터셋과 대표 샘플 선택

실험은 local public inventory에서 직접 benchmark 가능한 31개 데이터셋 전체를 대상으로 수행한다. 각 데이터셋에서는 대표 sample 하나를 선택하여 측정한다. representative sample은 현재 로컬에 존재하는 원시 데이터에서 benchmark script가 직접 복원 가능한 구조 중 가장 큰 sample을 기준으로 잡는다. 이 방식은 모든 데이터셋을 동일한 기준으로 비교할 수 있게 하며, 이후 dataset별 개별 CSV를 저장해 필요한 subset만 다시 조합한 figure를 생성할 수 있도록 한다.

### 4.2 계측 family와 직접 비교 가능 범위

본 논문은 네 가지 계측 family를 구분하되, 메인 성능 비교는 하나만 사용한다. 첫째, `SevenNet baseline vs proposal-only end-to-end`는 `e3nn_baseline`과 `e3nn_pair_full`에 대해 non-intrusive repeated wall-clock timing을 측정하는 family이며, 본 논문의 headline latency 결과는 전부 이 family에 기반한다. 둘째, `e3nn baseline detailed profile`은 intrusive synchronized stage timing으로 SH, radial basis, cutoff, `weight_nn`, TP, aggregation, force/stress output 등을 분해한다. 셋째, `e3nn pair detailed profile`은 같은 intrusive 조건에서 pair-aware 실행의 stage 변화를 측정한다. 넷째, `FlashTP end-to-end`와 representative Nsight는 후속 결합 가능성을 이해하기 위한 보조 family이다.

이들 family는 서로 역할이 다르다. `SevenNet baseline vs proposal-only end-to-end`는 headline latency 비교에 직접 사용할 수 있다. `e3nn baseline detailed`와 `e3nn pair detailed`는 stage share와 오버헤드 해석에 직접 사용할 수 있다. 반면 intrusive detailed time을 non-intrusive end-to-end latency와 직접 비교해서는 안 되며, representative Nsight 역시 kernel mix 검증용이지 본 논문의 headline latency 근거가 아니다.

### 4.3 반복 측정과 통계

메인 family인 `SevenNet baseline vs proposal-only end-to-end`는 warmup 3회 후 measured repeat 10회로 수집하며 `mean ± std`, median, p95를 저장한다. detailed profile은 warmup 이후 outer repeat 5회로 수행하고, 각 repeat마다 intrusive stage summary를 수집한 뒤 mean과 standard deviation을 계산한다. 이러한 통계는 dataset별 개별 CSV와 전체 통합 CSV에 모두 저장된다.

### 4.4 Nsight 사용 원칙

Nsight Systems는 kernel-level 구조를 확인하는 데 유용하지만, 31개 데이터셋 전수에 대한 repeated wall-clock 통계의 주 계측기로 사용하기에는 무겁고 host/toolchain 의존성이 크다. 따라서 본 논문에서는 synchronized repeated timing과 intrusive stage timing을 canonical 계측으로 두고, Nsight는 representative validation 도구로만 사용한다. 실제 KCC 패키지에는 representative Nsight 수집 스크립트를 포함하지만, 본 호스트의 `nsys 2020.3` 환경에서는 FlashTP Python 경로에 대한 per-run export 안정성이 충분하지 않아 canonical 본문 수치와 그림은 세 가지 primary family에만 기반한다. 즉, “Nsight를 쓰지 않는다”가 아니라 “전수 headline 통계의 주 수단으로 쓰지 않는다”가 본 논문의 입장이다.

## 5. 실험 결과

### 5.1 SevenNet baseline 대비 end-to-end 결과

본 논문의 메인 성능 결과는 `SevenNet baseline`과 `SevenNet + 제안기법`의 non-intrusive repeated end-to-end 비교이다. 31개 데이터셋 전체에서 `SevenNet + 제안기법`은 median `1.010x`의 speedup을 보였고, 31개 중 18개에서 baseline보다 빨랐다. 다만 geometric mean은 `0.857x`로 전체 분포가 한쪽으로만 개선되지는 않았으며, 성능은 데이터셋의 graph 규모와 밀도에 강하게 의존했다. 정확도는 유지되었다. 제안기법 적용 시 worst absolute energy delta는 `6.104e-05 eV`, worst force delta는 `1.831e-04 eV/A`였고, 모두 `omat24_1m_official`에서 관측되었다.

![그림 1](../figures/pair_end_to_end/pair_latency_all.png)

![그림 2](../figures/pair_end_to_end/pair_speedup_all.png)

best case는 `md22_buckyball_catcher (1.067x)`였고, `oc20_s2ef_val_ood_ads (1.065x)`, `oc20_s2ef_train_20m (1.055x)`가 뒤를 이었다. 반대로 worst case는 `md22_dha (0.598x)`였으며, `md22_ac_ala3_nhme`, `md22_at_at`, `qm9_hf`, `rmd17`, `iso17`, `spice_2023`, `phonondb_pbe` 등 small sparse workload는 모두 뚜렷한 손해를 보였다. 즉, 제안기법의 성능은 “항상 유리하다”가 아니라 “어떤 graph regime에서 유리한가”의 문제로 읽어야 한다.

### 5.2 제안기법이 유리해지는 조건

이번 전수 실험에서 가장 강한 분리 변수는 graph 크기와 이웃 밀도였다. `num_edges >= 3000`인 그래프의 승률은 `90%`였고 median speedup은 `1.013x`였다. 반대로 `num_edges < 3000`인 그래프의 승률은 `0%`였고 median speedup은 `0.613x`였다. 이웃 밀도 기준으로 보아도 `avg_neighbors_directed >= 40`에서는 승률이 `94.1%`, median speedup이 `1.013x`였고, 그보다 낮은 구간에서는 승률이 `14.3%`, median speedup이 `0.615x`에 그쳤다. Spearman 상관도 `num_edges`와 speedup 사이 `0.594`, `avg_neighbors_directed`와 speedup 사이 `0.629`, `natoms`와 speedup 사이 `0.681`로 모두 양의 방향을 보였다.

![그림 3](../figures/pair_end_to_end/pair_speedup_vs_num_edges.png)

![그림 4](../figures/pair_end_to_end/pair_speedup_size_density_map.png)

bucket별로 보면 `large_dense`는 17개 중 16개에서 승리했고 median speedup은 `1.013x`였다. `large_sparse`는 표본 수가 3개로 작지만 2개에서 승리했고 median speedup은 `1.055x`였다. 반면 `small_sparse` 11개는 전부 손해였고 median speedup은 `0.613x`였다. 이 결과는 본 연구의 노블티가 단순 구현 자체에만 있는 것이 아니라, **geometry-side exact reuse가 large/high-neighbor MLIP inference에서 유의미한 강점을 가진다는 조건을 정의했다는 점**에도 있음을 보여준다.

![그림 5](../figures/pair_end_to_end/pair_speedup_by_bucket.png)

### 5.3 Detailed profile과 오버헤드 해석

end-to-end 결과가 어디에서 오는지를 이해하기 위해 `e3nn baseline detailed`와 `e3nn pair detailed`를 같은 intrusive 조건에서 비교하였다. 이 결과는 absolute latency claim용이 아니라 stage decomposition용이다. 이 기준에서 `e3nn_pair_full`은 `e3nn_baseline` 대비 model-total 기준 median `1.318x`의 감소를 보였다. measured `conv_message_tp_ms`는 median `55.7 ms` 감소했고, `conv_weight_nn_ms`도 소폭 줄었다. 반면 명시적으로 추가되는 오버헤드는 작았다. `pair_indexing_ms`는 median 약 `0.51 ms`, `pair_expand_ms`는 median 약 `0.084 ms` 수준이었다.

![그림 6](../figures/figure_04_representative_stage_breakdown.png)

이 해석은 end-to-end 조건 분석과도 맞물린다. small sparse graph에서는 줄어드는 계산량의 절대 크기가 작아 sub-ms 수준의 pair bookkeeping 오버헤드와 반복 호출 비용을 상쇄하지 못한다. 반대로 large/high-neighbor graph에서는 geometry-side exact reuse와 pair-level `weight_nn` 재사용으로 줄어드는 절대량이 커져 오버헤드를 넘기기 시작한다. 다시 말해, 제안기법의 유효성은 “재사용 가능한 항을 얼마나 줄였는가”뿐 아니라 “그 절감량이 graph 규모에 따라 어느 정도의 절대 시간으로 나타나는가”에 의해 결정된다.

### 5.4 FlashTP와의 관계 및 후속 연구

FlashTP와의 결합은 본 논문의 메인 결과가 아니다. 본 논문은 먼저 SevenNet baseline 대비 제안기법만 적용했을 때 어떤 조건에서 이득이 생기는지를 밝히는 선행 연구로 위치시킨다. 이 점이 중요한 이유는, FlashTP와 무관한 현재 SevenNet baseline 비교만으로도 large/high-neighbor 조건에서 이미 오버헤드를 넘는 성능 향상이 관찰되었기 때문이다. 따라서 geometry-side exact reuse 자체가 large MLIP workload에서 의미 있는 실행 최적화 방향이라는 점은 독립적으로 성립한다.

FlashTP는 directed-edge convolution backend를 fused kernel로 최적화하지만 SH 계산 자체는 줄이지 않는다. 따라서 pair-major execution과 FlashTP 결합이 완성된다면, 현재와 같은 pair bookkeeping 오버헤드를 유지하더라도 더 큰 성능 향상이 가능할 것으로 기대할 수 있다. 다만 이 부분은 현재 구현으로 실증한 결과가 아니라 후속 연구 방향이다. 현재 코드에서는 Flash 경로가 pair-level weight를 다시 edge-major layout으로 확장하므로, 본 논문은 FlashTP와의 상보성을 설계 수준에서만 논의하고 강한 성능 claim은 하지 않는다.

### 5.5 논의

본 연구가 주는 가장 중요한 교훈은 “재사용 가능한 항을 정확히 구분하고, 그 유효 범위를 workload 조건과 함께 제시해야 한다”는 점이다. distance, radial basis, cutoff, spherical harmonics, pair-level `weight_nn` 입력은 exact reuse가 가능하다. 반면 source feature 의존 TP message와 force/stress backward는 현재 구조에서 직접 공유되지 않는다. 따라서 제안기법은 모든 graph에서 같은 방식으로 이득을 주지 않으며, large graph/high-neighbor regime에서만 오버헤드를 넘어서는 경향을 보인다.

이 점은 오히려 본 연구의 기여를 더 분명하게 만든다. 본 논문은 “모든 경우에 빠르다”는 과장된 주장을 하지 않는다. 대신 **어떤 MLIP workload에서 geometry-side exact reuse가 의미 있는가**를 실험적으로 구분하고, 그 조건에서 baseline 대비 실제 이득이 있음을 보인다. 이후 pair-major tensor-product execution과 FlashTP layout-level 결합이 구현된다면, 현재 large/high-neighbor 조건에서 관측된 이득을 더 확대할 가능성이 있다. 본 논문은 바로 그 후속 연구를 위한 조건 정의와 runtime 근거를 제공한다.

## 6. 결론

본 논문은 SevenNet에서 pair-aware geometry-side exact reuse를 구현하고, 그 의미와 한계를 31개 public-local benchmark dataset에 대해 분석하였다. 현재 구현은 pair-major TP 엔진이 아니라 geometry-side reusable terms의 exact reuse runtime이며, 본 논문의 메인 성능 비교는 `SevenNet baseline`과 `SevenNet + 제안기법`의 end-to-end repeated timing에 기반한다. 그 결과 전체 median speedup은 `1.010x`였고, 31개 중 18개에서 baseline보다 빨랐다. 그러나 이 이득은 workload 조건에 따라 뚜렷하게 갈렸다. `num_edges >= 3000`인 그래프의 승률은 `90%`, `avg_neighbors_directed >= 40`인 그래프의 승률은 `94.1%`였지만, `num_edges < 3000`인 그래프는 단 한 번도 이기지 못했다.

이 결과는 두 가지 의미를 갖는다. 첫째, geometry-side exact reuse는 구현 가능하고 수치적으로 안전하며, large/high-neighbor MLIP inference에서는 실제로 baseline 대비 이득을 만든다. 둘째, 이 조건 정의 자체가 하나의 기여이다. 즉, 제안기법은 모든 graph에 대한 범용 최적화가 아니라 특정 workload regime에 대해 효과적인 실행 최적화임을 밝혔다. 후속 연구에서는 pair-major tensor-product execution, pair-major fused reduction, FlashTP와의 layout-level 결합을 통해 현재 관측된 large/high-neighbor regime의 이득을 더 확대할 수 있다. 따라서 본 논문은 KCC 선행 연구로서 현재 구현의 범위, 유효 조건, 후속 확장 방향을 정리하고 이후 전체 pair-major 실행 재설계로 이어질 기반을 제공한다.

## 참 고 문 헌

[1] S. Batzner, A. Musaelian, L. Sun, et al., “E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials,” *Nature Communications*, vol. 13, 2453, 2022.  
[2] Y. Park, et al., “SevenNet: a graph neural network interatomic potential package supporting efficient multi-GPU parallel molecular dynamics simulations,” *Journal of Chemical Theory and Computation*, 2024.  
[3] J. Lee, et al., “FlashTP: fused, sparsity-aware tensor product for machine learning interatomic potentials,” 2024.  
[4] Y. Zhang and H. Guo, “Node-Equivariant Message Passing for Efficient and Accurate Machine Learning Interatomic Potentials,” 2025.
