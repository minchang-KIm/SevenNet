# Seminar_김민창 GNN-IP 최종 세미나 대본

## 2026-04-03 Correction Note

- 이 대본은 세미나 시점의 내부 발표 버전이다.
- 이후 benchmark sanitation에서 small-graph case 일부에 warm-up artifact가 확인되었다.
- 현재 canonical 해석은 `geometry-side reuse`, `small molecular에서는 대체로 손해`, `large periodic에서는 modest gain`이다.

## Slide 1. 제목 (1.0 min)
오늘 발표는 GNN-IP 추론 최적화, 그중에서도 pair execution 관점에서 지금까지 무엇을 이해했고 무엇을 구현했고, 어떤 실험 결과가 나왔는지를 정리하는 랩 세미나입니다. 핵심은 새 모델 제안이 아니라, equivariant interatomic potential의 실행 구조를 다시 보는 것입니다.

## Slide 2. 발표 구성 (1.0 min)
발표는 네 부분으로 진행하겠습니다. 먼저 GNN-IP 전체를 짧게 설명하고, 그 위에서 재사용 가능한 연산값을 어떻게 분류했는지 보겠습니다. 그 다음 SevenNet 기준 구현과 관련 최적화 연구를 연결하고, 마지막으로 실험 결과와 discussion으로 마무리하겠습니다.

## Slide 3. GNN-IP란 무엇인가 (2.0 min)
작업자 외 학생들을 위해 가장 짧게 정리하면, GNN-IP는 원자 구조를 그래프로 보고 원자별 local environment로부터 total energy를 예측한 뒤, force와 stress를 gradient로 얻는 모델입니다. 핵심은 원자 위치가 회전해도 물리량이 맞게 변해야 하므로 equivariance가 중요하다는 점입니다.

## Slide 4. Equivariant 추론 파이프라인 (2.0 min)
실제 추론은 atom과 neighbor list에서 시작해 edge geometry를 만들고, radial basis, cutoff, spherical harmonics로 edge feature를 만든 뒤, 여러 개의 interaction block에서 weight_nn, tensor product, aggregation을 반복합니다. 마지막에 energy를 읽고 force와 stress를 계산합니다.

## Slide 5. 재사용 연산값의 분류 (2.0 min)
지난 주 계획서의 첫 항목은 재사용 가능한 값을 분류하는 것이었습니다. 여기서 중요한 구분은 pair-shared geometry, parity로 복원 가능한 값, 방향별로 다시 계산해야 하는 message, 그리고 node/global reduction 결과입니다. 이 분류가 있어야 어디까지 exact하게 줄일 수 있는지가 보입니다.

## Slide 6. SevenNet 현재 구현과 목표 구조 (2.5 min)
현재 SevenNet 구현은 pair metadata를 만들고 geometry와 weight 일부를 재사용하지만, 최종 TP와 aggregation은 여전히 directed edge 기준입니다. 제가 최종적으로 노리는 구조는 pair-major layout을 오래 유지하고, 가능하면 pair-major TP까지 가는 구조입니다.

## Slide 7. 관련 연구 (2.0 min)
관련 연구는 세 범주만 보면 됩니다. 첫째는 FlashTP나 cuEquivariance처럼 tensor product 자체를 빠르게 만드는 backend 연구, 둘째는 graph/runtime 측면의 cache와 batching, 셋째는 distributed inference/runtime 연구입니다. 이번 발표는 이 세 범주 중 pair execution과 직접 연결되는 부분만 가져왔습니다.

## Slide 8. 실험 질문과 측정 항목 (1.5 min)
실험 질문은 간단합니다. 어떤 그래프에서 pair execution이 이득인지, 왜 그런지, 그리고 현재 구현이 줄이는 연산이 실제 전체 시간에서 얼마나 큰 비중을 차지하는지입니다. 그래서 latency, speedup, stage profiling, GPU utilization, 데이터셋 size-density를 같이 봤습니다.

## Slide 9. 데이터셋 맵 (2.0 min)
이 그림은 대표 샘플 기준으로 edge 수와 평균 이웃 수를 그린 것입니다. 발표용으로는 size와 density를 동시에 보게 하는 게 중요해서, large 기준은 대표 샘플 `num_edges >= 3000`, dense 기준은 `avg_neighbors >= 40`으로 잡았습니다.

## Slide 10. 네 구역 대표 비교 (2.5 min)
네 구역 대표는 small sparse로 SPICE 2023, small dense로 작은 periodic crystal 계열, large sparse로 OMol25 validation, large dense로 MPtrj validation을 잡았습니다. 이 비교가 중요한 이유는 pair execution의 성패가 크기만이 아니라 density와 edge load의 조합으로 결정된다는 점을 한 장에서 보여주기 때문입니다.

## Slide 11. Quadrant 결과 해석 (2.0 min)
결과를 보면 small sparse는 느리고, large periodic 쪽에서만 modest gain이 남습니다. 최신 정리에서는 small dense를 generic win으로 말하지 않고, large periodic graph에서 geometry-side reuse가 어느 정도 듣는다고 해석하는 것이 더 안전합니다. 즉 본질 변수는 atom 수 단독이 아니라 결국 edge load와 workload regime입니다.

## Slide 12. Baseline 상세 프로파일 (2.5 min)
baseline intrusive profile을 보면 small sparse에서는 message TP와 force_output이 지배적이고, large dense에서는 spherical harmonics까지 크게 올라옵니다. 반면 weight_nn과 gather, aggregation은 baseline에서도 상대적으로 작습니다. 그래서 현재 pair execution이 줄이는 부분만으로는 모든 그래프에서 큰 이득이 나기 어렵습니다.

## Slide 13. Lab benchmark 결과 (2.5 min)
랩 단일 GPU 환경에서 본 실제 결과를 요약하면, small sparse에서는 손해가 나고, large periodic 쪽에서만 몇 퍼센트 수준의 modest gain이 남습니다. 또 size-density map과 stage breakdown을 같이 보면, 왜 어떤 경우는 geometry reuse가 조금 듣고 어떤 경우는 TP와 force backward가 그대로 남아 발목을 잡는지가 같이 설명됩니다.

## Slide 14. Discussion (2.5 min)
현재 구현은 pair-major TP kernel이 아니라 geometry/SH/weight reuse 중심의 구현입니다. 그래서 speedup ceiling이 있고, small molecular에서는 손해가 나며 large periodic에서도 이득이 modest한 수준에 머뭅니다. 반대로 말하면 next step은 명확합니다. pair-major TP, backend별 policy, topology epoch cache, distributed schedule입니다.

## Slide 15. 지금까지 구현한 것과 남은 것 (2.0 min)
이미 구현한 것은 pair metadata, geometry reuse, weight reuse, baseline/pair profiling, dataset별 실험 인프라입니다. 남은 것은 pair-major TP, FlashTP와의 진짜 co-design, distributed path, 그리고 benchmark sanitation을 반영한 submission-grade evaluation입니다.

## Slide 16. 결론 (2.0 min)
정리하면 이번 단계의 가장 중요한 결론은 세 가지입니다. 첫째, pair execution을 논하려면 reusable value를 정확히 분류해야 합니다. 둘째, 현재 구현은 일부 edge-side work만 줄이기 때문에 small molecular에서는 손해가 나고 large periodic에서도 modest gain만 남습니다. 셋째, 다음 연구 포인트는 pair-major execution과 backend/runtime co-design입니다.

## Appendix. 데이터셋 확보 상태 (backup)
이 슬라이드는 Q&A용입니다. 현재는 대부분의 public large dataset이 로컬에 확보되어 있고, 남은 비공개 항목은 gated access가 필요한 `omol25_official_gated` 정도입니다. 따라서 앞으로의 핵심 병목은 데이터셋 확보보다는 benchmark sanitation, pair-major TP 구현, 그리고 LAMMPS end-to-end 검증입니다.
