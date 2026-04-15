# GNN-IP Pair Execution 제안 세미나 스크립트

## 2026-04-03 Note

- 현재 canonical repository state는 exact runtime reconstruction의 첫 단계로 보는 것이 맞다.
- 실제 구현된 것은 `pair-major TP`가 아니라 `pair-aware geometry-side reuse`다.
- `pair-major TP`, backend co-design, distributed exact pruning은 아직 future work다.
- 최신 canonical status는 `docs/papers/icpp_pair_execution/` 아래 문서들을 따른다.

## Slide 1. 제목
오늘 세미나는 구현 결과 보고가 아니라, GNN-IP 추론 실행 구조를 어떻게 다시 정의할지에 대한 제안 세미나입니다. 핵심 문제는 directed edge 기준 실행이 실제 물리적 상호작용 구조와 맞지 않아 중복 계산과 집계 오버헤드를 만든다는 점입니다.

## Slide 2. 왜 지금 이 문제를 보나
short-range MLIP는 그래프 이론 의미의 dense graph는 아니지만, local degree가 높고 기하적 규칙성이 강합니다. 이런 구조에서는 상호작용을 두 방향 간선으로 그대로 실행하는 것이 자연스러운 구현일 수는 있어도 효율적인 구현이라고 보기 어렵습니다.

## Slide 3. 현재 실행 경로
현재 실행은 edge 생성, 기하 feature 생성, weight_nn, tensor product, node reduction, 그리고 energy-force-stress 계산으로 이어집니다. 즉 edge 중심 생성과 node 중심 reduction이 분리되어 있고, 이 구조가 중복과 메모리 이동을 함께 키웁니다.

## Slide 4. 코드에서 확인한 병목
실제 코드에서도 pair metadata 생성, topology cache 검증, backend 선택이 아직 충분히 공격적으로 최적화되어 있지 않습니다. 그래서 단순히 연산 수를 줄이는 것만으로는 부족하고, control path와 backend path를 함께 재설계해야 합니다.

## Slide 5. 문제 정의
정리하면 문제는 세 가지입니다. 공유 가능한 기하 정보의 중복 생성, message 이후 node-centric reduction에 따른 메모리 비용, 그리고 분산 경로에서 과도하게 일반적인 통신 구조입니다.

## Slide 6. Exact Pair-Symmetric Execution
첫 번째 축은 pair를 기본 실행 단위로 보는 것입니다. geometry와 filter는 pair 기준으로 만들고, reverse 방향은 parity transform으로 복원합니다. 다만 source feature가 다르므로 message 자체는 공유하지 않는다는 점을 명확히 구분해야 합니다.

## Slide 7. Backend Co-Design
두 번째 축은 backend co-design입니다. 공통 옵션을 둔다고 해서 backend 내부까지 똑같이 실행하면 안 됩니다. 특히 FlashTP에서는 pair-major layout을 오래 유지하고, 실제 측정 기반으로 policy를 고르는 autotuning이 필요합니다.

## Slide 8. Topology-Epoch Caching
세 번째 축은 topology-epoch caching입니다. MD에서는 neighbor topology가 유지되는 구간이 있기 때문에, pair plan과 reduction schedule을 이 구간 단위로 재사용하는 것이 가능합니다. 핵심은 무엇을 cache하느냐보다 어떤 조건에서 invalidate하느냐입니다.

## Slide 9. Distributed Pair Schedule
네 번째 축은 distributed exactness입니다. ghost를 많이 들고 가는 구조 대신, pair-aware partial reduction과 scalar-only backward pruning이 가능하다면 통신 구조 자체를 바꿀 수 있습니다.

## Slide 10. 기대효과와 리스크
이 연구의 장점은 계산량, 메모리, backend utilization, distributed scaling을 함께 다룰 수 있다는 점입니다. 반면 pair metadata 생성이 새 병목이 되거나, backend별로 pair execution 이득이 상쇄될 수 있다는 위험도 분명히 있습니다.

## Slide 11. 검증 계획
검증은 정확성, 성능, ablation 세 축으로 가져갑니다. 특히 baseline과 수치적으로 같은지, 그리고 어떤 제안이 실제 속도 향상에 기여하는지를 분리해서 봐야 합니다.

## Slide 12. 핵심 메시지
결국 이 연구는 새 모델 제안이 아니라 exact runtime 재구성 연구입니다. 핵심 질문은 빠른 커널 하나가 아니라, 어떤 단위로 표현하고 언제 계산하며 무엇을 재사용할 것인가입니다.

## Slide 13. 마무리
단기적으로는 pair plan hot path와 backend autotuning을, 중기적으로는 FlashTP adapter와 topology-epoch runtime을, 장기적으로는 distributed exact pruning까지 포함한 시스템 논문 스토리를 목표로 가져가면 됩니다.
