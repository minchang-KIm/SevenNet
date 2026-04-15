# 20260331 Seminar 김민창 rev2 대본

전체 분량 기준은 약 30분입니다.  
각 슬라이드의 시간은 질문 없이 발표를 이어갈 때의 기준입니다.

## 2026-04-03 Correction Note

- 이 대본은 세미나 시점의 내부 발표 버전이다.
- 이후 benchmark sanitation에서 small-graph case 일부에 warm-up artifact가 확인되었다.
- 현재 canonical 해석은 `geometry-side reuse`, `small molecular에서는 대체로 손해`, `large periodic에서는 modest gain`이다.

## Slide 1. Title (1.0 min)
안녕하세요. 오늘 발표에서는 `GNN-IP Inference Optimization Reconstruction`이라는 제목으로, 지난주 계획서에 적었던 문제의식을 실제 구현과 실험 결과까지 연결해서 말씀드리겠습니다.  

이번 발표의 핵심은 새로운 물리 모델을 제안하는 것이 아닙니다. 오히려 그 반대입니다. 이미 널리 쓰이는 equivariant interatomic potential의 원래 formulation은 그대로 유지하면서, 실제 실행 파이프라인 안에서 어떤 값이 중복 계산되고 있는지, 그리고 어떤 값은 수학적으로 정확하게 재사용할 수 있는지를 다시 구성해 보는 것이 목적입니다.  

즉 오늘 발표는 모델 아키텍처 자체의 novelty보다는, `실행 구조를 다시 이해하고 재배치해서 추론을 더 효율적으로 만들 수 있는가`라는 질문에 대한 중간 보고라고 보시면 됩니다.

## Slide 2. Table of Contents (1.0 min)
발표는 크게 네 부분으로 진행하겠습니다.  

먼저 앞부분에서는 GNN-IP 전체 맥락과 equivariant model pipeline을 짧게 설명하겠습니다. 이 부분은 작업자 외 다른 학생들도 흐름을 따라올 수 있도록 아주 기본적인 수준에서 정리하겠습니다.  

그 다음에는 문제 정의와 challenge, 그리고 관련 연구를 연결해서 왜 제가 `pair reconstruction`이라는 방향을 잡았는지 설명드리겠습니다.  

세 번째로는 proposed method 파트에서 실제로 무엇을 재사용 대상으로 분류했고, 지금 SevenNet 안에서는 어디까지 구현되어 있는지 말씀드리겠습니다.  

마지막으로는 experiment plan이라고 적혀 있지만, 실제로는 단일 GPU 환경에서 이미 돌려본 결과까지 함께 보여드리면서 discussion으로 마무리하겠습니다.

## Slide 3. GNN-IP R&D Proposal (2.0 min)
이 장은 제 세부 주제가 전체 프로젝트 안에서 어디에 위치하는지 보여주는 페이지입니다.  

지금 랩의 큰 로드맵을 보면 Year 3에서는 GNN-IP kernel level optimization과 graph computation optimization이 함께 들어가 있습니다. 여기서 kernel optimization이라고 하면 보통 tensor product를 먼저 떠올리기 쉽습니다. 실제로 FlashTP 같은 연구도 그 방향입니다. 그런데 실제 추론 시간을 들여다보면 tensor product 이전에도 radial basis, cutoff, spherical harmonics, neighbor-based indexing처럼 꽤 많은 작업이 있습니다.  

저는 이 앞단의 non-TP geometric computation과 input representation restructuring에 초점을 두고 있습니다. 다시 말해, 연산 하나를 더 빠르게 만드는 문제와 별개로, 애초에 같은 정보를 두 번 계산하고 있는 부분이 있다면 입력 흐름 자체를 바꿔서 그 중복을 없앨 수 있지 않느냐는 관점입니다.  

그래서 오늘 발표는 `TP kernel을 새로 만들었다`는 이야기가 아니라, `TP 이전 단계와 message passing 경계에서 exact reuse가 가능한 부분을 어떻게 찾았는가`에 더 가깝습니다.

## Slide 4. Runtime / Distributed Roadmap (1.5 min)
다음 페이지는 이 문제가 왜 단순한 micro-optimization으로 끝나지 않는지를 보여줍니다.  

연차 계획을 보면 이후에는 adaptive skipping policy, runtime optimization, multi-GPU, multi-node inference까지 이어집니다. 그러면 자연스럽게 질문이 생깁니다. 지금 하는 pair-based reconstruction이 나중에 distributed setting에도 도움이 되는 표현인가, 아니면 단일 GPU에서만 잠깐 쓰고 버리는 일회성 최적화인가 하는 점입니다.  

제가 보기에 이 작업의 의미는 단순히 SH를 줄이는 데 있지 않습니다. `그래프를 directed edge 기준으로 볼 것인지, undirected pair 기준으로 먼저 볼 것인지`라는 중간 표현을 정의하는 데 있습니다. 이 중간 표현이 정리되어야 이후에 topology cache, graph-specific schedule, distributed partitioning 같은 문제로도 자연스럽게 확장할 수 있습니다.  

그래서 발표 전체를 들으실 때도, 단순한 SH optimization이라기보다 `GNN-IP 실행 단위 자체를 다시 정의해보는 시도`라고 이해하시면 좋겠습니다.

## Slide 5. Equivariant Model Pipeline (2.5 min)
여기서는 아주 짧게 GNN-IP, 그중에서도 NequIP류 equivariant 모델의 추론 파이프라인을 정리하겠습니다.  

이 슬라이드는 그림이 세 부분으로 나뉘어 있으니, 왼쪽부터 순서대로 보겠습니다.  

왼쪽 그림은 모델 전체 파이프라인입니다. 원자종 `Z`는 embedding으로 들어가서 초기 node feature를 만들고, 원자 좌표 `r`은 각 interaction block에 기하 정보를 제공합니다. interaction block이 여러 번 반복된 뒤 output block과 global pooling을 거쳐 최종 에너지 `E`가 나옵니다. 즉 왼쪽 그림은 모델의 가장 큰 흐름을 보여주는 그림입니다.  

가운데 그림은 interaction block 하나를 펼쳐놓은 것입니다. 여기에는 self-interaction 경로와 convolution 경로가 같이 있습니다. self-interaction은 이웃과 무관한 node-wise 변환이고, convolution 경로가 실제로 이웃 정보를 받아 message passing을 수행하는 부분입니다. 두 경로가 합쳐지고 non-linearity를 거쳐 다음 block으로 넘어갑니다.  

오른쪽 그림은 그 convolution 내부를 더 자세히 그린 것입니다. 좌표로부터 먼저 두 종류의 edge-side 정보가 만들어집니다. 하나는 방향 정보를 담는 spherical harmonics이고, 다른 하나는 거리 norm에서 출발하는 basis와 MLP입니다. 그리고 이 정보들이 tensor product에 들어가 source node feature와 결합되면서 실제 equivariant message가 생성됩니다.  

여기서 중요한 점은, spherical harmonics를 만드는 단계 자체가 곧바로 message 생성은 아니라는 것입니다. 이 단계는 `message 생성을 위한 edge attribute를 준비하는 단계`입니다. 실제 message 생성은 그 다음 tensor product에서 일어납니다.  

그래서 오늘 이야기할 재사용 가능성도 정확히 이 경계에서 나옵니다. distance, radial basis, cutoff, spherical harmonics 같은 geometry-side 값은 pair 기준으로 재사용 가능성이 있지만, 최종 message는 source node feature에 의존하므로 그대로 공유할 수 없습니다. 이 구분이 뒤의 제안기법과 실험 해석의 핵심입니다.

## Slide 6. Problem (2.5 min)
이제 본격적인 문제 정의입니다.  

기존 equivariant GNN-IP 구현은 대부분 directed edge 기준으로 계산합니다. 그러면 원자 i와 j가 있을 때, `i에서 j로 가는 edge`와 `j에서 i로 가는 edge`는 내부적으로 별도의 두 샘플처럼 취급됩니다. 이 방식은 구현이 단순하고 기존 message passing 프레임워크와도 잘 맞지만, 물리적으로 보면 두 edge는 같은 pair에서 나온 정보입니다.  

예를 들어 distance는 두 방향에서 완전히 같습니다. radial basis와 cutoff도 같습니다. spherical harmonics는 완전히 같은 값은 아니지만, 역방향으로 갔을 때 degree l에 따른 parity sign으로 복원 가능합니다. 다시 말해 `새로 계산하지 않아도 되는 값`이 이미 존재합니다.  

반면 모든 것이 재사용 가능한 것은 아닙니다. message는 source node feature를 사용하므로 `i to j`와 `j to i`에서 서로 다릅니다. aggregation도 destination이 다르기 때문에 단순 공유가 안 됩니다.  

그래서 제가 지난주 계획서에서 가장 먼저 잡았던 일이 `재사용 가능한 연산값의 분류`였습니다. 이것이 먼저 정리되어야, 원래 formulation을 깨지 않고 exact reuse가 가능한 경계가 보이기 때문입니다.

## Slide 7. Challenge 1 (2.0 min)
첫 번째 challenge는 structural inefficiency, 즉 구조적 비효율입니다.  

겉으로 보면 directed edge 두 개를 계산하는 것이 별문제 없어 보일 수 있습니다. 하지만 pair 관점에서 보면 같은 geometric information을 두 번 encoding하고 있는 셈입니다. 특히 neighbor 수가 많고 graph가 클수록 이 중복은 절대량으로 커집니다.  

그렇다고 단순히 한쪽 edge를 없애면 되는 문제는 아닙니다. 모델의 원래 의미를 유지해야 하기 때문입니다. 예를 들어 SH는 reverse edge에서 sign flip으로 복원할 수 있지만, message는 source node가 달라지므로 없앨 수 없습니다. 따라서 `줄일 수 있는 것`과 `남겨야 하는 것`을 정확히 구분해야 하고, 이 구분이 어긋나면 모델의 수학적 의미가 깨집니다.  

즉 challenge는 단순한 deduplication이 아니라, `exactness를 유지하는 형태로 computation graph를 재구성하는 것`입니다.

## Slide 8. Challenge 2 (2.0 min)
두 번째 challenge는 backend integration입니다.  

예를 들어 FlashTP는 tensor product를 fused kernel로 아주 잘 가속합니다. 하지만 그 인터페이스를 보면 여전히 directed edge 배열을 입력으로 받습니다. 즉 pair-based representation을 만들더라도, 마지막에 다시 directed edge로 풀어버리면 기대했던 이득이 줄어듭니다.  

실제로 제가 구현하면서 확인한 것도 바로 이 부분입니다. Python 레벨에서 pair metadata를 만들고 SH나 weight를 재사용하는 것 자체는 가능합니다. 하지만 backend가 pair-major execution을 이해하지 못하면, 중간에 다시 edge layout으로 펼쳐야 하고 그 과정에서 indexing overhead가 생깁니다.  

그래서 pair reconstruction의 난점은 알고리즘 아이디어보다도 `어디까지를 pair 기준으로 유지할 수 있는가`, 그리고 `backend와 어떤 계약을 새로 만들어야 하는가`에 있습니다. 여기서부터는 단순 코드 수정이 아니라 runtime co-design 문제가 됩니다.

## Slide 9. Related Work (2.5 min)
관련 연구는 세 갈래만 보면 충분합니다.  

첫 번째는 FlashTP나 cuEquivariance 같은 `tensor product acceleration` 계열입니다. 이들은 TP와 scatter-gather를 매우 효율적으로 처리하고, 시스템 수준에서 큰 개선을 가져옵니다. 하지만 upstream geometry redundancy, 즉 i to j와 j to i에서 중복되는 SH나 radial computation은 거의 다루지 않습니다.  

두 번째는 SevenNet 같은 `system-level MLIP acceleration`입니다. 여기서는 multi-GPU, domain decomposition, 실제 MD에 가까운 실행 인프라가 중요합니다. 그런데 이 역시 계산 단위 자체는 대체로 directed edge를 전제하고 있습니다.  

세 번째는 NEMP처럼 `algorithm 자체를 바꾸는 방식`입니다. NEMP는 node space에서 message passing을 다시 구성해 효율을 얻습니다. 다만 이 방식은 원래 edge-based equivariant formulation을 그대로 유지하는 접근은 아닙니다.  

그래서 제가 잡은 위치는 이 셋의 중간쯤입니다. 즉 formulation은 유지하되, geometry redundancy를 줄이는 exact reuse를 노리는 방향입니다. 제가 찾은 범위에서는 이 지점을 정면으로 다루는 선행연구가 상대적으로 적었습니다.

## Slide 10. Proposed Method Motivation (2.0 min)
이 장은 proposed method의 수학적 동기를 설명하는 슬라이드입니다.  

E3-equivariant 모델은 rotation, translation, reflection에 대해 정해진 방식으로 반응해야 합니다. translation은 relative coordinate를 쓰는 순간 자연스럽게 처리됩니다. rotation은 spherical harmonics와 irreps 표현을 통해 다뤄집니다. 그리고 reflection은 degree l에 따른 parity로 다뤄집니다.  

이 점이 중요한 이유는 reverse edge가 새 정보를 주는 것이 아니라는 데 있습니다. 같은 pair에 대해 방향만 반대로 뒤집으면, geometry는 essentially 동일하고 SH는 parity sign으로 연결됩니다. 그래서 한쪽 방향의 SH를 계산해두면 반대 방향은 expensive evaluation 없이 sign 처리로 복원할 수 있습니다.  

즉 제가 제안하는 sign-flip reuse는 경험적 heuristic이 아니라, equivariant formulation 안에서 이미 허용된 symmetry property를 execution level에서 이용하자는 것입니다.

## Slide 11. Reason Why choose this (2.5 min)
이 페이지를 따로 넣은 이유는, 왜 여러 reusable candidate 중에서 SH를 특히 강조하는지 분명히 말하기 위해서입니다.  

첫 번째 이유는 수학적으로 가장 깔끔하기 때문입니다. distance나 radial basis는 아예 동일하고, SH는 reverse 방향에서 parity sign으로 정확히 복원됩니다. 즉 approximate reuse가 아니라 exact reuse가 가능합니다.  

두 번째 이유는 profiling 결과입니다. 여기서 과장하지 않는 것이 중요합니다. 모든 데이터셋에서 SH가 지배적인 병목은 아닙니다. 실제 baseline intrusive profiling을 해보면 small sparse 계열에서는 SH 비중이 1퍼센트도 안 되는 경우가 많습니다. 반대로 large dense case, 예를 들어 MPtrj나 MD22 nanotube 같은 경우에는 SH가 model total time의 약 18퍼센트에서 26퍼센트를 차지했습니다. 평균만 내면 낮아 보일 수 있지만, 우리가 실제로 최적화 타깃으로 삼는 large edge-load regime에서는 분명히 substantial한 부분입니다.  

세 번째 이유는 해석이 명확하다는 점입니다. TP를 건드리면 backend, kernel, autograd 문제가 한꺼번에 커집니다. 반면 SH reuse는 `compute once, reverse by sign`이라는 설명이 분명하고, 실험 결과도 그 방향으로 잘 해석됩니다.  

그래서 이 페이지의 메시지는 단순합니다. `SH가 언제나 제일 느리다`가 아니라, `SH는 exact reuse가 가능하고, 큰 dense graph에서는 실제로 무시할 수 없는 비용이기 때문에 첫 번째 공략 대상으로 적합하다`입니다.

## Slide 12. Proposed Method (3.0 min)
이제 실제 구현 내용을 설명드리겠습니다.  

현재 제가 구현한 방식은 먼저 reverse 관계에 있는 두 directed edge를 하나의 undirected pair로 묶는 metadata를 만드는 것입니다. 여기에는 pair의 canonical 방향, reverse 방향, 각 directed edge가 어느 pair에 속하는지 같은 정보가 들어 있습니다.  

그 다음 geometry 단계에서는 pair 기준으로 distance, radial basis, cutoff, spherical harmonics를 한 번만 계산합니다. reverse edge의 SH는 새로 평가하지 않고 parity sign으로 복원합니다. 그리고 weight neural network도 pair 기준으로 한 번 계산해서 양방향에서 공유합니다.  

여기까지는 pair-aware reuse가 실제로 들어간 부분입니다. 하지만 중요한 제한이 있습니다. 최종 tensor product와 aggregation은 아직 directed edge 기준으로 남아 있습니다. 이유는 message가 source node feature에 의존하기 때문입니다. `i to j`와 `j to i`는 geometry는 공유해도 source가 다르므로 최종 message를 하나로 합칠 수 없습니다.  

그래서 현재 구현은 `pair-major full execution`이 아니라 `geometry, SH, pair-weight reuse`에 가까운 구조입니다. 이 차이를 정확히 말하는 것이 중요합니다. 지금 단계에서 얻는 speedup은 이 reusable part에서 오고, TP 자체를 줄여서 얻는 것은 아닙니다. 이 한계가 뒤의 실험 결과와도 정확히 연결됩니다.

## Slide 13. Experiment Plan (3.5 min)
이 슬라이드는 제목은 experiment plan이지만, 실제로는 계획과 현재 진행 결과를 같이 설명하는 장입니다.  

원래 계획은 세 가지였습니다. 첫째, SevenNet e3nn baseline과 SevenNet plus pair execution full을 직접 비교해서 `FlashTP가 없다고 가정했을 때` pair reuse만으로 어떤 이득이 있는지 보자는 것이었습니다. 둘째, FlashTP reference를 함께 두고, 현재 구현이 backend와 잘 결합되는지, 아니면 geometry reuse까지만 듣는지를 확인하자는 것이었습니다. 셋째, 데이터셋을 크기와 density 관점에서 나눠서 언제 효과가 나는지 해석하자는 것이었습니다.  

그래서 workload도 단순히 분자냐 고체냐로 나누지 않고, small sparse, small dense, large sparse, large dense 네 구역으로 보았습니다. small sparse 쪽에는 SPICE 2023, ANI-1x 같은 샘플을 넣었고, small dense에는 작은 periodic crystal 계열을 넣었습니다. large sparse에는 OMol25 validation이나 일부 OC20 계열을, large dense에는 MPtrj와 MD22 nanotube, OMat24 쪽을 대표로 봤습니다.  

평가 지표도 latency 하나만 본 것이 아니라, energy-force exactness, cold latency, steady latency, stage profiling, GPU utilization, size-density map을 함께 봤습니다. 이유는 단순히 빨라졌다, 느려졌다만으로는 왜 그런지 설명이 안 되기 때문입니다.  

현재까지 정리된 단일 GPU 결과를 요약하면 이렇습니다. e3nn baseline 대비 pair execution은 universal win이 아닙니다. small sparse에서는 대체로 손해이고, large periodic 쪽에서는 modest gain 또는 break-even이 나옵니다. 대표적으로 stable recheck 이후에는 OC20, sAlex, OMat24, MPtrj 같은 큰 periodic graph에서만 소폭 이득이 남고, SPICE 같은 small sparse에서는 손해가 났습니다.  

즉 계획 단계에서 세웠던 가설, 즉 `크기와 density를 같이 봐야 한다`는 가설은 현재 단일 GPU 결과로 부분적으로 지지되고 있습니다. 다만 최신 해석은 `large dense면 무조건 확실한 승리`가 아니라 `large periodic에서만 몇 퍼센트 수준의 이득이 남는다`는 쪽에 더 가깝습니다.

## Slide 14. Discussion (2.0 min)
마지막 discussion입니다.  

이번 단계에서 가장 중요하게 얻은 결론은 세 가지입니다. 첫째, pair execution을 이야기하려면 reusable value를 먼저 정확히 분류해야 합니다. 이걸 하지 않으면 무엇이 exact reuse인지, 무엇이 source-dependent인지가 섞여서 논의가 흐려집니다.  

둘째, 현재 구현은 pair-major TP가 아닙니다. geometry, SH, weight reuse는 들어갔지만 TP와 aggregation은 directed edge 기준으로 남아 있습니다. 그래서 small sparse에서는 이득이 잘 안 나고, large periodic에서도 이득이 크지 않고 modest한 수준에 머뭅니다. 이 점을 솔직하게 말해야 결과가 과장되지 않습니다.  

셋째, 다음 연구 방향은 꽤 분명합니다. pair-major TP kernel, backend co-design, topology cache, 그리고 distributed setting으로의 확장입니다. 다시 말해 지금 단계는 끝이 아니라, `어디까지 exact reuse가 가능하고 어디서부터 새 커널이 필요한지`를 분명히 한 첫 단계라고 볼 수 있습니다.  

이상으로 발표를 마치겠습니다.

## Appendix. Dataset Status (backup)

현재는 대부분의 public large dataset이 로컬에 확보되어 있고, 남은 비공개 항목은 gated access가 필요한 `omol25_official_gated` 정도입니다. 따라서 병목은 더 이상 데이터셋 확보 자체가 아니라, benchmark sanitation, pair-major TP 구현, 그리고 LAMMPS end-to-end 검증으로 옮겨갔다고 보는 것이 맞습니다.
