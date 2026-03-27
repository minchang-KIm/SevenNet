# GNN-IP 커널 및 그래프 실행 구조 최적화 연구 노트

## 개요

이 문서는 short-range equivariant GNN 기반 interatomic potential(이하 GNN-IP)의 추론 실행 구조를 재구성하기 위한 연구 메모이다. 목적은 새로운 MLIP 모델을 설계하는 것이 아니라, 기존 모델의 수식 구조와 학습된 파라미터를 유지한 채 실행 경로를 바꾸어 중복 계산, 메모리 트래픽, 집계 비용, 분산 실행 오버헤드를 줄일 수 있는지 검토하는 데 있다.

핵심 문제는 다음과 같다.

- 현재 GNN-IP 추론은 동일한 물리적 상호작용을 양방향 directed edge로 처리하면서 공통으로 재사용 가능하거나 방향 반전에 따라 변환 가능한 기하 정보를 반복적으로 생성한다.
- edge별로 생성된 message는 다시 정점 기준으로 합산되어야 하므로 중간 데이터 저장과 reduction 비용이 추가된다.
- 분산 실행에서는 이러한 구조가 고스트 정보 교환과 토폴로지 의존 전처리 비용으로 이어질 수 있다.

## 코드 기준 사실

현재 `main` 기준 SevenNet 추론 경로에서 아래 구조는 코드상 확인된다.

- serial/parallel LAMMPS 경로는 모두 full neighbor list를 사용한다. 관련 구현은 `sevenn/pair_e3gnn/pair_e3gnn.cpp`, `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`에 있다.
- convolution은 directed edge별로 weight와 message를 만든 뒤 `message_gather(..., reduce='sum')`를 통해 node 기준으로 합산한다. 관련 구현은 `sevenn/nn/convolution.py`에 있다.
- parallel 경로는 `x_ghost`를 통신하여 다음 segment 계산에 사용한다. 관련 구현은 `sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp`에 있다.

따라서 이 연구는 다음과 같은 문제 정의를 갖는다.

- reverse directed edge에서 발생하는 기하 정보 생성의 중복
- message 생성과 node 기준 집계가 분리되어 발생하는 reduction 및 메모리 오버헤드
- 분산 실행에서 발생하는 고스트 정보 및 토폴로지 스케줄 처리 비용

## 연구 문제 정의

본 연구가 해결하고자 하는 핵심 문제는, 기존 equivariant GNN-IP의 추론이 하나의 물리적 상호작용을 두 개의 방향 간선으로 처리하면서 공유 가능한 기하 정보를 반복 생성하고, 이후 각 방향 간선에서 얻은 결과를 다시 정점 기준으로 집계하는 구조를 사용함으로써, 계산량과 메모리 접근 비용이 함께 증가한다는 점이다.

이 문제는 개별 커널의 속도 부족만으로 설명되지 않는다. 거리, radial basis, cutoff와 같은 방향 불변 항과, spherical harmonics와 같이 방향 반전에 따라 정해진 규칙으로 변환 가능한 항이 분리되지 않은 채 directed edge 단위로 반복 처리되기 때문이다. 따라서 tensor-product나 개별 CUDA 커널만 빠르게 만드는 방식으로는 중복 생성 자체를 제거할 수 없다.

## 제안 방향

### 1. 상호작용 공유형 실행 표현

하나의 물리적 상호작용을 기본 실행 단위로 두고, 여기서 공통으로 계산할 수 있는 항과 방향별로 달라지는 항을 분리한다. 거리, radial basis, cutoff는 상호작용 단위에서 한 번만 계산하고, 방향 정보는 같은 상호작용의 기준 방향 표현으로부터 재구성 가능한 범위를 명확히 정리한다.

이 방향의 목적은 directed edge를 없애는 것이 아니라, 두 방향 간선이 공유할 수 있는 정보를 한 번만 계산하도록 실행 구조를 바꾸는 것이다. 최종 message 자체는 각 방향에서 사용하는 source node feature가 다르므로 여전히 방향별 계산이 필요하지만, 그 이전 단계의 geometry/filter 생성 비용은 줄일 수 있다.

### 2. message 생성과 reduction의 결합

기존 구조에서는 edge별로 message를 생성한 뒤 이를 다시 node 기준으로 합산한다. 본 연구에서는 상호작용 공유형 표현을 기반으로 message 생성과 집계를 더 밀접하게 결합한 실행 구조를 검토한다.

중요한 점은 sum aggregation 자체를 제거하는 것이 아니라, 중간 message를 별도 배열에 저장한 뒤 나중에 다시 읽어 합산하는 과정을 줄이는 것이다. 즉, 연구의 초점은 reduction 제거가 아니라 reduction 비용 완화와 온라인 누산 구조의 가능성 검토에 있다.

### 3. topology-epoch 재사용

MD에서는 좌표는 매 step 변해도 neighbor topology가 일정 구간 유지될 수 있다. 이 경우 상호작용 단위 매핑, 방향 변환 메타데이터, sparse reduction schedule, 통신 인덱스 등 토폴로지 의존 정보는 반복 생성하지 않고 일정 구간 재사용할 가능성이 있다.

본 연구에서는 이 구간을 topology epoch로 정의하고, 어떤 메타데이터가 재사용 가능하며 어떤 조건에서 무효화되어야 하는지 분석한다. 이는 단순 캐시 적용이 아니라 MD 실행 구조와 결합된 실행 스케줄 재사용 원리 정립에 해당한다.

### 4. 분산 실행 확장

분산 환경에서는 기존 공간 분할 방식이 고스트 정보 교환에 따른 비용을 유발할 수 있다. 본 연구는 이를 단순히 고스트 제거 문제로 다루지 않고, 상호작용 공유형 실행 표현이 적용될 때 통신되는 정보의 종류와 양이 어떻게 달라지는지 평가하는 방향으로 접근한다.

즉, 이 연구의 분산 확장 항목은 “통신이 사라진다”는 주장보다 “중복 생성되던 정보와 보조 데이터 이동을 줄일 수 있는가”를 분석하는 데 초점을 둔다.

## 이론적 검증 계획

본 연구는 실행 구조를 바꾸면서도 기존 모델의 의미를 유지해야 하므로, 구현보다 먼저 이론적 검증이 필요하다. 수식 전개 대신 아래 세 가지 명제를 검토한다.

- 거리, radial basis, cutoff와 같이 방향 반전에 대해 불변인 항은 상호작용 단위에서 완전 재사용 가능하다.
- spherical harmonics와 같은 방향 의존 항은 동일한 basis/normalization 가정 아래에서 반대 방향 표현을 정해진 변환 규칙으로 재구성할 수 있다.
- 최종 message는 source node feature를 사용하므로 방향별 계산이 필요하지만, 기하 정보 생성과 집계 순서를 바꾸더라도 최종 node update의 의미는 유지된다. 이때 차이가 발생할 수 있는 부분은 부동소수점 합산 순서뿐이다.

이 세 명제가 성립하면, 본 연구는 새로운 모델을 제안하는 것이 아니라 기존 모델의 exact execution path를 재구성하는 시스템 연구로 정당화될 수 있다.

## 적용 범위

강하게 주장할 수 있는 범위는 다음과 같다.

- short-range
- cutoff 기반
- message-passing
- sum aggregation 사용
- pairwise relative geometry를 edge 입력으로 사용하는 equivariant MLIP

따라서 논문 claim은 “all MLIPs”보다 “short-range message-passing equivariant MLIPs” 수준으로 제한하는 것이 안전하다.

## 실험 계획

기본 구현과 평가는 SevenNet `main` branch를 기준선으로 수행한다. 가능하다면 NequIP 계열 모델 한 종류를 추가하여 일반성을 보강한다.

평가 항목은 다음과 같다.

- geometry/filter 생성 시간
- interaction block 단위 실행 시간
- end-to-end inference latency
- memory footprint 및 메모리 트래픽
- 평균 이웃 수와 시스템 크기에 따른 성능 변화
- topology epoch 재사용 효과
- energy, force, stress 차이
- 분산 환경에서의 scaling 및 통신 비용

ablation은 다음 순서로 구성한다.

- geometry/filter 재사용만 적용
- geometry/filter 재사용 + reduction 결합
- geometry/filter 재사용 + reduction 결합 + topology-epoch caching

## 현재 시점의 논문 포지셔닝

이 연구는 graph partitioning 논문보다는 exact execution model 논문에 가깝다. 중심 contribution은 다음처럼 정리하는 것이 가장 안전하다.

- reverse-edge duplication을 first-class inefficiency로 정의
- pair-symmetric exact execution 제안
- fused online reduction 가능성 제시
- topology-epoch caching 설계 원리 제시

즉, “새로운 MLIP 모델”이 아니라 “기존 short-range equivariant MLIP의 추론 실행 구조를 재구성하는 시스템 논문”으로 포지셔닝한다.

## 작업 메모

- 현재 표현에서 “분자”보다 “원자” 또는 “상호작용 단위”를 사용하는 것이 더 정확하다.
- “reduction 제거”가 아니라 “reduction 비용 완화” 또는 “온라인 누산”으로 써야 한다.
- "정확도 유지"는 "부동소수점 합산 순서 차이를 제외하면 기존 연산 의미 유지"로 표현하는 것이 안전하다.
- 분산 확장 항목은 "ghost 제거"보다 "중복 정보 및 보조 데이터 이동 완화"로 쓰는 편이 좋다.
- 구현 시에는 `pair_execution_config`로 `use`, `policy`, `fuse_reduction`, `use_topology_cache`, `distributed_schedule`, `backend_policy`를 노출해 backend별 실행 경로를 선택할 수 있게 한다.
