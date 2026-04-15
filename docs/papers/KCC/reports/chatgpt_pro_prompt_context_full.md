# KCC Pair-Geometry Reuse 연구 컨텍스트 패키지

이 문서는 지금까지 SevenNet 코드베이스에서 수행한 구현, 실험, 해석, 열린 가설, 후속 실험 아이디어를 **한 번에 복사해서 상위 모델에게 넘길 수 있도록** 정리한 장문 메모다.  
목적은 세 가지다.

1. 지금까지 무엇을 구현했고 무엇이 아직 구현되지 않았는지 정확히 전달
2. 어떤 실험을 이미 했고, 무엇이 사실로 확인됐는지 구분
3. 낙관적인 가능성을 유지하되, 과장 없이 다음 연구 방향을 설계할 수 있게 함

이 문서는 **논문 본문 초안이 아니라 연구 컨텍스트 패키지**다.  
따라서 이미 검증된 사실, 현재 코드 기준 분석, 아직 검증되지 않은 가설, 추천 실험 설계를 분리해서 적는다.

---

## 1. 연구 한 줄 요약

- 대상: **NequIP/SevenNet 계열 equivariant GNN interatomic potential**
- 문제: 하나의 원자쌍 `(i, j)`가 directed edge `i -> j`, `j -> i` 두 개로 중복 표현되어 **geometry-side 계산이 반복**됨
- 현재 구현: **pair-aware geometry-side reuse runtime**
- 아직 아님: **완전한 pair-major message passing / fused pair-major kernel**
- 핵심 질문:
  - 어떤 계산이 실제로 재사용 가능한가
  - 어떤 조건에서 속도 이득이 나는가
  - 왜 현재 구현은 small graph에서 손해가 나고, large/dense graph에서만 개선되는가
  - `lmax`가 커질수록 이 방향의 가치가 더 커지는가

---

## 2. 현재 구현의 정확한 정의

### 2.1 현재 구현이 하는 일

현재 제안기법은 다음 값을 **pair 기준으로 한 번** 계산하거나 재사용한다.

- 거리 `|r_ij|`
- radial basis
- cutoff
- spherical harmonics
- pair-level `weight_nn` 입력

핵심 구현 파일:

- pair metadata: [pair_runtime.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_runtime.py)
- pair edge embedding: [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)
- pair-aware convolution path: [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)

### 2.2 현재 구현이 아직 하지 않는 일

현재 구현은 다음을 **아직 하지 않는다**.

- pair-major tensor product 1-pass 실행
- pair-major reduction / aggregation
- pair-aware custom backward
- FlashTP와의 완전한 layout-level 결합
- upstream neighbor builder(LAMMPS/TorchSim)에서 reverse pair 정보를 직접 받아 재사용하는 구조

즉 현재 구현은 다음 식으로 보는 게 맞다.

```text
Baseline:
directed-edge geometry 2회 + edge-major conv + generic backward

Current proposal:
pair geometry 1회 + pair weight 1회 + edge-major-ish conv + generic backward

Not yet implemented:
pair geometry 1회 + pair-major conv 1회 + pair-aware backward
```

### 2.3 현재 구현을 “SH만 최적화”라고 부르면 안 되는 이유

현재 구현은 SH만 건드리는 것이 아니다. 실제로는 아래 전체가 들어간다.

- pair metadata 생성
- pair distance / radial / cutoff / SH
- reverse SH sign 복원
- pair weight 생성
- current full path에서는 forward/reverse 경로 스케줄도 바뀜

따라서 실험 결과를 설명할 때도 “SH 최적화의 성능”이라고 부르면 안 되고,  
정확히는 **geometry-side reusable terms의 exact reuse**라고 쓰는 것이 맞다.

---

## 3. 코드 기준 파이프라인 설명

### 3.1 baseline SevenNet 추론 흐름

```text
Atoms
-> graph build
-> EDGE_VEC / EDGE_IDX
-> edge embedding
   - distance
   - radial basis
   - cutoff
   - spherical harmonics
-> interaction blocks
   - self interaction
   - convolution
   - gate
-> readout
-> total energy
-> force/stress output
   - autograd.grad(E, EDGE_VEC)
```

중요한 코드 경로:

- edge embedding: [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)
- interaction/convolution: [interaction_blocks.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/interaction_blocks.py), [convolution.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/convolution.py)
- force output: [force_output.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/force_output.py)

### 3.2 current full path 흐름

```text
Atoms
-> graph build
-> prepare_pair_metadata
-> pair edge embedding
   - pair distance
   - pair radial basis
   - pair cutoff
   - pair SH
   - reverse SH = sign flip
-> pair weight_nn
-> current full conv
   - 현재는 완전 pair-major가 아니라 edge-major 성격이 남아 있음
-> readout
-> total energy
-> force/stress backward
```

### 3.3 backward는 무엇을 다시 지나가나

힘 계산은 마지막 MLP만 다시 계산하는 것이 아니다.

- [model_build.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/model_build.py)에서 `ForceStressOutputFromEdge`를 붙임
- [force_output.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/force_output.py)에서 `torch.autograd.grad(energy, [EDGE_VEC])`

즉 backward는 readout만이 아니라 아래 전체를 다시 지난다.

- readout
- interaction block
- convolution
- `weight_nn`
- geometric path

이 점 때문에 `forward-only`와 `force 포함 step`의 성능 해석을 분리해야 한다.

---

## 4. pair metadata 분석

### 4.1 pair metadata가 무엇인가

pair metadata는 directed edge 목록에서 아래를 만드는 단계다.

- `EDGE_PAIR_MAP`
- `PAIR_EDGE_FORWARD_INDEX`
- `PAIR_EDGE_BACKWARD_INDEX`
- `PAIR_EDGE_HAS_REVERSE`

즉 “이 directed edge 둘이 같은 undirected pair인지”를 복원하는 단계다.

### 4.2 왜 병목이었나

이전 구현은:

- CPU 이동
- Python for-loop
- tuple/string key
- dict/hash lookup

기반이었다.

### 4.3 지금까지 확인한 개선 가능성

`pair_metadata_summary.csv` 기준:

- `cpu_original` median: `55.866 ms`
- `cpu_vectorized` median: `10.019 ms`
- `gpu_vectorized_kernel_only` median: `0.730 ms`

파일:

- [pair_metadata_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_validation_split/global/pair_metadata_summary.csv)

즉 pair metadata는 “무시 가능한 부수 비용”이 아니라 **실제 병목**이다.

### 4.4 현재 상태

이미 일부 개선을 적용했다.

- pair matching 본체는 vectorized/device 경로로 이동
- 계산기 경로에서 device 올린 뒤 pair metadata 생성

하지만 아직 완전한 device-only는 아니다.

- topology signature / cache 판단은 CPU 쪽이 남아 있음

따라서 현재 상태는:

- **pair matching 본체: 많이 개선**
- **cache/signature: 아직 잔존**

---

## 5. LAMMPS / TorchSim / ASE 경로 분석

### 5.1 ASE

현재 논문 메인 실험의 중심은 ASE calculator 경로다.

- graph를 만든 뒤
- `prepare_pair_metadata()`를 별도 호출

즉 upstream neighbor 정보를 직접 pair-aware하게 주는 구조는 아니다.

### 5.2 LAMMPS

LAMMPS pair style은 full neighbor list를 직접 받는다.

- serial: [pair_e3gnn.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn.cpp)
- parallel: [pair_e3gnn_parallel.cpp](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp)

하지만 current code는:

- LAMMPS neighbor list를 사용해 edge를 만들고
- 그 뒤 reverse pair를 **다시 복원**한다

즉 neighbor list는 활용하지만, pair 정보를 upstream에서 그대로 받는 구조는 아니다.

### 5.3 TorchSim

TorchSim은 optional dependency다.

- 문서: [torchsim.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/source/user_guide/torchsim.md)

현재 wrapper는 TorchSim neighbor list를 쓰지만,

- `prepare_pair_metadata()`를 붙이지 않았고
- pair execution 경로도 사실상 연결되지 않았다

즉 TorchSim은 **잠재력은 크지만 아직 pair-aware 활용이 거의 없는 상태**다.

### 5.4 결론

현재 구현은 아직 아래 수준이 아니다.

- upstream neighbor builder가 reverse edge map/pair id를 직접 주고
- downstream이 그걸 그대로 쓰는 구조

따라서 “LAMMPS/TorchSim neighborlist를 최대한 활용했다”고 쓰면 과장이다.

보다 정확한 표현:

- **현재는 generic edge graph 위에서 pair metadata를 후처리로 구성하는 구조**
- **upstream pair-aware interface는 후속 과제**

---

## 6. FlashTP와의 관계

### 6.1 FlashTP가 실제로 최적화하는 것

FlashTP는:

- fused tensor product
- gather/scatter fusion
- Clebsch-Gordan(CG) 희소 경로 skip

를 최적화한다.

코드:

- [flash_helper.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/flash_helper.py)
- [flashTP.py](/home/wise/miniconda3/lib/python3.12/site-packages/flashTP_e3nn/flashTP.py)

### 6.2 FlashTP가 하지 않는 것

FlashTP는 **SH 계산 자체를 줄이지 않는다.**

SH는 여전히:

- [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)

에서 계산된다.

### 6.3 흔한 오해 정리

“FlashTP에서 삼각법에 맞지 않는 70% 이상을 안 계산한다”는 식의 설명은 정확하지 않다.

정확히는:

- **SH 함수의 삼각함수 항을 건너뛰는 것**이 아니라
- **TP 내부의 CG sparse path를 건너뛰는 것**

이다.

### 6.4 현재 연구와의 관계

따라서 현재 연구와 FlashTP는:

- 같은 부분을 두 번 최적화하는 것이 아니라
- **서로 다른 계층을 최적화하는 상보적 방법**

이다.

정리:

- 우리 방법: **geometry-side**
- FlashTP: **TP-side**

다만 지금 current full path는 pair-major가 아니므로,
FlashTP와 결합했다고 자동으로 좋아지는 단계는 아니다.

---

## 7. 정확도 검증

### 7.1 핵심 메시지

논문의 중요한 전제는:

- **정확도를 바꾸지 않고 실행 시간만 바꾸는 것**

이다.

이를 확인하기 위해 기준 출력 대비 반복 실행 차이를 측정했다.

### 7.2 결과

`pair_accuracy_summary.csv` 기준:

- 에너지 차이 중앙값: baseline/proposal 모두 `0 eV`
- 힘 차이 중앙값:
  - baseline: `1.204e-06 eV/A`
  - proposal: `1.867e-06 eV/A`
- 최악의 에너지 차이: `1.221e-04 eV`
- 최악의 힘 차이:
  - baseline: `1.526e-04 eV/A`
  - proposal: `2.441e-04 eV/A`

파일:

- [pair_accuracy_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_accuracy/global/pair_accuracy_summary.csv)
- [pair_accuracy_energy_errorbar.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_accuracy/pair_accuracy_energy_errorbar.png)
- [pair_accuracy_force_errorbar.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_accuracy/pair_accuracy_force_errorbar.png)

### 7.3 해석

이 차이는 기본 방식 자체의 반복 잡음과 같은 수준이다.

즉 현재 제안기법은:

- **출력을 바꾸는 기법이 아니라**
- **실행 시간을 바꾸는 기법**

으로 해석할 수 있다.

---

## 8. 메인 성능 결과: SevenNet baseline vs proposal

### 8.1 메인 비교 기준

현재 논문의 본선 비교는 FlashTP가 아니라:

- `SevenNet baseline`
- `SevenNet + 제안기법`

이다.

메인 파일:

- [pair_end_to_end_comparison.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv)

### 8.2 전체 결과

31개 benchmarkable dataset 기준:

- median speedup: `1.0105x`
- geometric mean: `0.8571x`
- wins: `18`
- losses: `13`

### 8.3 조건별 결과

- `num_edges >= 3000`
  - count: `20`
  - wins: `18`
  - median speedup: `1.0132x`

- `num_edges < 3000`
  - count: `11`
  - wins: `0`
  - median speedup: `0.6135x`

- `avg_neighbors_directed >= 40`
  - count: `17`
  - wins: `16`
  - median speedup: `1.0131x`

- `avg_neighbors_directed < 40`
  - count: `14`
  - wins: `2`
  - median speedup: `0.6146x`

### 8.4 대표적인 win / loss

상위 개선:

- `md22_buckyball_catcher`: `1.0668x`
- `oc20_s2ef_val_ood_ads`: `1.0651x`
- `oc20_s2ef_train_20m`: `1.0546x`
- `omol25_train_neutral`: `1.0285x`
- `salex_train_official`: `1.0274x`

대표 손해:

- `md22_dha`: `0.5977x`
- `md22_ac_ala3_nhme`: `0.5992x`
- `md22_at_at`: `0.6073x`
- `qm9_hf`: `0.6117x`
- `rmd17`: `0.6135x`
- `iso17`: `0.6135x`
- `spice_2023`: `0.6141x`

### 8.5 방어 가능한 메인 메시지

- 모든 경우에 빠르다고 주장하면 안 됨
- **큰 graph / 높은 neighbor count 조건에서 유리**
- **작은 graph에서는 손해**

이 조건 정의 자체가 컨트리뷰션이 될 수 있다.

---

## 9. 분리 검증: baseline / geometry_only / full

메인 성능 결과만으로는 원인 분리가 되지 않아서, 아래 분리 검증을 추가했다.

비교 케이스:

- `baseline`
- `geometry_only`
- `full_legacy`
- `full_no_expand`

측정 모드:

- `forward_energy`
- `step_force`

### 9.1 핵심 수치

중앙값 기준:

- `baseline / geometry_only`
  - `forward_energy`: `0.977x`
  - `step_force`: `0.684x`

- `baseline / full_legacy`
  - `forward_energy`: `0.865x`
  - `step_force`: `0.696x`

- `baseline / full_no_expand`
  - `forward_energy`: `0.862x`
  - `step_force`: `0.694x`

- `full_legacy / full_no_expand`
  - `forward_energy`: `0.996x`
  - `step_force`: `0.998x`

파일:

- [pair_validation_split_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pair_validation_split_report.md)
- [pair_validation_interpretation.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pair_validation_interpretation.md)

### 9.2 해석

이 결과는 매우 중요하다.

1. `geometry_only`는 pure forward에서는 거의 본전이다
2. force를 포함하면 크게 나빠진다
3. current `full`은 pure forward에서도 느리다
4. `expand 제거` 패치 하나는 거의 효과가 없다

즉:

- geometry-side 재사용 아이디어 자체가 완전히 틀린 것은 아니다
- current full path가 구조적으로 좋지 않다
- 실제 step에서는 force/backward가 큰 부분을 차지한다

---

## 10. `lmax` 해석

### 10.1 `lmax`는 무엇인가

`lmax`는 **spherical harmonics의 최대 차수 `l`**가 맞다.

코드:

- [edge_embedding.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/nn/edge_embedding.py)  
  `Irreps.spherical_harmonics(lmax, parity)`

하지만 SevenNet에서는 기본 설정상 이 값이 **SH에만** 쓰이는 게 아니라:

- `lmax_edge`
- `lmax_node`

로 연결되며, `-1`이면 전역 `lmax`를 따른다.

코드:

- [model_build.py](/home/wise/minchang/DenseMLIP/SevenNet/sevenn/model_build.py)

### 10.2 왜 `lmax`가 커질수록 비용이 빨리 커지나

SH 출력 차원은:

```text
sum_{l=0}^{L}(2l+1) = (L+1)^2
```

즉 `lmax=1,2,3,4,...`로 올리면 SH 차원이 제곱으로 증가한다.

하지만 실제 전체 비용은 SH 차원 증가보다 더 빨리 커질 수 있다. 이유는:

- hidden irreps
- edge irreps
- output irreps

사이의 **tensor product path 수**가 같이 증가하기 때문이다.

즉 “숨겨진 차원 + 에지 차원 + 출력 차원의 조합 + 1”이라는 표현은,
`lmax` 정의 자체가 아니라 **비용이 커지는 메커니즘**을 설명하는 말로 이해해야 한다.

### 10.3 `lmax`가 높을수록 정확도가 무한정 좋아지나

아니다. 실제로 baseline-only 학습 sweep을 돌려서 확인했다.

대상:

- `rMD17 azobenzene`

실험:

- same baseline architecture
- `lmax = 1..8`
- accuracy, training time, inference latency, stage profile 모두 저장

핵심 결과:

- best force RMSE: `lmax=7`, `0.136777 eV/A`
- `lmax=8`: `0.147295 eV/A`로 다시 약간 나빠짐
- `lmax=1 -> 8`
  - step latency: `5.089 -> 52.680 ms` (`10.35x`)
  - training time: `18.95 -> 1391.69 s` (`73.44x`)
  - params: `32192 -> 555456` (`17.25x`)

파일:

- [lmax_sweep_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_sweep_summary.csv)
- [lmax_baseline_sweep_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/reports/lmax_baseline_sweep_report.md)

따라서 방어 가능한 메시지:

- `lmax`는 데이터셋이 정해주는 값이 아니라 설계자가 정하는 하이퍼파라미터
- 하지만 최적 `lmax`는 데이터셋/학습조건이 결정
- 높은 `lmax`가 항상 더 정확한 것은 아니다

---

## 11. `lmax`와 dataset 크기/밀도 조합

대표 4개 graph를 택해 baseline runtime만 따로 측정했다.

- `small_sparse`: `spice_2023`
- `small_dense`: `salex_val_official`
- `large_sparse`: `oc20_s2ef_train_20m`
- `large_dense`: `mptrj`

파일:

- [lmax_quadrant_runtime_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/reports/lmax_quadrant_runtime_report.md)
- [quadrant_lmax_latency_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_latency_summary.csv)
- [quadrant_lmax_stage_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_stage_summary.csv)
- [quadrant_lmax_growth_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_growth_summary.csv)

### 11.1 핵심 결과

`lmax=1 -> 8`에서 model total 증가 배수:

- `small_sparse`: `9.72x`
- `small_dense`: `85.43x`
- `large_sparse`: `59.34x`
- `large_dense`: `182.21x`

SH 증가 배수:

- `20.56x ~ 24.56x`

TP 증가 배수:

- `22.37x ~ 128.35x`

force output 증가 배수:

- `12.78x ~ 325.05x`

### 11.2 해석

이 결과가 말하는 건 분명하다.

- `lmax`가 커질수록 SH도 커진다
- 하지만 large/dense graph에서 전체 비용 폭증의 주범은 **SH만이 아니라**
  - `TP`
  - `force backward`
  이다

즉 “높은 `lmax`에서 SH만 줄이면 전체가 크게 좋아진다”고 쓰면 과장이다.

맞는 해석:

- geometry-side 절감 가치는 높아진다
- 하지만 전체 wall-clock을 지배하는 것은 TP/backward까지 합친 구조다

---

## 12. 현재 데이터가 지지하는 strongest message

아래 문장들은 현재 데이터로 방어 가능하다.

1. 현재 구현은 **정확도를 바꾸기보다 실행 시간을 바꾸는 최적화**다.
2. 현재 구현은 **SH만이 아니라 geometry-side reusable term 전체의 exact reuse**다.
3. 모든 graph에서 빨라지지 않는다.
4. **큰 graph / 높은 연결 수 조건**에서만 일관된 개선이 나타난다.
5. current full path는 아직 pair-major가 아니며 구조적으로 느리다.
6. pair metadata는 현재 실제 병목이다.
7. `lmax`는 설계 하이퍼파라미터이며, 높은 `lmax`가 항상 더 정확한 것은 아니다.
8. 높은 `lmax`와 large/dense graph에서는 geometry-side 비용이 커지므로, 이 방향의 최적화 가치가 커질 수 있다.

---

## 13. 지금 단계에서 하면 안 되는 주장

아래는 현재 데이터로는 과장이다.

1. “우리 방법은 모든 MLIP에서 빠르다”
2. “현재 구현은 pair-major runtime이다”
3. “FlashTP와 결합하면 반드시 큰 향상이 난다”
4. “SH가 현재 전체 시간의 주 병목이다”
5. “높은 `lmax`는 항상 더 정확하다”
6. “LAMMPS/TorchSim neighbor list를 현재 이미 최대한 활용하고 있다”

---

## 14. 그래도 낙관적으로 볼 수 있는 지점

여기서부터는 **가설/기대효과**다. 다만 코드와 데이터에 근거한 낙관적 가설이다.

### 14.1 완전한 pair-major까지 가면 줄일 수 있는 항목

현재 SH 비중이 작더라도, pair-major로 완전히 분리되면 줄일 수 있는 것은 SH만이 아니다.

#### 현재 이미 일부 줄이는 것

- distance
- radial basis
- cutoff
- spherical harmonics
- pair-level `weight_nn` 입력

#### pair-major가 되면 추가로 줄일 수 있는 후보

1. **pair metadata 오버헤드**
   - upstream reverse map / pair id 직접 사용 시 거의 제거 가능

2. **reverse SH 복원용 edge 확장**
   - `edge_attr_expand`
   - `reverse_sign`
   - `edge_attr_sign_apply`

3. **forward/reverse 분리 실행에서 생기는 indexing**
   - `forward_src_index`
   - `forward_dst_index`
   - `reverse_src_index`
   - `reverse_dst_index`
   - `reverse_weight_select`

4. **forward/reverse 분리 때문에 생기는 gather/scatter launch 수**
   - source gather 횟수
   - aggregation launch 수

5. **pair weight / pair attr의 중복 load**
   - pair-major kernel 내부에서 shared load 가능성

6. **backward에서 저장해야 하는 context**
   - per-edge보다 per-pair 저장량 감소 가능

7. **filter/weight gradient 경로의 중복**
   - pair-aware backward를 짜면 grad_filter / grad_weight도 pair 단위로 다룰 수 있음

8. **force 경로에서 generic autograd graph 일부**
   - 현재는 `autograd.grad(E, edge_vec)`가 전체 graph를 다시 돈다
   - pair-aware force path로 바꾸면 일부 구조적 비용을 줄일 수 있을 가능성 있음

### 14.2 FlashTP와 결합 시 기대할 수 있는 그림

정확한 표현은 이렇다.

- 현재는 geometry-side 최적화
- FlashTP는 TP-side 최적화
- 현재 구현이 아니라 **pair-major + FlashTP**가 되면
  - geometry-side 절감
  - TP-side fused sparse execution
  를 동시에 노릴 수 있다

즉 지금 결과가 작더라도, 방향 자체가 죽은 것은 아니다.

### 14.3 accuracy-latency tradeoff 측면의 낙관적 메시지

현재 `lmax` sweep은 다음을 시사한다.

- 높은 `lmax`가 어떤 경우에는 정확도를 개선
- 하지만 비용이 급격히 증가

따라서 geometry-side 절감이 성공하면:

- **같은 정확도에서 더 빠르게**
- 또는 **같은 시간 예산에서 더 높은 `lmax`를 시도**

라는 방향으로 연결될 수 있다.

단, 이 문장을 논문에 쓰려면 결국 다음 실험이 필요하다.

- 동일 시간 예산 비교
- baseline 낮은 `lmax` vs proposal 높은 `lmax`

---

## 15. 현재 실험 환경 정리

### 15.1 GPU 환경

현재 이 세션에서 확인된 GPU:

- `nvidia-smi -L`: `GPU 0: NVIDIA GeForce RTX 4090`
- `torch.cuda.device_count() = 1`

즉 지금까지 이 세션에서 수행한 `lmax` sweep은 **단일 GPU 실험**이다.

2-GPU 실험이라고 쓰면 안 된다.

### 15.2 반복 횟수

현재 기준:

- step latency 반복: `30회`
- 일부 과거 진단 실험은 `100회` 반복으로 남아 있음
- 문서/최종 보고서에는 어떤 표가 몇 회 반복인지 명시해야 함

### 15.3 현재 2-GPU 상태

실제 2-GPU가 보이는 세션에서는 아직 main KCC 실험을 다시 돌리지 않았다.

즉:

- LAMMPS 2-GPU runbook은 있음
- 하지만 이번 요약의 canonical 결과는 **single RTX 4090**

---

## 16. 지금 당장 가장 가치 있는 후속 실험

### A. baseline vs proposal를 `lmax=1..8`에서 대표 dataset 몇 개에 대해 직접 비교

목적:

- 높은 `lmax`에서 geometry-side proposal의 상대 가치가 실제로 커지는지 확인

대표 후보:

- `small_sparse`: `spice_2023`
- `small_dense`: `salex_val_official`
- `large_sparse`: `oc20_s2ef_train_20m`
- `large_dense`: `mptrj`

### B. 동일 시간 예산 비교

예:

- baseline `lmax=2`
- proposal `lmax=3` 또는 `4`

목적:

- “우리 방법은 정확도를 직접 올리는 것이 아니라, 더 좋은 accuracy-latency tradeoff를 연다”를 입증

### C. pair-major 전용 full path prototype

목적:

- current full의 구조적 손해 제거

핵심 목표:

- forward/reverse 분리 conv 제거
- pair 상태 유지
- pair-aware backward 가능성 탐색

### D. LAMMPS/TorchSim upstream pair-aware integration

목적:

- pair metadata 오버헤드를 근본적으로 줄이기

### E. representative Nsight / torch.profiler

목적:

- large/dense/high-lmax에서 실제 병목이 TP/backward인지 더 낮은 수준에서 재검증

---

## 17. 상위 모델에게 던질 수 있는 질문 예시

1. 현재 결과 기준으로, KCC 논문에서 가장 방어 가능한 claim set은 무엇인가?
2. “geometry-side exact reuse”를 중심으로 논문 제목/초록을 어떻게 다시 써야 하나?
3. `lmax` sweep 결과를 보면 “같은 시간 예산에서 더 높은 lmax” 실험이 가장 중요해 보이는데, 실험 설계를 어떻게 짜야 하나?
4. current full path를 진짜 pair-major로 바꿀 때, 최소 구현으로 가장 큰 이득을 낼 수 있는 구조는 무엇인가?
5. LAMMPS/TorchSim upstream neighbor 정보를 pair-aware하게 직접 쓰려면 인터페이스를 어떻게 바꿔야 하나?
6. FlashTP와의 결합을 실제 논문에서 어느 수준까지 주장할 수 있나?

---

## 18. 최종 요약

지금까지의 연구를 가장 압축해서 말하면 이렇다.

- 현재 구현은 **pair-major 완성형이 아니라 pair-aware geometry-side reuse**
- 정확도는 사실상 유지된다
- 모든 graph에서 빠르지 않다
- 하지만 **큰 graph / 높은 neighbor 조건**에서는 실제 이득이 이미 보인다
- current full path의 주 문제는 SH 하나가 아니라
  - pair metadata
  - pair-major 미구현
  - generic backward
  - forward/reverse split 구조
  다
- `lmax`는 중요한 축이고, 실제로 최적 `lmax`가 존재한다
- 높은 `lmax`, 큰 graph, dense graph일수록 geometry-side 절감의 가치는 커질 가능성이 높다
- 따라서 이 연구는 실패한 최적화가 아니라,
  **조건이 정의된 선행 단계 + pair-major/FlashTP 후속 연구를 위한 기반 연구**
  로 보는 것이 맞다

---

## 19. 실험 환경 상세

### 19.1 코드/브랜치

- repository: `SevenNet`
- working branch: `pair-major`
- 현재 연구 관련 산출물 루트:
  - `docs/papers/KCC/`

### 19.2 장치 환경

현재 이 세션에서 직접 확인한 GPU 환경:

```text
nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4090

torch.cuda.is_available() = True
torch.cuda.device_count() = 1
```

즉 본 요약에서 직접 수행한 주요 baseline / proposal / lmax 실험은 **단일 RTX 4090** 기준이다.

주의:

- 과거에 2-GPU LAMMPS 실험 계획과 runbook은 정리했지만
- 이 문서의 canonical 결과는 single-GPU 기준이다

### 19.3 반복 횟수와 계측 규칙

현재 canonical로 쓰는 반복 규칙:

- `pair_end_to_end` main latency:
  - 각 dataset representative sample
  - `mean ± std`
  - warmup/repeat는 해당 스크립트 구현 기준
- `pair_accuracy`:
  - warmup `2`
  - repeat `10`
- `lmax baseline sweep` step latency:
  - warmup `20`
  - repeat `30`
- `lmax baseline sweep` stage profile:
  - warmup `5`
  - repeat `20`
- `lmax quadrant runtime`:
  - latency repeat `30`
  - intrusive stage repeat `30`

주의:

- 과거 일부 split-validation diagnostic은 `repeat=100`으로 남아 있다
- 논문에서는 **어떤 표가 어떤 repeat 규칙을 쓰는지 표마다 명시**해야 한다

### 19.4 측정 장치 수준

현재 계측은 세 종류다.

1. **non-intrusive wall-clock latency**
   - 실제 end-to-end 성능 비교용

2. **intrusive stage timer**
   - `torch.cuda.synchronize()`를 구간 앞뒤로 넣은 decomposition용
   - 절대 지연시간 claim이 아니라 stage 비중 해석용

3. **representative profiler / Nsight 후보**
   - 구조 확인용
   - 전수 통계용 아님

즉 논문에서는:

- headline latency는 non-intrusive 기준
- stage 해석은 intrusive 기준
- profiler는 보조 증거

로 역할을 나눠야 한다.

---

## 20. 재현 명령과 실행 entrypoint

### 20.1 `lmax` baseline sweep

스크립트:

- [kcc_lmax_baseline_sweep.py](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/scripts/kcc_lmax_baseline_sweep.py)

실행:

```bash
python docs/papers/KCC/scripts/kcc_lmax_baseline_sweep.py
```

출력:

- `docs/papers/KCC/lmax_baseline_sweep/`

### 20.2 `lmax` 사분면 시간 분석

스크립트:

- [kcc_lmax_quadrant_runtime.py](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/scripts/kcc_lmax_quadrant_runtime.py)

실행:

```bash
python docs/papers/KCC/scripts/kcc_lmax_quadrant_runtime.py
```

출력:

- `docs/papers/KCC/lmax_quadrant_runtime/`

### 20.3 pair validation split

관련 결과:

- [pair_validation_split_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pair_validation_split_report.md)
- [pair_validation_interpretation.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pair_validation_interpretation.md)

### 20.4 pair accuracy

관련 결과:

- [pair_accuracy_repeat_report.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/pair_accuracy_repeat_report.md)

### 20.5 LAMMPS 2-GPU runbook

관련 메모:

- [lammps_2gpu_runbook.md](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/reports/lammps_2gpu_runbook.md)

주의:

- runbook은 있음
- canonical 결과는 아직 single-GPU
- 실제 2-GPU 수치가 논문 claim으로 들어가려면 **재실행 필요**

---

## 21. 결과 파일 인덱스

### 21.1 메인 baseline vs proposal

- [pair_end_to_end_comparison.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_end_to_end/global/pair_end_to_end_comparison.csv)
- [pair_latency_all.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_latency_all.png)
- [pair_speedup_all.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_speedup_all.png)
- [pair_speedup_vs_num_edges.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_speedup_vs_num_edges.png)
- [pair_speedup_vs_avg_neighbors.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_speedup_vs_avg_neighbors.png)
- [pair_speedup_by_bucket.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_speedup_by_bucket.png)
- [pair_speedup_size_density_map.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_end_to_end/pair_speedup_size_density_map.png)

### 21.2 정확도 보존

- [pair_accuracy_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_accuracy/global/pair_accuracy_summary.csv)
- [pair_accuracy_raw_repeats.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_accuracy/global/pair_accuracy_raw_repeats.csv)
- [pair_accuracy_energy_errorbar.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_accuracy/pair_accuracy_energy_errorbar.png)
- [pair_accuracy_force_errorbar.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_accuracy/pair_accuracy_force_errorbar.png)

### 21.3 원인 분리 실험

- [pair_validation_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_validation_split/global/pair_validation_summary.csv)
- [pair_validation_raw.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_validation_split/global/pair_validation_raw.csv)
- [pair_metadata_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/metrics/pair_validation_split/global/pair_metadata_summary.csv)
- [pair_validation_forward_energy_latency_all.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_validation_split/pair_validation_forward_energy_latency_all.png)
- [pair_validation_step_force_latency_all.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_validation_split/pair_validation_step_force_latency_all.png)
- [pair_validation_pair_metadata_methods.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/figures/pair_validation_split/pair_validation_pair_metadata_methods.png)

### 21.4 `lmax` 학습/정확도/시간

- [lmax_sweep_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_sweep_summary.csv)
- [lmax_training_history.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_training_history.csv)
- [lmax_latency_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_latency_summary.csv)
- [lmax_stage_profile_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_stage_profile_summary.csv)
- [lmax_inference_errors.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/metrics/lmax_inference_errors.csv)
- [lmax_accuracy_rmse.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_accuracy_rmse.png)
- [lmax_step_latency.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_step_latency.png)
- [lmax_accuracy_latency_frontier.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_accuracy_latency_frontier.png)
- [lmax_trainable_params.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_trainable_params.png)
- [lmax_training_wall_time.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_training_wall_time.png)
- [lmax_stage_profile.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_baseline_sweep/figures/lmax_stage_profile.png)

### 21.5 `lmax` 사분면 시간 분석

- [quadrant_lmax_latency_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_latency_summary.csv)
- [quadrant_lmax_stage_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_stage_summary.csv)
- [quadrant_lmax_growth_summary.csv](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/metrics/quadrant_lmax_growth_summary.csv)
- [quadrant_lmax_latency.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/figures/quadrant_lmax_latency.png)
- [quadrant_lmax_latency_growth.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/figures/quadrant_lmax_latency_growth.png)
- [quadrant_lmax_stage_shares.png](/home/wise/minchang/DenseMLIP/SevenNet/docs/papers/KCC/lmax_quadrant_runtime/figures/quadrant_lmax_stage_shares.png)

---

## 22. 논문 서사 후보

### 서사 A: 조건이 정의된 최적화

핵심 문장:

- 이 연구는 모든 graph를 빠르게 만드는 만능 최적화가 아니라,
- **large / high-neighbor MLIP 추론**에서 geometry-side reuse가 실효성 있는 조건을 규정한 연구다.

장점:

- 현재 데이터와 가장 잘 맞는다
- 과장 없이 방어 가능하다

### 서사 B: 정확도 유지형 실행 최적화

핵심 문장:

- 제안기법은 출력 정확도를 바꾸지 않고 실행 시간만 바꾸는 최적화다.
- 이는 baseline 반복 잡음 수준의 작은 차이로 검증되었다.

장점:

- 논문 핵심 메시지가 명확하다

### 서사 C: higher-`lmax` 시대의 geometry-side 비용

핵심 문장:

- 최신 equivariant MLIP는 더 큰 `lmax`를 쓰는 방향으로 갈 수 있고,
- 이때 geometry-side 비용은 절대량이 커진다.
- 본 연구는 그 비용을 줄이는 방향의 초기 실증이다.

장점:

- future-looking
- pair-major / FlashTP 후속 연구와 자연스럽게 이어짐

주의:

- 현재 데이터는 “SH가 전체 주병목”까지는 지지하지 않는다
- 더 정확한 표현은 “geometry-side 비용의 절대량이 커지고, 전체 구조 안에서 줄일 가치가 높아진다”이다

### 추천 조합

가장 무난한 조합은:

- 메인: **서사 A**
- 보조: **서사 B**
- 확장 방향: **서사 C**

---

## 23. pair-major가 되면 줄일 수 있는 비용 항목 상세

이 섹션은 낙관적인 가능성을 정리하기 위한 것이다.  
현재는 SH 비중이 작아 보일 수 있지만, pair-major 완전분리까지 가면 줄일 수 있는 항목은 SH 하나가 아니다.

### 이미 일부 줄인 항목

- pair distance
- radial basis
- cutoff
- SH
- pair-level weight input

### pair-major가 되면 추가로 직접 줄일 수 있는 항목

1. reverse edge용 geometry expansion
2. reverse sign application 경로
3. forward/reverse split를 위한 index_select / gather
4. forward/reverse split 때문에 2번 들어가는 aggregation
5. pair_attr / pair_weight의 redundant global memory traffic
6. backward context의 per-edge 저장량
7. grad filter / grad weight의 중복 구조
8. force backward에서 edge-major generic autograd 일부

### 특히 기대가 큰 부분

- `force_output`은 high-`lmax`, large/dense graph에서 폭증한다
- current implementation에서는 이것이 generic autograd graph 위에 올라간다
- pair-aware backward가 생기면, SH 절감보다 더 큰 구조적 개선이 나올 가능성이 있다

즉 “현재는 SH 비중이 작다”가 곧 “이 방향이 의미 없다”는 뜻은 아니다.  
오히려 **pair-major가 아직 아니라서 진짜 줄일 수 있는 다른 비용들이 아직 남아 있다**고 보는 것이 맞다.

---

## 24. 낙관적인 해석을 할 때 지켜야 할 선

상위 모델이나 논문 초안이 너무 비관적으로 가는 것도 문제지만,  
지금 데이터가 말하지 않는 것을 강하게 주장하면 더 위험하다.

권장 표현:

- “현재 구현은 하한선 성격의 초기 구현이다”
- “large/high-neighbor 조건에서는 이미 오버헤드를 넘는 개선이 관측된다”
- “pair-major와 pair-aware backward가 도입되면 추가 개선 가능성이 있다”
- “FlashTP와는 중복이 아니라 상보적이다”
- “higher-`lmax`와 large/dense graph에서 geometry-side 비용 절감의 가치가 커질 가능성이 있다”

피해야 할 표현:

- “현재 구현만으로 이미 완전한 pair-major 효과를 보였다”
- “FlashTP와 결합하면 반드시 크게 빨라진다”
- “높은 `lmax`에서는 SH가 전체 병목이다”

---

## 25. 상위 모델에게 요구할 출력 형식

상위 모델에게 이 문서를 주고 요청할 때는, 아래 형식까지 같이 명시하는 편이 좋다.

1. **논문 claim set**
   - strongest defensible claims
   - softer but promising claims
   - claims to avoid

2. **논문 구조**
   - abstract
   - intro
   - method
   - results
   - discussion
   - future work

3. **실험 계획**
   - must-run
   - nice-to-have
   - optional

4. **도표 계획**
   - main paper figures
   - appendix figures
   - tables

5. **후속 구현 우선순위**
   - pair metadata
   - pair-major conv
   - pair-aware backward
   - FlashTP integration
   - LAMMPS / TorchSim upstream integration

---

## 26. ChatGPT Pro에 바로 넣을 수 있는 요청 템플릿

아래는 상위 모델에게 함께 줄 수 있는 요청 템플릿이다.

```text
아래 markdown은 SevenNet 기반 pair-geometry reuse 연구의 현재 구현, 실험, 결과, 열린 가설을 정리한 컨텍스트 패키지다.

요청:
1. 현재 데이터로 방어 가능한 strongest claim set을 정리해라.
2. 과장 없이도 매력적인 논문 서사를 제안해라.
3. KCC/국내학회 수준 선행 논문 버전과, pair-major/FlashTP 후속 확장 논문 버전을 각각 설계해라.
4. 지금 당장 반드시 추가해야 할 실험과, 있으면 좋은 실험을 구분해라.
5. 본문의 결과 섹션을 어떤 그림/표 중심으로 구성해야 하는지 제안해라.
6. higher-lmax와 large/dense graph에서 이 연구가 왜 중요한지 이론+실험 관점에서 정리해라.
7. 현재 구현의 한계와, 이를 pair-major / pair-aware backward / upstream pair integration으로 어떻게 넘어갈지 설계해라.

중요:
- 현재 데이터가 직접 지지하는 사실과 아직 가설인 내용을 분리해라.
- overly pessimistic하게 쓰지 말고, 가능한 연구 가치와 후속 가능성도 적극적으로 정리해라.
- 다만 현재 데이터가 말하지 않는 것을 사실처럼 쓰지 마라.
```

---

## 27. 마지막 요약

이 연구를 현재 시점에서 가장 잘 요약하면:

- **정확도 보존형 geometry-side 실행 최적화**
- **조건이 정의된 성능 개선**
- **pair-major/FlashTP 후속 연구를 위한 실증 기반**

그리고 가장 중요한 열린 가능성은:

- 현재는 SH 비중이 작아도,
- pair-major 완전분리와 pair-aware backward까지 가면
- 지금 남아 있는 TP/backward/indexing/metadata 비용까지 줄일 수 있고,
- 그러면 높은 `lmax`, large/dense graph, 실사용 MD 추론 조건에서
- **정확도-시간 tradeoff를 실제로 더 좋은 방향으로 밀 수 있다**

