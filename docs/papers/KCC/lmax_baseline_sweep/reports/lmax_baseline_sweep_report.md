# lmax baseline sweep note

대상은 `rMD17 azobenzene` 고정 분할이며, 같은 baseline 구조에서 `lmax=1..8`만 바꿔 학습/평가했다.

## 핵심 질문

- `lmax`는 데이터셋이 자동으로 정하는 값인가?
- `lmax`를 올리면 정확도가 항상 좋아지는가?
- 그 대가로 시간은 얼마나 늘어나는가?

## 실험 설정

- dataset: `rmd17_azobenzene`
- split: train 512, valid 128, test 128
- cutoff: 5.0
- epoch: 15
- batch size: 8
- device: cuda
- latency repeat: warmup 20, repeat 30
- profile repeat: warmup 5, repeat 20

## 대표 결과

- 최저 test force RMSE: lmax=7, 0.136777 eV/A
- 최저 test energy RMSE: lmax=7, 0.001516 eV
- 최저 step latency: lmax=1, 5.089 ms
- 최대 학습 시간: 23.19 min

## 숫자로 보는 추세

- `lmax=1 -> 7`에서 test force RMSE는 `0.306315 -> 0.136777 eV/A`로 줄어들어 약 `55.3%` 개선됐다.
- 하지만 `lmax=8`에서는 force RMSE가 다시 `0.147295 eV/A`로 올라가, 더 큰 `lmax`가 항상 더 정확한 것은 아니었다.
- step latency는 `lmax=1 -> 8`에서 `5.089 -> 52.680 ms`로 약 `10.35x` 증가했다.
- 학습 시간은 `lmax=1 -> 8`에서 `18.95 -> 1391.69 s`로 약 `73.44x` 증가했다.
- trainable parameter는 `lmax=1 -> 8`에서 `32192 -> 555456`로 약 `17.25x` 증가했다.

## 해석 메모

- `lmax`는 데이터셋이 정해주는 값이 아니라 모델 설계자가 고르는 하이퍼파라미터다.
- 다만 어떤 데이터셋에서는 높은 각도 표현이 필요해 더 큰 `lmax`가 유리할 수 있고, 어떤 데이터셋에서는 그렇지 않을 수 있다.
- 따라서 실제로는 `정확도-시간 절충`을 실험으로 확인해야 한다.
- 이번 고정 조건 실험에서는 `rMD17 azobenzene`에 대해 `lmax=7`이 가장 좋은 힘 정확도를 보였고, `lmax=8`은 더 비싸지만 오히려 성능이 약간 나빠졌다.
- 따라서 `해상도가 높으면 정확도도 무조건 높아진다`고 쓰면 안 되고, `적절한 lmax가 존재한다`고 해석하는 것이 맞다.
- 대표 구조 기준에서 SH 절대 시간은 `lmax=1 -> 8`에서 `0.0763 -> 2.0272 ms`로 크게 늘었다.
- 그러나 SH 비중은 전체 model time의 `1.01% -> 2.81%` 수준에 머물렀다.
- 반면 force output 비중은 `40.78% -> 53.62%`로 더 컸다.
- 즉 높은 `lmax`의 비용은 SH 하나만이 아니라 TP와 force backward가 함께 커지는 구조로 이해해야 한다.

## 저장된 파일

- summary: `metrics/lmax_sweep_summary.csv`
- training history: `metrics/lmax_training_history.csv`
- latency: `metrics/lmax_latency_summary.csv`
- stage profile: `metrics/lmax_stage_profile_summary.csv`
- figures: `figures/*.png`
