# KCC_new Pair Profiler Interpretation

대표 profiler는 `qm9_hf`와 `mptrj`에서 `baseline`과 `geometry_only`를 각각 `forward_energy`, `force_model` 모드로 한 번씩 수집한 것이다.

관련 파일:

- `metrics/pair_profiler/pair_profiler_summary.csv`
- `metrics/pair_profiler/qm9_hf/*.csv`
- `metrics/pair_profiler/mptrj/*.csv`

## 핵심 관찰

### 1. 현재 dominant op는 여전히 일반 elementwise / copy 계열이다

forward와 force 모두에서 상위 device time은 주로 다음 연산이 차지한다.

- `aten::mul`
- `aten::fill_`
- `aten::copy_`
- `aten::clone`

즉 현재 geometry_only가 dominant kernel class 자체를 바꾸지는 못하고 있다.

### 2. geometry_only는 `index_select` 계열 비용을 분명히 늘린다

`index_select` 관련 device time 합은 representative profiler에서 다음과 같았다.

- `mptrj`, `forward_energy`
  - baseline: `1.600 us`
  - geometry_only: `2017.683 us`
- `mptrj`, `force_model`
  - baseline: `1.632 us`
  - geometry_only: `2027.508 us`
- `qm9_hf`, `forward_energy`
  - baseline: `1.920 us`
  - geometry_only: `42.402 us`
- `qm9_hf`, `force_model`
  - baseline: `1.921 us`
  - geometry_only: `43.233 us`

이는 pair->edge 재확장과 pair-aware indexing이 실제로 추가 비용을 만든다는 점을 정량적으로 보여준다.

### 3. force path에서는 backward 성격의 일반 연산이 그대로 dominant다

`force_model`에서는 `mul`, `fill_`, `copy_` 비중이 크게 올라가며, geometry_only가 이 dominant class를 줄이지는 못한다.  
즉 현재 geometry_only는 forward 앞단의 geometry 중복을 줄이더라도, generic force path의 큰 비용 구조는 그대로 남는다.

## 해석

이 profiler는 geometry_only가 왜 end-to-end에서 baseline을 아직 이기지 못하는지에 대한 정성적 근거를 제공한다.

1. dominant class가 아직 generic edge-major path에 머물러 있다.
2. geometry_only는 `index_select` / expand 계열을 새로 만든다.
3. force path에서는 generic backward 성격의 연산이 계속 상위를 차지한다.

따라서 다음 구현 우선순위는 다음과 일치한다.

1. pair->edge expand 최소화
2. pair weight expand 최소화
3. edge-major force path 축소
