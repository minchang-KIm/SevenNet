# Baseline Detailed Model Profiling Report

## Goal

`SevenNet e3nn baseline` 내부를 더 잘게 쪼개서, 아래 단계들의 시간을 모두 보이게 했다.

- `edge length norm`
- `radial basis`
- `cutoff`
- `radial combine`
- `spherical harmonics`
- block별 `weight_nn`
- block별 `source gather`
- block별 `message tensor product`
- block별 `aggregation`
- block별 `denominator`
- top-level `input embedding`
- top-level `interaction/gate/self-connection`
- top-level `readout`
- top-level `force_output`

## Where The Results Are

가장 중요한 파일은 아래 3개다.

- Wide summary: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/detailed_model_profile_main/metrics/summary.csv`
- Long-form every stage: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/detailed_model_profile_main/metrics/stage_breakdown_long.csv`
- Aggregate by semantic stage: `/home/wise/minchang/DenseMLIP/SevenNet/bench_runs/detailed_model_profile_main/metrics/stage_breakdown_aggregate.csv`

참고:

- `stage_breakdown_long.csv`는 `0_convolution.message_tp_ms` 같은 block별 raw stage를 전부 담는다.
- `stage_breakdown_aggregate.csv`는 다섯 개 convolution block의 같은 종류 stage를 합친 표다.
- 이 프로파일은 intrusive profiling이다. 즉 각 stage 전후로 synchronize하므로, 절대 end-to-end latency claim이 아니라 stage decomposition 용도로 써야 한다.

## Stage Name Guide

- `edge_embedding.edge_length_norm_ms`: `||r||` 계산
- `edge_embedding.radial_basis_ms`: bessel basis
- `edge_embedding.cutoff_ms`: cutoff 함수
- `edge_embedding.radial_combine_ms`: `basis * cutoff`
- `edge_embedding.spherical_harmonics_ms`: SH 계산
- `N_convolution.weight_nn_ms`: block `N`의 weight MLP
- `N_convolution.edge_src_gather_ms`: source node feature gather
- `N_convolution.message_tp_ms`: message tensor product
- `N_convolution.aggregation_ms`: scatter/reduce aggregation
- `N_convolution.denominator_ms`: convolution output normalization

## Baseline Aggregate Summary

아래 값은 `stage_breakdown_aggregate.csv`의 `e3nn_baseline` 행만 요약한 것이다.

| Dataset | Atoms | Model total | SH | Conv weight_nn | Conv src gather | Conv message TP | Conv aggregation | Force output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mptrj` | 444 | `666.33 ms` | `120.55 ms` | `1.53 ms` | `0.70 ms` | `104.08 ms` | `5.14 ms` | `367.11 ms` |
| `md22_double_walled_nanotube` | 370 | `856.66 ms` | `226.41 ms` | `1.41 ms` | `0.61 ms` | `96.99 ms` | `4.51 ms` | `460.36 ms` |
| `spice_2023` | 89 | `156.02 ms` | `0.29 ms` | `0.46 ms` | `0.09 ms` | `62.17 ms` | `0.30 ms` | `27.09 ms` |
| `md22_stachyose` | 87 | `175.28 ms` | `0.31 ms` | `0.55 ms` | `0.14 ms` | `64.85 ms` | `0.90 ms` | `42.94 ms` |
| `ani1x` | 63 | `157.65 ms` | `0.28 ms` | `0.49 ms` | `0.11 ms` | `63.23 ms` | `0.57 ms` | `27.41 ms` |
| `rmd17` | 24 | `156.73 ms` | `0.27 ms` | `0.45 ms` | `0.07 ms` | `62.59 ms` | `0.26 ms` | `27.65 ms` |
| `iso17` | 19 | `156.70 ms` | `0.31 ms` | `0.45 ms` | `0.09 ms` | `62.43 ms` | `0.26 ms` | `27.43 ms` |

## Main Reading

이 표가 보여주는 건 명확하다.

1. 작은 분자 그래프에서는 baseline 내부에서 `SH`, `weight_nn`, `gather`, `aggregation`은 매우 작다.
2. 작은 그래프에서는 convolution의 대부분이 사실상 `message TP`와 `force_output`에 있다.
3. 큰 periodic graph에서는 `spherical harmonics` 자체가 매우 커진다.
4. `mptrj`와 `nanotube`에서는 `force_output + SH + message TP`가 지배적이다.
5. 반대로 `weight_nn`, `source gather`, `aggregation`은 baseline에서도 절대시간이 작다.

즉 baseline 기준으로 보면, 현재 pair execution이 줄이는 `weight_nn`이나 일부 edge-side work는 실제 전체 시간의 일부에만 해당한다. 그래서 pair execution이 큰 graph에서만 들리고, 작은 graph에서는 전체를 뒤집을 만큼 크지 않다.

## Per-Block Inspection

block별 raw stage는 `stage_breakdown_long.csv`에서 직접 볼 수 있다.

예를 들어 `mptrj` baseline의 큰 항목들은 다음 순서였다.

- `top_force_output_ms`
- `edge_embedding.spherical_harmonics_ms`
- `3_convolution.message_tp_ms`
- `2_convolution.message_tp_ms`
- `1_convolution.message_tp_ms`

`spice_2023` baseline은 다음 순서였다.

- `top_interaction_other_ms`
- `top_force_output_ms`
- `3_convolution.message_tp_ms`
- `2_convolution.message_tp_ms`
- `1_convolution.message_tp_ms`

즉 작은 graph는 TP 중심, 큰 graph는 `force + SH + TP` 중심으로 보는 것이 맞다.
