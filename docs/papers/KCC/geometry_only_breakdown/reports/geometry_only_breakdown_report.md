# Geometry-Only Breakdown Report

- checkpoint: `tests/data/checkpoints/cp_0.pth`
- warmup: `5`
- repeats: `30`
- cases: `baseline`, `geometry_only`
- focus: `geometry_only` 내부의 pair->edge 재확장 비용 분해
- note: 이 계측은 `torch.cuda.synchronize()`를 각 stage 경계에 넣는 intrusive timing이다.
- note: 따라서 `model_total_ms` 절대값은 headline latency가 아니라 stage 비교용으로만 읽어야 한다.

## bulk_large

- baseline forward total: `3.012 ms`
- geometry_only forward total: `3.136 ms`
- pair->edge expansion: `0.030 ms`
- pair weight expand: `0.037 ms`
- pair geometry (norm+basis+sh): `0.191 ms`

## bulk_small

- baseline forward total: `2.946 ms`
- geometry_only forward total: `3.040 ms`
- pair->edge expansion: `0.028 ms`
- pair weight expand: `0.032 ms`
- pair geometry (norm+basis+sh): `0.187 ms`

## dimer_small

- baseline forward total: `2.789 ms`
- geometry_only forward total: `2.921 ms`
- pair->edge expansion: `0.027 ms`
- pair weight expand: `0.032 ms`
- pair geometry (norm+basis+sh): `0.173 ms`
