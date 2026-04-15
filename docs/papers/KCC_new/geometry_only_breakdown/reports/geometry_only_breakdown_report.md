# Geometry-Only Breakdown Report

- checkpoint: `tests/data/checkpoints/cp_0.pth`
- warmup: `5`
- repeats: `30`
- cases: `baseline`, `geometry_only`
- focus: `geometry_only` 내부의 pair->edge 재확장 비용 분해
- note: 이 계측은 `torch.cuda.synchronize()`를 각 stage 경계에 넣는 intrusive timing이다.
- note: 따라서 `model_total_ms` 절대값은 headline latency가 아니라 stage 비교용으로만 읽어야 한다.

## bulk_large

- baseline forward total: `2.997 ms`
- geometry_only forward total: `3.107 ms`
- pair->edge expansion: `0.029 ms`
- pair weight expand: `0.036 ms`
- pair geometry (norm+basis+sh): `0.194 ms`

## bulk_small

- baseline forward total: `2.940 ms`
- geometry_only forward total: `3.047 ms`
- pair->edge expansion: `0.028 ms`
- pair weight expand: `0.034 ms`
- pair geometry (norm+basis+sh): `0.187 ms`

## dimer_small

- baseline forward total: `2.830 ms`
- geometry_only forward total: `2.918 ms`
- pair->edge expansion: `0.028 ms`
- pair weight expand: `0.032 ms`
- pair geometry (norm+basis+sh): `0.173 ms`
