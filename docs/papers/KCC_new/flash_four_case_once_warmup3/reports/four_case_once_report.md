# FlashTP Four-Case One-Shot Result

## Purpose

성능이 잘 나온 대표 데이터셋에서 기본 SevenNet, 제안기법, FlashTP, FlashTP+제안기법을 한 번씩 실행해 결합 가능성과 대략적인 latency를 확인한다.

## Run Configuration

- datasets: `oc20_s2ef_val_ood_ads`
- warmup: `3`
- repeat: `1`
- note: repeat=1 결과이므로 논문 수치가 아니라 smoke benchmark다.

## Results

| case | status | timing_ms | resolved_policy | baseline/case or error |
| --- | --- | ---: | --- | --- |
| sevennet_baseline | ok | 102.741 | baseline | 1.000x |
| sevennet_pair_full | ok | 97.019 | full | 1.059x |
| flashtp_baseline | ok | 16.019 | baseline | 6.414x |
| flashtp_pair_full | ok | 16.880 | full | 6.087x |

## Interpretation Guardrail

- `sevennet_pair_full`은 최근 win을 복구한 two-pass 쌍 단위 실행 경로다.
- `flashtp_pair_full`은 FlashTP fused tensor-product backend 위에서 pair geometry/weight 재사용 입력을 붙인 결합 smoke test다.
- 현재 FlashTP convolution class는 일반 e3nn convolution의 two-pass gather 경로와 동일하지 않으므로, 이 결과를 최종 결합 성능 claim으로 쓰면 안 된다.
