# FlashTP Four-Case One-Shot Result

## Purpose

성능이 잘 나온 대표 데이터셋에서 기본 SevenNet, 제안기법, FlashTP, FlashTP+제안기법을 한 번씩 실행해 결합 가능성과 대략적인 latency를 확인한다.

## Run Configuration

- datasets: `md22_buckyball_catcher, oc20_s2ef_val_ood_ads`
- warmup: `3`
- repeat: `1`
- note: repeat=1 결과이므로 논문 수치가 아니라 smoke benchmark다.

## Results

| dataset | case | status | timing_ms | resolved_policy | baseline/case or error |
| --- | --- | --- | ---: | --- | --- |
| md22_buckyball_catcher | sevennet_baseline | ok | 103.543 | baseline | 1.000x |
| md22_buckyball_catcher | sevennet_pair_full | ok | 97.046 | full | 1.067x |
| md22_buckyball_catcher | flashtp_baseline | ok | 14.789 | baseline | 7.001x |
| md22_buckyball_catcher | flashtp_pair_full | ok | 15.512 | full | 6.675x |
| oc20_s2ef_val_ood_ads | sevennet_baseline | ok | 103.114 | baseline | 1.000x |
| oc20_s2ef_val_ood_ads | sevennet_pair_full | ok | 96.476 | full | 1.069x |
| oc20_s2ef_val_ood_ads | flashtp_baseline | ok | 16.025 | baseline | 6.435x |
| oc20_s2ef_val_ood_ads | flashtp_pair_full | ok | 16.514 | full | 6.244x |

## Cross-Case Observations

- `md22_buckyball_catcher`: 제안기법 단독은 baseline 대비 `1.067x`, FlashTP 단독은 baseline 대비 `7.001x`, FlashTP+제안기법은 FlashTP 단독 대비 `0.953x`다.
- `oc20_s2ef_val_ood_ads`: 제안기법 단독은 baseline 대비 `1.069x`, FlashTP 단독은 baseline 대비 `6.435x`, FlashTP+제안기법은 FlashTP 단독 대비 `0.970x`다.
- 두 대표 데이터셋 모두에서 제안기법 단독은 기존 30회 결과와 같은 방향으로 baseline을 이겼다.
- 두 대표 데이터셋 모두에서 현재 FlashTP+제안기법은 FlashTP 단독보다 약간 느렸다. 이는 결합이 불가능하다는 뜻이 아니라, 현재 결합 경로가 FlashTP fused TP 내부까지 pair-major하게 합쳐진 구조가 아니기 때문이다.

## Interpretation Guardrail

- `sevennet_pair_full`은 최근 win을 복구한 two-pass 쌍 단위 실행 경로다.
- `flashtp_pair_full`은 FlashTP fused tensor-product backend 위에서 pair geometry/weight 재사용 입력을 붙인 결합 smoke test다.
- 현재 FlashTP convolution class는 일반 e3nn convolution의 two-pass gather 경로와 동일하지 않으므로, 이 결과를 최종 결합 성능 claim으로 쓰면 안 된다.
