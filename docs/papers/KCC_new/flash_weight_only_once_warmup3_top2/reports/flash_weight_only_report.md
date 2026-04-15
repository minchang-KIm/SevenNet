# FlashTP Weight-Only Pair Reuse Smoke Test

## Purpose

FlashTP는 edge-major 입력을 요구하므로 edge list와 edge filter는 그대로 두고, `weight_nn` 입력/출력만 reverse pair에서 공유했을 때 FlashTP 단독보다 좋아지는지 확인한다.

## Run Configuration

- datasets: `md22_buckyball_catcher, oc20_s2ef_val_ood_ads`
- warmup: `3`
- repeat: `1`
- source modification: none. `EdgeEmbedding.forward` is monkey-patched only inside this script process.

## Results

| dataset | case | status | timing_ms | FlashTP/case | max force diff |
| --- | --- | --- | ---: | ---: | ---: |
| md22_buckyball_catcher | flashtp_baseline | ok | 14.702 | 1.000x | 1.967e-06 |
| md22_buckyball_catcher | flashtp_pair_full_current | ok | 15.743 | 0.934x | 2.384e-06 |
| md22_buckyball_catcher | flashtp_pair_weight_only | ok | 15.248 | 0.964x | 2.503e-06 |
| oc20_s2ef_val_ood_ads | flashtp_baseline | ok | 15.967 | 1.000x | 8.941e-07 |
| oc20_s2ef_val_ood_ads | flashtp_pair_full_current | ok | 16.824 | 0.949x | 1.609e-06 |
| oc20_s2ef_val_ood_ads | flashtp_pair_weight_only | ok | 16.774 | 0.952x | 1.386e-06 |

## Interpretation

- `flashtp_pair_weight_only` keeps normal edge construction and normal edge spherical harmonics.
- Pair metadata is used only to compute `weight_nn(pair_embedding)` once per reverse pair, then expand the resulting weight to FlashTP's edge-major input.
- Full-edge radial/cutoff embedding is intentionally skipped in the patched case because FlashTP consumes the expanded weight, not `EDGE_EMBEDDING`.
- If this case is slower than `flashtp_baseline`, the weight MLP saving is smaller than pair metadata plus pair-to-edge weight expansion overhead.
