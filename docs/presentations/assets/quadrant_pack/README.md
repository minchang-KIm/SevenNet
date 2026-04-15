# Seminar Asset Pack

## Representative Quadrants

Seminar framing thresholds used in the quadrant map:

- `large` if representative sample has `num_edges >= 3000`
- `dense` if representative sample has `avg_neighbors_directed >= 40`

- `Small Sparse`: `SPICE 2023` (atoms=`89`, edges=`772`, avg_neighbors=`8.67`, speedup=`0.743x`, source=`local_pair_size_main`)
- `Small Dense`: `phononDB PBE` (atoms=`8`, edges=`640`, avg_neighbors=`80.00`, speedup=`1.105x`, source=`real_e3nn_pair_k3`)
- `Large Sparse`: `OMol25 validation` (atoms=`110`, edges=`3826`, avg_neighbors=`34.78`, speedup=`1.126x`, source=`real_e3nn_pair_k3`)
- `Large Dense`: `MPtrj validation` (atoms=`426`, edges=`28964`, avg_neighbors=`67.99`, speedup=`1.669x`, source=`real_e3nn_pair_k3`)

## Largest Datasets Still Missing In `datasets/raw`

- `oc22_s2ef_train`: approx `30.13 GiB`, status=`pending`
- `omol25_validation`: approx `13.62 GiB`, status=`pending`
- `omol25_train_4m`: approx `11.49 GiB`, status=`pending`
- `salex_train_official`: approx `7.52 GiB`, status=`pending`
- `oc20_s2ef_train_2m`: approx `4.74 GiB`, status=`pending`
- `oc20_s2ef_val_ood_both`: approx `3.31 GiB`, status=`pending`
- `oc20_s2ef_val_ood_ads`: approx `3.13 GiB`, status=`pending`
- `oc20_s2ef_val_ood_cat`: approx `3.11 GiB`, status=`pending`
- `oc20_s2ef_val_id`: approx `3.00 GiB`, status=`pending`
- `oc22_s2ef_val_ood`: approx `1.61 GiB`, status=`pending`

## Created Files

- `quadrant_representatives.csv`
- `all_dataset_points.csv`
- `large_dataset_status.csv`
- `plots/quadrant_dataset_map.png`
- `plots/quadrant_latency_comparison.png`
- `plots/quadrant_speedup_comparison.png`
- `plots/large_dataset_download_status.png`
- `plots/extreme_stage_breakdown.png`
- `diagrams/quadrant_mechanism_diagram.png`
- `diagrams/quadrant_mechanism_diagram.svg`
