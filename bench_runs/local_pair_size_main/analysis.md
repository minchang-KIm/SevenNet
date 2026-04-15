# Local Size Effect + Stage Profiling Analysis

Run date: `2026-03-30T13:55:57.515837`

## Goal

Validate whether larger graphs consistently benefit more from the current pair-execution full path,
using only already-downloaded local datasets, and break down stage-wise runtime/load.

## Main Findings

- Spearman correlation between speedup and `natoms`: `0.541`
- Spearman correlation between speedup and `num_edges`: `0.487`

### Largest Wins

- `mptrj` / `mp-1204603`: `444` atoms, `28508` edges, speedup `1.444x`.
- `mptrj` / `mp-1204603`: `444` atoms, `24168` edges, speedup `1.007x`.
- `mptrj` / `mp-1204603`: `444` atoms, `27612` edges, speedup `1.006x`.
- `md22_double_walled_nanotube` / `md22_double-walled_nanotube_2001`: `370` atoms, `24032` edges, speedup `1.006x`.
- `md22_double_walled_nanotube` / `md22_double-walled_nanotube_1987`: `370` atoms, `24024` edges, speedup `1.005x`.

### Largest Losses

- `ani1x` / `ani1x-release_416625`: `63` atoms, `2008` edges, speedup `0.729x`.
- `spice_2023` / `hie__SPICE_Solvated_Amino_Acids_Single_Points_Dataset_v1.1__index_34`: `89` atoms, `772` edges, speedup `0.743x`.
- `md22_stachyose` / `md22_stachyose_10546`: `87` atoms, `3532` edges, speedup `0.747x`.
- `ani1x` / `ani1x-release_416622`: `62` atoms, `2086` edges, speedup `0.753x`.
- `ani1x` / `ani1x-release_273723`: `62` atoms, `1684` edges, speedup `0.755x`.

## Stage Profiling Highlights

- `mptrj`: model_total `557.53 -> 494.63` ms, edge_embedding `1.27 -> 1.10` ms, conv_weight_nn `3.54 -> 2.37` ms, conv_tensor_product `196.15 -> 193.42` ms.
- `md22_double_walled_nanotube`: model_total `487.85 -> 426.07` ms, edge_embedding `1.48 -> 1.23` ms, conv_weight_nn `3.07 -> 2.10` ms, conv_tensor_product `181.52 -> 179.11` ms.
- `spice_2023`: model_total `156.54 -> 118.19` ms, edge_embedding `1.45 -> 1.20` ms, conv_weight_nn `1.00 -> 0.95` ms, conv_tensor_product `113.83 -> 120.98` ms.
- `md22_stachyose`: model_total `174.81 -> 121.86` ms, edge_embedding `1.37 -> 1.22` ms, conv_weight_nn `1.77 -> 1.24` ms, conv_tensor_product `117.52 -> 123.69` ms.
- `ani1x`: model_total `157.36 -> 119.97` ms, edge_embedding `1.37 -> 1.21` ms, conv_weight_nn `1.43 -> 1.05` ms, conv_tensor_product `115.23 -> 123.60` ms.
- `rmd17`: model_total `156.13 -> 116.86` ms, edge_embedding `1.40 -> 1.26` ms, conv_weight_nn `1.05 -> 0.94` ms, conv_tensor_product `113.40 -> 121.23` ms.
- `iso17`: model_total `155.53 -> 116.29` ms, edge_embedding `1.41 -> 1.22` ms, conv_weight_nn `0.99 -> 0.93` ms, conv_tensor_product `112.56 -> 120.98` ms.

## Interpretation

- If speedup tracks `num_edges` more strongly than `natoms`, the current implementation is behaving like an edge-load optimization rather than an atom-count optimization.
- The load metrics explain why: pair execution reduces edge embedding and weight_nn rows from `num_edges` to `num_pairs`, but TP/scatter rows remain `num_edges`.
- Therefore large wins require graphs where saved geometry/weight work dominates indexing and unchanged TP/scatter work.
