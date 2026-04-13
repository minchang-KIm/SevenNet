# All Public Local Pair Benchmark Analysis

Run date: `2026-03-31T21:47:38.039039`

## Scope

- Benchmarked all locally available public datasets that can be converted back to ASE `Atoms` directly from the downloaded cache.
- Compared `e3nn_baseline` vs `e3nn_pair_full` with `7net-omni`.
- Logged unsupported datasets explicitly instead of silently dropping them.

## Main Findings

- Spearman(speedup, natoms): `0.502`
- Spearman(speedup, num_edges): `0.413`

### Best Wins

- `qm9_hf`: natoms=`29`, edges=`570`, avg_neighbors=`19.7`, speedup=`3.737x`.
- `oc20_s2ef_train_20m`: natoms=`225`, edges=`8260`, avg_neighbors=`36.7`, speedup=`1.034x`.
- `oc20_s2ef_val_ood_ads`: natoms=`225`, edges=`6686`, avg_neighbors=`29.7`, speedup=`1.028x`.
- `md22_buckyball_catcher`: natoms=`148`, edges=`6872`, avg_neighbors=`46.4`, speedup=`1.023x`.
- `salex_train_official`: natoms=`132`, edges=`9932`, avg_neighbors=`75.2`, speedup=`1.018x`.
- `omol25_train_neutral`: natoms=`181`, edges=`9926`, avg_neighbors=`54.8`, speedup=`1.017x`.
- `oc20_s2ef_val_id`: natoms=`225`, edges=`15230`, avg_neighbors=`67.7`, speedup=`1.013x`.
- `oc22_s2ef_val_id`: natoms=`200`, edges=`11704`, avg_neighbors=`58.5`, speedup=`1.012x`.

### Largest Losses

- `iso17`: natoms=`19`, edges=`340`, avg_neighbors=`17.9`, speedup=`0.730x`.
- `wbm_initial`: natoms=`100`, edges=`2260`, avg_neighbors=`22.6`, speedup=`0.748x`.
- `md22_dha`: natoms=`56`, edges=`1406`, avg_neighbors=`25.1`, speedup=`0.751x`.
- `ani1ccx`: natoms=`55`, edges=`2042`, avg_neighbors=`37.1`, speedup=`0.751x`.
- `ani1x`: natoms=`63`, edges=`2008`, avg_neighbors=`31.9`, speedup=`0.753x`.
- `md22_ac_ala3_nhme`: natoms=`42`, edges=`1142`, avg_neighbors=`27.2`, speedup=`0.754x`.
- `md22_at_at`: natoms=`60`, edges=`1312`, avg_neighbors=`21.9`, speedup=`0.757x`.
- `spice_2023`: natoms=`89`, edges=`728`, avg_neighbors=`8.2`, speedup=`0.757x`.

## Interpretation

- Current pair execution reduces geometry/SH/weight work, but TP/scatter rows remain edge-major.
- Therefore benefits should track edge load and repeated geometric work more than raw atom count alone.
- Remaining skips are dominated by graph-only mirrors, missing raw files, or gated access rather than public download availability.

## Skipped Datasets

- `md17_aspirin`: graph-only parquet; atomic positions unavailable
- `md17_benzene`: graph-only parquet; atomic positions unavailable
- `md17_ethanol`: no parquet files found
- `md17_malonaldehyde`: graph-only parquet; atomic positions unavailable
- `md17_naphthalene`: graph-only parquet; atomic positions unavailable
- `md17_salicylic_acid`: graph-only parquet; atomic positions unavailable
- `md17_toluene`: graph-only parquet; atomic positions unavailable
- `md17_uracil`: graph-only parquet; atomic positions unavailable
- `omol25_official_gated`: gated dataset

