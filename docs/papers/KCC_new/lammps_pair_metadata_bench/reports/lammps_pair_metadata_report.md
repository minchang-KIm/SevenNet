# LAMMPS Pair Metadata Bench

- LAMMPS binary: `/home/wise/minchang/DenseMLIP/lammps_sevenn/build/lmp`
- repeats: `30`
- cases: `baseline`, `geometry_only_legacy`, `geometry_only_upstream`
- note: current environment exposes a single GPU, so this report covers serial `pair_style e3gnn` only

## bulk_large

- baseline total: `180.406 ± 7.989 ms`
- geometry_only legacy pair_metadata: `4.636 ± 0.054 ms`
- geometry_only legacy total: `175.248 ± 5.417 ms`
- geometry_only upstream pair_metadata: `0.322 ± 0.029 ms`
- geometry_only upstream total: `173.153 ± 9.628 ms`
- pair_metadata reduction factor: `14.402x`
- geometry_only total reduction factor: `1.012x`

## bulk_small

- baseline total: `186.687 ± 13.058 ms`
- geometry_only legacy pair_metadata: `0.441 ± 0.011 ms`
- geometry_only legacy total: `167.673 ± 6.979 ms`
- geometry_only upstream pair_metadata: `0.100 ± 0.005 ms`
- geometry_only upstream total: `169.793 ± 10.672 ms`
- pair_metadata reduction factor: `4.405x`
- geometry_only total reduction factor: `0.988x`
