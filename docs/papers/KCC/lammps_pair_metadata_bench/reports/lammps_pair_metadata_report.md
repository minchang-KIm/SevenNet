# LAMMPS Pair Metadata Bench

- LAMMPS binary: `/home/wise/minchang/DenseMLIP/lammps_sevenn/build/lmp`
- repeats: `30`
- cases: `baseline`, `geometry_only_legacy`, `geometry_only_upstream`
- note: current environment exposes a single GPU, so this report covers serial `pair_style e3gnn` only

## bulk_large

- baseline total: `173.045 ± 6.574 ms`
- geometry_only legacy pair_metadata: `4.612 ± 0.052 ms`
- geometry_only legacy total: `170.320 ± 7.123 ms`
- geometry_only upstream pair_metadata: `0.304 ± 0.014 ms`
- geometry_only upstream total: `167.296 ± 7.276 ms`
- pair_metadata reduction factor: `15.194x`
- geometry_only total reduction factor: `1.018x`

## bulk_small

- baseline total: `184.012 ± 18.115 ms`
- geometry_only legacy pair_metadata: `0.439 ± 0.013 ms`
- geometry_only legacy total: `168.856 ± 13.518 ms`
- geometry_only upstream pair_metadata: `0.101 ± 0.008 ms`
- geometry_only upstream total: `162.871 ± 4.693 ms`
- pair_metadata reduction factor: `4.347x`
- geometry_only total reduction factor: `1.037x`
