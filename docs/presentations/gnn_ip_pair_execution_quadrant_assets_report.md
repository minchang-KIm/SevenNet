# Quadrant Comparison Asset Report

세미나 발표용 `small/large × sparse/dense` 비교 자원은 아래 경로에 생성했다.

## Main Assets

- Representative table: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/quadrant_representatives.csv`
- All benchmark points: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/all_dataset_points.csv`
- Large dataset status table: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/large_dataset_status.csv`

## Plots

- Dataset map on `num_edges × avg_neighbors`: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/plots/quadrant_dataset_map.png`
- Baseline vs pair latency on four representatives: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/plots/quadrant_latency_comparison.png`
- Speedup bar on four representatives: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/plots/quadrant_speedup_comparison.png`
- Largest public dataset download status: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/plots/large_dataset_download_status.png`
- Extreme stage breakdown (`Small Sparse` vs `Large Dense`): `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/plots/extreme_stage_breakdown.png`

## Diagrams

- Mechanism diagram PNG: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/diagrams/quadrant_mechanism_diagram.png`
- Mechanism diagram SVG: `/home/wise/minchang/DenseMLIP/SevenNet/docs/presentations/assets/quadrant_pack/diagrams/quadrant_mechanism_diagram.svg`

## Selected Representative Datasets

Quadrant framing used in the comparison figure:

- `large`: representative sample `num_edges >= 3000`
- `dense`: representative sample `avg_neighbors_directed >= 40`

- `Small Sparse`: `SPICE 2023`
- `Small Dense`: `phononDB PBE`
- `Large Sparse`: `OMol25 validation`
- `Large Dense`: `MPtrj validation`

주의:

- `OMol25 validation`은 기존 benchmark 결과에는 포함되어 있지만, 현재 repo의 `datasets/raw` 인벤토리 기준으로는 아직 `pending` 상태다.
- 따라서 발표에서는 "benchmark evidence exists"와 "repo local raw is fully downloaded"를 구분해서 말하는 것이 안전하다.
