from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ase.build import bulk, molecule

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    PolynomialCutoff,
    SphericalEncoding,
)
from sevenn.util import chemical_species_preprocess


OUTPUT_ROOT = Path("docs/papers/KCC")
METRIC_DIR = OUTPUT_ROOT / "metrics" / "lmax_strength"
FIG_DIR = OUTPUT_ROOT / "figures" / "lmax_strength"
REPORT_DIR = OUTPUT_ROOT / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
WARMUP = 20
REPEAT = 30
CUTOFF = 4.0
L_VALUES = [1, 2, 3, 4]

CASE_COLORS = {
    "baseline_sh": "#0072B2",
    "pair_sh": "#D55E00",
    "baseline_edge_embedding": "#0072B2",
    "pair_edge_embedding": "#D55E00",
}


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _measure(fn: Callable[[], None], *, warmup: int = WARMUP, repeat: int = REPEAT) -> Dict[str, float]:
    if DEVICE.type == "cuda":
        for _ in range(warmup):
            fn()
        _sync()

        times: List[float] = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    else:
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": float(pd.Series(times).mean()),
        "std_ms": float(pd.Series(times).std(ddof=1)),
        "median_ms": float(pd.Series(times).median()),
        "p95_ms": float(pd.Series(times).quantile(0.95)),
    }


def _graph_from_atoms(name: str) -> AtomGraphData:
    if name == "small_benzene":
        atoms = molecule("C6H6")
    elif name == "large_nacl_10x10x10":
        atoms = bulk("NaCl", "rocksalt", a=5.63).repeat((10, 10, 10))
    else:
        raise ValueError(name)
    graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(atoms, CUTOFF))
    return graph


def _pair_graph(graph: AtomGraphData) -> AtomGraphData:
    data = graph.clone()
    data = data.to(DEVICE)
    meta = pair_runtime.build_pair_metadata(
        data[KEY.EDGE_IDX],
        data[KEY.EDGE_VEC],
        cell_shift=data.get(KEY.CELL_SHIFT),
    )
    for key, value in meta.items():
        data[key] = value
    return data


def _baseline_graph(graph: AtomGraphData) -> AtomGraphData:
    return graph.clone().to(DEVICE)


def _model_config(lmax: int) -> Dict:
    cfg = {
        "cutoff": CUTOFF,
        "channel": 4,
        "radial_basis": {"radial_basis_name": "bessel"},
        "cutoff_function": {"cutoff_function_name": "poly_cut"},
        "interaction_type": "nequip",
        "lmax": lmax,
        "is_parity": True,
        "num_convolution_layer": 3,
        "weight_nn_hidden_neurons": [64, 64],
        "act_radial": "silu",
        "act_scalar": {"e": "silu", "o": "tanh"},
        "act_gate": {"e": "silu", "o": "tanh"},
        "conv_denominator": 30.0,
        "train_denominator": False,
        "self_connection_type": "nequip",
        "shift": -10.0,
        "scale": 10.0,
        "train_shift_scale": False,
        "irreps_manual": False,
        "lmax_edge": -1,
        "lmax_node": -1,
        "readout_as_fcn": False,
        "use_bias_in_linear": False,
        "_normalize_sph": True,
    }
    cfg.update(chemical_species_preprocess(["Na", "Cl"]))
    return cfg


def _edge_module(lmax: int, policy: str) -> EdgeEmbedding:
    return EdgeEmbedding(
        basis_module=BesselBasis(cutoff_length=CUTOFF),
        cutoff_module=PolynomialCutoff(cutoff_length=CUTOFF),
        spherical_module=SphericalEncoding(lmax, parity=-1, normalize=True),
        pair_execution_policy=policy,
    ).to(DEVICE)


def _structural_rows() -> pd.DataFrame:
    rows = []
    for lmax in L_VALUES:
        model = build_E3_equivariant_model(_model_config(lmax), parallel=False).to(DEVICE)
        edge = model._modules["edge_embedding"]
        conv = model._modules["0_convolution"]
        rows.append(
            {
                "lmax": lmax,
                "sh_dim": int(edge.spherical.irreps_out.dim),
                "sh_irreps": str(edge.spherical.irreps_out),
                "conv_instruction_count": int(len(conv.convolution.instructions)),
                "conv_weight_numel": int(conv.convolution.weight_numel),
                "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            }
        )
    return pd.DataFrame(rows)


def _timing_rows() -> pd.DataFrame:
    rows = []
    graph_specs = {
        "small_benzene": _graph_from_atoms("small_benzene"),
        "large_nacl_10x10x10": _graph_from_atoms("large_nacl_10x10x10"),
    }
    for graph_name, graph in graph_specs.items():
        baseline_graph = _baseline_graph(graph)
        pair_graph = _pair_graph(graph)
        reverse_sign_cache: Dict[int, torch.Tensor] = {}
        for lmax in L_VALUES:
            sph = SphericalEncoding(lmax, parity=-1, normalize=True).to(DEVICE)
            baseline_edge = _edge_module(lmax, "baseline")
            pair_edge = _edge_module(lmax, "geometry_only")

            if lmax not in reverse_sign_cache:
                reverse_sign_cache[lmax] = pair_edge.reverse_sh_sign.to(DTYPE).unsqueeze(0)
            reverse_sign = reverse_sign_cache[lmax]

            rvec = baseline_graph[KEY.EDGE_VEC]
            pair_rvec = pair_graph[KEY.PAIR_EDGE_VEC]
            pair_map = pair_graph[KEY.EDGE_PAIR_MAP]
            reverse_mask = pair_graph[KEY.EDGE_PAIR_REVERSE].to(DTYPE).unsqueeze(-1)

            def baseline_sh() -> None:
                _ = sph(rvec)

            def pair_sh_kernel_only() -> None:
                _ = sph(pair_rvec)

            def pair_sh() -> None:
                pair_attr = sph(pair_rvec)
                edge_attr = pair_attr.index_select(0, pair_map)
                sign = 1.0 + reverse_mask * (reverse_sign - 1.0)
                _ = edge_attr * sign

            def baseline_edge_embedding() -> None:
                baseline_edge(baseline_graph)

            def pair_edge_embedding() -> None:
                pair_edge(pair_graph)

            for case_name, fn in (
                ("baseline_sh", baseline_sh),
                ("pair_sh_kernel_only", pair_sh_kernel_only),
                ("pair_sh", pair_sh),
                ("baseline_edge_embedding", baseline_edge_embedding),
                ("pair_edge_embedding", pair_edge_embedding),
            ):
                row = {
                    "graph_name": graph_name,
                    "num_edges": int(baseline_graph[KEY.EDGE_IDX].shape[1]),
                    "num_pairs": int(pair_graph[KEY.PAIR_EDGE_VEC].shape[0]),
                    "lmax": lmax,
                    "case": case_name,
                    "device": DEVICE.type,
                    "warmup": WARMUP,
                    "repeat": REPEAT,
                }
                row.update(_measure(fn))
                rows.append(row)
    return pd.DataFrame(rows)


def _summary_rows(timing_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for graph_name in sorted(timing_df["graph_name"].unique()):
        graph_df = timing_df[timing_df["graph_name"] == graph_name]
        for lmax in L_VALUES:
            part = graph_df[graph_df["lmax"] == lmax].set_index("case")
            rows.append(
                {
                    "graph_name": graph_name,
                    "lmax": lmax,
                    "baseline_sh_mean_ms": part.loc["baseline_sh", "mean_ms"],
                    "pair_sh_kernel_only_mean_ms": part.loc["pair_sh_kernel_only", "mean_ms"],
                    "sh_kernel_only_speedup_x": (
                        part.loc["baseline_sh", "mean_ms"] / part.loc["pair_sh_kernel_only", "mean_ms"]
                    ),
                    "pair_sh_mean_ms": part.loc["pair_sh", "mean_ms"],
                    "sh_speedup_x": part.loc["baseline_sh", "mean_ms"] / part.loc["pair_sh", "mean_ms"],
                    "baseline_edge_embedding_mean_ms": part.loc["baseline_edge_embedding", "mean_ms"],
                    "pair_edge_embedding_mean_ms": part.loc["pair_edge_embedding", "mean_ms"],
                    "edge_embedding_speedup_x": (
                        part.loc["baseline_edge_embedding", "mean_ms"] / part.loc["pair_edge_embedding", "mean_ms"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _save_plot_sh_dim(struct_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(struct_df["lmax"], struct_df["sh_dim"], marker="o", color="#0072B2", label="SH dimension")
    ax.set_xlabel("lmax")
    ax.set_ylabel("SH dimension per edge")
    ax.set_title("Spherical Harmonics Dimension vs lmax")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lmax_sh_dimension.png", dpi=300)
    fig.savefig(FIG_DIR / "lmax_sh_dimension.svg")
    plt.close(fig)


def _save_plot_params(struct_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(struct_df["lmax"], struct_df["trainable_params"], marker="o", color="#D55E00", label="Parameters")
    ax.set_xlabel("lmax")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Model Capacity Growth with lmax")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lmax_model_params.png", dpi=300)
    fig.savefig(FIG_DIR / "lmax_model_params.svg")
    plt.close(fig)


def _save_plot_timing(summary_df: pd.DataFrame, case_prefix: str, filename: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, graph_name in zip(axes, sorted(summary_df["graph_name"].unique())):
        part = summary_df[summary_df["graph_name"] == graph_name]
        ax.plot(
            part["lmax"],
            part[f"baseline_{case_prefix}_mean_ms"],
            marker="o",
            color="#0072B2",
            label="Baseline",
        )
        ax.plot(
            part["lmax"],
            part[f"pair_{case_prefix}_mean_ms"],
            marker="o",
            color="#D55E00",
            label="Pair reuse",
        )
        ax.set_title(graph_name.replace("_", " "))
        ax.set_xlabel("lmax")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    fig.savefig(FIG_DIR / filename.replace(".png", ".svg"))
    plt.close(fig)


def _save_plot_speedup(summary_df: pd.DataFrame, column: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for graph_name, color in (("small_benzene", "#0072B2"), ("large_nacl_10x10x10", "#D55E00")):
        part = summary_df[summary_df["graph_name"] == graph_name]
        ax.plot(part["lmax"], part[column], marker="o", label=graph_name.replace("_", " "), color=color)
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel("lmax")
    ax.set_ylabel("Speedup (baseline / pair)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    fig.savefig(FIG_DIR / filename.replace(".png", ".svg"))
    plt.close(fig)


def _write_report(struct_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    rows = []
    for _, r in struct_df.iterrows():
        rows.append(
            f"| {int(r['lmax'])} | {int(r['sh_dim'])} | {int(r['conv_instruction_count'])} | "
            f"{int(r['conv_weight_numel'])} | {int(r['trainable_params'])} |"
        )
    struct_table = "\n".join(rows)

    speedup_rows = []
    for _, r in summary_df.iterrows():
        speedup_rows.append(
            f"| {r['graph_name']} | {int(r['lmax'])} | {r['sh_kernel_only_speedup_x']:.3f} | "
            f"{r['sh_speedup_x']:.3f} | {r['edge_embedding_speedup_x']:.3f} |"
        )
    speedup_table = "\n".join(speedup_rows)

    report = f"""# lmax와 Spherical Harmonics 재사용 메모

이 문서는 논문 본문을 다시 정리하지 않고, `lmax`가 왜 중요한지와 현재 제안기법이 이 축에서 어디에 강점을 가질 수 있는지를 정리해 둔 작업 메모다.

## 1. 공식 문서 기준으로 보통 어떤 `lmax`를 쓰는가

- NequIP 공식 문서는 `l_max=1`을 좋은 기본값으로 소개하고, `l_max=2`는 더 정확하지만 더 느리다고 설명한다. 또한 foundation preset은 `S/M/L/XL = l_max 1/2/3/4`로 제공한다.  
  출처: https://nequip.readthedocs.io/en/latest/_modules/nequip/model/nequip_models.html
- SevenNet 공식 문서는 `SevenNet-0`이 `lmax=2`, `SevenNet-l3i5`가 `lmax=3`이라고 밝히며, `l3i5`는 정확도는 좋아지지만 `SevenNet-0`보다 약 4배 느리다고 설명한다. 최신 Omni/MF/OMAT 계열도 모두 `lmax=3`을 사용한다.  
  출처: https://github.com/MDIL-SNU/SevenNet/blob/main/docs/source/user_guide/pretrained.md

정리하면, 실무적으로는 `1`이 아주 가벼운 기본값이고, `2`와 `3`이 더 흔한 정확도 지향 설정이며, `4`는 큰 모델에서만 쓰는 편이라고 보는 것이 맞다.

## 2. 이론적으로 `l`이 커지면 무엇이 좋아지고 무엇이 비싸지는가

- `lmax`는 각 edge에서 표현할 수 있는 각도 정보의 해상도를 정한다.
- `l=0`은 방향성이 없는 스칼라 성분이다.
- `l=1`은 1차 방향 정보를 담는다.
- `l=2`, `l=3`로 갈수록 더 복잡한 각도 변화를 표현할 수 있다.
- 따라서 방향성 결합이 강한 공유결합, 분자, 표면 흡착, 저대칭 결정 구조에서는 더 높은 `l`이 도움이 될 수 있다.

반대로 비용도 커진다.

- spherical harmonics 출력 차원은 `sum_(l=0)^L (2l+1) = (L+1)^2`로 증가한다.
- 즉 `lmax=1,2,3,4`에서 SH 차원은 `4, 9, 16, 25`로 커진다.
- 이 증가는 SH 단계뿐 아니라, 이후 irreps와 tensor product 경로 수, 파라미터 수 증가로 이어진다.
- 따라서 이론적으로도 `최적의 lmax`는 하나로 고정되지 않는다. 각도 표현력이 필요한 문제에서는 올리는 것이 이득이고, 그렇지 않으면 비용만 커진다.

## 3. 로컬 구조 확인

| lmax | SH dim | TP instructions (1st conv) | TP weight numel | Trainable params |
| --- | ---: | ---: | ---: | ---: |
{struct_table}

이 표는 같은 SevenNet 구조에서 `lmax`만 바꿨을 때의 변화다. 적어도 현재 코드 기준으로는 `lmax`가 올라갈수록 SH 차원은 정확히 제곱 꼴로 커지고, 파라미터 수도 함께 증가한다.

## 4. 로컬 timing 검증

측정 환경:

- device: `{DEVICE.type}`
- warmup: `{WARMUP}`
- repeat: `{REPEAT}`
- small graph: benzene (`114` directed edges)
- large graph: NaCl `10x10x10` supercell (`36000` directed edges)

측정한 값:

1. `baseline_sh`: 모든 directed edge에 대해 SH 직접 계산
2. `pair_sh_kernel_only`: pair 기준 SH 한 번만 계산
3. `pair_sh`: pair 기준 SH 한 번 계산 후 reverse는 sign flip으로 복원
4. `baseline_edge_embedding`: 기존 edge embedding
5. `pair_edge_embedding`: pair-aware geometry-only edge embedding

pair 적용 속도비:

| graph | lmax | SH kernel-only speedup | SH reconstructed speedup | Edge embedding speedup |
| --- | ---: | ---: | ---: | ---: |
{speedup_table}

## 5. 이 메모에서 바로 가져갈 수 있는 메시지

1. `lmax`가 커질수록 SH가 다루는 성분 수는 빠르게 증가한다.
2. 따라서 SH를 정확히 반으로 줄이는 pair reuse의 절대 절감량도 커질 가능성이 높다.
3. 최신 SevenNet 계열이 `lmax=3`을 쓰고 있다는 점은, 이 방향이 실제로 중요한 영역에 이미 들어와 있다는 뜻이다.
4. 즉 현재 제안기법의 강점은 단순히 "SH를 한 번 덜 계산한다"가 아니라, `lmax`가 커질수록 더 비싸지는 각도 표현 비용을 정확도 변화 없이 줄인다는 점이다.

## 6. 주의할 점

- 이 메모는 아직 논문 본문용으로 다듬지 않았다.
- 현재 timing은 SH/edge-embedding 단계의 국소 실험이다.
- 전체 force-including MD step 성능은 backward, pair metadata, full path 구조에 더 크게 좌우된다.
- 따라서 여기서 바로 "전체 모델이 lmax가 높을수록 pair가 더 빠르다"라고 주장하면 안 된다.
- 대신 "SH 중심의 geometry-side 절감 강도는 lmax가 커질수록 더 중요해진다"까지는 방어 가능하다.
"""
    (REPORT_DIR / "lmax_spherical_harmonics_strength_note.md").write_text(report)


def main() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    struct_df = _structural_rows()
    timing_df = _timing_rows()
    summary_df = _summary_rows(timing_df)

    struct_df.to_csv(METRIC_DIR / "lmax_structural_scaling.csv", index=False)
    timing_df.to_csv(METRIC_DIR / "lmax_timing_raw.csv", index=False)
    summary_df.to_csv(METRIC_DIR / "lmax_speedup_summary.csv", index=False)

    _save_plot_sh_dim(struct_df)
    _save_plot_params(struct_df)
    _save_plot_timing(summary_df, "sh", "lmax_sh_timing.png", "Mean latency (ms)")
    _save_plot_timing(
        summary_df,
        "edge_embedding",
        "lmax_edge_embedding_timing.png",
        "Mean latency (ms)",
    )
    _save_plot_speedup(summary_df, "sh_speedup_x", "lmax_sh_speedup.png", "SH Reuse Speedup vs lmax")
    _save_plot_speedup(
        summary_df,
        "sh_kernel_only_speedup_x",
        "lmax_sh_kernel_only_speedup.png",
        "SH Kernel-Only Speedup vs lmax",
    )
    _save_plot_speedup(
        summary_df,
        "edge_embedding_speedup_x",
        "lmax_edge_embedding_speedup.png",
        "Edge Embedding Reuse Speedup vs lmax",
    )
    _write_report(struct_df, summary_df)


if __name__ == "__main__":
    main()
