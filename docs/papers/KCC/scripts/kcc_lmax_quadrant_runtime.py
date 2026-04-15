from __future__ import annotations

import math
import sys
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sevenn._keys as KEY
import sevenn.nn.convolution as conv_mod
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import chemical_species_preprocess


REPO_ROOT = Path(__file__).resolve().parents[4]
KCC_ROOT = REPO_ROOT / "docs" / "papers" / "KCC"
if str(KCC_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(KCC_ROOT / "scripts"))

from kcc_common import supported_dataset_samples  # noqa: E402


OUTPUT_ROOT = KCC_ROOT / "lmax_quadrant_runtime"
METRIC_ROOT = OUTPUT_ROOT / "metrics"
FIG_ROOT = OUTPUT_ROOT / "figures"
REPORT_ROOT = OUTPUT_ROOT / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTOFF = 5.0
L_VALUES = list(range(1, 9))
LATENCY_WARMUP = 10
LATENCY_REPEAT = 30
PROFILE_WARMUP = 5
PROFILE_REPEAT = 30

QUADRANT_DATASETS = OrderedDict(
    [
        ("small_sparse", "spice_2023"),
        ("small_dense", "salex_val_official"),
        ("large_sparse", "oc20_s2ef_train_20m"),
        ("large_dense", "mptrj"),
    ]
)

QUADRANT_COLORS = {
    "small_sparse": "#0072B2",
    "small_dense": "#D55E00",
    "large_sparse": "#009E73",
    "large_dense": "#CC79A7",
}

STAGE_COLORS = {
    "spherical_harmonics_ms": "#E69F00",
    "tp_total_ms": "#4D4D4D",
    "force_output_ms": "#3B4C63",
    "weight_nn_total_ms": "#009E73",
    "radial_total_ms": "#56B4E9",
    "other_ms": "#B0B0B0",
}


def _ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, METRIC_ROOT, FIG_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _measure(fn: Callable[[], None], warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    times: list[float] = []
    for _ in range(repeat):
        if DEVICE.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    ser = pd.Series(times)
    return {
        "mean_ms": float(ser.mean()),
        "std_ms": float(ser.std(ddof=1)),
        "median_ms": float(ser.median()),
        "p95_ms": float(ser.quantile(0.95)),
    }


def _model_config(lmax: int, species: list[str]) -> Dict[str, Any]:
    cfg = {
        "cutoff": CUTOFF,
        "channel": 16,
        "radial_basis": {"radial_basis_name": "bessel", "bessel_basis_num": 8},
        "cutoff_function": {"cutoff_function_name": "poly_cut", "poly_cut_p_value": 6},
        "interaction_type": "nequip",
        "lmax": lmax,
        "is_parity": False,
        "num_convolution_layer": 3,
        "weight_nn_hidden_neurons": [64, 64],
        "act_radial": "silu",
        "act_scalar": {"e": "silu", "o": "tanh"},
        "act_gate": {"e": "silu", "o": "tanh"},
        "conv_denominator": 20.0,
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
    cfg.update(chemical_species_preprocess(species))
    return cfg


def _graph_from_atoms(atoms) -> AtomGraphData:
    return AtomGraphData.from_numpy_dict(
        dl.unlabeled_atoms_to_graph(atoms, CUTOFF, with_shift=False)
    ).to(DEVICE)


class StageTimer:
    def __init__(self) -> None:
        self.times_ms: Dict[str, float] = defaultdict(float)

    def reset(self) -> None:
        self.times_ms.clear()

    @contextmanager
    def section(self, key: str):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.times_ms[key] += (time.perf_counter() - t0) * 1000.0


def _patch_method(
    obj: Any, attr: str, wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Callable[[], None]:
    original = getattr(obj, attr)
    setattr(obj, attr, wrapper_factory(original))

    def restore() -> None:
        setattr(obj, attr, original)

    return restore


def _edge_forward_profile(module: Any, data: AtomGraphData, timed_call: Callable[..., Any]) -> AtomGraphData:
    rvec = data[KEY.EDGE_VEC]
    r = timed_call("edge_length_norm_ms", torch.linalg.norm, rvec, dim=-1)
    data[KEY.EDGE_LENGTH] = r
    basis = timed_call("radial_basis_ms", module.basis_function, r)
    cutoff = timed_call("cutoff_ms", module.cutoff_function, r)
    data[KEY.EDGE_EMBEDDING] = timed_call("radial_combine_ms", torch.mul, basis, cutoff.unsqueeze(-1))
    data[KEY.EDGE_ATTR] = timed_call("spherical_harmonics_ms", module.spherical, rvec)
    return data


def _conv_forward_profile(
    name: str, module: Any, data: AtomGraphData, timed_call: Callable[..., Any]
) -> AtomGraphData:
    x = data[module.key_x]
    edge_index = data[module.key_edge_idx]
    edge_src = edge_index[1]
    edge_dst = edge_index[0]
    weight = timed_call(f"{name}.weight_nn_ms", module.weight_nn, data[module.key_weight_input])
    x_src = timed_call(f"{name}.src_gather_ms", x.index_select, 0, edge_src)
    msg = timed_call(
        f"{name}.message_tp_ms",
        module.convolution,
        x_src,
        data[module.key_filter],
        weight,
    )
    out = timed_call(f"{name}.aggregation_ms", conv_mod.message_gather, x, edge_dst, msg)
    out = timed_call(f"{name}.denominator_ms", torch.div, out, module.denominator)
    data[module.key_x] = out
    return data


def _force_forward_profile(original: Callable[..., Any], data: AtomGraphData, timed_call: Callable[..., Any]) -> AtomGraphData:
    return timed_call("top_force_output_ms", original, data)


def profile_model_stages(model: Any, raw_graph: AtomGraphData) -> pd.DataFrame:
    timer = StageTimer()
    restore_stack: list[Callable[[], None]] = []

    def timed_call(key: str, fn: Callable[..., Any], *args, **kwargs):
        with timer.section(key):
            return fn(*args, **kwargs)

    edge_module = model._modules["edge_embedding"]
    restore_stack.append(
        _patch_method(
            edge_module,
            "forward",
            lambda _orig: lambda data: _edge_forward_profile(edge_module, data, timed_call),
        )
    )

    for name, module in model._modules.items():
        if not name.endswith("_convolution"):
            continue
        restore_stack.append(
            _patch_method(
                module,
                "forward",
                lambda _orig, name=name, module=module: lambda data: _conv_forward_profile(
                    name, module, data, timed_call
                ),
            )
        )

    force_module = model._modules.get("force_output")
    if force_module is not None:
        restore_stack.append(
            _patch_method(
                force_module,
                "forward",
                lambda original: lambda data: _force_forward_profile(original, data, timed_call),
            )
        )

    rows: list[dict[str, Any]] = []
    try:
        for rep in range(PROFILE_REPEAT):
            timer.reset()
            data = raw_graph.clone()
            with timer.section("model_total_ms"):
                _ = model(data)
            for stage, value in timer.times_ms.items():
                rows.append({"repeat": rep, "stage": stage, "time_ms": value})
    finally:
        for restore in reversed(restore_stack):
            restore()
    return pd.DataFrame(rows)


def _aggregate_stage_rows(stage_df: pd.DataFrame) -> pd.DataFrame:
    grouped_rows = []
    conv_tp = [c for c in stage_df["stage"].unique() if c.endswith(".message_tp_ms")]
    conv_w = [c for c in stage_df["stage"].unique() if c.endswith(".weight_nn_ms")]
    conv_g = [c for c in stage_df["stage"].unique() if c.endswith(".src_gather_ms") or c.endswith(".aggregation_ms")]

    for rep, part in stage_df.groupby("repeat"):
        stage_map = dict(zip(part["stage"], part["time_ms"]))
        radial_total = (
            stage_map.get("radial_basis_ms", 0.0)
            + stage_map.get("cutoff_ms", 0.0)
            + stage_map.get("radial_combine_ms", 0.0)
        )
        tp_total = sum(stage_map.get(k, 0.0) for k in conv_tp)
        weight_total = sum(stage_map.get(k, 0.0) for k in conv_w)
        gather_total = sum(stage_map.get(k, 0.0) for k in conv_g)
        sh = stage_map.get("spherical_harmonics_ms", 0.0)
        force = stage_map.get("top_force_output_ms", 0.0)
        total = stage_map.get("model_total_ms", 0.0)
        known = radial_total + tp_total + weight_total + gather_total + sh + force
        other = max(total - known, 0.0)
        grouped_rows.extend(
            [
                {"repeat": rep, "stage_group": "radial_total_ms", "time_ms": radial_total},
                {"repeat": rep, "stage_group": "weight_nn_total_ms", "time_ms": weight_total},
                {"repeat": rep, "stage_group": "gather_total_ms", "time_ms": gather_total},
                {"repeat": rep, "stage_group": "tp_total_ms", "time_ms": tp_total},
                {"repeat": rep, "stage_group": "spherical_harmonics_ms", "time_ms": sh},
                {"repeat": rep, "stage_group": "force_output_ms", "time_ms": force},
                {"repeat": rep, "stage_group": "other_ms", "time_ms": other},
                {"repeat": rep, "stage_group": "model_total_ms", "time_ms": total},
            ]
        )
    return pd.DataFrame(grouped_rows)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = list(QUADRANT_DATASETS.values())
    sample_lookup = {s.dataset: s for s in supported_dataset_samples(selected)}
    summary_rows: list[dict[str, Any]] = []
    raw_stage_rows: list[dict[str, Any]] = []
    stage_summary_rows: list[dict[str, Any]] = []

    for quadrant, dataset in QUADRANT_DATASETS.items():
        sample = sample_lookup[dataset]
        atoms = sample.atoms
        species = sorted(set(atoms.get_chemical_symbols()))
        raw_graph = _graph_from_atoms(atoms)
        num_edges = int(raw_graph[KEY.EDGE_IDX].shape[1])
        avg_neighbors = float(num_edges / max(len(atoms), 1))

        for lmax in L_VALUES:
            model = build_E3_equivariant_model(_model_config(lmax, species), parallel=False).to(DEVICE)
            model.eval()
            model.set_is_batch_data(False)

            def one_step() -> None:
                _ = model(raw_graph.clone())

            latency = _measure(one_step, warmup=LATENCY_WARMUP, repeat=LATENCY_REPEAT)
            params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            summary_rows.append(
                {
                    "quadrant": quadrant,
                    "dataset": dataset,
                    "natoms": len(atoms),
                    "num_edges": num_edges,
                    "avg_neighbors_directed": avg_neighbors,
                    "lmax": lmax,
                    "trainable_params": params,
                    **latency,
                }
            )

            stage_raw = profile_model_stages(model, raw_graph)
            stage_raw["quadrant"] = quadrant
            stage_raw["dataset"] = dataset
            stage_raw["lmax"] = lmax
            raw_stage_rows.extend(stage_raw.to_dict(orient="records"))

            grouped = _aggregate_stage_rows(stage_raw)
            grouped["quadrant"] = quadrant
            grouped["dataset"] = dataset
            grouped["lmax"] = lmax
            stage_summary = (
                grouped.groupby(["quadrant", "dataset", "lmax", "stage_group"], as_index=False)
                .agg(mean_ms=("time_ms", "mean"), std_ms=("time_ms", "std"))
            )
            stage_summary_rows.extend(stage_summary.to_dict(orient="records"))

            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return (
        pd.DataFrame(summary_rows).sort_values(["quadrant", "lmax"]),
        pd.DataFrame(raw_stage_rows),
        pd.DataFrame(stage_summary_rows).sort_values(["quadrant", "lmax", "stage_group"]),
    )


def _save_plots(summary_df: pd.DataFrame, stage_summary_df: pd.DataFrame, acc_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()
    for ax, (quadrant, _) in zip(axes, QUADRANT_DATASETS.items()):
        part = summary_df[summary_df["quadrant"] == quadrant]
        ax.errorbar(
            part["lmax"],
            part["mean_ms"],
            yerr=part["std_ms"],
            marker="o",
            capsize=3,
            color=QUADRANT_COLORS[quadrant],
        )
        ax.set_title(f"{quadrant} / {part['dataset'].iloc[0]}")
        ax.set_xlabel("lmax")
        ax.set_ylabel("Step latency (ms)")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "quadrant_lmax_latency.png", dpi=300)
    fig.savefig(FIG_ROOT / "quadrant_lmax_latency.svg")
    plt.close(fig)

    norm_rows = []
    for quadrant, part in summary_df.groupby("quadrant"):
        base = float(part.loc[part["lmax"] == 1, "mean_ms"].iloc[0])
        for _, row in part.iterrows():
            norm_rows.append({**row.to_dict(), "latency_growth_x": row["mean_ms"] / base})
    norm_df = pd.DataFrame(norm_rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for quadrant in QUADRANT_DATASETS:
        part = norm_df[norm_df["quadrant"] == quadrant]
        ax.plot(part["lmax"], part["latency_growth_x"], marker="o", label=quadrant, color=QUADRANT_COLORS[quadrant])
    ax.set_xlabel("lmax")
    ax.set_ylabel("Latency growth (x vs lmax=1)")
    ax.set_title("How quickly runtime grows with lmax")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "quadrant_lmax_latency_growth.png", dpi=300)
    fig.savefig(FIG_ROOT / "quadrant_lmax_latency_growth.svg")
    plt.close(fig)

    focus = stage_summary_df[stage_summary_df["stage_group"].isin(["spherical_harmonics_ms", "tp_total_ms", "force_output_ms", "model_total_ms"])].copy()
    pivot = focus.pivot_table(index=["quadrant", "dataset", "lmax"], columns="stage_group", values="mean_ms").reset_index()
    pivot["sh_share"] = pivot["spherical_harmonics_ms"] / pivot["model_total_ms"]
    pivot["tp_share"] = pivot["tp_total_ms"] / pivot["model_total_ms"]
    pivot["force_share"] = pivot["force_output_ms"] / pivot["model_total_ms"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    for ax, col, title in zip(
        axes,
        ["sh_share", "tp_share", "force_share"],
        ["SH share", "TP share", "Force share"],
    ):
        for quadrant in QUADRANT_DATASETS:
            part = pivot[pivot["quadrant"] == quadrant]
            ax.plot(part["lmax"], part[col] * 100.0, marker="o", label=quadrant, color=QUADRANT_COLORS[quadrant])
        ax.set_xlabel("lmax")
        ax.set_ylabel("Share of model time (%)")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "quadrant_lmax_stage_shares.png", dpi=300)
    fig.savefig(FIG_ROOT / "quadrant_lmax_stage_shares.svg")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(acc_df["lmax"], acc_df["test_force_rmse"], marker="o", color="#D55E00", label="azobenzene force RMSE")
    ax1.set_xlabel("lmax")
    ax1.set_ylabel("Force RMSE (eV/A)", color="#D55E00")
    ax2 = ax1.twinx()
    large_dense = norm_df[norm_df["quadrant"] == "large_dense"]
    ax2.plot(large_dense["lmax"], large_dense["latency_growth_x"], marker="s", color="#0072B2", label="large_dense latency growth")
    ax2.set_ylabel("Latency growth (x vs lmax=1)", color="#0072B2")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "azobenzene_accuracy_vs_large_dense_cost.png", dpi=300)
    fig.savefig(FIG_ROOT / "azobenzene_accuracy_vs_large_dense_cost.svg")
    plt.close(fig)


def _write_report(summary_df: pd.DataFrame, stage_summary_df: pd.DataFrame, acc_df: pd.DataFrame) -> None:
    pivot = stage_summary_df.pivot_table(index=["quadrant", "dataset", "lmax"], columns="stage_group", values="mean_ms").reset_index()
    pivot["sh_share_pct"] = pivot["spherical_harmonics_ms"] / pivot["model_total_ms"] * 100.0
    pivot["tp_share_pct"] = pivot["tp_total_ms"] / pivot["model_total_ms"] * 100.0
    pivot["force_share_pct"] = pivot["force_output_ms"] / pivot["model_total_ms"] * 100.0

    lines = []
    for quadrant in QUADRANT_DATASETS:
        part = summary_df[summary_df["quadrant"] == quadrant].sort_values("lmax")
        latency_growth = part.iloc[-1]["mean_ms"] / part.iloc[0]["mean_ms"]
        p1 = pivot[(pivot["quadrant"] == quadrant) & (pivot["lmax"] == 1)].iloc[0]
        p8 = pivot[(pivot["quadrant"] == quadrant) & (pivot["lmax"] == 8)].iloc[0]
        lines.append(
            f"- {quadrant} ({part.iloc[0]['dataset']}): latency {part.iloc[0]['mean_ms']:.3f} -> {part.iloc[-1]['mean_ms']:.3f} ms ({latency_growth:.2f}x), "
            f"SH share {p1['sh_share_pct']:.2f}% -> {p8['sh_share_pct']:.2f}%, "
            f"TP share {p1['tp_share_pct']:.2f}% -> {p8['tp_share_pct']:.2f}%, "
            f"force share {p1['force_share_pct']:.2f}% -> {p8['force_share_pct']:.2f}%"
        )

    best_acc = acc_df.loc[acc_df["test_force_rmse"].idxmin()]
    report = f"""# lmax와 데이터셋 크기/밀도 메모

이 문서는 `lmax`를 올릴 때 비용이 어떤 그래프에서 더 급격히 늘어나는지 보기 위한 보조 실험 메모다.

## 전제

- 정확도 실험은 현재 `rMD17 azobenzene` baseline-only sweep에서 직접 검증했다.
- 시간 실험은 대표 그래프 4개에서, 같은 baseline 구조를 랜덤 초기화 상태로 만들어 측정했다.
- 이 시간 실험은 학습된 가중치의 차이가 아니라 그래프 크기와 밀도, 그리고 `lmax` 증가가 만드는 구조적 비용을 보기 위한 것이다.

## 대표 데이터셋

- small_sparse: `{QUADRANT_DATASETS['small_sparse']}`
- small_dense: `{QUADRANT_DATASETS['small_dense']}`
- large_sparse: `{QUADRANT_DATASETS['large_sparse']}`
- large_dense: `{QUADRANT_DATASETS['large_dense']}`

선정 기준은 현재 public-local benchmark 샘플 분포의 `natoms`와 `avg_neighbors_directed` 중앙값을 기준으로 한 사분면이다.

## 정확도 쪽에서 이미 확인된 것

- `rMD17 azobenzene`에서는 `lmax=7`이 가장 좋은 힘 정확도(`{best_acc['test_force_rmse']:.6f} eV/A`)를 보였다.
- `lmax=8`은 더 비싸지만 정확도가 다시 약간 나빠졌다.
- 따라서 `lmax`가 높을수록 정확도가 무한정 좋아진다고 쓰면 안 된다.

## 시간/프로파일 쪽 핵심 관찰

{chr(10).join(lines)}

## 해석

- `lmax`는 spherical harmonics의 최대 차수이고, edge에서 다루는 각도 표현 차원을 직접 늘린다.
- 그러나 실제 비용은 SH 하나만의 문제가 아니다.
- `lmax`가 올라가면 SH 차원뿐 아니라 hidden irreps, edge irreps, output irreps 사이의 tensor product 경로 수가 함께 증가한다.
- 그래서 large/dense 그래프에서는 `TP`와 `force backward` 쪽 시간이 더 빠르게 커진다.
- small/sparse 그래프에서는 절대 시간이 작아서 증가 폭이 상대적으로 덜 커 보일 수 있다.

## 현재 단계에서 방어 가능한 메시지

- `lmax`는 모델 설계자가 정하는 값이며, 데이터셋이 자동으로 정해주지 않는다.
- 높은 `lmax`는 더 풍부한 각도 표현을 가능하게 하지만, 실제 최적값은 데이터셋과 학습 조건에 따라 달라진다.
- 큰 graph, 특히 dense graph일수록 `lmax` 증가가 시간 비용으로 더 크게 나타난다.
- 따라서 geometry-side 비용을 줄이는 현재 제안기법의 가치는 높은 `lmax`와 large/dense graph에서 더 커질 가능성이 있다.
- 다만 이 문장 자체는 시간 구조에 대한 해석이고, 정확도 이득까지 일반화하려면 각 데이터셋별 재학습이 추가로 필요하다.
"""
    (REPORT_ROOT / "lmax_quadrant_runtime_report.md").write_text(report)


def main() -> None:
    _ensure_dirs()
    summary_df, raw_stage_df, stage_summary_df = collect()
    summary_df.to_csv(METRIC_ROOT / "quadrant_lmax_latency_summary.csv", index=False)
    raw_stage_df.to_csv(METRIC_ROOT / "quadrant_lmax_stage_raw.csv", index=False)
    stage_summary_df.to_csv(METRIC_ROOT / "quadrant_lmax_stage_summary.csv", index=False)

    acc_df = pd.read_csv(KCC_ROOT / "lmax_baseline_sweep" / "metrics" / "lmax_sweep_summary.csv")
    _save_plots(summary_df, stage_summary_df, acc_df)
    _write_report(summary_df, stage_summary_df, acc_df)


if __name__ == "__main__":
    main()
