from __future__ import annotations

import csv
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ase import Atoms
from ase.build import bulk

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.convolution import IrrepsConvolution, message_gather
from sevenn.nn.edge_embedding import EdgeEmbedding
from sevenn.train.dataload import unlabeled_atoms_to_graph

from kcc_common import KCC_ROOT, ensure_dir, resolve_device, sync_if_needed


CHECKPOINT = Path("/home/wise/minchang/DenseMLIP/SevenNet/tests/data/checkpoints/cp_0.pth")
OUT_ROOT = KCC_ROOT / "geometry_only_breakdown"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
REPORTS_DIR = OUT_ROOT / "reports"

WARMUP = 5
REPEATS = 30

CASES = (
    {
        "case": "baseline",
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "geometry_only",
        "enable_pair_execution": True,
        "pair_execution_policy": "geometry_only",
    },
)

SYSTEMS = (
    {
        "system": "dimer_small",
        "atoms": Atoms(
            symbols=["Hf", "O"],
            positions=[[0.0, 0.0, 0.0], [1.85, 0.0, 0.0]],
            pbc=False,
        ),
        "category": "small_sparse",
    },
    {
        "system": "bulk_small",
        "atoms": bulk("HfO", "rocksalt", 4.0, orthorhombic=True) * (2, 2, 2),
        "category": "small_dense",
    },
    {
        "system": "bulk_large",
        "atoms": bulk("HfO", "rocksalt", 4.0, orthorhombic=True) * (6, 6, 3),
        "category": "large_dense",
    },
)

STAGE_ORDER = [
    "edge_pair_vec_select_ms",
    "edge_length_norm_ms",
    "radial_basis_cutoff_ms",
    "spherical_harmonics_ms",
    "edge_length_expand_ms",
    "edge_embedding_expand_ms",
    "edge_attr_expand_ms",
    "edge_attr_sign_ms",
    "conv_weight_nn_ms",
    "conv_weight_expand_ms",
    "conv_src_gather_ms",
    "conv_message_tp_ms",
    "conv_message_gather_ms",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_energy_only_model(model):
    if hasattr(model, "delete_module_by_key"):
        model.delete_module_by_key("force_output")
    if hasattr(model, "key_grad"):
        model.key_grad = None
    model.eval()
    return model


def _resolve_pair_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return pair_runtime.resolve_pair_execution_config(
        {
            KEY.PAIR_EXECUTION_CONFIG: {
                "use": case["enable_pair_execution"],
                "policy": case["pair_execution_policy"],
            }
        },
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case["pair_execution_policy"],
    )


def _build_data(
    atoms: Any,
    *,
    cutoff: float,
    pair_cfg: dict[str, Any],
    device: torch.device,
) -> AtomGraphData:
    graph = unlabeled_atoms_to_graph(
        atoms,
        cutoff,
        with_shift=pair_cfg["resolved_policy"] != "baseline",
    )
    data = AtomGraphData.from_numpy_dict(graph)
    data, _ = pair_runtime.prepare_pair_metadata(
        data,
        pair_cfg,
        num_atoms=len(atoms),
    )
    data.to(device)  # type: ignore[arg-type]
    return data


class StageProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._totals: dict[str, float] = defaultdict(float)

    def reset(self) -> None:
        self._totals.clear()

    def timed(self, key: str, fn, *args, **kwargs):
        sync_if_needed(self.device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        sync_if_needed(self.device)
        self._totals[key] += (time.perf_counter() - t0) * 1000.0
        return out

    def totals(self) -> dict[str, float]:
        return dict(self._totals)


def _instrument_model(model, profiler: StageProfiler):
    for module in model.modules():
        if isinstance(module, EdgeEmbedding):
            _patch_edge_embedding(module, profiler)
        elif isinstance(module, IrrepsConvolution):
            _patch_convolution(module, profiler)
    return model


def _patch_edge_embedding(module: EdgeEmbedding, profiler: StageProfiler) -> None:
    def wrapped_forward(self, data):
        if (
            self.pair_execution_policy != "baseline"
            and KEY.PAIR_EDGE_VEC in data
            and KEY.EDGE_PAIR_MAP in data
            and KEY.EDGE_PAIR_REVERSE in data
        ):
            pair_rvec = data[KEY.PAIR_EDGE_VEC]
            if KEY.PAIR_EDGE_FORWARD_INDEX in data and KEY.EDGE_VEC in data:
                pair_rvec = profiler.timed(
                    "edge_pair_vec_select_ms",
                    data[KEY.EDGE_VEC].index_select,
                    0,
                    data[KEY.PAIR_EDGE_FORWARD_INDEX],
                )
                data[KEY.PAIR_EDGE_VEC] = pair_rvec

            pair_r = profiler.timed(
                "edge_length_norm_ms",
                torch.linalg.norm,
                pair_rvec,
                dim=-1,
            )
            pair_embedding = profiler.timed(
                "radial_basis_cutoff_ms",
                lambda: self.basis_function(pair_r)
                * self.cutoff_function(pair_r).unsqueeze(-1),
            )
            pair_attr = profiler.timed(
                "spherical_harmonics_ms",
                self.spherical,
                pair_rvec,
            )

            data[KEY.PAIR_EDGE_EMBEDDING] = pair_embedding
            data[KEY.PAIR_EDGE_ATTR] = pair_attr
            data[KEY.PAIR_EDGE_REVERSE_ATTR] = pair_attr * self.reverse_sh_sign.to(
                pair_attr.dtype
            ).unsqueeze(0)
            if self.pair_execution_policy != "geometry_only":
                data[KEY.EDGE_LENGTH] = profiler.timed(
                    "edge_length_expand_ms",
                    pair_r.index_select,
                    0,
                    data[KEY.EDGE_PAIR_MAP],
                )

            edge_attr = profiler.timed(
                "edge_attr_expand_ms",
                pair_attr.index_select,
                0,
                data[KEY.EDGE_PAIR_MAP],
            )
            reverse_mask = data[KEY.EDGE_PAIR_REVERSE].to(edge_attr.dtype).unsqueeze(-1)
            sign = profiler.timed(
                "edge_attr_sign_ms",
                lambda: 1.0
                + reverse_mask
                * (self.reverse_sh_sign.to(edge_attr.dtype).unsqueeze(0) - 1.0),
            )
            if self.pair_execution_policy != "geometry_only":
                edge_embedding = profiler.timed(
                    "edge_embedding_expand_ms",
                    pair_embedding.index_select,
                    0,
                    data[KEY.EDGE_PAIR_MAP],
                )
                data[KEY.EDGE_EMBEDDING] = edge_embedding
            data[KEY.EDGE_ATTR] = edge_attr * sign
            return data

        rvec = data[KEY.EDGE_VEC]
        r = data[KEY.EDGE_LENGTH] if KEY.EDGE_LENGTH in data else profiler.timed(
            "edge_length_norm_ms", torch.linalg.norm, data[KEY.EDGE_VEC], dim=-1
        )
        data[KEY.EDGE_LENGTH] = r
        data[KEY.EDGE_EMBEDDING] = profiler.timed(
            "radial_basis_cutoff_ms",
            lambda: self.basis_function(r) * self.cutoff_function(r).unsqueeze(-1),
        )
        data[KEY.EDGE_ATTR] = profiler.timed("spherical_harmonics_ms", self.spherical, rvec)
        return data

    module.forward = MethodType(wrapped_forward, module)


def _patch_convolution(module: IrrepsConvolution, profiler: StageProfiler) -> None:
    def wrapped_forward(self, data):
        assert self.convolution is not None
        assert self.weight_nn is not None

        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        if self._use_pair_execution(data):
            pair_weight = profiler.timed(
                "conv_weight_nn_ms", self.weight_nn, data[KEY.PAIR_EDGE_EMBEDDING]
            )
            weight = profiler.timed(
                "conv_weight_expand_ms",
                pair_weight.index_select,
                0,
                data[KEY.EDGE_PAIR_MAP],
            )
            edge_src = data[self.key_edge_idx][1]
            edge_dst = data[self.key_edge_idx][0]
            src_feat = profiler.timed("conv_src_gather_ms", x.index_select, 0, edge_src)
            message = profiler.timed(
                "conv_message_tp_ms",
                self.convolution,
                src_feat,
                data[self.key_filter],
                weight,
            )
            x = profiler.timed("conv_message_gather_ms", message_gather, x, edge_dst, message)
        else:
            weight = profiler.timed(
                "conv_weight_nn_ms", self.weight_nn, data[self.key_weight_input]
            )
            edge_src = data[self.key_edge_idx][1]
            edge_dst = data[self.key_edge_idx][0]
            src_feat = profiler.timed("conv_src_gather_ms", x.index_select, 0, edge_src)
            message = profiler.timed(
                "conv_message_tp_ms",
                self.convolution,
                src_feat,
                data[self.key_filter],
                weight,
            )
            x = profiler.timed("conv_message_gather_ms", message_gather, x, edge_dst, message)

        x = x.div(self.denominator)
        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data

    module.forward = MethodType(wrapped_forward, module)


def _run_mode(
    *,
    case: dict[str, Any],
    system: dict[str, Any],
    mode: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    calc = SevenNetCalculator(
        model=str(CHECKPOINT),
        device=device,
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case["pair_execution_policy"],
    )
    pair_cfg = _resolve_pair_cfg(case)
    atoms = deepcopy(system["atoms"])
    data = _build_data(atoms, cutoff=calc.cutoff, pair_cfg=pair_cfg, device=device)

    model = calc.model
    if mode == "forward_energy":
        model = _make_energy_only_model(model)
    model.eval()

    for _ in range(WARMUP):
        _ = model(data.clone())
    sync_if_needed(device)

    profiler = StageProfiler(device)
    _instrument_model(model, profiler)

    raw_rows: list[dict[str, Any]] = []
    totals: list[float] = []
    for repeat in range(REPEATS):
        profiler.reset()
        batch = data.clone()
        sync_if_needed(device)
        t0 = time.perf_counter()
        _ = model(batch)
        sync_if_needed(device)
        total_ms = (time.perf_counter() - t0) * 1000.0
        totals.append(total_ms)
        stage_totals = profiler.totals()
        for stage in STAGE_ORDER:
            raw_rows.append(
                {
                    "system": system["system"],
                    "category": system["category"],
                    "case": case["case"],
                    "mode": mode,
                    "repeat": repeat,
                    "stage": stage,
                    "ms": stage_totals.get(stage, 0.0),
                    "num_edges": int(batch[KEY.EDGE_IDX].shape[1]),
                    "num_pairs": int(batch[KEY.PAIR_EDGE_VEC].shape[0])
                    if KEY.PAIR_EDGE_VEC in batch
                    else 0,
                }
            )
        raw_rows.append(
            {
                "system": system["system"],
                "category": system["category"],
                "case": case["case"],
                "mode": mode,
                "repeat": repeat,
                "stage": "model_total_ms",
                "ms": total_ms,
                "num_edges": int(batch[KEY.EDGE_IDX].shape[1]),
                "num_pairs": int(batch[KEY.PAIR_EDGE_VEC].shape[0])
                if KEY.PAIR_EDGE_VEC in batch
                else 0,
            }
        )

    total_summary = {
        "mean_total_ms": statistics.mean(totals),
        "std_total_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
    }
    return raw_rows, total_summary


def _summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (row["system"], row["category"], row["case"], row["stage"])
        grouped[key].append(row["ms"])
        meta[key] = {
            "mode": row["mode"],
            "num_edges": row["num_edges"],
            "num_pairs": row["num_pairs"],
        }
    out = []
    for (system, category, case, stage), values in grouped.items():
        info = meta[(system, category, case, stage)]
        out.append(
            {
                "system": system,
                "category": category,
                "case": case,
                "mode": info["mode"],
                "stage": stage,
                "mean_ms": statistics.mean(values),
                "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                "num_edges": info["num_edges"],
                "num_pairs": info["num_pairs"],
            }
        )
    return sorted(out, key=lambda row: (row["system"], row["case"], row["stage"]))


def _plot(summary_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(summary_rows)
    df = df[df["stage"] != "model_total_ms"]
    if df.empty:
        return
    ensure_dir(FIGURES_DIR)
    systems = list(dict.fromkeys(df["system"]))
    fig, axes = plt.subplots(len(systems), 1, figsize=(12, 4 * len(systems)), squeeze=False)
    for ax, system in zip(axes[:, 0], systems):
        sub = df[df["system"] == system]
        piv = (
            sub.pivot(index="case", columns="stage", values="mean_ms")
            .reindex(index=["baseline", "geometry_only"])
            .fillna(0.0)
        )
        bottom = None
        for stage in STAGE_ORDER:
            if stage not in piv.columns:
                continue
            vals = piv[stage].values
            ax.bar(piv.index, vals, bottom=bottom, label=stage)
            bottom = vals if bottom is None else bottom + vals
        ax.set_title(system)
        ax.set_ylabel("ms")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geometry_only_forward_breakdown.png", dpi=300)
    fig.savefig(FIGURES_DIR / "geometry_only_forward_breakdown.svg")
    plt.close(fig)


def _write_report(summary_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(summary_rows)
    lines = [
        "# Geometry-Only Breakdown Report",
        "",
        "- checkpoint: `tests/data/checkpoints/cp_0.pth`",
        f"- warmup: `{WARMUP}`",
        f"- repeats: `{REPEATS}`",
        "- cases: `baseline`, `geometry_only`",
        "- focus: `geometry_only` 내부의 pair->edge 재확장 비용 분해",
        "- note: 이 계측은 `torch.cuda.synchronize()`를 각 stage 경계에 넣는 intrusive timing이다.",
        "- note: 따라서 `model_total_ms` 절대값은 headline latency가 아니라 stage 비교용으로만 읽어야 한다.",
        "",
    ]
    for system in sorted(df["system"].unique()):
        sub = df[(df["system"] == system) & (df["mode"] == "forward_energy")]
        if sub.empty:
            continue
        base_total = sub[(sub["case"] == "baseline") & (sub["stage"] == "model_total_ms")]["mean_ms"].iloc[0]
        geo_total = sub[(sub["case"] == "geometry_only") & (sub["stage"] == "model_total_ms")]["mean_ms"].iloc[0]
        geo = sub[sub["case"] == "geometry_only"].set_index("stage")["mean_ms"].to_dict()
        lines.extend(
            [
                f"## {system}",
                "",
                f"- baseline forward total: `{base_total:.3f} ms`",
                f"- geometry_only forward total: `{geo_total:.3f} ms`",
                f"- pair->edge expansion: `{geo.get('edge_embedding_expand_ms', 0.0) + geo.get('edge_attr_expand_ms', 0.0) + geo.get('edge_attr_sign_ms', 0.0):.3f} ms`",
                f"- pair weight expand: `{geo.get('conv_weight_expand_ms', 0.0):.3f} ms`",
                f"- pair geometry (norm+basis+sh): `{geo.get('edge_length_norm_ms', 0.0) + geo.get('radial_basis_cutoff_ms', 0.0) + geo.get('spherical_harmonics_ms', 0.0):.3f} ms`",
                "",
            ]
        )
    (REPORTS_DIR / "geometry_only_breakdown_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    ensure_dir(METRICS_DIR)
    ensure_dir(REPORTS_DIR)
    device = resolve_device()
    raw_rows: list[dict[str, Any]] = []

    for system in SYSTEMS:
        for case in CASES:
            rows, _totals = _run_mode(
                case=case,
                system=system,
                mode="forward_energy",
                device=device,
            )
            raw_rows.extend(rows)

    summary_rows = _summarize(raw_rows)
    _write_csv(METRICS_DIR / "geometry_only_breakdown_raw.csv", raw_rows)
    _write_csv(METRICS_DIR / "geometry_only_breakdown_summary.csv", summary_rows)
    _plot(summary_rows)
    _write_report(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
