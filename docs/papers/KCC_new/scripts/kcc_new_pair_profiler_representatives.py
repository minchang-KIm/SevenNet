from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.train.dataload import unlabeled_atoms_to_graph

KCC_SCRIPTS = Path(__file__).resolve().parents[2] / "KCC" / "scripts"
if str(KCC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(KCC_SCRIPTS))

from kcc_common import ensure_dir, load_topk_samples, supported_dataset_specs


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIVES = ("qm9_hf", "mptrj")
CASES = (
    {
        "case": "sevennet_baseline",
        "enable_pair_execution": False,
        "pair_execution_policy": None,
        "expand_full_edges": True,
    },
    {
        "case": "sevennet_geometry_only",
        "enable_pair_execution": True,
        "pair_execution_policy": "geometry_only",
        "expand_full_edges": True,
    },
)


def _make_energy_only_model(model):
    if hasattr(model, "delete_module_by_key"):
        model.delete_module_by_key("force_output")
    if hasattr(model, "key_grad"):
        model.key_grad = None
    model.eval()
    return model


def _resolve_pair_cfg(case: dict[str, Any]) -> dict[str, Any]:
    return pair_runtime.resolve_pair_execution_config(
        {KEY.PAIR_EXECUTION_CONFIG: {"use": case["enable_pair_execution"], "policy": case["pair_execution_policy"]}},
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case["pair_execution_policy"],
    )


def _build_data(atoms: Any, *, modal: str, cutoff: float, case: dict[str, Any], device: torch.device) -> AtomGraphData:
    pair_cfg = _resolve_pair_cfg(case)
    graph = unlabeled_atoms_to_graph(atoms, cutoff, with_shift=pair_cfg["resolved_policy"] != "baseline")
    data = AtomGraphData.from_numpy_dict(graph)
    data[KEY.DATA_MODALITY] = modal
    data, _ = pair_runtime.prepare_pair_metadata(data, pair_cfg, num_atoms=len(atoms))
    data.to(device)  # type: ignore[arg-type]
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_root = ensure_dir(output_root / "metrics" / "pair_profiler")
    report_dir = ensure_dir(output_root / "reports")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for spec, _inventory_row in supported_dataset_specs(dataset_names=REPRESENTATIVES):
        sample_info = load_topk_samples(spec, top_k=1)[0]
        sample = SimpleNamespace(
            dataset=spec.name,
            modal=spec.modal,
            atoms=sample_info["atoms"],
            natoms=int(sample_info["natoms"]),
        )
        for case in CASES:
            calc = SevenNetCalculator(
                model="7net-omni",
                modal=sample.modal,
                device=device,
                enable_flash=False,
                enable_pair_execution=case["enable_pair_execution"],
                pair_execution_policy=case["pair_execution_policy"],
            )
            for mode in ("force_model", "forward_energy"):
                model = _make_energy_only_model(calc.model) if mode == "forward_energy" else calc.model
                model.eval()
                data = _build_data(sample.atoms, modal=sample.modal, cutoff=calc.cutoff, case=case, device=device)
                for _ in range(args.warmup):
                    _ = model(data.clone())
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    _ = model(data.clone())
                top = []
                for evt in prof.key_averages():
                    top.append(
                        {
                            "dataset": sample.dataset,
                            "natoms": sample.natoms,
                            "case": case["case"],
                            "mode": mode,
                            "op": evt.key,
                            "cpu_time_total_us": float(evt.cpu_time_total),
                            "device_time_total_us": float(
                                getattr(evt, "device_time_total", getattr(evt, "cuda_time_total", 0.0))
                            ),
                            "self_cpu_time_total_us": float(evt.self_cpu_time_total),
                            "count": int(evt.count),
                        }
                    )
                top_df = pd.DataFrame(top).sort_values("device_time_total_us", ascending=False)
                case_dir = ensure_dir(metrics_root / sample.dataset)
                top_df.to_csv(case_dir / f"{sample.dataset}_{case['case']}_{mode}_profiler.csv", index=False)
                rows.extend(top[:50])

    summary = pd.DataFrame(rows)
    summary.to_csv(metrics_root / "pair_profiler_summary.csv", index=False)
    report_lines = [
        "# KCC_new Pair Profiler Representative Report",
        "",
        "- representative datasets: qm9_hf (small), mptrj (large)",
        "- cases: sevennet_baseline, sevennet_geometry_only",
        "- modes: force_model, forward_energy",
        "- purpose: current geometry_only path에서 연산 분해와 호출 수를 확인하는 representative profiler",
        "",
    ]
    (report_dir / "pair_profiler_representative_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
