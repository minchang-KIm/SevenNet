from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

KCC_SCRIPTS = Path(__file__).resolve().parents[2] / "KCC" / "scripts"
if str(KCC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(KCC_SCRIPTS))

from kcc_common import (
    calculator_for_case,
    ensure_dir,
    gpu_info,
    load_topk_samples,
    supported_dataset_specs,
    sync_if_needed,
    write_json,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

PAIR_CASES = (
    {
        "case": "sevennet_baseline",
        "enable_flash": False,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "sevennet_geometry_only",
        "enable_flash": False,
        "enable_pair_execution": True,
        "pair_execution_policy": "geometry_only",
    },
)


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _reference_outputs(atoms: Any, *, modal: str, warmup: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = calculator_for_case(
        modal,
        {
            "case": "sevennet_baseline",
            "enable_flash": False,
            "enable_pair_execution": False,
            "pair_execution_policy": None,
        },
        device=device,
    )
    try:
        for _ in range(warmup):
            sync_if_needed(device)
            calc.calculate(atoms)
            sync_if_needed(device)
        calc.calculate(atoms)
        sync_if_needed(device)
        return {
            "energy": float(calc.results["energy"]),
            "forces": np.asarray(calc.results["forces"]),
            "device": str(device),
            "resolved_policy": calc.pair_execution_config["resolved_policy"],
        }
    finally:
        del calc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _evaluate_accuracy_repeats(
    atoms: Any,
    *,
    modal: str,
    case: dict[str, Any],
    ref_energy: float,
    ref_forces: np.ndarray,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = calculator_for_case(modal, case, device=device)
    raw_rows: list[dict[str, Any]] = []
    try:
        for _ in range(warmup):
            sync_if_needed(device)
            calc.calculate(atoms)
            sync_if_needed(device)
        for repeat_idx in range(repeat):
            sync_if_needed(device)
            calc.calculate(atoms)
            sync_if_needed(device)
            energy = float(calc.results["energy"])
            forces = np.asarray(calc.results["forces"])
            raw_rows.append(
                {
                    "repeat_idx": repeat_idx,
                    "energy": energy,
                    "abs_energy_diff_vs_reference": abs(energy - ref_energy),
                    "max_abs_force_diff_vs_reference": float(np.max(np.abs(forces - ref_forces))),
                    "mean_abs_force_diff_vs_reference": float(np.mean(np.abs(forces - ref_forces))),
                }
            )
        return {
            "resolved_policy": calc.pair_execution_config["resolved_policy"],
            "device": str(device),
            "raw_rows": raw_rows,
        }
    finally:
        del calc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", help="optional subset of dataset names")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_root = ensure_dir(output_root / "metrics" / "pair_accuracy")
    global_dir = ensure_dir(metrics_root / "global")
    per_dataset_dir = ensure_dir(metrics_root / "per_dataset")
    run_log = output_root / "run_pair_accuracy.log"
    run_log.write_text("")

    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for spec, inventory_row in selected_specs:
        sample_info = load_topk_samples(spec, top_k=1)
        if not sample_info:
            continue
        sample_info = sample_info[0]
        sample = SimpleNamespace(
            dataset=spec.name,
            modal=spec.modal,
            loader=spec.loader,
            sample_id=sample_info["sample_id"],
            natoms=int(sample_info["natoms"]),
            atoms=sample_info["atoms"],
            category=inventory_row.category,
            source_kind=spec.source_kind,
            local_path=str(spec.root),
        )
        line = f"[kcc-new accuracy] {sample.dataset} :: {sample.sample_id} :: natoms={sample.natoms}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        reference = _reference_outputs(sample.atoms, modal=sample.modal, warmup=args.warmup)
        reference_row = {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "modal": sample.modal,
            "natoms": sample.natoms,
            "reference_case": "sevennet_baseline",
            "energy": reference["energy"],
            "device": reference["device"],
            "resolved_policy": reference["resolved_policy"],
        }
        reference_rows.append(reference_row)
        manifest_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "modal": sample.modal,
                "natoms": sample.natoms,
                "category": sample.category,
                "source_kind": sample.source_kind,
                "local_path": sample.local_path,
            }
        )

        dataset_raw_rows: list[dict[str, Any]] = []
        dataset_summary_rows: list[dict[str, Any]] = []
        for case in PAIR_CASES:
            result = _evaluate_accuracy_repeats(
                sample.atoms,
                modal=sample.modal,
                case=case,
                ref_energy=reference["energy"],
                ref_forces=reference["forces"],
                warmup=args.warmup,
                repeat=args.repeat,
            )
            case_raw_rows: list[dict[str, Any]] = []
            for row in result["raw_rows"]:
                case_raw_rows.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "modal": sample.modal,
                        "natoms": sample.natoms,
                        "case": case["case"],
                        "repeat_idx": row["repeat_idx"],
                        "energy": row["energy"],
                        "abs_energy_diff_vs_reference": row["abs_energy_diff_vs_reference"],
                        "max_abs_force_diff_vs_reference": row["max_abs_force_diff_vs_reference"],
                        "mean_abs_force_diff_vs_reference": row["mean_abs_force_diff_vs_reference"],
                        "n_warmup": args.warmup,
                        "n_repeat": args.repeat,
                        "device": result["device"],
                        "resolved_policy": result["resolved_policy"],
                    }
                )
            case_raw_df = pd.DataFrame(case_raw_rows)
            dataset_raw_rows.extend(case_raw_rows)
            raw_rows.extend(case_raw_rows)
            dataset_summary_rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "modal": sample.modal,
                    "natoms": sample.natoms,
                    "case": case["case"],
                    "n_warmup": args.warmup,
                    "n_repeat": args.repeat,
                    "device": result["device"],
                    "resolved_policy": result["resolved_policy"],
                    "abs_energy_diff_mean": float(case_raw_df["abs_energy_diff_vs_reference"].mean()),
                    "abs_energy_diff_std": float(case_raw_df["abs_energy_diff_vs_reference"].std(ddof=0)),
                    "abs_energy_diff_max": float(case_raw_df["abs_energy_diff_vs_reference"].max()),
                    "max_force_diff_mean": float(case_raw_df["max_abs_force_diff_vs_reference"].mean()),
                    "max_force_diff_std": float(case_raw_df["max_abs_force_diff_vs_reference"].std(ddof=0)),
                    "max_force_diff_max": float(case_raw_df["max_abs_force_diff_vs_reference"].max()),
                    "mean_force_diff_mean": float(case_raw_df["mean_abs_force_diff_vs_reference"].mean()),
                    "mean_force_diff_std": float(case_raw_df["mean_abs_force_diff_vs_reference"].std(ddof=0)),
                    "mean_force_diff_max": float(case_raw_df["mean_abs_force_diff_vs_reference"].max()),
                }
            )
        summary_rows.extend(dataset_summary_rows)

        dataset_dir = ensure_dir(per_dataset_dir / sample.dataset)
        _write_csv(dataset_dir / "manifest.csv", [manifest_rows[-1]])
        write_json(dataset_dir / "manifest.json", manifest_rows[-1])
        _write_csv(dataset_dir / "reference_baseline.csv", [reference_row])
        _write_csv(dataset_dir / "pair_accuracy_raw_repeats.csv", dataset_raw_rows)
        _write_csv(dataset_dir / "pair_accuracy_summary.csv", dataset_summary_rows)

    _write_csv(global_dir / "manifest.csv", manifest_rows)
    _write_csv(global_dir / "reference_baseline.csv", reference_rows)
    summary_df = _write_csv(global_dir / "pair_accuracy_summary.csv", summary_rows)
    _write_csv(global_dir / "pair_accuracy_raw_repeats.csv", raw_rows)
    env = gpu_info()
    env["warmup"] = args.warmup
    env["repeat"] = args.repeat
    env["cases"] = [case["case"] for case in PAIR_CASES]
    write_json(global_dir / "environment.json", env)
    write_json(output_root / "environment.json", env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
