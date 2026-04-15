from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

KCC_SCRIPTS = Path(__file__).resolve().parents[2] / "KCC" / "scripts"
if str(KCC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(KCC_SCRIPTS))

from kcc_common import (  # noqa: E402
    ensure_dir,
    evaluate_case,
    gpu_info,
    graph_feature_manifest,
    load_topk_samples,
    reference_eval,
    supported_dataset_specs,
    write_json,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "flash_four_case_once"

FOUR_CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "sevennet_baseline",
        "label_ko": "기본 SevenNet",
        "enable_flash": False,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "sevennet_pair_full",
        "label_ko": "제안기법",
        "enable_flash": False,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
    },
    {
        "case": "flashtp_baseline",
        "label_ko": "FlashTP",
        "enable_flash": True,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
    },
    {
        "case": "flashtp_pair_full",
        "label_ko": "FlashTP + 제안기법",
        "enable_flash": True,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
    },
)


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _dataset_bucket(manifest: dict[str, Any]) -> str:
    large = manifest["num_edges"] >= 3000
    dense = manifest["avg_neighbors_directed"] >= 40.0
    if large and dense:
        return "large_dense"
    if large and not dense:
        return "large_sparse"
    if (not large) and dense:
        return "small_dense"
    return "small_sparse"


def _case_report_line(row: dict[str, Any]) -> str:
    if row["status"] != "ok":
        return (
            f"| {row['dataset']} | {row['case']} | failed | - | - | "
            f"{row['error_type']}: {row['error_message']} |"
        )
    speedup = row.get("speedup_vs_baseline", np.nan)
    speedup_text = "-" if pd.isna(speedup) else f"{speedup:.3f}x"
    return (
        f"| {row['dataset']} | {row['case']} | ok | {row['timing_ms']:.3f} | "
        f"{row['resolved_policy']} | {speedup_text} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["oc20_s2ef_val_ood_ads"],
        help="dataset names. Default uses the top restored-full win dataset.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    metrics_dir = ensure_dir(output_root / "metrics")
    reports_dir = ensure_dir(output_root / "reports")

    selected_specs = supported_dataset_specs(dataset_names=args.datasets)
    if not selected_specs:
        raise SystemExit("No benchmarkable datasets selected")

    manifest_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

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

        manifest = graph_feature_manifest(sample)
        manifest["density_bucket"] = _dataset_bucket(manifest)
        manifest_rows.append(manifest)

        ref = reference_eval(sample.atoms, modal=sample.modal)
        reference_rows.append(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "modal": sample.modal,
                "reference_case": "sevennet_baseline",
                "resolved_policy": ref["resolved_policy"],
                "energy": float(ref["energy"]),
            }
        )

        baseline_ms: float | None = None
        for case in FOUR_CASES:
            row_base = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "modal": sample.modal,
                "natoms": sample.natoms,
                "num_edges": manifest["num_edges"],
                "avg_neighbors_directed": manifest["avg_neighbors_directed"],
                "density_bucket": manifest["density_bucket"],
                "case": case["case"],
                "label_ko": case["label_ko"],
                "enable_flash": case["enable_flash"],
                "enable_pair_execution": case["enable_pair_execution"],
                "requested_pair_policy": case["pair_execution_policy"] or "none",
                "n_warmup": args.warmup,
                "n_repeat": args.repeat,
            }
            try:
                result = evaluate_case(
                    sample.atoms,
                    modal=sample.modal,
                    case=case,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                timings = [float(x) for x in result["timings_ms"]]
                for repeat_idx, timing in enumerate(timings):
                    raw_rows.append(
                        {
                            **row_base,
                            "repeat_idx": repeat_idx,
                            "timing_ms": timing,
                            "resolved_policy": result["resolved_policy"],
                            "device": result["device"],
                            "status": "ok",
                        }
                    )
                timing_ms = float(result["mean_ms"])
                if case["case"] == "sevennet_baseline":
                    baseline_ms = timing_ms
                speedup = (
                    float(baseline_ms / timing_ms)
                    if baseline_ms is not None and timing_ms > 0
                    else np.nan
                )
                summary_rows.append(
                    {
                        **row_base,
                        "status": "ok",
                        "device": result["device"],
                        "resolved_policy": result["resolved_policy"],
                        "timing_ms": timing_ms,
                        "mean_ms": float(result["mean_ms"]),
                        "std_ms": float(result["std_ms"]),
                        "median_ms": float(result["median_ms"]),
                        "p95_ms": float(result["p95_ms"]),
                        "warmup_mean_ms": float(result["warmup_mean_ms"]),
                        "warmup_std_ms": float(result["warmup_std_ms"]),
                        "speedup_vs_baseline": speedup,
                        "energy": float(result["energy"]),
                        "abs_energy_diff_vs_baseline": abs(
                            float(result["energy"]) - float(ref["energy"])
                        ),
                        "max_abs_force_diff_vs_baseline": float(
                            np.max(np.abs(np.asarray(result["forces"]) - ref["forces"]))
                        ),
                        "error_type": "",
                        "error_message": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - benchmark should preserve failures
                summary_rows.append(
                    {
                        **row_base,
                        "status": "failed",
                        "device": "",
                        "resolved_policy": "",
                        "timing_ms": np.nan,
                        "mean_ms": np.nan,
                        "std_ms": np.nan,
                        "median_ms": np.nan,
                        "p95_ms": np.nan,
                        "warmup_mean_ms": np.nan,
                        "warmup_std_ms": np.nan,
                        "speedup_vs_baseline": np.nan,
                        "energy": np.nan,
                        "abs_energy_diff_vs_baseline": np.nan,
                        "max_abs_force_diff_vs_baseline": np.nan,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc).splitlines()[0],
                    }
                )
                (reports_dir / f"{sample.dataset}_{case['case']}_traceback.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )

    manifest_df = _write_csv(metrics_dir / "dataset_manifest.csv", manifest_rows)
    summary_df = _write_csv(metrics_dir / "four_case_once_summary.csv", summary_rows)
    _write_csv(metrics_dir / "four_case_once_raw.csv", raw_rows)
    _write_csv(metrics_dir / "reference_baseline.csv", reference_rows)

    env = gpu_info()
    env["warmup"] = args.warmup
    env["repeat"] = args.repeat
    env["cases"] = [case["case"] for case in FOUR_CASES]
    write_json(metrics_dir / "environment.json", env)

    report_lines = [
        "# FlashTP Four-Case One-Shot Result",
        "",
        "## Purpose",
        "",
        "성능이 잘 나온 대표 데이터셋에서 기본 SevenNet, 제안기법, FlashTP, FlashTP+제안기법을 한 번씩 실행해 결합 가능성과 대략적인 latency를 확인한다.",
        "",
        "## Run Configuration",
        "",
        f"- datasets: `{', '.join(manifest_df['dataset'].tolist())}`",
        f"- warmup: `{args.warmup}`",
        f"- repeat: `{args.repeat}`",
        "- note: repeat=1 결과이므로 논문 수치가 아니라 smoke benchmark다.",
        "",
        "## Results",
        "",
        "| dataset | case | status | timing_ms | resolved_policy | baseline/case or error |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in summary_df.to_dict("records"):
        report_lines.append(_case_report_line(row))
    report_lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- `sevennet_pair_full`은 최근 win을 복구한 two-pass 쌍 단위 실행 경로다.",
            "- `flashtp_pair_full`은 FlashTP fused tensor-product backend 위에서 pair geometry/weight 재사용 입력을 붙인 결합 smoke test다.",
            "- 현재 FlashTP convolution class는 일반 e3nn convolution의 two-pass gather 경로와 동일하지 않으므로, 이 결과를 최종 결합 성능 claim으로 쓰면 안 된다.",
        ]
    )
    (reports_dir / "four_case_once_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
