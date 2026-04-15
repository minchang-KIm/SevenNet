from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch

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

import sevenn._keys as KEY  # noqa: E402
import sevenn.nn.edge_embedding as edge_embedding_mod  # noqa: E402


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "flash_weight_only_once"

CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "flashtp_baseline",
        "label_ko": "FlashTP",
        "enable_flash": True,
        "enable_pair_execution": False,
        "pair_execution_policy": None,
        "runtime_patch": "none",
    },
    {
        "case": "flashtp_pair_full_current",
        "label_ko": "FlashTP + 현재 pair full",
        "enable_flash": True,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
        "runtime_patch": "none",
    },
    {
        "case": "flashtp_pair_weight_only",
        "label_ko": "FlashTP + weight only",
        "enable_flash": True,
        "enable_pair_execution": True,
        "pair_execution_policy": "full",
        "runtime_patch": "weight_only",
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


def _weight_only_forward(self, data):
    """Runtime-only ablation: keep normal edge filters, share only weight_nn input."""
    if (
        self.pair_execution_policy != "baseline"
        and KEY.PAIR_EDGE_VEC in data
        and KEY.EDGE_PAIR_MAP in data
        and KEY.PAIR_EDGE_FORWARD_INDEX in data
    ):
        rvec = data[KEY.EDGE_VEC]
        # FlashTP only consumes edge filters; keep edge SH exact but do not build
        # full-edge radial embeddings because weight_nn uses pair embeddings below.
        data[KEY.EDGE_ATTR] = self.spherical(rvec)

        pair_rvec = data[KEY.PAIR_EDGE_VEC]
        if KEY.EDGE_VEC in data and data[KEY.EDGE_VEC].requires_grad:
            pair_rvec = data[KEY.EDGE_VEC].index_select(
                0, data[KEY.PAIR_EDGE_FORWARD_INDEX]
            )
            data[KEY.PAIR_EDGE_VEC] = pair_rvec
        pair_r = torch.linalg.norm(pair_rvec, dim=-1)
        data[KEY.EDGE_LENGTH] = pair_r.index_select(0, data[KEY.EDGE_PAIR_MAP])
        data[KEY.PAIR_EDGE_EMBEDDING] = self.basis_function(
            pair_r
        ) * self.cutoff_function(pair_r).unsqueeze(-1)
        return data

    return _ORIGINAL_EDGE_EMBEDDING_FORWARD(self, data)


_ORIGINAL_EDGE_EMBEDDING_FORWARD = edge_embedding_mod.EdgeEmbedding.forward


@contextmanager
def _maybe_weight_only_patch(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    original = edge_embedding_mod.EdgeEmbedding.forward
    edge_embedding_mod.EdgeEmbedding.forward = _weight_only_forward
    try:
        yield
    finally:
        edge_embedding_mod.EdgeEmbedding.forward = original


def _case_report_line(row: dict[str, Any]) -> str:
    flash_ms = row.get("flashtp_baseline_ms", np.nan)
    if row["status"] != "ok":
        return f"| {row['dataset']} | {row['case']} | failed | - | - | {row['error_type']} |"
    speedup = float(flash_ms / row["timing_ms"]) if flash_ms > 0 else np.nan
    return (
        f"| {row['dataset']} | {row['case']} | ok | {row['timing_ms']:.3f} | "
        f"{speedup:.3f}x | {row['max_abs_force_diff_vs_baseline']:.3e} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["md22_buckyball_catcher", "oc20_s2ef_val_ood_ads"],
    )
    parser.add_argument("--warmup", type=int, default=3)
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
                "energy": float(ref["energy"]),
            }
        )

        flash_baseline_ms: float | None = None
        for case in CASES:
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
                "runtime_patch": case["runtime_patch"],
                "enable_flash": case["enable_flash"],
                "enable_pair_execution": case["enable_pair_execution"],
                "requested_pair_policy": case["pair_execution_policy"] or "none",
                "n_warmup": args.warmup,
                "n_repeat": args.repeat,
            }
            try:
                with _maybe_weight_only_patch(case["runtime_patch"] == "weight_only"):
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
                if case["case"] == "flashtp_baseline":
                    flash_baseline_ms = timing_ms
                speedup_vs_flash = (
                    float(flash_baseline_ms / timing_ms)
                    if flash_baseline_ms is not None and timing_ms > 0
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
                        "flashtp_baseline_ms": flash_baseline_ms,
                        "speedup_vs_flashtp_baseline": speedup_vs_flash,
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
            except Exception as exc:  # noqa: BLE001
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
                        "flashtp_baseline_ms": flash_baseline_ms or np.nan,
                        "speedup_vs_flashtp_baseline": np.nan,
                        "energy": np.nan,
                        "abs_energy_diff_vs_baseline": np.nan,
                        "max_abs_force_diff_vs_baseline": np.nan,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc).splitlines()[0],
                    }
                )

    summary_df = _write_csv(metrics_dir / "flash_weight_only_summary.csv", summary_rows)
    _write_csv(metrics_dir / "flash_weight_only_raw.csv", raw_rows)
    _write_csv(metrics_dir / "dataset_manifest.csv", manifest_rows)
    _write_csv(metrics_dir / "reference_baseline.csv", reference_rows)

    env = gpu_info()
    env["warmup"] = args.warmup
    env["repeat"] = args.repeat
    env["cases"] = [case["case"] for case in CASES]
    write_json(metrics_dir / "environment.json", env)

    report_lines = [
        "# FlashTP Weight-Only Pair Reuse Smoke Test",
        "",
        "## Purpose",
        "",
        "FlashTP는 edge-major 입력을 요구하므로 edge list와 edge filter는 그대로 두고, `weight_nn` 입력/출력만 reverse pair에서 공유했을 때 FlashTP 단독보다 좋아지는지 확인한다.",
        "",
        "## Run Configuration",
        "",
        f"- datasets: `{', '.join(summary_df['dataset'].drop_duplicates().tolist())}`",
        f"- warmup: `{args.warmup}`",
        f"- repeat: `{args.repeat}`",
        "- source modification: none. `EdgeEmbedding.forward` is monkey-patched only inside this script process.",
        "",
        "## Results",
        "",
        "| dataset | case | status | timing_ms | FlashTP/case | max force diff |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_df.to_dict("records"):
        report_lines.append(_case_report_line(row))
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "- `flashtp_pair_weight_only` keeps normal edge construction and normal edge spherical harmonics.",
        "- Pair metadata is used only to compute `weight_nn(pair_embedding)` once per reverse pair, then expand the resulting weight to FlashTP's edge-major input.",
        "- Full-edge radial/cutoff embedding is intentionally skipped in the patched case because FlashTP consumes the expanded weight, not `EDGE_EMBEDDING`.",
        "- If this case is slower than `flashtp_baseline`, the weight MLP saving is smaller than pair metadata plus pair-to-edge weight expansion overhead.",
        ]
    )
    (reports_dir / "flash_weight_only_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
