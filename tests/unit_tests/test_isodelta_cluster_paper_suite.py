"""Unit tests for the IsoDelta-Halo cluster paper suite runner.

The cluster runner must remain testable without a real 8-GPU node, so these
tests use synthetic but fully validated benchmark and trace evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_SUITE_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_cluster_paper_suite.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_cluster_paper_suite",
    CLUSTER_SUITE_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
isodelta_cluster_suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_cluster_suite
SPEC.loader.exec_module(isodelta_cluster_suite)


PERCENT_SCALE = 100.0
RUN_TIMEOUT_SECONDS = 3600.0
BASELINE_LOOP_TIME_SECONDS = 12.0
ISODELTA_LOOP_TIME_SECONDS = 10.0
EXPECTED_SPEEDUP = BASELINE_LOOP_TIME_SECONDS / ISODELTA_LOOP_TIME_SECONDS
EXPECTED_RESULT_COUNT = 4
ENABLED_ATTEMPTS = 10.0
ENABLED_HITS = 8.0
DISABLED_ATTEMPTS = 10.0
PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE = 7
PIPELINE_MIN_SPEEDUP = 1.05
PIPELINE_MIN_SPEEDUP_LOWER_BOUND = 1.0
PAPER_REPEAT_COUNT = isodelta_cluster_suite.FINAL_PAPER_MIN_REPEAT_COUNT
UNIT_TEST_REQUIRED_ARTIFACT_NAME = "dataset"
UNIT_TEST_ARTIFACT_SIZE_BYTES = 1
UNIT_TEST_ARTIFACT_SHA256 = "0" * isodelta_cluster_suite.SHA256_HEX_LENGTH
TRACE_ATTEMPTS = 4.0
TRACE_HITS = 3.0
TRACE_HIT_RATE_PERCENT = PERCENT_SCALE * TRACE_HITS / TRACE_ATTEMPTS
TRACE_BASELINE_SECONDS = 100.0
TRACE_METADATA_SECONDS = 20.0
TRACE_ENABLED_SECONDS = 80.0
TRACE_SPEEDUP = TRACE_BASELINE_SECONDS / TRACE_ENABLED_SECONDS
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"


def _cache_summary(
    *,
    attempts: float = ENABLED_ATTEMPTS,
    hits: float = ENABLED_HITS,
    miss_disabled: float = 0.0,
    miss_no_cache: float = ENABLED_ATTEMPTS - ENABLED_HITS,
) -> dict[str, float]:
    """Create a complete cache summary with all miss counters."""
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": PERCENT_SCALE * hits / attempts,
        "summary_rank_count": 1.0,
        "miss_disabled": miss_disabled,
        "miss_no-cache": miss_no_cache,
        "miss_neighbor-list-rebuilt": 0.0,
        "miss_shape-changed": 0.0,
        "miss_index-tensor-shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _disabled_cache_summary() -> dict[str, float]:
    """Create the disabled-cache baseline summary expected by the checker."""
    return _cache_summary(
        attempts=DISABLED_ATTEMPTS,
        hits=0.0,
        miss_disabled=DISABLED_ATTEMPTS,
        miss_no_cache=0.0,
    )


def _benchmark_report() -> dict[str, object]:
    """Create a small passing paired benchmark report."""
    return {
        isodelta_cluster_suite.benchmark_check.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.benchmark_check.BENCHMARK_REPORT_COMMENT
        ),
        "provenance": {
            "report_schema_version": "isodelta-benchmark-report-v1",
            "git_commit": "0123456789abcdef",
            "git_branch": "isodelta-halo-runtime",
            "git_dirty": False,
            "python_executable": "python",
            "python_version": "3.13.0",
            "platform": "test-platform",
            "case_environment_overrides": {
                "baseline-disabled": {
                    PRINT_INFO_ENV: ENV_FLAG_ENABLED,
                    DISABLE_CACHE_ENV: ENV_FLAG_ENABLED,
                    PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
                },
                "isodelta-enabled": {
                    PRINT_INFO_ENV: ENV_FLAG_ENABLED,
                    PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
                },
            },
        },
        "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
        "summary": {
            "runs": EXPECTED_RESULT_COUNT,
            "speedup_vs_disabled_cache": EXPECTED_SPEEDUP,
            "cases": {
                "baseline-disabled": {
                    "mean_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": 0.0,
                    "sample_stddev_loop_time_seconds": 0.0,
                    "min_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
                "isodelta-enabled": {
                    "mean_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": 0.0,
                    "sample_stddev_loop_time_seconds": 0.0,
                    "min_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
            },
            "final_thermo_delta_vs_disabled_cache": {
                "PotEng": {"max_abs_delta": 1.0e-9, "paired_count": 2.0},
            },
        },
        "results": [
            {
                "case": "baseline-disabled",
                "repeat_index": 0,
                "returncode": 0,
                "loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                "cache_summary": _disabled_cache_summary(),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 0,
                "returncode": 0,
                "loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                "cache_summary": _cache_summary(),
            },
            {
                "case": "baseline-disabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                "cache_summary": _disabled_cache_summary(),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                "cache_summary": _cache_summary(),
            },
        ],
    }


def _trace_evidence(model_name: str) -> dict[str, object]:
    """Create portable trace evidence accepted by the trace checker."""
    return {
        isodelta_cluster_suite.trace_check.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.trace_check.TRACE_EVIDENCE_REPORT_COMMENT
        ),
        "status": "passed",
        "model": model_name,
        "attempts": TRACE_ATTEMPTS,
        "hits": TRACE_HITS,
        "hit_rate_percent": TRACE_HIT_RATE_PERCENT,
        "miss_breakdown": {
            "miss_disabled": 0.0,
            "miss_no-cache": TRACE_ATTEMPTS - TRACE_HITS,
            "miss_neighbor-list-rebuilt": 0.0,
            "miss_shape-changed": 0.0,
            "miss_index-tensor-shape-changed": 0.0,
            "miss_tag-count-changed": 0.0,
            "miss_tag-order-changed": 0.0,
            "miss_comm-topology-changed": 0.0,
            "miss_comm-list-tag-order-changed": 0.0,
        },
        "timing": {
            "baseline_step_time_seconds": TRACE_BASELINE_SECONDS,
            "metadata_build_time_seconds": TRACE_METADATA_SECONDS,
            "metadata_fraction_percent": (
                PERCENT_SCALE * TRACE_METADATA_SECONDS / TRACE_BASELINE_SECONDS
            ),
            "cache_lookup_overhead_seconds": 0.0,
            "estimated_average_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_worst_case_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_average_speedup": TRACE_SPEEDUP,
            "estimated_worst_case_speedup": TRACE_SPEEDUP,
        },
        "model_agnostic_requirements": {
            "uses_ordered_graph_node_tags": True,
            "uses_edge_count_shape_guard": True,
            "uses_neighbor_rebuild_guard": True,
            "uses_comm_topology_guard": True,
            "uses_comm_list_tag_order_guard": True,
        },
    }


def _experiment_report(command_count: int = 1) -> dict[str, object]:
    """Create a SevenNet experiment driver report with auditable metadata."""
    return {
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.EXPERIMENT_REPORT_COMMENT
        ),
        "provenance": {
            "report_schema_version": (
                isodelta_cluster_suite.EXPERIMENT_REPORT_SCHEMA_VERSION
            ),
        },
        "commands": [
            {
                "name": f"experiment-command-{index}",
            }
            for index in range(command_count)
        ],
    }


def _experiment_report_check(
    experiment_report: Path,
    command_count: int = 1,
) -> dict[str, object]:
    """Create report-check evidence that should match an experiment report."""
    return {
        "experiment_report_check_schema_version": (
            isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION
        ),
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_COMMENT
        ),
        "status": isodelta_cluster_suite.PASSED_STATUS,
        "experiment_report": str(experiment_report),
        "checked_command_count": command_count,
        "checked_log_fingerprint_count": (
            command_count
            * isodelta_cluster_suite.EXPERIMENT_LOG_STREAMS_PER_COMMAND
        ),
    }


def _external_timing_report(
    model_name: str,
    *,
    disabled_command: str = "run baseline",
    enabled_command: str = "run enabled",
    log_dir: Path | None = None,
) -> dict[str, object]:
    """Create an external-pair timing report for a non-SevenNet runtime."""
    case = isodelta_cluster_suite.CaseConfig(
        name=f"{model_name.lower()}-existing",
        model=model_name,
        kind="external_pair",
        disabled_command=disabled_command,
        enabled_command=enabled_command,
        repeat_count=2,
    )
    return _external_timing_report_for_case(case, log_dir=log_dir)


def _external_timing_report_for_case(
    case: "isodelta_cluster_suite.CaseConfig",
    *,
    baseline_times: list[float] | None = None,
    enabled_times: list[float] | None = None,
    log_dir: Path | None = None,
) -> dict[str, object]:
    """Create external-pair timing evidence for a concrete manifest case."""
    if baseline_times is None:
        baseline_times = [
            BASELINE_LOOP_TIME_SECONDS - 1.0,
            BASELINE_LOOP_TIME_SECONDS + 1.0,
        ]
    if enabled_times is None:
        enabled_times = [
            ISODELTA_LOOP_TIME_SECONDS - 1.0,
            ISODELTA_LOOP_TIME_SECONDS + 1.0,
        ]
    baseline_mean = isodelta_cluster_suite._mean(baseline_times)
    enabled_mean = isodelta_cluster_suite._mean(enabled_times)
    speedup = (
        baseline_mean / enabled_mean
        if baseline_mean is not None
        and enabled_mean is not None
        and enabled_mean > isodelta_cluster_suite.MIN_POSITIVE_VALUE
        else None
    )
    baseline_variance = isodelta_cluster_suite._sample_variance(baseline_times)
    enabled_variance = isodelta_cluster_suite._sample_variance(enabled_times)
    baseline_stddev = isodelta_cluster_suite._sample_stddev(baseline_times)
    enabled_stddev = isodelta_cluster_suite._sample_stddev(enabled_times)
    command_records = []
    for repeat_index, elapsed_seconds in enumerate(baseline_times):
        if log_dir is None:
            stdout_path = Path(f"{case.name}/disabled_{repeat_index}.stdout.log")
            stderr_path = Path(f"{case.name}/disabled_{repeat_index}.stderr.log")
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"{case.name}_disabled_{repeat_index}.stdout.log"
            stderr_path = log_dir / f"{case.name}_disabled_{repeat_index}.stderr.log"
            stdout_path.write_text(f"{case.name} disabled {repeat_index}\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
        command_records.append(
            {
                "name": f"{case.name}:{isodelta_cluster_suite.EXTERNAL_DISABLED_COMMAND_LABEL}:{repeat_index}",
                "command": str(case.disabled_command),
                "returncode": isodelta_cluster_suite.SUCCESS_RETURN_CODE,
                "elapsed_seconds": elapsed_seconds,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "cwd": str(REPO_ROOT),
                "tracked_env": isodelta_cluster_suite.command_environment_snapshot(
                    isodelta_cluster_suite._default_case_env(case.disabled_env, disabled=True)
                ),
            }
        )
    for repeat_index, elapsed_seconds in enumerate(enabled_times):
        if log_dir is None:
            stdout_path = Path(f"{case.name}/enabled_{repeat_index}.stdout.log")
            stderr_path = Path(f"{case.name}/enabled_{repeat_index}.stderr.log")
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"{case.name}_enabled_{repeat_index}.stdout.log"
            stderr_path = log_dir / f"{case.name}_enabled_{repeat_index}.stderr.log"
            stdout_path.write_text(f"{case.name} enabled {repeat_index}\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
        command_records.append(
            {
                "name": f"{case.name}:{isodelta_cluster_suite.EXTERNAL_ENABLED_COMMAND_LABEL}:{repeat_index}",
                "command": str(case.enabled_command),
                "returncode": isodelta_cluster_suite.SUCCESS_RETURN_CODE,
                "elapsed_seconds": elapsed_seconds,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "cwd": str(REPO_ROOT),
                "tracked_env": isodelta_cluster_suite.command_environment_snapshot(
                    isodelta_cluster_suite._default_case_env(case.enabled_env, disabled=False)
                ),
            }
        )
    command_log_fingerprints = [
        {
            "name": command["name"],
            "returncode": command["returncode"],
            "stdout": isodelta_cluster_suite.optional_file_fingerprint(
                Path(str(command["stdout_path"]))
            ),
            "stderr": isodelta_cluster_suite.optional_file_fingerprint(
                Path(str(command["stderr_path"]))
            ),
        }
        for command in command_records
    ]
    return {
        "schema_version": "isodelta-external-pair-timing-v1",
        "case_name": case.name,
        "model": case.model,
        "ablation_mode": case.ablation_mode,
        "timing_modes": list(
            isodelta_cluster_suite._external_timing_modes_for_ablation(case)
        ),
        "repeat_count": case.repeat_count,
        "disabled_success_count": len(baseline_times),
        "enabled_success_count": len(enabled_times),
        "baseline_times_seconds": baseline_times,
        "enabled_times_seconds": enabled_times,
        "baseline_mean_seconds": baseline_mean,
        "enabled_mean_seconds": enabled_mean,
        "baseline_sample_variance_seconds": baseline_variance,
        "enabled_sample_variance_seconds": enabled_variance,
        "baseline_sample_stddev_seconds": baseline_stddev,
        "enabled_sample_stddev_seconds": enabled_stddev,
        "speedup_vs_disabled_cache": speedup,
        "mode_controls": isodelta_cluster_suite.case_mode_control_record(case),
        "commands": command_records,
        "command_log_fingerprints": command_log_fingerprints,
    }


def _repeat_timing_rows_from_test_evidence(
    case_record: dict[str, object],
    *,
    benchmark_report: dict[str, object] | None = None,
    external_timing_report: dict[str, object] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build repeat timing artifact rows from test benchmark/external evidence."""
    return tuple(
        isodelta_cluster_suite._repeat_timing_rows_from_sources(
            str(case_record["case_name"]),
            str(case_record["model"]),
            str(case_record["kind"]),
            benchmark_payload=benchmark_report,
            external_payload=external_timing_report,
        )
    )


def _minimal_svg(title: str, *, description: str) -> str:
    """Return a tiny SVG figure that still exercises XML-based validation."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" '
        'viewBox="0 0 960 540">'
        f"<desc>{description}</desc>"
        f'<text x="20" y="40">{title}</text>'
        "</svg>\n"
    )


def _summary_case_record(
    case_name: str,
    *,
    model: str = "SevenNet",
    kind: str = "trace_only",
    status: str = isodelta_cluster_suite.CASE_STATUS_PASSED,
    **metrics: object,
) -> dict[str, object]:
    """Return the minimal case JSON fields needed to verify paper tables."""
    record = {
        "case_name": case_name,
        "model": model,
        "kind": kind,
        "status": status,
    }
    record.update(metrics)
    return record


def _summary_correlations(case_count: int) -> list[dict[str, object]]:
    """Return correlation rows matching the generated minimal CSV table."""
    return [
        {
            "x_metric": x_metric,
            "y_metric": y_metric,
            "n": case_count,
            "pearson": None,
            "spearman": None,
        }
        for x_metric, y_metric in isodelta_cluster_suite.CORRELATION_METRIC_PAIRS
    ]


def _write_required_paper_artifacts(
    output_dir: Path,
    *,
    case_names: tuple[str, ...] = ("case",),
    case_records: tuple[dict[str, object], ...] | None = None,
    command_records: tuple[dict[str, object], ...] = (),
    repeat_timing_rows: tuple[dict[str, object], ...] = (),
) -> dict[str, dict[str, object]]:
    """Create the required paper artifacts that bundle verification expects."""
    if case_records is None:
        case_records = tuple(_summary_case_record(case_name) for case_name in case_names)
    tables_dir = output_dir / isodelta_cluster_suite.TABLES_DIR_NAME
    figures_dir = output_dir / isodelta_cluster_suite.FIGURES_DIR_NAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    environment_snapshot = output_dir / isodelta_cluster_suite.ENVIRONMENT_SNAPSHOT_NAME
    manifest_snapshot = output_dir / isodelta_cluster_suite.MANIFEST_SNAPSHOT_NAME
    case_summary_csv = tables_dir / "case_summary.csv"
    case_summary_markdown = tables_dir / "case_summary.md"
    correlation_csv = tables_dir / "correlation.csv"
    command_timing_csv = tables_dir / "command_timing.csv"
    command_timing_markdown = tables_dir / "command_timing.md"
    repeat_timing_csv = tables_dir / "repeat_timing.csv"
    repeat_timing_markdown = tables_dir / "repeat_timing.md"
    speedup_svg = figures_dir / "speedup_by_case.svg"
    hit_rate_svg = figures_dir / "hit_rate_vs_speedup.svg"
    trace_svg = figures_dir / "trace_metadata_fraction_vs_speedup.svg"
    environment_snapshot.write_text(
        json.dumps(
            {
                isodelta_cluster_suite.GENERATED_ARTIFACT_COMMENT_KEY: (
                    isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                        "environment_snapshot"
                    ]
                ),
                "snapshot_schema_version": (
                    isodelta_cluster_suite.ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
                )
            }
        ),
        encoding="utf-8",
    )
    manifest_snapshot.write_text(
        (
            "# "
            + isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS["manifest_snapshot"]
            + '\n[suite]\nname = "test-suite"\n'
        ),
        encoding="utf-8",
    )
    case_rows = [
        {
            "case": record["case_name"],
            "model": record["model"],
            "kind": record["kind"],
            "status": record["status"],
        }
        for record in case_records
    ]
    correlation_rows = [
        {
            "x_metric": x_metric,
            "y_metric": y_metric,
            "n": len(case_records),
            "pearson": "",
            "spearman": "",
        }
        for x_metric, y_metric in isodelta_cluster_suite.CORRELATION_METRIC_PAIRS
    ]
    command_rows = [
        {
            "name": record["name"],
            "returncode": record["returncode"],
            "elapsed_seconds": record["elapsed_seconds"],
            "stdout_path": record["stdout_path"],
            "stderr_path": record["stderr_path"],
            "cwd": record["cwd"],
        }
        for record in command_records
    ]
    isodelta_cluster_suite.write_csv(
        case_summary_csv,
        case_rows,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS["case_summary_csv"],
    )
    isodelta_cluster_suite.write_markdown_table(
        case_summary_markdown,
        case_rows,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
            "case_summary_markdown"
        ],
    )
    isodelta_cluster_suite.write_csv(
        correlation_csv,
        correlation_rows,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS["correlation_csv"],
    )
    isodelta_cluster_suite.write_csv(
        command_timing_csv,
        command_rows,
        fieldnames=isodelta_cluster_suite.PAPER_COMMAND_TIMING_COLUMNS,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
            "command_timing_csv"
        ],
    )
    isodelta_cluster_suite.write_markdown_table(
        command_timing_markdown,
        command_rows,
        fieldnames=isodelta_cluster_suite.PAPER_COMMAND_TIMING_COLUMNS,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
            "command_timing_markdown"
        ],
    )
    isodelta_cluster_suite.write_csv(
        repeat_timing_csv,
        list(repeat_timing_rows),
        fieldnames=isodelta_cluster_suite.PAPER_REPEAT_TIMING_COLUMNS,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
            "repeat_timing_csv"
        ],
    )
    isodelta_cluster_suite.write_markdown_table(
        repeat_timing_markdown,
        list(repeat_timing_rows),
        fieldnames=isodelta_cluster_suite.PAPER_REPEAT_TIMING_COLUMNS,
        comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
            "repeat_timing_markdown"
        ],
    )
    speedup_svg.write_text(
        isodelta_cluster_suite._empty_svg(
            isodelta_cluster_suite.SPEEDUP_SVG_EMPTY_MESSAGE,
            description=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                "speedup_svg"
            ],
        ),
        encoding="utf-8",
    )
    hit_rate_svg.write_text(
        isodelta_cluster_suite._empty_svg(
            f"No paired values for {isodelta_cluster_suite.HIT_RATE_SCATTER_TITLE}",
            description=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                "hit_rate_svg"
            ],
        ),
        encoding="utf-8",
    )
    trace_svg.write_text(
        isodelta_cluster_suite._empty_svg(
            f"No paired values for {isodelta_cluster_suite.TRACE_METADATA_SCATTER_TITLE}",
            description=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                "trace_svg"
            ],
        ),
        encoding="utf-8",
    )
    artifact_paths = {
        "environment_snapshot": environment_snapshot,
        "case_summary_csv": case_summary_csv,
        "case_summary_markdown": case_summary_markdown,
        "correlation_csv": correlation_csv,
        "command_timing_csv": command_timing_csv,
        "command_timing_markdown": command_timing_markdown,
        "repeat_timing_csv": repeat_timing_csv,
        "repeat_timing_markdown": repeat_timing_markdown,
        "speedup_svg": speedup_svg,
        "hit_rate_svg": hit_rate_svg,
        "trace_svg": trace_svg,
        "manifest_snapshot": manifest_snapshot,
    }
    return {
        name: isodelta_cluster_suite.generated_artifact_record(path)
        for name, path in artifact_paths.items()
    }


def _artifact_index(
    artifact_fingerprints: dict[str, dict[str, object]],
) -> dict[str, str]:
    """Return the summary artifact index that should mirror fingerprints."""
    return {
        name: str(record["path"])
        for name, record in artifact_fingerprints.items()
    }


def _pipeline_report_modes(**overrides: bool) -> dict[str, bool]:
    """Return valid pipeline mode flags for synthetic verification reports."""
    modes = {
        "dry_run": False,
        "skip_downloads": False,
        "skip_gpu_check": False,
        "allow_gpu_mismatch": False,
        "keep_going": False,
        "reuse_passed": False,
    }
    modes.update(overrides)
    return modes


def _pipeline_suite_record(root: Path, output_dir: Path) -> dict[str, object]:
    """Return suite metadata matching a final-paper pipeline report."""
    manifest_path = root / "suite.toml"
    manifest_path.write_text('[suite]\nname = "pipeline-suite"\n', encoding="utf-8")
    return {
        "name": "pipeline-suite",
        "manifest_path": str(manifest_path),
        "manifest": isodelta_cluster_suite.generated_artifact_record(manifest_path),
        "output_dir": str(output_dir),
        "expected_gpus": isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
        "required_models": list(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS),
        "runtime_overrides": {},
    }


def _pipeline_success_stage_statuses() -> list[str]:
    """Return the accepted status value for each final-paper pipeline stage."""
    return [
        isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
        isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
        isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED,
        isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
        isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
        isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
    ]


def _pipeline_preflight_report(
    *,
    expected_gpus: int = isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
    detected_gpus: int | None = isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
    skip_gpu_check: bool = False,
    allow_gpu_mismatch: bool = False,
    skipped: bool = False,
) -> dict[str, object]:
    """Return preflight evidence for pipeline-report semantic verification."""
    return {
        "preflight_report_schema_version": (
            isodelta_cluster_suite.PREFLIGHT_REPORT_SCHEMA_VERSION
        ),
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.PREFLIGHT_REPORT_COMMENT
        ),
        "status": isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED,
        "skip_gpu_check": skip_gpu_check,
        "allow_gpu_mismatch": allow_gpu_mismatch,
        "gpu_check": {
            "expected_gpus": expected_gpus,
            "detected_gpus": detected_gpus,
            "detector": "unit-test",
            "allow_mismatch": allow_gpu_mismatch,
            "skipped": skipped,
        },
    }


def _pipeline_readiness_report(
    *,
    status: str = isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
    failed_check: str | None = None,
    expected_gpus: int = isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
    required_models: tuple[str, ...] = isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS,
    runtime_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return readiness evidence for pipeline-report semantic verification."""
    return {
        "readiness_schema_version": isodelta_cluster_suite.READINESS_SCHEMA_VERSION,
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.READINESS_REPORT_COMMENT
        ),
        "status": status,
        "suite": {
            "expected_gpus": expected_gpus,
            "required_models": list(required_models),
            "runtime_overrides": {} if runtime_overrides is None else runtime_overrides,
        },
        "checks": [
            {
                "name": check_name,
                "passed": check_name != failed_check,
                "detail": "unit-test readiness fixture",
            }
            for check_name in isodelta_cluster_suite.FINAL_PAPER_READINESS_CHECK_NAMES
        ],
    }


def _pipeline_artifact_preparation_report(
    suite_record: dict[str, object],
    *,
    missing_required: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Return artifact-preparation evidence for pipeline semantic checks."""
    return {
        "artifact_preparation_schema_version": (
            isodelta_cluster_suite.ARTIFACT_PREPARATION_SCHEMA_VERSION
        ),
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_COMMENT
        ),
        "status": isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
        "dry_run": dry_run,
        "suite": {
            "manifest": suite_record["manifest"],
            "require_artifact_sha256": True,
            "runtime_overrides": suite_record["runtime_overrides"],
        },
        "missing_required_artifacts": (
            [UNIT_TEST_REQUIRED_ARTIFACT_NAME] if missing_required else []
        ),
        "artifacts": [
            {
                "name": UNIT_TEST_REQUIRED_ARTIFACT_NAME,
                "required": True,
                "exists_after_prepare": not missing_required,
                "size_bytes": UNIT_TEST_ARTIFACT_SIZE_BYTES,
                "sha256": UNIT_TEST_ARTIFACT_SHA256,
                "actual_sha256": UNIT_TEST_ARTIFACT_SHA256,
            }
        ],
    }


def _pipeline_plan_report(
    suite_record: dict[str, object],
    *,
    skip_gpu_check: bool = False,
) -> dict[str, object]:
    """Return run-plan evidence for pipeline semantic verification."""
    paper_outputs = {
        output_key: f"paper_outputs/{output_key}.out"
        for output_key in isodelta_cluster_suite.PIPELINE_PLAN_REQUIRED_PAPER_OUTPUT_KEYS
    }
    return {
        "plan_schema_version": isodelta_cluster_suite.SUITE_SCHEMA_VERSION,
        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_cluster_suite.RUN_PLAN_REPORT_COMMENT
        ),
        "suite": {
            "manifest": suite_record["manifest"],
            "expected_gpus": suite_record["expected_gpus"],
            "required_models": suite_record["required_models"],
            "require_artifact_sha256": True,
            "runtime_overrides": suite_record["runtime_overrides"],
        },
        "modes": {
            "collect_only": False,
            "skip_downloads": False,
            "skip_gpu_check": skip_gpu_check,
            "reuse_passed": False,
        },
        "gpu_check_planned": not skip_gpu_check,
        "artifacts": [
            {
                "name": UNIT_TEST_REQUIRED_ARTIFACT_NAME,
                "required": True,
                "has_sha256": True,
                "sha256_required": True,
                "missing_required": False,
                "missing_without_url": False,
                "skip_downloads_would_fail": False,
                "sha256": UNIT_TEST_ARTIFACT_SHA256,
            }
        ],
        "cases": [
            {
                "name": f"{model_name.lower()}-plan",
                "model": model_name,
                "kind": "external_pair",
                "thresholds": {
                    "ablation_mode": isodelta_cluster_suite.ABLATION_MODE_PAIRED,
                    "repeat_count": PAPER_REPEAT_COUNT,
                },
                "expected_outputs": {
                    "benchmark_report": f"{model_name.lower()}_benchmark.json",
                },
            }
            for model_name in isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS
        ],
        "paper_outputs": paper_outputs,
    }


class IsoDeltaClusterPaperSuiteTest(unittest.TestCase):
    """Check manifest validation and paper artifact generation."""

    def test_verify_output_bundle_accepts_relocated_bundle_paths(self) -> None:
        """Bundle verification should survive archiving to a different directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_output_dir = root / "paper_outputs"
            logs_dir = original_output_dir / "logs"
            logs_dir.mkdir(parents=True)
            stdout_path = logs_dir / "case.stdout"
            missing_stderr_path = logs_dir / "case.stderr"
            stdout_path.write_text("completed\n", encoding="utf-8")
            command_record = {
                "name": "case",
                "command": "run-case",
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(missing_stderr_path),
                "cwd": str(original_output_dir),
                "tracked_env": isodelta_cluster_suite.command_environment_snapshot({}),
            }
            artifact_fingerprints = _write_required_paper_artifacts(
                original_output_dir,
                command_records=(command_record,),
            )
            summary_path = original_output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(original_output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [command_record],
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "command_log_fingerprints": [
                            {
                                "name": "case",
                                "returncode": 0,
                                "stdout": isodelta_cluster_suite.optional_file_fingerprint(
                                    stdout_path
                                ),
                                "stderr": isodelta_cluster_suite.optional_file_fingerprint(
                                    missing_stderr_path
                                ),
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            relocated_output_dir = root / "archived_outputs"
            shutil.copytree(original_output_dir, relocated_output_dir)
            shutil.rmtree(original_output_dir)
            verification = isodelta_cluster_suite.verify_output_bundle(relocated_output_dir)

        self.assertEqual(verification["status"], "passed")
        self.assertEqual(
            verification["verified_artifact_count"],
            len(isodelta_cluster_suite.REQUIRED_PAPER_ARTIFACT_NAMES),
        )
        self.assertEqual(
            verification["verified_artifact_index_count"],
            len(isodelta_cluster_suite.REQUIRED_PAPER_ARTIFACT_NAMES),
        )
        self.assertEqual(
            verification["verified_paper_artifact_semantic_count"],
            len(isodelta_cluster_suite.REQUIRED_PAPER_ARTIFACT_NAMES),
        )
        self.assertEqual(verification["verified_command_record_count"], 1)
        self.assertEqual(verification["verified_command_log_count"], 2)

    def test_verify_output_bundle_rejects_mismatched_command_fingerprints(self) -> None:
        """Bundle verification should bind command records to their log hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True)
            stdout_path = logs_dir / "case.stdout"
            stderr_path = logs_dir / "case.stderr"
            stdout_path.write_text("completed\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            command_record = {
                "name": "case",
                "command": "run-case",
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "cwd": str(output_dir),
                "tracked_env": isodelta_cluster_suite.command_environment_snapshot({}),
            }
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                command_records=(command_record,),
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [command_record],
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "command_log_fingerprints": [
                            {
                                "name": "different-case",
                                "returncode": 0,
                                "stdout": isodelta_cluster_suite.optional_file_fingerprint(
                                    stdout_path
                                ),
                                "stderr": isodelta_cluster_suite.optional_file_fingerprint(
                                    stderr_path
                                ),
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "must align by name",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_mutated_source_evidence(self) -> None:
        """Bundle verification should bind paper rows to source evidence files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            trace_path = root / "trace_evidence.json"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [
                                    isodelta_cluster_suite.optional_file_fingerprint(trace_path)
                                ],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "SHA-256 mismatch",
            ):
                isodelta_cluster_suite.verify_output_bundle(summary_path)

        self.assertEqual(verification["verified_evidence_file_count"], 1)

    def test_verify_output_bundle_rejects_mutated_external_command_log(self) -> None:
        """Bundle verification should recurse into external timing log hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir(parents=True)
            timing_report_path = output_dir / "cases" / "nequip-existing" / "external_pair_timing_report.json"
            timing_report_path.parent.mkdir(parents=True)
            timing_report = _external_timing_report("NequIP", log_dir=timing_report_path.parent / "logs")
            timing_report_path.write_text(json.dumps(timing_report), encoding="utf-8")
            nequip_case_record = _summary_case_record(
                "nequip-existing",
                model="NequIP",
                kind="external_pair",
            )
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                case_records=(nequip_case_record,),
                repeat_timing_rows=_repeat_timing_rows_from_test_evidence(
                    nequip_case_record,
                    external_timing_report=timing_report,
                ),
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [nequip_case_record],
                        "correlations": _summary_correlations(1),
                        "case_mode_controls": {
                            "nequip-existing": timing_report["mode_controls"]
                        },
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "nequip-existing": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": (
                                    isodelta_cluster_suite.generated_artifact_record(
                                        timing_report_path
                                    )
                                ),
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            mutated_log = Path(str(timing_report["commands"][0]["stdout_path"]))
            mutated_log.write_text("mutated external command log\n", encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "SHA-256 mismatch",
            ):
                isodelta_cluster_suite.verify_output_bundle(summary_path)

        self.assertEqual(verification["verified_external_command_log_count"], 4)

    def test_verify_output_bundle_rejects_semantically_invalid_svg_artifact(self) -> None:
        """Bundle verification should reject a hashed file that is not an SVG figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            speedup_svg = output_dir / "figures" / "speedup_by_case.svg"
            speedup_svg.write_text("this is not an svg document\n", encoding="utf-8")
            artifact_fingerprints["speedup_svg"] = (
                isodelta_cluster_suite.generated_artifact_record(speedup_svg)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "speedup_svg: invalid SVG XML",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_speedup_svg_label_drift(self) -> None:
        """The speedup figure should label every case with measured speedup data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            case_record = _summary_case_record(
                "case",
                speedup_vs_disabled_cache=1.25,
            )
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                case_records=(case_record,),
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [case_record],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "speedup_svg must include case label case",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_scatter_svg_point_count_drift(self) -> None:
        """Scatter figures should plot exactly one point per summary data pair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            case_record = _summary_case_record(
                "case",
                cache_hit_rate_percent=87.5,
                speedup_vs_disabled_cache=1.25,
            )
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                case_records=(case_record,),
            )
            speedup_svg = output_dir / "figures" / "speedup_by_case.svg"
            speedup_svg.write_text(
                _minimal_svg(
                    "case",
                    description=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                        "speedup_svg"
                    ],
                ),
                encoding="utf-8",
            )
            artifact_fingerprints["speedup_svg"] = (
                isodelta_cluster_suite.generated_artifact_record(speedup_svg)
            )
            hit_rate_svg = output_dir / "figures" / "hit_rate_vs_speedup.svg"
            hit_rate_svg.write_text(
                _minimal_svg(
                    isodelta_cluster_suite.HIT_RATE_SCATTER_TITLE,
                    description=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                        "hit_rate_svg"
                    ],
                ),
                encoding="utf-8",
            )
            artifact_fingerprints["hit_rate_svg"] = (
                isodelta_cluster_suite.generated_artifact_record(hit_rate_svg)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [case_record],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "hit_rate_svg circle count must match summary data pairs",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_missing_generated_file_comment(self) -> None:
        """Paper bundle tables should keep their generated-file explanation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            case_summary_csv = output_dir / "tables" / "case_summary.csv"
            uncommented_lines = case_summary_csv.read_text(
                encoding="utf-8"
            ).splitlines()[1:]
            case_summary_csv.write_text(
                "\n".join(uncommented_lines) + "\n",
                encoding="utf-8",
            )
            artifact_fingerprints["case_summary_csv"] = (
                isodelta_cluster_suite.generated_artifact_record(case_summary_csv)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "case_summary.csv: missing generated file comment",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_case_summary_value_drift(self) -> None:
        """The main paper table should not drift from summary JSON case values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            case_summary_csv = output_dir / "tables" / "case_summary.csv"
            case_summary_csv.write_text(
                (
                    isodelta_cluster_suite._csv_comment_line(
                        isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                            "case_summary_csv"
                        ]
                    )
                    + "case,model,kind,status\ncase,MACE,trace_only,passed\n"
                ),
                encoding="utf-8",
            )
            artifact_fingerprints["case_summary_csv"] = (
                isodelta_cluster_suite.generated_artifact_record(case_summary_csv)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "case_summary.csv.case.model must match summary cases",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_correlation_value_drift(self) -> None:
        """The correlation appendix table should match summary JSON rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            correlation_csv = output_dir / "tables" / "correlation.csv"
            drifted_rows = _summary_correlations(1)
            drifted_rows[0] = dict(drifted_rows[0])
            drifted_rows[0]["n"] = 2
            isodelta_cluster_suite.write_csv(
                correlation_csv,
                drifted_rows,
                comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                    "correlation_csv"
                ],
            )
            artifact_fingerprints["correlation_csv"] = (
                isodelta_cluster_suite.generated_artifact_record(correlation_csv)
            )
            first_metric_pair = (
                drifted_rows[0]["x_metric"],
                drifted_rows[0]["y_metric"],
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                (
                    "correlation.csv."
                    + re.escape(str(first_metric_pair))
                    + ".n must match summary correlations"
                ),
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_command_timing_value_drift(self) -> None:
        """The command timing table should match summary JSON command records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            stdout_path = output_dir / "logs" / "case.stdout"
            stderr_path = output_dir / "logs" / "case.stderr"
            stdout_path.parent.mkdir(parents=True)
            stdout_path.write_text("completed\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            command_record = {
                "name": "case",
                "command": "run-case",
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "cwd": str(output_dir),
                "tracked_env": isodelta_cluster_suite.command_environment_snapshot({}),
            }
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                command_records=(command_record,),
            )
            command_timing_csv = output_dir / "tables" / "command_timing.csv"
            command_timing_csv.write_text(
                (
                    isodelta_cluster_suite._csv_comment_line(
                        isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                            "command_timing_csv"
                        ]
                    )
                    + "name,returncode,elapsed_seconds,stdout_path,stderr_path,cwd\n"
                    f"case,0,2.0,{stdout_path},{stderr_path},{output_dir}\n"
                ),
                encoding="utf-8",
            )
            artifact_fingerprints["command_timing_csv"] = (
                isodelta_cluster_suite.generated_artifact_record(command_timing_csv)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [command_record],
                        "command_log_fingerprints": [
                            {
                                "name": "case",
                                "returncode": 0,
                                "stdout": isodelta_cluster_suite.optional_file_fingerprint(
                                    stdout_path
                                ),
                                "stderr": isodelta_cluster_suite.optional_file_fingerprint(
                                    stderr_path
                                ),
                            }
                        ],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                r"command_timing\.csv\[0\]\.elapsed_seconds must match summary commands",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_repeat_timing_value_drift(self) -> None:
        """The repeat timing table should match benchmark/external source evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            benchmark_path = output_dir / "cases" / "case" / "benchmark_report.json"
            benchmark_path.parent.mkdir(parents=True)
            benchmark_report = _benchmark_report()
            benchmark_path.write_text(json.dumps(benchmark_report), encoding="utf-8")
            case_record = _summary_case_record(
                "case",
                model="SevenNet",
                kind="sevennet_lammps",
            )
            repeat_rows = _repeat_timing_rows_from_test_evidence(
                case_record,
                benchmark_report=benchmark_report,
            )
            artifact_fingerprints = _write_required_paper_artifacts(
                output_dir,
                case_records=(case_record,),
                repeat_timing_rows=repeat_rows,
            )
            repeat_timing_csv = output_dir / "tables" / "repeat_timing.csv"
            drifted_rows = [dict(row) for row in repeat_rows]
            drifted_rows[0]["elapsed_seconds"] = BASELINE_LOOP_TIME_SECONDS + 1.0
            isodelta_cluster_suite.write_csv(
                repeat_timing_csv,
                drifted_rows,
                fieldnames=isodelta_cluster_suite.PAPER_REPEAT_TIMING_COLUMNS,
                comment=isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                    "repeat_timing_csv"
                ],
            )
            artifact_fingerprints["repeat_timing_csv"] = (
                isodelta_cluster_suite.generated_artifact_record(repeat_timing_csv)
            )
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [case_record],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": _artifact_index(artifact_fingerprints),
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": (
                                    isodelta_cluster_suite.generated_artifact_record(
                                        benchmark_path
                                    )
                                ),
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                r"repeat_timing\.csv\[0\]\.elapsed_seconds must match source timing evidence",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_verify_output_bundle_rejects_mismatched_artifact_index_path(self) -> None:
        """The summary artifact index should not drift from fingerprint paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_outputs"
            artifact_fingerprints = _write_required_paper_artifacts(output_dir)
            artifact_index = _artifact_index(artifact_fingerprints)
            artifact_index["case_summary_csv"] = str(output_dir / "tables" / "wrong.csv")
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(output_dir)},
                        "cases": [_summary_case_record("case")],
                        "correlations": _summary_correlations(1),
                        "commands": [],
                        "command_log_fingerprints": [],
                        "artifacts": artifact_index,
                        "artifact_fingerprints": artifact_fingerprints,
                        "evidence_fingerprints": {
                            "case": {
                                "benchmark_report": None,
                                "bundle_evidence": None,
                                "external_timing_report": None,
                                "trace_evidence": [],
                            }
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "artifacts.case_summary_csv must match artifact_fingerprints.case_summary_csv.path",
            ):
                isodelta_cluster_suite.verify_output_bundle(output_dir)

    def test_write_template_creates_commented_three_model_manifest(self) -> None:
        """The template should be editable and include the required models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "suite.toml"
            isodelta_cluster_suite.write_template(template_path)
            template = template_path.read_text(encoding="utf-8")

        self.assertTrue(template.lstrip().startswith("#"))
        self.assertIn('required_models = ["SevenNet", "MACE", "NequIP"]', template)
        self.assertIn("require_artifact_sha256 = true", template)
        self.assertIn('kind = "sevennet_lammps"', template)
        self.assertIn('kind = "external_pair"', template)
        self.assertIn('preflight_command = \'python -c "import mace"\'', template)
        self.assertIn('disabled_env = { SEVENN_ISODELTA_HALO_DISABLE = "1" }', template)
        self.assertIn("enabled_env = {}", template)
        self.assertIn('ablation_mode = "paired"', template)

    def test_sevennet_manifest_ablation_mode_reaches_experiment_driver(self) -> None:
        """A suite case should pass one-sided ablation mode to SevenNet runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            manifest_path.write_text(
                f"""
[suite]
name = "ablation-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-ablation"
model = "SevenNet"
kind = "sevennet_lammps"
lammps_command = "lmp"
input = "inputs/in.sevennet"
ablation_mode = "isodelta-enabled"
min_speedup = 1.2
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            benchmark_report, bundle_evidence, trace_paths, records = (
                isodelta_cluster_suite.run_sevennet_case(
                    config,
                    config.cases[0],
                    dry_run=True,
                )
            )

        self.assertEqual(config.cases[0].ablation_mode, "isodelta-enabled")
        self.assertIsNotNone(benchmark_report)
        self.assertIsNone(bundle_evidence)
        self.assertEqual(trace_paths, ())
        self.assertEqual(len(records), 1)
        command = records[0].command
        self.assertIsInstance(command, list)
        self.assertIn("--ablation-mode", command)
        self.assertIn("isodelta-enabled", command)
        self.assertNotIn("--min-speedup", command)

    def test_one_sided_sevennet_benchmark_report_validates_as_raw_ablation(self) -> None:
        """Suite validation should accept one-sided timing without speedup claims."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "benchmark.json"
            payload = _benchmark_report()
            payload["ablation_mode"] = "isodelta-enabled"
            payload["benchmark_cases"] = ["isodelta-enabled"]
            payload["summary"]["speedup_vs_disabled_cache"] = None
            payload["results"] = [
                result
                for result in payload["results"]
                if result["case"] == "isodelta-enabled"
            ]
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-ablation",
                model="SevenNet",
                kind="sevennet_lammps",
                ablation_mode="isodelta-enabled",
            )

            isodelta_cluster_suite.validate_case_outputs(
                case,
                report_path,
                None,
                (),
                None,
                dry_run=False,
            )

    def test_validate_case_outputs_accepts_commented_bundle_evidence(self) -> None:
        """Existing bundle evidence should identify itself before revalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "benchmark.json"
            trace_path = root / "mace_trace_evidence.json"
            bundle_path = root / "bundle_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-paper",
                model="SevenNet",
                kind="sevennet_lammps",
                required_trace_models=("MACE",),
            )
            bundle_path.write_text(
                json.dumps(
                    isodelta_cluster_suite.bundle_check.validate_bundle(
                        benchmark_report=benchmark_path,
                        trace_evidence_paths=[trace_path],
                        required_models=list(case.required_trace_models),
                        thresholds=isodelta_cluster_suite.bundle_check.BundleThresholds(
                            max_abs_thermo_delta=case.max_abs_thermo_delta,
                            min_paired_thermo_count=case.min_paired_thermo_count,
                            min_speedup=case.min_speedup,
                            min_hit_rate_percent=case.min_hit_rate_percent,
                            min_enabled_cache_attempts=(
                                case.min_enabled_cache_attempts
                            ),
                            min_enabled_cache_hits=case.min_enabled_cache_hits,
                            min_trace_hit_rate_percent=(
                                case.min_trace_hit_rate_percent
                            ),
                            min_trace_estimated_speedup=(
                                case.min_trace_estimated_speedup
                            ),
                            min_trace_metadata_fraction_percent=(
                                case.min_trace_metadata_fraction_percent
                            ),
                        ),
                    )
                ),
                encoding="utf-8",
            )

            isodelta_cluster_suite.validate_case_outputs(
                case,
                benchmark_path,
                bundle_path,
                (trace_path,),
                None,
                dry_run=False,
            )

    def test_validate_case_outputs_rejects_uncommented_bundle_evidence(self) -> None:
        """Archived bundle evidence should not pass as anonymous JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "benchmark.json"
            trace_path = root / "mace_trace_evidence.json"
            bundle_path = root / "bundle_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            bundle_path.write_text(json.dumps({}), encoding="utf-8")
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-paper",
                model="SevenNet",
                kind="sevennet_lammps",
                required_trace_models=("MACE",),
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "bundle_evidence.report_comment",
            ):
                isodelta_cluster_suite.validate_case_outputs(
                    case,
                    benchmark_path,
                    bundle_path,
                    (trace_path,),
                    None,
                    dry_run=False,
                )

    def test_validate_case_outputs_rejects_wrong_bundle_comment(self) -> None:
        """Bundle evidence comments should describe the exact report purpose."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "benchmark.json"
            trace_path = root / "mace_trace_evidence.json"
            bundle_path = root / "bundle_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            bundle_path.write_text(
                json.dumps(
                    {
                        isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY: (
                            "wrong evidence purpose"
                        )
                    }
                ),
                encoding="utf-8",
            )
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-paper",
                model="SevenNet",
                kind="sevennet_lammps",
                required_trace_models=("MACE",),
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "generated report purpose",
            ):
                isodelta_cluster_suite.validate_case_outputs(
                    case,
                    benchmark_path,
                    bundle_path,
                    (trace_path,),
                    None,
                    dry_run=False,
                )

    def test_validate_case_outputs_accepts_experiment_report_check(self) -> None:
        """SevenNet case evidence should include a passing driver report check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "isodelta_experiment_report.json"
            check_path = root / "experiment_report_check.json"
            report_path.write_text(json.dumps(_experiment_report()), encoding="utf-8")
            check_path.write_text(
                json.dumps(_experiment_report_check(report_path)),
                encoding="utf-8",
            )
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-paper",
                model="SevenNet",
                kind="sevennet_lammps",
            )

            isodelta_cluster_suite.validate_case_outputs(
                case,
                None,
                None,
                (),
                None,
                report_path,
                check_path,
                dry_run=False,
            )

    def test_validate_case_outputs_rejects_mismatched_experiment_check_count(self) -> None:
        """Report-check evidence should match the driver command count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "isodelta_experiment_report.json"
            check_path = root / "experiment_report_check.json"
            report_path.write_text(
                json.dumps(_experiment_report(command_count=2)),
                encoding="utf-8",
            )
            check_path.write_text(
                json.dumps(_experiment_report_check(report_path, command_count=1)),
                encoding="utf-8",
            )
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-paper",
                model="SevenNet",
                kind="sevennet_lammps",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "checked command count",
            ):
                isodelta_cluster_suite.validate_case_outputs(
                    case,
                    None,
                    None,
                    (),
                    None,
                    report_path,
                    check_path,
                    dry_run=False,
                )

    def test_summary_verifier_counts_experiment_report_check_evidence(self) -> None:
        """Output-bundle verification should reopen SevenNet report-check evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "isodelta_experiment_report.json"
            check_path = root / "experiment_report_check.json"
            report_path.write_text(json.dumps(_experiment_report()), encoding="utf-8")
            check_path.write_text(
                json.dumps(_experiment_report_check(report_path)),
                encoding="utf-8",
            )
            summary_payload = {
                "cases": [
                    {
                        "case_name": "sevennet-paper",
                        "kind": "sevennet_lammps",
                    }
                ],
                isodelta_cluster_suite.EVIDENCE_FINGERPRINTS_KEY: {
                    "sevennet-paper": {
                        isodelta_cluster_suite.EXPERIMENT_REPORT_KEY: (
                            isodelta_cluster_suite.generated_artifact_record(
                                report_path
                            )
                        ),
                        isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_KEY: (
                            isodelta_cluster_suite.generated_artifact_record(
                                check_path
                            )
                        ),
                    }
                },
            }

            verified_count = (
                isodelta_cluster_suite._require_experiment_report_checks_from_summary(
                    summary_payload,
                    bundle_root=root,
                    original_output_dir=None,
                )
            )

        self.assertEqual(verified_count, 1)

    def test_one_sided_sevennet_benchmark_rejects_speedup_claim(self) -> None:
        """One-sided raw timing should never be accepted as paired speedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "benchmark.json"
            payload = _benchmark_report()
            payload["ablation_mode"] = "baseline-disabled"
            payload["benchmark_cases"] = ["baseline-disabled"]
            payload["results"] = [
                result
                for result in payload["results"]
                if result["case"] == "baseline-disabled"
            ]
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-ablation",
                model="SevenNet",
                kind="sevennet_lammps",
                ablation_mode="baseline-disabled",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "must not claim speedup",
            ):
                isodelta_cluster_suite.validate_case_outputs(
                    case,
                    report_path,
                    None,
                    (),
                    None,
                    dry_run=False,
                )

    def test_external_pair_ablation_mode_runs_one_command_side(self) -> None:
        """External MACE/NequIP cases should support one-sided timing smoke runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            manifest_path.write_text(
                f"""
[suite]
name = "external-ablation-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["MACE"]

[[cases]]
name = "mace-ablation"
model = "MACE"
kind = "external_pair"
disabled_command = "run baseline"
ablation_mode = "baseline-disabled"
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            (
                _benchmark_report_path,
                _bundle_evidence,
                _trace_paths,
                timing_report_path,
                records,
            ) = isodelta_cluster_suite.run_external_pair_case(
                config,
                config.cases[0],
                dry_run=True,
            )

        command_names = [record.name for record in records]
        self.assertIsNotNone(timing_report_path)
        self.assertEqual(config.cases[0].ablation_mode, "baseline-disabled")
        self.assertIn("mace-ablation:disabled:0", command_names)
        self.assertNotIn("mace-ablation:enabled:0", command_names)

    def test_external_pair_enabled_ablation_omits_disabled_command(self) -> None:
        """External enabled-only ablation should not require a disabled command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            manifest_path.write_text(
                f"""
[suite]
name = "external-enabled-ablation-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["NequIP"]

[[cases]]
name = "nequip-ablation"
model = "NequIP"
kind = "external_pair"
enabled_command = "run enabled"
ablation_mode = "isodelta-enabled"
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            (
                _benchmark_report_path,
                _bundle_evidence,
                _trace_paths,
                timing_report_path,
                records,
            ) = isodelta_cluster_suite.run_external_pair_case(
                config,
                config.cases[0],
                dry_run=True,
            )

        command_names = [record.name for record in records]
        self.assertIsNotNone(timing_report_path)
        self.assertEqual(config.cases[0].ablation_mode, "isodelta-enabled")
        self.assertNotIn("nequip-ablation:disabled:0", command_names)
        self.assertIn("nequip-ablation:enabled:0", command_names)

    def test_external_pair_paired_mode_still_requires_both_commands(self) -> None:
        """Paired external timing must still define both disabled and enabled commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "external-paired-suite"
required_models = ["MACE"]

[[cases]]
name = "mace-paired"
model = "MACE"
kind = "external_pair"
disabled_command = "run baseline"
ablation_mode = "paired"
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "enabled_command is required",
            ):
                isodelta_cluster_suite.validate_suite_config(
                    isodelta_cluster_suite.load_manifest(manifest_path)
                )

    def test_cli_ablation_override_updates_runtime_timing_cases(self) -> None:
        """A CLI override should switch SevenNet and external-pair smoke runs."""
        config = isodelta_cluster_suite.SuiteConfig(
            name="override-suite",
            manifest_path=Path("suite.toml"),
            output_dir=Path("outputs"),
            cases=(
                isodelta_cluster_suite.CaseConfig(
                    name="sevennet",
                    model="SevenNet",
                    kind="sevennet_lammps",
                    ablation_mode="paired",
                ),
                isodelta_cluster_suite.CaseConfig(
                    name="mace",
                    model="MACE",
                    kind="external_pair",
                    ablation_mode="paired",
                ),
                isodelta_cluster_suite.CaseConfig(
                    name="trace",
                    model="NequIP",
                    kind="trace_only",
                    ablation_mode="paired",
                ),
            ),
        )
        args = isodelta_cluster_suite.parse_args(
            [
                "--manifest",
                "suite.toml",
                "--ablation-mode-override",
                "isodelta-enabled",
            ]
        )

        overridden = isodelta_cluster_suite._apply_cli_overrides(config, args)

        self.assertEqual(overridden.cases[0].ablation_mode, "isodelta-enabled")
        self.assertEqual(overridden.cases[1].ablation_mode, "isodelta-enabled")
        self.assertEqual(overridden.cases[2].ablation_mode, "paired")
        self.assertEqual(
            overridden.runtime_overrides["ablation_mode"],
            "isodelta-enabled",
        )

    def test_cli_ablation_override_is_recorded_in_run_plan(self) -> None:
        """Run plans should show when CLI options override the manifest mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            manifest_path.write_text(
                f"""
[suite]
name = "override-plan-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["MACE"]

[[cases]]
name = "mace"
model = "MACE"
kind = "external_pair"
disabled_command = "run baseline"
enabled_command = "run enabled"
ablation_mode = "paired"
""",
                encoding="utf-8",
            )
            args = isodelta_cluster_suite.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--ablation-mode-override",
                    "baseline-disabled",
                ]
            )
            config = isodelta_cluster_suite._apply_cli_overrides(
                isodelta_cluster_suite.load_manifest(manifest_path),
                args,
            )
            plan = isodelta_cluster_suite.build_run_plan(
                config,
                collect_only=False,
                skip_downloads=True,
                skip_gpu_check=True,
            )

        self.assertEqual(
            plan["suite"]["runtime_overrides"],
            {"ablation_mode": "baseline-disabled"},
        )
        self.assertEqual(
            plan["cases"][0]["thresholds"]["ablation_mode"],
            "baseline-disabled",
        )

    def test_cli_ablation_override_requires_runtime_timing_case(self) -> None:
        """A suite with only trace evidence should reject ablation overrides."""
        config = isodelta_cluster_suite.SuiteConfig(
            name="trace-suite",
            manifest_path=Path("suite.toml"),
            output_dir=Path("outputs"),
            cases=(
                isodelta_cluster_suite.CaseConfig(
                    name="trace",
                    model="SevenNet",
                    kind="trace_only",
                    ablation_mode="paired",
                ),
            ),
        )
        args = isodelta_cluster_suite.parse_args(
            [
                "--manifest",
                "suite.toml",
                "--ablation-mode-override",
                "baseline-disabled",
            ]
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "sevennet_lammps or external_pair",
        ):
            isodelta_cluster_suite._apply_cli_overrides(config, args)

    def test_external_pair_one_sided_timing_validates_without_speedup(self) -> None:
        """External one-sided timing reports should validate as raw ablation evidence."""
        case = isodelta_cluster_suite.CaseConfig(
            name="mace-existing",
            model="MACE",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            ablation_mode="isodelta-enabled",
        )
        report = _external_timing_report_for_case(
            case,
            baseline_times=[],
            enabled_times=[ISODELTA_LOOP_TIME_SECONDS - 0.5, ISODELTA_LOOP_TIME_SECONDS + 0.5],
        )

        verification = isodelta_cluster_suite.validate_external_timing_report(
            report,
            case,
        )

        self.assertEqual(verification["ablation_mode"], "isodelta-enabled")
        self.assertEqual(verification["disabled_success_count"], 0)
        self.assertEqual(verification["enabled_success_count"], 2)
        self.assertIsNone(verification["speedup_vs_disabled_cache"])

    def test_external_pair_one_sided_timing_rejects_speedup_claim(self) -> None:
        """External one-sided reports should fail if they claim paired speedup."""
        case = isodelta_cluster_suite.CaseConfig(
            name="mace-existing",
            model="MACE",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            ablation_mode="baseline-disabled",
        )
        report = _external_timing_report_for_case(
            case,
            baseline_times=[BASELINE_LOOP_TIME_SECONDS - 0.5, BASELINE_LOOP_TIME_SECONDS + 0.5],
            enabled_times=[],
        )
        report["speedup_vs_disabled_cache"] = EXPECTED_SPEEDUP

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "must be null for one-sided ablation",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_write_slurm_script_creates_commented_preflight_first_launcher(self) -> None:
        """The SLURM wrapper should submit reproducible preflight evidence first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper outputs"
            slurm_path = root / "run_isodelta.sbatch"
            manifest_path.write_text(
                f"""
[suite]
name = "slurm-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
""",
                encoding="utf-8",
            )
            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--write-slurm-script",
                    str(slurm_path),
                    "--skip-downloads",
                    "--keep-going",
                    "--reuse-passed",
                    "--slurm-job-name",
                    "paper suite",
                    "--slurm-cpus-per-task",
                    "12",
                    "--slurm-repo-root",
                    "/scratch/icpp/SevenNet-main",
                    "--slurm-manifest-path",
                    "/scratch/icpp/SevenNet-main/isodelta_cluster_suite.toml",
                    "--slurm-output-dir",
                    "/scratch/icpp/paper outputs",
                ]
            )
            script = slurm_path.read_text(encoding="utf-8")
            verification = isodelta_cluster_suite.verify_slurm_script(slurm_path)
            verify_exit_code = isodelta_cluster_suite.main(
                ["--verify-slurm-script", str(slurm_path)]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(verify_exit_code, 0)
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(
            verification["report_comment"],
            isodelta_cluster_suite.SLURM_SCRIPT_VERIFICATION_COMMENT,
        )
        self.assertTrue(script.startswith("#!/usr/bin/env bash"))
        self.assertIn("# IsoDelta-Halo cluster paper suite launcher.", script)
        self.assertIn("# CLI runtime overrides: none.", script)
        self.assertIn("#SBATCH --job-name=paper_suite", script)
        self.assertIn("#SBATCH --gres=gpu:8", script)
        self.assertIn("#SBATCH --cpus-per-task=12", script)
        self.assertIn('if [[ -z "${REPO_ROOT:-}" ]]; then', script)
        self.assertIn("  REPO_ROOT=/scratch/icpp/SevenNet-main", script)
        self.assertIn('if [[ -z "${MANIFEST_PATH:-}" ]]; then', script)
        self.assertIn(
            "  MANIFEST_PATH=/scratch/icpp/SevenNet-main/isodelta_cluster_suite.toml",
            script,
        )
        self.assertIn('if [[ -z "${ISODELTA_OUTPUT_DIR:-}" ]]; then', script)
        self.assertIn("  ISODELTA_OUTPUT_DIR='/scratch/icpp/paper outputs'", script)
        self.assertIn('cd "$REPO_ROOT"', script)
        self.assertIn(
            'SUITE_RUNNER="${SUITE_RUNNER:-$REPO_ROOT/tools/run_isodelta_cluster_paper_suite.py}"',
            script,
        )
        self.assertIn("COMMON_ARGS=(--manifest \"$MANIFEST_PATH\")", script)
        self.assertIn('COMMON_ARGS+=(--output-dir "${ISODELTA_OUTPUT_DIR}")', script)
        self.assertIn('PLAN_OUTPUT="${ISODELTA_OUTPUT_DIR}/isodelta_cluster_paper_plan.json"', script)
        self.assertIn('PREFLIGHT_OUTPUT="${ISODELTA_OUTPUT_DIR}/preflight_report.json"', script)
        self.assertIn('PIPELINE_OUTPUT="${ISODELTA_OUTPUT_DIR}/pipeline_report.json"', script)
        self.assertNotIn(str(root), script)
        self.assertIn("--preflight-only --preflight-output \"$PREFLIGHT_OUTPUT\"", script)
        self.assertIn("--plan-only --plan-output \"$PLAN_OUTPUT\"", script)
        self.assertIn("--pipeline --pipeline-report \"$PIPELINE_OUTPUT\"", script)
        self.assertIn("--verify-pipeline-report \"$PIPELINE_OUTPUT\"", script)
        self.assertIn("COMMON_ARGS+=(--skip-downloads)", script)
        self.assertIn("COMMON_ARGS+=(--keep-going)", script)
        self.assertIn("COMMON_ARGS+=(--reuse-passed)", script)
        self.assertIn("# Run the full paper pipeline", script)
        self.assertIn("# Re-open the finished pipeline report", script)

    def test_verify_slurm_script_rejects_missing_final_gate(self) -> None:
        """A launcher edited after generation must still keep final verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            slurm_path = root / "run_isodelta.sbatch"
            manifest_path.write_text(
                """
[suite]
name = "slurm-verify-suite"
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
""",
                encoding="utf-8",
            )
            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--write-slurm-script",
                    str(slurm_path),
                    "--skip-downloads",
                ]
            )
            script = slurm_path.read_text(encoding="utf-8")
            slurm_path.write_text(
                script.replace(
                    '"$PYTHON_BIN" "$SUITE_RUNNER" --verify-pipeline-report "$PIPELINE_OUTPUT"',
                    "",
                ),
                encoding="utf-8",
            )

            self.assertEqual(exit_code, 0)
            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "pipeline-report or output-bundle verification",
            ):
                isodelta_cluster_suite.verify_slurm_script(slurm_path)

    def test_write_slurm_script_rejects_collect_only_pipeline_launcher(self) -> None:
        """The generated cluster launcher should not combine collect-only with pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            slurm_path = root / "run_isodelta.sbatch"
            manifest_path.write_text(
                """
[suite]
name = "slurm-collect-only"
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--write-slurm-script",
                    str(slurm_path),
                    "--collect-only",
                ]
            )
            script_exists = slurm_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertFalse(script_exists)

    def test_write_slurm_script_uses_ablation_override_without_pipeline(self) -> None:
        """One-sided ablation launchers should skip the final-paper pipeline gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            slurm_path = root / "run_ablation.sbatch"
            manifest_path.write_text(
                f"""
[suite]
name = "slurm-ablation-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["MACE"]

[[cases]]
name = "mace-ablation"
model = "MACE"
kind = "external_pair"
disabled_command = "run baseline"
enabled_command = "run enabled"
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--write-slurm-script",
                    str(slurm_path),
                    "--ablation-mode-override",
                    "isodelta-enabled",
                ]
            )
            script = slurm_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "COMMON_ARGS+=(--ablation-mode-override isodelta-enabled)",
            script,
        )
        self.assertIn(
            "# CLI runtime overrides: ablation_mode=isodelta-enabled.",
            script,
        )
        self.assertIn("one-sided ablation suite", script)
        self.assertIn("--verify-output-bundle", script)
        self.assertNotIn("--pipeline --pipeline-report", script)
        self.assertNotIn("--verify-pipeline-report", script)

    def test_reuse_passed_skips_existing_valid_case_outputs(self) -> None:
        """Validated outputs should be reusable after an interrupted cluster run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_path = root / "sevennet_trace.json"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            manifest_path.write_text(
                f"""
[suite]
name = "reuse-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "external_pair"
disabled_command = "should-not-run-disabled"
enabled_command = "should-not-run-enabled"
repeat_count = 2
trace_evidence = ["{trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            timing_report_path = isodelta_cluster_suite._external_timing_report_path(
                config,
                config.cases[0],
            )
            timing_report_path.parent.mkdir(parents=True, exist_ok=True)
            timing_report_path.write_text(
                json.dumps(
                    _external_timing_report(
                        "SevenNet",
                        disabled_command="should-not-run-disabled",
                        enabled_command="should-not-run-enabled",
                        log_dir=timing_report_path.parent / "external_logs",
                    )
                ),
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.run_suite(
                config,
                skip_downloads=True,
                skip_gpu_check=True,
                reuse_passed=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["cases"][0]["status"], "reused")
        self.assertEqual(summary["suite_evidence"]["passed_models"], ["SevenNet"])
        self.assertEqual(summary["commands"], [])

    def test_preflight_failure_skips_expensive_case_commands(self) -> None:
        """A failed case preflight should stop the paired timing loop early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_path = root / "sevennet_trace.json"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            python_bin = Path(sys.executable).as_posix()
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-pass"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{trace_path.as_posix()}"]

[[cases]]
name = "mace-preflight"
model = "MACE"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(3)"'
disabled_command = "should-not-run-disabled"
enabled_command = "should-not-run-enabled"
repeat_count = 1
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                skip_downloads=True,
                skip_gpu_check=True,
                keep_going=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            command_names = [record["name"] for record in summary["commands"]]
            preflight_log = (
                output_dir
                / "cases"
                / "mace-preflight"
                / "logs"
                / "preflight.stderr.log"
            )
            preflight_log_exists = preflight_log.exists()
            preflight_log_digest = hashlib.sha256(preflight_log.read_bytes()).hexdigest()
            preflight_fingerprint = next(
                record
                for record in summary["command_log_fingerprints"]
                if record["name"] == "mace-preflight:preflight"
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("mace-preflight:preflight", command_names)
        self.assertFalse(any(":disabled:" in name for name in command_names))
        self.assertFalse(any(":enabled:" in name for name in command_names))
        self.assertTrue(preflight_log_exists)
        self.assertTrue(preflight_fingerprint["stderr"]["exists"])
        self.assertEqual(preflight_fingerprint["stderr"]["sha256"], preflight_log_digest)
        self.assertTrue(
            any(case["status"].startswith("failed:") for case in summary["cases"])
        )

    def test_manifest_validation_requires_all_default_foundation_models(self) -> None:
        """A paper manifest should not silently omit MACE or NequIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "missing-models"

[[cases]]
name = "sevennet-only"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "missing required model cases",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_rejects_unknown_artifact_required_by(self) -> None:
        """Artifact model references should not hide spelling mistakes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "bad-artifact-reference"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "data.ext"
required_by = ["TypoModel"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "required_by names unknown case models",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_requires_sha256_for_required_artifacts_when_enabled(self) -> None:
        """Final paper manifests can require immutable digests for all inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "strict-artifacts"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "data.ext"
required = true
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "required artifact needs sha256",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_rejects_malformed_artifact_sha256(self) -> None:
        """Digest fields should be full SHA-256 hex strings, not placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "bad-sha"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "data.ext"
sha256 = "replace-with-real-sha256"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "64-character hexadecimal SHA-256",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_rejects_enabled_external_pair_disable_env(self) -> None:
        """External enabled runs must not inherit the cache-disable switch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "bad-mode-controls"
required_models = ["MACE"]

[[cases]]
name = "mace-pair"
model = "MACE"
kind = "external_pair"
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
enabled_env = { SEVENN_ISODELTA_HALO_DISABLE = "1" }
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "enabled mode must leave SEVENN_ISODELTA_HALO_DISABLE unset or false",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_accepts_false_enabled_disable_env(self) -> None:
        """Explicit false disable flags should match the C++ enabled runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "false-mode-controls"
required_models = ["MACE"]

[[cases]]
name = "mace-pair"
model = "MACE"
kind = "external_pair"
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
enabled_env = { SEVENN_ISODELTA_HALO_DISABLE = " off " }
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        isodelta_cluster_suite.validate_suite_config(config)
        mode_controls = isodelta_cluster_suite.case_mode_control_record(
            config.cases[0]
        )
        self.assertTrue(mode_controls["disabled_cache_disabled"])
        self.assertFalse(mode_controls["enabled_cache_disabled"])
        self.assertEqual(
            mode_controls["enabled_env"][isodelta_cluster_suite.SEVENNET_DISABLE_ENV],
            "off",
        )
        self.assertEqual(
            mode_controls[isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES_KEY],
            list(isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES),
        )

    def test_manifest_validation_accepts_empty_enabled_disable_env(self) -> None:
        """An empty disable flag should also mean cache-enabled at runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "empty-mode-controls"
required_models = ["NequIP"]

[[cases]]
name = "nequip-pair"
model = "NequIP"
kind = "external_pair"
disabled_command = "python run_nequip.py --mode baseline"
enabled_command = "python run_nequip.py --mode isodelta"
enabled_env = { SEVENN_ISODELTA_HALO_DISABLE = "" }
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        isodelta_cluster_suite.validate_suite_config(config)
        mode_controls = isodelta_cluster_suite.case_mode_control_record(
            config.cases[0]
        )
        self.assertFalse(mode_controls["enabled_cache_disabled"])
        self.assertEqual(
            mode_controls["enabled_env"][isodelta_cluster_suite.SEVENNET_DISABLE_ENV],
            "",
        )
        self.assertEqual(
            mode_controls[isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES_KEY],
            list(isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES),
        )

    def test_download_artifact_copies_file_url_and_checks_sha256(self) -> None:
        """Artifact downloads should verify immutable paper inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.bin"
            destination_path = Path(tmpdir) / "downloaded.bin"
            artifact_bytes = b"isodelta artifact"
            source_path.write_bytes(artifact_bytes)
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            artifact = isodelta_cluster_suite.ArtifactConfig(
                name="local-artifact",
                path=destination_path,
                url=source_path.as_uri(),
                sha256=digest,
            )

            record = isodelta_cluster_suite.download_artifact(artifact)

        self.assertTrue(record["downloaded"])
        self.assertEqual(record["sha256"], digest)
        self.assertEqual(record["download_progress"]["bytes_total"], len(artifact_bytes))
        self.assertEqual(record["download_progress"]["bytes_written"], len(artifact_bytes))
        self.assertEqual(record["download_progress"]["percent"], 100.0)
        self.assertTrue(record["download_progress"]["complete"])

    def test_download_artifact_prints_terminal_progress_when_requested(self) -> None:
        """Cluster runs should expose byte-level artifact download progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.bin"
            destination_path = Path(tmpdir) / "downloaded.bin"
            source_path.write_bytes(b"progress bytes")
            artifact = isodelta_cluster_suite.ArtifactConfig(
                name="progress-artifact",
                path=destination_path,
                url=source_path.as_uri(),
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                record = isodelta_cluster_suite.download_artifact(
                    artifact,
                    progress_label="download-suite",
                )

        self.assertTrue(record["downloaded"])
        self.assertIn(
            "[download-suite] [download progress-artifact] progress-artifact: 100.0%",
            stdout.getvalue(),
        )

    def test_prepare_artifacts_downloads_and_writes_audit_report(self) -> None:
        """Artifact preparation should finish before any GPU case is launched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.bin"
            destination_path = root / "inputs" / "prepared.bin"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            source_path.write_bytes(b"prepared artifact bytes")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                f"""
[suite]
name = "prepare-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "{destination_path.as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--prepare-artifacts"]
            )
            report_path = output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            report = json.loads(report_path.read_text(encoding="utf-8"))
            destination_exists = destination_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(destination_exists)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["artifacts"][0]["actual_sha256"], digest)
        self.assertEqual(report["artifacts"][0]["size_bytes"], len(b"prepared artifact bytes"))
        self.assertEqual(
            report["artifacts"][0]["download_progress"]["bytes_total"],
            len(b"prepared artifact bytes"),
        )
        self.assertTrue(report["artifacts"][0]["download_progress"]["complete"])

    def test_download_artifact_skips_optional_missing_without_url(self) -> None:
        """Optional artifacts may be absent but should be recorded explicitly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = isodelta_cluster_suite.ArtifactConfig(
                name="optional-note",
                path=Path(tmpdir) / "optional.txt",
                required=False,
            )

            record = isodelta_cluster_suite.download_artifact(artifact)

        self.assertFalse(record["downloaded"])
        self.assertTrue(record["skipped_optional_missing"])

    def test_skip_downloads_rejects_missing_required_artifact(self) -> None:
        """A cluster run should not start when required inputs are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "missing-required-artifact"
required_models = ["SevenNet"]

[[artifacts]]
name = "required-dataset"
path = "{(root / "missing_dataset.bin").as_posix()}"
required = true

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
artifacts = ["required-dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "required artifacts are missing",
            ):
                isodelta_cluster_suite.run_suite(
                    config,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )

    def test_plan_only_writes_preflight_manifest_audit(self) -> None:
        """Researchers should inspect planned commands before using GPU time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"cluster input")
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            plan_path = root / "planned_run.json"
            manifest_path.write_text(
                f"""
[suite]
name = "plan-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "{(root / "missing_dataset.bin").as_posix()}"
url = "{source_path.as_uri()}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet-plan"
model = "SevenNet"
kind = "trace_only"
trace_input = "trace.json"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--plan-only",
                    "--plan-output",
                    str(plan_path),
                    "--skip-gpu-check",
                    "--reuse-passed",
                ]
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        self.assertEqual(exit_code, 0)
        self.assertFalse(plan["gpu_check_planned"])
        self.assertTrue(plan["modes"]["reuse_passed"])
        self.assertTrue(plan["cases"][0]["reuse"]["enabled"])
        self.assertTrue(plan["artifacts"][0]["will_download"])
        self.assertTrue(plan["artifacts"][0]["missing_required"])
        self.assertFalse(plan["suite"]["require_artifact_sha256"])
        self.assertFalse(plan["artifacts"][0]["has_sha256"])
        self.assertEqual(
            plan["suite"]["manifest"]["sha256"],
            manifest_digest,
        )
        self.assertEqual(plan["cases"][0]["model"], "SevenNet")
        self.assertEqual(plan["cases"][0]["mode_controls"]["kind"], "trace_only")
        self.assertIn("trace_evidence", plan["cases"][0]["expected_outputs"])
        self.assertIn("environment_snapshot.json", plan["paper_outputs"]["environment_snapshot"])
        self.assertIn("speedup_by_case.svg", plan["paper_outputs"]["speedup_svg"])

    def test_preflight_only_downloads_artifacts_and_runs_case_checks(self) -> None:
        """Preflight-only mode should verify inputs and model launch commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            target_path = root / "downloaded-data.bin"
            source_path.write_bytes(b"cluster preflight input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            preflight_path = root / "preflight_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "{target_path.as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet-preflight"
model = "SevenNet"
kind = "trace_only"
preflight_command = '"{python_bin}" -c "print(12345)"'
preflight_env = {{ OMP_NUM_THREADS = "2", SEVENN_ISODELTA_HALO_PROFILE = "1" }}
trace_input = "trace.json"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--preflight-only",
                    "--preflight-output",
                    str(preflight_path),
                    "--skip-gpu-check",
                ]
            )
            report = json.loads(preflight_path.read_text(encoding="utf-8"))
            stdout_path = Path(report["case_preflights"][0]["stdout_path"])
            stdout_text = stdout_path.read_text(encoding="utf-8")
            command_record = report["commands"][0]
            target_exists = target_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(target_exists)
        self.assertEqual(report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED)
        self.assertEqual(
            report["preflight_report_schema_version"],
            isodelta_cluster_suite.PREFLIGHT_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.PREFLIGHT_REPORT_COMMENT,
        )
        self.assertTrue(report["downloads"][0]["downloaded"])
        self.assertEqual(
            report["case_preflights"][0]["status"],
            isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED,
        )
        self.assertIn("12345", stdout_text)
        self.assertEqual(len(report["command_log_fingerprints"]), 1)
        self.assertIn(
            isodelta_cluster_suite.PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME,
            report["environment_snapshot"],
        )
        self.assertGreater(report["environment_snapshot_fingerprint"]["size_bytes"], 0)
        self.assertEqual(
            len(report["environment_snapshot_fingerprint"]["sha256"]),
            isodelta_cluster_suite.SHA256_HEX_LENGTH,
        )
        self.assertEqual(command_record["cwd"], str(REPO_ROOT))
        self.assertEqual(
            command_record["tracked_env"]["OMP_NUM_THREADS"],
            "2",
        )
        self.assertEqual(
            command_record["tracked_env"][PROFILE_CACHE_ENV],
            ENV_FLAG_ENABLED,
        )
        self.assertIn("SLURM_JOB_ID", command_record["tracked_env"])

    def test_command_records_include_cwd_and_tracked_environment(self) -> None:
        """Command records should explain where and under which mode they ran."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = isodelta_cluster_suite.run_shell_command(
                name="dry-run-command",
                command="unused",
                cwd=root,
                env={
                    DISABLE_CACHE_ENV: ENV_FLAG_ENABLED,
                    PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "OMP_NUM_THREADS": "4",
                },
                timeout_seconds=1.0,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                dry_run=True,
            )

        self.assertEqual(record.cwd, str(root))
        self.assertEqual(record.tracked_env[DISABLE_CACHE_ENV], ENV_FLAG_ENABLED)
        self.assertEqual(record.tracked_env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(record.tracked_env["OMP_NUM_THREADS"], "4")
        self.assertIn("SLURM_JOB_ID", record.tracked_env)

    def test_sevennet_case_runs_experiment_report_checker(self) -> None:
        """SevenNet runs should immediately verify the driver report and logs."""
        launched_commands: list[dict[str, object]] = []
        original_runner = isodelta_cluster_suite.run_argv_command

        def fake_run_argv_command(
            *,
            name: str,
            argv: list[str],
            cwd: Path,
            env: dict[str, str] | None,
            timeout_seconds: float,
            stdout_path: Path,
            stderr_path: Path,
            dry_run: bool,
        ) -> isodelta_cluster_suite.CommandRecord:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            launched_commands.append(
                {
                    "name": name,
                    "argv": argv,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "dry_run": dry_run,
                }
            )
            return isodelta_cluster_suite.CommandRecord(
                name=name,
                command=argv,
                returncode=0,
                elapsed_seconds=0.0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                cwd=str(cwd),
                tracked_env={},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = isodelta_cluster_suite.SuiteConfig(
                name="sevennet-checker-suite",
                manifest_path=root / "suite.toml",
                output_dir=root / "paper_outputs",
                expected_gpus=1,
                required_models=("SevenNet",),
            )
            case = isodelta_cluster_suite.CaseConfig(
                name="sevennet-checker",
                model="SevenNet",
                kind="sevennet_lammps",
                lammps_command="lmp",
                input_path=root / "in.sevenn",
                repeat_count=1,
                min_speedup=None,
            )
            try:
                isodelta_cluster_suite.run_argv_command = fake_run_argv_command
                _, _, _, records = isodelta_cluster_suite.run_sevennet_case(
                    config,
                    case,
                    dry_run=False,
                )
            finally:
                isodelta_cluster_suite.run_argv_command = original_runner

        command_names = [record.name for record in records]
        checker_command = launched_commands[1]
        checker_argv = checker_command["argv"]
        self.assertEqual(command_names, ["sevennet-checker:sevennet-experiment", "sevennet-checker:experiment-report-check"])
        self.assertIn("check_isodelta_experiment_report.py", checker_argv[1])
        self.assertIn("--report", checker_argv)
        self.assertIn("--output", checker_argv)
        self.assertTrue(str(checker_argv[-1]).endswith("experiment_report_check.json"))

    def test_preflight_only_reports_failed_case_check(self) -> None:
        """A nonzero case preflight should fail before expensive model runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            preflight_path = root / "preflight_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-failure-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-preflight-fail"
model = "SevenNet"
kind = "trace_only"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(7)"'
trace_input = "trace.json"
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--preflight-only",
                    "--preflight-output",
                    str(preflight_path),
                    "--skip-gpu-check",
                ]
            )
            report = json.loads(preflight_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_FAILED)
        self.assertEqual(report["case_preflights"][0]["returncode"], 7)
        self.assertEqual(report["failures"][0]["stage"], "case_preflight")

    def test_pipeline_runs_all_paper_stages_and_verifies_bundle(self) -> None:
        """Pipeline mode should chain readiness, preflight, run, and verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-pipeline"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "print(12345)"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = 3
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_trace_count = 3
min_distinct_trace_models = 3
min_speedup = 1.05
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            for case in config.cases:
                timing_path = isodelta_cluster_suite._external_timing_report_path(
                    config,
                    case,
                )
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing_path.write_text(
                    json.dumps(
                        _external_timing_report_for_case(
                            case,
                            baseline_times=[BASELINE_LOOP_TIME_SECONDS] * case.repeat_count,
                            enabled_times=[ISODELTA_LOOP_TIME_SECONDS] * case.repeat_count,
                            log_dir=timing_path.parent / "external_logs",
                        )
                    ),
                    encoding="utf-8",
                )

            original_detect_gpu_count = isodelta_cluster_suite.detect_gpu_count
            isodelta_cluster_suite.detect_gpu_count = lambda: (
                isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
                "unit-test",
            )
            try:
                exit_code = isodelta_cluster_suite.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--pipeline",
                        "--pipeline-report",
                        str(pipeline_report_path),
                        "--reuse-passed",
                    ]
                )
            finally:
                isodelta_cluster_suite.detect_gpu_count = original_detect_gpu_count
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]
            stage_fingerprints = pipeline_report[
                isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY
            ]
            pipeline_verification = pipeline_report[
                isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY
            ]
            pipeline_report_verification = isodelta_cluster_suite.verify_pipeline_report(
                pipeline_report_path
            )
            pipeline_cli_exit_code = isodelta_cluster_suite.main(
                ["--verify-pipeline-report", str(pipeline_report_path)]
            )
            readiness_report_path = output_dir / isodelta_cluster_suite.READINESS_REPORT_NAME
            artifact_report_path = (
                output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            )
            preflight_report_path = output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            readiness_report = json.loads(readiness_report_path.read_text(encoding="utf-8"))
            artifact_report = json.loads(artifact_report_path.read_text(encoding="utf-8"))
            preflight_report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
            uncommented_pipeline_report = dict(pipeline_report)
            uncommented_pipeline_report.pop(
                isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY
            )
            pipeline_report_path.write_text(
                json.dumps(uncommented_pipeline_report),
                encoding="utf-8",
            )
            try:
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)
            except isodelta_cluster_suite.ClusterSuiteError as exc:
                uncommented_pipeline_report_error = str(exc)
            else:
                uncommented_pipeline_report_error = ""
            pipeline_report_path.write_text(
                json.dumps(pipeline_report),
                encoding="utf-8",
            )
            readiness_report_path.write_text(
                readiness_report_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            try:
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)
            except isodelta_cluster_suite.ClusterSuiteError as exc:
                mutated_stage_report_error = str(exc)
            else:
                mutated_stage_report_error = ""

        self.assertEqual(exit_code, 0)
        self.assertEqual(pipeline_cli_exit_code, 0)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_PASSED)
        self.assertEqual(
            pipeline_report[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.PIPELINE_REPORT_COMMENT,
        )
        self.assertEqual(
            readiness_report[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.READINESS_REPORT_COMMENT,
        )
        self.assertEqual(
            artifact_report[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_COMMENT,
        )
        self.assertEqual(
            preflight_report[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.PREFLIGHT_REPORT_COMMENT,
        )
        self.assertEqual(
            stage_names,
            [
                "readiness",
                "prepare_artifacts",
                "preflight",
                "plan",
                "run_suite",
                "verify_output_bundle",
            ],
        )
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(len(stage_fingerprints), len(stage_names))
        self.assertEqual(pipeline_report_verification["status"], "passed")
        self.assertEqual(
            pipeline_report_verification["verified_stage_report_count"],
            len(stage_names),
        )
        self.assertEqual(
            pipeline_report_verification["preflight_gpu_check"]["detected_gpus"],
            isodelta_cluster_suite.DEFAULT_EXPECTED_GPU_COUNT,
        )
        self.assertEqual(
            pipeline_report_verification["readiness_report"]["verified_check_count"],
            len(isodelta_cluster_suite.FINAL_PAPER_READINESS_CHECK_NAMES),
        )
        self.assertGreaterEqual(
            pipeline_report_verification["artifact_preparation_report"][
                "verified_required_artifact_count"
            ],
            1,
        )
        self.assertEqual(
            pipeline_report_verification["run_plan_report"]["verified_case_plan_count"],
            len(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS),
        )
        self.assertEqual(pipeline_verification["status"], "passed")
        self.assertEqual(
            pipeline_verification["verified_artifact_index_count"],
            verification["verified_artifact_index_count"],
        )
        self.assertEqual(
            pipeline_verification["verified_evidence_file_count"],
            verification["verified_evidence_file_count"],
        )
        self.assertEqual(
            pipeline_verification["verified_paper_artifact_semantic_count"],
            verification["verified_paper_artifact_semantic_count"],
        )
        self.assertGreaterEqual(
            pipeline_verification["verified_artifact_count"],
            1,
        )
        self.assertIn("preflight_report", summary["artifact_fingerprints"])
        self.assertIn("preflight_environment_snapshot", summary["artifact_fingerprints"])
        self.assertIn("run_plan", summary["artifact_fingerprints"])
        self.assertNotIn("pipeline_report", summary["artifact_fingerprints"])
        self.assertIn("pipeline_report.report_comment", uncommented_pipeline_report_error)
        self.assertIn("SHA-256 mismatch", mutated_stage_report_error)

    def test_pipeline_stops_when_readiness_fails(self) -> None:
        """Pipeline mode should not prepare inputs or run cases after a failed gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-readiness-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["missing_sevennet_trace.json"]

[[cases]]
name = "mace-trace"
model = "MACE"
kind = "trace_only"
trace_evidence = ["missing_mace_trace.json"]

[[cases]]
name = "nequip-trace"
model = "NequIP"
kind = "trace_only"
trace_evidence = ["missing_nequip_trace.json"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pipeline",
                    "--pipeline-report",
                    str(pipeline_report_path),
                    "--skip-gpu-check",
                ]
            )
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            readiness_path = output_dir / isodelta_cluster_suite.READINESS_REPORT_NAME
            stopped_before_prepare = not (
                output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            ).exists()
            stopped_before_preflight = not (
                output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            ).exists()
            stopped_before_plan = not (
                output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            ).exists()
            stopped_before_summary = not (
                output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            ).exists()
            readiness_exists = readiness_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual([stage["name"] for stage in pipeline_report["stages"]], ["readiness"])
        self.assertTrue(readiness_exists)
        self.assertTrue(stopped_before_prepare)
        self.assertTrue(stopped_before_preflight)
        self.assertTrue(stopped_before_plan)
        self.assertTrue(stopped_before_summary)

    def test_pipeline_stops_when_preflight_fails(self) -> None:
        """A failed model preflight should stop before plan, run, and summary stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline preflight failure input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-preflight-fails"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                preflight_exit_code = (
                    PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE
                    if model_name == "MACE"
                    else isodelta_cluster_suite.SUCCESS_RETURN_CODE
                )
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit({preflight_exit_code})"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = {PAPER_REPEAT_COUNT}
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-preflight-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = {PAPER_REPEAT_COUNT}
min_trace_count = {PAPER_REPEAT_COUNT}
min_distinct_trace_models = {len(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS)}
min_speedup = {PIPELINE_MIN_SPEEDUP}
min_speedup_95ci_lower_bound = {PIPELINE_MIN_SPEEDUP_LOWER_BOUND}

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pipeline",
                    "--pipeline-report",
                    str(pipeline_report_path),
                    "--skip-gpu-check",
                ]
            )
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            preflight_path = output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]
            stopped_before_plan = not (
                output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            ).exists()
            stopped_before_summary = not (
                output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            ).exists()
            artifact_report_exists = (
                output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            ).exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual(stage_names, ["readiness", "prepare_artifacts", "preflight"])
        self.assertEqual(preflight_report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_FAILED)
        self.assertEqual(preflight_report["failures"][0]["stage"], "case_preflight")
        self.assertEqual(
            preflight_report["case_preflights"][1]["returncode"],
            PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE,
        )
        self.assertTrue(artifact_report_exists)
        self.assertTrue(stopped_before_plan)
        self.assertTrue(stopped_before_summary)

    def test_pipeline_reports_bundle_verification_failure(self) -> None:
        """A final fingerprint mismatch should fail the pipeline after run output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline verification failure input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-verification-fails"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(0)"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = {PAPER_REPEAT_COUNT}
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-verification-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = {PAPER_REPEAT_COUNT}
min_trace_count = {PAPER_REPEAT_COUNT}
min_distinct_trace_models = {len(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS)}
min_speedup = {PIPELINE_MIN_SPEEDUP}
min_speedup_95ci_lower_bound = {PIPELINE_MIN_SPEEDUP_LOWER_BOUND}

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            for case in config.cases:
                timing_path = isodelta_cluster_suite._external_timing_report_path(
                    config,
                    case,
                )
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing_path.write_text(
                    json.dumps(
                        _external_timing_report_for_case(
                            case,
                            baseline_times=[BASELINE_LOOP_TIME_SECONDS] * case.repeat_count,
                            enabled_times=[ISODELTA_LOOP_TIME_SECONDS] * case.repeat_count,
                            log_dir=timing_path.parent / "external_logs",
                        )
                    ),
                    encoding="utf-8",
                )

            original_verify_output_bundle = isodelta_cluster_suite.verify_output_bundle

            def corrupt_case_summary_before_verify(bundle_or_summary_path: Path) -> dict[str, object]:
                """Mutate a generated table so the real bundle verifier must reject it."""
                case_summary_path = output_dir / "tables" / "case_summary.csv"
                case_summary_path.write_text(
                    case_summary_path.read_text(encoding="utf-8") + "\ncorrupted-row\n",
                    encoding="utf-8",
                )
                return original_verify_output_bundle(bundle_or_summary_path)

            isodelta_cluster_suite.verify_output_bundle = corrupt_case_summary_before_verify
            try:
                exit_code = isodelta_cluster_suite.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--pipeline",
                        "--pipeline-report",
                        str(pipeline_report_path),
                        "--skip-gpu-check",
                        "--reuse-passed",
                    ]
                )
            finally:
                isodelta_cluster_suite.verify_output_bundle = original_verify_output_bundle
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]
            verification_stage = pipeline_report["stages"][-1]
            pipeline_verification = pipeline_report[
                isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY
            ]
            stage_fingerprints = pipeline_report[
                isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY
            ]

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual(
            stage_names,
            [
                "readiness",
                "prepare_artifacts",
                "preflight",
                "plan",
                "run_suite",
                "verify_output_bundle",
            ],
        )
        self.assertEqual(verification_stage["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertIn("SHA-256 mismatch", verification_stage["detail"])
        self.assertEqual(pipeline_verification["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertIn("SHA-256 mismatch", pipeline_verification["detail"])
        self.assertEqual(len(stage_fingerprints), len(stage_names))

    def test_verify_pipeline_report_rejects_failed_pipeline_status(self) -> None:
        """The publication verifier should not pass a failed pipeline report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_report_path = Path(tmpdir) / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_FAILED,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_REPORT_PASSED_STATUS_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)
            with contextlib.redirect_stderr(io.StringIO()):
                cli_exit_code = isodelta_cluster_suite.main(
                    ["--verify-pipeline-report", str(pipeline_report_path)]
                )

        self.assertEqual(cli_exit_code, 1)

    def test_verify_pipeline_report_rejects_dry_run_passed_mode(self) -> None:
        """Dry-run mode should remain planned rather than publication-passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_report_path = Path(tmpdir) / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(dry_run=True),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_DRY_RUN_PASSED_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_skipped_gpu_check_mode(self) -> None:
        """Final-paper pipeline evidence should prove the requested GPUs were checked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_report_path = Path(tmpdir) / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(skip_gpu_check=True),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_GPU_CHECK_SKIPPED_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_allowed_gpu_mismatch_mode(self) -> None:
        """Final-paper pipeline evidence should not allow an undersized GPU job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_report_path = Path(tmpdir) / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(allow_gpu_mismatch=True),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_GPU_MISMATCH_ALLOWED_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_requires_final_paper_suite_models(self) -> None:
        """Pipeline suite metadata should retain the three-model paper scope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            suite_record = _pipeline_suite_record(root, output_dir)
            suite_record["required_models"] = ["SevenNet"]
            pipeline_report_path = root / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_SUITE_REQUIRED_MODELS_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_unsupported_runtime_override(
        self,
    ) -> None:
        """Pipeline reports should not hide unknown CLI overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            suite_record = _pipeline_suite_record(root, output_dir)
            suite_record["runtime_overrides"] = {"unknown_override": "1"}
            pipeline_report_path = root / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_UNSUPPORTED_RUNTIME_OVERRIDE_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_one_sided_runtime_override(
        self,
    ) -> None:
        """Final-paper pipeline evidence should not pass as one-sided ablation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            suite_record = _pipeline_suite_record(root, output_dir)
            suite_record["runtime_overrides"] = {
                "ablation_mode": isodelta_cluster_suite.ABLATION_MODE_ENABLED_ONLY
            }
            pipeline_report_path = root / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_ONE_SIDED_RUNTIME_OVERRIDE_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_shallow_passed_report(self) -> None:
        """A top-level passed status should not replace the full stage sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            pipeline_report_path = root / "pipeline_report.json"
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": _pipeline_suite_record(root, output_dir),
                        "stages": [],
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_REQUIRED_STAGES_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_requires_success_stage_fingerprints(self) -> None:
        """Every non-skipped success stage should carry a report fingerprint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            pipeline_report_path = root / "pipeline_report.json"
            stages = [
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_READINESS,
                    "status": isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
                    "report_path": str(output_dir / "readiness.json"),
                    "detail": None,
                },
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_PREPARE_ARTIFACTS,
                    "status": isodelta_cluster_suite.PIPELINE_STAGE_STATUS_READY,
                    "report_path": str(output_dir / "artifacts.json"),
                    "detail": None,
                },
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_PREFLIGHT,
                    "status": isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED,
                    "report_path": str(output_dir / "preflight.json"),
                    "detail": None,
                },
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_PLAN,
                    "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                    "report_path": str(output_dir / "plan.json"),
                    "detail": None,
                },
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_RUN_SUITE,
                    "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                    "report_path": str(output_dir / "summary.json"),
                    "detail": None,
                },
                {
                    "name": isodelta_cluster_suite.PIPELINE_STAGE_VERIFY_OUTPUT_BUNDLE,
                    "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                    "report_path": str(output_dir / "summary.json"),
                    "detail": None,
                },
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": _pipeline_suite_record(root, output_dir),
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {"name": stage["name"], "report": None}
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "report is required for readiness",
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_absent_success_stage_fingerprint(self) -> None:
        """Success stage fingerprints should prove existing report files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            pipeline_report_path = root / "pipeline_report.json"
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(output_dir / f"{stage_name}.json"),
                    "detail": None,
                }
                for stage_name, stage_status in zip(
                    isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES,
                    _pipeline_success_stage_statuses(),
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": _pipeline_suite_record(root, output_dir),
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": {
                                    "path": stage["report_path"],
                                    "exists": False,
                                    "sha256": None,
                                    "size_bytes": None,
                                },
                            }
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "expected a present fingerprint record",
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_preflight_skipped_gpu_check(
        self,
    ) -> None:
        """A matching preflight fingerprint should still prove GPU semantics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir()
            pipeline_report_path = root / "pipeline_report.json"
            suite_record = _pipeline_suite_record(root, output_dir)
            stage_names = isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES
            stage_statuses = _pipeline_success_stage_statuses()
            report_paths = [
                output_dir / "readiness.json",
                output_dir / "artifacts.json",
                output_dir / "preflight.json",
                output_dir / "plan.json",
                output_dir / "summary.json",
                output_dir / "summary.json",
            ]
            for report_path in set(report_paths):
                report_path.write_text("{}", encoding="utf-8")
            report_paths[0].write_text(
                json.dumps(_pipeline_readiness_report()),
                encoding="utf-8",
            )
            report_paths[1].write_text(
                json.dumps(_pipeline_artifact_preparation_report(suite_record)),
                encoding="utf-8",
            )
            report_paths[3].write_text(
                json.dumps(_pipeline_plan_report(suite_record)),
                encoding="utf-8",
            )
            report_paths[2].write_text(
                json.dumps(
                    _pipeline_preflight_report(
                        detected_gpus=None,
                        skip_gpu_check=True,
                        skipped=True,
                    )
                ),
                encoding="utf-8",
            )
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(report_path),
                    "detail": None,
                }
                for stage_name, stage_status, report_path in zip(
                    stage_names,
                    stage_statuses,
                    report_paths,
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": isodelta_cluster_suite.generated_artifact_record(
                                    Path(str(stage["report_path"]))
                                ),
                            }
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_PREFLIGHT_GPU_CHECK_REQUIRED_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_failed_readiness_check(self) -> None:
        """A top-level ready stage should still prove every readiness check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir()
            pipeline_report_path = root / "pipeline_report.json"
            suite_record = _pipeline_suite_record(root, output_dir)
            report_paths = [
                output_dir / "readiness.json",
                output_dir / "artifacts.json",
                output_dir / "preflight.json",
                output_dir / "plan.json",
                output_dir / "summary.json",
                output_dir / "summary.json",
            ]
            for report_path in set(report_paths):
                report_path.write_text("{}", encoding="utf-8")
            report_paths[0].write_text(
                json.dumps(
                    _pipeline_readiness_report(failed_check="artifact_sha256_gate")
                ),
                encoding="utf-8",
            )
            report_paths[2].write_text(
                json.dumps(_pipeline_preflight_report()),
                encoding="utf-8",
            )
            stage_names = isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES
            stage_statuses = _pipeline_success_stage_statuses()
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(report_path),
                    "detail": None,
                }
                for stage_name, stage_status, report_path in zip(
                    stage_names,
                    stage_statuses,
                    report_paths,
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": isodelta_cluster_suite.generated_artifact_record(
                                    Path(str(stage["report_path"]))
                                ),
                            }
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_READINESS_CHECKS_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_failed_artifact_preparation(self) -> None:
        """A ready pipeline stage should still prove prepared artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir()
            pipeline_report_path = root / "pipeline_report.json"
            suite_record = _pipeline_suite_record(root, output_dir)
            report_paths = [
                output_dir / "readiness.json",
                output_dir / "artifacts.json",
                output_dir / "preflight.json",
                output_dir / "plan.json",
                output_dir / "summary.json",
                output_dir / "summary.json",
            ]
            for report_path in set(report_paths):
                report_path.write_text("{}", encoding="utf-8")
            report_paths[0].write_text(
                json.dumps(_pipeline_readiness_report()),
                encoding="utf-8",
            )
            report_paths[1].write_text(
                json.dumps(
                    _pipeline_artifact_preparation_report(
                        suite_record,
                        missing_required=True,
                    )
                ),
                encoding="utf-8",
            )
            report_paths[2].write_text(
                json.dumps(_pipeline_preflight_report()),
                encoding="utf-8",
            )
            stage_names = isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES
            stage_statuses = _pipeline_success_stage_statuses()
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(report_path),
                    "detail": None,
                }
                for stage_name, stage_status, report_path in zip(
                    stage_names,
                    stage_statuses,
                    report_paths,
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": isodelta_cluster_suite.generated_artifact_record(
                                    Path(str(stage["report_path"]))
                                ),
                            }
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_ARTIFACT_PREPARATION_ARTIFACTS_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_rejects_gpu_skipped_run_plan(self) -> None:
        """A passed pipeline should reject a plan that skipped GPU checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir()
            pipeline_report_path = root / "pipeline_report.json"
            suite_record = _pipeline_suite_record(root, output_dir)
            report_paths = [
                output_dir / "readiness.json",
                output_dir / "artifacts.json",
                output_dir / "preflight.json",
                output_dir / "plan.json",
                output_dir / "summary.json",
                output_dir / "summary.json",
            ]
            for report_path in set(report_paths):
                report_path.write_text("{}", encoding="utf-8")
            report_paths[0].write_text(
                json.dumps(_pipeline_readiness_report()),
                encoding="utf-8",
            )
            report_paths[1].write_text(
                json.dumps(_pipeline_artifact_preparation_report(suite_record)),
                encoding="utf-8",
            )
            report_paths[2].write_text(
                json.dumps(_pipeline_preflight_report()),
                encoding="utf-8",
            )
            report_paths[3].write_text(
                json.dumps(_pipeline_plan_report(suite_record, skip_gpu_check=True)),
                encoding="utf-8",
            )
            stage_names = isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES
            stage_statuses = _pipeline_success_stage_statuses()
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(report_path),
                    "detail": None,
                }
                for stage_name, stage_status, report_path in zip(
                    stage_names,
                    stage_statuses,
                    report_paths,
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": isodelta_cluster_suite.generated_artifact_record(
                                    Path(str(stage["report_path"]))
                                ),
                            }
                            for stage in stages
                        ],
                        isodelta_cluster_suite.OUTPUT_BUNDLE_VERIFICATION_KEY: {
                            "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(isodelta_cluster_suite.PIPELINE_PLAN_MODES_ERROR),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_verify_pipeline_report_requires_passed_bundle_verification(self) -> None:
        """A passed pipeline report should include final bundle verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            output_dir.mkdir()
            pipeline_report_path = root / "pipeline_report.json"
            suite_record = _pipeline_suite_record(root, output_dir)
            report_paths = [
                output_dir / "readiness.json",
                output_dir / "artifacts.json",
                output_dir / "preflight.json",
                output_dir / "plan.json",
                output_dir / "summary.json",
                output_dir / "summary.json",
            ]
            for report_path in set(report_paths):
                report_path.write_text("{}", encoding="utf-8")
            report_paths[0].write_text(
                json.dumps(_pipeline_readiness_report()),
                encoding="utf-8",
            )
            report_paths[1].write_text(
                json.dumps(_pipeline_artifact_preparation_report(suite_record)),
                encoding="utf-8",
            )
            report_paths[3].write_text(
                json.dumps(_pipeline_plan_report(suite_record)),
                encoding="utf-8",
            )
            report_paths[2].write_text(
                json.dumps(_pipeline_preflight_report()),
                encoding="utf-8",
            )
            stage_names = isodelta_cluster_suite.REQUIRED_PIPELINE_STAGE_NAMES
            stage_statuses = _pipeline_success_stage_statuses()
            stages = [
                {
                    "name": stage_name,
                    "status": stage_status,
                    "report_path": str(report_path),
                    "detail": None,
                }
                for stage_name, stage_status, report_path in zip(
                    stage_names,
                    stage_statuses,
                    report_paths,
                )
            ]
            pipeline_report_path.write_text(
                json.dumps(
                    {
                        "pipeline_report_schema_version": (
                            isodelta_cluster_suite.PIPELINE_REPORT_SCHEMA_VERSION
                        ),
                        "status": isodelta_cluster_suite.PIPELINE_STATUS_PASSED,
                        "modes": _pipeline_report_modes(),
                        "suite": suite_record,
                        "stages": stages,
                        isodelta_cluster_suite.STAGE_REPORT_FINGERPRINTS_KEY: [
                            {
                                "name": stage["name"],
                                "report": isodelta_cluster_suite.generated_artifact_record(
                                    Path(str(stage["report_path"]))
                                ),
                            }
                            for stage in stages
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                re.escape(
                    isodelta_cluster_suite.PIPELINE_BUNDLE_VERIFICATION_REQUIRED_ERROR
                ),
            ):
                isodelta_cluster_suite.verify_pipeline_report(pipeline_report_path)

    def test_readiness_check_accepts_strict_three_model_paired_manifest(self) -> None:
        """A final paper manifest should prove strict input and model coverage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"strict cluster input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "ready-suite"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-ready"
model = "SevenNet"
kind = "sevennet_lammps"
preflight_command = 'python -c "import sevenn"'
lammps_command = "mpiexec -n 8 lmp"
input = "inputs/in.sevennet"
artifacts = ["dataset"]

[[cases]]
name = "mace-ready"
model = "MACE"
kind = "external_pair"
preflight_command = 'python -c "import mace"'
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
artifacts = ["dataset"]

[[cases]]
name = "nequip-ready"
model = "NequIP"
kind = "external_pair"
preflight_command = 'python -c "import nequip"'
disabled_command = "python run_nequip.py --mode baseline"
enabled_command = "python run_nequip.py --mode isodelta"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--readiness-check"]
            )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        self.assertEqual(exit_code, 0)

    def test_readiness_check_rejects_one_sided_sevennet_ablation(self) -> None:
        """Final paper readiness must not accept one-sided SevenNet timings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"strict cluster input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "one-sided-ready-suite"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-ablation"
model = "SevenNet"
kind = "sevennet_lammps"
preflight_command = 'python -c "import sevenn"'
lammps_command = "mpiexec -n 8 lmp"
input = "inputs/in.sevennet"
ablation_mode = "isodelta-enabled"
artifacts = ["dataset"]

[[cases]]
name = "mace-ready"
model = "MACE"
kind = "external_pair"
preflight_command = 'python -c "import mace"'
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
artifacts = ["dataset"]

[[cases]]
name = "nequip-ready"
model = "NequIP"
kind = "external_pair"
preflight_command = 'python -c "import nequip"'
disabled_command = "python run_nequip.py --mode baseline"
enabled_command = "python run_nequip.py --mode isodelta"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            failed_checks = {
                check["name"] for check in report["checks"] if not check["passed"]
            }

        self.assertEqual(report["status"], "failed")
        self.assertIn("paired_enabled_disabled_cases", failed_checks)
        self.assertIn("sevennet_final_paper_ablation_mode", failed_checks)

    def test_readiness_check_rejects_one_sided_external_pair_ablation(self) -> None:
        """Final paper readiness must not accept one-sided MACE/NequIP timings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"strict cluster input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "external-one-sided-ready-suite"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-ready"
model = "SevenNet"
kind = "sevennet_lammps"
preflight_command = 'python -c "import sevenn"'
lammps_command = "mpiexec -n 8 lmp"
input = "inputs/in.sevennet"
artifacts = ["dataset"]

[[cases]]
name = "mace-ablation"
model = "MACE"
kind = "external_pair"
preflight_command = 'python -c "import mace"'
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
ablation_mode = "baseline-disabled"
artifacts = ["dataset"]

[[cases]]
name = "nequip-ready"
model = "NequIP"
kind = "external_pair"
preflight_command = 'python -c "import nequip"'
disabled_command = "python run_nequip.py --mode baseline"
enabled_command = "python run_nequip.py --mode isodelta"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            failed_checks = {
                check["name"] for check in report["checks"] if not check["passed"]
            }

        self.assertEqual(report["status"], "failed")
        self.assertIn("paired_enabled_disabled_cases", failed_checks)
        self.assertIn("external_pair_final_paper_ablation_mode", failed_checks)

    def test_readiness_check_rejects_unedited_template_manifest(self) -> None:
        """A generated template must be filled in before paper execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            isodelta_cluster_suite.write_template(manifest_path)
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--readiness-check"]
            )
            failed_checks = {
                check["name"] for check in report["checks"] if not check["passed"]
            }

        self.assertEqual(report["status"], "failed")
        self.assertEqual(exit_code, 1)
        self.assertIn("manifest_schema", failed_checks)
        self.assertIn("artifact_template_markers_removed", failed_checks)

    def test_external_timing_report_rejects_inconsistent_speedup(self) -> None:
        """External MACE/NequIP timing rows should be internally auditable."""
        report = _external_timing_report("NequIP")
        report["speedup_vs_disabled_cache"] = EXPECTED_SPEEDUP + 0.5
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "must match baseline / enabled seconds",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_rejects_inconsistent_raw_mean(self) -> None:
        """External timing reports should keep raw samples and means aligned."""
        report = _external_timing_report("NequIP")
        report["baseline_mean_seconds"] = BASELINE_LOOP_TIME_SECONDS + 0.25
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "baseline_mean_seconds must match raw timing samples",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_rejects_mismatched_mode_controls(self) -> None:
        """External timing evidence must prove the same on/off controls as the manifest."""
        report = _external_timing_report("NequIP")
        report["mode_controls"]["enabled_env"][
            isodelta_cluster_suite.SEVENNET_DISABLE_ENV
        ] = ENV_FLAG_ENABLED
        report["mode_controls"]["enabled_cache_disabled"] = True
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "mode_controls must match manifest disabled/enabled controls",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_requires_repeat_command_records(self) -> None:
        """Every external disabled/enabled repeat should have command provenance."""
        report = _external_timing_report("NequIP")
        missing_name = "nequip-existing:enabled:1"
        report["commands"] = [
            command for command in report["commands"] if command["name"] != missing_name
        ]
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            f"exactly one {missing_name} command record",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_rejects_failed_command_record(self) -> None:
        """External timing rows should not hide failed disabled/enabled commands."""
        report = _external_timing_report("NequIP")
        failed_name = "nequip-existing:disabled:0"
        for command in report["commands"]:
            if command["name"] == failed_name:
                command["returncode"] = 2
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            f"{failed_name}.returncode must be 0",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_rejects_mutated_command_log(self) -> None:
        """External timing report log fingerprints should protect archived logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "nequip_timing.json"
            report = _external_timing_report("NequIP", log_dir=root / "logs")
            case = isodelta_cluster_suite.CaseConfig(
                name="nequip-existing",
                model="NequIP",
                kind="external_pair",
                disabled_command="run baseline",
                enabled_command="run enabled",
                repeat_count=2,
                min_speedup=1.1,
            )
            verification = isodelta_cluster_suite.validate_external_timing_report(
                report,
                case,
                report_path=report_path,
            )
            mutated_log = Path(str(report["commands"][0]["stdout_path"]))
            mutated_log.write_text("changed after report\n", encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "SHA-256 mismatch",
            ):
                isodelta_cluster_suite.validate_external_timing_report(
                    report,
                    case,
                    report_path=report_path,
                )

        self.assertEqual(verification["verified_command_log_count"], 4)

    def test_speedup_lower_bound_gate_rejects_uncertain_case(self) -> None:
        """A mean speedup should fail when its conservative CI bound is weak."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sevennet_trace_path = root / "sevennet_trace.json"
            nequip_trace_path = root / "nequip_trace.json"
            nequip_timing_path = root / "nequip_timing.json"
            output_dir = root / "paper_outputs"
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            nequip_trace_path.write_text(json.dumps(_trace_evidence("NequIP")), encoding="utf-8")
            nequip_timing_path.write_text(
                json.dumps(
                    _external_timing_report(
                        "NequIP",
                        disabled_command="baseline",
                        enabled_command="enabled",
                        log_dir=nequip_timing_path.parent / "external_logs",
                    )
                ),
                encoding="utf-8",
            )
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "lower-bound-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-pass"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "external_pair"
disabled_command = "baseline"
enabled_command = "enabled"
repeat_count = 2
external_timing_report = "{nequip_timing_path.as_posix()}"
trace_evidence = ["{nequip_trace_path.as_posix()}"]
min_speedup_95ci_lower_bound = 1.1
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                collect_only=True,
                skip_downloads=True,
                skip_gpu_check=True,
                keep_going=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            nequip_case = next(case for case in summary["cases"] if case["model"] == "NequIP")

        self.assertEqual(exit_code, 1)
        self.assertIn("speedup 95% CI lower bound", nequip_case["status"])
        self.assertLess(nequip_case["speedup_95ci_lower_bound"], 1.1)

    def test_collect_only_writes_tables_correlations_and_svg_figures(self) -> None:
        """Existing evidence should become paper tables, correlations, and graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "sevennet_benchmark.json"
            sevennet_trace_path = root / "sevennet_trace.json"
            mace_trace_path = root / "mace_trace.json"
            nequip_trace_path = root / "nequip_trace.json"
            nequip_timing_path = root / "nequip_timing.json"
            output_dir = root / "paper_outputs"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            mace_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            nequip_trace_path.write_text(json.dumps(_trace_evidence("NequIP")), encoding="utf-8")
            nequip_timing_path.write_text(
                json.dumps(
                    _external_timing_report(
                        "NequIP",
                        disabled_command="python -c print('baseline')",
                        enabled_command="python -c print('enabled')",
                        log_dir=nequip_timing_path.parent / "external_logs",
                    )
                ),
                encoding="utf-8",
            )
            output_dir.mkdir(parents=True)
            preflight_report_path = output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            plan_path = output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            preflight_report_path.write_text(
                json.dumps({"status": "passed", "kind": "preflight"}),
                encoding="utf-8",
            )
            plan_path.write_text(
                json.dumps({"status": "planned", "kind": "plan"}),
                encoding="utf-8",
            )
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "collect-only-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
min_speedup = 1.1
min_hit_rate_percent = 75.0
min_trace_hit_rate_percent = 50.0
required_models = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "sevennet_lammps"
lammps_command = "lmp"
input = "in.sevennet"
benchmark_report = "{benchmark_path.as_posix()}"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]
min_enabled_cache_attempts = 10
min_enabled_cache_hits = 8

[[cases]]
name = "mace-existing"
model = "MACE"
kind = "trace_only"
trace_evidence = ["{mace_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "external_pair"
disabled_command = "python -c print('baseline')"
enabled_command = "python -c print('enabled')"
repeat_count = 2
external_timing_report = "{nequip_timing_path.as_posix()}"
trace_evidence = ["{nequip_trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                collect_only=True,
                skip_downloads=True,
                skip_gpu_check=True,
            )

            summary_path = output_dir / "isodelta_cluster_paper_summary.json"
            environment_snapshot = output_dir / "environment_snapshot.json"
            case_summary_csv = output_dir / "tables" / "case_summary.csv"
            correlation_csv = output_dir / "tables" / "correlation.csv"
            repeat_timing_csv = output_dir / "tables" / "repeat_timing.csv"
            repeat_timing_markdown = output_dir / "tables" / "repeat_timing.md"
            speedup_svg = output_dir / "figures" / "speedup_by_case.svg"
            manifest_snapshot = output_dir / "isodelta_cluster_suite_manifest.toml"

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            environment_payload = json.loads(environment_snapshot.read_text(encoding="utf-8"))
            environment_snapshot_exists = environment_snapshot.exists()
            case_summary_exists = case_summary_csv.exists()
            correlation_exists = correlation_csv.exists()
            repeat_timing_exists = repeat_timing_csv.exists()
            repeat_timing_markdown_exists = repeat_timing_markdown.exists()
            speedup_svg_exists = speedup_svg.exists()
            manifest_snapshot_exists = manifest_snapshot.exists()
            case_summary_text = case_summary_csv.read_text(encoding="utf-8")
            repeat_timing_text = repeat_timing_csv.read_text(encoding="utf-8")
            speedup_svg_text = speedup_svg.read_text(encoding="utf-8")
            manifest_snapshot_text = manifest_snapshot.read_text(encoding="utf-8")
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            environment_digest = hashlib.sha256(environment_snapshot.read_bytes()).hexdigest()
            case_summary_digest = hashlib.sha256(case_summary_csv.read_bytes()).hexdigest()
            repeat_timing_digest = hashlib.sha256(repeat_timing_csv.read_bytes()).hexdigest()
            manifest_snapshot_digest = hashlib.sha256(manifest_snapshot.read_bytes()).hexdigest()
            preflight_digest = hashlib.sha256(preflight_report_path.read_bytes()).hexdigest()
            plan_digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            benchmark_digest = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
            sevennet_trace_digest = hashlib.sha256(sevennet_trace_path.read_bytes()).hexdigest()
            nequip_timing_digest = hashlib.sha256(nequip_timing_path.read_bytes()).hexdigest()
            environment_size = environment_snapshot.stat().st_size
            case_summary_size = case_summary_csv.stat().st_size
            repeat_timing_size = repeat_timing_csv.stat().st_size
            manifest_snapshot_size = manifest_snapshot.stat().st_size
            preflight_size = preflight_report_path.stat().st_size
            plan_size = plan_path.stat().st_size
            nequip_case = next(
                case for case in summary["cases"] if case["model"] == "NequIP"
            )
            nequip_mode_controls = summary["case_mode_controls"]["nequip-existing"]
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            case_summary_csv.write_text(case_summary_text + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "SHA-256 mismatch",
            ):
                isodelta_cluster_suite.verify_output_bundle(summary_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(verification["status"], "passed")
        self.assertGreaterEqual(verification["verified_artifact_count"], 1)
        self.assertEqual(
            verification["verified_artifact_index_count"],
            verification["verified_artifact_count"],
        )
        self.assertEqual(
            verification["verified_paper_artifact_semantic_count"],
            len(isodelta_cluster_suite.REQUIRED_PAPER_ARTIFACT_NAMES),
        )
        self.assertEqual(verification["verified_evidence_file_count"], 5)
        self.assertTrue(environment_snapshot_exists)
        self.assertTrue(case_summary_exists)
        self.assertTrue(correlation_exists)
        self.assertTrue(repeat_timing_exists)
        self.assertTrue(repeat_timing_markdown_exists)
        self.assertTrue(speedup_svg_exists)
        self.assertTrue(manifest_snapshot_exists)
        self.assertEqual(
            summary["suite"]["manifest"]["sha256"],
            manifest_digest,
        )
        self.assertEqual(
            summary[isodelta_cluster_suite.GENERATED_REPORT_COMMENT_KEY],
            isodelta_cluster_suite.SUMMARY_REPORT_COMMENT,
        )
        self.assertEqual(
            environment_payload[
                isodelta_cluster_suite.GENERATED_ARTIFACT_COMMENT_KEY
            ],
            isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                "environment_snapshot"
            ],
        )
        self.assertEqual(len(summary["cases"]), 3)
        self.assertEqual(summary["suite_evidence"]["distinct_trace_model_count"], 3)
        self.assertTrue(nequip_mode_controls["disabled_cache_disabled"])
        self.assertFalse(nequip_mode_controls["enabled_cache_disabled"])
        self.assertEqual(
            nequip_mode_controls["disabled_env"][isodelta_cluster_suite.SEVENNET_DISABLE_ENV],
            ENV_FLAG_ENABLED,
        )
        self.assertIsNone(
            nequip_mode_controls["enabled_env"][isodelta_cluster_suite.SEVENNET_DISABLE_ENV]
        )
        self.assertEqual(
            nequip_mode_controls[isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES_KEY],
            list(isodelta_cluster_suite.ENV_FLAG_FALSE_VALUES),
        )
        self.assertIn("speedup_vs_disabled_cache", case_summary_text)
        self.assertIn("baseline_sample_variance_seconds", case_summary_text)
        self.assertIn("enabled_sample_stddev_seconds", case_summary_text)
        self.assertIn("baseline_mean_95ci_half_width_seconds", case_summary_text)
        self.assertIn("speedup_95ci_lower_bound", case_summary_text)
        self.assertTrue(
            case_summary_text.startswith(
                isodelta_cluster_suite._csv_comment_line(
                    isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                        "case_summary_csv"
                    ]
                )
            )
        )
        self.assertIn("baseline-disabled", repeat_timing_text)
        self.assertIn("isodelta-enabled", repeat_timing_text)
        self.assertIn("external_timing_report", repeat_timing_text)
        self.assertIn("disabled", repeat_timing_text)
        self.assertIn("enabled", repeat_timing_text)
        self.assertTrue(
            repeat_timing_text.startswith(
                isodelta_cluster_suite._csv_comment_line(
                    isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                        "repeat_timing_csv"
                    ]
                )
            )
        )
        self.assertEqual(nequip_case["baseline_timing_count"], 2)
        self.assertAlmostEqual(
            nequip_case["baseline_mean_95ci_half_width_seconds"],
            isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER,
        )
        self.assertAlmostEqual(
            nequip_case["enabled_mean_95ci_half_width_seconds"],
            isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER,
        )
        self.assertAlmostEqual(
            nequip_case["speedup_95ci_lower_bound"],
            (
                BASELINE_LOOP_TIME_SECONDS
                - isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            )
            / (
                ISODELTA_LOOP_TIME_SECONDS
                + isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            ),
        )
        self.assertAlmostEqual(
            nequip_case["speedup_95ci_upper_bound"],
            (
                BASELINE_LOOP_TIME_SECONDS
                + isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            )
            / (
                ISODELTA_LOOP_TIME_SECONDS
                - isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            ),
        )
        self.assertIn("<svg", speedup_svg_text)
        self.assertIn(
            f"<desc>{isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS['speedup_svg']}</desc>",
            speedup_svg_text,
        )
        self.assertTrue(
            manifest_snapshot_text.startswith(
                "# "
                + isodelta_cluster_suite.PAPER_ARTIFACT_COMMENTS[
                    "manifest_snapshot"
                ]
            )
        )
        self.assertEqual(
            environment_payload["snapshot_schema_version"],
            isodelta_cluster_suite.ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        )
        self.assertIn("torch", environment_payload["package_versions"])
        self.assertIn("CUDA_VISIBLE_DEVICES", environment_payload["selected_environment"])
        self.assertEqual(
            summary["artifact_fingerprints"]["environment_snapshot"]["sha256"],
            environment_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["environment_snapshot"]["size_bytes"],
            environment_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["case_summary_csv"]["sha256"],
            case_summary_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["case_summary_csv"]["size_bytes"],
            case_summary_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["repeat_timing_csv"]["sha256"],
            repeat_timing_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["repeat_timing_csv"]["size_bytes"],
            repeat_timing_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["manifest_snapshot"]["sha256"],
            manifest_snapshot_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["manifest_snapshot"]["size_bytes"],
            manifest_snapshot_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["preflight_report"]["sha256"],
            preflight_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["preflight_report"]["size_bytes"],
            preflight_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["run_plan"]["sha256"],
            plan_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["run_plan"]["size_bytes"],
            plan_size,
        )
        self.assertEqual(
            summary["evidence_fingerprints"]["sevennet-existing"]["benchmark_report"]["sha256"],
            benchmark_digest,
        )
        self.assertEqual(
            summary["evidence_fingerprints"]["sevennet-existing"]["trace_evidence"][0]["sha256"],
            sevennet_trace_digest,
        )
        self.assertEqual(
            summary["evidence_fingerprints"]["nequip-existing"]["external_timing_report"]["sha256"],
            nequip_timing_digest,
        )

    def test_run_suite_verifies_output_bundle_after_writing(self) -> None:
        """A normal suite run should reopen and verify its own paper bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            trace_path = root / "sevennet_trace.json"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "verify-after-write-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            verification_calls: list[Path] = []
            original_verify_output_bundle = isodelta_cluster_suite.verify_output_bundle

            def failing_verify_output_bundle(path: Path) -> dict[str, object]:
                verification_calls.append(Path(path))
                raise isodelta_cluster_suite.ClusterSuiteError("forced bundle failure")

            try:
                isodelta_cluster_suite.verify_output_bundle = failing_verify_output_bundle
                exit_code = isodelta_cluster_suite.run_suite(
                    config,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )
            finally:
                isodelta_cluster_suite.verify_output_bundle = original_verify_output_bundle

        self.assertEqual(exit_code, 1)
        self.assertEqual(verification_calls, [output_dir])

    def test_run_suite_verifies_sevennet_experiment_check_evidence(self) -> None:
        """A normal SevenNet run should archive and reverify report-check evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            trace_path = root / "sevennet_trace.json"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "sevennet-check-e2e-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-paper"
model = "SevenNet"
kind = "sevennet_lammps"
lammps_command = "lmp"
input = "inputs/in.sevennet"
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["SevenNet"]
min_enabled_cache_attempts = 1
min_enabled_cache_hits = 1
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            original_runner = isodelta_cluster_suite.run_argv_command

            def fake_run_argv_command(
                *,
                name: str,
                argv: list[str],
                cwd: Path,
                env: dict[str, str] | None,
                timeout_seconds: float,
                stdout_path: Path,
                stderr_path: Path,
                dry_run: bool,
            ) -> isodelta_cluster_suite.CommandRecord:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_text(f"{name} stdout\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                if name.endswith(":sevennet-experiment"):
                    experiment_dir = Path(argv[argv.index("--output-dir") + 1])
                    benchmark_path = (
                        experiment_dir
                        / "benchmark"
                        / isodelta_cluster_suite.BENCHMARK_REPORT_NAME
                    )
                    bundle_path = (
                        experiment_dir
                        / isodelta_cluster_suite.BUNDLE_EVIDENCE_NAME
                    )
                    report_path = (
                        experiment_dir
                        / isodelta_cluster_suite.EXPERIMENT_REPORT_NAME
                    )
                    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
                    benchmark_path.write_text(
                        json.dumps(_benchmark_report()),
                        encoding="utf-8",
                    )
                    bundle_path.write_text(
                        json.dumps(
                            isodelta_cluster_suite.bundle_check.validate_bundle(
                                benchmark_report=benchmark_path,
                                trace_evidence_paths=[trace_path],
                                required_models=["SevenNet"],
                                thresholds=(
                                    isodelta_cluster_suite.bundle_check.BundleThresholds(
                                        max_abs_thermo_delta=(
                                            config.cases[0].max_abs_thermo_delta
                                        ),
                                        min_paired_thermo_count=(
                                            config.cases[0].min_paired_thermo_count
                                        ),
                                        min_speedup=config.cases[0].min_speedup,
                                        min_hit_rate_percent=(
                                            config.cases[0].min_hit_rate_percent
                                        ),
                                        min_enabled_cache_attempts=(
                                            config.cases[0].min_enabled_cache_attempts
                                        ),
                                        min_enabled_cache_hits=(
                                            config.cases[0].min_enabled_cache_hits
                                        ),
                                        min_trace_hit_rate_percent=(
                                            config.cases[0].min_trace_hit_rate_percent
                                        ),
                                        min_trace_estimated_speedup=(
                                            config.cases[0].min_trace_estimated_speedup
                                        ),
                                        min_trace_metadata_fraction_percent=(
                                            config.cases[
                                                0
                                            ].min_trace_metadata_fraction_percent
                                        ),
                                    )
                                ),
                            )
                        ),
                        encoding="utf-8",
                    )
                    report_path.write_text(
                        json.dumps(_experiment_report(command_count=5)),
                        encoding="utf-8",
                    )
                if name.endswith(":experiment-report-check"):
                    report_path = Path(argv[argv.index("--report") + 1])
                    output_path = Path(argv[argv.index("--output") + 1])
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                    command_count = len(report_payload["commands"])
                    output_path.write_text(
                        json.dumps(
                            _experiment_report_check(
                                report_path,
                                command_count=command_count,
                            )
                        ),
                        encoding="utf-8",
                    )
                return isodelta_cluster_suite.CommandRecord(
                    name=name,
                    command=argv,
                    returncode=isodelta_cluster_suite.SUCCESS_RETURN_CODE,
                    elapsed_seconds=0.0,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    cwd=str(cwd),
                    tracked_env=isodelta_cluster_suite.command_environment_snapshot(
                        env
                    ),
                )

            try:
                isodelta_cluster_suite.run_argv_command = fake_run_argv_command
                exit_code = isodelta_cluster_suite.run_suite(
                    config,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )
            finally:
                isodelta_cluster_suite.run_argv_command = original_runner

            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            evidence = summary[isodelta_cluster_suite.EVIDENCE_FINGERPRINTS_KEY][
                "sevennet-paper"
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(verification["verified_experiment_report_check_count"], 1)
        self.assertIsNotNone(
            evidence[isodelta_cluster_suite.EXPERIMENT_REPORT_KEY]
        )
        self.assertIsNotNone(
            evidence[isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_KEY]
        )
        self.assertEqual(
            summary["cases"][0][isodelta_cluster_suite.EXPERIMENT_REPORT_KEY],
            evidence[isodelta_cluster_suite.EXPERIMENT_REPORT_KEY]["path"],
        )
        self.assertEqual(
            summary["cases"][0][isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_KEY],
            evidence[isodelta_cluster_suite.EXPERIMENT_REPORT_CHECK_KEY]["path"],
        )

    def test_collect_only_rejects_insufficient_distinct_trace_models(self) -> None:
        """Suite-level gates should count model labels inside trace evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sevennet_trace_path = root / "sevennet_trace.json"
            mace_trace_path = root / "mace_trace.json"
            mislabeled_nequip_trace_path = root / "nequip_trace.json"
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            mace_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            mislabeled_nequip_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "weak-distinct-traces"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
min_trace_count = 3
min_distinct_trace_models = 3

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]

[[cases]]
name = "mace-existing"
model = "MACE"
kind = "trace_only"
trace_evidence = ["{mace_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "trace_only"
trace_evidence = ["{mislabeled_nequip_trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "distinct trace model count",
            ):
                isodelta_cluster_suite.run_suite(
                    config,
                    collect_only=True,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )


if __name__ == "__main__":
    unittest.main()
