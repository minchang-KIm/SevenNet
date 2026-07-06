"""Audit local IsoDelta-Halo goal readiness from the current source tree.

The audit is intentionally dependency-free and evidence-oriented. It does not
claim that remote authentication works or that a heavy 8-GPU LAMMPS job has
already run; instead it verifies that the local implementation, validation
scripts, sync gate, paper-suite tooling, documentation, and CI workflow are
present and mutually connected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any


# These constants make the audit criteria explicit rather than hiding them in
# ad-hoc string checks.
REPO_ROOT_PARENT_DEPTH = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
GOAL_READINESS_SCHEMA_VERSION = "isodelta-goal-readiness-v1"
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
GIT_METADATA_TIMEOUT_SECONDS = 10.0
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_NOT_ENFORCED = "not_enforced"
EXPECTED_BRANCH = "codex/isodelta-halo-runtime"
PYTHON_HEADER_PREFIX = '"""'
ISODELTA_PYTHON_GLOB_PATTERNS = (
    "tools/*isodelta*.py",
    "tests/unit_tests/test_isodelta*.py",
)
REQUIRED_FILE_SNIPPETS = {
    "sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp": (
        "IsoDelta-Halo",
        "SEVENN_ISODELTA_HALO_DISABLE",
        "SEVENN_ISODELTA_HALO_PROFILE",
        "comm_cache_miss_reason_is_valid",
    ),
    "sevenn/pair_e3gnn/pair_e3gnn_parallel.h": (
        "IsoDelta",
        "metadata",
    ),
    "tools/run_isodelta_cluster_paper_suite.py": (
        "--prepare-artifacts",
        "--preflight-only",
        "--pipeline",
        "--readiness-check",
        "--verify-output-bundle",
        "--verify-pipeline-report",
        "verify_pipeline_report",
        "_validate_external_timing_command_records",
        "_require_external_command_log_fingerprints",
        "_require_external_timing_reports_from_summary",
        "COMMAND_LOG_FINGERPRINTS_KEY",
        "verified_external_command_log_count",
        "REQUIRED_PAPER_ARTIFACT_NAMES",
        "_require_artifact_index_alignment",
        "verified_artifact_index_count",
        "_require_case_summary_cell_values",
        "_summary_correlations_by_metric_pair",
        "_format_csv_value",
        "PAPER_COMMAND_TIMING_COLUMNS",
        "_command_timing_rows",
        "_require_command_timing_csv",
        "_require_command_timing_markdown",
        "PAPER_REPEAT_TIMING_COLUMNS",
        "_repeat_timing_rows",
        "_repeat_timing_rows_from_summary",
        "_require_repeat_timing_csv",
        "_require_repeat_timing_markdown",
        "_require_paper_artifact_semantics",
        "verified_paper_artifact_semantic_count",
        "_require_speedup_svg_semantics",
        "_require_scatter_svg_semantics",
        "_svg_element_count",
        "verify_output",
        "verifying output bundle",
        "DOWNLOAD_PROGRESS_INTERVAL_BYTES",
        "_update_download_progress",
        "_emit_download_progress",
        "download_progress",
        "EXTERNAL_DISABLED_COMMAND_LABEL",
        "EXTERNAL_ENABLED_COMMAND_LABEL",
        "PIPELINE_OUTPUT",
        '--verify-pipeline-report "$PIPELINE_OUTPUT"',
        "STAGE_REPORT_FINGERPRINTS_KEY",
        "OUTPUT_BUNDLE_VERIFICATION_KEY",
        "case_mode_controls",
        "MODE_CONTROLS_KEY",
        "COMMAND_ENV_SNAPSHOT_KEYS",
        "tracked_env",
        "ABLATION_MODE_CHOICES",
        "ABLATION_OVERRIDE_CASE_KINDS",
        "--ablation-mode-override",
        "_apply_ablation_mode_override",
        "_runtime_override_comment",
        "CLI runtime overrides",
        "runtime_overrides",
        "_has_one_sided_ablation_case",
        "one-sided ablation suite",
        "TIMING_MODES_KEY",
        "_external_timing_modes_for_ablation",
        "_validate_one_sided_benchmark_report",
        "external_pair_final_paper_ablation_mode",
        "sevennet_final_paper_ablation_mode",
        "EVIDENCE_FINGERPRINTS_KEY",
        "evidence_fingerprints",
        "MACE",
        "NequIP",
    ),
    "tools/run_isodelta_validation.py": (
        "VALIDATION_REPORT_SCHEMA_VERSION",
        "--expected-branch",
        "test_isodelta_sync_gate.py",
    ),
    "tools/run_isodelta_lammps_benchmark.py": (
        "ABLATION_MODE_CHOICES",
        "--ablation-mode",
        "benchmark_cases_for_ablation_mode",
        '"benchmark_cases": [case.name for case in benchmark_cases]',
    ),
    "tools/run_isodelta_experiment.py": (
        "ABLATION_MODE_CHOICES",
        "--ablation-mode",
        "ablation-benchmark",
        "should_run_publishable_pair_gates",
        "min_speedup requires paired ablation_mode",
    ),
    "tools/run_isodelta_sync_gate.py": (
        "SYNC_REPORT_SCHEMA_VERSION",
        "PUSH_AUTH_ENVIRONMENT",
        "--expected-branch",
        "_sync_git_provenance",
        "git_provenance",
        "--push-failure-bundle",
        "_write_push_failure_bundle",
        "PUSH_FAILURE_BUNDLE_KEY",
        "STATUS_PUSH_FAILED",
        "PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED",
        "_classify_push_failure",
        '"push_failure": _classify_push_failure(push_record)',
    ),
    "tools/check_isodelta_goal_readiness.py": (
        "ISODELTA_PYTHON_GLOB_PATTERNS",
        "_audit_isodelta_python_headers",
    ),
    ".github/workflows/isodelta-halo.yml": (
        "--report-path isodelta_validation_report.json",
        "actions/upload-artifact@v4",
    ),
    "docs/source/user_guide/isodelta_halo.md": (
        "--prepare-artifacts",
        "--preflight-only",
        "--pipeline",
        "--readiness-check",
        "--verify-output-bundle",
        "`commands` array",
        "`command_log_fingerprints` array",
        "MACE/NequIP stdout/stderr logs cannot drift silently",
        "semantic paper-artifact checks",
        'same cell values as `summary["cases"]`',
        '`summary["correlations"]`',
        "speedup chart must include every measured-speedup",
        "scatter plots must contain the same",
        "command_timing.csv",
        "repeat_timing.csv",
        '`summary["commands"]`',
        "source timing evidence",
        '`ablation_mode = "paired"`',
        "--ablation-mode-override",
        "`suite.runtime_overrides`",
        "CLI runtime overrides",
        "one-sided ablation suite",
        "`--verify-output-bundle`",
        "one-sided SevenNet",
        "external_pair ablation",
        "--ablation-mode baseline-disabled",
        "--ablation-mode isodelta-enabled",
        "one-sided ablation",
        "`tools/*isodelta*.py`",
        "`tests/unit_tests/test_isodelta*.py`",
        "normal suite run also reopens the completed",
        "bundle verification failure returns a nonzero exit code",
        "`git_provenance` fields",
        "`download_progress`",
        "byte-level download progress",
        "human-facing path index points to a different file",
        '`--verify-pipeline-report "$PIPELINE_OUTPUT"`',
        "run_isodelta_sync_gate.py",
        "`push_failure` object",
        "`--push-failure-bundle`",
        "`push_failure_bundle`",
        "`auth-prompt-disabled`",
    ),
    "tests/unit_tests/test_isodelta_cluster_paper_suite.py": (
        "test_prepare_artifacts_downloads_and_writes_audit_report",
        "test_download_artifact_prints_terminal_progress_when_requested",
        "test_preflight_only_downloads_artifacts_and_runs_case_checks",
        "test_pipeline_runs_all_paper_stages_and_verifies_bundle",
        "test_pipeline_stops_when_readiness_fails",
        "test_pipeline_stops_when_preflight_fails",
        "test_pipeline_reports_bundle_verification_failure",
        "test_manifest_validation_rejects_enabled_external_pair_disable_env",
        "test_external_timing_report_rejects_mismatched_mode_controls",
        "test_external_timing_report_requires_repeat_command_records",
        "test_external_timing_report_rejects_failed_command_record",
        "test_external_timing_report_rejects_mutated_command_log",
        "test_command_records_include_cwd_and_tracked_environment",
        "test_verify_output_bundle_rejects_mismatched_command_fingerprints",
        "test_verify_output_bundle_rejects_mutated_source_evidence",
        "test_verify_output_bundle_rejects_mutated_external_command_log",
        "test_verify_output_bundle_rejects_semantically_invalid_svg_artifact",
        "test_verify_output_bundle_rejects_speedup_svg_label_drift",
        "test_verify_output_bundle_rejects_scatter_svg_point_count_drift",
        "test_verify_output_bundle_rejects_mismatched_artifact_index_path",
        "test_verify_output_bundle_rejects_case_summary_value_drift",
        "test_verify_output_bundle_rejects_correlation_value_drift",
        "test_verify_output_bundle_rejects_command_timing_value_drift",
        "test_verify_output_bundle_rejects_repeat_timing_value_drift",
        "test_run_suite_verifies_output_bundle_after_writing",
        "test_write_slurm_script_rejects_collect_only_pipeline_launcher",
        "test_readiness_check_accepts_strict_three_model_paired_manifest",
        "test_sevennet_manifest_ablation_mode_reaches_experiment_driver",
        "test_one_sided_sevennet_benchmark_report_validates_as_raw_ablation",
        "test_one_sided_sevennet_benchmark_rejects_speedup_claim",
        "test_readiness_check_rejects_one_sided_sevennet_ablation",
        "test_external_pair_ablation_mode_runs_one_command_side",
        "test_cli_ablation_override_updates_runtime_timing_cases",
        "test_cli_ablation_override_is_recorded_in_run_plan",
        "# CLI runtime overrides: ablation_mode=isodelta-enabled.",
        "test_cli_ablation_override_requires_runtime_timing_case",
        "test_write_slurm_script_uses_ablation_override_without_pipeline",
        "test_external_pair_one_sided_timing_validates_without_speedup",
        "test_external_pair_one_sided_timing_rejects_speedup_claim",
        "test_readiness_check_rejects_one_sided_external_pair_ablation",
    ),
    "tests/unit_tests/test_isodelta_benchmark_parser.py": (
        "test_ablation_mode_selects_one_benchmark_case",
        "test_ablation_only_summary_has_no_speedup_claim",
        "test_validate_benchmark_options_rejects_unknown_ablation_mode",
    ),
    "tests/unit_tests/test_isodelta_experiment_runner.py": (
        "test_build_experiment_commands_supports_one_sided_ablation",
        "test_validate_config_rejects_speedup_gate_for_one_sided_ablation",
    ),
    "tests/unit_tests/test_isodelta_sync_gate.py": (
        "IsoDeltaSyncGateTest",
        "test_validation_command_enforces_target_branch",
        "test_run_sync_records_git_provenance_for_push_target",
        "test_run_sync_can_write_bundle_after_push_failure",
        "test_run_sync_classifies_noninteractive_auth_push_failure",
        "STATUS_VALIDATION_FAILED",
    ),
    "tests/unit_tests/test_isodelta_validation_runner.py": (
        "test_expected_branch_reaches_goal_readiness_command",
        "test_expected_branch_does_not_reach_py_compile",
        "codex/isodelta-halo-runtime",
    ),
    "tests/unit_tests/test_isodelta_goal_readiness.py": (
        "test_expected_branch_names_codex_work_branch",
        "test_goal_readiness_accepts_isodelta_python_headers",
        "test_goal_readiness_rejects_isodelta_python_without_header",
    ),
}
COMMENT_PREFIX_REQUIREMENTS = {
    ".github/workflows/isodelta-halo.yml": "#",
    "docs/source/user_guide/isodelta_halo.md": "<!--",
    "tools/check_isodelta_goal_readiness.py": '"""',
    "tools/run_isodelta_cluster_paper_suite.py": '"""',
    "tools/run_isodelta_sync_gate.py": '"""',
    "tools/run_isodelta_validation.py": '"""',
    "tests/unit_tests/test_isodelta_sync_gate.py": '"""',
}


def _read_text(path: Path) -> str:
    """Read one source file with the repository-wide encoding."""
    return path.read_text(encoding="utf-8")


def _check_record(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """Return one JSON-friendly audit check record."""
    return {"name": name, "passed": passed, "detail": detail}


def _git_metadata(root: Path, command: tuple[str, ...]) -> str | None:
    """Collect short git metadata without making the audit depend on git state."""
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=GIT_METADATA_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != SUCCESS_RETURN_CODE:
        return None
    return completed.stdout.strip()


def _audit_required_snippets(
    root: Path,
    required_file_snippets: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Check that core files exist and still expose required feature markers."""
    records: list[dict[str, Any]] = []
    for relative_path, snippets in required_file_snippets.items():
        path = root / relative_path
        if not path.exists():
            records.append(_check_record(f"required_file:{relative_path}", False, "missing file"))
            continue
        text = _read_text(path)
        missing = [snippet for snippet in snippets if snippet not in text]
        records.append(
            _check_record(
                f"required_file:{relative_path}",
                not missing,
                "missing snippets: " + ", ".join(missing) if missing else "all snippets present",
            )
        )
    return records


def _audit_comment_prefixes(
    root: Path,
    comment_prefix_requirements: dict[str, str],
) -> list[dict[str, Any]]:
    """Check that generated or added documentation/script files explain themselves."""
    records: list[dict[str, Any]] = []
    for relative_path, prefix in comment_prefix_requirements.items():
        path = root / relative_path
        if not path.exists():
            records.append(_check_record(f"comment_prefix:{relative_path}", False, "missing file"))
            continue
        stripped_text = _read_text(path).lstrip()
        records.append(
            _check_record(
                f"comment_prefix:{relative_path}",
                stripped_text.startswith(prefix),
                f"expected prefix {prefix!r}",
            )
        )
    return records


def _audit_isodelta_python_headers(
    root: Path,
    glob_patterns: tuple[str, ...] = ISODELTA_PYTHON_GLOB_PATTERNS,
) -> list[dict[str, Any]]:
    """Require every IsoDelta Python tool or test to start with a file comment."""
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for glob_pattern in glob_patterns:
        for path in sorted(root.glob(glob_pattern)):
            if not path.is_file() or path in seen_paths:
                continue
            seen_paths.add(path)
            relative_path = path.relative_to(root).as_posix()
            stripped_text = _read_text(path).lstrip()
            records.append(
                _check_record(
                    f"isodelta_python_header:{relative_path}",
                    stripped_text.startswith(PYTHON_HEADER_PREFIX),
                    f"expected prefix {PYTHON_HEADER_PREFIX!r}",
                )
            )
    return records


def build_goal_readiness_report(
    *,
    root: Path = REPO_ROOT,
    expected_branch: str | None = None,
    required_file_snippets: dict[str, tuple[str, ...]] | None = None,
    comment_prefix_requirements: dict[str, str] | None = None,
    isodelta_python_glob_patterns: tuple[str, ...] = ISODELTA_PYTHON_GLOB_PATTERNS,
) -> dict[str, Any]:
    """Build a report proving local source readiness for the active goal."""
    snippet_requirements = (
        REQUIRED_FILE_SNIPPETS
        if required_file_snippets is None
        else required_file_snippets
    )
    prefix_requirements = (
        COMMENT_PREFIX_REQUIREMENTS
        if comment_prefix_requirements is None
        else comment_prefix_requirements
    )
    branch = _git_metadata(root, ("git", "branch", "--show-current"))
    checks = []
    if expected_branch is None:
        checks.append(_check_record("branch", True, STATUS_NOT_ENFORCED))
    else:
        checks.append(
            _check_record(
                "branch",
                branch == expected_branch,
                f"observed={branch!r}, expected={expected_branch!r}",
            )
        )
    checks.extend(_audit_required_snippets(root, snippet_requirements))
    checks.extend(_audit_comment_prefixes(root, prefix_requirements))
    checks.extend(_audit_isodelta_python_headers(root, isodelta_python_glob_patterns))
    status = STATUS_PASSED if all(record["passed"] for record in checks) else STATUS_FAILED
    return {
        "goal_readiness_schema_version": GOAL_READINESS_SCHEMA_VERSION,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "expected_branch": expected_branch,
        "git_branch": branch,
        "git_commit": _git_metadata(root, ("git", "rev-parse", "HEAD")),
        "git_status_short": _git_metadata(root, ("git", "status", "--short")),
        "checks": checks,
    }


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    """Persist a goal-readiness report for later completion audits."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the readiness audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path for local goal-readiness evidence",
    )
    parser.add_argument(
        "--expected-branch",
        default=None,
        help="Optionally enforce an expected working branch",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the local goal-readiness audit."""
    args = parse_args(argv)
    report = build_goal_readiness_report(expected_branch=args.expected_branch)
    if args.report_path is not None:
        _write_report(args.report_path, report)
    print(
        json.dumps(
            {
                "goal_readiness_report": str(args.report_path) if args.report_path is not None else None,
                "status": report["status"],
            },
            indent=2,
        )
    )
    return SUCCESS_RETURN_CODE if report["status"] == STATUS_PASSED else FAILURE_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
