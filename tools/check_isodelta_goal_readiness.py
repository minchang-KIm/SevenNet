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
EXPECTED_BRANCH = "isodelta-halo-runtime"
REQUIRED_FILE_SNIPPETS = {
    "sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp": (
        "IsoDelta-Halo",
        "SEVENN_ISODELTA_HALO_DISABLE",
        "SEVENN_ISODELTA_HALO_PROFILE",
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
        "EVIDENCE_FINGERPRINTS_KEY",
        "evidence_fingerprints",
        "MACE",
        "NequIP",
    ),
    "tools/run_isodelta_validation.py": (
        "VALIDATION_REPORT_SCHEMA_VERSION",
        "test_isodelta_sync_gate.py",
    ),
    "tools/run_isodelta_sync_gate.py": (
        "SYNC_REPORT_SCHEMA_VERSION",
        "PUSH_AUTH_ENVIRONMENT",
        "STATUS_PUSH_FAILED",
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
        '`summary["commands"]`',
        "normal suite run also reopens the completed",
        "bundle verification failure returns a nonzero exit code",
        "`download_progress`",
        "byte-level download progress",
        "human-facing path index points to a different file",
        '`--verify-pipeline-report "$PIPELINE_OUTPUT"`',
        "run_isodelta_sync_gate.py",
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
        "test_run_suite_verifies_output_bundle_after_writing",
        "test_write_slurm_script_rejects_collect_only_pipeline_launcher",
        "test_readiness_check_accepts_strict_three_model_paired_manifest",
    ),
    "tests/unit_tests/test_isodelta_sync_gate.py": (
        "IsoDeltaSyncGateTest",
        "STATUS_VALIDATION_FAILED",
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


def build_goal_readiness_report(
    *,
    root: Path = REPO_ROOT,
    expected_branch: str | None = None,
    required_file_snippets: dict[str, tuple[str, ...]] | None = None,
    comment_prefix_requirements: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a report proving local source readiness for the active goal."""
    snippet_requirements = required_file_snippets or REQUIRED_FILE_SNIPPETS
    prefix_requirements = comment_prefix_requirements or COMMENT_PREFIX_REQUIREMENTS
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
