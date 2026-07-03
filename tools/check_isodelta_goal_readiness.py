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
        "--readiness-check",
        "--verify-output-bundle",
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
        "--readiness-check",
        "--verify-output-bundle",
        "run_isodelta_sync_gate.py",
    ),
    "tests/unit_tests/test_isodelta_cluster_paper_suite.py": (
        "test_prepare_artifacts_downloads_and_writes_audit_report",
        "test_preflight_only_downloads_artifacts_and_runs_case_checks",
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
