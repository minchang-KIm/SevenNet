"""Run the complete lightweight validation suite for IsoDelta-Halo changes.

The real LAMMPS/LibTorch build is still required for full runtime validation,
but this script keeps every dependency-free check in one reproducible command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


# Each command is kept as an argument list so Windows, Linux, and CI shells do
# not reinterpret paths or quoting differently.
REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_REPORT_SCHEMA_VERSION = "isodelta-lightweight-validation-report-v1"
DEFAULT_OUTPUT_TAIL_CHARS = 4000
SUCCESS_RETURN_CODE = 0
VALIDATION_COMMANDS = (
    (sys.executable, "tools/run_isodelta_static_checks.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_halo_static.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_evidence_bundle_check.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_benchmark_report_check.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_benchmark_parser.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_build_prereqs.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_experiment_runner.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_cluster_paper_suite.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_lammps_binary_check.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_mlip_trace_check.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_mlip_trace_demo.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_validation_runner.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_sync_gate.py"),
    (
        sys.executable,
        "-m",
        "py_compile",
        "tools/check_isodelta_build_prereqs.py",
        "tools/check_isodelta_benchmark_report.py",
        "tools/check_isodelta_evidence_bundle.py",
        "tools/check_isodelta_lammps_binary.py",
        "tools/check_isodelta_mlip_trace.py",
        "tools/run_isodelta_experiment.py",
        "tools/run_isodelta_cluster_paper_suite.py",
        "tools/run_isodelta_lammps_benchmark.py",
        "tools/run_isodelta_mlip_trace_demo.py",
        "tools/run_isodelta_static_checks.py",
        "tools/run_isodelta_sync_gate.py",
        "tools/run_isodelta_validation.py",
        "tests/unit_tests/test_isodelta_evidence_bundle_check.py",
        "tests/unit_tests/test_isodelta_benchmark_report_check.py",
        "tests/unit_tests/test_isodelta_benchmark_parser.py",
        "tests/unit_tests/test_isodelta_build_prereqs.py",
        "tests/unit_tests/test_isodelta_experiment_runner.py",
        "tests/unit_tests/test_isodelta_cluster_paper_suite.py",
        "tests/unit_tests/test_isodelta_lammps_binary_check.py",
        "tests/unit_tests/test_isodelta_mlip_trace_check.py",
        "tests/unit_tests/test_isodelta_mlip_trace_demo.py",
        "tests/unit_tests/test_isodelta_validation_runner.py",
        "tests/unit_tests/test_isodelta_sync_gate.py",
        "tests/unit_tests/test_isodelta_halo_static.py",
    ),
    ("git", "diff", "--check"),
)


def _command_tail(text: str) -> str:
    """Keep report records bounded while preserving the end of command output."""
    return text[-DEFAULT_OUTPUT_TAIL_CHARS:]


def _run_metadata_command(command: tuple[str, ...]) -> str | None:
    """Run a short repository metadata command for validation provenance."""
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != SUCCESS_RETURN_CODE:
        return None
    return completed.stdout.strip()


def _run(command: tuple[str, ...]) -> dict[str, object]:
    """Run one validation command and return a machine-readable record."""
    print(f"[IsoDelta-Halo validation] {' '.join(command)}")
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started_at
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": _command_tail(completed.stdout),
        "stderr_tail": _command_tail(completed.stderr),
    }


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    """Write a validation report that can be archived with a sync attempt."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_validation(report_path: Path | None = None) -> int:
    """Execute checks and optionally persist a JSON sync evidence report."""
    command_records: list[dict[str, object]] = []
    status = "passed"
    for command in VALIDATION_COMMANDS:
        record = _run(command)
        command_records.append(record)
        if record["returncode"] != SUCCESS_RETURN_CODE:
            status = "failed"
            break
    payload: dict[str, object] = {
        "validation_report_schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _run_metadata_command(("git", "rev-parse", "HEAD")),
        "git_branch": _run_metadata_command(("git", "branch", "--show-current")),
        "git_status_short": _run_metadata_command(("git", "status", "--short")),
        "commands": command_records,
    }
    if report_path is not None:
        _write_report(report_path, payload)
        print(f"[IsoDelta-Halo validation] wrote report to {report_path}")
    if status == "passed":
        print("[IsoDelta-Halo validation] all checks passed")
        return SUCCESS_RETURN_CODE
    print("[IsoDelta-Halo validation] checks failed", file=sys.stderr)
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse validation runner options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Write a JSON validation report for commit/push evidence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute all lightweight checks in the same order before every commit."""
    args = parse_args(argv)
    return run_validation(args.report_path)


if __name__ == "__main__":
    raise SystemExit(main())
