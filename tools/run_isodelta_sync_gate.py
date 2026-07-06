"""Run IsoDelta-Halo validation and record the following git push attempt.

This script is intentionally small and dependency-free so a researcher can run
the same gate before every sync: validate first, attempt the requested push
only after validation passes, and keep a JSON report even when authentication or
network state prevents the push from succeeding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


# Constants keep the sync contract explicit for reviewers and test fixtures.
REPO_ROOT_PARENT_DEPTH = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
VALIDATION_RUNNER_PATH = REPO_ROOT / "tools" / "run_isodelta_validation.py"
SYNC_REPORT_SCHEMA_VERSION = "isodelta-sync-gate-report-v1"
DEFAULT_SYNC_REPORT_PATH = Path("isodelta_sync_report.json")
DEFAULT_VALIDATION_REPORT_PATH = Path("isodelta_validation_report.json")
DEFAULT_REMOTE = "fork"
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
COMMAND_OUTPUT_TAIL_CHARS = 4000
STATUS_SYNCED = "synced"
STATUS_VALIDATED = "validated"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_PUSH_FAILED = "push_failed"
PUSH_AUTH_ENVIRONMENT = {
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
    "GIT_ASKPASS": "",
    "SSH_ASKPASS": "",
}


def _tail(text: str) -> str:
    """Keep command output records bounded without hiding final errors."""
    return text[-COMMAND_OUTPUT_TAIL_CHARS:]


def _run_command(
    command: tuple[str, ...],
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one command and return a JSON-serializable command record."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _metadata_command(command: tuple[str, ...]) -> str | None:
    """Return short git metadata, or None when the repository is unavailable."""
    record = _run_command(command)
    if record["returncode"] != SUCCESS_RETURN_CODE:
        return None
    return str(record["stdout_tail"]).strip()


def _current_branch() -> str | None:
    """Return the currently checked-out branch name for the default push target."""
    return _metadata_command(("git", "branch", "--show-current"))


def _validation_command(
    validation_report_path: Path,
    expected_branch: str | None,
) -> tuple[str, ...]:
    """Build the validation command used before any push attempt."""
    command = (
        sys.executable,
        str(VALIDATION_RUNNER_PATH),
        "--report-path",
        str(validation_report_path),
    )
    if expected_branch is not None:
        command = (*command, "--expected-branch", expected_branch)
    return command


def _push_command(remote: str, branch: str) -> tuple[str, ...]:
    """Build the non-interactive push command for one remote branch."""
    return ("git", "push", "-u", remote, branch)


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    """Persist the sync report so failed pushes are still auditable."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sync(
    *,
    remote: str,
    branch: str | None,
    report_path: Path,
    validation_report_path: Path,
    skip_push: bool = False,
    validation_command: tuple[str, ...] | None = None,
    push_command: tuple[str, ...] | None = None,
) -> int:
    """Run validation, then optionally push, and write one sync evidence report."""
    resolved_branch = branch or _current_branch()
    command_records: list[dict[str, Any]] = []
    validation_record = _run_command(
        validation_command or _validation_command(validation_report_path, resolved_branch)
    )
    command_records.append({"name": "validation", **validation_record})
    push_record: dict[str, Any] | None = None
    status = STATUS_VALIDATION_FAILED
    if validation_record["returncode"] == SUCCESS_RETURN_CODE:
        if skip_push:
            status = STATUS_VALIDATED
        elif resolved_branch:
            push_record = _run_command(
                push_command or _push_command(remote, resolved_branch),
                env_overrides=PUSH_AUTH_ENVIRONMENT,
            )
            command_records.append({"name": "push", **push_record})
            status = (
                STATUS_SYNCED
                if push_record["returncode"] == SUCCESS_RETURN_CODE
                else STATUS_PUSH_FAILED
            )
        else:
            status = STATUS_PUSH_FAILED
            push_record = {
                "name": "push",
                "command": [],
                "returncode": FAILURE_RETURN_CODE,
                "elapsed_seconds": 0.0,
                "stdout_tail": "",
                "stderr_tail": "could not determine current branch",
            }
            command_records.append(push_record)

    payload = {
        "sync_report_schema_version": SYNC_REPORT_SCHEMA_VERSION,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "remote": remote,
        "branch": resolved_branch,
        "validation_report_path": str(validation_report_path),
        "git_commit": _metadata_command(("git", "rev-parse", "HEAD")),
        "git_status_short": _metadata_command(("git", "status", "--short")),
        "commands": command_records,
    }
    _write_report(report_path, payload)
    print(json.dumps({"sync_report": str(report_path), "status": status}, indent=2))
    return SUCCESS_RETURN_CODE if status in (STATUS_SYNCED, STATUS_VALIDATED) else FAILURE_RETURN_CODE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sync gate command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Git remote to push after validation")
    parser.add_argument("--branch", help="Branch to push; defaults to the current branch")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_SYNC_REPORT_PATH,
        help="JSON report path for validation and push evidence",
    )
    parser.add_argument(
        "--validation-report-path",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT_PATH,
        help="JSON report path passed to run_isodelta_validation.py",
    )
    parser.add_argument("--skip-push", action="store_true", help="Validate and report without attempting git push")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the validation-plus-push sync gate."""
    args = parse_args(argv)
    return run_sync(
        remote=args.remote,
        branch=args.branch,
        report_path=args.report_path,
        validation_report_path=args.validation_report_path,
        skip_push=args.skip_push,
    )


if __name__ == "__main__":
    raise SystemExit(main())
