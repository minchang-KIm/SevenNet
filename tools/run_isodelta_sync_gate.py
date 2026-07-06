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
PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED = "auth-prompt-disabled"
PUSH_FAILURE_REASON_NETWORK_UNREACHABLE = "network-unreachable"
PUSH_FAILURE_REASON_UNKNOWN = "unknown"
AUTH_PROMPT_DISABLED_MARKERS = (
    "Cannot prompt because user interactivity has been disabled",
    "could not read Username for",
    "terminal prompts disabled",
)
NETWORK_UNREACHABLE_MARKERS = (
    "Could not connect to server",
    "Failed to connect to",
    "Could not resolve host",
)
PUSH_FAILURE_SUGGESTED_ACTIONS = {
    PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED: (
        "Authenticate the Git HTTPS remote outside the non-interactive sync gate "
        "or switch the remote to an already-authenticated SSH URL, then rerun "
        "run_isodelta_sync_gate.py."
    ),
    PUSH_FAILURE_REASON_NETWORK_UNREACHABLE: (
        "Run the sync gate from a network that can reach the Git remote, then "
        "rerun run_isodelta_sync_gate.py."
    ),
    PUSH_FAILURE_REASON_UNKNOWN: (
        "Inspect commands[-1].stderr_tail and rerun run_isodelta_sync_gate.py "
        "after resolving the reported git push error."
    ),
}
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


def _sync_git_provenance(remote: str, branch: str | None) -> dict[str, str | None]:
    """Collect local and remote refs that define one sync attempt."""
    remote_tracking_ref = f"refs/remotes/{remote}/{branch}" if branch else None
    return {
        "current_branch": _current_branch(),
        "head_commit": _metadata_command(("git", "rev-parse", "HEAD")),
        "target_branch_commit": (
            _metadata_command(("git", "rev-parse", branch)) if branch else None
        ),
        "remote_url": _metadata_command(("git", "remote", "get-url", remote)),
        "remote_tracking_ref": remote_tracking_ref,
        "remote_tracking_commit": (
            _metadata_command(("git", "rev-parse", "--verify", remote_tracking_ref))
            if remote_tracking_ref
            else None
        ),
    }


def _classify_push_failure(push_record: dict[str, Any] | None) -> dict[str, str] | None:
    """Classify a failed push so sync reports are actionable without logs open."""
    if push_record is None:
        return None
    if push_record.get("returncode") == SUCCESS_RETURN_CODE:
        return None
    stderr_tail = str(push_record.get("stderr_tail", ""))
    stdout_tail = str(push_record.get("stdout_tail", ""))
    combined_output = f"{stderr_tail}\n{stdout_tail}"
    if any(marker in combined_output for marker in AUTH_PROMPT_DISABLED_MARKERS):
        reason = PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED
    elif any(marker in combined_output for marker in NETWORK_UNREACHABLE_MARKERS):
        reason = PUSH_FAILURE_REASON_NETWORK_UNREACHABLE
    else:
        reason = PUSH_FAILURE_REASON_UNKNOWN
    return {
        "reason": reason,
        "detail": _tail(combined_output.strip()),
        "suggested_action": PUSH_FAILURE_SUGGESTED_ACTIONS[reason],
    }


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
        "git_provenance": _sync_git_provenance(remote, resolved_branch),
        "commands": command_records,
        "push_failure": _classify_push_failure(push_record),
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
