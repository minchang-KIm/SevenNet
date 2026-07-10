"""Run IsoDelta-Halo validation and record the following git push attempt.

This script is intentionally small and dependency-free so a researcher can run
the same gate before every sync: validate first, attempt the requested push
only after validation passes, and keep a JSON report even when authentication or
network state prevents the push from succeeding.
"""

from __future__ import annotations

import argparse
import hashlib
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
EXPECTED_VALIDATION_REPORT_SCHEMA_VERSION = "isodelta-lightweight-validation-report-v1"
GENERATED_REPORT_COMMENT_KEY = "report_comment"
SYNC_REPORT_COMMENT = (
    "IsoDelta-Halo sync gate report linking validation evidence, push outcome, "
    "remote ref verification, worktree state, and failure handoff data."
)
EXPECTED_VALIDATION_REPORT_COMMENT = (
    "IsoDelta-Halo lightweight validation report for pre-commit and pre-push "
    "evidence; records every dependency-free command used by the sync gate."
)
DEFAULT_SYNC_REPORT_PATH = Path("isodelta_sync_report.json")
DEFAULT_VALIDATION_REPORT_PATH = Path("isodelta_validation_report.json")
DEFAULT_REMOTE = "fork"
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
COMMAND_OUTPUT_TAIL_CHARS = 4000
AUDIT_FILE_HASH_READ_CHUNK_BYTES = 1024 * 1024
BUNDLE_HASH_READ_CHUNK_BYTES = AUDIT_FILE_HASH_READ_CHUNK_BYTES
GIT_SHA1_HEX_LENGTH = 40
GIT_SHA256_HEX_LENGTH = 64
GIT_OBJECT_ID_HEX_LENGTHS = frozenset((GIT_SHA1_HEX_LENGTH, GIT_SHA256_HEX_LENGTH))
LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
STATUS_SYNCED = "synced"
STATUS_VALIDATED = "validated"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_VALIDATION_REPORT_MISSING = "validation_report_missing"
STATUS_VALIDATION_REPORT_INVALID = "validation_report_invalid"
STATUS_DIRTY_WORKTREE = "dirty_worktree"
STATUS_GIT_HEAD_UNAVAILABLE = "git_head_unavailable"
STATUS_TARGET_BRANCH_MISMATCH = "target_branch_mismatch"
STATUS_PUSH_FAILED = "push_failed"
STATUS_REMOTE_VERIFICATION_FAILED = "remote_verification_failed"
LOCAL_HEAD_PRECONDITION_KEY = "local_head_precondition"
LOCAL_HEAD_PRECONDITION_COMMAND_NAME = "local_head_precondition"
LOCAL_HEAD_PRECONDITION_DETAIL = "current HEAD commit is unavailable or malformed"
TARGET_BRANCH_PRECONDITION_KEY = "target_branch_precondition"
TARGET_BRANCH_PRECONDITION_COMMAND_NAME = "target_branch_precondition"
TARGET_BRANCH_PRECONDITION_DETAIL = (
    "target branch commit is unavailable, malformed, or different from validated HEAD"
)
REMOTE_REF_VERIFICATION_KEY = "remote_ref_verification"
REMOTE_REF_VERIFY_COMMAND_NAME = "remote_ref_verify"
CURRENT_BRANCH_COMMAND = ("git", "branch", "--show-current")
HEAD_COMMIT_COMMAND = ("git", "rev-parse", "HEAD")
VALIDATION_REPORT_FINGERPRINT_KEY = "validation_report_fingerprint"
VALIDATION_REPORT_SUMMARY_KEY = "validation_report_summary"
SYNC_COMMAND_SUMMARY_KEY = "sync_command_summary"
SYNC_COMMAND_NAME_FIELD = "name"
VALIDATION_REPORT_COMMAND_FIELD = "command"
VALIDATION_REPORT_COMMAND_RETURNCODE_FIELD = "returncode"
VALIDATION_REPORT_COMMAND_ELAPSED_SECONDS_FIELD = "elapsed_seconds"
VALIDATION_REPORT_COMMAND_TEXT_FIELDS = ("stdout_tail", "stderr_tail")
VALIDATION_REPORT_COMMAND_REQUIRED_FIELDS = (
    VALIDATION_REPORT_COMMAND_FIELD,
    VALIDATION_REPORT_COMMAND_RETURNCODE_FIELD,
    VALIDATION_REPORT_COMMAND_ELAPSED_SECONDS_FIELD,
    *VALIDATION_REPORT_COMMAND_TEXT_FIELDS,
)
SYNC_COMMAND_REQUIRED_FIELDS = (
    SYNC_COMMAND_NAME_FIELD,
    *VALIDATION_REPORT_COMMAND_REQUIRED_FIELDS,
)
WORKTREE_STATUS_KEY = "worktree_status"
PUSH_FAILURE_BUNDLE_KEY = "push_failure_bundle"
PUSH_BRANCH_PRECONDITION_COMMAND_NAME = "push_branch_precondition"
PUSH_FAILURE_BUNDLE_COMMAND_NAME = "push_failure_bundle"
PUSH_FAILURE_BUNDLE_VERIFY_COMMAND_NAME = "push_failure_bundle_verify"
PUSH_FAILURE_BUNDLE_STATUS_CREATED = "created"
PUSH_FAILURE_BUNDLE_STATUS_FAILED = "failed"
PUSH_FAILURE_BUNDLE_STATUS_SKIPPED = "skipped"
PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED = "auth-prompt-disabled"
PUSH_FAILURE_REASON_NETWORK_UNREACHABLE = "network-unreachable"
PUSH_FAILURE_REASON_UNKNOWN = "unknown"
VALIDATION_REPORT_PASSED_STATUS = "passed"
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


def _is_git_object_id(value: Any) -> bool:
    """Return whether a value is a full SHA-1 or SHA-256 Git object id."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return (
        len(normalized) in GIT_OBJECT_ID_HEX_LENGTHS
        and all(character in LOWERCASE_HEX_DIGITS for character in normalized)
    )


def _current_branch() -> str | None:
    """Return the currently checked-out branch name for the default push target."""
    return _metadata_command(CURRENT_BRANCH_COMMAND)


def _target_branch_commit_command(branch: str) -> tuple[str, str, str]:
    """Build the git command that resolves the branch intended for push."""
    return ("git", "rev-parse", branch)


def _parse_status_short(status_short: str | None) -> list[dict[str, str]]:
    """Parse git status --short output into stable JSON entries."""
    if status_short is None:
        return []
    entries: list[dict[str, str]] = []
    for line in status_short.splitlines():
        if not line:
            continue
        index_status = line[0] if len(line) >= 1 else " "
        worktree_status = line[1] if len(line) >= 2 else " "
        path = line[3:] if len(line) >= 4 and line[2] == " " else line[2:].strip()
        entries.append(
            {
                "index_status": index_status,
                "worktree_status": worktree_status,
                "path": path,
            }
        )
    return entries


def _worktree_status() -> dict[str, Any]:
    """Return a structured dirty-worktree snapshot for sync provenance."""
    status_short = _metadata_command(("git", "status", "--short"))
    if status_short is None:
        return {
            "available": False,
            "clean": False,
            "entry_count": 0,
            "entries": [],
            "raw": None,
        }
    entries = _parse_status_short(status_short)
    return {
        "available": True,
        "clean": not entries,
        "entry_count": len(entries),
        "entries": entries,
        "raw": status_short,
    }


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


def _remote_ref_verify_command(remote: str, branch: str) -> tuple[str, ...]:
    """Build the command that reads the pushed branch from the remote."""
    return ("git", "ls-remote", "--heads", remote, branch)


def _bundle_command(bundle_path: Path, branch: str) -> tuple[str, ...]:
    """Build a git bundle command for a validated branch after push failure."""
    return ("git", "bundle", "create", str(bundle_path), branch)


def _bundle_verify_command(bundle_path: Path) -> tuple[str, ...]:
    """Build a git command that verifies a generated bundle is readable."""
    return ("git", "bundle", "verify", str(bundle_path))


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a SHA-256 fingerprint for an audit artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(AUDIT_FILE_HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _bundle_file_fingerprint(bundle_path: Path) -> dict[str, Any]:
    """Return a SHA-256 fingerprint for a generated git bundle."""
    return _file_fingerprint(bundle_path)


def _validation_command_record_has_valid_shape(command_record: Any) -> bool:
    """Return whether a validation command record is replayable JSON evidence."""
    if not isinstance(command_record, dict):
        return False
    command = command_record.get(VALIDATION_REPORT_COMMAND_FIELD)
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(command_token, str) or not command_token
            for command_token in command
        )
    ):
        return False
    returncode = command_record.get(VALIDATION_REPORT_COMMAND_RETURNCODE_FIELD)
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        return False
    elapsed_seconds = command_record.get(VALIDATION_REPORT_COMMAND_ELAPSED_SECONDS_FIELD)
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or elapsed_seconds < 0
    ):
        return False
    return all(
        isinstance(command_record.get(text_field), str)
        for text_field in VALIDATION_REPORT_COMMAND_TEXT_FIELDS
    )


def _sync_command_record_has_valid_shape(command_record: Any) -> bool:
    """Return whether a sync command record has a replayable evidence shape."""
    return (
        isinstance(command_record, dict)
        and isinstance(command_record.get(SYNC_COMMAND_NAME_FIELD), str)
        and bool(command_record.get(SYNC_COMMAND_NAME_FIELD))
        and _validation_command_record_has_valid_shape(command_record)
    )


def _sync_command_summary(command_records: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize command record health for quick sync report audits."""
    return {
        "command_count": len(command_records),
        "command_failure_count": sum(
            1
            for command_record in command_records
            if not isinstance(command_record, dict)
            or command_record.get(VALIDATION_REPORT_COMMAND_RETURNCODE_FIELD)
            != SUCCESS_RETURN_CODE
        ),
        "command_missing_field_count": sum(
            1
            for command_record in command_records
            if not isinstance(command_record, dict)
            or any(
                required_field not in command_record
                for required_field in SYNC_COMMAND_REQUIRED_FIELDS
            )
        ),
        "command_invalid_field_count": sum(
            1
            for command_record in command_records
            if not _sync_command_record_has_valid_shape(command_record)
        ),
    }


def _branch_precondition_failure_record(name: str, detail: str) -> dict[str, Any]:
    """Record a replayable branch-discovery precondition failure."""
    record = _run_command(CURRENT_BRANCH_COMMAND)
    if record["returncode"] == SUCCESS_RETURN_CODE:
        record["returncode"] = FAILURE_RETURN_CODE
    stderr_tail = str(record["stderr_tail"])
    record["stderr_tail"] = _tail(
        "\n".join(part for part in (stderr_tail, detail) if part)
    )
    return {"name": name, **record}


def _local_head_precondition_report(head_commit: str | None) -> dict[str, Any]:
    """Describe whether the current checkout exposes a full HEAD object id."""
    verified = _is_git_object_id(head_commit)
    return {
        "command": list(HEAD_COMMIT_COMMAND),
        "head_commit": head_commit,
        "verified": verified,
        "detail": None if verified else LOCAL_HEAD_PRECONDITION_DETAIL,
    }


def _local_head_precondition_failure_record(head_commit: str | None) -> dict[str, Any]:
    """Record a replayable local-HEAD precondition failure."""
    record = _run_command(HEAD_COMMIT_COMMAND)
    if record["returncode"] == SUCCESS_RETURN_CODE:
        record["returncode"] = FAILURE_RETURN_CODE
    stderr_tail = str(record["stderr_tail"])
    detail = f"{LOCAL_HEAD_PRECONDITION_DETAIL}; observed={head_commit!r}"
    record["stderr_tail"] = _tail(
        "\n".join(part for part in (stderr_tail, detail) if part)
    )
    return {"name": LOCAL_HEAD_PRECONDITION_COMMAND_NAME, **record}


def _target_branch_precondition_report(
    *,
    branch: str,
    head_commit: str | None,
    target_branch_commit: str | None,
) -> dict[str, Any]:
    """Describe whether the push target resolves to the validated HEAD."""
    verified = (
        _is_git_object_id(head_commit)
        and _is_git_object_id(target_branch_commit)
        and target_branch_commit == head_commit
    )
    return {
        "command": list(_target_branch_commit_command(branch)),
        "branch": branch,
        "expected_head_commit": head_commit,
        "target_branch_commit": target_branch_commit,
        "verified": verified,
        "detail": None if verified else TARGET_BRANCH_PRECONDITION_DETAIL,
    }


def _target_branch_precondition_failure_record(
    *,
    branch: str,
    head_commit: str | None,
    target_branch_commit: str | None,
) -> dict[str, Any]:
    """Record a replayable target-branch precondition failure."""
    record = _run_command(_target_branch_commit_command(branch))
    if record["returncode"] == SUCCESS_RETURN_CODE:
        record["returncode"] = FAILURE_RETURN_CODE
    stderr_tail = str(record["stderr_tail"])
    detail = (
        f"{TARGET_BRANCH_PRECONDITION_DETAIL}; "
        f"expected_head={head_commit!r}; observed_target={target_branch_commit!r}"
    )
    record["stderr_tail"] = _tail(
        "\n".join(part for part in (stderr_tail, detail) if part)
    )
    return {"name": TARGET_BRANCH_PRECONDITION_COMMAND_NAME, **record}


def _validation_report_summary(
    validation_report_path: Path,
    *,
    expected_branch: str | None,
    expected_commit: str | None,
) -> dict[str, Any]:
    """Validate and summarize the lightweight validation report JSON."""
    summary: dict[str, Any] = {
        "path": str(validation_report_path),
        "valid": False,
        "schema_version": None,
        GENERATED_REPORT_COMMENT_KEY: None,
        "status": None,
        "expected_branch": None,
        "git_commit": None,
        "command_count": None,
        "command_failure_count": None,
        "command_missing_field_count": None,
        "command_invalid_field_count": None,
        "detail": None,
    }
    try:
        payload = json.loads(validation_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary["detail"] = f"validation report is not readable JSON: {exc}"
        return summary
    if not isinstance(payload, dict):
        summary["detail"] = "validation report root must be a JSON object"
        return summary
    schema_version = payload.get("validation_report_schema_version")
    report_comment = payload.get(GENERATED_REPORT_COMMENT_KEY)
    status = payload.get("status")
    recorded_expected_branch = payload.get("expected_branch")
    git_commit = payload.get("git_commit")
    commands = payload.get("commands")
    command_count = len(commands) if isinstance(commands, list) else None
    command_failure_count = (
        sum(
            1
            for command_record in commands
            if not isinstance(command_record, dict)
            or command_record.get(VALIDATION_REPORT_COMMAND_RETURNCODE_FIELD)
            != SUCCESS_RETURN_CODE
        )
        if isinstance(commands, list)
        else None
    )
    command_missing_field_count = (
        sum(
            1
            for command_record in commands
            if not isinstance(command_record, dict)
            or any(
                required_field not in command_record
                for required_field in VALIDATION_REPORT_COMMAND_REQUIRED_FIELDS
            )
        )
        if isinstance(commands, list)
        else None
    )
    command_invalid_field_count = (
        sum(
            1
            for command_record in commands
            if not _validation_command_record_has_valid_shape(command_record)
        )
        if isinstance(commands, list)
        else None
    )
    summary.update(
        {
            "schema_version": schema_version,
            GENERATED_REPORT_COMMENT_KEY: report_comment,
            "status": status,
            "expected_branch": recorded_expected_branch,
            "git_commit": git_commit,
            "command_count": command_count,
            "command_failure_count": command_failure_count,
            "command_missing_field_count": command_missing_field_count,
            "command_invalid_field_count": command_invalid_field_count,
        }
    )
    if schema_version != EXPECTED_VALIDATION_REPORT_SCHEMA_VERSION:
        summary["detail"] = "validation report schema_version does not match expected version"
        return summary
    if report_comment != EXPECTED_VALIDATION_REPORT_COMMENT:
        summary["detail"] = "validation report report_comment does not describe sync validation evidence"
        return summary
    if status != VALIDATION_REPORT_PASSED_STATUS:
        summary["detail"] = "validation report status is not passed"
        return summary
    if expected_branch is not None and recorded_expected_branch != expected_branch:
        summary["detail"] = "validation report expected_branch does not match push branch"
        return summary
    if not _is_git_object_id(git_commit):
        summary["detail"] = "validation report git_commit is not a full Git object id"
        return summary
    if _is_git_object_id(expected_commit) and git_commit != expected_commit:
        summary["detail"] = "validation report git_commit does not match current HEAD"
        return summary
    if command_count is None:
        summary["detail"] = "validation report commands must be a JSON array"
        return summary
    if command_count < 1:
        summary["detail"] = "validation report commands must not be empty"
        return summary
    if command_missing_field_count != 0:
        summary["detail"] = "validation report commands are missing required fields"
        return summary
    if command_invalid_field_count != 0:
        summary["detail"] = "validation report commands have invalid field values"
        return summary
    if command_failure_count != 0:
        summary["detail"] = "validation report commands include nonzero returncodes"
        return summary
    summary["valid"] = True
    return summary


def _sync_git_provenance(remote: str, branch: str | None) -> dict[str, str | None]:
    """Collect local and remote refs that define one sync attempt."""
    remote_tracking_ref = f"refs/remotes/{remote}/{branch}" if branch else None
    return {
        "current_branch": _current_branch(),
        "head_commit": _metadata_command(HEAD_COMMIT_COMMAND),
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


def _parse_remote_head_commit(ls_remote_stdout: str, branch: str) -> str | None:
    """Extract one branch commit from a git ls-remote --heads response."""
    expected_ref = f"refs/heads/{branch}"
    for raw_line in ls_remote_stdout.splitlines():
        fields = raw_line.split()
        if len(fields) != 2:
            continue
        commit, ref_name = fields
        if ref_name == expected_ref:
            return commit
    return None


def _verify_remote_ref(
    *,
    remote: str,
    branch: str,
    expected_commit: str | None,
    remote_ref_verify_command: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that the remote branch points at the commit that was pushed."""
    record = _run_command(
        remote_ref_verify_command or _remote_ref_verify_command(remote, branch),
        env_overrides=PUSH_AUTH_ENVIRONMENT,
    )
    named_record = {"name": REMOTE_REF_VERIFY_COMMAND_NAME, **record}
    observed_commit = (
        _parse_remote_head_commit(str(record["stdout_tail"]), branch)
        if record["returncode"] == SUCCESS_RETURN_CODE
        else None
    )
    expected_commit_valid = _is_git_object_id(expected_commit)
    observed_commit_valid = _is_git_object_id(observed_commit)
    verified = (
        record["returncode"] == SUCCESS_RETURN_CODE
        and expected_commit_valid
        and observed_commit_valid
        and observed_commit == expected_commit
    )
    report = {
        "remote_ref": f"refs/heads/{branch}",
        "expected_commit": expected_commit,
        "observed_commit": observed_commit,
        "returncode": record["returncode"],
        "verified": verified,
    }
    if record["returncode"] == SUCCESS_RETURN_CODE and not expected_commit_valid:
        report["detail"] = "expected commit is not a full Git object id"
    elif record["returncode"] == SUCCESS_RETURN_CODE and observed_commit is None:
        report["detail"] = "remote branch was not present in git ls-remote output"
    elif record["returncode"] == SUCCESS_RETURN_CODE and not observed_commit_valid:
        report["detail"] = "remote branch commit is not a full Git object id"
    elif record["returncode"] == SUCCESS_RETURN_CODE and not verified:
        report["detail"] = "remote branch commit does not match the pushed branch"
    return named_record, report


def _write_push_failure_bundle(
    *,
    bundle_path: Path,
    branch: str | None,
    bundle_command: tuple[str, ...] | None = None,
    bundle_verify_command: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a portable git bundle for a branch that could not be pushed."""
    if branch is None:
        skipped_record = _branch_precondition_failure_record(
            PUSH_FAILURE_BUNDLE_COMMAND_NAME,
            "could not determine branch for git bundle",
        )
        skipped_report = {
            "path": str(bundle_path),
            "status": PUSH_FAILURE_BUNDLE_STATUS_SKIPPED,
            "returncode": FAILURE_RETURN_CODE,
        }
        return [skipped_record], skipped_report
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    record = _run_command(bundle_command or _bundle_command(bundle_path, branch))
    named_record = {"name": PUSH_FAILURE_BUNDLE_COMMAND_NAME, **record}
    fingerprint = (
        _bundle_file_fingerprint(bundle_path)
        if record["returncode"] == SUCCESS_RETURN_CODE and bundle_path.exists()
        else None
    )
    verify_record: dict[str, Any] | None = None
    if fingerprint:
        verify_record = {
            "name": PUSH_FAILURE_BUNDLE_VERIFY_COMMAND_NAME,
            **_run_command(bundle_verify_command or _bundle_verify_command(bundle_path)),
        }
    verify_returncode = (
        verify_record["returncode"] if verify_record is not None else FAILURE_RETURN_CODE
    )
    bundle_status = (
        PUSH_FAILURE_BUNDLE_STATUS_CREATED
        if fingerprint and verify_returncode == SUCCESS_RETURN_CODE
        else PUSH_FAILURE_BUNDLE_STATUS_FAILED
    )
    report = {
        "path": str(bundle_path),
        "status": bundle_status,
        "returncode": record["returncode"],
        "verify_returncode": verify_returncode,
        "fingerprint": fingerprint,
    }
    if record["returncode"] == SUCCESS_RETURN_CODE and fingerprint is None:
        report["detail"] = "bundle command succeeded but output file is missing"
    command_records = [named_record]
    if verify_record is not None:
        command_records.append(verify_record)
    return command_records, report


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
    push_failure_bundle_path: Path | None = None,
    require_clean_worktree: bool = False,
    validation_command: tuple[str, ...] | None = None,
    push_command: tuple[str, ...] | None = None,
    remote_ref_verify_command: tuple[str, ...] | None = None,
    bundle_command: tuple[str, ...] | None = None,
    bundle_verify_command: tuple[str, ...] | None = None,
) -> int:
    """Run validation, then optionally push, and write one sync evidence report."""
    resolved_branch = branch or _current_branch()
    command_records: list[dict[str, Any]] = []
    worktree_report = _worktree_status()
    validation_report_fingerprint: dict[str, Any] | None = None
    validation_report_summary: dict[str, Any] | None = None
    push_record: dict[str, Any] | None = None
    remote_ref_verification_report: dict[str, Any] | None = None
    target_branch_precondition_report: dict[str, Any] | None = None
    expected_head_commit = _metadata_command(HEAD_COMMIT_COMMAND)
    local_head_precondition_report = _local_head_precondition_report(
        expected_head_commit
    )
    status = (
        STATUS_DIRTY_WORKTREE
        if require_clean_worktree and not worktree_report["clean"]
        else STATUS_VALIDATION_FAILED
    )
    if status != STATUS_DIRTY_WORKTREE:
        validation_record = _run_command(
            validation_command or _validation_command(validation_report_path, resolved_branch)
        )
        command_records.append({"name": "validation", **validation_record})
        validation_report_fingerprint = (
            _file_fingerprint(validation_report_path)
            if validation_report_path.exists()
            else None
        )
        if validation_report_fingerprint is not None:
            validation_report_summary = _validation_report_summary(
                validation_report_path,
                expected_branch=resolved_branch,
                expected_commit=expected_head_commit,
            )
    else:
        validation_record = None

    if validation_record is not None and validation_record["returncode"] == SUCCESS_RETURN_CODE:
        if validation_report_fingerprint is None:
            status = STATUS_VALIDATION_REPORT_MISSING
        elif validation_report_summary is None or not validation_report_summary["valid"]:
            status = STATUS_VALIDATION_REPORT_INVALID
        elif not local_head_precondition_report["verified"]:
            status = STATUS_GIT_HEAD_UNAVAILABLE
            command_records.append(
                _local_head_precondition_failure_record(expected_head_commit)
            )
        elif resolved_branch:
            target_branch_commit = _metadata_command(
                _target_branch_commit_command(resolved_branch)
            )
            target_branch_precondition_report = _target_branch_precondition_report(
                branch=resolved_branch,
                head_commit=expected_head_commit,
                target_branch_commit=target_branch_commit,
            )
            if not target_branch_precondition_report["verified"]:
                status = STATUS_TARGET_BRANCH_MISMATCH
                command_records.append(
                    _target_branch_precondition_failure_record(
                        branch=resolved_branch,
                        head_commit=expected_head_commit,
                        target_branch_commit=target_branch_commit,
                    )
                )
            elif skip_push:
                status = STATUS_VALIDATED
            else:
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
                if status == STATUS_SYNCED:
                    expected_commit = target_branch_commit
                    (
                        remote_ref_record,
                        remote_ref_verification_report,
                    ) = _verify_remote_ref(
                        remote=remote,
                        branch=resolved_branch,
                        expected_commit=expected_commit,
                        remote_ref_verify_command=remote_ref_verify_command,
                    )
                    command_records.append(remote_ref_record)
                    if not remote_ref_verification_report["verified"]:
                        status = STATUS_REMOTE_VERIFICATION_FAILED
        elif skip_push:
            status = STATUS_VALIDATED
        else:
            status = STATUS_PUSH_FAILED
            push_record = _branch_precondition_failure_record(
                PUSH_BRANCH_PRECONDITION_COMMAND_NAME,
                "could not determine current branch",
            )
            command_records.append(push_record)

    push_failure_bundle_report: dict[str, Any] | None = None
    if status == STATUS_PUSH_FAILED and push_failure_bundle_path is not None:
        bundle_records, push_failure_bundle_report = _write_push_failure_bundle(
            bundle_path=push_failure_bundle_path,
            branch=resolved_branch,
            bundle_command=bundle_command,
            bundle_verify_command=bundle_verify_command,
        )
        command_records.extend(bundle_records)

    payload = {
        "sync_report_schema_version": SYNC_REPORT_SCHEMA_VERSION,
        GENERATED_REPORT_COMMENT_KEY: SYNC_REPORT_COMMENT,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "remote": remote,
        "branch": resolved_branch,
        "validation_report_path": str(validation_report_path),
        VALIDATION_REPORT_FINGERPRINT_KEY: validation_report_fingerprint,
        VALIDATION_REPORT_SUMMARY_KEY: validation_report_summary,
        LOCAL_HEAD_PRECONDITION_KEY: local_head_precondition_report,
        TARGET_BRANCH_PRECONDITION_KEY: target_branch_precondition_report,
        WORKTREE_STATUS_KEY: worktree_report,
        "git_commit": expected_head_commit,
        "git_status_short": _metadata_command(("git", "status", "--short")),
        "git_provenance": _sync_git_provenance(remote, resolved_branch),
        "commands": command_records,
        SYNC_COMMAND_SUMMARY_KEY: _sync_command_summary(command_records),
        REMOTE_REF_VERIFICATION_KEY: remote_ref_verification_report,
        "push_failure": _classify_push_failure(push_record),
        PUSH_FAILURE_BUNDLE_KEY: push_failure_bundle_report,
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
    parser.add_argument(
        "--require-clean-worktree",
        action="store_true",
        help="Fail before validation and push if git status --short is not clean",
    )
    parser.add_argument(
        "--push-failure-bundle",
        type=Path,
        help="Create a git bundle for the validated branch when git push fails",
    )
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
        push_failure_bundle_path=args.push_failure_bundle,
        require_clean_worktree=args.require_clean_worktree,
    )


if __name__ == "__main__":
    raise SystemExit(main())
