"""Unit tests for the IsoDelta-Halo validation-plus-push sync gate.

The tests use tiny Python commands instead of real git remotes so the sync
state machine can be checked without network credentials.
"""

from __future__ import annotations

import importlib.util
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT_PARENT_DEPTH = 2
VALIDATION_COMMAND_INDEX = 0
PUSH_COMMAND_INDEX = 1
REMOTE_REF_VERIFY_COMMAND_INDEX = 2
COMMAND_COUNT_AFTER_VALIDATION_FAILURE = 1
COMMAND_COUNT_AFTER_DIRTY_WORKTREE = 0
INTENTIONAL_VALIDATION_FAILURE_CODE = 3
INTENTIONAL_PUSH_FAILURE_CODE = 128
VALIDATION_REPORT_COMMAND_COUNT = 1
VALIDATION_REPORT_ELAPSED_SECONDS = 0.0
MISSING_VALIDATION_COMMAND_FIELD_COUNT = 1
INVALID_VALIDATION_COMMAND_FIELD_COUNT = 1
INVALID_VALIDATION_REPORT_ELAPSED_SECONDS = -1.0
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
SYNC_GATE_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_sync_gate.py"
SPEC = importlib.util.spec_from_file_location("isodelta_sync_gate", SYNC_GATE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_gate
SPEC.loader.exec_module(sync_gate)


def _validation_report_command(
    validation_report_path: Path,
    *,
    expected_branch: str = "feature",
    git_commit: str = "feature-sha",
    status: str | None = None,
    command_returncode: int = sync_gate.SUCCESS_RETURN_CODE,
    include_command_required_fields: bool = True,
    command_record_updates: dict[str, object] | None = None,
) -> tuple[str, ...]:
    """Return a tiny command that writes the validation report under test."""
    report_status = status or sync_gate.VALIDATION_REPORT_PASSED_STATUS
    command_record: dict[str, object] = {
        "command": ["test"],
        "returncode": command_returncode,
    }
    if include_command_required_fields:
        command_record.update(
            {
                "elapsed_seconds": VALIDATION_REPORT_ELAPSED_SECONDS,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
    if command_record_updates is not None:
        command_record.update(command_record_updates)
    report_payload = {
        "validation_report_schema_version": (
            sync_gate.EXPECTED_VALIDATION_REPORT_SCHEMA_VERSION
        ),
        "status": report_status,
        "expected_branch": expected_branch,
        "git_commit": git_commit,
        "commands": [command_record],
    }
    report_text = json.dumps(report_payload, indent=2)
    script = (
        "from pathlib import Path; "
        f"Path({str(validation_report_path)!r}).write_text("
        f"{report_text!r}, encoding='utf-8'); "
        "print('valid')"
    )
    return (sys.executable, "-c", script)


def _validation_report_fingerprint(validation_report_path: Path) -> dict[str, object]:
    """Return the expected fingerprint for the test validation report."""
    report_bytes = validation_report_path.read_bytes()
    return {
        "path": str(validation_report_path),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
        "size_bytes": len(report_bytes),
    }


def _passed_validation_report_summary(
    validation_report_path: Path,
    *,
    git_commit: str = "feature-sha",
) -> dict[str, object]:
    """Return the expected sync-gate summary for a passing test report."""
    return {
        "path": str(validation_report_path),
        "valid": True,
        "schema_version": sync_gate.EXPECTED_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": sync_gate.VALIDATION_REPORT_PASSED_STATUS,
        "expected_branch": "feature",
        "git_commit": git_commit,
        "command_count": VALIDATION_REPORT_COMMAND_COUNT,
        "command_failure_count": 0,
        "command_missing_field_count": 0,
        "command_invalid_field_count": 0,
        "detail": None,
    }


class IsoDeltaSyncGateTest(unittest.TestCase):
    """Check that validation and push outcomes are reported correctly."""

    def test_validation_command_enforces_target_branch(self) -> None:
        """The sync gate should validate the same branch it intends to push."""
        command = sync_gate._validation_command(
            Path("validation.json"),
            "codex/isodelta-halo-runtime",
        )

        self.assertEqual(
            command[-2:],
            ("--expected-branch", "codex/isodelta-halo-runtime"),
        )

    def test_run_sync_records_successful_validation_and_push(self) -> None:
        """A passing validation and push command should mark the sync as done."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "feature-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path
                        ),
                        push_command=(sys.executable, "-c", "print('pushed')"),
                        remote_ref_verify_command=(
                            sys.executable,
                            "-c",
                            "print('feature-sha\\trefs/heads/feature')",
                        ),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_validation_fingerprint = _validation_report_fingerprint(
                validation_report_path
            )
            expected_validation_summary = _passed_validation_report_summary(
                validation_report_path
            )

        self.assertEqual(exit_code, sync_gate.SUCCESS_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_SYNCED)
        self.assertEqual(
            [record["name"] for record in report["commands"]],
            ["validation", "push", sync_gate.REMOTE_REF_VERIFY_COMMAND_NAME],
        )
        self.assertIn("pushed", report["commands"][PUSH_COMMAND_INDEX]["stdout_tail"])
        self.assertIn(
            "refs/heads/feature",
            report["commands"][REMOTE_REF_VERIFY_COMMAND_INDEX]["stdout_tail"],
        )
        self.assertEqual(
            report[sync_gate.REMOTE_REF_VERIFICATION_KEY],
            {
                "remote_ref": "refs/heads/feature",
                "expected_commit": "feature-sha",
                "observed_commit": "feature-sha",
                "returncode": sync_gate.SUCCESS_RETURN_CODE,
                "verified": True,
            },
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_FINGERPRINT_KEY],
            expected_validation_fingerprint,
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY],
            expected_validation_summary,
        )
        self.assertEqual(
            report[sync_gate.WORKTREE_STATUS_KEY],
            {
                "available": True,
                "clean": True,
                "entry_count": 0,
                "entries": [],
                "raw": "",
            },
        )
        self.assertIsNone(report["push_failure"])

    def test_run_sync_records_git_provenance_for_push_target(self) -> None:
        """Sync reports should identify the local and remote refs being synced."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        expected_remote_ref = "refs/remotes/origin/feature"
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", expected_remote_ref): "remote-feature-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path
                        ),
                        push_command=(sys.executable, "-c", "print('pushed')"),
                        remote_ref_verify_command=(
                            sys.executable,
                            "-c",
                            "print('feature-sha\\trefs/heads/feature')",
                        ),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_validation_fingerprint = _validation_report_fingerprint(
                validation_report_path
            )
            expected_validation_summary = _passed_validation_report_summary(
                validation_report_path
            )

        self.assertEqual(exit_code, sync_gate.SUCCESS_RETURN_CODE)
        self.assertEqual(
            report["git_provenance"],
            {
                "current_branch": "feature",
                "head_commit": "feature-sha",
                "target_branch_commit": "feature-sha",
                "remote_url": "https://example.invalid/repo.git",
                "remote_tracking_ref": expected_remote_ref,
                "remote_tracking_commit": "remote-feature-sha",
            },
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_FINGERPRINT_KEY],
            expected_validation_fingerprint,
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY],
            expected_validation_summary,
        )

    def test_run_sync_can_require_clean_worktree(self) -> None:
        """A final-paper sync can refuse dirty source trees before pushing."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        status_short = " M tools/run.py\n?? scratch.txt"
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): status_short,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            forbidden_marker = root / "validation-ran"
            forbidden_validation_script = (
                "from pathlib import Path; "
                f"Path({str(forbidden_marker)!r}).write_text('ran', encoding='utf-8')"
            )
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        require_clean_worktree=True,
                        validation_command=(sys.executable, "-c", forbidden_validation_script),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertFalse(forbidden_marker.exists())
        self.assertEqual(report["status"], sync_gate.STATUS_DIRTY_WORKTREE)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_DIRTY_WORKTREE)
        self.assertEqual(
            report[sync_gate.WORKTREE_STATUS_KEY],
            {
                "available": True,
                "clean": False,
                "entry_count": 2,
                "entries": [
                    {
                        "index_status": " ",
                        "worktree_status": "M",
                        "path": "tools/run.py",
                    },
                    {
                        "index_status": "?",
                        "worktree_status": "?",
                        "path": "scratch.txt",
                    },
                ],
                "raw": status_short,
            },
        )
        self.assertIsNone(report[sync_gate.VALIDATION_REPORT_FINGERPRINT_KEY])

    def test_run_sync_fails_when_remote_ref_does_not_match(self) -> None:
        """A push is not synced until the remote branch reports the same commit."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path
                        ),
                        push_command=(sys.executable, "-c", "print('pushed')"),
                        remote_ref_verify_command=(
                            sys.executable,
                            "-c",
                            "print('other-sha\\trefs/heads/feature')",
                        ),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_validation_fingerprint = _validation_report_fingerprint(
                validation_report_path
            )

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_REMOTE_VERIFICATION_FAILED)
        self.assertEqual(
            report[sync_gate.REMOTE_REF_VERIFICATION_KEY],
            {
                "remote_ref": "refs/heads/feature",
                "expected_commit": "feature-sha",
                "observed_commit": "other-sha",
                "returncode": sync_gate.SUCCESS_RETURN_CODE,
                "verified": False,
                "detail": "remote branch commit does not match the pushed branch",
            },
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_FINGERPRINT_KEY],
            expected_validation_fingerprint,
        )
        self.assertIsNone(report["push_failure"])

    def test_run_sync_rejects_failed_validation_report_json(self) -> None:
        """A zero exit code cannot override a failed validation report payload."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path,
                            status="failed",
                        ),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_REPORT_INVALID)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY]["detail"],
            "validation report status is not passed",
        )

    def test_run_sync_rejects_failed_validation_report_command(self) -> None:
        """A passed validation report must not hide failed command records."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.get
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path,
                            command_returncode=sync_gate.FAILURE_RETURN_CODE,
                        ),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_REPORT_INVALID)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY]["command_failure_count"],
            1,
        )
        self.assertEqual(
            report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY]["detail"],
            "validation report commands include nonzero returncodes",
        )

    def test_run_sync_rejects_incomplete_validation_command_record(self) -> None:
        """A passed validation report must include replayable command records."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.__getitem__
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path,
                            include_command_required_fields=False,
                        ),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validation_summary = report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY]

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_REPORT_INVALID)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            validation_summary["command_missing_field_count"],
            MISSING_VALIDATION_COMMAND_FIELD_COUNT,
        )
        self.assertEqual(
            validation_summary["detail"],
            "validation report commands are missing required fields",
        )

    def test_run_sync_rejects_invalid_validation_command_record_values(self) -> None:
        """A passed validation report must use typed command record fields."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "feature-sha",
            ("git", "rev-parse", "feature"): "feature-sha",
            ("git", "remote", "get-url", "origin"): "https://example.invalid/repo.git",
            ("git", "rev-parse", "--verify", "refs/remotes/origin/feature"): "old-sha",
            ("git", "status", "--short"): "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            sync_gate._metadata_command = fake_metadata.__getitem__
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path,
                            command_record_updates={
                                sync_gate.VALIDATION_REPORT_COMMAND_ELAPSED_SECONDS_FIELD: (
                                    INVALID_VALIDATION_REPORT_ELAPSED_SECONDS
                                )
                            },
                        ),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validation_summary = report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY]

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_REPORT_INVALID)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            validation_summary["command_invalid_field_count"],
            INVALID_VALIDATION_COMMAND_FIELD_COUNT,
        )
        self.assertEqual(
            validation_summary["detail"],
            "validation report commands have invalid field values",
        )

    def test_run_sync_rejects_missing_validation_report(self) -> None:
        """A zero exit code is not enough evidence without the JSON report."""
        original_root = sync_gate.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=(sys.executable, "-c", "print('valid')"),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_REPORT_MISSING)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertIsNone(report[sync_gate.VALIDATION_REPORT_FINGERPRINT_KEY])
        self.assertIsNone(report[sync_gate.VALIDATION_REPORT_SUMMARY_KEY])

    def test_run_sync_skips_push_after_validation_failure(self) -> None:
        """A failing validation command should prevent the push command."""
        original_root = sync_gate.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=(
                            sys.executable,
                            "-c",
                            (
                                "import sys; print('invalid', file=sys.stderr); "
                                f"sys.exit({INTENTIONAL_VALIDATION_FAILURE_CODE})"
                            ),
                        ),
                        push_command=(sys.executable, "-c", "print('should-not-push')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_VALIDATION_FAILED)
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            report["commands"][VALIDATION_COMMAND_INDEX]["returncode"],
            INTENTIONAL_VALIDATION_FAILURE_CODE,
        )
        self.assertIn("invalid", report["commands"][VALIDATION_COMMAND_INDEX]["stderr_tail"])

    def test_run_sync_classifies_noninteractive_auth_push_failure(self) -> None:
        """Credential prompts disabled by the sync gate should be explicit."""
        original_root = sync_gate.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            sync_gate.REPO_ROOT = root
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        validation_command=_validation_report_command(
                            validation_report_path
                        ),
                        push_command=(
                            sys.executable,
                            "-c",
                            (
                                "import sys; "
                                "print('fatal: Cannot prompt because user "
                                "interactivity has been disabled.', file=sys.stderr); "
                                "print(\"fatal: could not read Username for "
                                "'https://github.com': terminal prompts disabled\", "
                                "file=sys.stderr); "
                                f"sys.exit({INTENTIONAL_PUSH_FAILURE_CODE})"
                            ),
                        ),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_PUSH_FAILED)
        self.assertEqual(
            report["push_failure"]["reason"],
            sync_gate.PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED,
        )
        self.assertIn(
            "Authenticate the Git HTTPS remote",
            report["push_failure"]["suggested_action"],
        )

    def test_run_sync_can_write_bundle_after_push_failure(self) -> None:
        """A failed push can still produce a portable bundle for handoff."""
        original_root = sync_gate.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "sync_report.json"
            validation_report_path = root / "validation_report.json"
            bundle_path = root / "failed_push.bundle"
            bundle_script = (
                "from pathlib import Path; "
                f"Path({str(bundle_path)!r}).write_text('bundle', encoding='utf-8')"
            )
            sync_gate.REPO_ROOT = root
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = sync_gate.run_sync(
                        remote="origin",
                        branch="feature",
                        report_path=report_path,
                        validation_report_path=validation_report_path,
                        push_failure_bundle_path=bundle_path,
                        validation_command=_validation_report_command(
                            validation_report_path
                        ),
                        push_command=(
                            sys.executable,
                            "-c",
                            (
                                "import sys; "
                                "print('fatal: Could not resolve host', file=sys.stderr); "
                                f"sys.exit({INTENTIONAL_PUSH_FAILURE_CODE})"
                            ),
                        ),
                        bundle_command=(sys.executable, "-c", bundle_script),
                        bundle_verify_command=(sys.executable, "-c", "print('verified')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            bundle_exists = bundle_path.exists()
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertTrue(bundle_exists)
        self.assertEqual(
            [record["name"] for record in report["commands"]],
            [
                "validation",
                "push",
                sync_gate.PUSH_FAILURE_BUNDLE_COMMAND_NAME,
                sync_gate.PUSH_FAILURE_BUNDLE_VERIFY_COMMAND_NAME,
            ],
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["status"],
            sync_gate.PUSH_FAILURE_BUNDLE_STATUS_CREATED,
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["path"],
            str(bundle_path),
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["verify_returncode"],
            sync_gate.SUCCESS_RETURN_CODE,
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["fingerprint"],
            {
                "path": str(bundle_path),
                "sha256": hashlib.sha256(b"bundle").hexdigest(),
                "size_bytes": len(b"bundle"),
            },
        )


if __name__ == "__main__":
    unittest.main()
