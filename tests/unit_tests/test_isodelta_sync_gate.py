"""Unit tests for the IsoDelta-Halo validation-plus-push sync gate.

The tests use tiny Python commands instead of real git remotes so the sync
state machine can be checked without network credentials.
"""

from __future__ import annotations

import importlib.util
import contextlib
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
COMMAND_COUNT_AFTER_VALIDATION_FAILURE = 1
INTENTIONAL_VALIDATION_FAILURE_CODE = 3
INTENTIONAL_PUSH_FAILURE_CODE = 128
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
SYNC_GATE_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_sync_gate.py"
SPEC = importlib.util.spec_from_file_location("isodelta_sync_gate", SYNC_GATE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_gate
SPEC.loader.exec_module(sync_gate)


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
                        push_command=(sys.executable, "-c", "print('pushed')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.SUCCESS_RETURN_CODE)
        self.assertEqual(report["status"], sync_gate.STATUS_SYNCED)
        self.assertEqual([record["name"] for record in report["commands"]], ["validation", "push"])
        self.assertIn("pushed", report["commands"][PUSH_COMMAND_INDEX]["stdout_tail"])
        self.assertIsNone(report["push_failure"])

    def test_run_sync_records_git_provenance_for_push_target(self) -> None:
        """Sync reports should identify the local and remote refs being synced."""
        original_root = sync_gate.REPO_ROOT
        original_metadata_command = sync_gate._metadata_command
        expected_remote_ref = "refs/remotes/origin/feature"
        fake_metadata = {
            ("git", "branch", "--show-current"): "feature",
            ("git", "rev-parse", "HEAD"): "head-sha",
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
                        validation_command=(sys.executable, "-c", "print('valid')"),
                        push_command=(sys.executable, "-c", "print('pushed')"),
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
                sync_gate._metadata_command = original_metadata_command
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.SUCCESS_RETURN_CODE)
        self.assertEqual(
            report["git_provenance"],
            {
                "current_branch": "feature",
                "head_commit": "head-sha",
                "target_branch_commit": "feature-sha",
                "remote_url": "https://example.invalid/repo.git",
                "remote_tracking_ref": expected_remote_ref,
                "remote_tracking_commit": "remote-feature-sha",
            },
        )

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
                        validation_command=(sys.executable, "-c", "print('valid')"),
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
                        validation_command=(sys.executable, "-c", "print('valid')"),
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
                    )
            finally:
                sync_gate.REPO_ROOT = original_root
            bundle_exists = bundle_path.exists()
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, sync_gate.FAILURE_RETURN_CODE)
        self.assertTrue(bundle_exists)
        self.assertEqual(
            [record["name"] for record in report["commands"]],
            ["validation", "push", sync_gate.PUSH_FAILURE_BUNDLE_COMMAND_NAME],
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["status"],
            sync_gate.PUSH_FAILURE_BUNDLE_STATUS_CREATED,
        )
        self.assertEqual(
            report[sync_gate.PUSH_FAILURE_BUNDLE_KEY]["path"],
            str(bundle_path),
        )


if __name__ == "__main__":
    unittest.main()
