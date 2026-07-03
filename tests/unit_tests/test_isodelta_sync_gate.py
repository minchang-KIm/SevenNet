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
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
SYNC_GATE_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_sync_gate.py"
SPEC = importlib.util.spec_from_file_location("isodelta_sync_gate", SYNC_GATE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sync_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_gate
SPEC.loader.exec_module(sync_gate)


class IsoDeltaSyncGateTest(unittest.TestCase):
    """Check that validation and push outcomes are reported correctly."""

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


if __name__ == "__main__":
    unittest.main()
