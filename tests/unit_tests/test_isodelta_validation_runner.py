"""Unit tests for the IsoDelta-Halo lightweight validation orchestrator.

These tests keep the commit/push safety gate auditable without running the full
suite recursively. They replace the command list with tiny Python snippets and
then verify the generated JSON report shape.
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
FIRST_COMMAND_INDEX = 0
COMMAND_COUNT_AFTER_VALIDATION_FAILURE = 1
INTENTIONAL_VALIDATION_FAILURE_CODE = 7
VALIDATION_FAILURE_RETURN_CODE = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
VALIDATION_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_validation_runner",
    VALIDATION_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
validation_runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_runner
SPEC.loader.exec_module(validation_runner)


class IsoDeltaValidationRunnerTest(unittest.TestCase):
    """Check JSON evidence emitted by the lightweight validation runner."""

    def test_run_validation_writes_success_report(self) -> None:
        """A passing command should produce a passed validation report."""
        original_commands = validation_runner.VALIDATION_COMMANDS
        original_root = validation_runner.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "validation_report.json"
            validation_runner.REPO_ROOT = root
            validation_runner.VALIDATION_COMMANDS = (
                (sys.executable, "-c", "print('validation-ok')"),
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = validation_runner.run_validation(report_path)
            finally:
                validation_runner.VALIDATION_COMMANDS = original_commands
                validation_runner.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, validation_runner.SUCCESS_RETURN_CODE)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["validation_report_schema_version"],
            validation_runner.VALIDATION_REPORT_SCHEMA_VERSION,
        )
        self.assertIn("validation-ok", report["commands"][FIRST_COMMAND_INDEX]["stdout_tail"])

    def test_run_validation_writes_failure_report_and_stops(self) -> None:
        """A failing command should be recorded and stop later commands."""
        original_commands = validation_runner.VALIDATION_COMMANDS
        original_root = validation_runner.REPO_ROOT
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "validation_report.json"
            validation_runner.REPO_ROOT = root
            validation_runner.VALIDATION_COMMANDS = (
                (
                    sys.executable,
                    "-c",
                    (
                        "import sys; print('bad', file=sys.stderr); "
                        f"sys.exit({INTENTIONAL_VALIDATION_FAILURE_CODE})"
                    ),
                ),
                (sys.executable, "-c", "print('should-not-run')"),
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = validation_runner.run_validation(report_path)
            finally:
                validation_runner.VALIDATION_COMMANDS = original_commands
                validation_runner.REPO_ROOT = original_root
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, VALIDATION_FAILURE_RETURN_CODE)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["commands"]), COMMAND_COUNT_AFTER_VALIDATION_FAILURE)
        self.assertEqual(
            report["commands"][FIRST_COMMAND_INDEX]["returncode"],
            INTENTIONAL_VALIDATION_FAILURE_CODE,
        )
        self.assertIn("bad", report["commands"][FIRST_COMMAND_INDEX]["stderr_tail"])


if __name__ == "__main__":
    unittest.main()
