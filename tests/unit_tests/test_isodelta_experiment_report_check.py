"""Unit tests for the IsoDelta-Halo experiment report verifier.

The checker is tested with small temporary reports and log files so the
fingerprint contract can be validated without running LAMMPS.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Load tools by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
CHECK_SCRIPT = TOOLS_DIR / "check_isodelta_experiment_report.py"
SPEC = importlib.util.spec_from_file_location("isodelta_experiment_report_check", CHECK_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_experiment_report_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_experiment_report_check
SPEC.loader.exec_module(isodelta_experiment_report_check)
experiment_driver = isodelta_experiment_report_check.experiment_driver


EXPECTED_CHECK_REPORT_COMMENT = (
    "IsoDelta-Halo experiment report verification evidence recording driver "
    "report schema, comment, command log fingerprints, and command-result checks."
)
EXPECTED_CHECK_SCHEMA_VERSION = "isodelta-experiment-report-check-v1"


def _write_valid_report(root: Path) -> Path:
    """Create a minimal valid experiment report and fingerprinted logs."""
    stdout_path = root / "logs" / "stage.stdout.log"
    stderr_path = root / "logs" / "stage.stderr.log"
    stdout_path.parent.mkdir(parents=True)
    stdout_path.write_text("stage stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    report_path = root / "isodelta_experiment_report.json"
    payload = {
        "report_comment": experiment_driver.EXPERIMENT_REPORT_COMMENT,
        "ok": True,
        "failed_stage": None,
        "provenance": {
            "report_schema_version": experiment_driver.EXPERIMENT_REPORT_SCHEMA_VERSION,
        },
        "config": {},
        "benchmark_report": str(root / "benchmark" / "isodelta_benchmark_report.json"),
        "bundle_evidence_report": None,
        "commands": [
            {
                "name": "stage",
                "argv": ["python", "stage.py"],
                "returncode": 0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_fingerprint": experiment_driver.file_fingerprint(stdout_path),
                "stderr_fingerprint": experiment_driver.file_fingerprint(stderr_path),
            }
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


class IsoDeltaExperimentReportCheckTest(unittest.TestCase):
    """Check experiment report verification without launching external tools."""

    def test_validate_experiment_report_accepts_matching_log_fingerprints(self) -> None:
        """A report with matching command log fingerprints should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_valid_report(Path(tmpdir))
            evidence = isodelta_experiment_report_check.validate_experiment_report(report_path)

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["experiment_report_check_schema_version"],
            EXPECTED_CHECK_SCHEMA_VERSION,
        )
        self.assertEqual(evidence["report_comment"], EXPECTED_CHECK_REPORT_COMMENT)
        self.assertEqual(evidence["checked_command_count"], 1)
        self.assertEqual(evidence["checked_log_fingerprint_count"], 2)

    def test_validate_experiment_report_rejects_missing_report_comment(self) -> None:
        """The verifier should reject copied JSON without its evidence comment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_valid_report(Path(tmpdir))
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            del payload["report_comment"]
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "report_comment"):
                isodelta_experiment_report_check.validate_experiment_report(report_path)

    def test_validate_experiment_report_rejects_changed_stdout_log(self) -> None:
        """A log edit after report creation should invalidate the fingerprint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_valid_report(Path(tmpdir))
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            stdout_path = Path(payload["commands"][0]["stdout_path"])
            stdout_path.write_text("changed stdout\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stdout_fingerprint.sha256"):
                isodelta_experiment_report_check.validate_experiment_report(report_path)


if __name__ == "__main__":
    unittest.main()
