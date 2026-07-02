"""Unit tests for the IsoDelta-Halo benchmark report checker.

The checker must validate published benchmark evidence without requiring a
LAMMPS binary, so these tests use compact in-memory reports.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_report_check",
    REPORT_CHECK_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
isodelta_report_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_report_check
SPEC.loader.exec_module(isodelta_report_check)


def _valid_report() -> dict[str, object]:
    """Create a small benchmark report with passing correctness evidence."""
    return {
        "summary": {
            "speedup_vs_disabled_cache": 1.15,
            "final_thermo_delta_vs_disabled_cache": {
                "PotEng": {"max_abs_delta": 1.0e-9, "paired_count": 2.0},
                "TotEng": {"max_abs_delta": 2.0e-9, "paired_count": 2.0},
            },
        },
        "results": [
            {
                "case": "baseline-disabled",
                "returncode": 0,
                "cache_summary": {
                    "attempts": 10.0,
                    "hits": 0.0,
                    "hit_rate_percent": 0.0,
                },
            },
            {
                "case": "isodelta-enabled",
                "returncode": 0,
                "cache_summary": {
                    "attempts": 10.0,
                    "hits": 8.0,
                    "hit_rate_percent": 80.0,
                },
            },
            {
                "case": "isodelta-enabled",
                "returncode": 0,
                "cache_summary": {
                    "attempts": 11.0,
                    "hits": 9.0,
                    "hit_rate_percent": 82.0,
                },
            },
        ],
    }


class IsoDeltaBenchmarkReportCheckTest(unittest.TestCase):
    """Check pass/fail gates for report-level correctness claims."""

    def test_validate_report_accepts_passing_evidence(self) -> None:
        """A report with low thermo drift and enough speedup should pass."""
        thresholds = isodelta_report_check.ReportThresholds(
            max_abs_thermo_delta=1.0e-8,
            min_paired_thermo_count=2,
            min_speedup=1.1,
            min_hit_rate_percent=75.0,
            min_enabled_cache_attempts=10,
            min_enabled_cache_hits=8,
        )
        evidence = isodelta_report_check.validate_report(_valid_report(), thresholds)
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["checked_observables"],
            ["PotEng", "TotEng"],
        )
        self.assertEqual(evidence["min_enabled_cache_attempts"], 10.0)
        self.assertEqual(evidence["min_enabled_cache_hits"], 8.0)
        self.assertEqual(evidence["min_enabled_hit_rate_percent"], 80.0)

    def test_validate_report_rejects_large_thermo_delta(self) -> None:
        """Thermo drift above the configured tolerance should fail."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        deltas = summary["final_thermo_delta_vs_disabled_cache"]
        assert isinstance(deltas, dict)
        poteng = deltas["PotEng"]
        assert isinstance(poteng, dict)
        poteng["max_abs_delta"] = 1.0e-4

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "PotEng",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(
                    max_abs_thermo_delta=1.0e-8,
                ),
            )

    def test_validate_report_rejects_weak_speedup(self) -> None:
        """Optional effect gates should reject slow enabled-cache runs."""
        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "speedup_vs_disabled_cache",
        ):
            isodelta_report_check.validate_report(
                _valid_report(),
                isodelta_report_check.ReportThresholds(min_speedup=1.2),
            )

    def test_validate_report_rejects_failed_runs_by_default(self) -> None:
        """A nonzero run returncode should fail unless explicitly allowed."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        first_result = results[0]
        assert isinstance(first_result, dict)
        first_result["returncode"] = 2

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "returncode 2",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_missing_cache_activity(self) -> None:
        """Enabled runs should prove cache attempts and hits, not just speedup."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["hits"] = 0.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "minimum enabled hits",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(
                    min_enabled_cache_attempts=1,
                    min_enabled_cache_hits=1,
                ),
            )

    def test_main_reads_json_report_and_returns_success(self) -> None:
        """The CLI should return zero for a valid benchmark report file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(_valid_report()), encoding="utf-8")
            exit_code = isodelta_report_check.main(
                [
                    "--report",
                    str(report_path),
                    "--max-abs-thermo-delta",
                    "1.0e-8",
                    "--min-speedup",
                    "1.1",
                    "--min-hit-rate-percent",
                    "75.0",
                    "--min-enabled-cache-attempts",
                    "10",
                    "--min-enabled-cache-hits",
                    "8",
                ]
            )
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
