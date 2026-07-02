"""Unit tests for the IsoDelta-Halo benchmark report checker.

The checker must validate published benchmark evidence without requiring a
LAMMPS binary, so these tests use compact in-memory reports.
"""

from __future__ import annotations

import importlib.util
import json
import math
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


PERCENT_SCALE = 100.0
SECOND_ENABLED_ATTEMPTS = 11.0
SECOND_ENABLED_HITS = 9.0
EXPECTED_ZERO_RESIDUAL = 0.0
RUN_TIMEOUT_SECONDS = 3600.0


def _cache_summary(
    attempts: float,
    hits: float,
    hit_rate_percent: float,
) -> dict[str, float]:
    """Create a cache summary with every IsoDelta-Halo miss counter."""
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": hit_rate_percent,
        "miss_disabled": 0.0,
        "miss_no-cache": 1.0,
        "miss_neighbor-list-rebuilt": 1.0,
        "miss_shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _valid_report() -> dict[str, object]:
    """Create a small benchmark report with passing correctness evidence."""
    return {
        "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
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
                "repeat_index": 0,
                "returncode": 0,
                "cache_summary": _cache_summary(
                    attempts=10.0,
                    hits=0.0,
                    hit_rate_percent=0.0,
                ),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 0,
                "returncode": 0,
                "cache_summary": _cache_summary(
                    attempts=10.0,
                    hits=8.0,
                    hit_rate_percent=80.0,
                ),
            },
            {
                "case": "baseline-disabled",
                "repeat_index": 1,
                "returncode": 0,
                "cache_summary": _cache_summary(
                    attempts=10.0,
                    hits=0.0,
                    hit_rate_percent=0.0,
                ),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 1,
                "returncode": 0,
                "cache_summary": _cache_summary(
                    attempts=SECOND_ENABLED_ATTEMPTS,
                    hits=SECOND_ENABLED_HITS,
                    hit_rate_percent=(
                        PERCENT_SCALE * SECOND_ENABLED_HITS / SECOND_ENABLED_ATTEMPTS
                    ),
                ),
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
        self.assertEqual(evidence["max_cache_count_residual"], EXPECTED_ZERO_RESIDUAL)
        self.assertEqual(evidence["verified_cache_miss_key_count"], 8.0)
        self.assertEqual(evidence["run_timeout_seconds"], RUN_TIMEOUT_SECONDS)
        self.assertEqual(evidence["paired_repeat_count"], 2)

    def test_validate_report_rejects_missing_run_timeout(self) -> None:
        """Paper evidence should record the timeout used for each run."""
        report = _valid_report()
        del report["run_timeout_seconds"]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "run_timeout_seconds",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_nonpositive_run_timeout(self) -> None:
        """A nonpositive timeout would not bound benchmark execution."""
        report = _valid_report()
        report["run_timeout_seconds"] = 0.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "run_timeout_seconds",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_unpaired_repeat(self) -> None:
        """Every repeat should include both disabled and enabled cases."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        del results[2]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "missing paired cases",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_duplicate_case_for_repeat(self) -> None:
        """A repeat should not contain two enabled or two disabled entries."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        duplicate = dict(results[1])
        results.append(duplicate)

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "duplicate",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

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

    def test_validate_report_rejects_negative_thermo_delta(self) -> None:
        """Thermo delta magnitudes should never be negative."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        deltas = summary["final_thermo_delta_vs_disabled_cache"]
        assert isinstance(deltas, dict)
        poteng = deltas["PotEng"]
        assert isinstance(poteng, dict)
        poteng["max_abs_delta"] = -1.0e-9

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "max_abs_delta",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_fractional_thermo_pair_count(self) -> None:
        """Paired thermo counts should represent whole baseline/enabled pairs."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        deltas = summary["final_thermo_delta_vs_disabled_cache"]
        assert isinstance(deltas, dict)
        poteng = deltas["PotEng"]
        assert isinstance(poteng, dict)
        poteng["paired_count"] = 1.5

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "paired_count",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
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
        cache_summary["hit_rate_percent"] = 0.0
        cache_summary["miss_no-cache"] = 9.0

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

    def test_validate_report_rejects_inconsistent_cache_hit_rate(self) -> None:
        """Reported hit rate should match the enabled hits and attempts."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["hit_rate_percent"] = 99.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "hits / attempts",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_nonfinite_numeric_values(self) -> None:
        """Benchmark evidence should not accept NaN or infinity values."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        deltas = summary["final_thermo_delta_vs_disabled_cache"]
        assert isinstance(deltas, dict)
        poteng = deltas["PotEng"]
        assert isinstance(poteng, dict)
        poteng["max_abs_delta"] = math.nan

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "must be finite",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_hits_above_attempts(self) -> None:
        """Cache hits cannot exceed cache lookup attempts in a valid report."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["hits"] = 11.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "cannot exceed attempts",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_inconsistent_miss_breakdown(self) -> None:
        """Miss reason counters should sum to attempts minus cache hits."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["miss_shape-changed"] = 3.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "attempts - hits",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_negative_miss_counter(self) -> None:
        """Miss reason counters should be nonnegative profiling counts."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["miss_shape-changed"] = -1.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "must be nonnegative",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_missing_miss_breakdown(self) -> None:
        """Enabled runs should include every miss counter for diagnosis."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        del cache_summary["miss_comm-list-tag-order-changed"]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "miss_comm-list-tag-order-changed",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_thresholds_rejects_impossible_cache_gate(self) -> None:
        """Cache-hit thresholds should not exceed the required attempts."""
        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "cannot exceed",
        ):
            isodelta_report_check.validate_thresholds(
                isodelta_report_check.ReportThresholds(
                    min_enabled_cache_attempts=2,
                    min_enabled_cache_hits=3,
                )
            )

    def test_validate_thresholds_rejects_out_of_range_percent(self) -> None:
        """Percentage thresholds should stay in the physical range."""
        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "between 0 and 100",
        ):
            isodelta_report_check.validate_thresholds(
                isodelta_report_check.ReportThresholds(
                    min_hit_rate_percent=125.0,
                )
            )

    def test_validate_thresholds_rejects_nonpositive_speedup(self) -> None:
        """Speedup thresholds should require a positive multiplier."""
        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "positive",
        ):
            isodelta_report_check.validate_thresholds(
                isodelta_report_check.ReportThresholds(min_speedup=0.0)
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
