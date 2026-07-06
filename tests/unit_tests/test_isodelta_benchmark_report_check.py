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
DEFAULT_CACHE_ATTEMPTS = 10.0
ZERO_CACHE_COUNT = 0.0
ZERO_HIT_RATE_PERCENT = 0.0
DEFAULT_ENABLED_NO_CACHE_MISSES = 1.0
DEFAULT_ENABLED_NEIGHBOR_REBUILT_MISSES = 1.0
DEFAULT_SUMMARY_RANK_COUNT = 1.0
FRACTIONAL_SUMMARY_RANK_COUNT = 1.5
FRACTIONAL_CACHE_COUNT = 1.5
ONE_CACHE_COUNT = 1.0
WRONG_BASELINE_HIT_RATE_PERCENT = 10.0
WRONG_BASELINE_DISABLED_MISSES = 9.0
RUN_TIMEOUT_SECONDS = 3600.0
BASELINE_MEAN_LOOP_TIME_SECONDS = 11.5
BASELINE_SAMPLE_VARIANCE_LOOP_TIME_SECONDS = 0.5
BASELINE_SAMPLE_STDDEV_LOOP_TIME_SECONDS = (
    BASELINE_SAMPLE_VARIANCE_LOOP_TIME_SECONDS ** 0.5
)
ISODELTA_MEAN_LOOP_TIME_SECONDS = 10.0
ISODELTA_SAMPLE_VARIANCE_LOOP_TIME_SECONDS = 0.0
ISODELTA_SAMPLE_STDDEV_LOOP_TIME_SECONDS = 0.0
EXPECTED_RESULT_COUNT = 4
EXPECTED_REPORT_SCHEMA_VERSION = "isodelta-benchmark-report-v1"
EXPECTED_GIT_COMMIT = "0123456789abcdef"
EXPECTED_GIT_BRANCH = "isodelta-halo-runtime"
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"
ENV_FLAG_DISABLED = "0"


def _cache_summary(
    attempts: float,
    hits: float,
    hit_rate_percent: float,
    miss_disabled: float = ZERO_CACHE_COUNT,
    miss_no_cache: float = DEFAULT_ENABLED_NO_CACHE_MISSES,
    miss_neighbor_list_rebuilt: float = DEFAULT_ENABLED_NEIGHBOR_REBUILT_MISSES,
    summary_rank_count: float = DEFAULT_SUMMARY_RANK_COUNT,
) -> dict[str, float]:
    """Create a cache summary with every IsoDelta-Halo miss counter."""
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": hit_rate_percent,
        "summary_rank_count": summary_rank_count,
        "miss_disabled": miss_disabled,
        "miss_no-cache": miss_no_cache,
        "miss_neighbor-list-rebuilt": miss_neighbor_list_rebuilt,
        "miss_shape-changed": 0.0,
        "miss_index-tensor-shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _disabled_cache_summary(attempts: float) -> dict[str, float]:
    """Create the cache summary shape produced by the disabled baseline."""
    return _cache_summary(
        attempts=attempts,
        hits=ZERO_CACHE_COUNT,
        hit_rate_percent=ZERO_HIT_RATE_PERCENT,
        miss_disabled=attempts,
        miss_no_cache=ZERO_CACHE_COUNT,
        miss_neighbor_list_rebuilt=ZERO_CACHE_COUNT,
    )


def _valid_report() -> dict[str, object]:
    """Create a small benchmark report with passing correctness evidence."""
    return {
        isodelta_report_check.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_report_check.BENCHMARK_REPORT_COMMENT
        ),
        "provenance": {
            "report_schema_version": EXPECTED_REPORT_SCHEMA_VERSION,
            "git_commit": EXPECTED_GIT_COMMIT,
            "git_branch": EXPECTED_GIT_BRANCH,
            "git_dirty": False,
            "python_executable": "python",
            "python_version": "3.13.0",
            "platform": "test-platform",
            "case_environment_overrides": {
                "baseline-disabled": {
                    PRINT_INFO_ENV: ENV_FLAG_ENABLED,
                    DISABLE_CACHE_ENV: ENV_FLAG_ENABLED,
                    PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
                },
                "isodelta-enabled": {
                    PRINT_INFO_ENV: ENV_FLAG_ENABLED,
                    PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
                },
            },
        },
        "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
        "summary": {
            "runs": EXPECTED_RESULT_COUNT,
            "speedup_vs_disabled_cache": 1.15,
            "cases": {
                "baseline-disabled": {
                    "mean_loop_time_seconds": BASELINE_MEAN_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": (
                        BASELINE_SAMPLE_VARIANCE_LOOP_TIME_SECONDS
                    ),
                    "sample_stddev_loop_time_seconds": (
                        BASELINE_SAMPLE_STDDEV_LOOP_TIME_SECONDS
                    ),
                    "min_loop_time_seconds": 11.0,
                    "max_loop_time_seconds": 12.0,
                    "valid_loop_time_count": 2,
                },
                "isodelta-enabled": {
                    "mean_loop_time_seconds": ISODELTA_MEAN_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": (
                        ISODELTA_SAMPLE_VARIANCE_LOOP_TIME_SECONDS
                    ),
                    "sample_stddev_loop_time_seconds": (
                        ISODELTA_SAMPLE_STDDEV_LOOP_TIME_SECONDS
                    ),
                    "min_loop_time_seconds": ISODELTA_MEAN_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": ISODELTA_MEAN_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
            },
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
                "loop_time_seconds": 12.0,
                "cache_summary": _disabled_cache_summary(DEFAULT_CACHE_ATTEMPTS),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 0,
                "returncode": 0,
                "loop_time_seconds": 10.0,
                "cache_summary": _cache_summary(
                    attempts=DEFAULT_CACHE_ATTEMPTS,
                    hits=8.0,
                    hit_rate_percent=80.0,
                ),
            },
            {
                "case": "baseline-disabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": 11.0,
                "cache_summary": _disabled_cache_summary(DEFAULT_CACHE_ATTEMPTS),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": 10.0,
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
            evidence["report_schema_version"],
            EXPECTED_REPORT_SCHEMA_VERSION,
        )
        self.assertEqual(
            evidence[isodelta_report_check.GENERATED_REPORT_COMMENT_KEY],
            isodelta_report_check.BENCHMARK_REPORT_COMMENT,
        )
        self.assertEqual(evidence["git_commit"], EXPECTED_GIT_COMMIT)
        self.assertEqual(evidence["git_branch"], EXPECTED_GIT_BRANCH)
        self.assertFalse(evidence["git_dirty"])
        self.assertEqual(
            evidence["checked_observables"],
            ["PotEng", "TotEng"],
        )
        self.assertEqual(evidence["min_enabled_cache_attempts"], 10.0)
        self.assertEqual(evidence["min_enabled_cache_hits"], 8.0)
        self.assertEqual(evidence["min_enabled_hit_rate_percent"], 80.0)
        self.assertEqual(evidence["max_cache_count_residual"], EXPECTED_ZERO_RESIDUAL)
        self.assertEqual(
            evidence["min_cache_summary_rank_count"],
            DEFAULT_SUMMARY_RANK_COUNT,
        )
        self.assertEqual(
            evidence["verified_cache_miss_key_count"],
            float(len(isodelta_report_check.REQUIRED_CACHE_MISS_KEYS)),
        )
        self.assertEqual(evidence["result_count"], EXPECTED_RESULT_COUNT)
        self.assertEqual(evidence["run_timeout_seconds"], RUN_TIMEOUT_SECONDS)
        self.assertEqual(evidence["paired_repeat_count"], 2)
        self.assertEqual(
            evidence["baseline_mean_loop_time_seconds"],
            BASELINE_MEAN_LOOP_TIME_SECONDS,
        )
        self.assertEqual(
            evidence["isodelta_mean_loop_time_seconds"],
            ISODELTA_MEAN_LOOP_TIME_SECONDS,
        )
        self.assertEqual(evidence["timing_speedup_residual"], 0.0)

    def test_validate_report_rejects_missing_report_comment(self) -> None:
        """Benchmark JSON should explain what source evidence it contains."""
        report = _valid_report()
        del report[isodelta_report_check.GENERATED_REPORT_COMMENT_KEY]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "report_comment",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_wrong_report_comment(self) -> None:
        """The benchmark report comment should not be a generic JSON label."""
        report = _valid_report()
        report[isodelta_report_check.GENERATED_REPORT_COMMENT_KEY] = "generic report"

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "benchmark evidence",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

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

    def test_validate_report_rejects_missing_provenance(self) -> None:
        """Benchmark evidence should include reproducibility metadata."""
        report = _valid_report()
        del report["provenance"]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "provenance",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_wrong_schema_version(self) -> None:
        """Benchmark evidence should pin the expected report schema version."""
        report = _valid_report()
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        provenance["report_schema_version"] = "old-schema"

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "report_schema_version",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_wrong_baseline_env_override(self) -> None:
        """Baseline evidence should prove that IsoDelta-Halo was disabled."""
        report = _valid_report()
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        overrides = provenance["case_environment_overrides"]
        assert isinstance(overrides, dict)
        baseline = overrides["baseline-disabled"]
        assert isinstance(baseline, dict)
        baseline[DISABLE_CACHE_ENV] = ENV_FLAG_DISABLED

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            DISABLE_CACHE_ENV,
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_disabled_env_in_enabled_case(self) -> None:
        """Enabled evidence should prove that the disable flag was absent."""
        report = _valid_report()
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        overrides = provenance["case_environment_overrides"]
        assert isinstance(overrides, dict)
        enabled = overrides["isodelta-enabled"]
        assert isinstance(enabled, dict)
        enabled[DISABLE_CACHE_ENV] = ENV_FLAG_ENABLED

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "must be absent",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_accepts_false_disable_env_in_enabled_case(self) -> None:
        """Enabled evidence may record an explicit false cache-disable flag."""
        report = _valid_report()
        provenance = report["provenance"]
        assert isinstance(provenance, dict)
        overrides = provenance["case_environment_overrides"]
        assert isinstance(overrides, dict)
        enabled = overrides["isodelta-enabled"]
        assert isinstance(enabled, dict)
        enabled[DISABLE_CACHE_ENV] = " off "

        evidence = isodelta_report_check.validate_report(
            report,
            isodelta_report_check.ReportThresholds(),
        )

        self.assertEqual(evidence["status"], "passed")

    def test_validate_report_rejects_wrong_summary_run_count(self) -> None:
        """The summary run count should match result rows."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        summary["runs"] = 3

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "summary.runs",
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
        summary = report["summary"]
        assert isinstance(summary, dict)
        summary["runs"] = len(results)

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
        summary = report["summary"]
        assert isinstance(summary, dict)
        summary["runs"] = len(results)

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "duplicate",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_inconsistent_case_mean_time(self) -> None:
        """Summary means should be recomputed from raw loop times."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        cases = summary["cases"]
        assert isinstance(cases, dict)
        baseline = cases["baseline-disabled"]
        assert isinstance(baseline, dict)
        baseline["mean_loop_time_seconds"] = 99.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "mean_loop_time_seconds",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_inconsistent_case_min_time(self) -> None:
        """Summary min timing should be recomputed from raw loop times."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        cases = summary["cases"]
        assert isinstance(cases, dict)
        baseline = cases["baseline-disabled"]
        assert isinstance(baseline, dict)
        baseline["min_loop_time_seconds"] = 1.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "min_loop_time_seconds",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_inconsistent_sample_variance(self) -> None:
        """Summary timing variance should be recomputed from raw loop times."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        cases = summary["cases"]
        assert isinstance(cases, dict)
        baseline = cases["baseline-disabled"]
        assert isinstance(baseline, dict)
        baseline["sample_variance_loop_time_seconds"] = 9.0

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "sample_variance_loop_time_seconds",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_inconsistent_speedup(self) -> None:
        """Reported speedup should match the case mean loop times."""
        report = _valid_report()
        summary = report["summary"]
        assert isinstance(summary, dict)
        summary["speedup_vs_disabled_cache"] = 1.5

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "speedup_vs_disabled_cache",
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

    def test_validate_report_rejects_baseline_cache_hits(self) -> None:
        """Disabled baseline profiling should not report cache hits."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        baseline_result = results[0]
        assert isinstance(baseline_result, dict)
        cache_summary = baseline_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["hits"] = ONE_CACHE_COUNT
        cache_summary["hit_rate_percent"] = WRONG_BASELINE_HIT_RATE_PERCENT
        cache_summary["miss_disabled"] = WRONG_BASELINE_DISABLED_MISSES

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "baseline-disabled cache_summary.hits",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_wrong_baseline_disabled_misses(self) -> None:
        """Disabled baseline misses should account for every cache attempt."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        baseline_result = results[0]
        assert isinstance(baseline_result, dict)
        cache_summary = baseline_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["miss_disabled"] = WRONG_BASELINE_DISABLED_MISSES
        cache_summary["miss_no-cache"] = ONE_CACHE_COUNT

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "miss_disabled",
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

    def test_validate_report_rejects_fractional_cache_attempts(self) -> None:
        """Cache attempts should be whole profiling counts."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["attempts"] = FRACTIONAL_CACHE_COUNT

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "nonnegative integer",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_zero_cache_attempts(self) -> None:
        """Cache summaries with no lookup attempts cannot prove runtime behavior."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["attempts"] = ZERO_CACHE_COUNT
        cache_summary["hits"] = ZERO_CACHE_COUNT
        cache_summary["hit_rate_percent"] = ZERO_HIT_RATE_PERCENT
        for miss_key in isodelta_report_check.REQUIRED_CACHE_MISS_KEYS:
            cache_summary[miss_key] = ZERO_CACHE_COUNT

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "at least 1",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_missing_summary_rank_count(self) -> None:
        """Cache summaries should prove how many MPI rank summaries were parsed."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        del cache_summary["summary_rank_count"]

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "summary_rank_count",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_fractional_summary_rank_count(self) -> None:
        """MPI summary rank counts should be whole positive counts."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["summary_rank_count"] = FRACTIONAL_SUMMARY_RANK_COUNT

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "positive integer",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
            )

    def test_validate_report_rejects_fractional_miss_counter(self) -> None:
        """Miss reason counters should also be whole profiling counts."""
        report = _valid_report()
        results = report["results"]
        assert isinstance(results, list)
        enabled_result = results[1]
        assert isinstance(enabled_result, dict)
        cache_summary = enabled_result["cache_summary"]
        assert isinstance(cache_summary, dict)
        cache_summary["miss_shape-changed"] = FRACTIONAL_CACHE_COUNT

        with self.assertRaisesRegex(
            isodelta_report_check.ReportCheckError,
            "nonnegative integer",
        ):
            isodelta_report_check.validate_report(
                report,
                isodelta_report_check.ReportThresholds(),
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
            "nonnegative integer",
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
