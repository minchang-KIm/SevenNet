"""Validate an IsoDelta-Halo benchmark report against publishable evidence.

The benchmark runner records timing, cache, and final-thermo fields. This
checker turns that JSON report into a pass/fail gate for correctness and effect
claims before the numbers are used in a paper or a performance table.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


# These keys mirror the benchmark report schema rather than scattering JSON
# field names through the validation logic.
PROVENANCE_KEY = "provenance"
REPORT_SCHEMA_VERSION_KEY = "report_schema_version"
EXPECTED_REPORT_SCHEMA_VERSION = "isodelta-benchmark-report-v1"
GIT_COMMIT_KEY = "git_commit"
GIT_BRANCH_KEY = "git_branch"
GIT_DIRTY_KEY = "git_dirty"
PYTHON_EXECUTABLE_KEY = "python_executable"
PYTHON_VERSION_KEY = "python_version"
PLATFORM_KEY = "platform"
CASE_ENVIRONMENT_OVERRIDES_KEY = "case_environment_overrides"
SUMMARY_KEY = "summary"
SUMMARY_CASES_KEY = "cases"
SUMMARY_RUNS_KEY = "runs"
RESULTS_KEY = "results"
RUN_TIMEOUT_SECONDS_KEY = "run_timeout_seconds"
CASE_KEY = "case"
REPEAT_INDEX_KEY = "repeat_index"
RETURNCODE_KEY = "returncode"
LOOP_TIME_SECONDS_KEY = "loop_time_seconds"
CACHE_SUMMARY_KEY = "cache_summary"
ATTEMPTS_KEY = "attempts"
HITS_KEY = "hits"
HIT_RATE_KEY = "hit_rate_percent"
REQUIRED_CACHE_MISS_KEYS = (
    "miss_disabled",
    "miss_no-cache",
    "miss_neighbor-list-rebuilt",
    "miss_shape-changed",
    "miss_tag-count-changed",
    "miss_tag-order-changed",
    "miss_comm-topology-changed",
    "miss_comm-list-tag-order-changed",
)
SPEEDUP_KEY = "speedup_vs_disabled_cache"
MEAN_LOOP_TIME_KEY = "mean_loop_time_seconds"
SAMPLE_VARIANCE_LOOP_TIME_KEY = "sample_variance_loop_time_seconds"
SAMPLE_STDDEV_LOOP_TIME_KEY = "sample_stddev_loop_time_seconds"
MIN_LOOP_TIME_KEY = "min_loop_time_seconds"
MAX_LOOP_TIME_KEY = "max_loop_time_seconds"
VALID_LOOP_TIME_COUNT_KEY = "valid_loop_time_count"
FINAL_THERMO_DELTA_KEY = "final_thermo_delta_vs_disabled_cache"
MAX_ABS_DELTA_KEY = "max_abs_delta"
PAIRED_COUNT_KEY = "paired_count"
BASELINE_CASE = "baseline-disabled"
ISODELTA_CASE = "isodelta-enabled"
EXPECTED_CASES = frozenset((BASELINE_CASE, ISODELTA_CASE))
EXPECTED_CASE_COUNT_PER_REPEAT = len(EXPECTED_CASES)
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"
REQUIRED_CASE_ENVIRONMENT_OVERRIDES = {
    BASELINE_CASE: {
        PRINT_INFO_ENV: ENV_FLAG_ENABLED,
        DISABLE_CACHE_ENV: ENV_FLAG_ENABLED,
        PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
    },
    ISODELTA_CASE: {
        PRINT_INFO_ENV: ENV_FLAG_ENABLED,
        PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
    },
}
DEFAULT_MAX_ABS_THERMO_DELTA = 1.0e-8
DEFAULT_MIN_PAIRED_THERMO_COUNT = 1
DEFAULT_MIN_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS = 1
DEFAULT_MIN_ENABLED_CACHE_HITS = 0
MIN_COUNT_VALUE = 0
MIN_REQUIRED_PAIRED_THERMO_COUNT = 1
MIN_NONNEGATIVE_VALUE = 0.0
MIN_PERCENT_VALUE = 0.0
MAX_PERCENT_VALUE = 100.0
MIN_POSITIVE_SPEEDUP = 0.0
MIN_POSITIVE_TIMEOUT_SECONDS = 0.0
MIN_POSITIVE_LOOP_TIME_SECONDS = 0.0
MIN_SAMPLE_VARIANCE_COUNT = 2
SAMPLE_VARIANCE_DEGREES_OF_FREEDOM = 1
CACHE_HIT_RATE_TOLERANCE_PERCENT = 1.0e-9
CACHE_COUNT_TOLERANCE = 1.0e-9
TIMING_ABSOLUTE_TOLERANCE_SECONDS = 1.0e-12
TIMING_RELATIVE_TOLERANCE = 1.0e-9


class ReportCheckError(ValueError):
    """Raised when a benchmark report lacks required correctness evidence."""


@dataclass(frozen=True)
class ReportThresholds:
    """Store numerical gates used to judge a benchmark report."""

    max_abs_thermo_delta: float = DEFAULT_MAX_ABS_THERMO_DELTA
    min_paired_thermo_count: int = DEFAULT_MIN_PAIRED_THERMO_COUNT
    min_speedup: float | None = None
    min_hit_rate_percent: float = DEFAULT_MIN_HIT_RATE_PERCENT
    min_enabled_cache_attempts: int = DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS
    min_enabled_cache_hits: int = DEFAULT_MIN_ENABLED_CACHE_HITS
    require_successful_runs: bool = True


def load_report(path: Path) -> dict[str, Any]:
    """Read a benchmark report JSON file and return its object payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReportCheckError("benchmark report root must be a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    """Raise a report-check error with a concise message."""
    if not condition:
        raise ReportCheckError(message)


def _validate_percent(value: float, field_name: str) -> None:
    """Require a percentage threshold to stay within the physical range."""
    _require(
        MIN_PERCENT_VALUE <= value <= MAX_PERCENT_VALUE,
        f"{field_name} must be between {MIN_PERCENT_VALUE:g} and {MAX_PERCENT_VALUE:g}",
    )


def validate_thresholds(thresholds: ReportThresholds) -> None:
    """Reject report acceptance criteria that would make evidence meaningless."""
    _require(
        thresholds.max_abs_thermo_delta >= MIN_NONNEGATIVE_VALUE,
        "max_abs_thermo_delta must be nonnegative",
    )
    _require(
        thresholds.min_paired_thermo_count >= MIN_REQUIRED_PAIRED_THERMO_COUNT,
        "min_paired_thermo_count must be at least one",
    )
    _require(
        thresholds.min_enabled_cache_attempts >= MIN_COUNT_VALUE,
        "min_enabled_cache_attempts must be nonnegative",
    )
    _require(
        thresholds.min_enabled_cache_hits >= MIN_COUNT_VALUE,
        "min_enabled_cache_hits must be nonnegative",
    )
    _require(
        thresholds.min_enabled_cache_hits <= thresholds.min_enabled_cache_attempts,
        "min_enabled_cache_hits cannot exceed min_enabled_cache_attempts",
    )
    if thresholds.min_speedup is not None:
        _require(
            thresholds.min_speedup > MIN_POSITIVE_SPEEDUP,
            "min_speedup must be positive when provided",
        )
    _validate_percent(thresholds.min_hit_rate_percent, "min_hit_rate_percent")


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a dictionary field or fail with a schema-oriented message."""
    _require(isinstance(value, dict), f"{field_name} must be a JSON object")
    return value


def _as_sequence(value: Any, field_name: str) -> list[Any]:
    """Return a list field or fail with a schema-oriented message."""
    _require(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _as_nonempty_string(value: Any, field_name: str) -> str:
    """Return a non-empty string field with a schema-oriented message."""
    _require(isinstance(value, str), f"{field_name} must be a string")
    _require(bool(value.strip()), f"{field_name} must not be empty")
    return value


def _as_boolean(value: Any, field_name: str) -> bool:
    """Return a boolean field without accepting integer aliases."""
    _require(isinstance(value, bool), f"{field_name} must be boolean")
    return value


def _as_number(value: Any, field_name: str) -> float:
    """Return a numeric field without accepting booleans as numbers."""
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    numeric_value = float(value)
    _require(math.isfinite(numeric_value), f"{field_name} must be finite")
    return numeric_value


def _require_optional_none(value: Any, field_name: str) -> None:
    """Require an optional report field to be null when a statistic is undefined."""
    _require(value is None, f"{field_name} must be null")


def _is_close(
    observed: float,
    expected: float,
    absolute_tolerance: float = TIMING_ABSOLUTE_TOLERANCE_SECONDS,
    relative_tolerance: float = TIMING_RELATIVE_TOLERANCE,
) -> bool:
    """Return whether two report numbers agree within a named tolerance."""
    tolerance = max(
        absolute_tolerance,
        relative_tolerance * max(abs(observed), abs(expected)),
    )
    return abs(observed - expected) <= tolerance


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return the report summary object."""
    return _as_mapping(report.get(SUMMARY_KEY), SUMMARY_KEY)


def _results(report: dict[str, Any]) -> list[Any]:
    """Return the report result list."""
    return _as_sequence(report.get(RESULTS_KEY), RESULTS_KEY)


def _check_case_environment_overrides(overrides: dict[str, Any]) -> None:
    """Require provenance to prove the disabled and enabled runtime controls."""
    for case_name, expected_env in REQUIRED_CASE_ENVIRONMENT_OVERRIDES.items():
        field_prefix = f"{PROVENANCE_KEY}.{CASE_ENVIRONMENT_OVERRIDES_KEY}.{case_name}"
        case_overrides = _as_mapping(overrides.get(case_name), field_prefix)
        for env_name, expected_value in expected_env.items():
            actual_value = _as_nonempty_string(
                case_overrides.get(env_name),
                f"{field_prefix}.{env_name}",
            )
            _require(
                actual_value == expected_value,
                f"{field_prefix}.{env_name} must be {expected_value!r}",
            )
        if case_name == ISODELTA_CASE:
            _require(
                DISABLE_CACHE_ENV not in case_overrides,
                f"{field_prefix}.{DISABLE_CACHE_ENV} must be absent for enabled case",
            )


def _check_provenance(report: dict[str, Any]) -> dict[str, Any]:
    """Require reproducibility metadata for a publishable benchmark report."""
    provenance = _as_mapping(report.get(PROVENANCE_KEY), PROVENANCE_KEY)
    schema_version = _as_nonempty_string(
        provenance.get(REPORT_SCHEMA_VERSION_KEY),
        f"{PROVENANCE_KEY}.{REPORT_SCHEMA_VERSION_KEY}",
    )
    _require(
        schema_version == EXPECTED_REPORT_SCHEMA_VERSION,
        (
            f"{PROVENANCE_KEY}.{REPORT_SCHEMA_VERSION_KEY} must be "
            f"{EXPECTED_REPORT_SCHEMA_VERSION}"
        ),
    )
    git_commit = _as_nonempty_string(
        provenance.get(GIT_COMMIT_KEY),
        f"{PROVENANCE_KEY}.{GIT_COMMIT_KEY}",
    )
    git_branch = _as_nonempty_string(
        provenance.get(GIT_BRANCH_KEY),
        f"{PROVENANCE_KEY}.{GIT_BRANCH_KEY}",
    )
    git_dirty = _as_boolean(
        provenance.get(GIT_DIRTY_KEY),
        f"{PROVENANCE_KEY}.{GIT_DIRTY_KEY}",
    )
    _as_nonempty_string(
        provenance.get(PYTHON_EXECUTABLE_KEY),
        f"{PROVENANCE_KEY}.{PYTHON_EXECUTABLE_KEY}",
    )
    _as_nonempty_string(
        provenance.get(PYTHON_VERSION_KEY),
        f"{PROVENANCE_KEY}.{PYTHON_VERSION_KEY}",
    )
    _as_nonempty_string(
        provenance.get(PLATFORM_KEY),
        f"{PROVENANCE_KEY}.{PLATFORM_KEY}",
    )
    case_environment_overrides = _as_mapping(
        provenance.get(CASE_ENVIRONMENT_OVERRIDES_KEY),
        f"{PROVENANCE_KEY}.{CASE_ENVIRONMENT_OVERRIDES_KEY}",
    )
    _check_case_environment_overrides(case_environment_overrides)
    return {
        "report_schema_version": schema_version,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }


def _check_run_timeout(report: dict[str, Any]) -> float:
    """Require the report to record a finite per-run timeout."""
    run_timeout_seconds = _as_number(
        report.get(RUN_TIMEOUT_SECONDS_KEY),
        RUN_TIMEOUT_SECONDS_KEY,
    )
    _require(
        run_timeout_seconds > MIN_POSITIVE_TIMEOUT_SECONDS,
        f"{RUN_TIMEOUT_SECONDS_KEY} must be positive",
    )
    return run_timeout_seconds


def _check_result_count(report: dict[str, Any], summary: dict[str, Any]) -> int:
    """Require summary run count to match the result rows."""
    results = _results(report)
    reported_runs = _as_number(
        summary.get(SUMMARY_RUNS_KEY),
        f"{SUMMARY_KEY}.{SUMMARY_RUNS_KEY}",
    )
    _require(
        reported_runs >= MIN_REQUIRED_PAIRED_THERMO_COUNT and reported_runs.is_integer(),
        f"{SUMMARY_KEY}.{SUMMARY_RUNS_KEY} must be a positive integer",
    )
    _require(
        int(reported_runs) == len(results),
        f"{SUMMARY_KEY}.{SUMMARY_RUNS_KEY} must match results length",
    )
    return len(results)


def _check_paired_runs(report: dict[str, Any]) -> int:
    """Require each repeat to include exactly one baseline and enabled run."""
    cases_by_repeat: dict[int, set[str]] = {}
    for index, result in enumerate(_results(report)):
        result_map = _as_mapping(result, f"{RESULTS_KEY}[{index}]")
        case_name = result_map.get(CASE_KEY)
        _require(
            isinstance(case_name, str),
            f"{RESULTS_KEY}[{index}].{CASE_KEY} must be a string",
        )
        _require(
            case_name in EXPECTED_CASES,
            f"{RESULTS_KEY}[{index}].{CASE_KEY} must be one of {sorted(EXPECTED_CASES)}",
        )
        repeat_index_value = _as_number(
            result_map.get(REPEAT_INDEX_KEY),
            f"{RESULTS_KEY}[{index}].{REPEAT_INDEX_KEY}",
        )
        _require(
            repeat_index_value >= MIN_COUNT_VALUE and repeat_index_value.is_integer(),
            f"{RESULTS_KEY}[{index}].{REPEAT_INDEX_KEY} must be a nonnegative integer",
        )
        repeat_index = int(repeat_index_value)
        seen_cases = cases_by_repeat.setdefault(repeat_index, set())
        _require(
            case_name not in seen_cases,
            f"repeat_index {repeat_index} has duplicate {case_name}",
        )
        seen_cases.add(case_name)

    _require(cases_by_repeat, "benchmark report must contain paired repeat results")
    for repeat_index, seen_cases in sorted(cases_by_repeat.items()):
        missing_cases = EXPECTED_CASES - seen_cases
        _require(
            not missing_cases,
            (
                f"repeat_index {repeat_index} missing paired cases: "
                f"{', '.join(sorted(missing_cases))}"
            ),
        )
        _require(
            len(seen_cases) == EXPECTED_CASE_COUNT_PER_REPEAT,
            f"repeat_index {repeat_index} must contain exactly two paired cases",
        )
    return len(cases_by_repeat)


def _check_successful_runs(report: dict[str, Any]) -> int:
    """Require every recorded benchmark process to exit successfully."""
    successful_count = 0
    for index, result in enumerate(_results(report)):
        result_map = _as_mapping(result, f"{RESULTS_KEY}[{index}]")
        returncode = _as_number(
            result_map.get(RETURNCODE_KEY),
            f"{RESULTS_KEY}[{index}].{RETURNCODE_KEY}",
        )
        if returncode != 0:
            case_name = result_map.get(CASE_KEY, f"index {index}")
            raise ReportCheckError(f"{case_name} failed with returncode {returncode:g}")
        successful_count += 1
    return successful_count


def _check_timing_summary(report: dict[str, Any], summary: dict[str, Any]) -> dict[str, float]:
    """Require summary timing and speedup to match raw run loop times."""
    summary_cases = _as_mapping(
        summary.get(SUMMARY_CASES_KEY),
        f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}",
    )
    loop_times_by_case: dict[str, list[float]] = {
        case_name: [] for case_name in EXPECTED_CASES
    }
    for index, result in enumerate(_results(report)):
        result_map = _as_mapping(result, f"{RESULTS_KEY}[{index}]")
        case_name = result_map.get(CASE_KEY)
        if case_name not in EXPECTED_CASES:
            continue
        loop_time = _as_number(
            result_map.get(LOOP_TIME_SECONDS_KEY),
            f"{RESULTS_KEY}[{index}].{LOOP_TIME_SECONDS_KEY}",
        )
        _require(
            loop_time > MIN_POSITIVE_LOOP_TIME_SECONDS,
            f"{RESULTS_KEY}[{index}].{LOOP_TIME_SECONDS_KEY} must be positive",
        )
        loop_times_by_case[str(case_name)].append(loop_time)

    mean_loop_times: dict[str, float] = {}
    for case_name in sorted(EXPECTED_CASES):
        loop_times = loop_times_by_case[case_name]
        _require(loop_times, f"no loop times recorded for {case_name}")
        expected_mean = sum(loop_times) / len(loop_times)
        expected_min = min(loop_times)
        expected_max = max(loop_times)
        case_summary = _as_mapping(
            summary_cases.get(case_name),
            f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}",
        )
        reported_mean = _as_number(
            case_summary.get(MEAN_LOOP_TIME_KEY),
            f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}.{MEAN_LOOP_TIME_KEY}",
        )
        reported_count = _as_number(
            case_summary.get(VALID_LOOP_TIME_COUNT_KEY),
            f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}.{VALID_LOOP_TIME_COUNT_KEY}",
        )
        reported_min = _as_number(
            case_summary.get(MIN_LOOP_TIME_KEY),
            f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}.{MIN_LOOP_TIME_KEY}",
        )
        reported_max = _as_number(
            case_summary.get(MAX_LOOP_TIME_KEY),
            f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}.{MAX_LOOP_TIME_KEY}",
        )
        _require(
            reported_mean > MIN_POSITIVE_LOOP_TIME_SECONDS,
            f"{case_name} {MEAN_LOOP_TIME_KEY} must be positive",
        )
        _require(
            reported_count == len(loop_times) and reported_count.is_integer(),
            f"{case_name} {VALID_LOOP_TIME_COUNT_KEY} must match raw loop times",
        )
        _require(
            _is_close(reported_mean, expected_mean),
            f"{case_name} {MEAN_LOOP_TIME_KEY} must match raw loop times",
        )
        _require(
            _is_close(reported_min, expected_min),
            f"{case_name} {MIN_LOOP_TIME_KEY} must match raw loop times",
        )
        _require(
            _is_close(reported_max, expected_max),
            f"{case_name} {MAX_LOOP_TIME_KEY} must match raw loop times",
        )
        if len(loop_times) >= MIN_SAMPLE_VARIANCE_COUNT:
            squared_delta_sum = sum(
                (loop_time - expected_mean) ** 2 for loop_time in loop_times
            )
            expected_variance = squared_delta_sum / (
                len(loop_times) - SAMPLE_VARIANCE_DEGREES_OF_FREEDOM
            )
            expected_stddev = math.sqrt(expected_variance)
            reported_variance = _as_number(
                case_summary.get(SAMPLE_VARIANCE_LOOP_TIME_KEY),
                (
                    f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}."
                    f"{SAMPLE_VARIANCE_LOOP_TIME_KEY}"
                ),
            )
            reported_stddev = _as_number(
                case_summary.get(SAMPLE_STDDEV_LOOP_TIME_KEY),
                (
                    f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}."
                    f"{SAMPLE_STDDEV_LOOP_TIME_KEY}"
                ),
            )
            _require(
                reported_variance >= MIN_NONNEGATIVE_VALUE,
                f"{case_name} {SAMPLE_VARIANCE_LOOP_TIME_KEY} must be nonnegative",
            )
            _require(
                reported_stddev >= MIN_NONNEGATIVE_VALUE,
                f"{case_name} {SAMPLE_STDDEV_LOOP_TIME_KEY} must be nonnegative",
            )
            _require(
                _is_close(reported_variance, expected_variance),
                f"{case_name} {SAMPLE_VARIANCE_LOOP_TIME_KEY} must match raw loop times",
            )
            _require(
                _is_close(reported_stddev, expected_stddev),
                f"{case_name} {SAMPLE_STDDEV_LOOP_TIME_KEY} must match raw loop times",
            )
        else:
            _require_optional_none(
                case_summary.get(SAMPLE_VARIANCE_LOOP_TIME_KEY),
                (
                    f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}."
                    f"{SAMPLE_VARIANCE_LOOP_TIME_KEY}"
                ),
            )
            _require_optional_none(
                case_summary.get(SAMPLE_STDDEV_LOOP_TIME_KEY),
                (
                    f"{SUMMARY_KEY}.{SUMMARY_CASES_KEY}.{case_name}."
                    f"{SAMPLE_STDDEV_LOOP_TIME_KEY}"
                ),
            )
        mean_loop_times[case_name] = reported_mean

    expected_speedup = mean_loop_times[BASELINE_CASE] / mean_loop_times[ISODELTA_CASE]
    reported_speedup = _as_number(
        summary.get(SPEEDUP_KEY),
        f"{SUMMARY_KEY}.{SPEEDUP_KEY}",
    )
    _require(
        _is_close(reported_speedup, expected_speedup),
        f"{SPEEDUP_KEY} must match mean loop times",
    )
    return {
        "baseline_mean_loop_time_seconds": mean_loop_times[BASELINE_CASE],
        "isodelta_mean_loop_time_seconds": mean_loop_times[ISODELTA_CASE],
        "timing_speedup_residual": abs(reported_speedup - expected_speedup),
    }


def _check_thermo_deltas(
    summary: dict[str, Any],
    thresholds: ReportThresholds,
    paired_repeat_count: int,
) -> tuple[list[str], float]:
    """Require final thermo deltas to stay within the configured tolerance."""
    delta_report = _as_mapping(
        summary.get(FINAL_THERMO_DELTA_KEY),
        f"{SUMMARY_KEY}.{FINAL_THERMO_DELTA_KEY}",
    )
    _require(
        bool(delta_report),
        f"{SUMMARY_KEY}.{FINAL_THERMO_DELTA_KEY} must not be empty",
    )

    checked_observables: list[str] = []
    max_seen_delta = 0.0
    for observable, metrics in sorted(delta_report.items()):
        metric_map = _as_mapping(
            metrics,
            f"{SUMMARY_KEY}.{FINAL_THERMO_DELTA_KEY}.{observable}",
        )
        max_abs_delta = _as_number(
            metric_map.get(MAX_ABS_DELTA_KEY),
            f"{observable}.{MAX_ABS_DELTA_KEY}",
        )
        paired_count = _as_number(
            metric_map.get(PAIRED_COUNT_KEY),
            f"{observable}.{PAIRED_COUNT_KEY}",
        )
        _require(
            max_abs_delta >= MIN_NONNEGATIVE_VALUE,
            f"{observable} {MAX_ABS_DELTA_KEY} must be nonnegative",
        )
        _require(
            paired_count >= MIN_REQUIRED_PAIRED_THERMO_COUNT
            and paired_count.is_integer(),
            f"{observable} {PAIRED_COUNT_KEY} must be a positive integer",
        )
        _require(
            paired_count >= thresholds.min_paired_thermo_count,
            (
                f"{observable} paired_count {paired_count:g} is below "
                f"{thresholds.min_paired_thermo_count}"
            ),
        )
        _require(
            paired_count <= paired_repeat_count,
            (
                f"{observable} paired_count {paired_count:g} exceeds "
                f"paired_repeat_count {paired_repeat_count:g}"
            ),
        )
        _require(
            max_abs_delta <= thresholds.max_abs_thermo_delta,
            (
                f"{observable} max_abs_delta {max_abs_delta:g} exceeds "
                f"{thresholds.max_abs_thermo_delta:g}"
            ),
        )
        checked_observables.append(observable)
        max_seen_delta = max(max_seen_delta, max_abs_delta)
    return checked_observables, max_seen_delta


def _check_speedup(summary: dict[str, Any], min_speedup: float | None) -> float | None:
    """Optionally require the enabled cache to outperform the disabled baseline."""
    speedup_value = summary.get(SPEEDUP_KEY)
    if min_speedup is None and speedup_value is None:
        return None
    speedup = _as_number(speedup_value, f"{SUMMARY_KEY}.{SPEEDUP_KEY}")
    if min_speedup is not None:
        _require(
            speedup >= min_speedup,
            f"{SPEEDUP_KEY} {speedup:g} is below {min_speedup:g}",
        )
    return speedup


def _check_cache_evidence(
    report: dict[str, Any],
    min_hit_rate_percent: float,
    min_enabled_cache_attempts: int,
    min_enabled_cache_hits: int,
) -> dict[str, float]:
    """Require every enabled run to report enough cache activity."""
    hit_rates: list[float] = []
    attempt_counts: list[float] = []
    hit_counts: list[float] = []
    cache_count_residuals: list[float] = []
    verified_miss_keys: set[str] = set()
    for index, result in enumerate(_results(report)):
        result_map = _as_mapping(result, f"{RESULTS_KEY}[{index}]")
        if result_map.get(CASE_KEY) != ISODELTA_CASE:
            continue
        cache_summary = _as_mapping(
            result_map.get(CACHE_SUMMARY_KEY),
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}",
        )
        attempts = _as_number(
            cache_summary.get(ATTEMPTS_KEY),
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{ATTEMPTS_KEY}",
        )
        hits = _as_number(
            cache_summary.get(HITS_KEY),
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HITS_KEY}",
        )
        hit_rate_percent = _as_number(
            cache_summary.get(HIT_RATE_KEY),
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HIT_RATE_KEY}",
        )
        _require(
            attempts >= MIN_NONNEGATIVE_VALUE,
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{ATTEMPTS_KEY} must be nonnegative",
        )
        _require(
            hits >= MIN_NONNEGATIVE_VALUE,
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HITS_KEY} must be nonnegative",
        )
        _require(
            hits <= attempts,
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HITS_KEY} cannot exceed attempts",
        )
        _validate_percent(
            hit_rate_percent,
            f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HIT_RATE_KEY}",
        )
        expected_hit_rate = (
            MIN_PERCENT_VALUE
            if attempts == MIN_NONNEGATIVE_VALUE
            else MAX_PERCENT_VALUE * hits / attempts
        )
        _require(
            abs(hit_rate_percent - expected_hit_rate)
            <= CACHE_HIT_RATE_TOLERANCE_PERCENT,
            (
                f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{HIT_RATE_KEY} "
                "must match hits / attempts"
            ),
        )
        attempt_counts.append(attempts)
        hit_counts.append(hits)
        hit_rates.append(hit_rate_percent)
        miss_count_sum = 0.0
        for miss_key in REQUIRED_CACHE_MISS_KEYS:
            miss_count = _as_number(
                cache_summary.get(miss_key),
                f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{miss_key}",
            )
            _require(
                miss_count >= MIN_NONNEGATIVE_VALUE,
                f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{miss_key} must be nonnegative",
            )
            miss_count_sum += miss_count
            verified_miss_keys.add(miss_key)
        expected_miss_count = attempts - hits
        cache_count_residual = abs(miss_count_sum - expected_miss_count)
        _require(
            cache_count_residual <= CACHE_COUNT_TOLERANCE,
            (
                f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY} miss counters "
                "must match attempts - hits"
            ),
        )
        cache_count_residuals.append(cache_count_residual)

    _require(hit_rates, f"no {ISODELTA_CASE} cache hit-rate entries found")
    min_seen_attempts = min(attempt_counts)
    min_seen_hits = min(hit_counts)
    min_seen_hit_rate = min(hit_rates)
    _require(
        min_seen_attempts >= min_enabled_cache_attempts,
        f"minimum enabled attempts {min_seen_attempts:g} is below "
        f"{min_enabled_cache_attempts:g}",
    )
    _require(
        min_seen_hits >= min_enabled_cache_hits,
        f"minimum enabled hits {min_seen_hits:g} is below "
        f"{min_enabled_cache_hits:g}",
    )
    _require(
        min_seen_hit_rate >= min_hit_rate_percent,
        f"minimum enabled hit rate {min_seen_hit_rate:g}% is below "
        f"{min_hit_rate_percent:g}%",
    )
    return {
        "min_enabled_cache_attempts": min_seen_attempts,
        "min_enabled_cache_hits": min_seen_hits,
        "min_enabled_hit_rate_percent": min_seen_hit_rate,
        "max_cache_count_residual": max(cache_count_residuals),
        "verified_cache_miss_key_count": float(len(verified_miss_keys)),
    }


def validate_report(
    report: dict[str, Any], thresholds: ReportThresholds
) -> dict[str, Any]:
    """Validate one report and return a compact evidence summary."""
    validate_thresholds(thresholds)
    provenance_evidence = _check_provenance(report)
    summary = _summary(report)
    run_timeout_seconds = _check_run_timeout(report)
    result_count = _check_result_count(report, summary)
    paired_repeat_count = _check_paired_runs(report)
    successful_run_count = (
        _check_successful_runs(report) if thresholds.require_successful_runs else None
    )
    timing_evidence = _check_timing_summary(report, summary)
    checked_observables, max_seen_delta = _check_thermo_deltas(
        summary,
        thresholds,
        paired_repeat_count,
    )
    speedup = _check_speedup(summary, thresholds.min_speedup)
    cache_evidence = _check_cache_evidence(
        report,
        thresholds.min_hit_rate_percent,
        thresholds.min_enabled_cache_attempts,
        thresholds.min_enabled_cache_hits,
    )
    return {
        "status": "passed",
        "thresholds": asdict(thresholds),
        **provenance_evidence,
        "successful_run_count": successful_run_count,
        "result_count": result_count,
        "run_timeout_seconds": run_timeout_seconds,
        "paired_repeat_count": paired_repeat_count,
        **timing_evidence,
        "checked_observables": checked_observables,
        "max_seen_abs_thermo_delta": max_seen_delta,
        "speedup_vs_disabled_cache": speedup,
        **cache_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and validate a benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--max-abs-thermo-delta",
        type=float,
        default=DEFAULT_MAX_ABS_THERMO_DELTA,
        help="Largest allowed final thermo difference versus disabled cache",
    )
    parser.add_argument(
        "--min-paired-thermo-count",
        type=int,
        default=DEFAULT_MIN_PAIRED_THERMO_COUNT,
        help="Minimum paired baseline/enabled thermo samples per observable",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help="Optional minimum speedup versus disabled cache",
    )
    parser.add_argument(
        "--min-hit-rate-percent",
        type=float,
        default=DEFAULT_MIN_HIT_RATE_PERCENT,
        help="Minimum cache hit rate required for every enabled run",
    )
    parser.add_argument(
        "--min-enabled-cache-attempts",
        type=int,
        default=DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS,
        help="Minimum cache reuse attempts required for every enabled run",
    )
    parser.add_argument(
        "--min-enabled-cache-hits",
        type=int,
        default=DEFAULT_MIN_ENABLED_CACHE_HITS,
        help="Minimum cache hits required for every enabled run",
    )
    parser.add_argument(
        "--allow-failed-runs",
        action="store_true",
        help="Skip returncode checks when inspecting partial keep-going reports",
    )
    args = parser.parse_args(argv)

    thresholds = ReportThresholds(
        max_abs_thermo_delta=args.max_abs_thermo_delta,
        min_paired_thermo_count=args.min_paired_thermo_count,
        min_speedup=args.min_speedup,
        min_hit_rate_percent=args.min_hit_rate_percent,
        min_enabled_cache_attempts=args.min_enabled_cache_attempts,
        min_enabled_cache_hits=args.min_enabled_cache_hits,
        require_successful_runs=not args.allow_failed_runs,
    )
    try:
        evidence = validate_report(load_report(args.report), thresholds)
    except ReportCheckError as exc:
        print(f"IsoDelta-Halo benchmark report check failed: {exc}")
        return 1

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
