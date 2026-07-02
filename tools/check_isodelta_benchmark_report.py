"""Validate an IsoDelta-Halo benchmark report against publishable evidence.

The benchmark runner records timing, cache, and final-thermo fields. This
checker turns that JSON report into a pass/fail gate for correctness and effect
claims before the numbers are used in a paper or a performance table.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


# These keys mirror the benchmark report schema rather than scattering JSON
# field names through the validation logic.
SUMMARY_KEY = "summary"
RESULTS_KEY = "results"
CASE_KEY = "case"
RETURNCODE_KEY = "returncode"
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
FINAL_THERMO_DELTA_KEY = "final_thermo_delta_vs_disabled_cache"
MAX_ABS_DELTA_KEY = "max_abs_delta"
PAIRED_COUNT_KEY = "paired_count"
ISODELTA_CASE = "isodelta-enabled"
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
CACHE_HIT_RATE_TOLERANCE_PERCENT = 1.0e-9


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


def _as_number(value: Any, field_name: str) -> float:
    """Return a numeric field without accepting booleans as numbers."""
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    return float(value)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return the report summary object."""
    return _as_mapping(report.get(SUMMARY_KEY), SUMMARY_KEY)


def _results(report: dict[str, Any]) -> list[Any]:
    """Return the report result list."""
    return _as_sequence(report.get(RESULTS_KEY), RESULTS_KEY)


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


def _check_thermo_deltas(
    summary: dict[str, Any], thresholds: ReportThresholds
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
            paired_count >= thresholds.min_paired_thermo_count,
            (
                f"{observable} paired_count {paired_count:g} is below "
                f"{thresholds.min_paired_thermo_count}"
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
        for miss_key in REQUIRED_CACHE_MISS_KEYS:
            _as_number(
                cache_summary.get(miss_key),
                f"{RESULTS_KEY}[{index}].{CACHE_SUMMARY_KEY}.{miss_key}",
            )
            verified_miss_keys.add(miss_key)

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
        "verified_cache_miss_key_count": float(len(verified_miss_keys)),
    }


def validate_report(
    report: dict[str, Any], thresholds: ReportThresholds
) -> dict[str, Any]:
    """Validate one report and return a compact evidence summary."""
    validate_thresholds(thresholds)
    summary = _summary(report)
    successful_run_count = (
        _check_successful_runs(report) if thresholds.require_successful_runs else None
    )
    checked_observables, max_seen_delta = _check_thermo_deltas(summary, thresholds)
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
        "successful_run_count": successful_run_count,
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
