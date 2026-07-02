"""Run paired LAMMPS benchmarks for IsoDelta-Halo and baseline comparison.

The script executes the same LAMMPS input twice per repeat: once with the
metadata cache disabled and once with IsoDelta-Halo enabled. It stores raw logs
and emits a JSON report that can be used directly in profiling tables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


# Environment names mirror the C++ constants so the benchmark toggles the same
# runtime controls that PairE3GNNParallel reads.
REPO_ROOT_PARENT_DEPTH = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
REPORT_SCHEMA_VERSION = "isodelta-benchmark-report-v1"
DEFAULT_OUTPUT_DIR = Path("isodelta_benchmark_runs")
DEFAULT_REPEAT_COUNT = 3
DEFAULT_RUN_TIMEOUT_SECONDS = 3600.0
MIN_REPEAT_COUNT = 1
MIN_POSITIVE_TIMEOUT_SECONDS = 0.0
PERCENT_SCALE = 100.0
GIT_METADATA_TIMEOUT_SECONDS = 10.0
TIMEOUT_RETURN_CODE = 124
LAMMPS_INPUT_FLAG = "-in"
BASELINE_CASE = "baseline-disabled"
ISODELTA_CASE = "isodelta-enabled"
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"
THERMO_STEP_COLUMN = "Step"
MIN_THERMO_HEADER_COLUMNS = 2
ATTEMPTS_KEY = "attempts"
HITS_KEY = "hits"
HIT_RATE_PERCENT_KEY = "hit_rate_percent"
SUMMARY_RANK_COUNT_KEY = "summary_rank_count"
FINAL_THERMO_DELTA_KEY = "final_thermo_delta_vs_disabled_cache"
MAX_ABS_DELTA_KEY = "max_abs_delta"
PAIRED_COUNT_KEY = "paired_count"
MEAN_LOOP_TIME_KEY = "mean_loop_time_seconds"
SAMPLE_VARIANCE_LOOP_TIME_KEY = "sample_variance_loop_time_seconds"
SAMPLE_STDDEV_LOOP_TIME_KEY = "sample_stddev_loop_time_seconds"
MIN_LOOP_TIME_KEY = "min_loop_time_seconds"
MAX_LOOP_TIME_KEY = "max_loop_time_seconds"
VALID_LOOP_TIME_COUNT_KEY = "valid_loop_time_count"
MIN_SAMPLE_VARIANCE_COUNT = 2
SAMPLE_VARIANCE_DEGREES_OF_FREEDOM = 1
TIMEOUT_DETAIL_PREFIX = "LAMMPS benchmark timed out after"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOOP_TIME_RE = re.compile(
    rf"Loop time of\s+(?P<seconds>{FLOAT_PATTERN})\s+on\b",
    re.IGNORECASE,
)
SUMMARY_RE = re.compile(r"IsoDelta-Halo summary:\s+(?P<body>.*)")
SUMMARY_VALUE_RE = re.compile(
    rf"(?P<key>[A-Za-z0-9_-]+)=(?P<value>{FLOAT_PATTERN})"
)
FLOAT_TOKEN_RE = re.compile(FLOAT_PATTERN)


@dataclass(frozen=True)
class BenchmarkCase:
    """Describe one benchmark variant and the environment overrides it needs."""

    name: str
    env_updates: dict[str, str]


@dataclass(frozen=True)
class BenchmarkResult:
    """Store one run result in a JSON-friendly shape."""

    case: str
    repeat_index: int
    returncode: int
    loop_time_seconds: float | None
    cache_summary: dict[str, float]
    final_thermo_observables: dict[str, float]
    stdout_path: str
    stderr_path: str


BENCHMARK_CASES = (
    BenchmarkCase(
        name=BASELINE_CASE,
        env_updates={
            PRINT_INFO_ENV: ENV_FLAG_ENABLED,
            DISABLE_CACHE_ENV: ENV_FLAG_ENABLED,
            PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
        },
    ),
    BenchmarkCase(
        name=ISODELTA_CASE,
        env_updates={
            PRINT_INFO_ENV: ENV_FLAG_ENABLED,
            PROFILE_CACHE_ENV: ENV_FLAG_ENABLED,
        },
    ),
)


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    """Return timeout-captured output as text for raw log persistence."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _split_lammps_command(lammps_command: str) -> list[str]:
    """Split a user command and reject empty command lines."""
    command_tokens = shlex.split(lammps_command)
    if not command_tokens:
        raise ValueError("lammps_command must not be empty")
    return command_tokens


def validate_benchmark_options(
    lammps_command: str,
    repeat_count: int,
    run_timeout_seconds: float,
) -> None:
    """Reject benchmark options that cannot produce paired timing evidence."""
    _split_lammps_command(lammps_command)
    if repeat_count < MIN_REPEAT_COUNT:
        raise ValueError(f"repeat_count must be at least {MIN_REPEAT_COUNT}")
    if not math.isfinite(run_timeout_seconds):
        raise ValueError("run_timeout_seconds must be finite")
    if run_timeout_seconds <= MIN_POSITIVE_TIMEOUT_SECONDS:
        raise ValueError("run_timeout_seconds must be positive")


def parse_loop_time(log_text: str) -> float | None:
    """Extract the LAMMPS loop time from stdout/stderr text when present."""
    match = LOOP_TIME_RE.search(log_text)
    if match is None:
        return None
    return float(match.group("seconds"))


def _run_metadata_command(argv: list[str]) -> str | None:
    """Run a short metadata command and return stripped stdout when it works."""
    try:
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=GIT_METADATA_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_run_provenance() -> dict[str, Any]:
    """Collect report provenance so benchmark numbers remain auditable."""
    git_status_short = _run_metadata_command(["git", "status", "--short"])
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "git_commit": _run_metadata_command(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_metadata_command(["git", "branch", "--show-current"]),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "case_environment_overrides": {
            case.name: case.env_updates for case in BENCHMARK_CASES
        },
    }


def parse_cache_summary(log_text: str) -> dict[str, float]:
    """Aggregate IsoDelta-Halo summary counters from all MPI rank logs."""
    summary: dict[str, float] = {}
    summary_rank_count = 0.0
    for summary_match in SUMMARY_RE.finditer(log_text):
        summary_rank_count += 1.0
        body = summary_match.group("body")
        for value_match in SUMMARY_VALUE_RE.finditer(body):
            key = value_match.group("key")
            if key == HIT_RATE_PERCENT_KEY:
                continue
            summary[key] = summary.get(key, 0.0) + float(value_match.group("value"))
    if summary_rank_count:
        summary[SUMMARY_RANK_COUNT_KEY] = summary_rank_count
        attempts = summary.get(ATTEMPTS_KEY, 0.0)
        hits = summary.get(HITS_KEY, 0.0)
        summary[HIT_RATE_PERCENT_KEY] = (
            0.0 if attempts == 0.0 else PERCENT_SCALE * hits / attempts
        )
    return summary


def _parse_float_token(token: str) -> float | None:
    """Return a float for plain numeric LAMMPS table tokens."""
    if FLOAT_TOKEN_RE.fullmatch(token) is None:
        return None
    return float(token)


def parse_final_thermo_observables(log_text: str) -> dict[str, float]:
    """Extract the final numeric row from the latest LAMMPS thermo table."""
    active_headers: list[str] | None = None
    final_observables: dict[str, float] = {}

    for line in log_text.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == THERMO_STEP_COLUMN and len(tokens) >= MIN_THERMO_HEADER_COLUMNS:
            active_headers = tokens
            continue
        if active_headers is None or len(tokens) < len(active_headers):
            continue

        row_values: list[float] = []
        for token in tokens[: len(active_headers)]:
            value = _parse_float_token(token)
            if value is None:
                row_values = []
                break
            row_values.append(value)
        if row_values:
            final_observables = dict(zip(active_headers, row_values))

    return final_observables


def parse_run_output(
    stdout_text: str, stderr_text: str
) -> tuple[float | None, dict[str, float], dict[str, float]]:
    """Parse both streams because MPI launchers may route logs differently."""
    combined_log = stdout_text + "\n" + stderr_text
    return (
        parse_loop_time(combined_log),
        parse_cache_summary(combined_log),
        parse_final_thermo_observables(combined_log),
    )


def _case_environment(case: BenchmarkCase) -> dict[str, str]:
    """Create an isolated environment for a benchmark case."""
    env = os.environ.copy()
    env.update(case.env_updates)
    if case.name == ISODELTA_CASE:
        env.pop(DISABLE_CACHE_ENV, None)
    return env


def _run_case(
    command: list[str],
    case: BenchmarkCase,
    repeat_index: int,
    work_dir: Path,
    output_dir: Path,
    keep_going: bool,
    run_timeout_seconds: float,
) -> BenchmarkResult:
    """Run one case, persist logs, parse metrics, and optionally fail fast."""
    stdout_path = output_dir / f"{case.name}_repeat{repeat_index}.stdout.log"
    stderr_path = output_dir / f"{case.name}_repeat{repeat_index}.stderr.log"
    try:
        completed = subprocess.run(
            command,
            cwd=work_dir,
            env=_case_environment(case),
            text=True,
            capture_output=True,
            check=False,
            timeout=run_timeout_seconds,
        )
        stdout_text = completed.stdout
        stderr_text = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_text = _coerce_timeout_stream(exc.output)
        timeout_stderr = _coerce_timeout_stream(exc.stderr)
        timeout_detail = f"{TIMEOUT_DETAIL_PREFIX} {run_timeout_seconds:g} seconds"
        stderr_text = (
            f"{timeout_stderr}\n{timeout_detail}" if timeout_stderr else timeout_detail
        )
        returncode = TIMEOUT_RETURN_CODE

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    loop_time, cache_summary, final_thermo_observables = parse_run_output(
        stdout_text, stderr_text
    )

    result = BenchmarkResult(
        case=case.name,
        repeat_index=repeat_index,
        returncode=returncode,
        loop_time_seconds=loop_time,
        cache_summary=cache_summary,
        final_thermo_observables=final_thermo_observables,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
    if returncode != 0 and not keep_going:
        raise RuntimeError(
            f"{case.name} repeat {repeat_index} failed with exit code "
            f"{returncode}. See {stderr_path}"
        )
    return result


def _build_command(lammps_command: str, input_path: Path) -> list[str]:
    """Build a LAMMPS command without relying on shell-specific quoting."""
    return [*_split_lammps_command(lammps_command), LAMMPS_INPUT_FLAG, str(input_path)]


def _summarize(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Compute simple aggregate metrics for quick terminal inspection."""
    summary: dict[str, Any] = {"runs": len(results), "cases": {}}
    for case_name in (BASELINE_CASE, ISODELTA_CASE):
        case_times = [
            result.loop_time_seconds
            for result in results
            if result.case == case_name and result.loop_time_seconds is not None
        ]
        summary["cases"][case_name] = _summarize_loop_times(case_times)
    baseline_mean = summary["cases"][BASELINE_CASE][MEAN_LOOP_TIME_KEY]
    isodelta_mean = summary["cases"][ISODELTA_CASE][MEAN_LOOP_TIME_KEY]
    if baseline_mean and isodelta_mean:
        summary["speedup_vs_disabled_cache"] = baseline_mean / isodelta_mean
    else:
        summary["speedup_vs_disabled_cache"] = None
    summary[FINAL_THERMO_DELTA_KEY] = _summarize_final_thermo_deltas(results)
    return summary


def _summarize_loop_times(loop_times: list[float]) -> dict[str, float | int | None]:
    """Compute repeat statistics for one benchmark case."""
    if not loop_times:
        return {
            MEAN_LOOP_TIME_KEY: None,
            SAMPLE_VARIANCE_LOOP_TIME_KEY: None,
            SAMPLE_STDDEV_LOOP_TIME_KEY: None,
            MIN_LOOP_TIME_KEY: None,
            MAX_LOOP_TIME_KEY: None,
            VALID_LOOP_TIME_COUNT_KEY: 0,
        }

    loop_time_count = len(loop_times)
    mean_loop_time = sum(loop_times) / loop_time_count
    if loop_time_count >= MIN_SAMPLE_VARIANCE_COUNT:
        squared_delta_sum = sum(
            (loop_time - mean_loop_time) ** 2 for loop_time in loop_times
        )
        sample_variance = squared_delta_sum / (
            loop_time_count - SAMPLE_VARIANCE_DEGREES_OF_FREEDOM
        )
        sample_stddev = math.sqrt(sample_variance)
    else:
        sample_variance = None
        sample_stddev = None

    return {
        MEAN_LOOP_TIME_KEY: mean_loop_time,
        SAMPLE_VARIANCE_LOOP_TIME_KEY: sample_variance,
        SAMPLE_STDDEV_LOOP_TIME_KEY: sample_stddev,
        MIN_LOOP_TIME_KEY: min(loop_times),
        MAX_LOOP_TIME_KEY: max(loop_times),
        VALID_LOOP_TIME_COUNT_KEY: loop_time_count,
    }


def _summarize_final_thermo_deltas(results: list[BenchmarkResult]) -> dict[str, dict[str, float]]:
    """Compare final thermo scalars between paired baseline and enabled runs."""
    paired_by_repeat: dict[int, dict[str, BenchmarkResult]] = {}
    for result in results:
        paired_by_repeat.setdefault(result.repeat_index, {})[result.case] = result

    deltas_by_observable: dict[str, list[float]] = {}
    for paired_results in paired_by_repeat.values():
        baseline = paired_results.get(BASELINE_CASE)
        enabled = paired_results.get(ISODELTA_CASE)
        if baseline is None or enabled is None:
            continue
        common_observables = (
            set(baseline.final_thermo_observables)
            & set(enabled.final_thermo_observables)
            - {THERMO_STEP_COLUMN}
        )
        for observable in common_observables:
            delta = abs(
                enabled.final_thermo_observables[observable]
                - baseline.final_thermo_observables[observable]
            )
            deltas_by_observable.setdefault(observable, []).append(delta)

    return {
        observable: {
            MAX_ABS_DELTA_KEY: max(deltas),
            PAIRED_COUNT_KEY: float(len(deltas)),
        }
        for observable, deltas in sorted(deltas_by_observable.items())
        if deltas
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run paired benchmarks, and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-command", required=True, help="Example: lmp")
    parser.add_argument("--input", required=True, type=Path, help="LAMMPS input script")
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="Number of baseline/enabled pairs to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for logs and JSON report",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for each LAMMPS benchmark run",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory where LAMMPS should run; defaults to the input directory",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Write partial reports even if one benchmark command fails",
    )
    args = parser.parse_args(argv)
    try:
        validate_benchmark_options(
            args.lammps_command,
            args.repeat,
            args.run_timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve() if args.work_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(args.lammps_command, input_path)

    results: list[BenchmarkResult] = []
    for repeat_index in range(args.repeat):
        for case in BENCHMARK_CASES:
            results.append(
                _run_case(
                    command=command,
                    case=case,
                    repeat_index=repeat_index,
                    work_dir=work_dir,
                    output_dir=output_dir,
                    keep_going=args.keep_going,
                    run_timeout_seconds=args.run_timeout_seconds,
                )
            )

    report = {
        "provenance": collect_run_provenance(),
        "command": command,
        "input": str(input_path),
        "work_dir": str(work_dir),
        "run_timeout_seconds": args.run_timeout_seconds,
        "summary": _summarize(results),
        "results": [asdict(result) for result in results],
    }
    report_path = output_dir / "isodelta_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"IsoDelta-Halo benchmark report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
