"""Run the full IsoDelta-Halo experiment gate in a reproducible sequence.

This driver connects the lightweight source checks, LAMMPS binary smoke check,
paired benchmark, and benchmark report gate. It writes every command log and a
machine-readable experiment report so paper results can be audited later.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Callable


# All filenames and defaults are named so experimental acceptance criteria stay
# visible in one place rather than being hidden in command construction.
REPO_ROOT_PARENT_DEPTH = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
EXPERIMENT_REPORT_SCHEMA_VERSION = "isodelta-experiment-report-v1"
PREREQ_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_build_prereqs.py"
BINARY_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_lammps_binary.py"
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_lammps_benchmark.py"
REPORT_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
EVIDENCE_BUNDLE_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_evidence_bundle.py"
DEFAULT_OUTPUT_DIR = Path("isodelta_experiment_runs")
DEFAULT_REPEAT_COUNT = 3
MIN_REPEAT_COUNT = 1
DEFAULT_MAX_ABS_THERMO_DELTA = 1.0e-8
DEFAULT_MIN_PAIRED_THERMO_COUNT = 1
DEFAULT_MIN_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS = 1
DEFAULT_MIN_ENABLED_CACHE_HITS = 0
DEFAULT_MIN_TRACE_COUNT = 1
DEFAULT_MIN_TRACE_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT = 0.0
DEFAULT_BINARY_TIMEOUT_SECONDS = 60.0
MIN_POSITIVE_TIMEOUT_SECONDS = 0.0
MIN_POSITIVE_SPEEDUP = 0.0
MIN_NONNEGATIVE_VALUE = 0.0
MIN_COUNT_VALUE = 0
MIN_PERCENT_VALUE = 0.0
MAX_PERCENT_VALUE = 100.0
GIT_METADATA_TIMEOUT_SECONDS = 10.0
LOG_DIR_NAME = "logs"
BENCHMARK_DIR_NAME = "benchmark"
EXPERIMENT_REPORT_NAME = "isodelta_experiment_report.json"
BENCHMARK_REPORT_NAME = "isodelta_benchmark_report.json"
BUNDLE_EVIDENCE_REPORT_NAME = "bundle_evidence.json"
SUCCESS_RETURN_CODE = 0


@dataclass(frozen=True)
class ExperimentConfig:
    """Store user-facing options for one end-to-end experiment."""

    lammps_command: str
    input_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    repeat_count: int = DEFAULT_REPEAT_COUNT
    work_dir: Path | None = None
    lammps_root: Path | None = None
    require_torch: bool = False
    benchmark_keep_going: bool = False
    allow_failed_report_runs: bool = False
    max_abs_thermo_delta: float = DEFAULT_MAX_ABS_THERMO_DELTA
    min_paired_thermo_count: int = DEFAULT_MIN_PAIRED_THERMO_COUNT
    min_speedup: float | None = None
    min_hit_rate_percent: float = DEFAULT_MIN_HIT_RATE_PERCENT
    min_enabled_cache_attempts: int = DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS
    min_enabled_cache_hits: int = DEFAULT_MIN_ENABLED_CACHE_HITS
    trace_evidence_paths: tuple[Path, ...] = ()
    required_trace_models: tuple[str, ...] = ()
    min_trace_count: int = DEFAULT_MIN_TRACE_COUNT
    min_trace_hit_rate_percent: float = DEFAULT_MIN_TRACE_HIT_RATE_PERCENT
    min_trace_estimated_speedup: float | None = None
    min_trace_metadata_fraction_percent: float = (
        DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT
    )
    binary_timeout_seconds: float = DEFAULT_BINARY_TIMEOUT_SECONDS

    def benchmark_output_dir(self) -> Path:
        """Return the directory where the paired benchmark writes logs."""
        return self.output_dir / BENCHMARK_DIR_NAME

    def benchmark_report_path(self) -> Path:
        """Return the JSON report produced by the paired benchmark runner."""
        return self.benchmark_output_dir() / BENCHMARK_REPORT_NAME

    def experiment_report_path(self) -> Path:
        """Return the JSON report produced by this driver."""
        return self.output_dir / EXPERIMENT_REPORT_NAME

    def bundle_evidence_report_path(self) -> Path:
        """Return the optional combined evidence bundle report path."""
        return self.output_dir / BUNDLE_EVIDENCE_REPORT_NAME

    def should_run_bundle_gate(self) -> bool:
        """Return whether trace evidence options request the bundle gate."""
        return bool(self.trace_evidence_paths or self.required_trace_models)


@dataclass(frozen=True)
class ExperimentCommand:
    """Represent one external command and its log destinations."""

    name: str
    argv: list[str]
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class ExperimentCommandResult:
    """Represent one completed experiment stage for JSON reporting."""

    name: str
    argv: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _require_valid_config(condition: bool, message: str) -> None:
    """Raise a configuration error before launching external commands."""
    if not condition:
        raise ValueError(message)


def _validate_finite(value: float, field_name: str) -> None:
    """Reject NaN and infinity in numeric experiment gates."""
    _require_valid_config(math.isfinite(value), f"{field_name} must be finite")


def _validate_percent(value: float, field_name: str) -> None:
    """Require a percentage threshold to stay within its physical range."""
    _validate_finite(value, field_name)
    _require_valid_config(
        MIN_PERCENT_VALUE <= value <= MAX_PERCENT_VALUE,
        f"{field_name} must be between {MIN_PERCENT_VALUE:g} and {MAX_PERCENT_VALUE:g}",
    )


def validate_config(config: ExperimentConfig) -> None:
    """Reject experiment settings that cannot produce auditable evidence."""
    _require_valid_config(
        bool(config.lammps_command.strip()),
        "lammps_command must not be empty",
    )
    _require_valid_config(
        config.repeat_count >= MIN_REPEAT_COUNT,
        f"repeat_count must be at least {MIN_REPEAT_COUNT}",
    )
    _validate_finite(config.binary_timeout_seconds, "binary_timeout_seconds")
    _require_valid_config(
        config.binary_timeout_seconds > MIN_POSITIVE_TIMEOUT_SECONDS,
        "binary_timeout_seconds must be positive",
    )
    _validate_finite(config.max_abs_thermo_delta, "max_abs_thermo_delta")
    _require_valid_config(
        config.max_abs_thermo_delta >= MIN_NONNEGATIVE_VALUE,
        "max_abs_thermo_delta must be nonnegative",
    )
    _require_valid_config(
        config.min_paired_thermo_count >= MIN_REPEAT_COUNT,
        "min_paired_thermo_count must be at least one",
    )
    _require_valid_config(
        config.min_enabled_cache_attempts >= MIN_COUNT_VALUE,
        "min_enabled_cache_attempts must be nonnegative",
    )
    _require_valid_config(
        config.min_enabled_cache_hits >= MIN_COUNT_VALUE,
        "min_enabled_cache_hits must be nonnegative",
    )
    _require_valid_config(
        config.min_enabled_cache_hits <= config.min_enabled_cache_attempts,
        "min_enabled_cache_hits cannot exceed min_enabled_cache_attempts",
    )
    _require_valid_config(
        config.min_trace_count >= MIN_REPEAT_COUNT,
        "min_trace_count must be at least one",
    )
    if config.min_speedup is not None:
        _validate_finite(config.min_speedup, "min_speedup")
        _require_valid_config(
            config.min_speedup > MIN_POSITIVE_SPEEDUP,
            "min_speedup must be positive when provided",
        )
    if config.min_trace_estimated_speedup is not None:
        _validate_finite(
            config.min_trace_estimated_speedup,
            "min_trace_estimated_speedup",
        )
        _require_valid_config(
            config.min_trace_estimated_speedup > MIN_POSITIVE_SPEEDUP,
            "min_trace_estimated_speedup must be positive when provided",
        )
    _validate_percent(config.min_hit_rate_percent, "min_hit_rate_percent")
    _validate_percent(
        config.min_trace_hit_rate_percent,
        "min_trace_hit_rate_percent",
    )
    _validate_percent(
        config.min_trace_metadata_fraction_percent,
        "min_trace_metadata_fraction_percent",
    )
    for model_name in config.required_trace_models:
        _require_valid_config(
            bool(model_name.strip()),
            "required_trace_models must not include empty names",
        )


def _python_script_command(script_path: Path) -> list[str]:
    """Build a Python command that works from Windows and POSIX shells."""
    return [sys.executable, str(script_path)]


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
    if completed.returncode != SUCCESS_RETURN_CODE:
        return None
    return completed.stdout.strip()


def collect_run_provenance() -> dict[str, str | bool | None]:
    """Collect driver provenance so experiment reports can be audited later."""
    git_status_short = _run_metadata_command(["git", "status", "--short"])
    return {
        "report_schema_version": EXPERIMENT_REPORT_SCHEMA_VERSION,
        "git_commit": _run_metadata_command(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_metadata_command(["git", "branch", "--show-current"]),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def build_experiment_commands(config: ExperimentConfig) -> list[ExperimentCommand]:
    """Build the ordered commands that make up the full experiment gate."""
    validate_config(config)
    log_dir = config.output_dir / LOG_DIR_NAME

    prereq_argv = _python_script_command(PREREQ_SCRIPT)
    if config.lammps_root is not None:
        prereq_argv.extend(["--lammps-root", str(config.lammps_root)])
    if config.require_torch:
        prereq_argv.append("--require-torch")

    binary_argv = [
        *_python_script_command(BINARY_CHECK_SCRIPT),
        "--lammps-command",
        config.lammps_command,
        "--timeout-seconds",
        str(config.binary_timeout_seconds),
    ]

    benchmark_argv = [
        *_python_script_command(BENCHMARK_SCRIPT),
        "--lammps-command",
        config.lammps_command,
        "--input",
        str(config.input_path),
        "--repeat",
        str(config.repeat_count),
        "--output-dir",
        str(config.benchmark_output_dir()),
    ]
    if config.work_dir is not None:
        benchmark_argv.extend(["--work-dir", str(config.work_dir)])
    if config.benchmark_keep_going:
        benchmark_argv.append("--keep-going")

    report_argv = [
        *_python_script_command(REPORT_CHECK_SCRIPT),
        "--report",
        str(config.benchmark_report_path()),
        "--max-abs-thermo-delta",
        str(config.max_abs_thermo_delta),
        "--min-paired-thermo-count",
        str(config.min_paired_thermo_count),
        "--min-hit-rate-percent",
        str(config.min_hit_rate_percent),
        "--min-enabled-cache-attempts",
        str(config.min_enabled_cache_attempts),
        "--min-enabled-cache-hits",
        str(config.min_enabled_cache_hits),
    ]
    if config.min_speedup is not None:
        report_argv.extend(["--min-speedup", str(config.min_speedup)])
    if config.allow_failed_report_runs:
        report_argv.append("--allow-failed-runs")

    commands = [
        ExperimentCommand(
            name="prerequisites",
            argv=prereq_argv,
            stdout_path=log_dir / "prerequisites.stdout.log",
            stderr_path=log_dir / "prerequisites.stderr.log",
        ),
        ExperimentCommand(
            name="binary-smoke",
            argv=binary_argv,
            stdout_path=log_dir / "binary_smoke.stdout.log",
            stderr_path=log_dir / "binary_smoke.stderr.log",
        ),
        ExperimentCommand(
            name="paired-benchmark",
            argv=benchmark_argv,
            stdout_path=log_dir / "paired_benchmark.stdout.log",
            stderr_path=log_dir / "paired_benchmark.stderr.log",
        ),
        ExperimentCommand(
            name="report-gate",
            argv=report_argv,
            stdout_path=log_dir / "report_gate.stdout.log",
            stderr_path=log_dir / "report_gate.stderr.log",
        ),
    ]

    if config.should_run_bundle_gate():
        bundle_argv = [
            *_python_script_command(EVIDENCE_BUNDLE_CHECK_SCRIPT),
            "--benchmark-report",
            str(config.benchmark_report_path()),
            "--min-trace-count",
            str(config.min_trace_count),
            "--max-abs-thermo-delta",
            str(config.max_abs_thermo_delta),
            "--min-paired-thermo-count",
            str(config.min_paired_thermo_count),
            "--min-hit-rate-percent",
            str(config.min_hit_rate_percent),
            "--min-enabled-cache-attempts",
            str(config.min_enabled_cache_attempts),
            "--min-enabled-cache-hits",
            str(config.min_enabled_cache_hits),
            "--min-trace-hit-rate-percent",
            str(config.min_trace_hit_rate_percent),
            "--min-trace-metadata-fraction-percent",
            str(config.min_trace_metadata_fraction_percent),
            "--output",
            str(config.bundle_evidence_report_path()),
        ]
        if config.min_speedup is not None:
            bundle_argv.extend(["--min-speedup", str(config.min_speedup)])
        if config.min_trace_estimated_speedup is not None:
            bundle_argv.extend(
                [
                    "--min-trace-estimated-speedup",
                    str(config.min_trace_estimated_speedup),
                ]
            )
        if config.allow_failed_report_runs:
            bundle_argv.append("--allow-failed-runs")
        for trace_evidence_path in config.trace_evidence_paths:
            bundle_argv.extend(["--trace-evidence", str(trace_evidence_path)])
        for model_name in config.required_trace_models:
            bundle_argv.extend(["--require-trace-model", model_name])
        commands.append(
            ExperimentCommand(
                name="evidence-bundle",
                argv=bundle_argv,
                stdout_path=log_dir / "evidence_bundle.stdout.log",
                stderr_path=log_dir / "evidence_bundle.stderr.log",
            )
        )

    return commands


def _run_command(
    command: ExperimentCommand,
    runner: CommandRunner = subprocess.run,
) -> ExperimentCommandResult:
    """Execute one command, write its logs, and return a serializable result."""
    completed = runner(
        command.argv,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    command.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command.stdout_path.write_text(completed.stdout, encoding="utf-8")
    command.stderr_path.write_text(completed.stderr, encoding="utf-8")
    return ExperimentCommandResult(
        name=command.name,
        argv=command.argv,
        returncode=int(completed.returncode),
        stdout_path=str(command.stdout_path),
        stderr_path=str(command.stderr_path),
    )


def _write_report(
    config: ExperimentConfig,
    command_results: list[ExperimentCommandResult],
    ok: bool,
    failed_stage: str | None,
) -> None:
    """Write the top-level experiment report after each completed stage."""
    config_payload = {
        **asdict(config),
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "work_dir": str(config.work_dir) if config.work_dir else None,
        "lammps_root": str(config.lammps_root) if config.lammps_root else None,
        "trace_evidence_paths": [
            str(path) for path in config.trace_evidence_paths
        ],
    }
    payload = {
        "ok": ok,
        "failed_stage": failed_stage,
        "provenance": collect_run_provenance(),
        "config": config_payload,
        "benchmark_report": str(config.benchmark_report_path()),
        "bundle_evidence_report": (
            str(config.bundle_evidence_report_path())
            if config.should_run_bundle_gate()
            else None
        ),
        "commands": [asdict(result) for result in command_results],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.experiment_report_path().write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig,
    runner: CommandRunner = subprocess.run,
) -> int:
    """Run the full gate and stop on the first failed command."""
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    command_results: list[ExperimentCommandResult] = []
    failed_stage: str | None = None

    for command in build_experiment_commands(config):
        print(f"[IsoDelta-Halo experiment] {command.name}: {' '.join(command.argv)}")
        result = _run_command(command, runner)
        command_results.append(result)
        if result.returncode != SUCCESS_RETURN_CODE:
            failed_stage = result.name
            _write_report(config, command_results, ok=False, failed_stage=failed_stage)
            print(
                "[IsoDelta-Halo experiment] failed at "
                f"{failed_stage}; see {config.experiment_report_path()}"
            )
            return result.returncode

    _write_report(config, command_results, ok=True, failed_stage=None)
    print(
        "[IsoDelta-Halo experiment] all gates passed; report written to "
        f"{config.experiment_report_path()}"
    )
    return SUCCESS_RETURN_CODE


def _parse_args(argv: list[str] | None) -> ExperimentConfig:
    """Parse command-line options into an experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-command", required=True, help="Example: mpiexec -n 4 lmp")
    parser.add_argument("--input", required=True, type=Path, help="LAMMPS input script")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for experiment logs and reports",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="Number of baseline/enabled benchmark pairs",
    )
    parser.add_argument("--work-dir", type=Path, help="LAMMPS working directory")
    parser.add_argument("--lammps-root", type=Path, help="LAMMPS source root")
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Require Python torch import during prerequisite checks",
    )
    parser.add_argument(
        "--benchmark-keep-going",
        action="store_true",
        help="Ask the paired benchmark runner to write partial reports on failures",
    )
    parser.add_argument(
        "--allow-failed-report-runs",
        action="store_true",
        help="Allow failed benchmark runs when checking a keep-going report",
    )
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
        help="Minimum paired thermo samples per observable",
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
        "--trace-evidence",
        action="append",
        default=[],
        type=Path,
        help="Optional MLIP trace evidence JSON for the bundle gate",
    )
    parser.add_argument(
        "--require-trace-model",
        action="append",
        default=[],
        help="Model label that must appear in optional trace evidence",
    )
    parser.add_argument(
        "--min-trace-count",
        type=int,
        default=DEFAULT_MIN_TRACE_COUNT,
        help="Minimum number of trace evidence files for the bundle gate",
    )
    parser.add_argument(
        "--min-trace-hit-rate-percent",
        type=float,
        default=DEFAULT_MIN_TRACE_HIT_RATE_PERCENT,
        help="Minimum reusable-step hit rate in each trace evidence file",
    )
    parser.add_argument(
        "--min-trace-estimated-speedup",
        type=float,
        help="Optional minimum trace-estimated speedup for the bundle gate",
    )
    parser.add_argument(
        "--min-trace-metadata-fraction-percent",
        type=float,
        default=DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT,
        help="Minimum metadata-build fraction in each trace evidence file",
    )
    parser.add_argument(
        "--binary-timeout-seconds",
        type=float,
        default=DEFAULT_BINARY_TIMEOUT_SECONDS,
        help="Timeout for the LAMMPS help smoke check",
    )
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        lammps_command=args.lammps_command,
        input_path=args.input,
        output_dir=args.output_dir,
        repeat_count=args.repeat,
        work_dir=args.work_dir,
        lammps_root=args.lammps_root,
        require_torch=args.require_torch,
        benchmark_keep_going=args.benchmark_keep_going,
        allow_failed_report_runs=args.allow_failed_report_runs,
        max_abs_thermo_delta=args.max_abs_thermo_delta,
        min_paired_thermo_count=args.min_paired_thermo_count,
        min_speedup=args.min_speedup,
        min_hit_rate_percent=args.min_hit_rate_percent,
        min_enabled_cache_attempts=args.min_enabled_cache_attempts,
        min_enabled_cache_hits=args.min_enabled_cache_hits,
        trace_evidence_paths=tuple(args.trace_evidence),
        required_trace_models=tuple(args.require_trace_model),
        min_trace_count=args.min_trace_count,
        min_trace_hit_rate_percent=args.min_trace_hit_rate_percent,
        min_trace_estimated_speedup=args.min_trace_estimated_speedup,
        min_trace_metadata_fraction_percent=args.min_trace_metadata_fraction_percent,
        binary_timeout_seconds=args.binary_timeout_seconds,
    )
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the end-to-end experiment driver."""
    return run_experiment(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
