"""Run an IsoDelta-Halo 8-GPU cluster suite and produce paper artifacts.

The suite is intentionally manifest-driven. SevenNet can use the in-repository
LAMMPS experiment driver directly, while MACE, NequIP, and other MLIP runtimes
can provide their own disabled/enabled commands and portable trace evidence.
This keeps paper automation reproducible without pretending that every external
foundation model uses the same checkpoint format, dataset URL, or launcher.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


# Constants are named because this script becomes part of the experimental
# method: reviewers should see every gate and unit without hunting literals.
REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_SCHEMA_VERSION = "isodelta-cluster-paper-suite-v1"
READINESS_SCHEMA_VERSION = "isodelta-cluster-readiness-v1"
ARTIFACT_PREPARATION_SCHEMA_VERSION = "isodelta-artifact-preparation-v1"
PREFLIGHT_REPORT_SCHEMA_VERSION = "isodelta-cluster-preflight-v1"
EXTERNAL_TIMING_SCHEMA_VERSION = "isodelta-external-pair-timing-v1"
DEFAULT_OUTPUT_DIR = Path("isodelta_cluster_paper_runs")
DEFAULT_EXPECTED_GPU_COUNT = 8
DEFAULT_REPEAT_COUNT = 3
DEFAULT_BINARY_TIMEOUT_SECONDS = 60.0
DEFAULT_BENCHMARK_TIMEOUT_SECONDS = 3600.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 3600.0
DEFAULT_MAX_ABS_THERMO_DELTA = 1.0e-8
DEFAULT_MIN_PAIRED_THERMO_COUNT = 1
DEFAULT_MIN_SPEEDUP = 1.0
DEFAULT_MIN_SPEEDUP_95CI_LOWER_BOUND: float | None = None
DEFAULT_MIN_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS = 1
DEFAULT_MIN_ENABLED_CACHE_HITS = 0
DEFAULT_MIN_TRACE_COUNT = 1
DEFAULT_MIN_DISTINCT_TRACE_MODELS = 1
DEFAULT_MIN_TRACE_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT = 0.0
DEFAULT_REQUIRED_MODELS = ("SevenNet", "MACE", "NequIP")
DEFAULT_REQUIRE_ARTIFACT_SHA256 = False
FINAL_PAPER_REQUIRED_MODELS = DEFAULT_REQUIRED_MODELS
FINAL_PAPER_PAIRED_CASE_KINDS = frozenset(("sevennet_lammps", "external_pair"))
FINAL_PAPER_MIN_REPEAT_COUNT = DEFAULT_REPEAT_COUNT
UNRESOLVED_TEMPLATE_MARKERS = ("example.org", "replace-with-real")
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_SLURM_JOB_NAME = "isodelta-halo-paper-suite"
DEFAULT_SLURM_TIME_LIMIT = "24:00:00"
DEFAULT_SLURM_CPUS_PER_TASK = 8
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 300.0
CASE_STATUS_PASSED = "passed"
CASE_STATUS_REUSED = "reused"
PASSING_CASE_STATUSES = frozenset((CASE_STATUS_PASSED, CASE_STATUS_REUSED))
PREFLIGHT_STATUS_PASSED = "passed"
PREFLIGHT_STATUS_FAILED = "failed"
PREFLIGHT_STATUS_PLANNED = "planned"
PREFLIGHT_STATUS_SKIPPED = "skipped"
PREFLIGHT_SKIP_DOWNLOADS_REASON = "skip_downloads"
PREFLIGHT_NO_COMMAND_REASON = "no preflight_command"
SUPPORTED_CASE_KINDS = frozenset(("sevennet_lammps", "external_pair", "trace_only"))
BENCHMARK_REPORT_NAME = "isodelta_benchmark_report.json"
BUNDLE_EVIDENCE_NAME = "bundle_evidence.json"
EXPERIMENT_REPORT_NAME = "isodelta_experiment_report.json"
EXTERNAL_TIMING_REPORT_NAME = "external_pair_timing_report.json"
TRACE_EVIDENCE_SUFFIX = "_trace_evidence.json"
PLAN_REPORT_NAME = "isodelta_cluster_paper_plan.json"
SUMMARY_REPORT_NAME = "isodelta_cluster_paper_summary.json"
ARTIFACT_PREPARATION_REPORT_NAME = "artifact_preparation_report.json"
PREFLIGHT_REPORT_NAME = "preflight_report.json"
MANIFEST_SNAPSHOT_NAME = "isodelta_cluster_suite_manifest.toml"
SLURM_LOG_DIR_NAME = "slurm_logs"
ENVIRONMENT_SNAPSHOT_NAME = "environment_snapshot.json"
ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION = "isodelta-cluster-environment-snapshot-v1"
ENVIRONMENT_PACKAGE_NAMES = (
    "sevenn",
    "torch",
    "e3nn",
    "ase",
    "numpy",
    "scipy",
    "mace-torch",
    "nequip",
)
ENVIRONMENT_VARIABLE_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_CPUS_PER_TASK",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
)
SCHEMA_VERSION_KEY = "schema_version"
CASE_NAME_KEY = "case_name"
MODEL_KEY = "model"
REPEAT_COUNT_KEY = "repeat_count"
DISABLED_SUCCESS_COUNT_KEY = "disabled_success_count"
ENABLED_SUCCESS_COUNT_KEY = "enabled_success_count"
BASELINE_MEAN_SECONDS_KEY = "baseline_mean_seconds"
ENABLED_MEAN_SECONDS_KEY = "enabled_mean_seconds"
BASELINE_TIMES_SECONDS_KEY = "baseline_times_seconds"
ENABLED_TIMES_SECONDS_KEY = "enabled_times_seconds"
BASELINE_SAMPLE_VARIANCE_SECONDS_KEY = "baseline_sample_variance_seconds"
ENABLED_SAMPLE_VARIANCE_SECONDS_KEY = "enabled_sample_variance_seconds"
BASELINE_SAMPLE_STDDEV_SECONDS_KEY = "baseline_sample_stddev_seconds"
ENABLED_SAMPLE_STDDEV_SECONDS_KEY = "enabled_sample_stddev_seconds"
SPEEDUP_VS_DISABLED_CACHE_KEY = "speedup_vs_disabled_cache"
COMMANDS_KEY = "commands"
LOGS_DIR_NAME = "logs"
CASES_DIR_NAME = "cases"
TABLES_DIR_NAME = "tables"
FIGURES_DIR_NAME = "figures"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
HASH_CHUNK_BYTES = DOWNLOAD_CHUNK_BYTES
PERCENT_SCALE = 100.0
SUCCESS_RETURN_CODE = 0
COMMAND_TIMEOUT_RETURN_CODE = 124
MIN_REQUIRED_CASE_COUNT = 1
MIN_REQUIRED_TRACE_COUNT = 1
MIN_POSITIVE_VALUE = 0.0
MIN_NONNEGATIVE_VALUE = 0.0
MIN_PERCENT_VALUE = 0.0
MAX_PERCENT_VALUE = 100.0
MIN_CORRELATION_SAMPLE_COUNT = 2
MIN_SAMPLE_VARIANCE_COUNT = 2
SAMPLE_VARIANCE_DEGREES_OF_FREEDOM = 1
NORMAL_APPROX_95_CI_MULTIPLIER = 1.96
TIMING_ABSOLUTE_TOLERANCE_SECONDS = 1.0e-12
TIMING_RELATIVE_TOLERANCE = 1.0e-9
SHA256_HEX_LENGTH = 64
SHA256_HEX_PATTERN = re.compile(rf"^[0-9a-fA-F]{{{SHA256_HEX_LENGTH}}}$")
SVG_WIDTH = 960
SVG_HEIGHT = 540
SVG_MARGIN_LEFT = 88
SVG_MARGIN_RIGHT = 40
SVG_MARGIN_TOP = 56
SVG_MARGIN_BOTTOM = 88
BAR_GAP_RATIO = 0.28
SCATTER_POINT_RADIUS = 5
MODEL_NAME_JOINER = ", "
NVIDIA_SMI_TIMEOUT_SECONDS = 20.0
TORCH_GPU_TIMEOUT_SECONDS = 30.0
GIT_METADATA_TIMEOUT_SECONDS = 10.0
SEVENNET_DISABLE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
SEVENNET_PROFILE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
SEVENNET_PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
ENV_FLAG_ENABLED = "1"
BASELINE_CASE_NAME = "baseline-disabled"
ISODELTA_CASE_NAME = "isodelta-enabled"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

BENCHMARK_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
BUNDLE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_evidence_bundle.py"
TRACE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
EXPERIMENT_DRIVER_PATH = REPO_ROOT / "tools" / "run_isodelta_experiment.py"


class ClusterSuiteError(ValueError):
    """Raised when the cluster paper suite cannot produce auditable outputs."""


@dataclass(frozen=True)
class ArtifactConfig:
    """Describe one dataset, checkpoint, or input bundle needed by the suite."""

    name: str
    path: Path
    url: str | None = None
    sha256: str | None = None
    required: bool = True
    required_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseConfig:
    """Describe one model case in the paper experiment matrix."""

    name: str
    model: str
    kind: str
    lammps_command: str | None = None
    input_path: Path | None = None
    work_dir: Path | None = None
    lammps_root: Path | None = None
    disabled_command: str | None = None
    enabled_command: str | None = None
    trace_command: str | None = None
    preflight_command: str | None = None
    trace_input: Path | None = None
    benchmark_report: Path | None = None
    bundle_evidence: Path | None = None
    external_timing_report: Path | None = None
    trace_evidence_paths: tuple[Path, ...] = ()
    required_trace_models: tuple[str, ...] = ()
    repeat_count: int = DEFAULT_REPEAT_COUNT
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    binary_timeout_seconds: float = DEFAULT_BINARY_TIMEOUT_SECONDS
    benchmark_timeout_seconds: float = DEFAULT_BENCHMARK_TIMEOUT_SECONDS
    max_abs_thermo_delta: float = DEFAULT_MAX_ABS_THERMO_DELTA
    min_paired_thermo_count: int = DEFAULT_MIN_PAIRED_THERMO_COUNT
    min_speedup: float | None = DEFAULT_MIN_SPEEDUP
    min_speedup_95ci_lower_bound: float | None = DEFAULT_MIN_SPEEDUP_95CI_LOWER_BOUND
    min_hit_rate_percent: float = DEFAULT_MIN_HIT_RATE_PERCENT
    min_enabled_cache_attempts: int = DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS
    min_enabled_cache_hits: int = DEFAULT_MIN_ENABLED_CACHE_HITS
    min_trace_hit_rate_percent: float = DEFAULT_MIN_TRACE_HIT_RATE_PERCENT
    min_trace_estimated_speedup: float | None = None
    min_trace_metadata_fraction_percent: float = (
        DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT
    )
    preflight_timeout_seconds: float = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
    preflight_env: dict[str, str] = field(default_factory=dict)
    disabled_env: dict[str, str] = field(default_factory=dict)
    enabled_env: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteConfig:
    """Store the full manifest after path and default resolution."""

    name: str
    manifest_path: Path
    output_dir: Path
    expected_gpus: int = DEFAULT_EXPECTED_GPU_COUNT
    required_models: tuple[str, ...] = DEFAULT_REQUIRED_MODELS
    min_trace_count: int = DEFAULT_MIN_TRACE_COUNT
    min_distinct_trace_models: int = DEFAULT_MIN_DISTINCT_TRACE_MODELS
    require_artifact_sha256: bool = DEFAULT_REQUIRE_ARTIFACT_SHA256
    artifacts: tuple[ArtifactConfig, ...] = ()
    cases: tuple[CaseConfig, ...] = ()


@dataclass(frozen=True)
class CommandRecord:
    """Record one launched command for the suite report."""

    name: str
    command: str | list[str]
    returncode: int
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str


@dataclass(frozen=True)
class CaseSummary:
    """Represent one model row in the generated paper tables."""

    case_name: str
    model: str
    kind: str
    status: str
    benchmark_report: str | None
    bundle_evidence: str | None
    trace_evidence: tuple[str, ...]
    external_timing_report: str | None
    baseline_mean_seconds: float | None
    enabled_mean_seconds: float | None
    baseline_sample_variance_seconds: float | None
    enabled_sample_variance_seconds: float | None
    baseline_sample_stddev_seconds: float | None
    enabled_sample_stddev_seconds: float | None
    baseline_timing_count: int | None
    enabled_timing_count: int | None
    baseline_mean_95ci_half_width_seconds: float | None
    enabled_mean_95ci_half_width_seconds: float | None
    speedup_vs_disabled_cache: float | None
    speedup_95ci_lower_bound: float | None
    speedup_95ci_upper_bound: float | None
    cache_attempts: float | None
    cache_hits: float | None
    cache_hit_rate_percent: float | None
    max_abs_thermo_delta: float | None
    trace_hit_rate_percent: float | None
    trace_estimated_average_speedup: float | None
    trace_estimated_worst_case_speedup: float | None
    trace_metadata_fraction_percent: float | None


def _load_module(module_name: str, module_path: Path) -> Any:
    """Load a sibling validation module without turning tools/ into a package."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ClusterSuiteError(f"cannot load validation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark_check = _load_module("cluster_suite_benchmark_check", BENCHMARK_CHECK_PATH)
bundle_check = _load_module("cluster_suite_bundle_check", BUNDLE_CHECK_PATH)
trace_check = _load_module("cluster_suite_trace_check", TRACE_CHECK_PATH)


def _require(condition: bool, message: str) -> None:
    """Raise a concise suite error when an input or artifact is invalid."""
    if not condition:
        raise ClusterSuiteError(message)


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a TOML object with a schema-oriented error."""
    _require(isinstance(value, dict), f"{field_name} must be a table")
    return value


def _as_json_object(value: Any, field_name: str) -> dict[str, Any]:
    """Return a JSON object field with a schema-oriented error."""
    _require(isinstance(value, dict), f"{field_name} must be a JSON object")
    return value


def _as_string(value: Any, field_name: str) -> str:
    """Return a non-empty manifest string."""
    _require(isinstance(value, str), f"{field_name} must be a string")
    _require(bool(value.strip()), f"{field_name} must not be empty")
    return value.strip()


def _as_json_string(value: Any, field_name: str) -> str:
    """Return a non-empty JSON string field."""
    _require(isinstance(value, str), f"{field_name} must be a string")
    _require(bool(value.strip()), f"{field_name} must not be empty")
    return value.strip()


def _as_json_number(value: Any, field_name: str) -> float:
    """Return a finite JSON number without accepting boolean aliases."""
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    numeric_value = float(value)
    _require(math.isfinite(numeric_value), f"{field_name} must be finite")
    return numeric_value


def _as_json_positive_number(value: Any, field_name: str) -> float:
    """Return a strictly positive finite JSON number."""
    numeric_value = _as_json_number(value, field_name)
    _require(numeric_value > MIN_POSITIVE_VALUE, f"{field_name} must be positive")
    return numeric_value


def _as_json_optional_nonnegative_number(value: Any, field_name: str) -> float | None:
    """Return an optional nonnegative finite JSON number."""
    if value is None:
        return None
    numeric_value = _as_json_number(value, field_name)
    _require(numeric_value >= MIN_NONNEGATIVE_VALUE, f"{field_name} must be nonnegative")
    return numeric_value


def _as_json_positive_number_list(value: Any, field_name: str) -> list[float]:
    """Return a JSON array of positive finite timing values."""
    _require(isinstance(value, list), f"{field_name} must be a JSON array")
    return [
        _as_json_positive_number(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    ]


def _as_json_nonnegative_int(value: Any, field_name: str) -> int:
    """Return a whole nonnegative JSON count."""
    numeric_value = _as_json_number(value, field_name)
    int_value = int(numeric_value)
    _require(
        numeric_value == int_value and int_value >= 0,
        f"{field_name} must be a nonnegative integer",
    )
    return int_value


def _as_optional_string(value: Any, field_name: str) -> str | None:
    """Return an optional non-empty manifest string."""
    if value is None:
        return None
    return _as_string(value, field_name)


def _as_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Return a tuple of non-empty manifest strings."""
    if value is None:
        return ()
    _require(isinstance(value, list), f"{field_name} must be an array")
    return tuple(_as_string(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _as_bool(value: Any, field_name: str, default: bool) -> bool:
    """Return an optional manifest boolean."""
    if value is None:
        return default
    _require(isinstance(value, bool), f"{field_name} must be a boolean")
    return value


def _as_int(value: Any, field_name: str, default: int) -> int:
    """Return an optional manifest integer."""
    if value is None:
        return default
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field_name} must be an integer")
    return value


def _as_float(value: Any, field_name: str, default: float) -> float:
    """Return an optional finite manifest number as a float."""
    if value is None:
        return default
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    numeric_value = float(value)
    _require(math.isfinite(numeric_value), f"{field_name} must be finite")
    return numeric_value


def _as_optional_float(value: Any, field_name: str, default: float | None) -> float | None:
    """Return an optional finite manifest number."""
    if value is None:
        return default
    return _as_float(value, field_name, DEFAULT_MIN_SPEEDUP)


def _as_env_mapping(value: Any, field_name: str) -> dict[str, str]:
    """Return a string-to-string environment override mapping."""
    if value is None:
        return {}
    mapping = _as_mapping(value, field_name)
    return {
        _as_string(key, f"{field_name}.key"): _as_string(raw_value, f"{field_name}.{key}")
        for key, raw_value in mapping.items()
    }


def _resolve_path(raw_path: str | Path | None, base_dir: Path) -> Path | None:
    """Resolve manifest paths relative to the manifest directory."""
    if raw_path is None:
        return None
    path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


def _safe_name(value: str) -> str:
    """Return a filesystem-safe label for case directories and log files."""
    safe_value = SAFE_NAME_PATTERN.sub("_", value.strip())
    return safe_value.strip("._") or "case"


def _validate_percent(value: float, field_name: str) -> None:
    """Require percentages to stay within their physical range."""
    _require(
        MIN_PERCENT_VALUE <= value <= MAX_PERCENT_VALUE,
        f"{field_name} must be between {MIN_PERCENT_VALUE:g} and {MAX_PERCENT_VALUE:g}",
    )


def _validate_case_thresholds(case: CaseConfig) -> None:
    """Reject case gates that cannot support a paper claim."""
    _require(case.repeat_count >= MIN_REQUIRED_CASE_COUNT, f"{case.name}: repeat_count must be at least one")
    _require(
        case.preflight_timeout_seconds > MIN_POSITIVE_VALUE,
        f"{case.name}: preflight_timeout_seconds must be positive",
    )
    _require(case.command_timeout_seconds > MIN_POSITIVE_VALUE, f"{case.name}: command_timeout_seconds must be positive")
    _require(case.binary_timeout_seconds > MIN_POSITIVE_VALUE, f"{case.name}: binary_timeout_seconds must be positive")
    _require(case.benchmark_timeout_seconds > MIN_POSITIVE_VALUE, f"{case.name}: benchmark_timeout_seconds must be positive")
    _require(case.max_abs_thermo_delta >= MIN_NONNEGATIVE_VALUE, f"{case.name}: max_abs_thermo_delta must be nonnegative")
    _require(case.min_paired_thermo_count >= MIN_REQUIRED_TRACE_COUNT, f"{case.name}: min_paired_thermo_count must be at least one")
    _require(case.min_enabled_cache_attempts >= 0, f"{case.name}: min_enabled_cache_attempts must be nonnegative")
    _require(case.min_enabled_cache_hits >= 0, f"{case.name}: min_enabled_cache_hits must be nonnegative")
    _require(
        case.min_enabled_cache_hits <= case.min_enabled_cache_attempts,
        f"{case.name}: min_enabled_cache_hits cannot exceed min_enabled_cache_attempts",
    )
    if case.min_speedup is not None:
        _require(case.min_speedup > MIN_POSITIVE_VALUE, f"{case.name}: min_speedup must be positive")
    if case.min_speedup_95ci_lower_bound is not None:
        _require(
            case.min_speedup_95ci_lower_bound > MIN_POSITIVE_VALUE,
            f"{case.name}: min_speedup_95ci_lower_bound must be positive",
        )
    if case.min_trace_estimated_speedup is not None:
        _require(
            case.min_trace_estimated_speedup > MIN_POSITIVE_VALUE,
            f"{case.name}: min_trace_estimated_speedup must be positive",
        )
    _validate_percent(case.min_hit_rate_percent, f"{case.name}: min_hit_rate_percent")
    _validate_percent(case.min_trace_hit_rate_percent, f"{case.name}: min_trace_hit_rate_percent")
    _validate_percent(
        case.min_trace_metadata_fraction_percent,
        f"{case.name}: min_trace_metadata_fraction_percent",
    )


def _validate_optional_sha256(value: str | None, field_name: str) -> None:
    """Reject missing or malformed SHA-256 digests when a field is present."""
    if value is None:
        return
    _require(
        bool(SHA256_HEX_PATTERN.fullmatch(value)),
        f"{field_name} must be a {SHA256_HEX_LENGTH}-character hexadecimal SHA-256 digest",
    )


def load_manifest(manifest_path: Path) -> SuiteConfig:
    """Load a TOML manifest and resolve all relative paths."""
    manifest_path = manifest_path.resolve()
    base_dir = manifest_path.parent
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    suite_payload = _as_mapping(payload.get("suite"), "suite")
    suite_name = _as_string(suite_payload.get("name", "isodelta-paper-suite"), "suite.name")
    resolved_output_dir = _resolve_path(
        _as_optional_string(suite_payload.get("output_dir"), "suite.output_dir")
        or str(DEFAULT_OUTPUT_DIR),
        base_dir,
    )
    _require(resolved_output_dir is not None, "suite.output_dir could not be resolved")

    artifact_configs: list[ArtifactConfig] = []
    for index, artifact_payload in enumerate(payload.get("artifacts", [])):
        artifact = _as_mapping(artifact_payload, f"artifacts[{index}]")
        artifact_path = _resolve_path(
            _as_string(artifact.get("path"), f"artifacts[{index}].path"),
            base_dir,
        )
        _require(artifact_path is not None, f"artifacts[{index}].path could not be resolved")
        artifact_configs.append(
            ArtifactConfig(
                name=_as_string(artifact.get("name"), f"artifacts[{index}].name"),
                path=artifact_path,
                url=_as_optional_string(artifact.get("url"), f"artifacts[{index}].url"),
                sha256=_as_optional_string(
                    artifact.get("sha256"),
                    f"artifacts[{index}].sha256",
                ),
                required=_as_bool(artifact.get("required"), f"artifacts[{index}].required", True),
                required_by=_as_string_tuple(
                    artifact.get("required_by"),
                    f"artifacts[{index}].required_by",
                ),
            )
        )

    default_repeat_count = _as_int(
        suite_payload.get("repeat_count"),
        "suite.repeat_count",
        DEFAULT_REPEAT_COUNT,
    )
    default_command_timeout_seconds = _as_float(
        suite_payload.get("command_timeout_seconds"),
        "suite.command_timeout_seconds",
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    default_binary_timeout_seconds = _as_float(
        suite_payload.get("binary_timeout_seconds"),
        "suite.binary_timeout_seconds",
        DEFAULT_BINARY_TIMEOUT_SECONDS,
    )
    default_benchmark_timeout_seconds = _as_float(
        suite_payload.get("benchmark_timeout_seconds"),
        "suite.benchmark_timeout_seconds",
        DEFAULT_BENCHMARK_TIMEOUT_SECONDS,
    )
    default_min_speedup = _as_optional_float(
        suite_payload.get("min_speedup"),
        "suite.min_speedup",
        DEFAULT_MIN_SPEEDUP,
    )
    default_min_speedup_95ci_lower_bound = _as_optional_float(
        suite_payload.get("min_speedup_95ci_lower_bound"),
        "suite.min_speedup_95ci_lower_bound",
        DEFAULT_MIN_SPEEDUP_95CI_LOWER_BOUND,
    )
    default_min_hit_rate_percent = _as_float(
        suite_payload.get("min_hit_rate_percent"),
        "suite.min_hit_rate_percent",
        DEFAULT_MIN_HIT_RATE_PERCENT,
    )
    default_min_trace_hit_rate_percent = _as_float(
        suite_payload.get("min_trace_hit_rate_percent"),
        "suite.min_trace_hit_rate_percent",
        DEFAULT_MIN_TRACE_HIT_RATE_PERCENT,
    )
    default_min_trace_estimated_speedup = _as_optional_float(
        suite_payload.get("min_trace_estimated_speedup"),
        "suite.min_trace_estimated_speedup",
        None,
    )
    default_min_trace_metadata_fraction_percent = _as_float(
        suite_payload.get("min_trace_metadata_fraction_percent"),
        "suite.min_trace_metadata_fraction_percent",
        DEFAULT_MIN_TRACE_METADATA_FRACTION_PERCENT,
    )

    case_configs: list[CaseConfig] = []
    case_payloads = payload.get("cases", [])
    _require(isinstance(case_payloads, list), "cases must be an array of tables")
    for index, case_payload in enumerate(case_payloads):
        case = _as_mapping(case_payload, f"cases[{index}]")
        case_name = _as_string(case.get("name"), f"cases[{index}].name")
        trace_input = _resolve_path(
            _as_optional_string(case.get("trace_input"), f"cases[{index}].trace_input"),
            base_dir,
        )
        benchmark_report = _resolve_path(
            _as_optional_string(case.get("benchmark_report"), f"cases[{index}].benchmark_report"),
            base_dir,
        )
        bundle_evidence = _resolve_path(
            _as_optional_string(case.get("bundle_evidence"), f"cases[{index}].bundle_evidence"),
            base_dir,
        )
        external_timing_report = _resolve_path(
            _as_optional_string(
                case.get("external_timing_report"),
                f"cases[{index}].external_timing_report",
            ),
            base_dir,
        )
        trace_evidence_paths = tuple(
            path
            for raw_path in _as_string_tuple(
                case.get("trace_evidence"),
                f"cases[{index}].trace_evidence",
            )
            for path in [_resolve_path(raw_path, base_dir)]
            if path is not None
        )
        input_path = _resolve_path(
            _as_optional_string(case.get("input"), f"cases[{index}].input"),
            base_dir,
        )
        work_dir = _resolve_path(
            _as_optional_string(case.get("work_dir"), f"cases[{index}].work_dir"),
            base_dir,
        )
        lammps_root = _resolve_path(
            _as_optional_string(case.get("lammps_root"), f"cases[{index}].lammps_root"),
            base_dir,
        )
        case_configs.append(
            CaseConfig(
                name=case_name,
                model=_as_string(case.get("model"), f"cases[{index}].model"),
                kind=_as_string(case.get("kind"), f"cases[{index}].kind"),
                lammps_command=_as_optional_string(
                    case.get("lammps_command"),
                    f"cases[{index}].lammps_command",
                ),
                input_path=input_path,
                work_dir=work_dir,
                lammps_root=lammps_root,
                disabled_command=_as_optional_string(
                    case.get("disabled_command"),
                    f"cases[{index}].disabled_command",
                ),
                enabled_command=_as_optional_string(
                    case.get("enabled_command"),
                    f"cases[{index}].enabled_command",
                ),
                trace_command=_as_optional_string(
                    case.get("trace_command"),
                    f"cases[{index}].trace_command",
                ),
                preflight_command=_as_optional_string(
                    case.get("preflight_command"),
                    f"cases[{index}].preflight_command",
                ),
                trace_input=trace_input,
                benchmark_report=benchmark_report,
                bundle_evidence=bundle_evidence,
                external_timing_report=external_timing_report,
                trace_evidence_paths=trace_evidence_paths,
                required_trace_models=_as_string_tuple(
                    case.get("required_trace_models"),
                    f"cases[{index}].required_trace_models",
                ),
                repeat_count=_as_int(
                    case.get("repeat_count"),
                    f"cases[{index}].repeat_count",
                    default_repeat_count,
                ),
                command_timeout_seconds=_as_float(
                    case.get("command_timeout_seconds"),
                    f"cases[{index}].command_timeout_seconds",
                    default_command_timeout_seconds,
                ),
                binary_timeout_seconds=_as_float(
                    case.get("binary_timeout_seconds"),
                    f"cases[{index}].binary_timeout_seconds",
                    default_binary_timeout_seconds,
                ),
                benchmark_timeout_seconds=_as_float(
                    case.get("benchmark_timeout_seconds"),
                    f"cases[{index}].benchmark_timeout_seconds",
                    default_benchmark_timeout_seconds,
                ),
                max_abs_thermo_delta=_as_float(
                    case.get("max_abs_thermo_delta"),
                    f"cases[{index}].max_abs_thermo_delta",
                    DEFAULT_MAX_ABS_THERMO_DELTA,
                ),
                min_paired_thermo_count=_as_int(
                    case.get("min_paired_thermo_count"),
                    f"cases[{index}].min_paired_thermo_count",
                    DEFAULT_MIN_PAIRED_THERMO_COUNT,
                ),
                min_speedup=_as_optional_float(
                    case.get("min_speedup"),
                    f"cases[{index}].min_speedup",
                    default_min_speedup,
                ),
                min_speedup_95ci_lower_bound=_as_optional_float(
                    case.get("min_speedup_95ci_lower_bound"),
                    f"cases[{index}].min_speedup_95ci_lower_bound",
                    default_min_speedup_95ci_lower_bound,
                ),
                min_hit_rate_percent=_as_float(
                    case.get("min_hit_rate_percent"),
                    f"cases[{index}].min_hit_rate_percent",
                    default_min_hit_rate_percent,
                ),
                min_enabled_cache_attempts=_as_int(
                    case.get("min_enabled_cache_attempts"),
                    f"cases[{index}].min_enabled_cache_attempts",
                    DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS,
                ),
                min_enabled_cache_hits=_as_int(
                    case.get("min_enabled_cache_hits"),
                    f"cases[{index}].min_enabled_cache_hits",
                    DEFAULT_MIN_ENABLED_CACHE_HITS,
                ),
                min_trace_hit_rate_percent=_as_float(
                    case.get("min_trace_hit_rate_percent"),
                    f"cases[{index}].min_trace_hit_rate_percent",
                    default_min_trace_hit_rate_percent,
                ),
                min_trace_estimated_speedup=_as_optional_float(
                    case.get("min_trace_estimated_speedup"),
                    f"cases[{index}].min_trace_estimated_speedup",
                    default_min_trace_estimated_speedup,
                ),
                min_trace_metadata_fraction_percent=_as_float(
                    case.get("min_trace_metadata_fraction_percent"),
                    f"cases[{index}].min_trace_metadata_fraction_percent",
                    default_min_trace_metadata_fraction_percent,
                ),
                preflight_timeout_seconds=_as_float(
                    case.get("preflight_timeout_seconds"),
                    f"cases[{index}].preflight_timeout_seconds",
                    DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
                ),
                preflight_env=_as_env_mapping(
                    case.get("preflight_env"),
                    f"cases[{index}].preflight_env",
                ),
                disabled_env=_as_env_mapping(
                    case.get("disabled_env"),
                    f"cases[{index}].disabled_env",
                ),
                enabled_env=_as_env_mapping(
                    case.get("enabled_env"),
                    f"cases[{index}].enabled_env",
                ),
                artifacts=_as_string_tuple(case.get("artifacts"), f"cases[{index}].artifacts"),
            )
        )

    return SuiteConfig(
        name=suite_name,
        manifest_path=manifest_path,
        output_dir=resolved_output_dir,
        expected_gpus=_as_int(
            suite_payload.get("expected_gpus"),
            "suite.expected_gpus",
            DEFAULT_EXPECTED_GPU_COUNT,
        ),
        required_models=_as_string_tuple(
            suite_payload.get("required_models"),
            "suite.required_models",
        )
        or DEFAULT_REQUIRED_MODELS,
        min_trace_count=_as_int(
            suite_payload.get("min_trace_count"),
            "suite.min_trace_count",
            DEFAULT_MIN_TRACE_COUNT,
        ),
        min_distinct_trace_models=_as_int(
            suite_payload.get("min_distinct_trace_models"),
            "suite.min_distinct_trace_models",
            DEFAULT_MIN_DISTINCT_TRACE_MODELS,
        ),
        require_artifact_sha256=_as_bool(
            suite_payload.get("require_artifact_sha256"),
            "suite.require_artifact_sha256",
            DEFAULT_REQUIRE_ARTIFACT_SHA256,
        ),
        artifacts=tuple(artifact_configs),
        cases=tuple(case_configs),
    )


def validate_suite_config(config: SuiteConfig) -> None:
    """Reject manifests that cannot cover the requested paper matrix."""
    _require(config.expected_gpus >= MIN_REQUIRED_CASE_COUNT, "suite.expected_gpus must be at least one")
    _require(config.min_trace_count >= MIN_REQUIRED_TRACE_COUNT, "suite.min_trace_count must be at least one")
    _require(
        config.min_distinct_trace_models >= MIN_REQUIRED_TRACE_COUNT,
        "suite.min_distinct_trace_models must be at least one",
    )
    _require(len(config.cases) >= MIN_REQUIRED_CASE_COUNT, "manifest must define at least one case")
    artifact_names = [artifact.name for artifact in config.artifacts]
    duplicate_artifacts = sorted({name for name in artifact_names if artifact_names.count(name) > 1})
    _require(not duplicate_artifacts, "duplicate artifact names: " + MODEL_NAME_JOINER.join(duplicate_artifacts))
    case_names = [case.name for case in config.cases]
    duplicate_cases = sorted({name for name in case_names if case_names.count(name) > 1})
    _require(not duplicate_cases, "duplicate case names: " + MODEL_NAME_JOINER.join(duplicate_cases))

    present_models = {case.model for case in config.cases}
    missing_models = [model for model in config.required_models if model not in present_models]
    _require(
        not missing_models,
        "manifest is missing required model cases: " + MODEL_NAME_JOINER.join(missing_models),
    )
    artifact_name_set = set(artifact_names)
    for artifact in config.artifacts:
        _validate_optional_sha256(artifact.sha256, f"{artifact.name}: sha256")
        _require(
            not (config.require_artifact_sha256 and artifact.required and artifact.sha256 is None),
            f"{artifact.name}: required artifact needs sha256 because suite.require_artifact_sha256 is true",
        )
        unknown_required_by = [
            model_name
            for model_name in artifact.required_by
            if model_name not in present_models
        ]
        _require(
            not unknown_required_by,
            f"{artifact.name}: required_by names unknown case models: "
            + MODEL_NAME_JOINER.join(unknown_required_by),
        )
    for case in config.cases:
        _require(case.kind in SUPPORTED_CASE_KINDS, f"{case.name}: unsupported kind {case.kind}")
        _validate_case_thresholds(case)
        missing_artifacts = [artifact for artifact in case.artifacts if artifact not in artifact_name_set]
        _require(
            not missing_artifacts,
            f"{case.name}: unknown artifacts: " + MODEL_NAME_JOINER.join(missing_artifacts),
        )
        if case.kind == "sevennet_lammps":
            _require(bool(case.lammps_command), f"{case.name}: lammps_command is required")
            _require(case.input_path is not None, f"{case.name}: input is required")
        elif case.kind == "external_pair":
            _require(bool(case.disabled_command), f"{case.name}: disabled_command is required")
            _require(bool(case.enabled_command), f"{case.name}: enabled_command is required")
        elif case.kind == "trace_only":
            _require(
                bool(case.trace_command) or case.trace_input is not None or bool(case.trace_evidence_paths),
                f"{case.name}: trace_command, trace_input, or trace_evidence is required",
            )
        if case.trace_evidence_paths:
            normalized_trace_paths = [
                os.path.normcase(str(path.resolve(strict=False)))
                for path in case.trace_evidence_paths
            ]
            duplicate_trace_paths = sorted(
                {path for path in normalized_trace_paths if normalized_trace_paths.count(path) > 1}
            )
            _require(
                not duplicate_trace_paths,
                f"{case.name}: duplicate trace evidence paths: "
                + MODEL_NAME_JOINER.join(duplicate_trace_paths),
            )


def _readiness_record(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """Create one machine-readable final-paper readiness check row."""
    return {"name": name, "passed": passed, "detail": detail}


def _has_unresolved_template_marker(value: str | None) -> bool:
    """Return whether a manifest field still contains a template marker."""
    if value is None:
        return False
    lowered_value = value.lower()
    return any(marker in lowered_value for marker in UNRESOLVED_TEMPLATE_MARKERS)


def build_readiness_report(config: SuiteConfig) -> dict[str, Any]:
    """Audit whether a manifest is strict enough for final paper execution."""
    checks: list[dict[str, Any]] = []
    try:
        validate_suite_config(config)
    except ClusterSuiteError as exc:
        checks.append(_readiness_record("manifest_schema", False, str(exc)))
    else:
        checks.append(_readiness_record("manifest_schema", True, "manifest validation passed"))

    missing_required_models = [
        model for model in FINAL_PAPER_REQUIRED_MODELS if model not in config.required_models
    ]
    checks.append(
        _readiness_record(
            "foundation_models_required",
            not missing_required_models,
            "missing required models: " + MODEL_NAME_JOINER.join(missing_required_models)
            if missing_required_models
            else "SevenNet, MACE, and NequIP are required",
        )
    )

    paired_cases = [
        case
        for case in config.cases
        if case.model in FINAL_PAPER_REQUIRED_MODELS
        and case.kind in FINAL_PAPER_PAIRED_CASE_KINDS
    ]
    paired_models = {case.model for case in paired_cases}
    missing_paired_models = [
        model for model in FINAL_PAPER_REQUIRED_MODELS if model not in paired_models
    ]
    checks.append(
        _readiness_record(
            "paired_enabled_disabled_cases",
            not missing_paired_models,
            "missing paired cases: " + MODEL_NAME_JOINER.join(missing_paired_models)
            if missing_paired_models
            else "all required models have paired enabled/disabled cases",
        )
    )

    checks.append(
        _readiness_record(
            "expected_gpu_count",
            config.expected_gpus >= DEFAULT_EXPECTED_GPU_COUNT,
            f"expected_gpus={config.expected_gpus}, required>={DEFAULT_EXPECTED_GPU_COUNT}",
        )
    )
    checks.append(
        _readiness_record(
            "artifact_sha256_gate",
            config.require_artifact_sha256,
            "suite.require_artifact_sha256 is enabled"
            if config.require_artifact_sha256
            else "suite.require_artifact_sha256 must be true",
        )
    )

    required_artifacts = [artifact for artifact in config.artifacts if artifact.required]
    checks.append(
        _readiness_record(
            "required_artifacts_declared",
            bool(required_artifacts),
            "required artifact count: " + str(len(required_artifacts)),
        )
    )
    missing_artifact_sources = [
        artifact.name
        for artifact in required_artifacts
        if not artifact.path.exists() and artifact.url is None
    ]
    checks.append(
        _readiness_record(
            "required_artifacts_materializable",
            not missing_artifact_sources,
            "missing path and URL: " + MODEL_NAME_JOINER.join(missing_artifact_sources)
            if missing_artifact_sources
            else "required artifacts exist locally or have a download URL",
        )
    )

    unresolved_artifact_fields: list[str] = []
    for artifact in config.artifacts:
        artifact_fields = {
            "path": str(artifact.path),
            "url": artifact.url,
            "sha256": artifact.sha256,
        }
        unresolved_artifact_fields.extend(
            f"{artifact.name}.{field_name}"
            for field_name, field_value in artifact_fields.items()
            if _has_unresolved_template_marker(field_value)
        )
    checks.append(
        _readiness_record(
            "artifact_template_markers_removed",
            not unresolved_artifact_fields,
            "unresolved fields: " + MODEL_NAME_JOINER.join(unresolved_artifact_fields)
            if unresolved_artifact_fields
            else "artifact fields contain no unresolved template markers",
        )
    )

    missing_preflight_cases = [case.name for case in paired_cases if not case.preflight_command]
    checks.append(
        _readiness_record(
            "paired_case_preflights",
            not missing_preflight_cases,
            "missing preflight_command: " + MODEL_NAME_JOINER.join(missing_preflight_cases)
            if missing_preflight_cases
            else "paired cases define preflight commands",
        )
    )
    low_repeat_cases = [
        case.name for case in paired_cases if case.repeat_count < FINAL_PAPER_MIN_REPEAT_COUNT
    ]
    checks.append(
        _readiness_record(
            "paired_case_repeats",
            not low_repeat_cases,
            "repeat_count below "
            + str(FINAL_PAPER_MIN_REPEAT_COUNT)
            + ": "
            + MODEL_NAME_JOINER.join(low_repeat_cases)
            if low_repeat_cases
            else f"paired cases repeat at least {FINAL_PAPER_MIN_REPEAT_COUNT} times",
        )
    )
    missing_uncertainty_gate_cases = [
        case.name for case in paired_cases if case.min_speedup_95ci_lower_bound is None
    ]
    checks.append(
        _readiness_record(
            "paired_case_uncertainty_gate",
            not missing_uncertainty_gate_cases,
            "missing min_speedup_95ci_lower_bound: "
            + MODEL_NAME_JOINER.join(missing_uncertainty_gate_cases)
            if missing_uncertainty_gate_cases
            else "paired cases require conservative speedup bounds",
        )
    )

    ready = all(check["passed"] for check in checks)
    return {
        "readiness_schema_version": READINESS_SCHEMA_VERSION,
        "status": "ready" if ready else "failed",
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "output_dir": str(config.output_dir),
            "expected_gpus": config.expected_gpus,
            "required_models": list(config.required_models),
            "final_paper_required_models": list(FINAL_PAPER_REQUIRED_MODELS),
        },
        "checks": checks,
    }


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


def collect_run_provenance() -> dict[str, Any]:
    """Collect enough context to reproduce the generated paper artifacts."""
    git_status_short = _run_metadata_command(["git", "status", "--short"])
    return {
        "suite_schema_version": SUITE_SCHEMA_VERSION,
        "generated_at": time.strftime(TIMESTAMP_FORMAT),
        "git_commit": _run_metadata_command(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_metadata_command(["git", "branch", "--show-current"]),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _package_version(package_name: str) -> str | None:
    """Return an installed package version without importing heavy frameworks."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_nvidia_smi_snapshot() -> dict[str, Any]:
    """Collect lightweight GPU identity details when nvidia-smi is available."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "available": False,
            "path": None,
            "query": None,
            "rows": [],
        }
    query_fields = "index,name,driver_version,memory.total"
    query_output = _run_metadata_command(
        [
            nvidia_smi,
            f"--query-gpu={query_fields}",
            "--format=csv,noheader",
        ]
    )
    return {
        "available": query_output is not None,
        "path": nvidia_smi,
        "query": query_fields,
        "rows": [] if query_output is None else query_output.splitlines(),
    }


def collect_environment_snapshot(
    config: SuiteConfig,
    gpu_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect reproducibility context that should travel with paper outputs."""
    return {
        "snapshot_schema_version": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        "suite_name": config.name,
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "provenance": collect_run_provenance(),
        "gpu_check": gpu_record,
        "package_versions": {
            package_name: _package_version(package_name)
            for package_name in ENVIRONMENT_PACKAGE_NAMES
        },
        "selected_environment": {
            variable_name: os.environ.get(variable_name)
            for variable_name in ENVIRONMENT_VARIABLE_NAMES
        },
        "nvidia_smi": collect_nvidia_smi_snapshot(),
    }


def write_environment_snapshot(
    config: SuiteConfig,
    gpu_record: dict[str, Any] | None,
) -> Path:
    """Write the environment snapshot next to tables and figures."""
    snapshot_path = config.output_dir / ENVIRONMENT_SNAPSHOT_NAME
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_environment_snapshot(config, gpu_record)
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return snapshot_path


def detect_gpu_count() -> tuple[int | None, str]:
    """Detect visible NVIDIA GPUs with nvidia-smi, then fall back to torch."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "-L"],
                text=True,
                capture_output=True,
                check=False,
                timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"nvidia-smi failed: {exc}"
        if completed.returncode == SUCCESS_RETURN_CODE:
            gpu_lines = [
                line
                for line in completed.stdout.splitlines()
                if line.strip().startswith("GPU ")
            ]
            return len(gpu_lines), "nvidia-smi"
        return None, completed.stderr.strip() or "nvidia-smi returned a nonzero status"

    torch_probe = (
        "import torch; "
        "print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", torch_probe],
            text=True,
            capture_output=True,
            check=False,
            timeout=TORCH_GPU_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"torch GPU probe failed: {exc}"
    if completed.returncode != SUCCESS_RETURN_CODE:
        return None, completed.stderr.strip() or "torch GPU probe returned a nonzero status"
    try:
        return int(completed.stdout.strip()), "torch"
    except ValueError:
        return None, f"torch GPU probe returned non-integer output: {completed.stdout!r}"


def validate_gpu_count(expected_gpus: int, allow_mismatch: bool) -> dict[str, Any]:
    """Validate that the cluster exposes enough GPUs before expensive runs."""
    detected_gpus, detector = detect_gpu_count()
    result = {
        "expected_gpus": expected_gpus,
        "detected_gpus": detected_gpus,
        "detector": detector,
        "allow_mismatch": allow_mismatch,
    }
    if detected_gpus is None:
        _require(allow_mismatch, f"cannot detect GPU count: {detector}")
        return result
    _require(
        allow_mismatch or detected_gpus >= expected_gpus,
        f"detected {detected_gpus} GPUs but manifest expects {expected_gpus}",
    )
    return result


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_url(url: str, destination: Path) -> None:
    """Copy a file:// artifact so tests and offline mirrors share one path."""
    parsed_url = urlparse(url)
    source_path_text = parsed_url.path
    if os.name == "nt":
        if parsed_url.netloc:
            source_path_text = f"//{parsed_url.netloc}{parsed_url.path}"
        elif re.match(r"^/[A-Za-z]:", parsed_url.path):
            source_path_text = parsed_url.path[1:]
    source_path = Path(source_path_text)
    shutil.copyfile(source_path, destination)


def download_artifact(artifact: ArtifactConfig, dry_run: bool = False) -> dict[str, Any]:
    """Download a missing artifact and verify its optional SHA-256 digest."""
    path = artifact.path
    record: dict[str, Any] = {
        "name": artifact.name,
        "path": str(path),
        "url": artifact.url,
        "required": artifact.required,
        "downloaded": False,
        "skipped_optional_missing": False,
        "sha256": None,
    }
    if path.exists():
        if artifact.sha256:
            digest = sha256_file(path)
            _require(
                digest.lower() == artifact.sha256.lower(),
                f"{artifact.name}: SHA-256 mismatch for existing artifact {path}",
            )
            record["sha256"] = digest
        return record

    if artifact.url is None:
        _require(
            not artifact.required,
            f"{artifact.name}: missing required artifact and no url: {path}",
        )
        record["skipped_optional_missing"] = True
        return record
    if dry_run:
        record["downloaded"] = "planned"
        return record

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".download")
    parsed_url = urlparse(artifact.url)
    if parsed_url.scheme == "file":
        _copy_file_url(artifact.url, temporary_path)
    else:
        with urlopen(artifact.url, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
    if artifact.sha256:
        digest = sha256_file(temporary_path)
        _require(
            digest.lower() == artifact.sha256.lower(),
            f"{artifact.name}: downloaded artifact SHA-256 mismatch",
        )
        record["sha256"] = digest
    temporary_path.replace(path)
    record["downloaded"] = True
    return record


def _augment_artifact_record(artifact: ArtifactConfig, record: dict[str, Any]) -> dict[str, Any]:
    """Add post-prepare existence, size, and digest evidence to one record."""
    enriched_record = dict(record)
    artifact_exists = artifact.path.exists()
    enriched_record["exists_after_prepare"] = artifact_exists
    if artifact_exists:
        enriched_record["size_bytes"] = artifact.path.stat().st_size
        enriched_record["actual_sha256"] = sha256_file(artifact.path)
    else:
        enriched_record["size_bytes"] = None
        enriched_record["actual_sha256"] = None
    return enriched_record


def prepare_artifacts(config: SuiteConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Download and verify all declared artifacts before reserving GPUs."""
    validate_suite_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    total_stages = max(len(config.artifacts), MIN_REQUIRED_CASE_COUNT)
    records: list[dict[str, Any]] = []
    if not config.artifacts:
        _progress(config.name, MIN_REQUIRED_CASE_COUNT, total_stages, "no artifacts declared")
    for index, artifact in enumerate(config.artifacts, start=MIN_REQUIRED_CASE_COUNT):
        _progress(config.name, index, total_stages, f"preparing artifact {artifact.name}")
        record = download_artifact(artifact, dry_run=dry_run)
        records.append(_augment_artifact_record(artifact, record))

    missing_required = [
        record["name"]
        for record in records
        if record["required"] and not record["exists_after_prepare"]
    ]
    status = "planned" if dry_run else "ready"
    if missing_required and not dry_run:
        status = "failed"
    report_path = config.output_dir / ARTIFACT_PREPARATION_REPORT_NAME
    payload = {
        "artifact_preparation_schema_version": ARTIFACT_PREPARATION_SCHEMA_VERSION,
        "status": status,
        "dry_run": dry_run,
        "report_path": str(report_path),
        "provenance": collect_run_provenance(),
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "manifest": manifest_record(config),
            "output_dir": str(config.output_dir),
            "require_artifact_sha256": config.require_artifact_sha256,
        },
        "missing_required_artifacts": missing_required,
        "artifacts": records,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _require(
        dry_run or not missing_required,
        "required artifacts were not prepared: " + MODEL_NAME_JOINER.join(missing_required),
    )
    return payload


def _base_preflight_artifact_record(artifact: ArtifactConfig) -> dict[str, Any]:
    """Return the shared artifact fields used by preflight records."""
    return {
        "name": artifact.name,
        "path": str(artifact.path),
        "url": artifact.url,
        "required": artifact.required,
        "downloaded": False,
        "skipped_optional_missing": False,
        "sha256": None,
    }


def _preflight_download_failure_record(
    artifact: ArtifactConfig,
    exc: BaseException,
) -> dict[str, Any]:
    """Return a failed artifact-preflight record with post-check evidence."""
    record = _base_preflight_artifact_record(artifact)
    record["error"] = str(exc)
    record["status"] = PREFLIGHT_STATUS_FAILED
    return _augment_artifact_record(artifact, record)


def _preflight_skip_download_record(
    artifact: ArtifactConfig,
    *,
    reason: str,
) -> dict[str, Any]:
    """Return an artifact record when preflight intentionally skips downloads."""
    record = _base_preflight_artifact_record(artifact)
    record["status"] = PREFLIGHT_STATUS_SKIPPED
    record["skip_reason"] = reason
    return _augment_artifact_record(artifact, record)


def _preflight_status_from_failures(
    *,
    dry_run: bool,
    failures: list[dict[str, Any]],
) -> str:
    """Return the suite-level preflight status from collected failures."""
    if failures:
        return PREFLIGHT_STATUS_FAILED
    return PREFLIGHT_STATUS_PLANNED if dry_run else PREFLIGHT_STATUS_PASSED


def _preflight_case_skip_record(case: CaseConfig) -> dict[str, Any]:
    """Return a case-preflight record when no command was configured."""
    return {
        "name": case.name,
        "model": case.model,
        "kind": case.kind,
        "status": PREFLIGHT_STATUS_SKIPPED,
        "reason": PREFLIGHT_NO_COMMAND_REASON,
        "command": None,
        "returncode": None,
        "stdout_path": None,
        "stderr_path": None,
    }


def _preflight_case_command_record(
    case: CaseConfig,
    record: CommandRecord,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Return a JSON row for one executed or planned case preflight command."""
    if dry_run:
        status = PREFLIGHT_STATUS_PLANNED
    elif record.returncode == SUCCESS_RETURN_CODE:
        status = PREFLIGHT_STATUS_PASSED
    else:
        status = PREFLIGHT_STATUS_FAILED
    return {
        "name": case.name,
        "model": case.model,
        "kind": case.kind,
        "status": status,
        "reason": None,
        "command": record.command,
        "returncode": record.returncode,
        "elapsed_seconds": record.elapsed_seconds,
        "stdout_path": record.stdout_path,
        "stderr_path": record.stderr_path,
    }


def run_preflight_only(
    config: SuiteConfig,
    *,
    dry_run: bool = False,
    skip_downloads: bool = False,
    skip_gpu_check: bool = False,
    allow_gpu_mismatch: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run artifact, GPU, and per-model preflight checks without benchmark jobs."""
    validate_suite_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = report_path or config.output_dir / PREFLIGHT_REPORT_NAME
    download_stage_count = MIN_REQUIRED_CASE_COUNT if skip_downloads or not config.artifacts else len(config.artifacts)
    total_stages = MIN_REQUIRED_CASE_COUNT + download_stage_count + len(config.cases) + MIN_REQUIRED_CASE_COUNT
    stage_index = MIN_REQUIRED_CASE_COUNT
    failures: list[dict[str, Any]] = []
    command_records: list[CommandRecord] = []

    gpu_record: dict[str, Any] | None = None
    if skip_gpu_check:
        _progress(config.name, stage_index, total_stages, "GPU check skipped")
        gpu_record = {
            "expected_gpus": config.expected_gpus,
            "detected_gpus": None,
            "detector": None,
            "allow_mismatch": allow_gpu_mismatch,
            "skipped": True,
        }
    else:
        _progress(config.name, stage_index, total_stages, f"checking for {config.expected_gpus} GPUs")
        try:
            gpu_record = validate_gpu_count(config.expected_gpus, allow_gpu_mismatch)
            gpu_record["skipped"] = False
        except ClusterSuiteError as exc:
            gpu_record = {
                "expected_gpus": config.expected_gpus,
                "detected_gpus": None,
                "detector": None,
                "allow_mismatch": allow_gpu_mismatch,
                "skipped": False,
                "error": str(exc),
            }
            failures.append({"stage": "gpu", "message": str(exc)})
    stage_index += MIN_REQUIRED_CASE_COUNT

    download_records: list[dict[str, Any]] = []
    if skip_downloads:
        _progress(config.name, stage_index, total_stages, "artifact downloads skipped")
        for artifact in config.artifacts:
            record = _preflight_skip_download_record(
                artifact,
                reason=PREFLIGHT_SKIP_DOWNLOADS_REASON,
            )
            download_records.append(record)
            if artifact.required and not artifact.path.exists():
                failures.append(
                    {
                        "stage": "artifact",
                        "name": artifact.name,
                        "message": f"required artifact is missing: {artifact.path}",
                    }
                )
            if (
                artifact.sha256 is not None
                and record["actual_sha256"] is not None
                and record["actual_sha256"].lower() != artifact.sha256.lower()
            ):
                failures.append(
                    {
                        "stage": "artifact",
                        "name": artifact.name,
                        "message": f"SHA-256 mismatch for existing artifact {artifact.path}",
                    }
                )
        stage_index += MIN_REQUIRED_CASE_COUNT
    elif not config.artifacts:
        _progress(config.name, stage_index, total_stages, "no artifacts declared")
        stage_index += MIN_REQUIRED_CASE_COUNT
    else:
        for artifact in config.artifacts:
            _progress(config.name, stage_index, total_stages, f"preflighting artifact {artifact.name}")
            try:
                record = download_artifact(artifact, dry_run=dry_run)
                download_records.append(_augment_artifact_record(artifact, record))
            except (ClusterSuiteError, OSError) as exc:
                download_records.append(_preflight_download_failure_record(artifact, exc))
                failures.append(
                    {
                        "stage": "artifact",
                        "name": artifact.name,
                        "message": str(exc),
                    }
                )
            stage_index += MIN_REQUIRED_CASE_COUNT

    case_preflights: list[dict[str, Any]] = []
    for case in config.cases:
        _progress(config.name, stage_index, total_stages, f"preflighting case {case.name} ({case.model})")
        if not case.preflight_command:
            case_preflights.append(_preflight_case_skip_record(case))
        else:
            records = run_case_preflight(config, case, dry_run=dry_run)
            command_records.extend(records)
            for record in records:
                case_record = _preflight_case_command_record(case, record, dry_run=dry_run)
                case_preflights.append(case_record)
                if record.returncode != SUCCESS_RETURN_CODE:
                    failures.append(
                        {
                            "stage": "case_preflight",
                            "name": case.name,
                            "message": f"returncode={record.returncode}",
                        }
                    )
        stage_index += MIN_REQUIRED_CASE_COUNT

    _progress(config.name, stage_index, total_stages, "writing preflight report")
    environment_snapshot_path = write_environment_snapshot(config, gpu_record)
    payload = {
        "preflight_report_schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "status": _preflight_status_from_failures(dry_run=dry_run, failures=failures),
        "dry_run": dry_run,
        "skip_downloads": skip_downloads,
        "skip_gpu_check": skip_gpu_check,
        "allow_gpu_mismatch": allow_gpu_mismatch,
        "report_path": str(final_report_path),
        "provenance": collect_run_provenance(),
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "manifest": manifest_record(config),
            "output_dir": str(config.output_dir),
            "expected_gpus": config.expected_gpus,
            "required_models": list(config.required_models),
        },
        "gpu_check": gpu_record,
        "downloads": download_records,
        "case_preflights": case_preflights,
        "failures": failures,
        "commands": [asdict(record) for record in command_records],
        "command_log_fingerprints": command_log_fingerprints(command_records),
        "environment_snapshot": str(environment_snapshot_path),
        "environment_snapshot_fingerprint": generated_artifact_record(environment_snapshot_path),
    }
    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_required_artifacts_available(
    config: SuiteConfig,
    *,
    skip_downloads: bool,
    collect_only: bool,
    dry_run: bool,
) -> None:
    """Fail before execution when required artifacts cannot be materialized."""
    if not skip_downloads or collect_only or dry_run:
        return
    missing_required_artifacts = [
        artifact
        for artifact in config.artifacts
        if artifact.required and not artifact.path.exists()
    ]
    _require(
        not missing_required_artifacts,
        "required artifacts are missing while --skip-downloads is active: "
        + MODEL_NAME_JOINER.join(artifact.name for artifact in missing_required_artifacts),
    )


def _write_command_streams(
    stdout_path: Path,
    stderr_path: Path,
    stdout_text: str,
    stderr_text: str,
) -> None:
    """Persist command streams before returning to the orchestration layer."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")


def run_argv_command(
    *,
    name: str,
    argv: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
) -> CommandRecord:
    """Run an argv command, capture logs, and return timing metadata."""
    if dry_run:
        _write_command_streams(stdout_path, stderr_path, "DRY RUN\n", "")
        return CommandRecord(name, argv, SUCCESS_RETURN_CODE, 0.0, str(stdout_path), str(stderr_path))
    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            completed.stdout,
            completed.stderr,
        )
        return CommandRecord(
            name=name,
            command=argv,
            returncode=int(completed.returncode),
            elapsed_seconds=elapsed_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            exc.stdout or "",
            f"Command timed out after {timeout_seconds:g} seconds\n{exc.stderr or ''}",
        )
        return CommandRecord(
            name=name,
            command=argv,
            returncode=COMMAND_TIMEOUT_RETURN_CODE,
            elapsed_seconds=elapsed_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


def run_shell_command(
    *,
    name: str,
    command: str,
    cwd: Path,
    env: dict[str, str] | None,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
) -> CommandRecord:
    """Run a manifest shell command and capture its logs."""
    if dry_run:
        _write_command_streams(stdout_path, stderr_path, "DRY RUN\n", "")
        return CommandRecord(name, command, SUCCESS_RETURN_CODE, 0.0, str(stdout_path), str(stderr_path))
    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            completed.stdout,
            completed.stderr,
        )
        return CommandRecord(
            name=name,
            command=command,
            returncode=int(completed.returncode),
            elapsed_seconds=elapsed_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            exc.stdout or "",
            f"Command timed out after {timeout_seconds:g} seconds\n{exc.stderr or ''}",
        )
        return CommandRecord(
            name=name,
            command=command,
            returncode=COMMAND_TIMEOUT_RETURN_CODE,
            elapsed_seconds=elapsed_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


def _case_output_dir(config: SuiteConfig, case: CaseConfig) -> Path:
    """Return the output directory for one case."""
    return config.output_dir / CASES_DIR_NAME / _safe_name(case.name)


def _trace_output_path(config: SuiteConfig, case: CaseConfig) -> Path:
    """Return the generated trace evidence path for trace_input cases."""
    return _case_output_dir(config, case) / f"{_safe_name(case.name)}{TRACE_EVIDENCE_SUFFIX}"


def _external_timing_report_path(config: SuiteConfig, case: CaseConfig) -> Path:
    """Return the generated timing report path for external-pair cases."""
    return _case_output_dir(config, case) / EXTERNAL_TIMING_REPORT_NAME


def _path_text(path: Path | None) -> str | None:
    """Return a JSON-friendly path string while preserving absent fields."""
    return str(path) if path is not None else None


def _planned_benchmark_report(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    collect_only: bool,
) -> Path | None:
    """Return the benchmark report path expected for one planned case."""
    if collect_only or case.kind != "sevennet_lammps":
        return case.benchmark_report
    return _case_output_dir(config, case) / "experiment" / "benchmark" / BENCHMARK_REPORT_NAME


def _planned_bundle_evidence(
    config: SuiteConfig,
    case: CaseConfig,
    trace_paths: tuple[Path, ...],
    *,
    collect_only: bool,
) -> Path | None:
    """Return the bundle evidence path expected for one planned case."""
    if collect_only or case.kind != "sevennet_lammps" or not trace_paths:
        return case.bundle_evidence
    return _case_output_dir(config, case) / "experiment" / BUNDLE_EVIDENCE_NAME


def _planned_trace_paths(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    collect_only: bool,
) -> tuple[Path, ...]:
    """Return trace evidence paths that should exist after a planned case."""
    generated_paths: tuple[Path, ...] = ()
    if not collect_only and case.trace_input is not None:
        generated_paths = (_trace_output_path(config, case),)
    return case.trace_evidence_paths + generated_paths


def planned_case_outputs(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    collect_only: bool,
) -> tuple[Path | None, Path | None, tuple[Path, ...], Path | None]:
    """Return the output paths that a case should validate or reuse."""
    trace_paths = _planned_trace_paths(config, case, collect_only=collect_only)
    external_timing_report = (
        case.external_timing_report
        if collect_only
        else (
            _external_timing_report_path(config, case)
            if case.kind == "external_pair"
            else None
        )
    )
    return (
        _planned_benchmark_report(config, case, collect_only=collect_only),
        _planned_bundle_evidence(
            config,
            case,
            trace_paths,
            collect_only=collect_only,
        ),
        trace_paths,
        external_timing_report,
    )


def _planned_paths_exist(paths: tuple[Path | None, ...]) -> bool:
    """Return whether every planned non-null path exists for reuse planning."""
    concrete_paths = [path for path in paths if path is not None]
    return bool(concrete_paths) and all(path.exists() for path in concrete_paths)


def _has_reusable_case_outputs(
    benchmark_report: Path | None,
    bundle_evidence: Path | None,
    trace_evidence_paths: tuple[Path, ...],
    external_timing_report: Path | None,
) -> bool:
    """Return whether a case has at least one planned artifact to validate."""
    return any(
        (
            benchmark_report is not None,
            bundle_evidence is not None,
            bool(trace_evidence_paths),
            external_timing_report is not None,
        )
    )


def manifest_record(config: SuiteConfig) -> dict[str, Any]:
    """Return a reproducible fingerprint for the suite manifest file."""
    return {
        "path": str(config.manifest_path),
        "sha256": sha256_file(config.manifest_path),
        "size_bytes": config.manifest_path.stat().st_size,
    }


def generated_artifact_record(path: Path) -> dict[str, Any]:
    """Return a reproducible fingerprint for one generated paper artifact."""
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def optional_file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a fingerprint record that explicitly handles absent log files."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def command_log_fingerprints(command_records: list[CommandRecord]) -> list[dict[str, Any]]:
    """Fingerprint stdout/stderr logs for every launched command."""
    return [
        {
            "name": record.name,
            "returncode": record.returncode,
            "stdout": optional_file_fingerprint(Path(record.stdout_path)),
            "stderr": optional_file_fingerprint(Path(record.stderr_path)),
        }
        for record in command_records
    ]


def _resolve_summary_path(bundle_or_summary_path: Path) -> Path:
    """Resolve either an output directory or a direct summary JSON path."""
    candidate_path = bundle_or_summary_path.resolve()
    return candidate_path / SUMMARY_REPORT_NAME if candidate_path.is_dir() else candidate_path


def _candidate_fingerprint_paths(
    recorded_path: Path,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> list[Path]:
    """Return filesystem locations that may hold one recorded bundle file."""
    candidates = [recorded_path]
    if not recorded_path.is_absolute():
        candidates.append(bundle_root / recorded_path)
    if original_output_dir is not None and recorded_path.is_absolute():
        try:
            relative_path = recorded_path.resolve().relative_to(original_output_dir.resolve())
        except ValueError:
            relative_path = None
        if relative_path is not None:
            candidates.append(bundle_root / relative_path)
    unique_candidates: list[Path] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.resolve())
        if candidate_key in seen_paths:
            continue
        seen_paths.add(candidate_key)
        unique_candidates.append(candidate)
    return unique_candidates


def _original_output_dir(summary_payload: dict[str, Any]) -> Path | None:
    """Return the original output_dir stored in a summary if it is available."""
    suite_record = summary_payload.get("suite")
    if not isinstance(suite_record, dict):
        return None
    output_dir = suite_record.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return None
    return Path(output_dir)


def _require_fingerprint_match(
    record: dict[str, Any],
    label: str,
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> None:
    """Validate one fingerprint record against the current filesystem."""
    path_text = _as_json_string(record.get("path"), f"{label}.path")
    recorded_path = Path(path_text)
    candidate_paths = _candidate_fingerprint_paths(
        recorded_path,
        bundle_root,
        original_output_dir,
    )
    expected_exists = record.get("exists", True)
    if expected_exists is False:
        existing_paths = [path for path in candidate_paths if path.exists()]
        if existing_paths:
            _require(False, f"{label}: expected absent file exists: {existing_paths[0]}")
        return
    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    _require(
        path is not None,
        f"{label}: missing file {recorded_path}; checked {', '.join(str(path) for path in candidate_paths)}",
    )
    expected_sha256 = _as_json_string(record.get("sha256"), f"{label}.sha256")
    expected_size = _as_json_nonnegative_int(record.get("size_bytes"), f"{label}.size_bytes")
    actual_sha256 = sha256_file(path)
    actual_size = path.stat().st_size
    _require(
        actual_sha256 == expected_sha256,
        f"{label}: SHA-256 mismatch for {path}",
    )
    _require(
        actual_size == expected_size,
        f"{label}: byte size mismatch for {path}",
    )


def verify_output_bundle(bundle_or_summary_path: Path) -> dict[str, Any]:
    """Verify summary-recorded artifact and command-log fingerprints."""
    summary_path = _resolve_summary_path(bundle_or_summary_path)
    _require(summary_path.exists(), f"missing output summary {summary_path}")
    bundle_root = summary_path.parent
    summary_payload = _as_json_object(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "summary",
    )
    artifact_fingerprints = _as_json_object(
        summary_payload.get("artifact_fingerprints"),
        "artifact_fingerprints",
    )
    command_fingerprints = summary_payload.get("command_log_fingerprints", [])
    _require(
        isinstance(command_fingerprints, list),
        "command_log_fingerprints must be a JSON array",
    )
    original_output_dir = _original_output_dir(summary_payload)

    verified_artifact_count = 0
    for artifact_name, raw_record in artifact_fingerprints.items():
        record = _as_json_object(raw_record, f"artifact_fingerprints.{artifact_name}")
        _require_fingerprint_match(
            record,
            f"artifact_fingerprints.{artifact_name}",
            bundle_root=bundle_root,
            original_output_dir=original_output_dir,
        )
        verified_artifact_count += 1

    verified_log_count = 0
    for index, raw_command_record in enumerate(command_fingerprints):
        command_record = _as_json_object(
            raw_command_record,
            f"command_log_fingerprints[{index}]",
        )
        command_name = _as_json_string(
            command_record.get("name"),
            f"command_log_fingerprints[{index}].name",
        )
        for stream_name in ("stdout", "stderr"):
            stream_record = _as_json_object(
                command_record.get(stream_name),
                f"command_log_fingerprints[{index}].{stream_name}",
            )
            _require_fingerprint_match(
                stream_record,
                f"{command_name}.{stream_name}",
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
            verified_log_count += 1

    return {
        "status": "passed",
        "summary_json": str(summary_path),
        "verified_artifact_count": verified_artifact_count,
        "verified_command_log_count": verified_log_count,
    }


def write_manifest_snapshot(config: SuiteConfig) -> Path:
    """Copy the manifest into the output bundle for archival review."""
    snapshot_path = config.output_dir / MANIFEST_SNAPSHOT_NAME
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.manifest_path, snapshot_path)
    return snapshot_path


def build_run_plan(
    config: SuiteConfig,
    *,
    collect_only: bool,
    skip_downloads: bool,
    skip_gpu_check: bool,
    reuse_passed: bool = False,
) -> dict[str, Any]:
    """Build a machine-readable preflight plan before using cluster time."""
    validate_suite_config(config)
    artifact_plan = []
    for artifact in config.artifacts:
        artifact_exists = artifact.path.exists()
        artifact_plan.append(
            {
                "name": artifact.name,
                "path": str(artifact.path),
                "url": artifact.url,
                "required": artifact.required,
                "required_by": list(artifact.required_by),
                "exists": artifact_exists,
                "sha256": artifact.sha256,
                "has_sha256": artifact.sha256 is not None,
                "sha256_required": config.require_artifact_sha256 and artifact.required,
                "missing_required": artifact.required and not artifact_exists,
                "missing_optional": not artifact.required and not artifact_exists,
                "will_download": (
                    not skip_downloads
                    and not collect_only
                    and not artifact_exists
                    and artifact.url is not None
                ),
                "missing_without_url": (
                    not skip_downloads
                    and not collect_only
                    and not artifact_exists
                    and artifact.url is None
                ),
                "skip_downloads_would_fail": (
                    skip_downloads
                    and not collect_only
                    and artifact.required
                    and not artifact_exists
                ),
            }
        )

    case_plan = []
    for case in config.cases:
        (
            planned_benchmark_report,
            planned_bundle_evidence,
            trace_paths,
            external_timing_report,
        ) = planned_case_outputs(config, case, collect_only=collect_only)
        case_plan.append(
            {
                "name": case.name,
                "model": case.model,
                "kind": case.kind,
                "output_dir": str(_case_output_dir(config, case)),
                "artifacts": list(case.artifacts),
                "commands": {
                    "preflight_command": case.preflight_command,
                    "lammps_command": case.lammps_command,
                    "disabled_command": case.disabled_command,
                    "enabled_command": case.enabled_command,
                    "trace_command": case.trace_command,
                },
                "inputs": {
                    "input": _path_text(case.input_path),
                    "work_dir": _path_text(case.work_dir),
                    "lammps_root": _path_text(case.lammps_root),
                    "trace_input": _path_text(case.trace_input),
                },
                "expected_outputs": {
                    "benchmark_report": _path_text(planned_benchmark_report),
                    "bundle_evidence": _path_text(planned_bundle_evidence),
                    "trace_evidence": [str(path) for path in trace_paths],
                    "external_timing_report": _path_text(external_timing_report),
                },
                "reuse": {
                    "enabled": reuse_passed,
                    "eligible": _has_reusable_case_outputs(
                        planned_benchmark_report,
                        planned_bundle_evidence,
                        trace_paths,
                        external_timing_report,
                    ),
                    "all_expected_outputs_exist": _planned_paths_exist(
                        (
                            planned_benchmark_report,
                            planned_bundle_evidence,
                            external_timing_report,
                            *trace_paths,
                        )
                    ),
                },
                "thresholds": {
                    "repeat_count": case.repeat_count,
                    "preflight_timeout_seconds": case.preflight_timeout_seconds,
                    "min_speedup": case.min_speedup,
                    "min_speedup_95ci_lower_bound": case.min_speedup_95ci_lower_bound,
                    "min_hit_rate_percent": case.min_hit_rate_percent,
                    "min_enabled_cache_attempts": case.min_enabled_cache_attempts,
                    "min_enabled_cache_hits": case.min_enabled_cache_hits,
                    "min_trace_hit_rate_percent": case.min_trace_hit_rate_percent,
                    "min_trace_estimated_speedup": case.min_trace_estimated_speedup,
                    "min_trace_metadata_fraction_percent": (
                        case.min_trace_metadata_fraction_percent
                    ),
                },
            }
        )

    return {
        "plan_schema_version": SUITE_SCHEMA_VERSION,
        "provenance": collect_run_provenance(),
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "manifest": manifest_record(config),
            "output_dir": str(config.output_dir),
            "expected_gpus": config.expected_gpus,
            "required_models": list(config.required_models),
            "min_trace_count": config.min_trace_count,
            "min_distinct_trace_models": config.min_distinct_trace_models,
            "require_artifact_sha256": config.require_artifact_sha256,
        },
        "modes": {
            "collect_only": collect_only,
            "skip_downloads": skip_downloads,
            "skip_gpu_check": skip_gpu_check,
            "reuse_passed": reuse_passed,
        },
        "gpu_check_planned": not skip_gpu_check,
        "artifacts": artifact_plan,
        "cases": case_plan,
        "paper_outputs": {
            "summary_json": str(config.output_dir / SUMMARY_REPORT_NAME),
            "environment_snapshot": str(config.output_dir / ENVIRONMENT_SNAPSHOT_NAME),
            "case_summary_csv": str(config.output_dir / TABLES_DIR_NAME / "case_summary.csv"),
            "case_summary_markdown": str(config.output_dir / TABLES_DIR_NAME / "case_summary.md"),
            "correlation_csv": str(config.output_dir / TABLES_DIR_NAME / "correlation.csv"),
            "speedup_svg": str(config.output_dir / FIGURES_DIR_NAME / "speedup_by_case.svg"),
            "hit_rate_svg": str(config.output_dir / FIGURES_DIR_NAME / "hit_rate_vs_speedup.svg"),
            "trace_svg": str(
                config.output_dir / FIGURES_DIR_NAME / "trace_metadata_fraction_vs_speedup.svg"
            ),
        },
    }


def write_run_plan(
    config: SuiteConfig,
    plan_path: Path,
    *,
    collect_only: bool,
    skip_downloads: bool,
    skip_gpu_check: bool,
    reuse_passed: bool = False,
) -> Path:
    """Write the cluster preflight plan to a JSON file."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(
        config,
        collect_only=collect_only,
        skip_downloads=skip_downloads,
        skip_gpu_check=skip_gpu_check,
        reuse_passed=reuse_passed,
    )
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def _default_case_env(case_env: dict[str, str], disabled: bool) -> dict[str, str]:
    """Build environment overrides for disabled/enabled external commands."""
    env = os.environ.copy()
    env[SEVENNET_PRINT_INFO_ENV] = ENV_FLAG_ENABLED
    env[SEVENNET_PROFILE_ENV] = ENV_FLAG_ENABLED
    if disabled:
        env[SEVENNET_DISABLE_ENV] = ENV_FLAG_ENABLED
    else:
        env.pop(SEVENNET_DISABLE_ENV, None)
    env.update(case_env)
    return env


def _progress(prefix: str, current: int, total: int, message: str) -> None:
    """Print one human-readable progress line for cluster terminals."""
    print(f"[{prefix}] [{current}/{total}] {message}", flush=True)


def run_case_preflight(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    dry_run: bool,
) -> list[CommandRecord]:
    """Run a cheap environment check before spending GPU time on one case."""
    if not case.preflight_command:
        return []
    case_dir = _case_output_dir(config, case)
    log_dir = case_dir / LOGS_DIR_NAME
    env = os.environ.copy()
    env.update(case.preflight_env)
    record = run_shell_command(
        name=f"{case.name}:preflight",
        command=case.preflight_command,
        cwd=REPO_ROOT,
        env=env,
        timeout_seconds=case.preflight_timeout_seconds,
        stdout_path=log_dir / "preflight.stdout.log",
        stderr_path=log_dir / "preflight.stderr.log",
        dry_run=dry_run,
    )
    return [record]


def run_trace_generation(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    dry_run: bool,
) -> tuple[tuple[Path, ...], list[CommandRecord]]:
    """Generate portable trace evidence when the manifest provides trace inputs."""
    command_records: list[CommandRecord] = []
    generated_paths: list[Path] = []
    case_dir = _case_output_dir(config, case)
    log_dir = case_dir / LOGS_DIR_NAME
    if case.trace_input is not None:
        output_path = _trace_output_path(config, case)
        argv = [
            sys.executable,
            str(TRACE_CHECK_PATH),
            "--trace",
            str(case.trace_input),
            "--min-hit-rate-percent",
            str(case.min_trace_hit_rate_percent),
            "--min-metadata-fraction-percent",
            str(case.min_trace_metadata_fraction_percent),
            "--output",
            str(output_path),
        ]
        if case.min_trace_estimated_speedup is not None:
            argv.extend(["--min-estimated-speedup", str(case.min_trace_estimated_speedup)])
        record = run_argv_command(
            name=f"{case.name}:trace-check",
            argv=argv,
            cwd=REPO_ROOT,
            env=None,
            timeout_seconds=case.command_timeout_seconds,
            stdout_path=log_dir / "trace_check.stdout.log",
            stderr_path=log_dir / "trace_check.stderr.log",
            dry_run=dry_run,
        )
        command_records.append(record)
        if record.returncode == SUCCESS_RETURN_CODE:
            generated_paths.append(output_path)
    if case.trace_command:
        record = run_shell_command(
            name=f"{case.name}:trace-command",
            command=case.trace_command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            timeout_seconds=case.command_timeout_seconds,
            stdout_path=log_dir / "trace_command.stdout.log",
            stderr_path=log_dir / "trace_command.stderr.log",
            dry_run=dry_run,
        )
        command_records.append(record)
    return tuple(generated_paths), command_records


def run_sevennet_case(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    dry_run: bool,
) -> tuple[Path | None, Path | None, tuple[Path, ...], list[CommandRecord]]:
    """Run a SevenNet/LAMMPS case through the existing experiment driver."""
    case_dir = _case_output_dir(config, case)
    log_dir = case_dir / LOGS_DIR_NAME
    generated_traces, trace_records = run_trace_generation(config, case, dry_run=dry_run)
    trace_paths = case.trace_evidence_paths + generated_traces
    experiment_dir = case_dir / "experiment"
    argv = [
        sys.executable,
        str(EXPERIMENT_DRIVER_PATH),
        "--lammps-command",
        str(case.lammps_command),
        "--input",
        str(case.input_path),
        "--output-dir",
        str(experiment_dir),
        "--repeat",
        str(case.repeat_count),
        "--binary-timeout-seconds",
        str(case.binary_timeout_seconds),
        "--benchmark-timeout-seconds",
        str(case.benchmark_timeout_seconds),
        "--max-abs-thermo-delta",
        str(case.max_abs_thermo_delta),
        "--min-paired-thermo-count",
        str(case.min_paired_thermo_count),
        "--min-hit-rate-percent",
        str(case.min_hit_rate_percent),
        "--min-enabled-cache-attempts",
        str(case.min_enabled_cache_attempts),
        "--min-enabled-cache-hits",
        str(case.min_enabled_cache_hits),
    ]
    if case.work_dir is not None:
        argv.extend(["--work-dir", str(case.work_dir)])
    if case.lammps_root is not None:
        argv.extend(["--lammps-root", str(case.lammps_root)])
    if case.min_speedup is not None:
        argv.extend(["--min-speedup", str(case.min_speedup)])
    if trace_paths:
        argv.extend(["--min-trace-count", str(max(DEFAULT_MIN_TRACE_COUNT, len(trace_paths)))])
        argv.extend(["--min-distinct-trace-models", str(max(DEFAULT_MIN_DISTINCT_TRACE_MODELS, len(case.required_trace_models) or 1))])
        argv.extend(["--min-trace-hit-rate-percent", str(case.min_trace_hit_rate_percent)])
        argv.extend(["--min-trace-metadata-fraction-percent", str(case.min_trace_metadata_fraction_percent)])
        if case.min_trace_estimated_speedup is not None:
            argv.extend(["--min-trace-estimated-speedup", str(case.min_trace_estimated_speedup)])
        for trace_path in trace_paths:
            argv.extend(["--trace-evidence", str(trace_path)])
        for model_name in case.required_trace_models:
            argv.extend(["--require-trace-model", model_name])

    record = run_argv_command(
        name=f"{case.name}:sevennet-experiment",
        argv=argv,
        cwd=REPO_ROOT,
        env=None,
        timeout_seconds=case.command_timeout_seconds,
        stdout_path=log_dir / "sevennet_experiment.stdout.log",
        stderr_path=log_dir / "sevennet_experiment.stderr.log",
        dry_run=dry_run,
    )
    command_records = trace_records + [record]
    benchmark_report = experiment_dir / "benchmark" / BENCHMARK_REPORT_NAME
    bundle_evidence = experiment_dir / BUNDLE_EVIDENCE_NAME if trace_paths else None
    return benchmark_report, bundle_evidence, trace_paths, command_records


def run_external_pair_case(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    dry_run: bool,
) -> tuple[Path | None, Path | None, tuple[Path, ...], Path | None, list[CommandRecord]]:
    """Run manifest-provided disabled/enabled commands and time each repeat."""
    case_dir = _case_output_dir(config, case)
    log_dir = case_dir / LOGS_DIR_NAME
    case_dir.mkdir(parents=True, exist_ok=True)
    generated_traces, trace_records = run_trace_generation(config, case, dry_run=dry_run)
    command_records = list(trace_records)
    disabled_times: list[float] = []
    enabled_times: list[float] = []

    for repeat_index in range(case.repeat_count):
        disabled_record = run_shell_command(
            name=f"{case.name}:disabled:{repeat_index}",
            command=str(case.disabled_command),
            cwd=REPO_ROOT,
            env=_default_case_env(case.disabled_env, disabled=True),
            timeout_seconds=case.command_timeout_seconds,
            stdout_path=log_dir / f"disabled_{repeat_index}.stdout.log",
            stderr_path=log_dir / f"disabled_{repeat_index}.stderr.log",
            dry_run=dry_run,
        )
        command_records.append(disabled_record)
        if disabled_record.returncode == SUCCESS_RETURN_CODE:
            disabled_times.append(disabled_record.elapsed_seconds)

        enabled_record = run_shell_command(
            name=f"{case.name}:enabled:{repeat_index}",
            command=str(case.enabled_command),
            cwd=REPO_ROOT,
            env=_default_case_env(case.enabled_env, disabled=False),
            timeout_seconds=case.command_timeout_seconds,
            stdout_path=log_dir / f"enabled_{repeat_index}.stdout.log",
            stderr_path=log_dir / f"enabled_{repeat_index}.stderr.log",
            dry_run=dry_run,
        )
        command_records.append(enabled_record)
        if enabled_record.returncode == SUCCESS_RETURN_CODE:
            enabled_times.append(enabled_record.elapsed_seconds)

    timing_report_path = _external_timing_report_path(config, case)
    if not dry_run:
        timing_report = _build_external_timing_report(
            case=case,
            disabled_times=disabled_times,
            enabled_times=enabled_times,
            command_records=command_records,
        )
        timing_report_path.write_text(json.dumps(timing_report, indent=2), encoding="utf-8")
    trace_paths = case.trace_evidence_paths + generated_traces
    return case.benchmark_report, case.bundle_evidence, trace_paths, timing_report_path, command_records


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for a non-empty list."""
    return None if not values else sum(values) / len(values)


def _sample_variance(values: list[float]) -> float | None:
    """Return sample variance for repeat timings when enough samples exist."""
    if len(values) < MIN_SAMPLE_VARIANCE_COUNT:
        return None
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / (
        len(values) - SAMPLE_VARIANCE_DEGREES_OF_FREEDOM
    )


def _sample_stddev(values: list[float]) -> float | None:
    """Return sample standard deviation for repeat timings."""
    variance = _sample_variance(values)
    return None if variance is None else variance ** 0.5


def _mean_ci_half_width(stddev: float | None, count: int | None) -> float | None:
    """Return a 95% normal-approximation half-width for a mean timing."""
    if stddev is None or count is None or count < MIN_SAMPLE_VARIANCE_COUNT:
        return None
    return NORMAL_APPROX_95_CI_MULTIPLIER * stddev / math.sqrt(count)


def _speedup_ci_bounds(
    baseline_mean_seconds: float | None,
    enabled_mean_seconds: float | None,
    baseline_ci_half_width_seconds: float | None,
    enabled_ci_half_width_seconds: float | None,
) -> tuple[float | None, float | None]:
    """Return conservative speedup bounds derived from mean timing CIs."""
    if (
        baseline_mean_seconds is None
        or enabled_mean_seconds is None
        or baseline_ci_half_width_seconds is None
        or enabled_ci_half_width_seconds is None
    ):
        return None, None
    lower_baseline = baseline_mean_seconds - baseline_ci_half_width_seconds
    upper_baseline = baseline_mean_seconds + baseline_ci_half_width_seconds
    lower_enabled = enabled_mean_seconds - enabled_ci_half_width_seconds
    upper_enabled = enabled_mean_seconds + enabled_ci_half_width_seconds
    if lower_baseline <= MIN_POSITIVE_VALUE or lower_enabled <= MIN_POSITIVE_VALUE:
        return None, None
    return lower_baseline / upper_enabled, upper_baseline / lower_enabled


def _build_external_timing_report(
    *,
    case: CaseConfig,
    disabled_times: list[float],
    enabled_times: list[float],
    command_records: list[CommandRecord],
) -> dict[str, Any]:
    """Build a minimal timing report for non-SevenNet external pair runners."""
    disabled_mean = _mean(disabled_times)
    enabled_mean = _mean(enabled_times)
    speedup = None
    if disabled_mean is not None and enabled_mean is not None and enabled_mean > MIN_POSITIVE_VALUE:
        speedup = disabled_mean / enabled_mean
    return {
        SCHEMA_VERSION_KEY: EXTERNAL_TIMING_SCHEMA_VERSION,
        CASE_NAME_KEY: case.name,
        MODEL_KEY: case.model,
        REPEAT_COUNT_KEY: case.repeat_count,
        DISABLED_SUCCESS_COUNT_KEY: len(disabled_times),
        ENABLED_SUCCESS_COUNT_KEY: len(enabled_times),
        BASELINE_TIMES_SECONDS_KEY: disabled_times,
        ENABLED_TIMES_SECONDS_KEY: enabled_times,
        BASELINE_MEAN_SECONDS_KEY: disabled_mean,
        ENABLED_MEAN_SECONDS_KEY: enabled_mean,
        BASELINE_SAMPLE_VARIANCE_SECONDS_KEY: _sample_variance(disabled_times),
        ENABLED_SAMPLE_VARIANCE_SECONDS_KEY: _sample_variance(enabled_times),
        BASELINE_SAMPLE_STDDEV_SECONDS_KEY: _sample_stddev(disabled_times),
        ENABLED_SAMPLE_STDDEV_SECONDS_KEY: _sample_stddev(enabled_times),
        SPEEDUP_VS_DISABLED_CACHE_KEY: speedup,
        COMMANDS_KEY: [asdict(record) for record in command_records],
    }


def _timing_values_close(observed: float, expected: float) -> bool:
    """Return whether two timing-derived values agree within named tolerance."""
    tolerance = max(
        TIMING_ABSOLUTE_TOLERANCE_SECONDS,
        TIMING_RELATIVE_TOLERANCE * max(abs(observed), abs(expected)),
    )
    return abs(observed - expected) <= tolerance


def _validate_timing_statistic(
    report: dict[str, Any],
    key: str,
    expected_value: float | None,
) -> float | None:
    """Validate an optional timing statistic against a recomputed value."""
    observed_value = _as_json_optional_nonnegative_number(report.get(key), key)
    if expected_value is None:
        _require(observed_value is None, f"{key} must be null")
        return None
    _require(observed_value is not None, f"{key} must be numeric")
    _require(
        _timing_values_close(observed_value, expected_value),
        f"{key} must match raw timing samples",
    )
    return observed_value


def validate_external_timing_report(
    report: dict[str, Any],
    case: CaseConfig,
) -> dict[str, Any]:
    """Validate external-pair timing evidence before it reaches paper tables."""
    report = _as_json_object(report, "external_timing_report")
    schema_version = _as_json_string(
        report.get(SCHEMA_VERSION_KEY),
        SCHEMA_VERSION_KEY,
    )
    _require(
        schema_version == EXTERNAL_TIMING_SCHEMA_VERSION,
        (
            f"{SCHEMA_VERSION_KEY} must be "
            f"{EXTERNAL_TIMING_SCHEMA_VERSION!r}"
        ),
    )
    case_name = _as_json_string(report.get(CASE_NAME_KEY), CASE_NAME_KEY)
    model_name = _as_json_string(report.get(MODEL_KEY), MODEL_KEY)
    _require(case_name == case.name, f"{CASE_NAME_KEY} must match manifest case name")
    _require(model_name == case.model, f"{MODEL_KEY} must match manifest model")
    repeat_count = _as_json_nonnegative_int(
        report.get(REPEAT_COUNT_KEY),
        REPEAT_COUNT_KEY,
    )
    disabled_success_count = _as_json_nonnegative_int(
        report.get(DISABLED_SUCCESS_COUNT_KEY),
        DISABLED_SUCCESS_COUNT_KEY,
    )
    enabled_success_count = _as_json_nonnegative_int(
        report.get(ENABLED_SUCCESS_COUNT_KEY),
        ENABLED_SUCCESS_COUNT_KEY,
    )
    _require(repeat_count == case.repeat_count, f"{REPEAT_COUNT_KEY} must match manifest repeat_count")
    _require(
        disabled_success_count == repeat_count,
        f"{DISABLED_SUCCESS_COUNT_KEY} must equal {REPEAT_COUNT_KEY}",
    )
    _require(
        enabled_success_count == repeat_count,
        f"{ENABLED_SUCCESS_COUNT_KEY} must equal {REPEAT_COUNT_KEY}",
    )
    baseline_times = _as_json_positive_number_list(
        report.get(BASELINE_TIMES_SECONDS_KEY),
        BASELINE_TIMES_SECONDS_KEY,
    )
    enabled_times = _as_json_positive_number_list(
        report.get(ENABLED_TIMES_SECONDS_KEY),
        ENABLED_TIMES_SECONDS_KEY,
    )
    _require(
        len(baseline_times) == disabled_success_count,
        f"{BASELINE_TIMES_SECONDS_KEY} length must equal {DISABLED_SUCCESS_COUNT_KEY}",
    )
    _require(
        len(enabled_times) == enabled_success_count,
        f"{ENABLED_TIMES_SECONDS_KEY} length must equal {ENABLED_SUCCESS_COUNT_KEY}",
    )
    baseline_mean_seconds = _as_json_positive_number(
        report.get(BASELINE_MEAN_SECONDS_KEY),
        BASELINE_MEAN_SECONDS_KEY,
    )
    enabled_mean_seconds = _as_json_positive_number(
        report.get(ENABLED_MEAN_SECONDS_KEY),
        ENABLED_MEAN_SECONDS_KEY,
    )
    expected_baseline_mean = _mean(baseline_times)
    expected_enabled_mean = _mean(enabled_times)
    _require(
        expected_baseline_mean is not None
        and _timing_values_close(baseline_mean_seconds, expected_baseline_mean),
        f"{BASELINE_MEAN_SECONDS_KEY} must match raw timing samples",
    )
    _require(
        expected_enabled_mean is not None
        and _timing_values_close(enabled_mean_seconds, expected_enabled_mean),
        f"{ENABLED_MEAN_SECONDS_KEY} must match raw timing samples",
    )
    baseline_variance = _validate_timing_statistic(
        report,
        BASELINE_SAMPLE_VARIANCE_SECONDS_KEY,
        _sample_variance(baseline_times),
    )
    enabled_variance = _validate_timing_statistic(
        report,
        ENABLED_SAMPLE_VARIANCE_SECONDS_KEY,
        _sample_variance(enabled_times),
    )
    baseline_stddev = _validate_timing_statistic(
        report,
        BASELINE_SAMPLE_STDDEV_SECONDS_KEY,
        _sample_stddev(baseline_times),
    )
    enabled_stddev = _validate_timing_statistic(
        report,
        ENABLED_SAMPLE_STDDEV_SECONDS_KEY,
        _sample_stddev(enabled_times),
    )
    speedup = _as_json_positive_number(
        report.get(SPEEDUP_VS_DISABLED_CACHE_KEY),
        SPEEDUP_VS_DISABLED_CACHE_KEY,
    )
    expected_speedup = baseline_mean_seconds / enabled_mean_seconds
    _require(
        _timing_values_close(speedup, expected_speedup),
        f"{SPEEDUP_VS_DISABLED_CACHE_KEY} must match baseline / enabled seconds",
    )
    if case.min_speedup is not None:
        _require(
            speedup >= case.min_speedup,
            f"{case.name}: external timing speedup {speedup:g} is below {case.min_speedup:g}",
        )
    _require(
        isinstance(report.get(COMMANDS_KEY), list),
        f"{COMMANDS_KEY} must be a JSON array",
    )
    return {
        "status": "passed",
        SCHEMA_VERSION_KEY: schema_version,
        CASE_NAME_KEY: case_name,
        MODEL_KEY: model_name,
        REPEAT_COUNT_KEY: repeat_count,
        DISABLED_SUCCESS_COUNT_KEY: disabled_success_count,
        ENABLED_SUCCESS_COUNT_KEY: enabled_success_count,
        BASELINE_TIMES_SECONDS_KEY: baseline_times,
        ENABLED_TIMES_SECONDS_KEY: enabled_times,
        BASELINE_MEAN_SECONDS_KEY: baseline_mean_seconds,
        ENABLED_MEAN_SECONDS_KEY: enabled_mean_seconds,
        BASELINE_SAMPLE_VARIANCE_SECONDS_KEY: baseline_variance,
        ENABLED_SAMPLE_VARIANCE_SECONDS_KEY: enabled_variance,
        BASELINE_SAMPLE_STDDEV_SECONDS_KEY: baseline_stddev,
        ENABLED_SAMPLE_STDDEV_SECONDS_KEY: enabled_stddev,
        SPEEDUP_VS_DISABLED_CACHE_KEY: speedup,
    }


def run_trace_only_case(
    config: SuiteConfig,
    case: CaseConfig,
    *,
    dry_run: bool,
) -> tuple[tuple[Path, ...], list[CommandRecord]]:
    """Run or collect trace evidence for a model without paired timing."""
    generated_traces, command_records = run_trace_generation(config, case, dry_run=dry_run)
    return case.trace_evidence_paths + generated_traces, command_records


def _benchmark_thresholds(case: CaseConfig) -> Any:
    """Build benchmark thresholds from one case config."""
    return benchmark_check.ReportThresholds(
        max_abs_thermo_delta=case.max_abs_thermo_delta,
        min_paired_thermo_count=case.min_paired_thermo_count,
        min_speedup=case.min_speedup,
        min_hit_rate_percent=case.min_hit_rate_percent,
        min_enabled_cache_attempts=case.min_enabled_cache_attempts,
        min_enabled_cache_hits=case.min_enabled_cache_hits,
    )


def _trace_thresholds(case: CaseConfig) -> Any:
    """Build trace thresholds from one case config."""
    return trace_check.TraceThresholds(
        min_hit_rate_percent=case.min_trace_hit_rate_percent,
        min_estimated_speedup=case.min_trace_estimated_speedup,
        min_metadata_fraction_percent=case.min_trace_metadata_fraction_percent,
    )


def validate_case_outputs(
    case: CaseConfig,
    benchmark_report: Path | None,
    bundle_evidence: Path | None,
    trace_evidence_paths: tuple[Path, ...],
    external_timing_report: Path | None,
    *,
    dry_run: bool,
) -> None:
    """Validate generated files with the same gates used for paper evidence."""
    if dry_run:
        return
    if benchmark_report is not None:
        _require(benchmark_report.exists(), f"{case.name}: missing benchmark report {benchmark_report}")
        benchmark_check.validate_report(
            benchmark_check.load_report(benchmark_report),
            _benchmark_thresholds(case),
        )
    for trace_path in trace_evidence_paths:
        _require(trace_path.exists(), f"{case.name}: missing trace evidence {trace_path}")
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        trace_check.validate_trace_evidence(trace_payload, _trace_thresholds(case))
    if bundle_evidence is not None:
        _require(bundle_evidence.exists(), f"{case.name}: missing bundle evidence {bundle_evidence}")
        json.loads(bundle_evidence.read_text(encoding="utf-8"))
        if benchmark_report is not None and trace_evidence_paths:
            bundle_check.validate_bundle(
                benchmark_report=benchmark_report,
                trace_evidence_paths=list(trace_evidence_paths),
                required_models=list(case.required_trace_models),
                thresholds=bundle_check.BundleThresholds(
                    max_abs_thermo_delta=case.max_abs_thermo_delta,
                    min_paired_thermo_count=case.min_paired_thermo_count,
                    min_speedup=case.min_speedup,
                    min_hit_rate_percent=case.min_hit_rate_percent,
                    min_enabled_cache_attempts=case.min_enabled_cache_attempts,
                    min_enabled_cache_hits=case.min_enabled_cache_hits,
                    min_trace_hit_rate_percent=case.min_trace_hit_rate_percent,
                    min_trace_estimated_speedup=case.min_trace_estimated_speedup,
                    min_trace_metadata_fraction_percent=(
                        case.min_trace_metadata_fraction_percent
                    ),
                    min_trace_count=max(DEFAULT_MIN_TRACE_COUNT, len(trace_evidence_paths)),
                    min_distinct_trace_models=max(
                        DEFAULT_MIN_DISTINCT_TRACE_MODELS,
                        len(case.required_trace_models) or DEFAULT_MIN_DISTINCT_TRACE_MODELS,
                    ),
                ),
            )
    if external_timing_report is not None:
        _require(
            external_timing_report.exists(),
            f"{case.name}: missing external timing report {external_timing_report}",
        )
        timing_payload = json.loads(external_timing_report.read_text(encoding="utf-8"))
        validate_external_timing_report(timing_payload, case)


def try_reuse_case_outputs(
    config: SuiteConfig,
    case: CaseConfig,
) -> tuple[Path | None, Path | None, tuple[Path, ...], Path | None] | None:
    """Return reusable outputs only when existing artifacts pass current gates."""
    (
        benchmark_report,
        bundle_evidence,
        trace_evidence_paths,
        external_timing_report,
    ) = planned_case_outputs(config, case, collect_only=False)
    if not _has_reusable_case_outputs(
        benchmark_report,
        bundle_evidence,
        trace_evidence_paths,
        external_timing_report,
    ):
        return None
    try:
        validate_case_outputs(
            case,
            benchmark_report,
            bundle_evidence,
            trace_evidence_paths,
            external_timing_report,
            dry_run=False,
        )
    except (
        ClusterSuiteError,
        benchmark_check.ReportCheckError,
        trace_check.TraceCheckError,
        bundle_check.EvidenceBundleError,
        json.JSONDecodeError,
    ):
        return None
    return benchmark_report, bundle_evidence, trace_evidence_paths, external_timing_report


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON object when the path exists."""
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def validate_suite_evidence(
    config: SuiteConfig,
    case_summaries: list[CaseSummary],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Gate the complete paper matrix after all per-case checks pass."""
    passed_summaries = [
        summary for summary in case_summaries if summary.status in PASSING_CASE_STATUSES
    ]
    passed_models = {summary.model for summary in passed_summaries}
    missing_passed_models = [
        model_name
        for model_name in config.required_models
        if model_name not in passed_models
    ]
    _require(
        not missing_passed_models,
        "suite is missing passed evidence for required models: "
        + MODEL_NAME_JOINER.join(missing_passed_models),
    )

    trace_paths = sorted(
        {
            trace_path
            for summary in passed_summaries
            for trace_path in summary.trace_evidence
        }
    )
    _require(
        len(trace_paths) >= config.min_trace_count,
        f"suite trace evidence count {len(trace_paths)} is below {config.min_trace_count}",
    )

    if dry_run:
        trace_models = sorted(
            {
                summary.model
                for summary in passed_summaries
                if summary.trace_evidence
            }
        )
    else:
        trace_models = sorted(
            {
                _as_string(
                    _load_json_if_exists(Path(trace_path)).get("model"),
                    f"{trace_path}.model",
                )
                for trace_path in trace_paths
            }
        )
    _require(
        len(trace_models) >= config.min_distinct_trace_models,
        (
            f"suite distinct trace model count {len(trace_models)} is below "
            f"{config.min_distinct_trace_models}"
        ),
    )
    return {
        "required_models": list(config.required_models),
        "passed_models": sorted(passed_models),
        "trace_evidence_count": len(trace_paths),
        "distinct_trace_model_count": len(trace_models),
        "trace_models": trace_models,
        "min_trace_count": config.min_trace_count,
        "min_distinct_trace_models": config.min_distinct_trace_models,
    }


def _extract_benchmark_metrics(report: dict[str, Any] | None) -> dict[str, float | int | None]:
    """Extract timing, cache, and correctness metrics from a benchmark report."""
    if report is None:
        return {
            "baseline_mean_seconds": None,
            "enabled_mean_seconds": None,
            "baseline_sample_variance_seconds": None,
            "enabled_sample_variance_seconds": None,
            "baseline_sample_stddev_seconds": None,
            "enabled_sample_stddev_seconds": None,
            "baseline_timing_count": None,
            "enabled_timing_count": None,
            "speedup": None,
            "attempts": None,
            "hits": None,
            "hit_rate_percent": None,
            "max_abs_thermo_delta": None,
        }
    summary = report.get("summary", {})
    cases = summary.get("cases", {}) if isinstance(summary, dict) else {}
    baseline = cases.get(BASELINE_CASE_NAME, {}) if isinstance(cases, dict) else {}
    enabled = cases.get(ISODELTA_CASE_NAME, {}) if isinstance(cases, dict) else {}
    enabled_results = [
        result
        for result in report.get("results", [])
        if isinstance(result, dict) and result.get("case") == ISODELTA_CASE_NAME
    ]
    attempts = 0.0
    hits = 0.0
    for result in enabled_results:
        cache_summary = result.get("cache_summary", {})
        if isinstance(cache_summary, dict):
            attempts += float(cache_summary.get("attempts", 0.0))
            hits += float(cache_summary.get("hits", 0.0))
    max_delta = None
    delta_payload = summary.get("final_thermo_delta_vs_disabled_cache")
    if isinstance(delta_payload, dict):
        deltas = [
            float(metrics.get("max_abs_delta"))
            for metrics in delta_payload.values()
            if isinstance(metrics, dict)
            and isinstance(metrics.get("max_abs_delta"), (int, float))
        ]
        max_delta = max(deltas) if deltas else None
    hit_rate = None if attempts == 0.0 else PERCENT_SCALE * hits / attempts
    return {
        "baseline_mean_seconds": _coerce_optional_float(baseline.get("mean_loop_time_seconds")),
        "enabled_mean_seconds": _coerce_optional_float(enabled.get("mean_loop_time_seconds")),
        "baseline_sample_variance_seconds": _coerce_optional_float(
            baseline.get("sample_variance_loop_time_seconds")
        ),
        "enabled_sample_variance_seconds": _coerce_optional_float(
            enabled.get("sample_variance_loop_time_seconds")
        ),
        "baseline_sample_stddev_seconds": _coerce_optional_float(
            baseline.get("sample_stddev_loop_time_seconds")
        ),
        "enabled_sample_stddev_seconds": _coerce_optional_float(
            enabled.get("sample_stddev_loop_time_seconds")
        ),
        "baseline_timing_count": _coerce_optional_int(baseline.get("valid_loop_time_count")),
        "enabled_timing_count": _coerce_optional_int(enabled.get("valid_loop_time_count")),
        "speedup": _coerce_optional_float(summary.get("speedup_vs_disabled_cache")),
        "attempts": attempts if enabled_results else None,
        "hits": hits if enabled_results else None,
        "hit_rate_percent": hit_rate,
        "max_abs_thermo_delta": max_delta,
    }


def _coerce_optional_float(value: Any) -> float | None:
    """Return a finite float or None for absent report values."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    return None


def _coerce_optional_int(value: Any) -> int | None:
    """Return a nonnegative integer for optional count fields."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= MIN_NONNEGATIVE_VALUE else None
    if isinstance(value, float) and value.is_integer():
        integer_value = int(value)
        return integer_value if integer_value >= MIN_NONNEGATIVE_VALUE else None
    return None


def _extract_external_metrics(report: dict[str, Any] | None) -> dict[str, float | int | None]:
    """Extract timing metrics from the generated external-pair report."""
    if report is None:
        return {
            "baseline_mean_seconds": None,
            "enabled_mean_seconds": None,
            "baseline_sample_variance_seconds": None,
            "enabled_sample_variance_seconds": None,
            "baseline_sample_stddev_seconds": None,
            "enabled_sample_stddev_seconds": None,
            "baseline_timing_count": None,
            "enabled_timing_count": None,
            "speedup": None,
        }
    return {
        "baseline_mean_seconds": _coerce_optional_float(report.get(BASELINE_MEAN_SECONDS_KEY)),
        "enabled_mean_seconds": _coerce_optional_float(report.get(ENABLED_MEAN_SECONDS_KEY)),
        "baseline_sample_variance_seconds": _coerce_optional_float(
            report.get(BASELINE_SAMPLE_VARIANCE_SECONDS_KEY)
        ),
        "enabled_sample_variance_seconds": _coerce_optional_float(
            report.get(ENABLED_SAMPLE_VARIANCE_SECONDS_KEY)
        ),
        "baseline_sample_stddev_seconds": _coerce_optional_float(
            report.get(BASELINE_SAMPLE_STDDEV_SECONDS_KEY)
        ),
        "enabled_sample_stddev_seconds": _coerce_optional_float(
            report.get(ENABLED_SAMPLE_STDDEV_SECONDS_KEY)
        ),
        "baseline_timing_count": _coerce_optional_int(report.get(DISABLED_SUCCESS_COUNT_KEY)),
        "enabled_timing_count": _coerce_optional_int(report.get(ENABLED_SUCCESS_COUNT_KEY)),
        "speedup": _coerce_optional_float(report.get(SPEEDUP_VS_DISABLED_CACHE_KEY)),
    }


def _extract_trace_metrics(trace_payloads: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate trace evidence metrics for one case."""
    if not trace_payloads:
        return {
            "trace_hit_rate_percent": None,
            "trace_estimated_average_speedup": None,
            "trace_estimated_worst_case_speedup": None,
            "trace_metadata_fraction_percent": None,
        }
    hit_rates: list[float] = []
    average_speedups: list[float] = []
    worst_speedups: list[float] = []
    metadata_fractions: list[float] = []
    for payload in trace_payloads:
        hit_rate = _coerce_optional_float(payload.get("hit_rate_percent"))
        if hit_rate is not None:
            hit_rates.append(hit_rate)
        timing = payload.get("timing")
        if isinstance(timing, dict):
            average_speedup = _coerce_optional_float(timing.get("estimated_average_speedup"))
            worst_speedup = _coerce_optional_float(timing.get("estimated_worst_case_speedup"))
            metadata_fraction = _coerce_optional_float(timing.get("metadata_fraction_percent"))
            if average_speedup is not None:
                average_speedups.append(average_speedup)
            if worst_speedup is not None:
                worst_speedups.append(worst_speedup)
            if metadata_fraction is not None:
                metadata_fractions.append(metadata_fraction)
    return {
        "trace_hit_rate_percent": _mean(hit_rates),
        "trace_estimated_average_speedup": _mean(average_speedups),
        "trace_estimated_worst_case_speedup": _mean(worst_speedups),
        "trace_metadata_fraction_percent": _mean(metadata_fractions),
    }


def build_case_summary(
    *,
    case: CaseConfig,
    benchmark_report: Path | None,
    bundle_evidence: Path | None,
    trace_evidence_paths: tuple[Path, ...],
    external_timing_report: Path | None,
    status: str,
) -> CaseSummary:
    """Build one table row from validated benchmark and trace artifacts."""
    benchmark_metrics = _extract_benchmark_metrics(_load_json_if_exists(benchmark_report))
    external_metrics = _extract_external_metrics(_load_json_if_exists(external_timing_report))
    trace_payloads = [
        payload
        for path in trace_evidence_paths
        for payload in [_load_json_if_exists(path)]
        if payload is not None
    ]
    trace_metrics = _extract_trace_metrics(trace_payloads)
    baseline_seconds = (
        benchmark_metrics["baseline_mean_seconds"]
        if benchmark_metrics["baseline_mean_seconds"] is not None
        else external_metrics["baseline_mean_seconds"]
    )
    enabled_seconds = (
        benchmark_metrics["enabled_mean_seconds"]
        if benchmark_metrics["enabled_mean_seconds"] is not None
        else external_metrics["enabled_mean_seconds"]
    )
    speedup = (
        benchmark_metrics["speedup"]
        if benchmark_metrics["speedup"] is not None
        else external_metrics["speedup"]
    )
    baseline_variance = (
        benchmark_metrics["baseline_sample_variance_seconds"]
        if benchmark_metrics["baseline_sample_variance_seconds"] is not None
        else external_metrics["baseline_sample_variance_seconds"]
    )
    enabled_variance = (
        benchmark_metrics["enabled_sample_variance_seconds"]
        if benchmark_metrics["enabled_sample_variance_seconds"] is not None
        else external_metrics["enabled_sample_variance_seconds"]
    )
    baseline_stddev = (
        benchmark_metrics["baseline_sample_stddev_seconds"]
        if benchmark_metrics["baseline_sample_stddev_seconds"] is not None
        else external_metrics["baseline_sample_stddev_seconds"]
    )
    enabled_stddev = (
        benchmark_metrics["enabled_sample_stddev_seconds"]
        if benchmark_metrics["enabled_sample_stddev_seconds"] is not None
        else external_metrics["enabled_sample_stddev_seconds"]
    )
    baseline_timing_count = (
        benchmark_metrics["baseline_timing_count"]
        if benchmark_metrics["baseline_timing_count"] is not None
        else external_metrics["baseline_timing_count"]
    )
    enabled_timing_count = (
        benchmark_metrics["enabled_timing_count"]
        if benchmark_metrics["enabled_timing_count"] is not None
        else external_metrics["enabled_timing_count"]
    )
    baseline_ci_half_width = _mean_ci_half_width(
        baseline_stddev,
        baseline_timing_count,
    )
    enabled_ci_half_width = _mean_ci_half_width(
        enabled_stddev,
        enabled_timing_count,
    )
    speedup_ci_lower_bound, speedup_ci_upper_bound = _speedup_ci_bounds(
        baseline_seconds,
        enabled_seconds,
        baseline_ci_half_width,
        enabled_ci_half_width,
    )
    return CaseSummary(
        case_name=case.name,
        model=case.model,
        kind=case.kind,
        status=status,
        benchmark_report=str(benchmark_report) if benchmark_report is not None else None,
        bundle_evidence=str(bundle_evidence) if bundle_evidence is not None else None,
        trace_evidence=tuple(str(path) for path in trace_evidence_paths),
        external_timing_report=str(external_timing_report) if external_timing_report is not None else None,
        baseline_mean_seconds=baseline_seconds,
        enabled_mean_seconds=enabled_seconds,
        baseline_sample_variance_seconds=baseline_variance,
        enabled_sample_variance_seconds=enabled_variance,
        baseline_sample_stddev_seconds=baseline_stddev,
        enabled_sample_stddev_seconds=enabled_stddev,
        baseline_timing_count=baseline_timing_count,
        enabled_timing_count=enabled_timing_count,
        baseline_mean_95ci_half_width_seconds=baseline_ci_half_width,
        enabled_mean_95ci_half_width_seconds=enabled_ci_half_width,
        speedup_vs_disabled_cache=speedup,
        speedup_95ci_lower_bound=speedup_ci_lower_bound,
        speedup_95ci_upper_bound=speedup_ci_upper_bound,
        cache_attempts=benchmark_metrics["attempts"],
        cache_hits=benchmark_metrics["hits"],
        cache_hit_rate_percent=benchmark_metrics["hit_rate_percent"],
        max_abs_thermo_delta=benchmark_metrics["max_abs_thermo_delta"],
        trace_hit_rate_percent=trace_metrics["trace_hit_rate_percent"],
        trace_estimated_average_speedup=trace_metrics["trace_estimated_average_speedup"],
        trace_estimated_worst_case_speedup=trace_metrics["trace_estimated_worst_case_speedup"],
        trace_metadata_fraction_percent=trace_metrics["trace_metadata_fraction_percent"],
    )


def validate_case_summary_thresholds(case: CaseConfig, summary: CaseSummary) -> None:
    """Validate thresholds that depend on derived summary-table metrics."""
    if case.min_speedup_95ci_lower_bound is None:
        return
    _require(
        summary.speedup_95ci_lower_bound is not None,
        f"{case.name}: speedup 95% CI lower bound is unavailable",
    )
    _require(
        summary.speedup_95ci_lower_bound >= case.min_speedup_95ci_lower_bound,
        (
            f"{case.name}: speedup 95% CI lower bound "
            f"{summary.speedup_95ci_lower_bound:g} is below "
            f"{case.min_speedup_95ci_lower_bound:g}"
        ),
    )


def _format_table_value(value: Any) -> str:
    """Format values for markdown tables without losing numeric readability."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return MODEL_NAME_JOINER.join(value)
    return str(value)


def _summary_rows(case_summaries: list[CaseSummary]) -> list[dict[str, Any]]:
    """Return CSV/Markdown rows for the main paper summary table."""
    rows: list[dict[str, Any]] = []
    for summary in case_summaries:
        rows.append(
            {
                "case": summary.case_name,
                "model": summary.model,
                "kind": summary.kind,
                "status": summary.status,
                "baseline_mean_seconds": summary.baseline_mean_seconds,
                "enabled_mean_seconds": summary.enabled_mean_seconds,
                "baseline_sample_variance_seconds": summary.baseline_sample_variance_seconds,
                "enabled_sample_variance_seconds": summary.enabled_sample_variance_seconds,
                "baseline_sample_stddev_seconds": summary.baseline_sample_stddev_seconds,
                "enabled_sample_stddev_seconds": summary.enabled_sample_stddev_seconds,
                "baseline_timing_count": summary.baseline_timing_count,
                "enabled_timing_count": summary.enabled_timing_count,
                "baseline_mean_95ci_half_width_seconds": (
                    summary.baseline_mean_95ci_half_width_seconds
                ),
                "enabled_mean_95ci_half_width_seconds": (
                    summary.enabled_mean_95ci_half_width_seconds
                ),
                "speedup_vs_disabled_cache": summary.speedup_vs_disabled_cache,
                "speedup_95ci_lower_bound": summary.speedup_95ci_lower_bound,
                "speedup_95ci_upper_bound": summary.speedup_95ci_upper_bound,
                "cache_hit_rate_percent": summary.cache_hit_rate_percent,
                "cache_attempts": summary.cache_attempts,
                "cache_hits": summary.cache_hits,
                "trace_hit_rate_percent": summary.trace_hit_rate_percent,
                "trace_estimated_average_speedup": summary.trace_estimated_average_speedup,
                "trace_estimated_worst_case_speedup": summary.trace_estimated_worst_case_speedup,
                "trace_metadata_fraction_percent": summary.trace_metadata_fraction_percent,
                "max_abs_thermo_delta": summary.max_abs_thermo_delta,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a compact GitHub-flavored markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("| empty |\n| --- |\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_table_value(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson correlation for paired finite values."""
    if len(xs) < MIN_CORRELATION_SAMPLE_COUNT:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys))
    x_variance = sum((x_value - mean_x) ** 2 for x_value in xs)
    y_variance = sum((y_value - mean_y) ** 2 for y_value in ys)
    denominator = math.sqrt(x_variance * y_variance)
    return None if denominator == 0.0 else numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    """Return average ranks for Spearman correlation with tie support."""
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(sorted_pairs):
        tie_end = cursor + 1
        while tie_end < len(sorted_pairs) and sorted_pairs[tie_end][0] == sorted_pairs[cursor][0]:
            tie_end += 1
        average_rank = (cursor + 1 + tie_end) / 2.0
        for _, original_index in sorted_pairs[cursor:tie_end]:
            ranks[original_index] = average_rank
        cursor = tie_end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Return Spearman rank correlation for paired finite values."""
    if len(xs) < MIN_CORRELATION_SAMPLE_COUNT:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _metric_pairs(
    case_summaries: list[CaseSummary],
    x_name: str,
    y_name: str,
) -> tuple[list[float], list[float]]:
    """Return paired metric vectors by dataclass field name."""
    xs: list[float] = []
    ys: list[float] = []
    for summary in case_summaries:
        x_value = getattr(summary, x_name)
        y_value = getattr(summary, y_name)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            xs.append(float(x_value))
            ys.append(float(y_value))
    return xs, ys


def build_correlation_rows(case_summaries: list[CaseSummary]) -> list[dict[str, Any]]:
    """Build correlation rows for the paper appendix."""
    metric_pairs = (
        ("cache_hit_rate_percent", "speedup_vs_disabled_cache"),
        ("trace_hit_rate_percent", "trace_estimated_average_speedup"),
        ("trace_metadata_fraction_percent", "trace_estimated_average_speedup"),
        ("trace_estimated_average_speedup", "speedup_vs_disabled_cache"),
    )
    rows: list[dict[str, Any]] = []
    for x_name, y_name in metric_pairs:
        xs, ys = _metric_pairs(case_summaries, x_name, y_name)
        rows.append(
            {
                "x_metric": x_name,
                "y_metric": y_name,
                "n": len(xs),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
            }
        )
    return rows


def _svg_escape(text: str) -> str:
    """Escape a label for SVG text nodes."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_speedup_svg(path: Path, case_summaries: list[CaseSummary]) -> None:
    """Write a dependency-free bar chart of measured speedup by case."""
    points = [
        (summary.case_name, summary.speedup_vs_disabled_cache)
        for summary in case_summaries
        if summary.speedup_vs_disabled_cache is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        path.write_text(_empty_svg("No measured speedup values"), encoding="utf-8")
        return
    plot_width = SVG_WIDTH - SVG_MARGIN_LEFT - SVG_MARGIN_RIGHT
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    max_value = max(max(value for _, value in points), DEFAULT_MIN_SPEEDUP)
    bar_slot = plot_width / len(points)
    bar_width = bar_slot * (1.0 - BAR_GAP_RATIO)
    bars: list[str] = []
    for index, (label, value) in enumerate(points):
        bar_height = plot_height * value / max_value
        x_pos = SVG_MARGIN_LEFT + index * bar_slot + (bar_slot - bar_width) / 2.0
        y_pos = SVG_MARGIN_TOP + plot_height - bar_height
        bars.append(
            f'<rect x="{x_pos:.2f}" y="{y_pos:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="#2f6f9f"/>'
        )
        bars.append(
            f'<text x="{x_pos + bar_width / 2.0:.2f}" y="{y_pos - 8:.2f}" '
            f'text-anchor="middle" font-size="13">{value:.3g}x</text>'
        )
        bars.append(
            f'<text x="{x_pos + bar_width / 2.0:.2f}" y="{SVG_HEIGHT - 30}" '
            f'text-anchor="middle" font-size="12" transform="rotate(-25 '
            f'{x_pos + bar_width / 2.0:.2f} {SVG_HEIGHT - 30})">{_svg_escape(label)}</text>'
        )
    axis = _svg_axes("Speedup vs disabled cache", "case", "speedup")
    path.write_text(_svg_document(axis + "\n".join(bars)), encoding="utf-8")


def write_scatter_svg(
    path: Path,
    case_summaries: list[CaseSummary],
    *,
    x_field: str,
    y_field: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Write a dependency-free scatter plot for correlation inspection."""
    points = []
    for summary in case_summaries:
        x_value = getattr(summary, x_field)
        y_value = getattr(summary, y_field)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            points.append((summary.case_name, float(x_value), float(y_value)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        path.write_text(_empty_svg(f"No paired values for {title}"), encoding="utf-8")
        return
    plot_width = SVG_WIDTH - SVG_MARGIN_LEFT - SVG_MARGIN_RIGHT
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    x_values = [point[1] for point in points]
    y_values = [point[2] for point in points]
    x_min, x_max = _expanded_range(min(x_values), max(x_values))
    y_min, y_max = _expanded_range(min(y_values), max(y_values))
    circles: list[str] = []
    for label, x_value, y_value in points:
        x_pos = SVG_MARGIN_LEFT + plot_width * (x_value - x_min) / (x_max - x_min)
        y_pos = SVG_MARGIN_TOP + plot_height - plot_height * (y_value - y_min) / (y_max - y_min)
        circles.append(
            f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="{SCATTER_POINT_RADIUS}" fill="#b23a48"/>'
        )
        circles.append(
            f'<text x="{x_pos + 8:.2f}" y="{y_pos - 8:.2f}" font-size="12">{_svg_escape(label)}</text>'
        )
    axis = _svg_axes(title, x_label, y_label)
    path.write_text(_svg_document(axis + "\n".join(circles)), encoding="utf-8")


def _expanded_range(min_value: float, max_value: float) -> tuple[float, float]:
    """Return a nonzero plotting range with a small visual pad."""
    if min_value == max_value:
        pad = 1.0 if min_value == 0.0 else abs(min_value) * 0.1
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * 0.08
    return min_value - pad, max_value + pad


def _svg_axes(title: str, x_label: str, y_label: str) -> str:
    """Return shared SVG axes and labels."""
    plot_bottom = SVG_HEIGHT - SVG_MARGIN_BOTTOM
    return (
        f'<text x="{SVG_WIDTH / 2:.2f}" y="30" text-anchor="middle" font-size="22" '
        f'font-weight="700">{_svg_escape(title)}</text>'
        f'<line x1="{SVG_MARGIN_LEFT}" y1="{plot_bottom}" x2="{SVG_WIDTH - SVG_MARGIN_RIGHT}" '
        f'y2="{plot_bottom}" stroke="#222" stroke-width="1.5"/>'
        f'<line x1="{SVG_MARGIN_LEFT}" y1="{SVG_MARGIN_TOP}" x2="{SVG_MARGIN_LEFT}" '
        f'y2="{plot_bottom}" stroke="#222" stroke-width="1.5"/>'
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 8}" text-anchor="middle" '
        f'font-size="14">{_svg_escape(x_label)}</text>'
        f'<text x="20" y="{SVG_HEIGHT / 2:.2f}" text-anchor="middle" font-size="14" '
        f'transform="rotate(-90 20 {SVG_HEIGHT / 2:.2f})">{_svg_escape(y_label)}</text>'
    )


def _svg_document(body: str) -> str:
    """Wrap SVG body content in a complete document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" '
        f'height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{body}</svg>\n"
    )


def _empty_svg(message: str) -> str:
    """Return an SVG placeholder when a metric is unavailable."""
    return _svg_document(
        f'<text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT / 2:.2f}" text-anchor="middle" '
        f'font-size="20">{_svg_escape(message)}</text>'
    )


def write_paper_outputs(
    config: SuiteConfig,
    case_summaries: list[CaseSummary],
    command_records: list[CommandRecord],
    download_records: list[dict[str, Any]],
    gpu_record: dict[str, Any] | None,
    suite_evidence: dict[str, Any],
) -> dict[str, str]:
    """Write tables, correlations, figures, and a suite-level JSON summary."""
    tables_dir = config.output_dir / TABLES_DIR_NAME
    figures_dir = config.output_dir / FIGURES_DIR_NAME
    summary_rows = _summary_rows(case_summaries)
    correlation_rows = build_correlation_rows(case_summaries)
    case_summary_csv = tables_dir / "case_summary.csv"
    case_summary_md = tables_dir / "case_summary.md"
    correlation_csv = tables_dir / "correlation.csv"
    write_csv(case_summary_csv, summary_rows)
    write_markdown_table(case_summary_md, summary_rows)
    write_csv(correlation_csv, correlation_rows)
    speedup_svg = figures_dir / "speedup_by_case.svg"
    hit_rate_svg = figures_dir / "hit_rate_vs_speedup.svg"
    trace_svg = figures_dir / "trace_metadata_fraction_vs_speedup.svg"
    write_speedup_svg(speedup_svg, case_summaries)
    write_scatter_svg(
        hit_rate_svg,
        case_summaries,
        x_field="cache_hit_rate_percent",
        y_field="speedup_vs_disabled_cache",
        title="Cache hit rate vs measured speedup",
        x_label="cache hit rate (%)",
        y_label="measured speedup",
    )
    write_scatter_svg(
        trace_svg,
        case_summaries,
        x_field="trace_metadata_fraction_percent",
        y_field="trace_estimated_average_speedup",
        title="Trace metadata fraction vs estimated speedup",
        x_label="metadata build fraction (%)",
        y_label="trace estimated speedup",
    )
    summary_path = config.output_dir / SUMMARY_REPORT_NAME
    manifest_snapshot_path = write_manifest_snapshot(config)
    environment_snapshot_path = write_environment_snapshot(config, gpu_record)
    artifact_paths = {
        "environment_snapshot": environment_snapshot_path,
        "case_summary_csv": case_summary_csv,
        "case_summary_markdown": case_summary_md,
        "correlation_csv": correlation_csv,
        "speedup_svg": speedup_svg,
        "hit_rate_svg": hit_rate_svg,
        "trace_svg": trace_svg,
        "manifest_snapshot": manifest_snapshot_path,
    }
    artifact_fingerprints = {
        name: generated_artifact_record(path) for name, path in artifact_paths.items()
    }
    payload = {
        "provenance": collect_run_provenance(),
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "manifest": manifest_record(config),
            "output_dir": str(config.output_dir),
            "expected_gpus": config.expected_gpus,
            "required_models": list(config.required_models),
            "require_artifact_sha256": config.require_artifact_sha256,
        },
        "gpu_check": gpu_record,
        "suite_evidence": suite_evidence,
        "downloads": download_records,
        "cases": [asdict(summary) for summary in case_summaries],
        "correlations": correlation_rows,
        "commands": [asdict(record) for record in command_records],
        "command_log_fingerprints": command_log_fingerprints(command_records),
        "artifacts": {
            name: str(path) for name, path in artifact_paths.items()
        },
        "artifact_fingerprints": artifact_fingerprints,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "environment_snapshot": str(environment_snapshot_path),
        "case_summary_csv": str(case_summary_csv),
        "case_summary_markdown": str(case_summary_md),
        "correlation_csv": str(correlation_csv),
        "speedup_svg": str(speedup_svg),
        "hit_rate_svg": str(hit_rate_svg),
        "trace_svg": str(trace_svg),
        "manifest_snapshot": str(manifest_snapshot_path),
    }


def run_suite(
    config: SuiteConfig,
    *,
    dry_run: bool = False,
    collect_only: bool = False,
    skip_downloads: bool = False,
    skip_gpu_check: bool = False,
    allow_gpu_mismatch: bool = False,
    keep_going: bool = False,
    reuse_passed: bool = False,
) -> int:
    """Run the full cluster suite and write all paper-ready artifacts."""
    validate_suite_config(config)
    validate_required_artifacts_available(
        config,
        skip_downloads=skip_downloads,
        collect_only=collect_only,
        dry_run=dry_run,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    download_stage_count = 1 if skip_downloads or not config.artifacts else len(config.artifacts)
    total_stages = 1 + download_stage_count + len(config.cases) + 1
    stage_index = 1
    gpu_record = None
    if skip_gpu_check:
        _progress(config.name, stage_index, total_stages, "GPU check skipped")
    else:
        _progress(config.name, stage_index, total_stages, f"checking for {config.expected_gpus} GPUs")
        gpu_record = validate_gpu_count(config.expected_gpus, allow_gpu_mismatch)
    stage_index += 1

    download_records: list[dict[str, Any]] = []
    if skip_downloads:
        _progress(config.name, stage_index, total_stages, "artifact downloads skipped")
        stage_index += 1
    else:
        if not config.artifacts:
            _progress(config.name, stage_index, total_stages, "no artifacts declared")
            stage_index += 1
        else:
            for artifact in config.artifacts:
                _progress(config.name, stage_index, total_stages, f"checking artifact {artifact.name}")
                download_records.append(download_artifact(artifact, dry_run=dry_run or collect_only))
                stage_index += 1

    command_records: list[CommandRecord] = []
    case_summaries: list[CaseSummary] = []
    failed = False
    for case in config.cases:
        _progress(config.name, stage_index, total_stages, f"running case {case.name} ({case.model})")
        stage_index += 1
        case_status = CASE_STATUS_PASSED
        benchmark_report = case.benchmark_report
        bundle_evidence = case.bundle_evidence
        trace_evidence_paths = case.trace_evidence_paths
        external_timing_report: Path | None = None
        case_summary: CaseSummary | None = None
        if collect_only:
            external_timing_report = case.external_timing_report
        try:
            case_command_records: list[CommandRecord] = []
            if not collect_only:
                reused_outputs = (
                    None if dry_run or not reuse_passed else try_reuse_case_outputs(config, case)
                )
                if reused_outputs is not None:
                    (
                        benchmark_report,
                        bundle_evidence,
                        trace_evidence_paths,
                        external_timing_report,
                    ) = reused_outputs
                    case_status = CASE_STATUS_REUSED
                else:
                    preflight_records = run_case_preflight(config, case, dry_run=dry_run)
                    case_command_records.extend(preflight_records)
                    command_records.extend(preflight_records)
                    if any(record.returncode != SUCCESS_RETURN_CODE for record in preflight_records):
                        raise ClusterSuiteError(f"{case.name}: preflight command failed")
                    if case.kind == "sevennet_lammps":
                        benchmark_report, bundle_evidence, trace_evidence_paths, records = run_sevennet_case(
                            config,
                            case,
                            dry_run=dry_run,
                        )
                        case_command_records.extend(records)
                        command_records.extend(records)
                    elif case.kind == "external_pair":
                        (
                            benchmark_report,
                            bundle_evidence,
                            trace_evidence_paths,
                            external_timing_report,
                            records,
                        ) = run_external_pair_case(config, case, dry_run=dry_run)
                        case_command_records.extend(records)
                        command_records.extend(records)
                    elif case.kind == "trace_only":
                        trace_evidence_paths, records = run_trace_only_case(
                            config,
                            case,
                            dry_run=dry_run,
                        )
                        case_command_records.extend(records)
                        command_records.extend(records)
            validate_case_outputs(
                case,
                benchmark_report,
                bundle_evidence,
                trace_evidence_paths,
                external_timing_report,
                dry_run=dry_run,
            )
            if any(record.returncode != SUCCESS_RETURN_CODE for record in case_command_records):
                raise ClusterSuiteError(f"{case.name}: one or more commands failed")
            case_summary = build_case_summary(
                case=case,
                benchmark_report=benchmark_report,
                bundle_evidence=bundle_evidence,
                trace_evidence_paths=trace_evidence_paths,
                external_timing_report=external_timing_report,
                status=case_status,
            )
            if not dry_run:
                validate_case_summary_thresholds(case, case_summary)
        except (
            ClusterSuiteError,
            benchmark_check.ReportCheckError,
            trace_check.TraceCheckError,
            bundle_check.EvidenceBundleError,
        ) as exc:
            case_status = f"failed: {exc}"
            failed = True
            case_summary = build_case_summary(
                case=case,
                benchmark_report=benchmark_report,
                bundle_evidence=bundle_evidence,
                trace_evidence_paths=trace_evidence_paths,
                external_timing_report=external_timing_report,
                status=case_status,
            )
            if not keep_going:
                print(f"[{config.name}] {case_status}", file=sys.stderr)
                return 1
        _require(case_summary is not None, f"{case.name}: missing case summary")
        case_summaries.append(case_summary)

    _progress(config.name, stage_index, total_stages, "writing tables, correlations, and figures")
    suite_evidence = validate_suite_evidence(
        config,
        case_summaries,
        dry_run=dry_run,
    )
    artifacts = write_paper_outputs(
        config,
        case_summaries,
        command_records,
        download_records,
        gpu_record,
        suite_evidence,
    )
    print(json.dumps(artifacts, indent=2), flush=True)
    return 1 if failed else SUCCESS_RETURN_CODE


def write_template(path: Path) -> None:
    """Write a commented TOML template for a three-model paper suite."""
    template = f"""# IsoDelta-Halo cluster paper suite manifest.
# Fill in real dataset/checkpoint URLs and commands for your cluster. The runner
# verifies SHA-256 values when provided and fails if required model cases are
# missing, so this template is meant to be edited rather than blindly executed.

[suite]
name = "icpp-isodelta-8gpu"
output_dir = "isodelta_cluster_paper_runs"
expected_gpus = {DEFAULT_EXPECTED_GPU_COUNT}
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 5
command_timeout_seconds = 7200
benchmark_timeout_seconds = 3600
min_speedup = 1.05
# Optional but recommended for the final paper run: require the conservative
# 95% CI lower bound for speedup to stay above no-speedup.
min_speedup_95ci_lower_bound = 1.0
min_hit_rate_percent = 50.0
min_trace_hit_rate_percent = 50.0
min_trace_estimated_speedup = 1.05
min_trace_metadata_fraction_percent = 5.0
min_trace_count = 3
min_distinct_trace_models = 3

# Add one artifact per dataset, checkpoint, input deck, or runtime bundle.
# Paths are relative to this manifest unless absolute. URLs may be https:// or
# file://. Keep sha256 values once the paper artifact is frozen.
[[artifacts]]
name = "shared-dataset"
path = "data/shared_dataset.ext"
url = "https://example.org/replace-with-real-dataset"
sha256 = "replace-with-real-sha256"
required_by = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-lammps"
model = "SevenNet"
kind = "sevennet_lammps"
preflight_command = 'python -c "import sevenn"'
lammps_command = "mpiexec -n 8 lmp"
input = "inputs/in.sevennet"
work_dir = "inputs"
lammps_root = "../lammps"
repeat_count = 5
min_speedup = 1.05
min_hit_rate_percent = 50.0
min_enabled_cache_attempts = 1
min_enabled_cache_hits = 1
trace_input = "traces/sevennet_trace.json"
required_trace_models = ["SevenNet"]
artifacts = ["shared-dataset"]

[[cases]]
name = "mace-external"
model = "MACE"
kind = "external_pair"
preflight_command = 'python -c "import mace"'
disabled_command = "python scripts/run_mace_case.py --mode baseline --dataset data/shared_dataset.ext"
enabled_command = "python scripts/run_mace_case.py --mode isodelta --dataset data/shared_dataset.ext"
repeat_count = 5
min_speedup = 1.05
trace_input = "traces/mace_trace.json"
required_trace_models = ["MACE"]
artifacts = ["shared-dataset"]

[[cases]]
name = "nequip-external"
model = "NequIP"
kind = "external_pair"
preflight_command = 'python -c "import nequip"'
disabled_command = "python scripts/run_nequip_case.py --mode baseline --dataset data/shared_dataset.ext"
enabled_command = "python scripts/run_nequip_case.py --mode isodelta --dataset data/shared_dataset.ext"
repeat_count = 5
min_speedup = 1.05
trace_input = "traces/nequip_trace.json"
required_trace_models = ["NequIP"]
artifacts = ["shared-dataset"]
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")


def _bash_quote(value: str | Path) -> str:
    """Quote one literal for the POSIX shell used by generated SLURM scripts."""
    return shlex.quote(str(value))


def _append_bash_array_args(lines: list[str], *values: str | Path) -> None:
    """Append one Bash array extension line with safely quoted literals."""
    quoted_values = " ".join(_bash_quote(value) for value in values)
    lines.append(f"COMMON_ARGS+=({quoted_values})")


def write_slurm_script(
    path: Path,
    config: SuiteConfig,
    *,
    collect_only: bool = False,
    dry_run: bool = False,
    skip_downloads: bool = False,
    skip_gpu_check: bool = False,
    allow_gpu_mismatch: bool = False,
    keep_going: bool = False,
    reuse_passed: bool = False,
    job_name: str = DEFAULT_SLURM_JOB_NAME,
    time_limit: str = DEFAULT_SLURM_TIME_LIMIT,
    cpus_per_task: int = DEFAULT_SLURM_CPUS_PER_TASK,
) -> None:
    """Write a commented SLURM wrapper that runs the plan and full suite."""
    validate_suite_config(config)
    _require(config.expected_gpus >= MIN_REQUIRED_CASE_COUNT, "SLURM GPU count must be positive")
    _require(cpus_per_task >= MIN_REQUIRED_CASE_COUNT, "SLURM cpus-per-task must be positive")

    plan_path = config.output_dir / PLAN_REPORT_NAME
    slurm_job_name = _safe_name(job_name)
    lines = [
        "#!/usr/bin/env bash",
        "# IsoDelta-Halo cluster paper suite launcher.",
        "# Submit with: sbatch <this-file>",
        "# The script writes a preflight plan first, then runs the full suite.",
        f"#SBATCH --job-name={slurm_job_name}",
        f"#SBATCH --gres=gpu:{config.expected_gpus}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={SLURM_LOG_DIR_NAME}/%x-%j.out",
        f"#SBATCH --error={SLURM_LOG_DIR_NAME}/%x-%j.err",
        "",
        "set -euo pipefail",
        "",
        "# Override PYTHON_BIN or SUITE_RUNNER at submit time if the cluster uses modules.",
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        'SUITE_RUNNER="${SUITE_RUNNER:-tools/run_isodelta_cluster_paper_suite.py}"',
        f"MANIFEST_PATH={_bash_quote(config.manifest_path)}",
        f"PLAN_OUTPUT={_bash_quote(plan_path)}",
        f"PREFLIGHT_OUTPUT={_bash_quote(config.output_dir / PREFLIGHT_REPORT_NAME)}",
        "",
        "# Keep scheduler stdout/stderr directories explicit and reproducible.",
        f"mkdir -p {_bash_quote(SLURM_LOG_DIR_NAME)}",
        f"mkdir -p {_bash_quote(config.output_dir)}",
        "",
        "# COMMON_ARGS is reused for planning and execution to prevent argument drift.",
        'COMMON_ARGS=(--manifest "$MANIFEST_PATH")',
    ]
    _append_bash_array_args(lines, "--output-dir", config.output_dir)
    _append_bash_array_args(lines, "--expected-gpus", str(config.expected_gpus))
    if collect_only:
        _append_bash_array_args(lines, "--collect-only")
    if dry_run:
        _append_bash_array_args(lines, "--dry-run")
    if skip_downloads:
        _append_bash_array_args(lines, "--skip-downloads")
    if skip_gpu_check:
        _append_bash_array_args(lines, "--skip-gpu-check")
    if allow_gpu_mismatch:
        _append_bash_array_args(lines, "--allow-gpu-mismatch")
    if keep_going:
        _append_bash_array_args(lines, "--keep-going")
    if reuse_passed:
        _append_bash_array_args(lines, "--reuse-passed")
    lines.extend(
        [
            "",
            "# Run artifact, GPU, and model import checks before launching model runs.",
            '"$PYTHON_BIN" "$SUITE_RUNNER" "${COMMON_ARGS[@]}" --preflight-only --preflight-output "$PREFLIGHT_OUTPUT"',
            "",
            "# Generate the auditable plan JSON before launching model runs.",
            '"$PYTHON_BIN" "$SUITE_RUNNER" "${COMMON_ARGS[@]}" --plan-only --plan-output "$PLAN_OUTPUT"',
            "",
            "# Run SevenNet, MACE, NequIP, and any extra manifest cases.",
            '"$PYTHON_BIN" "$SUITE_RUNNER" "${COMMON_ARGS[@]}"',
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the cluster paper suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="TOML suite manifest")
    parser.add_argument("--write-template", type=Path, help="Write a commented TOML template and exit")
    parser.add_argument("--write-slurm-script", type=Path, help="Write a commented SLURM sbatch script and exit")
    parser.add_argument("--verify-output-bundle", type=Path, help="Verify summary artifact and log fingerprints")
    parser.add_argument("--readiness-check", action="store_true", help="Audit a manifest before final paper execution")
    parser.add_argument("--prepare-artifacts", action="store_true", help="Download and verify artifacts without using GPUs")
    parser.add_argument("--preflight-only", action="store_true", help="Run artifact, GPU, and case preflight checks only")
    parser.add_argument("--preflight-output", type=Path, help="Path for --preflight-only JSON output")
    parser.add_argument("--plan-only", action="store_true", help="Write a preflight JSON plan and exit")
    parser.add_argument("--plan-output", type=Path, help="Path for --plan-only JSON output")
    parser.add_argument("--output-dir", type=Path, help="Override suite.output_dir")
    parser.add_argument("--expected-gpus", type=int, help="Override suite.expected_gpus")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned outputs without executing commands")
    parser.add_argument("--collect-only", action="store_true", help="Only collect and validate existing reports")
    parser.add_argument("--skip-downloads", action="store_true", help="Do not download missing artifacts")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Do not probe GPU count")
    parser.add_argument("--allow-gpu-mismatch", action="store_true", help="Record GPU mismatch instead of failing")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed case and mark it in tables")
    parser.add_argument("--reuse-passed", action="store_true", help="Reuse existing case outputs that pass current gates")
    parser.add_argument("--slurm-job-name", default=DEFAULT_SLURM_JOB_NAME, help="Job name for --write-slurm-script")
    parser.add_argument("--slurm-time-limit", default=DEFAULT_SLURM_TIME_LIMIT, help="Time limit for --write-slurm-script")
    parser.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=DEFAULT_SLURM_CPUS_PER_TASK,
        help="CPU cores requested by --write-slurm-script",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(config: SuiteConfig, args: argparse.Namespace) -> SuiteConfig:
    """Return a config with CLI output/GPU overrides applied."""
    return SuiteConfig(
        name=config.name,
        manifest_path=config.manifest_path,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else config.output_dir,
        expected_gpus=args.expected_gpus if args.expected_gpus is not None else config.expected_gpus,
        required_models=config.required_models,
        min_trace_count=config.min_trace_count,
        min_distinct_trace_models=config.min_distinct_trace_models,
        require_artifact_sha256=config.require_artifact_sha256,
        artifacts=config.artifacts,
        cases=config.cases,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the cluster paper suite."""
    args = parse_args(argv)
    if args.write_template is not None:
        write_template(args.write_template)
        print(f"Wrote IsoDelta-Halo cluster suite template to {args.write_template}")
        return SUCCESS_RETURN_CODE
    if args.verify_output_bundle is not None:
        try:
            verification = verify_output_bundle(args.verify_output_bundle)
        except ClusterSuiteError as exc:
            print(f"IsoDelta-Halo output bundle verification failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(verification, indent=2))
        return SUCCESS_RETURN_CODE
    if args.manifest is None:
        raise SystemExit("--manifest is required unless --write-template or --verify-output-bundle is used")
    try:
        config = _apply_cli_overrides(load_manifest(args.manifest), args)
        _require(
            args.preflight_output is None or args.preflight_only,
            "--preflight-output requires --preflight-only",
        )
        if args.readiness_check:
            _require(args.write_slurm_script is None, "--readiness-check cannot be combined with --write-slurm-script")
            _require(not args.plan_only, "--readiness-check cannot be combined with --plan-only")
            _require(not args.prepare_artifacts, "--readiness-check cannot be combined with --prepare-artifacts")
            _require(not args.preflight_only, "--readiness-check cannot be combined with --preflight-only")
            report = build_readiness_report(config)
            print(json.dumps(report, indent=2))
            return SUCCESS_RETURN_CODE if report["status"] == "ready" else 1
        if args.prepare_artifacts:
            _require(args.write_slurm_script is None, "--prepare-artifacts cannot be combined with --write-slurm-script")
            _require(not args.plan_only, "--prepare-artifacts cannot be combined with --plan-only")
            _require(not args.preflight_only, "--prepare-artifacts cannot be combined with --preflight-only")
            _require(not args.collect_only, "--prepare-artifacts cannot be combined with --collect-only")
            _require(not args.skip_downloads, "--prepare-artifacts cannot be combined with --skip-downloads")
            report = prepare_artifacts(config, dry_run=args.dry_run)
            print(json.dumps({"artifact_preparation_report": report["report_path"], "status": report["status"]}, indent=2))
            return SUCCESS_RETURN_CODE
        if args.preflight_only:
            _require(args.write_slurm_script is None, "--preflight-only cannot be combined with --write-slurm-script")
            _require(not args.plan_only, "--preflight-only cannot be combined with --plan-only")
            _require(not args.collect_only, "--preflight-only cannot be combined with --collect-only")
            report = run_preflight_only(
                config,
                dry_run=args.dry_run,
                skip_downloads=args.skip_downloads,
                skip_gpu_check=args.skip_gpu_check,
                allow_gpu_mismatch=args.allow_gpu_mismatch,
                report_path=args.preflight_output,
            )
            print(json.dumps({"preflight_report": report["report_path"], "status": report["status"]}, indent=2))
            return SUCCESS_RETURN_CODE if report["status"] != PREFLIGHT_STATUS_FAILED else 1
        if args.write_slurm_script is not None:
            _require(not args.plan_only, "--write-slurm-script cannot be combined with --plan-only")
            write_slurm_script(
                args.write_slurm_script,
                config,
                collect_only=args.collect_only,
                dry_run=args.dry_run,
                skip_downloads=args.skip_downloads,
                skip_gpu_check=args.skip_gpu_check,
                allow_gpu_mismatch=args.allow_gpu_mismatch,
                keep_going=args.keep_going,
                reuse_passed=args.reuse_passed,
                job_name=args.slurm_job_name,
                time_limit=args.slurm_time_limit,
                cpus_per_task=args.slurm_cpus_per_task,
            )
            print(f"Wrote IsoDelta-Halo SLURM launcher to {args.write_slurm_script}")
            return SUCCESS_RETURN_CODE
        if args.plan_only:
            plan_path = (
                args.plan_output
                if args.plan_output is not None
                else config.output_dir / PLAN_REPORT_NAME
            )
            written_plan = write_run_plan(
                config,
                plan_path,
                collect_only=args.collect_only,
                skip_downloads=args.skip_downloads,
                skip_gpu_check=args.skip_gpu_check,
                reuse_passed=args.reuse_passed,
            )
            print(json.dumps({"plan_json": str(written_plan)}, indent=2))
            return SUCCESS_RETURN_CODE
        return run_suite(
            config,
            dry_run=args.dry_run,
            collect_only=args.collect_only,
            skip_downloads=args.skip_downloads,
            skip_gpu_check=args.skip_gpu_check,
            allow_gpu_mismatch=args.allow_gpu_mismatch,
            keep_going=args.keep_going,
            reuse_passed=args.reuse_passed,
        )
    except ClusterSuiteError as exc:
        print(f"IsoDelta-Halo cluster paper suite failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
