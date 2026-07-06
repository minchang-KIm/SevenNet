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
from dataclasses import asdict, dataclass, field, replace
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
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ElementTree


# Constants are named because this script becomes part of the experimental
# method: reviewers should see every gate and unit without hunting literals.
REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_SCHEMA_VERSION = "isodelta-cluster-paper-suite-v1"
READINESS_SCHEMA_VERSION = "isodelta-cluster-readiness-v1"
ARTIFACT_PREPARATION_SCHEMA_VERSION = "isodelta-artifact-preparation-v1"
PREFLIGHT_REPORT_SCHEMA_VERSION = "isodelta-cluster-preflight-v1"
PIPELINE_REPORT_SCHEMA_VERSION = "isodelta-cluster-pipeline-v1"
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
ABLATION_OVERRIDE_CASE_KINDS = FINAL_PAPER_PAIRED_CASE_KINDS
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
PIPELINE_STATUS_PASSED = "passed"
PIPELINE_STATUS_FAILED = "failed"
PIPELINE_STATUS_PLANNED = "planned"
PIPELINE_REPORT_PASSED_STATUS_ERROR = (
    "pipeline report status must be 'passed' before publication verification"
)
PIPELINE_REQUIRED_STAGES_ERROR = (
    "passed pipeline report must contain the final-paper stages in order"
)
PIPELINE_BUNDLE_VERIFICATION_REQUIRED_ERROR = (
    "output_bundle_verification.status must be 'passed' for a passed pipeline report"
)
PIPELINE_DRY_RUN_PASSED_ERROR = "passed pipeline report must record dry_run=false"
PIPELINE_SUITE_GPU_ERROR = (
    f"pipeline suite expected_gpus must be at least {DEFAULT_EXPECTED_GPU_COUNT}"
)
PIPELINE_SUITE_REQUIRED_MODELS_ERROR = (
    "pipeline suite required_models must include SevenNet, MACE, and NequIP"
)
PIPELINE_REQUIRED_MODE_KEYS = (
    "dry_run",
    "skip_downloads",
    "skip_gpu_check",
    "allow_gpu_mismatch",
    "keep_going",
    "reuse_passed",
)
PIPELINE_STAGE_STATUS_READY = "ready"
PIPELINE_STAGE_READINESS = "readiness"
PIPELINE_STAGE_PREPARE_ARTIFACTS = "prepare_artifacts"
PIPELINE_STAGE_PREFLIGHT = "preflight"
PIPELINE_STAGE_PLAN = "plan"
PIPELINE_STAGE_RUN_SUITE = "run_suite"
PIPELINE_STAGE_VERIFY_OUTPUT_BUNDLE = "verify_output_bundle"
REQUIRED_PIPELINE_STAGE_NAMES = (
    PIPELINE_STAGE_READINESS,
    PIPELINE_STAGE_PREPARE_ARTIFACTS,
    PIPELINE_STAGE_PREFLIGHT,
    PIPELINE_STAGE_PLAN,
    PIPELINE_STAGE_RUN_SUITE,
    PIPELINE_STAGE_VERIFY_OUTPUT_BUNDLE,
)
PIPELINE_SUCCESS_STAGE_STATUSES = {
    PIPELINE_STAGE_READINESS: (PIPELINE_STAGE_STATUS_READY,),
    PIPELINE_STAGE_PREPARE_ARTIFACTS: (
        PIPELINE_STAGE_STATUS_READY,
        PREFLIGHT_STATUS_SKIPPED,
    ),
    PIPELINE_STAGE_PREFLIGHT: (PREFLIGHT_STATUS_PASSED,),
    PIPELINE_STAGE_PLAN: (PIPELINE_STATUS_PASSED,),
    PIPELINE_STAGE_RUN_SUITE: (PIPELINE_STATUS_PASSED,),
    PIPELINE_STAGE_VERIFY_OUTPUT_BUNDLE: (PIPELINE_STATUS_PASSED,),
}
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
PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME = "preflight_environment_snapshot.json"
READINESS_REPORT_NAME = "readiness_report.json"
PIPELINE_REPORT_NAME = "pipeline_report.json"
MANIFEST_SNAPSHOT_NAME = "isodelta_cluster_suite_manifest.toml"
SLURM_LOG_DIR_NAME = "slurm_logs"
ENVIRONMENT_SNAPSHOT_NAME = "environment_snapshot.json"
STAGE_REPORT_FINGERPRINTS_KEY = "stage_report_fingerprints"
OUTPUT_BUNDLE_VERIFICATION_KEY = "output_bundle_verification"
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
MODE_CONTROLS_KEY = "mode_controls"
TIMING_MODES_KEY = "timing_modes"
ENV_FLAG_FALSE_VALUES_KEY = "env_flag_false_values"
COMMANDS_KEY = "commands"
COMMAND_LOG_FINGERPRINTS_KEY = "command_log_fingerprints"
EVIDENCE_FINGERPRINTS_KEY = "evidence_fingerprints"
TRACE_EVIDENCE_KEY = "trace_evidence"
CASE_EVIDENCE_FINGERPRINT_FIELDS = (
    "benchmark_report",
    "bundle_evidence",
    "external_timing_report",
)
LOGS_DIR_NAME = "logs"
CASES_DIR_NAME = "cases"
TABLES_DIR_NAME = "tables"
FIGURES_DIR_NAME = "figures"
REQUIRED_PAPER_ARTIFACT_NAMES = (
    "environment_snapshot",
    "case_summary_csv",
    "case_summary_markdown",
    "correlation_csv",
    "command_timing_csv",
    "command_timing_markdown",
    "repeat_timing_csv",
    "repeat_timing_markdown",
    "speedup_svg",
    "hit_rate_svg",
    "trace_svg",
    "manifest_snapshot",
)
PAPER_CASE_SUMMARY_COLUMNS = ("case", "model", "kind", "status")
PAPER_CASE_SUMMARY_FIELD_MAP = {"case": "case_name"}
PAPER_CORRELATION_COLUMNS = ("x_metric", "y_metric", "n", "pearson", "spearman")
PAPER_COMMAND_TIMING_COLUMNS = (
    "name",
    "returncode",
    "elapsed_seconds",
    "stdout_path",
    "stderr_path",
    "cwd",
)
PAPER_REPEAT_TIMING_COLUMNS = (
    "case",
    "model",
    "kind",
    "source",
    "mode",
    "repeat_index",
    "elapsed_seconds",
    "returncode",
)
BENCHMARK_TIMING_SOURCE = "benchmark_report"
EXTERNAL_TIMING_SOURCE = "external_timing_report"
PAPER_SVG_ARTIFACT_NAMES = ("speedup_svg", "hit_rate_svg", "trace_svg")
SPEEDUP_SVG_EMPTY_MESSAGE = "No measured speedup values"
HIT_RATE_SCATTER_TITLE = "Cache hit rate vs measured speedup"
TRACE_METADATA_SCATTER_TITLE = "Trace metadata fraction vs estimated speedup"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_PROGRESS_INTERVAL_BYTES = 64 * DOWNLOAD_CHUNK_BYTES
DOWNLOAD_PROGRESS_PERCENT_DECIMALS = 1
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
EXTERNAL_DISABLED_COMMAND_LABEL = "disabled"
EXTERNAL_ENABLED_COMMAND_LABEL = "enabled"
SVG_WIDTH = 960
SVG_HEIGHT = 540
SVG_MARGIN_LEFT = 88
SVG_MARGIN_RIGHT = 40
SVG_MARGIN_TOP = 56
SVG_MARGIN_BOTTOM = 88
BAR_GAP_RATIO = 0.28
SCATTER_POINT_RADIUS = 5
MODEL_NAME_JOINER = ", "
CORRELATION_METRIC_PAIRS = (
    ("cache_hit_rate_percent", "speedup_vs_disabled_cache"),
    ("trace_hit_rate_percent", "trace_estimated_average_speedup"),
    ("trace_metadata_fraction_percent", "trace_estimated_average_speedup"),
    ("trace_estimated_average_speedup", "speedup_vs_disabled_cache"),
)
NVIDIA_SMI_TIMEOUT_SECONDS = 20.0
TORCH_GPU_TIMEOUT_SECONDS = 30.0
GIT_METADATA_TIMEOUT_SECONDS = 10.0
SEVENNET_DISABLE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
SEVENNET_PROFILE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
SEVENNET_PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
ENV_FLAG_ENABLED = "1"
ENV_FLAG_FALSE_VALUES = ("", "0", "false", "no", "off")
MODE_CONTROL_ENV_KEYS = (
    SEVENNET_DISABLE_ENV,
    SEVENNET_PROFILE_ENV,
    SEVENNET_PRINT_INFO_ENV,
)
COMMAND_RUNTIME_ENVIRONMENT_VARIABLE_NAMES = (
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
COMMAND_ENV_SNAPSHOT_KEYS = (
    MODE_CONTROL_ENV_KEYS
    + ENVIRONMENT_VARIABLE_NAMES
    + COMMAND_RUNTIME_ENVIRONMENT_VARIABLE_NAMES
)
BASELINE_CASE_NAME = "baseline-disabled"
ISODELTA_CASE_NAME = "isodelta-enabled"
ABLATION_MODE_PAIRED = "paired"
ABLATION_MODE_BASELINE_ONLY = BASELINE_CASE_NAME
ABLATION_MODE_ENABLED_ONLY = ISODELTA_CASE_NAME
ABLATION_MODE_CHOICES = (
    ABLATION_MODE_PAIRED,
    ABLATION_MODE_BASELINE_ONLY,
    ABLATION_MODE_ENABLED_ONLY,
)
ABLATION_MODE_BENCHMARK_CASES = {
    ABLATION_MODE_PAIRED: (BASELINE_CASE_NAME, ISODELTA_CASE_NAME),
    ABLATION_MODE_BASELINE_ONLY: (BASELINE_CASE_NAME,),
    ABLATION_MODE_ENABLED_ONLY: (ISODELTA_CASE_NAME,),
}
ABLATION_MODE_EXTERNAL_TIMING_MODES = {
    ABLATION_MODE_PAIRED: (
        EXTERNAL_DISABLED_COMMAND_LABEL,
        EXTERNAL_ENABLED_COMMAND_LABEL,
    ),
    ABLATION_MODE_BASELINE_ONLY: (EXTERNAL_DISABLED_COMMAND_LABEL,),
    ABLATION_MODE_ENABLED_ONLY: (EXTERNAL_ENABLED_COMMAND_LABEL,),
}
PIPELINE_ALLOWED_RUNTIME_OVERRIDE_KEYS = ("ablation_mode",)
PIPELINE_UNSUPPORTED_RUNTIME_OVERRIDE_ERROR = (
    "pipeline suite runtime_overrides contains unsupported keys"
)
PIPELINE_ONE_SIDED_RUNTIME_OVERRIDE_ERROR = (
    "passed pipeline report only permits paired ablation runtime overrides"
)
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
    ablation_mode: str = ABLATION_MODE_PAIRED
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
    runtime_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandRecord:
    """Record one launched command for the suite report."""

    name: str
    command: str | list[str]
    returncode: int
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str
    cwd: str
    tracked_env: dict[str, str | None]


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


def _as_json_bool(value: Any, field_name: str) -> bool:
    """Return a JSON boolean without accepting numeric aliases."""
    _require(isinstance(value, bool), f"{field_name} must be boolean")
    return value


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


def _as_json_nonnegative_number(value: Any, field_name: str) -> float:
    """Return a nonnegative finite JSON number."""
    numeric_value = _as_json_number(value, field_name)
    _require(numeric_value >= MIN_NONNEGATIVE_VALUE, f"{field_name} must be nonnegative")
    return numeric_value


def _as_json_optional_nonnegative_number(value: Any, field_name: str) -> float | None:
    """Return an optional nonnegative finite JSON number."""
    if value is None:
        return None
    return _as_json_nonnegative_number(value, field_name)


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
        _as_string(key, f"{field_name}.key"): _as_env_value(
            raw_value,
            f"{field_name}.{key}",
        )
        for key, raw_value in mapping.items()
    }


def _as_env_value(value: Any, field_name: str) -> str:
    """Return a manifest environment value, allowing explicit empty strings."""
    _require(isinstance(value, str), f"{field_name} must be a string")
    return value.strip()


def env_flag_is_enabled(value: str | None) -> bool:
    """Return whether a runtime env flag is enabled under PairE3GNN rules."""
    if value is None:
        return False
    return value.strip().lower() not in ENV_FLAG_FALSE_VALUES


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


def _uses_paired_ablation_mode(case: CaseConfig) -> bool:
    """Return whether a SevenNet case generates paired paper timing evidence."""
    return case.ablation_mode == ABLATION_MODE_PAIRED


def _is_final_paper_paired_case(case: CaseConfig) -> bool:
    """Return whether one case can count toward final enabled/disabled timing."""
    if case.kind in FINAL_PAPER_PAIRED_CASE_KINDS:
        return _uses_paired_ablation_mode(case)
    return False


def _has_one_sided_ablation_case(config: SuiteConfig) -> bool:
    """Return whether the suite contains ablation-only timing cases."""
    return any(
        case.kind in ABLATION_OVERRIDE_CASE_KINDS
        and not _uses_paired_ablation_mode(case)
        for case in config.cases
    )


def _external_timing_modes_for_ablation(case: CaseConfig) -> tuple[str, ...]:
    """Return the external command modes requested by one ablation setting."""
    return ABLATION_MODE_EXTERNAL_TIMING_MODES[case.ablation_mode]


def _external_pair_command_requirement_errors(case: CaseConfig) -> list[str]:
    """Return missing external commands for the requested timing modes."""
    timing_modes = _external_timing_modes_for_ablation(case)
    errors: list[str] = []
    if EXTERNAL_DISABLED_COMMAND_LABEL in timing_modes and not case.disabled_command:
        errors.append("disabled_command is required")
    if EXTERNAL_ENABLED_COMMAND_LABEL in timing_modes and not case.enabled_command:
        errors.append("enabled_command is required")
    return errors


def _apply_ablation_mode_override(
    config: SuiteConfig,
    ablation_mode_override: str | None,
) -> SuiteConfig:
    """Return a suite with runtime-timing cases forced to one ablation mode."""
    if ablation_mode_override is None:
        return config
    _require(
        ablation_mode_override in ABLATION_MODE_CHOICES,
        "--ablation-mode-override must be one of "
        + MODEL_NAME_JOINER.join(ABLATION_MODE_CHOICES),
    )
    overridden_cases: list[CaseConfig] = []
    overridden_case_count = 0
    for case in config.cases:
        if case.kind in ABLATION_OVERRIDE_CASE_KINDS:
            overridden_cases.append(replace(case, ablation_mode=ablation_mode_override))
            overridden_case_count += 1
        else:
            overridden_cases.append(case)
    _require(
        overridden_case_count > 0,
        "--ablation-mode-override requires at least one sevennet_lammps or "
        "external_pair case",
    )
    runtime_overrides = dict(config.runtime_overrides)
    runtime_overrides["ablation_mode"] = ablation_mode_override
    return replace(
        config,
        cases=tuple(overridden_cases),
        runtime_overrides=runtime_overrides,
    )


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
        case.ablation_mode in ABLATION_MODE_CHOICES,
        f"{case.name}: ablation_mode must be one of "
        + MODEL_NAME_JOINER.join(ABLATION_MODE_CHOICES),
    )
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
                ablation_mode=_as_string(
                    case.get("ablation_mode", ABLATION_MODE_PAIRED),
                    f"cases[{index}].ablation_mode",
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
            command_errors = _external_pair_command_requirement_errors(case)
            _require(
                not command_errors,
                f"{case.name}: invalid requested timing commands: "
                + MODEL_NAME_JOINER.join(command_errors),
            )
            mode_control_errors = _external_pair_mode_control_errors(case)
            _require(
                not mode_control_errors,
                f"{case.name}: invalid disabled/enabled mode controls: "
                + MODEL_NAME_JOINER.join(mode_control_errors),
            )
        elif case.kind == "trace_only":
            _require(
                case.ablation_mode == ABLATION_MODE_PAIRED,
                f"{case.name}: ablation_mode is only supported for sevennet_lammps cases",
            )
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
        and _is_final_paper_paired_case(case)
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
    one_sided_ablation_cases = [
        case.name
        for case in config.cases
        if case.kind == "sevennet_lammps" and not _uses_paired_ablation_mode(case)
    ]
    checks.append(
        _readiness_record(
            "sevennet_final_paper_ablation_mode",
            not one_sided_ablation_cases,
            "one-sided SevenNet cases are ablation-only: "
            + MODEL_NAME_JOINER.join(one_sided_ablation_cases)
            if one_sided_ablation_cases
            else "SevenNet final-paper cases use paired ablation_mode",
        )
    )
    one_sided_external_pair_cases = [
        case.name
        for case in config.cases
        if case.kind == "external_pair" and not _uses_paired_ablation_mode(case)
    ]
    checks.append(
        _readiness_record(
            "external_pair_final_paper_ablation_mode",
            not one_sided_external_pair_cases,
            "one-sided external_pair cases are ablation-only: "
            + MODEL_NAME_JOINER.join(one_sided_external_pair_cases)
            if one_sided_external_pair_cases
            else "external_pair final-paper cases use paired ablation_mode",
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
    external_mode_control_errors = [
        f"{case.name}: " + MODEL_NAME_JOINER.join(_external_pair_mode_control_errors(case))
        for case in paired_cases
        if case.kind == "external_pair" and _external_pair_mode_control_errors(case)
    ]
    checks.append(
        _readiness_record(
            "external_pair_mode_controls",
            not external_mode_control_errors,
            "invalid mode controls: " + MODEL_NAME_JOINER.join(external_mode_control_errors)
            if external_mode_control_errors
            else "external pairs record disabled cache-off and enabled cache-on controls",
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
            "runtime_overrides": dict(config.runtime_overrides),
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
    *,
    snapshot_name: str = ENVIRONMENT_SNAPSHOT_NAME,
) -> Path:
    """Write the environment snapshot next to tables and figures."""
    snapshot_path = config.output_dir / snapshot_name
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


def _download_progress_template() -> dict[str, Any]:
    """Return the audit fields used for every artifact download attempt."""
    return {
        "bytes_total": None,
        "bytes_written": 0,
        "percent": None,
        "complete": False,
    }


def _download_percent(bytes_written: int, bytes_total: int | None) -> float | None:
    """Return a bounded download percent when the total byte count is known."""
    if bytes_total is None or bytes_total <= 0:
        return None
    percent = PERCENT_SCALE * float(bytes_written) / float(bytes_total)
    return round(min(percent, PERCENT_SCALE), DOWNLOAD_PROGRESS_PERCENT_DECIMALS)


def _update_download_progress(
    record: dict[str, Any],
    *,
    bytes_written: int,
    bytes_total: int | None,
    complete: bool,
) -> None:
    """Update the machine-readable download progress nested in one record."""
    progress = _as_json_object(record["download_progress"], "download_progress")
    progress["bytes_total"] = bytes_total
    progress["bytes_written"] = bytes_written
    progress["percent"] = _download_percent(bytes_written, bytes_total)
    progress["complete"] = complete


def _download_progress_message(
    artifact_name: str,
    *,
    bytes_written: int,
    bytes_total: int | None,
) -> str:
    """Return one compact terminal message for a download progress event."""
    percent = _download_percent(bytes_written, bytes_total)
    if percent is None:
        return f"{artifact_name}: {bytes_written} bytes downloaded"
    return f"{artifact_name}: {percent:.1f}% ({bytes_written}/{bytes_total} bytes)"


def _emit_download_progress(
    progress_label: str | None,
    artifact_name: str,
    *,
    bytes_written: int,
    bytes_total: int | None,
) -> None:
    """Print one download progress line when a suite run requested terminal updates."""
    if progress_label is None:
        return
    message = _download_progress_message(
        artifact_name,
        bytes_written=bytes_written,
        bytes_total=bytes_total,
    )
    print(f"[{progress_label}] [download {artifact_name}] {message}", flush=True)


def _response_content_length(response: Any) -> int | None:
    """Return an HTTP Content-Length value when the server provides one."""
    header_value = response.headers.get("Content-Length")
    if header_value is None:
        return None
    text = str(header_value).strip()
    return int(text) if text.isdecimal() else None


def _file_url_path(url: str) -> Path:
    """Resolve a file:// artifact URL to a local path on Unix or Windows."""
    parsed_url = urlparse(url)
    source_path_text = parsed_url.path
    if os.name == "nt":
        if parsed_url.netloc:
            source_path_text = f"//{parsed_url.netloc}{parsed_url.path}"
        elif re.match(r"^/[A-Za-z]:", parsed_url.path):
            source_path_text = parsed_url.path[1:]
    return Path(source_path_text)


def _copy_file_url(
    url: str,
    destination: Path,
    *,
    progress_callback: Callable[[int, int | None, bool], None] | None = None,
) -> tuple[int, int]:
    """Copy a file:// artifact so tests and offline mirrors share one path."""
    source_path = _file_url_path(url)
    bytes_total = source_path.stat().st_size
    bytes_written = 0
    with source_path.open("rb") as source, destination.open("wb") as output:
        while True:
            chunk = source.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            bytes_written += len(chunk)
            if progress_callback is not None:
                progress_callback(bytes_written, bytes_total, False)
    if progress_callback is not None:
        progress_callback(bytes_written, bytes_total, True)
    return bytes_written, bytes_total


def download_artifact(
    artifact: ArtifactConfig,
    dry_run: bool = False,
    *,
    progress_label: str | None = None,
) -> dict[str, Any]:
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
        "download_progress": _download_progress_template(),
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
    last_progress_bytes = 0

    def record_progress(
        bytes_written: int,
        bytes_total: int | None,
        complete: bool,
    ) -> None:
        """Record progress and emit throttled terminal updates for one artifact."""
        nonlocal last_progress_bytes
        final_bytes_total = bytes_total if bytes_total is not None else bytes_written
        _update_download_progress(
            record,
            bytes_written=bytes_written,
            bytes_total=final_bytes_total if complete else bytes_total,
            complete=complete,
        )
        should_emit = complete or (
            bytes_written - last_progress_bytes >= DOWNLOAD_PROGRESS_INTERVAL_BYTES
        )
        if should_emit:
            _emit_download_progress(
                progress_label,
                artifact.name,
                bytes_written=bytes_written,
                bytes_total=final_bytes_total if complete else bytes_total,
            )
            last_progress_bytes = bytes_written

    if parsed_url.scheme == "file":
        _copy_file_url(
            artifact.url,
            temporary_path,
            progress_callback=record_progress,
        )
    else:
        with (
            urlopen(artifact.url, timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS) as response,
            temporary_path.open("wb") as output,
        ):
            bytes_total = _response_content_length(response)
            bytes_written = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                bytes_written += len(chunk)
                record_progress(bytes_written, bytes_total, False)
            record_progress(bytes_written, bytes_total, True)
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
        record = download_artifact(
            artifact,
            dry_run=dry_run,
            progress_label=config.name,
        )
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
            "runtime_overrides": dict(config.runtime_overrides),
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
                record = download_artifact(
                    artifact,
                    dry_run=dry_run,
                    progress_label=config.name,
                )
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
    environment_snapshot_path = write_environment_snapshot(
        config,
        gpu_record,
        snapshot_name=PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME,
    )
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
            "runtime_overrides": dict(config.runtime_overrides),
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


def _write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON report and return its path for stage records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _pipeline_stage_record(
    *,
    name: str,
    status: str,
    report_path: Path | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Return one machine-readable pipeline stage record."""
    return {
        "name": name,
        "status": status,
        "report_path": str(report_path) if report_path is not None else None,
        "detail": detail,
    }


def _pipeline_status(*, dry_run: bool, failed: bool) -> str:
    """Return the final pipeline status from dry-run and failure state."""
    if failed:
        return PIPELINE_STATUS_FAILED
    return PIPELINE_STATUS_PLANNED if dry_run else PIPELINE_STATUS_PASSED


def _pipeline_stage_report_fingerprints(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fingerprint every stage report already written before the pipeline report."""
    records: list[dict[str, Any]] = []
    for stage in stages:
        report_path = stage.get("report_path")
        if report_path is None:
            records.append(
                {
                    "name": stage.get("name"),
                    "report": None,
                }
            )
            continue
        records.append(
            {
                "name": stage.get("name"),
                "report": optional_file_fingerprint(Path(str(report_path))),
            }
        )
    return records


def _write_pipeline_report(
    *,
    config: SuiteConfig,
    report_path: Path,
    dry_run: bool,
    skip_downloads: bool,
    skip_gpu_check: bool,
    allow_gpu_mismatch: bool,
    keep_going: bool,
    reuse_passed: bool,
    stages: list[dict[str, Any]],
    failed: bool,
    output_bundle_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the top-level paper pipeline report."""
    payload = {
        "pipeline_report_schema_version": PIPELINE_REPORT_SCHEMA_VERSION,
        "status": _pipeline_status(dry_run=dry_run, failed=failed),
        "report_path": str(report_path),
        "provenance": collect_run_provenance(),
        "modes": {
            "dry_run": dry_run,
            "skip_downloads": skip_downloads,
            "skip_gpu_check": skip_gpu_check,
            "allow_gpu_mismatch": allow_gpu_mismatch,
            "keep_going": keep_going,
            "reuse_passed": reuse_passed,
        },
        "suite": {
            "name": config.name,
            "manifest_path": str(config.manifest_path),
            "manifest": manifest_record(config),
            "output_dir": str(config.output_dir),
            "expected_gpus": config.expected_gpus,
            "required_models": list(config.required_models),
            "runtime_overrides": dict(config.runtime_overrides),
        },
        "stages": stages,
        STAGE_REPORT_FINGERPRINTS_KEY: _pipeline_stage_report_fingerprints(stages),
        OUTPUT_BUNDLE_VERIFICATION_KEY: output_bundle_verification,
    }
    _write_json_report(report_path, payload)
    return payload


def run_pipeline(
    config: SuiteConfig,
    *,
    dry_run: bool = False,
    skip_downloads: bool = False,
    skip_gpu_check: bool = False,
    allow_gpu_mismatch: bool = False,
    keep_going: bool = False,
    reuse_passed: bool = False,
    verify_output: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full paper pipeline from readiness checks through bundle verify."""
    validate_suite_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = report_path or config.output_dir / PIPELINE_REPORT_NAME
    stages: list[dict[str, Any]] = []

    readiness_path = config.output_dir / READINESS_REPORT_NAME
    readiness_report = build_readiness_report(config)
    _write_json_report(readiness_path, readiness_report)
    stages.append(
        _pipeline_stage_record(
            name="readiness",
            status=readiness_report["status"],
            report_path=readiness_path,
        )
    )
    if readiness_report["status"] != "ready":
        return _write_pipeline_report(
            config=config,
            report_path=final_report_path,
            dry_run=dry_run,
            skip_downloads=skip_downloads,
            skip_gpu_check=skip_gpu_check,
            allow_gpu_mismatch=allow_gpu_mismatch,
            keep_going=keep_going,
            reuse_passed=reuse_passed,
            stages=stages,
            failed=True,
        )

    if skip_downloads:
        stages.append(
            _pipeline_stage_record(
                name="prepare_artifacts",
                status=PREFLIGHT_STATUS_SKIPPED,
                detail=PREFLIGHT_SKIP_DOWNLOADS_REASON,
            )
        )
    else:
        try:
            artifact_report = prepare_artifacts(config, dry_run=dry_run)
            stages.append(
                _pipeline_stage_record(
                    name="prepare_artifacts",
                    status=str(artifact_report["status"]),
                    report_path=Path(str(artifact_report["report_path"])),
                )
            )
        except (ClusterSuiteError, OSError) as exc:
            stages.append(
                _pipeline_stage_record(
                    name="prepare_artifacts",
                    status=PIPELINE_STATUS_FAILED,
                    report_path=config.output_dir / ARTIFACT_PREPARATION_REPORT_NAME,
                    detail=str(exc),
                )
            )
            return _write_pipeline_report(
                config=config,
                report_path=final_report_path,
                dry_run=dry_run,
                skip_downloads=skip_downloads,
                skip_gpu_check=skip_gpu_check,
                allow_gpu_mismatch=allow_gpu_mismatch,
                keep_going=keep_going,
                reuse_passed=reuse_passed,
                stages=stages,
                failed=True,
            )

    preflight_report = run_preflight_only(
        config,
        dry_run=dry_run,
        skip_downloads=skip_downloads,
        skip_gpu_check=skip_gpu_check,
        allow_gpu_mismatch=allow_gpu_mismatch,
        report_path=config.output_dir / PREFLIGHT_REPORT_NAME,
    )
    stages.append(
        _pipeline_stage_record(
            name="preflight",
            status=str(preflight_report["status"]),
            report_path=Path(str(preflight_report["report_path"])),
        )
    )
    if preflight_report["status"] == PREFLIGHT_STATUS_FAILED:
        return _write_pipeline_report(
            config=config,
            report_path=final_report_path,
            dry_run=dry_run,
            skip_downloads=skip_downloads,
            skip_gpu_check=skip_gpu_check,
            allow_gpu_mismatch=allow_gpu_mismatch,
            keep_going=keep_going,
            reuse_passed=reuse_passed,
            stages=stages,
            failed=True,
        )

    plan_path = write_run_plan(
        config,
        config.output_dir / PLAN_REPORT_NAME,
        collect_only=False,
        skip_downloads=skip_downloads,
        skip_gpu_check=skip_gpu_check,
        reuse_passed=reuse_passed,
    )
    stages.append(
        _pipeline_stage_record(
            name="plan",
            status=PIPELINE_STATUS_PASSED,
            report_path=plan_path,
        )
    )

    run_returncode = run_suite(
        config,
        dry_run=dry_run,
        collect_only=False,
        skip_downloads=skip_downloads,
        skip_gpu_check=skip_gpu_check,
        allow_gpu_mismatch=allow_gpu_mismatch,
        keep_going=keep_going,
        reuse_passed=reuse_passed,
        verify_output=verify_output,
    )
    run_failed = run_returncode != SUCCESS_RETURN_CODE
    stages.append(
        _pipeline_stage_record(
            name="run_suite",
            status=PIPELINE_STATUS_FAILED if run_failed else PIPELINE_STATUS_PASSED,
            report_path=config.output_dir / SUMMARY_REPORT_NAME,
            detail=f"returncode={run_returncode}",
        )
    )
    if run_failed:
        return _write_pipeline_report(
            config=config,
            report_path=final_report_path,
            dry_run=dry_run,
            skip_downloads=skip_downloads,
            skip_gpu_check=skip_gpu_check,
            allow_gpu_mismatch=allow_gpu_mismatch,
            keep_going=keep_going,
            reuse_passed=reuse_passed,
            stages=stages,
            failed=True,
        )

    try:
        verification = verify_output_bundle(config.output_dir)
        stages.append(
            _pipeline_stage_record(
                name="verify_output_bundle",
                status=str(verification["status"]),
                report_path=Path(str(verification["summary_json"])),
            )
        )
    except ClusterSuiteError as exc:
        verification = {
            "status": PIPELINE_STATUS_FAILED,
            "summary_json": str(config.output_dir / SUMMARY_REPORT_NAME),
            "detail": str(exc),
        }
        stages.append(
            _pipeline_stage_record(
                name="verify_output_bundle",
                status=PIPELINE_STATUS_FAILED,
                report_path=config.output_dir / SUMMARY_REPORT_NAME,
                detail=str(exc),
            )
        )
        return _write_pipeline_report(
            config=config,
            report_path=final_report_path,
            dry_run=dry_run,
            skip_downloads=skip_downloads,
            skip_gpu_check=skip_gpu_check,
            allow_gpu_mismatch=allow_gpu_mismatch,
            keep_going=keep_going,
            reuse_passed=reuse_passed,
            stages=stages,
            failed=True,
            output_bundle_verification=verification,
        )

    return _write_pipeline_report(
        config=config,
        report_path=final_report_path,
        dry_run=dry_run,
        skip_downloads=skip_downloads,
        skip_gpu_check=skip_gpu_check,
        allow_gpu_mismatch=allow_gpu_mismatch,
        keep_going=keep_going,
        reuse_passed=reuse_passed,
        stages=stages,
        failed=False,
        output_bundle_verification=verification,
    )


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


def _command_record(
    *,
    name: str,
    command: str | list[str],
    returncode: int,
    elapsed_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
    env: dict[str, str] | None,
) -> CommandRecord:
    """Build the normalized command provenance record used by all runners."""
    return CommandRecord(
        name=name,
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        cwd=str(cwd),
        tracked_env=command_environment_snapshot(env),
    )


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
        return _command_record(
            name=name,
            command=argv,
            returncode=SUCCESS_RETURN_CODE,
            elapsed_seconds=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
        )
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
        return _command_record(
            name=name,
            command=argv,
            returncode=int(completed.returncode),
            elapsed_seconds=elapsed_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            exc.stdout or "",
            f"Command timed out after {timeout_seconds:g} seconds\n{exc.stderr or ''}",
        )
        return _command_record(
            name=name,
            command=argv,
            returncode=COMMAND_TIMEOUT_RETURN_CODE,
            elapsed_seconds=elapsed_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
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
        return _command_record(
            name=name,
            command=command,
            returncode=SUCCESS_RETURN_CODE,
            elapsed_seconds=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
        )
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
        return _command_record(
            name=name,
            command=command,
            returncode=int(completed.returncode),
            elapsed_seconds=elapsed_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_seconds = time.perf_counter() - start_time
        _write_command_streams(
            stdout_path,
            stderr_path,
            exc.stdout or "",
            f"Command timed out after {timeout_seconds:g} seconds\n{exc.stderr or ''}",
        )
        return _command_record(
            name=name,
            command=command,
            returncode=COMMAND_TIMEOUT_RETURN_CODE,
            elapsed_seconds=elapsed_seconds,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            env=env,
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
    if (
        collect_only
        or case.kind != "sevennet_lammps"
        or not trace_paths
        or not _uses_paired_ablation_mode(case)
    ):
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


def _optional_path_fingerprint(path_text: str | None) -> dict[str, Any] | None:
    """Return a fingerprint for an optional evidence path."""
    return None if path_text is None else generated_artifact_record(Path(path_text))


def evidence_fingerprints(case_summaries: list[CaseSummary]) -> dict[str, Any]:
    """Fingerprint every source evidence file behind the generated paper tables."""
    return {
        summary.case_name: {
            "benchmark_report": _optional_path_fingerprint(summary.benchmark_report),
            "bundle_evidence": _optional_path_fingerprint(summary.bundle_evidence),
            "external_timing_report": _optional_path_fingerprint(
                summary.external_timing_report
            ),
            TRACE_EVIDENCE_KEY: [
                generated_artifact_record(Path(path_text))
                for path_text in summary.trace_evidence
            ],
        }
        for summary in case_summaries
    }


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


def _resolve_present_fingerprint_path(
    record: dict[str, Any],
    label: str,
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> Path:
    """Validate one present fingerprint record and return its local path."""
    _require(
        record.get("exists", True) is not False,
        f"{label}: expected a present fingerprint record",
    )
    _require_fingerprint_match(
        record,
        label,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )
    recorded_path = Path(_as_json_string(record.get("path"), f"{label}.path"))
    for candidate in _candidate_fingerprint_paths(
        recorded_path,
        bundle_root,
        original_output_dir,
    ):
        if candidate.exists():
            return candidate
    _require(False, f"{label}: validated fingerprint path disappeared")
    raise AssertionError("unreachable after _require failure")


def _require_command_record_alignment(
    summary_payload: dict[str, Any],
    command_fingerprints: list[Any],
) -> int:
    """Verify command provenance records match their log fingerprints."""
    command_records = summary_payload.get(COMMANDS_KEY)
    _require(isinstance(command_records, list), f"{COMMANDS_KEY} must be a JSON array")
    _require(
        len(command_records) == len(command_fingerprints),
        f"{COMMANDS_KEY} and command_log_fingerprints must have the same length",
    )
    required_env_keys = tuple(dict.fromkeys(COMMAND_ENV_SNAPSHOT_KEYS))
    for index, raw_command_record in enumerate(command_records):
        command_record = _as_json_object(
            raw_command_record,
            f"{COMMANDS_KEY}[{index}]",
        )
        fingerprint_record = _as_json_object(
            command_fingerprints[index],
            f"command_log_fingerprints[{index}]",
        )
        command_name = _as_json_string(
            command_record.get("name"),
            f"{COMMANDS_KEY}[{index}].name",
        )
        fingerprint_name = _as_json_string(
            fingerprint_record.get("name"),
            f"command_log_fingerprints[{index}].name",
        )
        _require(
            command_name == fingerprint_name,
            f"{COMMANDS_KEY}[{index}] and command_log_fingerprints[{index}] must align by name",
        )
        command_returncode = _as_json_nonnegative_int(
            command_record.get("returncode"),
            f"{COMMANDS_KEY}[{index}].returncode",
        )
        fingerprint_returncode = _as_json_nonnegative_int(
            fingerprint_record.get("returncode"),
            f"command_log_fingerprints[{index}].returncode",
        )
        _require(
            command_returncode == fingerprint_returncode,
            f"{COMMANDS_KEY}[{index}] and command_log_fingerprints[{index}] must align by returncode",
        )
        _as_json_string(command_record.get("cwd"), f"{COMMANDS_KEY}[{index}].cwd")
        tracked_env = _as_json_object(
            command_record.get("tracked_env"),
            f"{COMMANDS_KEY}[{index}].tracked_env",
        )
        for env_key in required_env_keys:
            _require(
                env_key in tracked_env,
                f"{COMMANDS_KEY}[{index}].tracked_env missing {env_key}",
            )
            env_value = tracked_env[env_key]
            _require(
                env_value is None or isinstance(env_value, str),
                f"{COMMANDS_KEY}[{index}].tracked_env.{env_key} must be a string or null",
            )
        for stream_name, command_path_key in (
            ("stdout", "stdout_path"),
            ("stderr", "stderr_path"),
        ):
            command_path = _as_json_string(
                command_record.get(command_path_key),
                f"{COMMANDS_KEY}[{index}].{command_path_key}",
            )
            stream_record = _as_json_object(
                fingerprint_record.get(stream_name),
                f"command_log_fingerprints[{index}].{stream_name}",
            )
            fingerprint_path = _as_json_string(
                stream_record.get("path"),
                f"command_log_fingerprints[{index}].{stream_name}.path",
            )
            _require(
                command_path == fingerprint_path,
                (
                    f"{COMMANDS_KEY}[{index}].{command_path_key} must match "
                    f"command_log_fingerprints[{index}].{stream_name}.path"
                ),
            )
    return len(command_records)


def _require_present_fingerprint_match(
    record: dict[str, Any],
    label: str,
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> None:
    """Validate a fingerprint for a source evidence file that must exist."""
    _require(record.get("exists", True) is not False, f"{label}: source evidence must exist")
    _require_fingerprint_match(
        record,
        label,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )


def _summary_case_names(summary_payload: dict[str, Any]) -> list[str]:
    """Return case names recorded in the summary payload."""
    raw_cases = summary_payload.get("cases")
    _require(isinstance(raw_cases, list), "cases must be a JSON array")
    case_names: list[str] = []
    for index, raw_case in enumerate(raw_cases):
        case_record = _as_json_object(raw_case, f"cases[{index}]")
        case_names.append(_as_json_string(case_record.get("case_name"), f"cases[{index}].case_name"))
    return case_names


def _require_evidence_fingerprint_matches(
    summary_payload: dict[str, Any],
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> int:
    """Verify source evidence fingerprints that feed the paper tables."""
    case_names = _summary_case_names(summary_payload)
    raw_evidence_records = summary_payload.get(EVIDENCE_FINGERPRINTS_KEY)
    _require(
        isinstance(raw_evidence_records, dict),
        f"{EVIDENCE_FINGERPRINTS_KEY} must be a JSON object",
    )
    recorded_case_names = set(raw_evidence_records)
    expected_case_names = set(case_names)
    _require(
        recorded_case_names == expected_case_names,
        f"{EVIDENCE_FINGERPRINTS_KEY} case names must match summary cases",
    )
    verified_count = 0
    for case_name in case_names:
        case_record = _as_json_object(
            raw_evidence_records.get(case_name),
            f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}",
        )
        for field_name in CASE_EVIDENCE_FINGERPRINT_FIELDS:
            raw_record = case_record.get(field_name)
            if raw_record is None:
                continue
            record = _as_json_object(
                raw_record,
                f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}.{field_name}",
            )
            _require_present_fingerprint_match(
                record,
                f"{case_name}.{field_name}",
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
            verified_count += 1
        trace_records = case_record.get(TRACE_EVIDENCE_KEY)
        _require(
            isinstance(trace_records, list),
            f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}.{TRACE_EVIDENCE_KEY} must be a JSON array",
        )
        for trace_index, raw_record in enumerate(trace_records):
            record = _as_json_object(
                raw_record,
                f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}.{TRACE_EVIDENCE_KEY}[{trace_index}]",
            )
            _require_present_fingerprint_match(
                record,
                f"{case_name}.{TRACE_EVIDENCE_KEY}[{trace_index}]",
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
            verified_count += 1
    return verified_count


def _summary_cases_by_name(summary_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return summary case records keyed by case name."""
    raw_cases = summary_payload.get("cases")
    _require(isinstance(raw_cases, list), "cases must be a JSON array")
    cases_by_name: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        case_record = _as_json_object(raw_case, f"cases[{index}]")
        case_name = _as_json_string(case_record.get("case_name"), f"cases[{index}].case_name")
        _require(case_name not in cases_by_name, f"duplicate summary case {case_name}")
        cases_by_name[case_name] = case_record
    return cases_by_name


def _case_config_from_external_summary(
    *,
    case_name: str,
    case_record: dict[str, Any],
    mode_controls: dict[str, Any],
    timing_payload: dict[str, Any],
) -> CaseConfig:
    """Reconstruct the manifest fields needed to audit external timing evidence."""
    kind = _as_json_string(case_record.get("kind"), f"cases.{case_name}.kind")
    _require(kind == "external_pair", f"{case_name}: kind must be external_pair")
    disabled_env = _as_json_object(
        mode_controls.get("disabled_env"),
        f"case_mode_controls.{case_name}.disabled_env",
    )
    enabled_env = _as_json_object(
        mode_controls.get("enabled_env"),
        f"case_mode_controls.{case_name}.enabled_env",
    )
    return CaseConfig(
        name=case_name,
        model=_as_json_string(case_record.get("model"), f"cases.{case_name}.model"),
        kind=kind,
        ablation_mode=_as_json_string(
            mode_controls.get("ablation_mode"),
            f"case_mode_controls.{case_name}.ablation_mode",
        ),
        disabled_command=_as_json_string(
            mode_controls.get("disabled_command"),
            f"case_mode_controls.{case_name}.disabled_command",
        ),
        enabled_command=_as_json_string(
            mode_controls.get("enabled_command"),
            f"case_mode_controls.{case_name}.enabled_command",
        ),
        disabled_env=disabled_env,
        enabled_env=enabled_env,
        repeat_count=_as_json_nonnegative_int(
            timing_payload.get(REPEAT_COUNT_KEY),
            f"{case_name}.{REPEAT_COUNT_KEY}",
        ),
    )


def _require_external_timing_reports_from_summary(
    summary_payload: dict[str, Any],
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> int:
    """Verify nested external timing reports and their command-log fingerprints."""
    cases_by_name = _summary_cases_by_name(summary_payload)
    raw_mode_controls = _as_json_object(
        summary_payload.get("case_mode_controls", {}),
        "case_mode_controls",
    )
    raw_evidence_records = _as_json_object(
        summary_payload.get(EVIDENCE_FINGERPRINTS_KEY),
        EVIDENCE_FINGERPRINTS_KEY,
    )
    verified_log_count = 0
    for case_name, case_record in cases_by_name.items():
        if case_record.get("kind") != "external_pair":
            continue
        case_evidence = _as_json_object(
            raw_evidence_records.get(case_name),
            f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}",
        )
        raw_timing_record = case_evidence.get("external_timing_report")
        if raw_timing_record is None:
            continue
        timing_record = _as_json_object(
            raw_timing_record,
            f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}.external_timing_report",
        )
        timing_report_path = _resolve_present_fingerprint_path(
            timing_record,
            f"{case_name}.external_timing_report",
            bundle_root=bundle_root,
            original_output_dir=original_output_dir,
        )
        timing_payload = _as_json_object(
            json.loads(timing_report_path.read_text(encoding="utf-8")),
            f"{case_name}.external_timing_report",
        )
        mode_controls = _as_json_object(
            raw_mode_controls.get(case_name),
            f"case_mode_controls.{case_name}",
        )
        case_config = _case_config_from_external_summary(
            case_name=case_name,
            case_record=case_record,
            mode_controls=mode_controls,
            timing_payload=timing_payload,
        )
        verification = validate_external_timing_report(
            timing_payload,
            case_config,
            report_path=timing_report_path,
            bundle_root=bundle_root,
            original_output_dir=original_output_dir,
        )
        verified_log_count += _as_json_nonnegative_int(
            verification.get("verified_command_log_count"),
            f"{case_name}.verified_command_log_count",
        )
    return verified_log_count


def _read_csv_rows(path: Path, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read a generated CSV artifact with schema-oriented validation errors."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        _require(fieldnames, f"{label}: missing CSV header")
        rows = list(reader)
    for index, row in enumerate(rows):
        _require(None not in row, f"{label}: row {index} has more cells than headers")
    return fieldnames, rows


def _require_columns(
    fieldnames: tuple[str, ...],
    required_columns: tuple[str, ...],
    label: str,
) -> None:
    """Require generated table columns that downstream paper scripts depend on."""
    missing_columns = [column for column in required_columns if column not in fieldnames]
    _require(not missing_columns, f"{label}: missing columns {missing_columns}")


def _as_csv_nonnegative_int(value: str | None, field_name: str) -> int:
    """Return a whole nonnegative count from a generated CSV cell."""
    _require(value is not None, f"{field_name} must be present")
    text = value.strip()
    _require(text.isdecimal(), f"{field_name} must be a nonnegative integer")
    return int(text)


def _format_csv_value(value: Any) -> str:
    """Format a summary JSON value the way csv.DictWriter writes table cells."""
    return "" if value is None else str(value)


def _case_summary_field(column_name: str) -> str:
    """Map a paper table column to the matching summary JSON case field."""
    return PAPER_CASE_SUMMARY_FIELD_MAP.get(column_name, column_name)


def _require_case_summary_cell_values(
    *,
    column_names: tuple[str, ...],
    row_values_by_case: dict[str, dict[str, str]],
    cases_by_name: dict[str, dict[str, Any]],
    label: str,
    formatter: Any,
) -> None:
    """Verify generated case-summary table cells against summary JSON cases."""
    for case_name, row_values in row_values_by_case.items():
        case_record = cases_by_name[case_name]
        for column_name in column_names:
            summary_field = _case_summary_field(column_name)
            _require(
                summary_field in case_record,
                f"{label}.{case_name}: summary field {summary_field} is missing",
            )
            expected_value = formatter(case_record.get(summary_field))
            actual_value = row_values.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"{label}.{case_name}.{column_name} must match summary cases",
            )


def _summary_correlations_by_metric_pair(
    summary_payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return summary correlation records keyed by their metric pair."""
    raw_correlations = summary_payload.get("correlations")
    _require(isinstance(raw_correlations, list), "correlations must be a JSON array")
    correlations_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_record in enumerate(raw_correlations):
        record = _as_json_object(raw_record, f"correlations[{index}]")
        metric_pair = (
            _as_json_string(record.get("x_metric"), f"correlations[{index}].x_metric"),
            _as_json_string(record.get("y_metric"), f"correlations[{index}].y_metric"),
        )
        _require(metric_pair not in correlations_by_pair, f"duplicate correlation row {metric_pair}")
        correlations_by_pair[metric_pair] = record
    return correlations_by_pair


def _summary_numeric_value(case_record: dict[str, Any], field_name: str) -> float | None:
    """Return a finite numeric case field when a paper figure can plot it."""
    value = case_record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def _svg_local_name(element: Any) -> str:
    """Return an SVG element tag without an XML namespace prefix."""
    return str(element.tag).rsplit("}", maxsplit=1)[-1]


def _svg_text_content(root: Any) -> str:
    """Collect SVG text node content for figure-label semantic checks."""
    return "\n".join(
        "".join(element.itertext()).strip()
        for element in root.iter()
        if _svg_local_name(element) == "text"
    )


def _svg_element_count(root: Any, element_name: str) -> int:
    """Count SVG elements by local tag name so namespaces do not matter."""
    return sum(1 for element in root.iter() if _svg_local_name(element) == element_name)


def _markdown_cells(line: str) -> list[str]:
    """Split one GitHub-flavored markdown table row into trimmed cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _require_case_summary_csv(path: Path, cases_by_name: dict[str, dict[str, Any]]) -> None:
    """Verify that the main CSV table has one row per summary case."""
    fieldnames, rows = _read_csv_rows(path, "case_summary.csv")
    _require_columns(fieldnames, PAPER_CASE_SUMMARY_COLUMNS, "case_summary.csv")
    expected_case_names = set(cases_by_name)
    row_case_names = {
        _as_json_string(row.get("case"), f"case_summary.csv[{index}].case")
        for index, row in enumerate(rows)
    }
    _require(
        len(rows) == len(expected_case_names),
        "case_summary.csv: row count must match summary cases",
    )
    _require(
        row_case_names == expected_case_names,
        "case_summary.csv: case names must match summary cases",
    )
    _require_case_summary_cell_values(
        column_names=fieldnames,
        row_values_by_case={
            _as_json_string(row.get("case"), f"case_summary.csv[{index}].case"): row
            for index, row in enumerate(rows)
        },
        cases_by_name=cases_by_name,
        label="case_summary.csv",
        formatter=_format_csv_value,
    )


def _require_case_summary_markdown(
    path: Path,
    cases_by_name: dict[str, dict[str, Any]],
) -> None:
    """Verify that the markdown table is readable and aligned with summary cases."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_case_names = set(cases_by_name)
    _require(
        len(lines) == len(expected_case_names) + 2,
        "case_summary.md: row count must match summary cases plus header",
    )
    header = _markdown_cells(lines[0])
    separator = _markdown_cells(lines[1])
    _require_columns(tuple(header), PAPER_CASE_SUMMARY_COLUMNS, "case_summary.md")
    _require(
        len(separator) == len(header),
        "case_summary.md: separator width must match header",
    )
    _require(
        all(cell == "---" for cell in separator),
        "case_summary.md: separator row must contain markdown column markers",
    )
    case_index = header.index("case")
    row_case_names: set[str] = set()
    for index, line in enumerate(lines[2:]):
        cells = _markdown_cells(line)
        _require(
            len(cells) == len(header),
            f"case_summary.md[{index}]: row width must match header",
        )
        row_case_names.add(
            _as_json_string(cells[case_index], f"case_summary.md[{index}].case")
        )
    _require(
        row_case_names == expected_case_names,
        "case_summary.md: case names must match summary cases",
    )
    _require_case_summary_cell_values(
        column_names=tuple(header),
        row_values_by_case={
            _as_json_string(
                _markdown_cells(line)[case_index],
                f"case_summary.md[{index}].case",
            ): dict(zip(header, _markdown_cells(line), strict=True))
            for index, line in enumerate(lines[2:])
        },
        cases_by_name=cases_by_name,
        label="case_summary.md",
        formatter=_format_table_value,
    )


def _require_correlation_csv(path: Path, summary_payload: dict[str, Any]) -> None:
    """Verify that the correlation table matches summary JSON correlation rows."""
    fieldnames, rows = _read_csv_rows(path, "correlation.csv")
    _require_columns(fieldnames, PAPER_CORRELATION_COLUMNS, "correlation.csv")
    summary_correlations = _summary_correlations_by_metric_pair(summary_payload)
    _require(
        len(rows) == len(summary_correlations),
        "correlation.csv: row count must match summary correlations",
    )
    expected_pairs = set(CORRELATION_METRIC_PAIRS)
    observed_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        metric_pair = (
            _as_json_string(row.get("x_metric"), f"correlation.csv[{index}].x_metric"),
            _as_json_string(row.get("y_metric"), f"correlation.csv[{index}].y_metric"),
        )
        observed_pairs.add(metric_pair)
        _as_csv_nonnegative_int(row.get("n"), f"correlation.csv[{index}].n")
        summary_row = _as_json_object(
            summary_correlations.get(metric_pair),
            f"correlations.{metric_pair}",
        )
        for column_name in fieldnames:
            expected_value = _format_csv_value(summary_row.get(column_name))
            actual_value = row.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"correlation.csv.{metric_pair}.{column_name} must match summary correlations",
            )
    _require(
        observed_pairs == expected_pairs,
        "correlation.csv: metric pairs must match configured paper correlations",
    )


def _summary_command_rows(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return summary command records as table rows with the paper command schema."""
    raw_commands = summary_payload.get(COMMANDS_KEY)
    _require(isinstance(raw_commands, list), f"{COMMANDS_KEY} must be a JSON array")
    rows: list[dict[str, Any]] = []
    for index, raw_command in enumerate(raw_commands):
        command_record = _as_json_object(raw_command, f"{COMMANDS_KEY}[{index}]")
        rows.append(
            {
                "name": command_record.get("name"),
                "returncode": command_record.get("returncode"),
                "elapsed_seconds": command_record.get("elapsed_seconds"),
                "stdout_path": command_record.get("stdout_path"),
                "stderr_path": command_record.get("stderr_path"),
                "cwd": command_record.get("cwd"),
            }
        )
    return rows


def _benchmark_repeat_timing_rows(
    case_name: str,
    model: str,
    kind: str,
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return raw per-repeat rows from a SevenNet paired benchmark report."""
    if report is None:
        return []
    raw_results = report.get("results")
    _require(
        isinstance(raw_results, list),
        f"{case_name}.benchmark_report.results must be a JSON array",
    )
    rows: list[dict[str, Any]] = []
    for index, raw_result in enumerate(raw_results):
        result = _as_json_object(
            raw_result,
            f"{case_name}.benchmark_report.results[{index}]",
        )
        mode = _as_json_string(
            result.get("case"),
            f"{case_name}.benchmark_report.results[{index}].case",
        )
        if mode not in (BASELINE_CASE_NAME, ISODELTA_CASE_NAME):
            continue
        rows.append(
            {
                "case": case_name,
                "model": model,
                "kind": kind,
                "source": BENCHMARK_TIMING_SOURCE,
                "mode": mode,
                "repeat_index": _as_json_nonnegative_int(
                    result.get("repeat_index"),
                    f"{case_name}.benchmark_report.results[{index}].repeat_index",
                ),
                "elapsed_seconds": _as_json_optional_nonnegative_number(
                    result.get("loop_time_seconds"),
                    f"{case_name}.benchmark_report.results[{index}].loop_time_seconds",
                ),
                "returncode": _as_json_nonnegative_int(
                    result.get("returncode"),
                    f"{case_name}.benchmark_report.results[{index}].returncode",
                ),
            }
        )
    return rows


def _external_repeat_timing_rows(
    case_name: str,
    model: str,
    kind: str,
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return raw disabled/enabled timing samples from an external timing report."""
    if report is None:
        return []
    observed_case_name = _as_json_string(
        report.get(CASE_NAME_KEY),
        f"{case_name}.external_timing_report.{CASE_NAME_KEY}",
    )
    observed_model = _as_json_string(
        report.get(MODEL_KEY),
        f"{case_name}.external_timing_report.{MODEL_KEY}",
    )
    _require(
        observed_case_name == case_name,
        f"{case_name}.external_timing_report.{CASE_NAME_KEY} must match summary case",
    )
    _require(
        observed_model == model,
        f"{case_name}.external_timing_report.{MODEL_KEY} must match summary model",
    )
    rows: list[dict[str, Any]] = []
    for mode, timing_key in (
        (EXTERNAL_DISABLED_COMMAND_LABEL, BASELINE_TIMES_SECONDS_KEY),
        (EXTERNAL_ENABLED_COMMAND_LABEL, ENABLED_TIMES_SECONDS_KEY),
    ):
        timing_values = _as_json_positive_number_list(
            report.get(timing_key),
            f"{case_name}.external_timing_report.{timing_key}",
        )
        for repeat_index, elapsed_seconds in enumerate(timing_values):
            rows.append(
                {
                    "case": case_name,
                    "model": model,
                    "kind": kind,
                    "source": EXTERNAL_TIMING_SOURCE,
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "elapsed_seconds": elapsed_seconds,
                    "returncode": SUCCESS_RETURN_CODE,
                }
            )
    return rows


def _repeat_timing_rows_from_sources(
    case_name: str,
    model: str,
    kind: str,
    *,
    benchmark_payload: dict[str, Any] | None,
    external_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return paper-table timing rows from the source evidence for one case."""
    return (
        _benchmark_repeat_timing_rows(case_name, model, kind, benchmark_payload)
        + _external_repeat_timing_rows(case_name, model, kind, external_payload)
    )


def _load_fingerprinted_json_payload(
    case_evidence: dict[str, Any],
    case_name: str,
    field_name: str,
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> dict[str, Any] | None:
    """Load one source evidence JSON object through its verified fingerprint record."""
    raw_record = case_evidence.get(field_name)
    if raw_record is None:
        return None
    record = _as_json_object(
        raw_record,
        f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}.{field_name}",
    )
    evidence_path = _resolve_present_fingerprint_path(
        record,
        f"{case_name}.{field_name}",
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )
    return _as_json_object(
        json.loads(evidence_path.read_text(encoding="utf-8")),
        f"{case_name}.{field_name}",
    )


def _repeat_timing_rows_from_summary(
    summary_payload: dict[str, Any],
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> list[dict[str, Any]]:
    """Recompute repeat timing rows from summary-protected source evidence."""
    cases_by_name = _summary_cases_by_name(summary_payload)
    raw_evidence_records = _as_json_object(
        summary_payload.get(EVIDENCE_FINGERPRINTS_KEY),
        EVIDENCE_FINGERPRINTS_KEY,
    )
    rows: list[dict[str, Any]] = []
    for case_name, case_record in cases_by_name.items():
        case_evidence = _as_json_object(
            raw_evidence_records.get(case_name),
            f"{EVIDENCE_FINGERPRINTS_KEY}.{case_name}",
        )
        model = _as_json_string(case_record.get("model"), f"cases.{case_name}.model")
        kind = _as_json_string(case_record.get("kind"), f"cases.{case_name}.kind")
        benchmark_payload = _load_fingerprinted_json_payload(
            case_evidence,
            case_name,
            "benchmark_report",
            bundle_root=bundle_root,
            original_output_dir=original_output_dir,
        )
        external_payload = _load_fingerprinted_json_payload(
            case_evidence,
            case_name,
            "external_timing_report",
            bundle_root=bundle_root,
            original_output_dir=original_output_dir,
        )
        rows.extend(
            _repeat_timing_rows_from_sources(
                case_name,
                model,
                kind,
                benchmark_payload=benchmark_payload,
                external_payload=external_payload,
            )
        )
    return rows


def _require_repeat_timing_csv(
    path: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    """Verify that repeat timing CSV rows match source timing evidence."""
    fieldnames, rows = _read_csv_rows(path, "repeat_timing.csv")
    _require_columns(fieldnames, PAPER_REPEAT_TIMING_COLUMNS, "repeat_timing.csv")
    _require(
        len(rows) == len(expected_rows),
        "repeat_timing.csv: row count must match source timing evidence",
    )
    for index, expected_row in enumerate(expected_rows):
        csv_row = rows[index]
        for column_name in PAPER_REPEAT_TIMING_COLUMNS:
            expected_value = _format_csv_value(expected_row.get(column_name))
            actual_value = csv_row.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"repeat_timing.csv[{index}].{column_name} must match source timing evidence",
            )


def _require_repeat_timing_markdown(
    path: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    """Verify that repeat timing Markdown rows match source timing evidence."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(
        len(lines) == len(expected_rows) + 2,
        "repeat_timing.md: row count must match source timing evidence plus header",
    )
    header = _markdown_cells(lines[0])
    separator = _markdown_cells(lines[1])
    _require_columns(tuple(header), PAPER_REPEAT_TIMING_COLUMNS, "repeat_timing.md")
    _require(
        len(separator) == len(header),
        "repeat_timing.md: separator width must match header",
    )
    _require(
        all(cell == "---" for cell in separator),
        "repeat_timing.md: separator row must contain markdown column markers",
    )
    for index, expected_row in enumerate(expected_rows):
        cells = _markdown_cells(lines[index + 2])
        _require(
            len(cells) == len(header),
            f"repeat_timing.md[{index}]: row width must match header",
        )
        row = dict(zip(header, cells, strict=True))
        for column_name in PAPER_REPEAT_TIMING_COLUMNS:
            expected_value = _format_table_value(expected_row.get(column_name))
            actual_value = row.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"repeat_timing.md[{index}].{column_name} must match source timing evidence",
            )


def _require_command_timing_csv(
    path: Path,
    summary_payload: dict[str, Any],
) -> None:
    """Verify that the command timing CSV matches summary command records."""
    fieldnames, rows = _read_csv_rows(path, "command_timing.csv")
    _require_columns(fieldnames, PAPER_COMMAND_TIMING_COLUMNS, "command_timing.csv")
    summary_rows = _summary_command_rows(summary_payload)
    _require(
        len(rows) == len(summary_rows),
        "command_timing.csv: row count must match summary commands",
    )
    for index, summary_row in enumerate(summary_rows):
        csv_row = rows[index]
        for column_name in PAPER_COMMAND_TIMING_COLUMNS:
            expected_value = _format_csv_value(summary_row.get(column_name))
            actual_value = csv_row.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"command_timing.csv[{index}].{column_name} must match summary commands",
            )


def _require_command_timing_markdown(
    path: Path,
    summary_payload: dict[str, Any],
) -> None:
    """Verify that the command timing Markdown table matches summary commands."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_rows = _summary_command_rows(summary_payload)
    _require(
        len(lines) == len(summary_rows) + 2,
        "command_timing.md: row count must match summary commands plus header",
    )
    header = _markdown_cells(lines[0])
    separator = _markdown_cells(lines[1])
    _require_columns(tuple(header), PAPER_COMMAND_TIMING_COLUMNS, "command_timing.md")
    _require(
        len(separator) == len(header),
        "command_timing.md: separator width must match header",
    )
    _require(
        all(cell == "---" for cell in separator),
        "command_timing.md: separator row must contain markdown column markers",
    )
    for index, summary_row in enumerate(summary_rows):
        cells = _markdown_cells(lines[index + 2])
        _require(
            len(cells) == len(header),
            f"command_timing.md[{index}]: row width must match header",
        )
        row = dict(zip(header, cells, strict=True))
        for column_name in PAPER_COMMAND_TIMING_COLUMNS:
            expected_value = _format_table_value(summary_row.get(column_name))
            actual_value = row.get(column_name, "")
            _require(
                actual_value == expected_value,
                f"command_timing.md[{index}].{column_name} must match summary commands",
            )


def _require_svg_document(path: Path, label: str) -> Any:
    """Verify that a generated figure is parseable SVG with stable dimensions."""
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ClusterSuiteError(f"{label}: invalid SVG XML: {exc}") from exc
    root_name = _svg_local_name(root)
    _require(root_name == "svg", f"{label}: root element must be svg")
    _require(root.attrib.get("width") is not None, f"{label}: missing width")
    _require(root.attrib.get("height") is not None, f"{label}: missing height")
    _require(root.attrib.get("viewBox") is not None, f"{label}: missing viewBox")
    return root


def _require_speedup_svg_semantics(
    path: Path,
    cases_by_name: dict[str, dict[str, Any]],
) -> None:
    """Verify that the speedup bar chart labels every measured-speedup case."""
    root = _require_svg_document(path, "speedup_svg")
    text_content = _svg_text_content(root)
    expected_case_names = [
        case_name
        for case_name, case_record in cases_by_name.items()
        if _summary_numeric_value(case_record, SPEEDUP_VS_DISABLED_CACHE_KEY) is not None
    ]
    if not expected_case_names:
        _require(
            SPEEDUP_SVG_EMPTY_MESSAGE in text_content,
            "speedup_svg must state that no measured speedup values are available",
        )
        return
    for case_name in expected_case_names:
        _require(
            case_name in text_content,
            f"speedup_svg must include case label {case_name}",
        )


def _require_scatter_svg_semantics(
    path: Path,
    *,
    label: str,
    cases_by_name: dict[str, dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
) -> None:
    """Verify that a scatter figure has one plotted point per summary data pair."""
    root = _require_svg_document(path, label)
    text_content = _svg_text_content(root)
    expected_point_count = sum(
        1
        for case_record in cases_by_name.values()
        if _summary_numeric_value(case_record, x_field) is not None
        and _summary_numeric_value(case_record, y_field) is not None
    )
    if expected_point_count == 0:
        empty_message = f"No paired values for {title}"
        _require(
            empty_message in text_content,
            f"{label} must state that no paired values are available",
        )
    else:
        _require(title in text_content, f"{label} must include figure title {title!r}")
    observed_point_count = _svg_element_count(root, "circle")
    _require(
        observed_point_count == expected_point_count,
        f"{label} circle count must match summary data pairs",
    )


def _require_environment_snapshot(path: Path) -> None:
    """Verify that the environment snapshot has the expected schema marker."""
    payload = _as_json_object(
        json.loads(path.read_text(encoding="utf-8")),
        "environment_snapshot",
    )
    schema_version = _as_json_string(
        payload.get("snapshot_schema_version"),
        "environment_snapshot.snapshot_schema_version",
    )
    _require(
        schema_version == ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        f"environment_snapshot.snapshot_schema_version must be {ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION!r}",
    )


def _require_manifest_snapshot(path: Path) -> None:
    """Verify that the archived manifest snapshot is not an empty placeholder."""
    manifest_text = path.read_text(encoding="utf-8")
    _require(bool(manifest_text.strip()), "manifest_snapshot: file must not be empty")
    _require("[suite]" in manifest_text, "manifest_snapshot: missing [suite] table")


def _require_paper_artifact_semantics(
    summary_payload: dict[str, Any],
    resolved_artifact_paths: dict[str, Path],
    *,
    bundle_root: Path,
    original_output_dir: Path | None,
) -> int:
    """Verify that required paper artifacts are not only hashed but readable."""
    cases_by_name = _summary_cases_by_name(summary_payload)
    repeat_timing_rows = _repeat_timing_rows_from_summary(
        summary_payload,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )
    missing_artifacts = [
        name for name in REQUIRED_PAPER_ARTIFACT_NAMES if name not in resolved_artifact_paths
    ]
    _require(
        not missing_artifacts,
        f"artifact_fingerprints: missing required paper artifacts {missing_artifacts}",
    )
    _require_environment_snapshot(resolved_artifact_paths["environment_snapshot"])
    _require_case_summary_csv(resolved_artifact_paths["case_summary_csv"], cases_by_name)
    _require_case_summary_markdown(
        resolved_artifact_paths["case_summary_markdown"],
        cases_by_name,
    )
    _require_correlation_csv(resolved_artifact_paths["correlation_csv"], summary_payload)
    _require_command_timing_csv(
        resolved_artifact_paths["command_timing_csv"],
        summary_payload,
    )
    _require_command_timing_markdown(
        resolved_artifact_paths["command_timing_markdown"],
        summary_payload,
    )
    _require_repeat_timing_csv(
        resolved_artifact_paths["repeat_timing_csv"],
        repeat_timing_rows,
    )
    _require_repeat_timing_markdown(
        resolved_artifact_paths["repeat_timing_markdown"],
        repeat_timing_rows,
    )
    _require_speedup_svg_semantics(resolved_artifact_paths["speedup_svg"], cases_by_name)
    _require_scatter_svg_semantics(
        resolved_artifact_paths["hit_rate_svg"],
        label="hit_rate_svg",
        cases_by_name=cases_by_name,
        x_field="cache_hit_rate_percent",
        y_field=SPEEDUP_VS_DISABLED_CACHE_KEY,
        title=HIT_RATE_SCATTER_TITLE,
    )
    _require_scatter_svg_semantics(
        resolved_artifact_paths["trace_svg"],
        label="trace_svg",
        cases_by_name=cases_by_name,
        x_field="trace_metadata_fraction_percent",
        y_field="trace_estimated_average_speedup",
        title=TRACE_METADATA_SCATTER_TITLE,
    )
    _require_manifest_snapshot(resolved_artifact_paths["manifest_snapshot"])
    return len(REQUIRED_PAPER_ARTIFACT_NAMES)


def _require_artifact_index_alignment(
    summary_payload: dict[str, Any],
    artifact_fingerprints: dict[str, Any],
) -> int:
    """Verify that the human-facing artifact index matches fingerprint records."""
    artifact_index = _as_json_object(summary_payload.get("artifacts"), "artifacts")
    fingerprint_names = set(artifact_fingerprints)
    index_names = set(artifact_index)
    missing_index_names = sorted(fingerprint_names - index_names)
    unexpected_index_names = sorted(index_names - fingerprint_names)
    _require(
        not missing_index_names,
        f"artifacts: missing entries for fingerprints {missing_index_names}",
    )
    _require(
        not unexpected_index_names,
        f"artifacts: unexpected entries without fingerprints {unexpected_index_names}",
    )
    for artifact_name in sorted(fingerprint_names):
        indexed_path = _as_json_string(
            artifact_index.get(artifact_name),
            f"artifacts.{artifact_name}",
        )
        fingerprint_record = _as_json_object(
            artifact_fingerprints.get(artifact_name),
            f"artifact_fingerprints.{artifact_name}",
        )
        fingerprint_path = _as_json_string(
            fingerprint_record.get("path"),
            f"artifact_fingerprints.{artifact_name}.path",
        )
        _require(
            indexed_path == fingerprint_path,
            f"artifacts.{artifact_name} must match artifact_fingerprints.{artifact_name}.path",
        )
    return len(fingerprint_names)


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
    verified_artifact_index_count = _require_artifact_index_alignment(
        summary_payload,
        artifact_fingerprints,
    )
    command_fingerprints = summary_payload.get("command_log_fingerprints", [])
    _require(
        isinstance(command_fingerprints, list),
        "command_log_fingerprints must be a JSON array",
    )
    verified_command_record_count = _require_command_record_alignment(
        summary_payload,
        command_fingerprints,
    )
    original_output_dir = _original_output_dir(summary_payload)
    verified_evidence_file_count = _require_evidence_fingerprint_matches(
        summary_payload,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )
    verified_external_command_log_count = _require_external_timing_reports_from_summary(
        summary_payload,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )

    verified_artifact_count = 0
    resolved_artifact_paths: dict[str, Path] = {}
    for artifact_name, raw_record in artifact_fingerprints.items():
        record = _as_json_object(raw_record, f"artifact_fingerprints.{artifact_name}")
        artifact_label = f"artifact_fingerprints.{artifact_name}"
        if record.get("exists", True) is False:
            _require_fingerprint_match(
                record,
                artifact_label,
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
        else:
            resolved_artifact_paths[artifact_name] = _resolve_present_fingerprint_path(
                record,
                artifact_label,
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
        verified_artifact_count += 1
    verified_paper_artifact_semantic_count = _require_paper_artifact_semantics(
        summary_payload,
        resolved_artifact_paths,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )

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
        "verified_artifact_index_count": verified_artifact_index_count,
        "verified_paper_artifact_semantic_count": verified_paper_artifact_semantic_count,
        "verified_evidence_file_count": verified_evidence_file_count,
        "verified_command_record_count": verified_command_record_count,
        "verified_command_log_count": verified_log_count,
        "verified_external_command_log_count": verified_external_command_log_count,
    }


def _pipeline_report_output_dir(report_path: Path, original_output_dir: Path) -> Path:
    """Return the best local output directory candidate for a pipeline report."""
    if original_output_dir.exists():
        return original_output_dir
    sibling_output_dir = report_path.parent / original_output_dir.name
    if sibling_output_dir.exists():
        return sibling_output_dir
    return report_path.parent


def _require_pipeline_stage_report_fingerprints(
    pipeline_payload: dict[str, Any],
    *,
    pipeline_report_path: Path,
    original_output_dir: Path,
    expected_stage_names: tuple[str, ...],
) -> int:
    """Verify the stage report fingerprints embedded in a pipeline report."""
    stage_fingerprints = pipeline_payload.get(STAGE_REPORT_FINGERPRINTS_KEY)
    _require(
        isinstance(stage_fingerprints, list),
        f"{STAGE_REPORT_FINGERPRINTS_KEY} must be a JSON array",
    )
    _require(
        len(stage_fingerprints) == len(expected_stage_names),
        f"{STAGE_REPORT_FINGERPRINTS_KEY} must match the pipeline stage count",
    )
    raw_stages = pipeline_payload.get("stages")
    _require(isinstance(raw_stages, list), "stages must be a JSON array")
    bundle_root = _pipeline_report_output_dir(pipeline_report_path, original_output_dir)
    verified_count = 0
    for index, raw_record in enumerate(stage_fingerprints):
        record = _as_json_object(raw_record, f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}]")
        stage_name = _as_json_string(
            record.get("name"),
            f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}].name",
        )
        _require(
            stage_name == expected_stage_names[index],
            f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}].name must match stages[{index}].name",
        )
        stage = _as_json_object(raw_stages[index], f"stages[{index}]")
        stage_status = _as_json_string(stage.get("status"), f"stages[{index}].status")
        fingerprint_required = not (
            stage_name == PIPELINE_STAGE_PREPARE_ARTIFACTS
            and stage_status == PREFLIGHT_STATUS_SKIPPED
        )
        raw_report_record = record.get("report")
        if raw_report_record is None:
            _require(
                not fingerprint_required,
                f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}].report is required for {stage_name}",
            )
            continue
        report_record = _as_json_object(
            raw_report_record,
            f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}].report",
        )
        report_label = f"{STAGE_REPORT_FINGERPRINTS_KEY}[{index}].report"
        if fingerprint_required:
            _resolve_present_fingerprint_path(
                report_record,
                report_label,
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
        else:
            _require_fingerprint_match(
                report_record,
                report_label,
                bundle_root=bundle_root,
                original_output_dir=original_output_dir,
            )
        verified_count += 1
    return verified_count


def _require_pipeline_success_stages(
    pipeline_payload: dict[str, Any],
) -> tuple[str, ...]:
    """Verify that a passed pipeline report contains the full success path."""
    raw_stages = pipeline_payload.get("stages")
    _require(isinstance(raw_stages, list), "stages must be a JSON array")
    stage_names = tuple(
        _as_json_string(
            _as_json_object(raw_stage, f"stages[{index}]").get("name"),
            f"stages[{index}].name",
        )
        for index, raw_stage in enumerate(raw_stages)
    )
    _require(stage_names == REQUIRED_PIPELINE_STAGE_NAMES, PIPELINE_REQUIRED_STAGES_ERROR)
    for index, stage_name in enumerate(REQUIRED_PIPELINE_STAGE_NAMES):
        stage = _as_json_object(raw_stages[index], f"stages[{index}]")
        stage_status = _as_json_string(stage.get("status"), f"stages[{index}].status")
        allowed_statuses = PIPELINE_SUCCESS_STAGE_STATUSES[stage_name]
        _require(
            stage_status in allowed_statuses,
            (
                f"stages[{index}].status for {stage_name} must be one of "
                f"{MODEL_NAME_JOINER.join(allowed_statuses)}"
            ),
        )
        if (
            stage_name == PIPELINE_STAGE_PREPARE_ARTIFACTS
            and stage_status == PREFLIGHT_STATUS_SKIPPED
        ):
            continue
        _as_json_string(stage.get("report_path"), f"stages[{index}].report_path")
    return stage_names


def _require_pipeline_report_modes(pipeline_payload: dict[str, Any]) -> dict[str, bool]:
    """Verify execution-mode flags stored in a passed pipeline report."""
    raw_modes = _as_json_object(pipeline_payload.get("modes"), "modes")
    modes = {
        mode_key: _as_json_bool(raw_modes.get(mode_key), f"modes.{mode_key}")
        for mode_key in PIPELINE_REQUIRED_MODE_KEYS
    }
    _require(not modes["dry_run"], PIPELINE_DRY_RUN_PASSED_ERROR)
    return modes


def _require_pipeline_runtime_overrides(
    runtime_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Verify CLI overrides do not weaken a final-paper pipeline report."""
    unsupported_keys = [
        override_key
        for override_key in runtime_overrides
        if override_key not in PIPELINE_ALLOWED_RUNTIME_OVERRIDE_KEYS
    ]
    _require(not unsupported_keys, PIPELINE_UNSUPPORTED_RUNTIME_OVERRIDE_ERROR)
    if "ablation_mode" in runtime_overrides:
        ablation_mode = _as_json_string(
            runtime_overrides.get("ablation_mode"),
            "suite.runtime_overrides.ablation_mode",
        )
        _require(
            ablation_mode == ABLATION_MODE_PAIRED,
            PIPELINE_ONE_SIDED_RUNTIME_OVERRIDE_ERROR,
        )
    return runtime_overrides


def _require_pipeline_suite_metadata(
    pipeline_payload: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Verify suite metadata needed to interpret a passed pipeline report."""
    suite_record = _as_json_object(pipeline_payload.get("suite"), "suite")
    _as_json_string(suite_record.get("name"), "suite.name")
    _as_json_string(suite_record.get("manifest_path"), "suite.manifest_path")
    manifest = _as_json_object(suite_record.get("manifest"), "suite.manifest")
    _as_json_string(manifest.get("path"), "suite.manifest.path")
    manifest_digest = _as_json_string(manifest.get("sha256"), "suite.manifest.sha256")
    _require(
        SHA256_HEX_PATTERN.fullmatch(manifest_digest) is not None,
        "suite.manifest.sha256 must be a 64-character hexadecimal SHA-256 digest",
    )
    manifest_size = _as_json_nonnegative_int(
        manifest.get("size_bytes"),
        "suite.manifest.size_bytes",
    )
    _require(manifest_size > 0, "suite.manifest.size_bytes must be positive")
    original_output_dir = Path(
        _as_json_string(suite_record.get("output_dir"), "suite.output_dir")
    )
    expected_gpus = _as_json_nonnegative_int(
        suite_record.get("expected_gpus"),
        "suite.expected_gpus",
    )
    _require(expected_gpus >= DEFAULT_EXPECTED_GPU_COUNT, PIPELINE_SUITE_GPU_ERROR)
    required_models = _as_string_tuple(
        suite_record.get("required_models"),
        "suite.required_models",
    )
    missing_models = [
        model_name
        for model_name in FINAL_PAPER_REQUIRED_MODELS
        if model_name not in required_models
    ]
    _require(not missing_models, PIPELINE_SUITE_REQUIRED_MODELS_ERROR)
    _require_pipeline_runtime_overrides(
        _as_json_object(suite_record.get("runtime_overrides"), "suite.runtime_overrides")
    )
    return suite_record, original_output_dir


def _require_pipeline_bundle_verification(
    pipeline_payload: dict[str, Any],
    *,
    pipeline_report_path: Path,
    original_output_dir: Path,
) -> dict[str, Any] | None:
    """Re-run the final output-bundle verification referenced by a pipeline report."""
    raw_verification = pipeline_payload.get(OUTPUT_BUNDLE_VERIFICATION_KEY)
    if raw_verification is None:
        return None
    recorded_verification = _as_json_object(
        raw_verification,
        OUTPUT_BUNDLE_VERIFICATION_KEY,
    )
    recorded_status = _as_json_string(
        recorded_verification.get("status"),
        f"{OUTPUT_BUNDLE_VERIFICATION_KEY}.status",
    )
    if recorded_status != "passed":
        _as_json_string(
            recorded_verification.get("detail"),
            f"{OUTPUT_BUNDLE_VERIFICATION_KEY}.detail",
        )
        return recorded_verification
    bundle_root = _pipeline_report_output_dir(pipeline_report_path, original_output_dir)
    verification = verify_output_bundle(bundle_root)
    for count_key in (
        "verified_artifact_count",
        "verified_artifact_index_count",
        "verified_paper_artifact_semantic_count",
        "verified_evidence_file_count",
        "verified_command_record_count",
        "verified_command_log_count",
        "verified_external_command_log_count",
    ):
        recorded_count = _as_json_nonnegative_int(
            recorded_verification.get(count_key),
            f"{OUTPUT_BUNDLE_VERIFICATION_KEY}.{count_key}",
        )
        actual_count = _as_json_nonnegative_int(
            verification.get(count_key),
            f"actual_{count_key}",
        )
        _require(
            recorded_count == actual_count,
            f"{OUTPUT_BUNDLE_VERIFICATION_KEY}.{count_key} must match current bundle verification",
        )
    return verification


def verify_pipeline_report(pipeline_report_path: Path) -> dict[str, Any]:
    """Verify a pipeline report and all stage reports it fingerprints."""
    _require(pipeline_report_path.exists(), f"missing pipeline report {pipeline_report_path}")
    pipeline_payload = _as_json_object(
        json.loads(pipeline_report_path.read_text(encoding="utf-8")),
        "pipeline_report",
    )
    schema_version = _as_json_string(
        pipeline_payload.get("pipeline_report_schema_version"),
        "pipeline_report_schema_version",
    )
    _require(
        schema_version == PIPELINE_REPORT_SCHEMA_VERSION,
        f"pipeline_report_schema_version must be {PIPELINE_REPORT_SCHEMA_VERSION!r}",
    )
    pipeline_status = _as_json_string(pipeline_payload.get("status"), "status")
    _require(
        pipeline_status == PIPELINE_STATUS_PASSED,
        f"{PIPELINE_REPORT_PASSED_STATUS_ERROR}; observed {pipeline_status!r}",
    )
    _require_pipeline_report_modes(pipeline_payload)
    _suite_record, original_output_dir = _require_pipeline_suite_metadata(
        pipeline_payload
    )
    expected_stage_names = _require_pipeline_success_stages(pipeline_payload)
    verified_stage_report_count = _require_pipeline_stage_report_fingerprints(
        pipeline_payload,
        pipeline_report_path=pipeline_report_path,
        original_output_dir=original_output_dir,
        expected_stage_names=expected_stage_names,
    )
    bundle_verification = _require_pipeline_bundle_verification(
        pipeline_payload,
        pipeline_report_path=pipeline_report_path,
        original_output_dir=original_output_dir,
    )
    _require(
        bundle_verification is not None
        and bundle_verification.get("status") == PIPELINE_STATUS_PASSED,
        PIPELINE_BUNDLE_VERIFICATION_REQUIRED_ERROR,
    )
    return {
        "status": "passed",
        "pipeline_report": str(pipeline_report_path),
        "pipeline_status": pipeline_status,
        "verified_stage_report_count": verified_stage_report_count,
        OUTPUT_BUNDLE_VERIFICATION_KEY: bundle_verification,
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
                "mode_controls": case_mode_control_record(case),
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
                    "ablation_mode": case.ablation_mode,
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
            "runtime_overrides": dict(config.runtime_overrides),
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


def _mode_control_env_values(env: dict[str, str]) -> dict[str, str | None]:
    """Return only the cache mode environment values that matter for auditing."""
    return {key: env.get(key) for key in MODE_CONTROL_ENV_KEYS}


def command_environment_snapshot(env: dict[str, str] | None) -> dict[str, str | None]:
    """Return a small, non-secret environment snapshot for one command."""
    source_env = os.environ if env is None else env
    return {
        key: source_env.get(key)
        for key in dict.fromkeys(COMMAND_ENV_SNAPSHOT_KEYS)
    }


def _external_pair_mode_control_record(case: CaseConfig) -> dict[str, Any]:
    """Record the effective disabled/enabled controls for an external pair."""
    disabled_controls = _mode_control_env_values(
        _default_case_env(case.disabled_env, disabled=True)
    )
    enabled_controls = _mode_control_env_values(
        _default_case_env(case.enabled_env, disabled=False)
    )
    return {
        "kind": case.kind,
        "ablation_mode": case.ablation_mode,
        TIMING_MODES_KEY: list(_external_timing_modes_for_ablation(case)),
        "disabled_command": case.disabled_command,
        "enabled_command": case.enabled_command,
        "disabled_env": disabled_controls,
        "enabled_env": enabled_controls,
        ENV_FLAG_FALSE_VALUES_KEY: list(ENV_FLAG_FALSE_VALUES),
        "disabled_cache_disabled": env_flag_is_enabled(
            disabled_controls[SEVENNET_DISABLE_ENV]
        ),
        "enabled_cache_disabled": env_flag_is_enabled(
            enabled_controls[SEVENNET_DISABLE_ENV]
        ),
    }


def _external_pair_mode_control_errors(case: CaseConfig) -> list[str]:
    """Return configuration errors that would make paired modes ambiguous."""
    mode_control = _external_pair_mode_control_record(case)
    timing_modes = _external_timing_modes_for_ablation(case)
    errors: list[str] = []
    if (
        EXTERNAL_DISABLED_COMMAND_LABEL in timing_modes
        and not mode_control["disabled_cache_disabled"]
    ):
        errors.append(f"disabled mode must set {SEVENNET_DISABLE_ENV}")
    if (
        EXTERNAL_ENABLED_COMMAND_LABEL in timing_modes
        and mode_control["enabled_cache_disabled"]
    ):
        errors.append(f"enabled mode must leave {SEVENNET_DISABLE_ENV} unset or false")
    return errors


def case_mode_control_record(case: CaseConfig) -> dict[str, Any]:
    """Return a compact mode-control record for plan and summary artifacts."""
    if case.kind == "external_pair":
        return _external_pair_mode_control_record(case)
    if case.kind == "sevennet_lammps":
        return {
            "kind": case.kind,
            "paired_mode_source": str(EXPERIMENT_DRIVER_PATH),
            "ablation_mode": case.ablation_mode,
            "disabled_case": BASELINE_CASE_NAME,
            "enabled_case": ISODELTA_CASE_NAME,
            "disabled_env": {SEVENNET_DISABLE_ENV: ENV_FLAG_ENABLED},
            "enabled_env": {SEVENNET_DISABLE_ENV: None},
            ENV_FLAG_FALSE_VALUES_KEY: list(ENV_FLAG_FALSE_VALUES),
        }
    return {"kind": case.kind, "paired_mode_source": None}


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
        "--ablation-mode",
        case.ablation_mode,
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
    if _uses_paired_ablation_mode(case) and case.min_speedup is not None:
        argv.extend(["--min-speedup", str(case.min_speedup)])
    if _uses_paired_ablation_mode(case) and trace_paths:
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
    bundle_evidence = (
        experiment_dir / BUNDLE_EVIDENCE_NAME
        if _uses_paired_ablation_mode(case) and trace_paths
        else None
    )
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
    timing_modes = _external_timing_modes_for_ablation(case)

    for repeat_index in range(case.repeat_count):
        if EXTERNAL_DISABLED_COMMAND_LABEL in timing_modes:
            disabled_record = run_shell_command(
                name=f"{case.name}:{EXTERNAL_DISABLED_COMMAND_LABEL}:{repeat_index}",
                command=str(case.disabled_command),
                cwd=REPO_ROOT,
                env=_default_case_env(case.disabled_env, disabled=True),
                timeout_seconds=case.command_timeout_seconds,
                stdout_path=log_dir / f"{EXTERNAL_DISABLED_COMMAND_LABEL}_{repeat_index}.stdout.log",
                stderr_path=log_dir / f"{EXTERNAL_DISABLED_COMMAND_LABEL}_{repeat_index}.stderr.log",
                dry_run=dry_run,
            )
            command_records.append(disabled_record)
            if disabled_record.returncode == SUCCESS_RETURN_CODE:
                disabled_times.append(disabled_record.elapsed_seconds)

        if EXTERNAL_ENABLED_COMMAND_LABEL in timing_modes:
            enabled_record = run_shell_command(
                name=f"{case.name}:{EXTERNAL_ENABLED_COMMAND_LABEL}:{repeat_index}",
                command=str(case.enabled_command),
                cwd=REPO_ROOT,
                env=_default_case_env(case.enabled_env, disabled=False),
                timeout_seconds=case.command_timeout_seconds,
                stdout_path=log_dir / f"{EXTERNAL_ENABLED_COMMAND_LABEL}_{repeat_index}.stdout.log",
                stderr_path=log_dir / f"{EXTERNAL_ENABLED_COMMAND_LABEL}_{repeat_index}.stderr.log",
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
        "ablation_mode": case.ablation_mode,
        TIMING_MODES_KEY: list(_external_timing_modes_for_ablation(case)),
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
        MODE_CONTROLS_KEY: case_mode_control_record(case),
        COMMANDS_KEY: [asdict(record) for record in command_records],
        COMMAND_LOG_FINGERPRINTS_KEY: command_log_fingerprints(command_records),
    }


def _validate_external_timing_mode_controls(
    report: dict[str, Any],
    case: CaseConfig,
) -> dict[str, Any]:
    """Require timing-report mode controls to match the manifest case."""
    observed_controls = _as_json_object(
        report.get(MODE_CONTROLS_KEY),
        MODE_CONTROLS_KEY,
    )
    expected_controls = case_mode_control_record(case)
    _require(
        observed_controls == expected_controls,
        f"{MODE_CONTROLS_KEY} must match manifest disabled/enabled controls",
    )
    return observed_controls


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


def _validate_timing_mean(
    report: dict[str, Any],
    key: str,
    timing_values: list[float],
) -> float | None:
    """Validate a mean timing, allowing null only when that mode did not run."""
    expected_value = _mean(timing_values)
    if expected_value is None:
        observed_value = _as_json_optional_nonnegative_number(report.get(key), key)
        _require(observed_value is None, f"{key} must be null")
        return None
    observed_value = _as_json_positive_number(report.get(key), key)
    _require(
        _timing_values_close(observed_value, expected_value),
        f"{key} must match raw timing samples",
    )
    return observed_value


def _as_json_command_payload(value: Any, field_name: str) -> str | list[str]:
    """Validate a command payload recorded in external timing provenance."""
    if isinstance(value, str):
        _require(bool(value.strip()), f"{field_name} must not be empty")
        return value
    _require(isinstance(value, list), f"{field_name} must be a string or array")
    _require(bool(value), f"{field_name} must not be empty")
    for index, item in enumerate(value):
        _as_json_string(item, f"{field_name}[{index}]")
    return value


def _validate_external_timing_command_records(
    report: dict[str, Any],
    case: CaseConfig,
    repeat_count: int,
    mode_controls: dict[str, Any],
) -> int:
    """Require external timing reports to include one successful command per repeat."""
    raw_commands = report.get(COMMANDS_KEY)
    _require(isinstance(raw_commands, list), f"{COMMANDS_KEY} must be a JSON array")
    required_env_keys = tuple(dict.fromkeys(COMMAND_ENV_SNAPSHOT_KEYS))
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    for index, raw_record in enumerate(raw_commands):
        command_record = _as_json_object(raw_record, f"{COMMANDS_KEY}[{index}]")
        command_name = _as_json_string(command_record.get("name"), f"{COMMANDS_KEY}[{index}].name")
        command_payload = _as_json_command_payload(
            command_record.get("command"),
            f"{COMMANDS_KEY}[{index}].command",
        )
        returncode = _as_json_nonnegative_int(
            command_record.get("returncode"),
            f"{COMMANDS_KEY}[{index}].returncode",
        )
        _as_json_nonnegative_number(
            command_record.get("elapsed_seconds"),
            f"{COMMANDS_KEY}[{index}].elapsed_seconds",
        )
        _as_json_string(command_record.get("stdout_path"), f"{COMMANDS_KEY}[{index}].stdout_path")
        _as_json_string(command_record.get("stderr_path"), f"{COMMANDS_KEY}[{index}].stderr_path")
        _as_json_string(command_record.get("cwd"), f"{COMMANDS_KEY}[{index}].cwd")
        tracked_env = _as_json_object(
            command_record.get("tracked_env"),
            f"{COMMANDS_KEY}[{index}].tracked_env",
        )
        for env_key in required_env_keys:
            _require(
                env_key in tracked_env,
                f"{COMMANDS_KEY}[{index}].tracked_env missing {env_key}",
            )
            env_value = tracked_env[env_key]
            _require(
                env_value is None or isinstance(env_value, str),
                f"{COMMANDS_KEY}[{index}].tracked_env.{env_key} must be a string or null",
            )
        records_by_name.setdefault(command_name, []).append(
            {
                "command": command_payload,
                "returncode": returncode,
                "tracked_env": tracked_env,
            }
        )

    verified_count = 0
    expected_modes: list[tuple[str, str, dict[str, Any]]] = []
    timing_modes = _external_timing_modes_for_ablation(case)
    if EXTERNAL_DISABLED_COMMAND_LABEL in timing_modes:
        expected_modes.append(
            (
                EXTERNAL_DISABLED_COMMAND_LABEL,
                str(case.disabled_command),
                _as_json_object(
                    mode_controls.get("disabled_env"),
                    "mode_controls.disabled_env",
                ),
            )
        )
    if EXTERNAL_ENABLED_COMMAND_LABEL in timing_modes:
        expected_modes.append(
            (
                EXTERNAL_ENABLED_COMMAND_LABEL,
                str(case.enabled_command),
                _as_json_object(
                    mode_controls.get("enabled_env"),
                    "mode_controls.enabled_env",
                ),
            )
        )
    for mode_label, expected_command, expected_env in expected_modes:
        for repeat_index in range(repeat_count):
            expected_name = f"{case.name}:{mode_label}:{repeat_index}"
            matching_records = records_by_name.get(expected_name, [])
            _require(
                len(matching_records) == 1,
                f"{COMMANDS_KEY} must contain exactly one {expected_name} command record",
            )
            record = matching_records[0]
            _require(
                record["command"] == expected_command,
                f"{COMMANDS_KEY}.{expected_name}.command must match manifest {mode_label}_command",
            )
            _require(
                record["returncode"] == SUCCESS_RETURN_CODE,
                f"{COMMANDS_KEY}.{expected_name}.returncode must be {SUCCESS_RETURN_CODE}",
            )
            tracked_env = _as_json_object(
                record["tracked_env"],
                f"{COMMANDS_KEY}.{expected_name}.tracked_env",
            )
            for env_key in MODE_CONTROL_ENV_KEYS:
                _require(
                    tracked_env.get(env_key) == expected_env.get(env_key),
                    f"{COMMANDS_KEY}.{expected_name}.tracked_env.{env_key} must match {mode_label} mode control",
                )
            verified_count += 1
    return verified_count


def _require_external_command_log_fingerprints(
    report: dict[str, Any],
    *,
    report_path: Path | None,
    bundle_root: Path | None = None,
    original_output_dir: Path | None = None,
) -> int:
    """Verify external timing command log fingerprints and optionally their files."""
    command_fingerprints = report.get(COMMAND_LOG_FINGERPRINTS_KEY)
    _require(
        isinstance(command_fingerprints, list),
        f"{COMMAND_LOG_FINGERPRINTS_KEY} must be a JSON array",
    )
    verified_count = _require_command_record_alignment(report, command_fingerprints)
    if report_path is None:
        return verified_count
    fingerprint_root = bundle_root if bundle_root is not None else report_path.parent
    for index, raw_fingerprint in enumerate(command_fingerprints):
        fingerprint_record = _as_json_object(
            raw_fingerprint,
            f"{COMMAND_LOG_FINGERPRINTS_KEY}[{index}]",
        )
        for stream_name in ("stdout", "stderr"):
            stream_record = _as_json_object(
                fingerprint_record.get(stream_name),
                f"{COMMAND_LOG_FINGERPRINTS_KEY}[{index}].{stream_name}",
            )
            _require(
                stream_record.get("exists") is True,
                f"{COMMAND_LOG_FINGERPRINTS_KEY}[{index}].{stream_name} must exist",
            )
            _require_fingerprint_match(
                stream_record,
                f"{COMMAND_LOG_FINGERPRINTS_KEY}[{index}].{stream_name}",
                bundle_root=fingerprint_root,
                original_output_dir=original_output_dir,
            )
    return verified_count


def validate_external_timing_report(
    report: dict[str, Any],
    case: CaseConfig,
    *,
    report_path: Path | None = None,
    bundle_root: Path | None = None,
    original_output_dir: Path | None = None,
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
    ablation_mode = _as_json_string(report.get("ablation_mode"), "ablation_mode")
    _require(
        ablation_mode == case.ablation_mode,
        "ablation_mode must match manifest ablation_mode",
    )
    raw_timing_modes = report.get(TIMING_MODES_KEY)
    _require(isinstance(raw_timing_modes, list), f"{TIMING_MODES_KEY} must be a JSON array")
    timing_modes = tuple(
        _as_json_string(item, f"{TIMING_MODES_KEY}[{index}]")
        for index, item in enumerate(raw_timing_modes)
    )
    expected_timing_modes = _external_timing_modes_for_ablation(case)
    _require(
        timing_modes == expected_timing_modes,
        f"{TIMING_MODES_KEY} must match manifest ablation_mode",
    )
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
    expected_disabled_success_count = (
        repeat_count if EXTERNAL_DISABLED_COMMAND_LABEL in timing_modes else 0
    )
    expected_enabled_success_count = (
        repeat_count if EXTERNAL_ENABLED_COMMAND_LABEL in timing_modes else 0
    )
    _require(
        disabled_success_count == expected_disabled_success_count,
        f"{DISABLED_SUCCESS_COUNT_KEY} must match requested timing modes",
    )
    _require(
        enabled_success_count == expected_enabled_success_count,
        f"{ENABLED_SUCCESS_COUNT_KEY} must match requested timing modes",
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
    baseline_mean_seconds = _validate_timing_mean(
        report,
        BASELINE_MEAN_SECONDS_KEY,
        baseline_times,
    )
    enabled_mean_seconds = _validate_timing_mean(
        report,
        ENABLED_MEAN_SECONDS_KEY,
        enabled_times,
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
    if _uses_paired_ablation_mode(case):
        speedup = _as_json_positive_number(
            report.get(SPEEDUP_VS_DISABLED_CACHE_KEY),
            SPEEDUP_VS_DISABLED_CACHE_KEY,
        )
        _require(
            baseline_mean_seconds is not None and enabled_mean_seconds is not None,
            f"{SPEEDUP_VS_DISABLED_CACHE_KEY} requires both timing modes",
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
    else:
        speedup = _as_json_optional_nonnegative_number(
            report.get(SPEEDUP_VS_DISABLED_CACHE_KEY),
            SPEEDUP_VS_DISABLED_CACHE_KEY,
        )
        _require(
            speedup is None,
            f"{SPEEDUP_VS_DISABLED_CACHE_KEY} must be null for one-sided ablation",
        )
    mode_controls = _validate_external_timing_mode_controls(report, case)
    verified_command_record_count = _validate_external_timing_command_records(
        report,
        case,
        repeat_count,
        mode_controls,
    )
    verified_command_log_count = _require_external_command_log_fingerprints(
        report,
        report_path=report_path,
        bundle_root=bundle_root,
        original_output_dir=original_output_dir,
    )
    return {
        "status": "passed",
        SCHEMA_VERSION_KEY: schema_version,
        CASE_NAME_KEY: case_name,
        MODEL_KEY: model_name,
        "ablation_mode": ablation_mode,
        TIMING_MODES_KEY: list(timing_modes),
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
        MODE_CONTROLS_KEY: mode_controls,
        "verified_command_record_count": verified_command_record_count,
        "verified_command_log_count": verified_command_log_count,
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


def _validate_one_sided_benchmark_report(
    report: dict[str, Any],
    case: CaseConfig,
) -> None:
    """Validate raw one-sided SevenNet timing without claiming a speedup."""
    expected_cases = ABLATION_MODE_BENCHMARK_CASES[case.ablation_mode]
    observed_ablation_mode = _as_json_string(
        report.get("ablation_mode"),
        f"{case.name}: ablation_mode",
    )
    _require(
        observed_ablation_mode == case.ablation_mode,
        f"{case.name}: benchmark ablation_mode must match manifest",
    )
    raw_benchmark_cases = report.get("benchmark_cases")
    _require(
        isinstance(raw_benchmark_cases, list),
        f"{case.name}: benchmark_cases must be a JSON array",
    )
    benchmark_cases = tuple(
        _as_json_string(item, f"{case.name}: benchmark_cases[{index}]")
        for index, item in enumerate(raw_benchmark_cases)
    )
    _require(
        benchmark_cases == expected_cases,
        f"{case.name}: benchmark_cases must match ablation_mode",
    )
    summary = _as_json_object(report.get("summary"), f"{case.name}: summary")
    _require(
        summary.get("speedup_vs_disabled_cache") is None,
        f"{case.name}: one-sided ablation report must not claim speedup",
    )
    results = report.get("results")
    _require(isinstance(results, list), f"{case.name}: results must be a JSON array")
    _require(bool(results), f"{case.name}: one-sided ablation report must include results")
    observed_cases: list[str] = []
    for index, result in enumerate(results):
        result_record = _as_json_object(result, f"{case.name}: results[{index}]")
        observed_cases.append(
            _as_json_string(result_record.get("case"), f"{case.name}: results[{index}].case")
        )
    unexpected_cases = sorted(
        {case_name for case_name in observed_cases if case_name not in expected_cases}
    )
    missing_cases = [
        case_name for case_name in expected_cases if case_name not in observed_cases
    ]
    _require(
        not unexpected_cases,
        f"{case.name}: unexpected benchmark result cases: "
        + MODEL_NAME_JOINER.join(unexpected_cases),
    )
    _require(
        not missing_cases,
        f"{case.name}: missing benchmark result cases: "
        + MODEL_NAME_JOINER.join(missing_cases),
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
        benchmark_payload = benchmark_check.load_report(benchmark_report)
        if case.kind == "sevennet_lammps" and not _uses_paired_ablation_mode(case):
            _validate_one_sided_benchmark_report(benchmark_payload, case)
        else:
            benchmark_check.validate_report(
                benchmark_payload,
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
        validate_external_timing_report(
            timing_payload,
            case,
            report_path=external_timing_report,
        )


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


def _command_timing_rows(command_records: list[CommandRecord]) -> list[dict[str, Any]]:
    """Return paper-table rows for launched command timing evidence."""
    return [
        {
            "name": record.name,
            "returncode": record.returncode,
            "elapsed_seconds": record.elapsed_seconds,
            "stdout_path": record.stdout_path,
            "stderr_path": record.stderr_path,
            "cwd": record.cwd,
        }
        for record in command_records
    ]


def _repeat_timing_rows(case_summaries: list[CaseSummary]) -> list[dict[str, Any]]:
    """Return raw repeat timing rows from validated benchmark/external evidence."""
    rows: list[dict[str, Any]] = []
    for summary in case_summaries:
        benchmark_payload = _load_json_if_exists(
            Path(summary.benchmark_report) if summary.benchmark_report is not None else None
        )
        external_payload = _load_json_if_exists(
            Path(summary.external_timing_report)
            if summary.external_timing_report is not None
            else None
        )
        rows.extend(
            _repeat_timing_rows_from_sources(
                summary.case_name,
                summary.model,
                summary.kind,
                benchmark_payload=benchmark_payload,
                external_payload=external_payload,
            )
        )
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    """Write rows to CSV with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    resolved_fieldnames = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    """Write rows to a compact GitHub-flavored markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("| empty |\n| --- |\n", encoding="utf-8")
        return
    headers = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
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
    rows: list[dict[str, Any]] = []
    for x_name, y_name in CORRELATION_METRIC_PAIRS:
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
        path.write_text(_empty_svg(SPEEDUP_SVG_EMPTY_MESSAGE), encoding="utf-8")
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
    command_timing_rows = _command_timing_rows(command_records)
    repeat_timing_rows = _repeat_timing_rows(case_summaries)
    case_summary_csv = tables_dir / "case_summary.csv"
    case_summary_md = tables_dir / "case_summary.md"
    correlation_csv = tables_dir / "correlation.csv"
    command_timing_csv = tables_dir / "command_timing.csv"
    command_timing_md = tables_dir / "command_timing.md"
    repeat_timing_csv = tables_dir / "repeat_timing.csv"
    repeat_timing_md = tables_dir / "repeat_timing.md"
    write_csv(case_summary_csv, summary_rows)
    write_markdown_table(case_summary_md, summary_rows)
    write_csv(correlation_csv, correlation_rows)
    write_csv(
        command_timing_csv,
        command_timing_rows,
        fieldnames=PAPER_COMMAND_TIMING_COLUMNS,
    )
    write_markdown_table(
        command_timing_md,
        command_timing_rows,
        fieldnames=PAPER_COMMAND_TIMING_COLUMNS,
    )
    write_csv(
        repeat_timing_csv,
        repeat_timing_rows,
        fieldnames=PAPER_REPEAT_TIMING_COLUMNS,
    )
    write_markdown_table(
        repeat_timing_md,
        repeat_timing_rows,
        fieldnames=PAPER_REPEAT_TIMING_COLUMNS,
    )
    speedup_svg = figures_dir / "speedup_by_case.svg"
    hit_rate_svg = figures_dir / "hit_rate_vs_speedup.svg"
    trace_svg = figures_dir / "trace_metadata_fraction_vs_speedup.svg"
    write_speedup_svg(speedup_svg, case_summaries)
    write_scatter_svg(
        hit_rate_svg,
        case_summaries,
        x_field="cache_hit_rate_percent",
        y_field="speedup_vs_disabled_cache",
        title=HIT_RATE_SCATTER_TITLE,
        x_label="cache hit rate (%)",
        y_label="measured speedup",
    )
    write_scatter_svg(
        trace_svg,
        case_summaries,
        x_field="trace_metadata_fraction_percent",
        y_field="trace_estimated_average_speedup",
        title=TRACE_METADATA_SCATTER_TITLE,
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
        "command_timing_csv": command_timing_csv,
        "command_timing_markdown": command_timing_md,
        "repeat_timing_csv": repeat_timing_csv,
        "repeat_timing_markdown": repeat_timing_md,
        "speedup_svg": speedup_svg,
        "hit_rate_svg": hit_rate_svg,
        "trace_svg": trace_svg,
        "manifest_snapshot": manifest_snapshot_path,
    }
    optional_artifact_paths = {
        "preflight_report": config.output_dir / PREFLIGHT_REPORT_NAME,
        "preflight_environment_snapshot": config.output_dir / PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME,
        "run_plan": config.output_dir / PLAN_REPORT_NAME,
    }
    artifact_fingerprints = {
        name: generated_artifact_record(path) for name, path in artifact_paths.items()
    }
    artifact_fingerprints.update(
        {
            name: optional_file_fingerprint(path)
            for name, path in optional_artifact_paths.items()
        }
    )
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
            "runtime_overrides": dict(config.runtime_overrides),
        },
        "gpu_check": gpu_record,
        "suite_evidence": suite_evidence,
        "downloads": download_records,
        "cases": [asdict(summary) for summary in case_summaries],
        "case_mode_controls": {
            case.name: case_mode_control_record(case) for case in config.cases
        },
        "correlations": correlation_rows,
        "commands": [asdict(record) for record in command_records],
        "command_log_fingerprints": command_log_fingerprints(command_records),
        EVIDENCE_FINGERPRINTS_KEY: evidence_fingerprints(case_summaries),
        "artifacts": {
            name: str(path) for name, path in artifact_paths.items()
        }
        | {name: str(path) for name, path in optional_artifact_paths.items()},
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
        "command_timing_csv": str(command_timing_csv),
        "command_timing_markdown": str(command_timing_md),
        "repeat_timing_csv": str(repeat_timing_csv),
        "repeat_timing_markdown": str(repeat_timing_md),
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
    verify_output: bool = True,
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
    verify_stage_count = 1 if verify_output and not dry_run else 0
    total_stages = 1 + download_stage_count + len(config.cases) + 1 + verify_stage_count
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
                download_records.append(
                    download_artifact(
                        artifact,
                        dry_run=dry_run or collect_only,
                        progress_label=config.name,
                    )
                )
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
    stage_index += 1
    artifact_report: dict[str, Any] = dict(artifacts)
    if verify_output and not dry_run:
        _progress(config.name, stage_index, total_stages, "verifying output bundle")
        try:
            artifact_report["output_bundle_verification"] = verify_output_bundle(
                config.output_dir
            )
        except ClusterSuiteError as exc:
            print(f"[{config.name}] output bundle verification failed: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(artifact_report, indent=2), flush=True)
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
# Use "baseline-disabled" or "isodelta-enabled" only for quick ablation timing;
# final paper readiness requires the default paired mode.
ablation_mode = "paired"
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
disabled_env = {{ SEVENN_ISODELTA_HALO_DISABLE = "1" }}
enabled_env = {{}}
ablation_mode = "paired"
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
disabled_env = {{ SEVENN_ISODELTA_HALO_DISABLE = "1" }}
enabled_env = {{}}
ablation_mode = "paired"
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


def _runtime_override_comment(config: SuiteConfig) -> str:
    """Return one readable provenance comment for generated launch scripts."""
    if not config.runtime_overrides:
        return "# CLI runtime overrides: none."
    assignments = ", ".join(
        f"{key}={value}" for key, value in sorted(config.runtime_overrides.items())
    )
    return f"# CLI runtime overrides: {assignments}."


def write_slurm_script(
    path: Path,
    config: SuiteConfig,
    *,
    ablation_mode_override: str | None = None,
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
    """Write a commented SLURM wrapper that runs the full paper pipeline."""
    validate_suite_config(config)
    _require(config.expected_gpus >= MIN_REQUIRED_CASE_COUNT, "SLURM GPU count must be positive")
    _require(cpus_per_task >= MIN_REQUIRED_CASE_COUNT, "SLURM cpus-per-task must be positive")
    _require(
        not collect_only,
        "--write-slurm-script cannot be combined with --collect-only because the launcher runs --pipeline",
    )

    has_one_sided_ablation = _has_one_sided_ablation_case(config)
    plan_path = config.output_dir / PLAN_REPORT_NAME
    slurm_job_name = _safe_name(job_name)
    lines = [
        "#!/usr/bin/env bash",
        "# IsoDelta-Halo cluster paper suite launcher.",
        "# Submit with: sbatch <this-file>",
        "# The script writes pre-run evidence first, then runs the full pipeline.",
        _runtime_override_comment(config),
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
        f"PIPELINE_OUTPUT={_bash_quote(config.output_dir / PIPELINE_REPORT_NAME)}",
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
    if ablation_mode_override is not None:
        _append_bash_array_args(lines, "--ablation-mode-override", ablation_mode_override)
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
        ]
    )
    if has_one_sided_ablation:
        lines.extend(
            [
                "# Run the one-sided ablation suite; final-paper --pipeline is reserved for paired mode.",
                '"$PYTHON_BIN" "$SUITE_RUNNER" "${COMMON_ARGS[@]}"',
                "",
                "# Re-open the finished output bundle before allowing the SLURM job to succeed.",
                f'"$PYTHON_BIN" "$SUITE_RUNNER" --verify-output-bundle {_bash_quote(config.output_dir)}',
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Run the full paper pipeline: readiness, prepare, preflight, plan, suite, and bundle verify.",
                '"$PYTHON_BIN" "$SUITE_RUNNER" "${COMMON_ARGS[@]}" --pipeline --pipeline-report "$PIPELINE_OUTPUT"',
                "",
                "# Re-open the finished pipeline report before allowing the SLURM job to succeed.",
                '"$PYTHON_BIN" "$SUITE_RUNNER" --verify-pipeline-report "$PIPELINE_OUTPUT"',
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
    parser.add_argument("--verify-pipeline-report", type=Path, help="Verify a pipeline report and its stage fingerprints")
    parser.add_argument("--pipeline", action="store_true", help="Run readiness, prepare, preflight, plan, suite, and bundle verification")
    parser.add_argument("--pipeline-report", type=Path, help="Path for --pipeline JSON output")
    parser.add_argument("--readiness-check", action="store_true", help="Audit a manifest before final paper execution")
    parser.add_argument("--prepare-artifacts", action="store_true", help="Download and verify artifacts without using GPUs")
    parser.add_argument("--preflight-only", action="store_true", help="Run artifact, GPU, and case preflight checks only")
    parser.add_argument("--preflight-output", type=Path, help="Path for --preflight-only JSON output")
    parser.add_argument("--plan-only", action="store_true", help="Write a preflight JSON plan and exit")
    parser.add_argument("--plan-output", type=Path, help="Path for --plan-only JSON output")
    parser.add_argument("--output-dir", type=Path, help="Override suite.output_dir")
    parser.add_argument("--expected-gpus", type=int, help="Override suite.expected_gpus")
    parser.add_argument(
        "--ablation-mode-override",
        choices=ABLATION_MODE_CHOICES,
        help=(
            "Temporarily override ablation_mode for sevennet_lammps and "
            "external_pair cases"
        ),
    )
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
    overridden_config = SuiteConfig(
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
        runtime_overrides=dict(config.runtime_overrides),
    )
    return _apply_ablation_mode_override(
        overridden_config,
        args.ablation_mode_override,
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
    if args.verify_pipeline_report is not None:
        try:
            verification = verify_pipeline_report(args.verify_pipeline_report)
        except ClusterSuiteError as exc:
            print(f"IsoDelta-Halo pipeline report verification failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(verification, indent=2))
        return SUCCESS_RETURN_CODE
    if args.manifest is None:
        raise SystemExit("--manifest is required unless --write-template or a verify mode is used")
    try:
        config = _apply_cli_overrides(load_manifest(args.manifest), args)
        _require(
            args.preflight_output is None or args.preflight_only,
            "--preflight-output requires --preflight-only",
        )
        _require(
            args.pipeline_report is None or args.pipeline,
            "--pipeline-report requires --pipeline",
        )
        if args.pipeline:
            _require(args.write_slurm_script is None, "--pipeline cannot be combined with --write-slurm-script")
            _require(not args.readiness_check, "--pipeline cannot be combined with --readiness-check")
            _require(not args.prepare_artifacts, "--pipeline cannot be combined with --prepare-artifacts")
            _require(not args.preflight_only, "--pipeline cannot be combined with --preflight-only")
            _require(not args.plan_only, "--pipeline cannot be combined with --plan-only")
            _require(not args.collect_only, "--pipeline cannot be combined with --collect-only")
            report = run_pipeline(
                config,
                dry_run=args.dry_run,
                skip_downloads=args.skip_downloads,
                skip_gpu_check=args.skip_gpu_check,
                allow_gpu_mismatch=args.allow_gpu_mismatch,
                keep_going=args.keep_going,
                reuse_passed=args.reuse_passed,
                report_path=args.pipeline_report,
            )
            print(json.dumps({"pipeline_report": report["report_path"], "status": report["status"]}, indent=2))
            return SUCCESS_RETURN_CODE if report["status"] != PIPELINE_STATUS_FAILED else 1
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
                ablation_mode_override=args.ablation_mode_override,
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
