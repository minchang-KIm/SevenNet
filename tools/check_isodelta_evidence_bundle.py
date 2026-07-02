"""Validate a paper-ready IsoDelta-Halo evidence bundle.

The runtime benchmark report proves the concrete SevenNet/LAMMPS effect, while
model-agnostic MLIP trace evidence supports portability to other cutoff-graph
halo-exchange runtimes. This checker validates both kinds of artifacts together
so a manuscript can archive one reproducible acceptance gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


# Keep sibling tool imports path-local so tools/ does not need to become a
# Python package and command-line usage stays identical on developer machines.
REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
TRACE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
DEFAULT_MIN_TRACE_COUNT = 1
DEFAULT_MIN_DISTINCT_TRACE_MODELS = 1
MIN_COUNT_VALUE = 0
MIN_REQUIRED_TRACE_COUNT = 1
MIN_REQUIRED_DISTINCT_TRACE_MODELS = 1
MIN_NONNEGATIVE_VALUE = 0.0
MIN_PERCENT_VALUE = 0.0
MAX_PERCENT_VALUE = 100.0
MIN_POSITIVE_SPEEDUP = 0.0
MODEL_KEY = "model"
STATUS_KEY = "status"
PASSED_STATUS = "passed"
ARTIFACTS_KEY = "artifacts"
ARTIFACT_PATH_KEY = "path"
ARTIFACT_SHA256_KEY = "sha256"
ARTIFACT_SIZE_BYTES_KEY = "size_bytes"
ARTIFACT_ROLE_BENCHMARK_REPORT = "benchmark_report"
ARTIFACT_ROLE_TRACE_EVIDENCE = "trace_evidence"
TRACE_EVIDENCE_KEY = "trace_evidence"
BENCHMARK_EVIDENCE_KEY = "benchmark_evidence"
PATH_SEPARATOR = ", "
BYTES_PER_KIBIBYTE = 1024
BYTES_PER_MEBIBYTE = BYTES_PER_KIBIBYTE * BYTES_PER_KIBIBYTE
HASH_READ_CHUNK_BYTES = BYTES_PER_MEBIBYTE


class EvidenceBundleError(ValueError):
    """Raised when the combined evidence package is incomplete or weak."""


@dataclass(frozen=True)
class BundleThresholds:
    """Store cross-artifact gates for a publication evidence bundle."""

    max_abs_thermo_delta: float
    min_paired_thermo_count: int
    min_speedup: float | None
    min_hit_rate_percent: float
    min_enabled_cache_attempts: int
    min_enabled_cache_hits: int
    min_trace_hit_rate_percent: float
    min_trace_estimated_speedup: float | None
    min_trace_metadata_fraction_percent: float
    min_trace_count: int = DEFAULT_MIN_TRACE_COUNT
    min_distinct_trace_models: int = DEFAULT_MIN_DISTINCT_TRACE_MODELS
    require_successful_runs: bool = True


def _load_module(module_name: str, module_path: Path) -> Any:
    """Load a sibling validation tool by path without changing sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise EvidenceBundleError(f"cannot load validation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark_check = _load_module("isodelta_bundle_benchmark_check", BENCHMARK_CHECK_PATH)
trace_check = _load_module("isodelta_bundle_trace_check", TRACE_CHECK_PATH)


def _load_json_object(path: Path, field_name: str) -> dict[str, Any]:
    """Load one JSON object from disk with a schema-oriented error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceBundleError(f"{field_name} must be a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    """Raise a bundle-check error with a concise validation message."""
    if not condition:
        raise EvidenceBundleError(message)


def _validate_percent(value: float, field_name: str) -> None:
    """Require a percentage threshold to stay within the physical range."""
    _require(
        MIN_PERCENT_VALUE <= value <= MAX_PERCENT_VALUE,
        f"{field_name} must be between {MIN_PERCENT_VALUE:g} and {MAX_PERCENT_VALUE:g}",
    )


def _canonical_path_key(path: Path) -> str:
    """Return a stable path key for duplicate evidence detection."""
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _validate_unique_trace_evidence_paths(trace_evidence_paths: list[Path]) -> None:
    """Reject duplicate trace paths before counting portability evidence."""
    seen_paths: dict[str, Path] = {}
    duplicate_paths: list[str] = []
    for path in trace_evidence_paths:
        path_key = _canonical_path_key(path)
        if path_key in seen_paths:
            duplicate_paths.append(str(path))
        else:
            seen_paths[path_key] = path
    _require(
        not duplicate_paths,
        "duplicate trace evidence paths: " + PATH_SEPARATOR.join(duplicate_paths),
    )


def _artifact_record(path: Path) -> dict[str, str | int]:
    """Return a content fingerprint for one archived evidence artifact."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(HASH_READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise EvidenceBundleError(f"cannot fingerprint evidence artifact: {path}") from exc
    return {
        ARTIFACT_PATH_KEY: str(path),
        ARTIFACT_SHA256_KEY: digest.hexdigest(),
        ARTIFACT_SIZE_BYTES_KEY: size_bytes,
    }


def _validate_required_model_names(required_models: list[str]) -> list[str]:
    """Return sorted required model labels after rejecting empty or duplicate names."""
    _require(
        all(isinstance(model, str) and model.strip() for model in required_models),
        "required trace models must not include empty names",
    )
    normalized_models = [model.strip() for model in required_models]
    duplicate_models = sorted(
        {
            model
            for model in normalized_models
            if normalized_models.count(model) > 1
        }
    )
    _require(
        not duplicate_models,
        "duplicate required trace models: " + PATH_SEPARATOR.join(duplicate_models),
    )
    return sorted(normalized_models)


def validate_thresholds(thresholds: BundleThresholds) -> None:
    """Reject acceptance criteria that would make the bundle gate meaningless."""
    _require(
        thresholds.max_abs_thermo_delta >= MIN_NONNEGATIVE_VALUE,
        "max_abs_thermo_delta must be nonnegative",
    )
    _require(
        thresholds.min_paired_thermo_count >= MIN_REQUIRED_TRACE_COUNT,
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
    _require(
        thresholds.min_trace_count >= MIN_REQUIRED_TRACE_COUNT,
        "min_trace_count must be at least one",
    )
    _require(
        thresholds.min_distinct_trace_models >= MIN_REQUIRED_DISTINCT_TRACE_MODELS,
        "min_distinct_trace_models must be at least one",
    )
    if thresholds.min_speedup is not None:
        _require(
            thresholds.min_speedup > MIN_POSITIVE_SPEEDUP,
            "min_speedup must be positive when provided",
        )
    if thresholds.min_trace_estimated_speedup is not None:
        _require(
            thresholds.min_trace_estimated_speedup > MIN_POSITIVE_SPEEDUP,
            "min_trace_estimated_speedup must be positive when provided",
        )
    _validate_percent(thresholds.min_hit_rate_percent, "min_hit_rate_percent")
    _validate_percent(
        thresholds.min_trace_hit_rate_percent,
        "min_trace_hit_rate_percent",
    )
    _validate_percent(
        thresholds.min_trace_metadata_fraction_percent,
        "min_trace_metadata_fraction_percent",
    )


def _validate_trace_evidence_file(
    path: Path,
    thresholds: BundleThresholds,
) -> dict[str, Any]:
    """Validate one precomputed MLIP trace evidence JSON file."""
    evidence = _load_json_object(path, TRACE_EVIDENCE_KEY)
    trace_thresholds = trace_check.TraceThresholds(
        min_hit_rate_percent=thresholds.min_trace_hit_rate_percent,
        min_estimated_speedup=thresholds.min_trace_estimated_speedup,
        min_metadata_fraction_percent=thresholds.min_trace_metadata_fraction_percent,
    )
    try:
        validated = trace_check.validate_trace_evidence(evidence, trace_thresholds)
    except trace_check.TraceCheckError as exc:
        raise EvidenceBundleError(f"{path}: {exc}") from exc
    validated["path"] = str(path)
    return validated


def _validate_required_models(
    trace_evidence: list[dict[str, Any]],
    required_models: list[str],
) -> list[str]:
    """Require named MLIP models to appear in the validated trace evidence."""
    seen_models = {
        evidence.get(MODEL_KEY)
        for evidence in trace_evidence
        if isinstance(evidence.get(MODEL_KEY), str)
    }
    missing_models = [model for model in required_models if model not in seen_models]
    if missing_models:
        raise EvidenceBundleError(
            "missing required trace models: " + ", ".join(sorted(missing_models))
        )
    return sorted(model for model in seen_models if isinstance(model, str))


def _validate_distinct_model_count(
    validated_models: list[str],
    min_distinct_trace_models: int,
) -> None:
    """Require enough distinct MLIP labels to support a portability claim."""
    distinct_model_count = len(validated_models)
    if distinct_model_count < min_distinct_trace_models:
        raise EvidenceBundleError(
            f"distinct trace model count {distinct_model_count} is below "
            f"{min_distinct_trace_models}"
        )


def validate_bundle(
    *,
    benchmark_report: Path,
    trace_evidence_paths: list[Path],
    required_models: list[str],
    thresholds: BundleThresholds,
) -> dict[str, Any]:
    """Validate benchmark and portability evidence as one paper gate."""
    validate_thresholds(thresholds)
    _validate_unique_trace_evidence_paths(trace_evidence_paths)
    normalized_required_models = _validate_required_model_names(required_models)
    if len(trace_evidence_paths) < thresholds.min_trace_count:
        raise EvidenceBundleError(
            f"trace evidence count {len(trace_evidence_paths)} is below "
            f"{thresholds.min_trace_count}"
        )

    report_thresholds = benchmark_check.ReportThresholds(
        max_abs_thermo_delta=thresholds.max_abs_thermo_delta,
        min_paired_thermo_count=thresholds.min_paired_thermo_count,
        min_speedup=thresholds.min_speedup,
        min_hit_rate_percent=thresholds.min_hit_rate_percent,
        min_enabled_cache_attempts=thresholds.min_enabled_cache_attempts,
        min_enabled_cache_hits=thresholds.min_enabled_cache_hits,
        require_successful_runs=thresholds.require_successful_runs,
    )
    try:
        benchmark_evidence = benchmark_check.validate_report(
            benchmark_check.load_report(benchmark_report),
            report_thresholds,
        )
    except benchmark_check.ReportCheckError as exc:
        raise EvidenceBundleError(f"{benchmark_report}: {exc}") from exc

    trace_evidence = [
        _validate_trace_evidence_file(path, thresholds)
        for path in trace_evidence_paths
    ]
    validated_models = _validate_required_models(
        trace_evidence,
        normalized_required_models,
    )
    _validate_distinct_model_count(
        validated_models,
        thresholds.min_distinct_trace_models,
    )
    return {
        STATUS_KEY: PASSED_STATUS,
        "thresholds": asdict(thresholds),
        ARTIFACTS_KEY: {
            ARTIFACT_ROLE_BENCHMARK_REPORT: _artifact_record(benchmark_report),
            ARTIFACT_ROLE_TRACE_EVIDENCE: [
                _artifact_record(path) for path in trace_evidence_paths
            ],
        },
        BENCHMARK_EVIDENCE_KEY: benchmark_evidence,
        TRACE_EVIDENCE_KEY: trace_evidence,
        "trace_model_count": len(validated_models),
        "trace_models": validated_models,
        "required_trace_models": normalized_required_models,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and validate a complete evidence bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument(
        "--trace-evidence",
        action="append",
        default=[],
        type=Path,
        help="MLIP trace evidence JSON produced by check_isodelta_mlip_trace.py",
    )
    parser.add_argument(
        "--require-trace-model",
        action="append",
        default=[],
        help="Model label that must appear in at least one trace evidence file",
    )
    parser.add_argument(
        "--min-trace-count",
        type=int,
        default=DEFAULT_MIN_TRACE_COUNT,
        help="Minimum number of trace evidence files required",
    )
    parser.add_argument(
        "--min-distinct-trace-models",
        type=int,
        default=DEFAULT_MIN_DISTINCT_TRACE_MODELS,
        help="Minimum number of distinct model labels required in trace evidence",
    )
    parser.add_argument(
        "--max-abs-thermo-delta",
        type=float,
        default=benchmark_check.DEFAULT_MAX_ABS_THERMO_DELTA,
    )
    parser.add_argument(
        "--min-paired-thermo-count",
        type=int,
        default=benchmark_check.DEFAULT_MIN_PAIRED_THERMO_COUNT,
    )
    parser.add_argument("--min-speedup", type=float)
    parser.add_argument(
        "--min-hit-rate-percent",
        type=float,
        default=benchmark_check.DEFAULT_MIN_HIT_RATE_PERCENT,
    )
    parser.add_argument(
        "--min-enabled-cache-attempts",
        type=int,
        default=benchmark_check.DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS,
    )
    parser.add_argument(
        "--min-enabled-cache-hits",
        type=int,
        default=benchmark_check.DEFAULT_MIN_ENABLED_CACHE_HITS,
    )
    parser.add_argument(
        "--min-trace-hit-rate-percent",
        type=float,
        default=trace_check.DEFAULT_MIN_HIT_RATE_PERCENT,
    )
    parser.add_argument("--min-trace-estimated-speedup", type=float)
    parser.add_argument(
        "--min-trace-metadata-fraction-percent",
        type=float,
        default=trace_check.DEFAULT_MIN_METADATA_FRACTION_PERCENT,
    )
    parser.add_argument(
        "--allow-failed-runs",
        action="store_true",
        help="Skip benchmark returncode checks when inspecting partial reports",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    thresholds = BundleThresholds(
        max_abs_thermo_delta=args.max_abs_thermo_delta,
        min_paired_thermo_count=args.min_paired_thermo_count,
        min_speedup=args.min_speedup,
        min_hit_rate_percent=args.min_hit_rate_percent,
        min_enabled_cache_attempts=args.min_enabled_cache_attempts,
        min_enabled_cache_hits=args.min_enabled_cache_hits,
        min_trace_hit_rate_percent=args.min_trace_hit_rate_percent,
        min_trace_estimated_speedup=args.min_trace_estimated_speedup,
        min_trace_metadata_fraction_percent=args.min_trace_metadata_fraction_percent,
        min_trace_count=args.min_trace_count,
        min_distinct_trace_models=args.min_distinct_trace_models,
        require_successful_runs=not args.allow_failed_runs,
    )
    try:
        evidence = validate_bundle(
            benchmark_report=args.benchmark_report,
            trace_evidence_paths=args.trace_evidence,
            required_models=args.require_trace_model,
            thresholds=thresholds,
        )
    except EvidenceBundleError as exc:
        print(f"IsoDelta-Halo evidence bundle check failed: {exc}")
        return 1

    output_text = json.dumps(evidence, indent=2)
    if args.output is not None:
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
