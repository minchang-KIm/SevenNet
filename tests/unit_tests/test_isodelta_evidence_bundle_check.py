"""Unit tests for the IsoDelta-Halo paper evidence bundle checker.

The bundle checker combines SevenNet benchmark evidence with portable MLIP trace
evidence. These tests keep that publication gate reproducible without requiring
a LAMMPS binary or an external model runtime.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Load the bundle checker by path because tools/ is intentionally not a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_evidence_bundle.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_evidence_bundle",
    BUNDLE_CHECK_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
isodelta_evidence_bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_evidence_bundle
SPEC.loader.exec_module(isodelta_evidence_bundle)


# Named test values describe the acceptance rule instead of hiding paper gates
# inside raw literals spread through the assertions.
MAX_ABS_THERMO_DELTA = 1.0e-8
MIN_PAIRED_THERMO_COUNT = 2
MIN_SPEEDUP = 1.1
MIN_HIT_RATE_PERCENT = 75.0
MIN_ENABLED_ATTEMPTS = 10
MIN_ENABLED_HITS = 8
MIN_TRACE_METADATA_FRACTION_PERCENT = 20.0
MIN_TRACE_ESTIMATED_SPEEDUP = 1.15
OUT_OF_RANGE_PERCENT = 101.0
PERCENT_SCALE = 100.0
TRACE_BASELINE_SECONDS = 100.0
TRACE_ESTIMATED_SPEEDUP = 1.18
TRACE_METADATA_SECONDS = (
    TRACE_BASELINE_SECONDS * MIN_TRACE_METADATA_FRACTION_PERCENT / PERCENT_SCALE
)
TRACE_ENABLED_SECONDS = TRACE_BASELINE_SECONDS / TRACE_ESTIMATED_SPEEDUP
TRACE_LOOKUP_OVERHEAD_SECONDS = 0.0
RUN_TIMEOUT_SECONDS = 3600.0
BASELINE_LOOP_TIME_SECONDS = 12.0
ISODELTA_LOOP_TIME_SECONDS = 10.0
ZERO_SAMPLE_VARIANCE_LOOP_TIME_SECONDS = 0.0
ZERO_SAMPLE_STDDEV_LOOP_TIME_SECONDS = 0.0
EXPECTED_RESULT_COUNT = 4
EXPECTED_REPORT_SCHEMA_VERSION = "isodelta-benchmark-report-v1"
EXPECTED_BUNDLE_SCHEMA_VERSION = "isodelta-evidence-bundle-v1"
MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY = 2
ZERO_CACHE_COUNT = 0.0
ZERO_HIT_RATE_PERCENT = 0.0
DEFAULT_SUMMARY_RANK_COUNT = 1.0
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a test evidence artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_summary(
    attempts: float = MIN_ENABLED_ATTEMPTS,
    hits: float = MIN_ENABLED_HITS,
    hit_rate_percent: float | None = None,
    miss_disabled: float = ZERO_CACHE_COUNT,
    miss_no_cache: float | None = None,
    summary_rank_count: float = DEFAULT_SUMMARY_RANK_COUNT,
) -> dict[str, float]:
    """Create a complete IsoDelta-Halo cache summary for one run."""
    resolved_hit_rate_percent = (
        PERCENT_SCALE * hits / attempts
        if hit_rate_percent is None
        else hit_rate_percent
    )
    resolved_no_cache_misses = (
        attempts - hits if miss_no_cache is None else miss_no_cache
    )
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": resolved_hit_rate_percent,
        "summary_rank_count": summary_rank_count,
        "miss_disabled": miss_disabled,
        "miss_no-cache": resolved_no_cache_misses,
        "miss_neighbor-list-rebuilt": 0.0,
        "miss_shape-changed": 0.0,
        "miss_index-tensor-shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _disabled_cache_summary(attempts: float = MIN_ENABLED_ATTEMPTS) -> dict[str, float]:
    """Create a cache summary proving that the baseline ran with the cache off."""
    return _cache_summary(
        attempts=attempts,
        hits=ZERO_CACHE_COUNT,
        hit_rate_percent=ZERO_HIT_RATE_PERCENT,
        miss_disabled=attempts,
        miss_no_cache=ZERO_CACHE_COUNT,
    )


def _benchmark_report() -> dict[str, object]:
    """Create a benchmark report that proves correctness and speedup."""
    return {
        isodelta_evidence_bundle.benchmark_check.GENERATED_REPORT_COMMENT_KEY: (
            isodelta_evidence_bundle.benchmark_check.BENCHMARK_REPORT_COMMENT
        ),
        "provenance": {
            "report_schema_version": EXPECTED_REPORT_SCHEMA_VERSION,
            "git_commit": "0123456789abcdef",
            "git_branch": "isodelta-halo-runtime",
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
            "speedup_vs_disabled_cache": 1.2,
            "cases": {
                "baseline-disabled": {
                    "mean_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": (
                        ZERO_SAMPLE_VARIANCE_LOOP_TIME_SECONDS
                    ),
                    "sample_stddev_loop_time_seconds": (
                        ZERO_SAMPLE_STDDEV_LOOP_TIME_SECONDS
                    ),
                    "min_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
                "isodelta-enabled": {
                    "mean_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": (
                        ZERO_SAMPLE_VARIANCE_LOOP_TIME_SECONDS
                    ),
                    "sample_stddev_loop_time_seconds": (
                        ZERO_SAMPLE_STDDEV_LOOP_TIME_SECONDS
                    ),
                    "min_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
            },
            "final_thermo_delta_vs_disabled_cache": {
                "PotEng": {
                    "max_abs_delta": 1.0e-9,
                    "paired_count": float(MIN_PAIRED_THERMO_COUNT),
                },
            },
        },
        "results": [
            {
                "case": "baseline-disabled",
                "repeat_index": 0,
                "returncode": 0,
                "loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                "cache_summary": _disabled_cache_summary(),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 0,
                "returncode": 0,
                "loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                "cache_summary": _cache_summary(),
            },
            {
                "case": "baseline-disabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                "cache_summary": _disabled_cache_summary(),
            },
            {
                "case": "isodelta-enabled",
                "repeat_index": 1,
                "returncode": 0,
                "loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                "cache_summary": _cache_summary(),
            },
        ],
    }


def _trace_evidence(model_name: str = "MACE") -> dict[str, object]:
    """Create portable trace evidence produced by the trace checker."""
    return {
        "status": "evaluated",
        "model": model_name,
        "attempts": 4.0,
        "hits": 3.0,
        "hit_rate_percent": MIN_HIT_RATE_PERCENT,
        "miss_breakdown": {
            "miss_disabled": 0.0,
            "miss_no-cache": 1.0,
            "miss_neighbor-list-rebuilt": 0.0,
            "miss_shape-changed": 0.0,
            "miss_index-tensor-shape-changed": 0.0,
            "miss_tag-count-changed": 0.0,
            "miss_tag-order-changed": 0.0,
            "miss_comm-topology-changed": 0.0,
            "miss_comm-list-tag-order-changed": 0.0,
        },
        "timing": {
            "baseline_step_time_seconds": TRACE_BASELINE_SECONDS,
            "metadata_build_time_seconds": TRACE_METADATA_SECONDS,
            "metadata_fraction_percent": MIN_TRACE_METADATA_FRACTION_PERCENT,
            "cache_lookup_overhead_seconds": TRACE_LOOKUP_OVERHEAD_SECONDS,
            "estimated_average_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_worst_case_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_average_speedup": TRACE_ESTIMATED_SPEEDUP,
            "estimated_worst_case_speedup": TRACE_ESTIMATED_SPEEDUP,
        },
        "model_agnostic_requirements": {
            "uses_ordered_graph_node_tags": True,
            "uses_edge_count_shape_guard": True,
            "uses_neighbor_rebuild_guard": True,
            "uses_comm_topology_guard": True,
            "uses_comm_list_tag_order_guard": True,
        },
    }


def _thresholds(
    min_distinct_trace_models: int = (
        isodelta_evidence_bundle.DEFAULT_MIN_DISTINCT_TRACE_MODELS
    ),
) -> object:
    """Return the shared bundle thresholds used by direct tests."""
    return isodelta_evidence_bundle.BundleThresholds(
        max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
        min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
        min_speedup=MIN_SPEEDUP,
        min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
        min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
        min_enabled_cache_hits=MIN_ENABLED_HITS,
        min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
        min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
        min_trace_metadata_fraction_percent=MIN_TRACE_METADATA_FRACTION_PERCENT,
        min_distinct_trace_models=min_distinct_trace_models,
    )


class IsoDeltaEvidenceBundleCheckTest(unittest.TestCase):
    """Check pass/fail behavior for combined publication evidence."""

    def test_validate_bundle_accepts_benchmark_and_trace_evidence(self) -> None:
        """A complete bundle should pass and report the trace model labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            evidence = isodelta_evidence_bundle.validate_bundle(
                benchmark_report=benchmark_path,
                trace_evidence_paths=[trace_path],
                required_models=["MACE"],
                thresholds=_thresholds(),
            )
            expected_benchmark_sha256 = _sha256_file(benchmark_path)
            expected_benchmark_size_bytes = benchmark_path.stat().st_size
            expected_trace_sha256 = _sha256_file(trace_path)

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["bundle_schema_version"],
            EXPECTED_BUNDLE_SCHEMA_VERSION,
        )
        self.assertIn("git_commit", evidence["provenance"])
        self.assertIn("python_executable", evidence["provenance"])
        self.assertIsInstance(evidence["provenance"]["git_dirty"], bool)
        self.assertEqual(evidence["trace_models"], ["MACE"])
        self.assertEqual(evidence["trace_model_count"], 1)
        self.assertIn("benchmark_evidence", evidence)
        self.assertEqual(
            evidence["artifacts"]["benchmark_report"]["sha256"],
            expected_benchmark_sha256,
        )
        self.assertEqual(
            evidence["artifacts"]["benchmark_report"]["size_bytes"],
            expected_benchmark_size_bytes,
        )
        self.assertEqual(
            evidence["artifacts"]["trace_evidence"][0]["sha256"],
            expected_trace_sha256,
        )

    def test_validate_bundle_rejects_missing_required_model(self) -> None:
        """Required model labels should prevent vague portability claims."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "NequIP",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[trace_path],
                    required_models=["NequIP"],
                    thresholds=_thresholds(),
                )

    def test_validate_bundle_matches_trimmed_trace_model(self) -> None:
        """Whitespace around a trace model label should not break required matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence(" MACE ")), encoding="utf-8")

            evidence = isodelta_evidence_bundle.validate_bundle(
                benchmark_report=benchmark_path,
                trace_evidence_paths=[trace_path],
                required_models=["MACE"],
                thresholds=_thresholds(),
            )

        self.assertEqual(evidence["trace_models"], ["MACE"])

    def test_validate_bundle_accepts_distinct_trace_models(self) -> None:
        """Portability evidence should pass when distinct MLIP labels are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            mace_trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            nequip_trace_path = Path(tmpdir) / "nequip_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            mace_trace_path.write_text(
                json.dumps(_trace_evidence("MACE")),
                encoding="utf-8",
            )
            nequip_trace_path.write_text(
                json.dumps(_trace_evidence("NequIP")),
                encoding="utf-8",
            )

            evidence = isodelta_evidence_bundle.validate_bundle(
                benchmark_report=benchmark_path,
                trace_evidence_paths=[mace_trace_path, nequip_trace_path],
                required_models=["MACE", "NequIP"],
                thresholds=_thresholds(
                    min_distinct_trace_models=(
                        MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY
                    ),
                ),
            )

        self.assertEqual(evidence["trace_models"], ["MACE", "NequIP"])
        self.assertEqual(
            evidence["trace_model_count"],
            MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY,
        )

    def test_validate_bundle_rejects_repeated_model_for_distinct_gate(self) -> None:
        """Two files from one MLIP should not prove a multi-model portability claim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            first_trace_path = Path(tmpdir) / "first_mace_trace_evidence.json"
            second_trace_path = Path(tmpdir) / "second_mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            first_trace_path.write_text(
                json.dumps(_trace_evidence("MACE")),
                encoding="utf-8",
            )
            second_trace_path.write_text(
                json.dumps(_trace_evidence("MACE")),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "distinct trace model count",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[first_trace_path, second_trace_path],
                    required_models=["MACE"],
                    thresholds=isodelta_evidence_bundle.BundleThresholds(
                        max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                        min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                        min_speedup=MIN_SPEEDUP,
                        min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                        min_enabled_cache_hits=MIN_ENABLED_HITS,
                        min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                        min_trace_metadata_fraction_percent=(
                            MIN_TRACE_METADATA_FRACTION_PERCENT
                        ),
                        min_trace_count=MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY,
                        min_distinct_trace_models=(
                            MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY
                        ),
                    ),
                )

    def test_validate_bundle_rejects_duplicate_trace_paths(self) -> None:
        """One trace evidence file should not satisfy count gates twice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "duplicate trace evidence paths",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[trace_path, trace_path],
                    required_models=["MACE"],
                    thresholds=isodelta_evidence_bundle.BundleThresholds(
                        max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                        min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                        min_speedup=MIN_SPEEDUP,
                        min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                        min_enabled_cache_hits=MIN_ENABLED_HITS,
                        min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                        min_trace_metadata_fraction_percent=(
                            MIN_TRACE_METADATA_FRACTION_PERCENT
                        ),
                        min_trace_count=2,
                    ),
                )

    def test_validate_bundle_rejects_duplicate_required_models(self) -> None:
        """Required model labels should not be duplicated in bundle evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "duplicate required trace models",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[trace_path],
                    required_models=["MACE", "MACE"],
                    thresholds=_thresholds(),
                )

    def test_validate_bundle_rejects_empty_required_model(self) -> None:
        """Required model labels should be non-empty strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "empty names",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[trace_path],
                    required_models=[" "],
                    thresholds=_thresholds(),
                )

    def test_validate_bundle_rejects_meaningless_thresholds(self) -> None:
        """Bundle gates should reject thresholds that cannot support a paper claim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            trace_path = Path(tmpdir) / "mace_trace_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with self.assertRaisesRegex(
                isodelta_evidence_bundle.EvidenceBundleError,
                "min_trace_count",
            ):
                isodelta_evidence_bundle.validate_bundle(
                    benchmark_report=benchmark_path,
                    trace_evidence_paths=[trace_path],
                    required_models=["MACE"],
                    thresholds=isodelta_evidence_bundle.BundleThresholds(
                        max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                        min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                        min_speedup=MIN_SPEEDUP,
                        min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                        min_enabled_cache_hits=MIN_ENABLED_HITS,
                        min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                        min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                        min_trace_metadata_fraction_percent=(
                            MIN_TRACE_METADATA_FRACTION_PERCENT
                        ),
                        min_trace_count=0,
                    ),
                )

    def test_validate_thresholds_rejects_impossible_cache_gate(self) -> None:
        """Minimum cache hits should not exceed minimum cache attempts."""
        with self.assertRaisesRegex(
            isodelta_evidence_bundle.EvidenceBundleError,
            "min_enabled_cache_hits",
        ):
            isodelta_evidence_bundle.validate_thresholds(
                isodelta_evidence_bundle.BundleThresholds(
                    max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                    min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                    min_speedup=MIN_SPEEDUP,
                    min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                    min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                    min_enabled_cache_hits=MIN_ENABLED_ATTEMPTS + 1,
                    min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                    min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                    min_trace_metadata_fraction_percent=(
                        MIN_TRACE_METADATA_FRACTION_PERCENT
                    ),
                )
            )

    def test_validate_thresholds_rejects_zero_distinct_model_gate(self) -> None:
        """Distinct model gates should require at least one model label."""
        with self.assertRaisesRegex(
            isodelta_evidence_bundle.EvidenceBundleError,
            "min_distinct_trace_models",
        ):
            isodelta_evidence_bundle.validate_thresholds(
                isodelta_evidence_bundle.BundleThresholds(
                    max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                    min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                    min_speedup=MIN_SPEEDUP,
                    min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                    min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                    min_enabled_cache_hits=MIN_ENABLED_HITS,
                    min_trace_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                    min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                    min_trace_metadata_fraction_percent=(
                        MIN_TRACE_METADATA_FRACTION_PERCENT
                    ),
                    min_distinct_trace_models=0,
                )
            )

    def test_validate_thresholds_rejects_out_of_range_percent(self) -> None:
        """Percent thresholds should stay inside the zero-to-hundred range."""
        with self.assertRaisesRegex(
            isodelta_evidence_bundle.EvidenceBundleError,
            "min_trace_hit_rate_percent",
        ):
            isodelta_evidence_bundle.validate_thresholds(
                isodelta_evidence_bundle.BundleThresholds(
                    max_abs_thermo_delta=MAX_ABS_THERMO_DELTA,
                    min_paired_thermo_count=MIN_PAIRED_THERMO_COUNT,
                    min_speedup=MIN_SPEEDUP,
                    min_hit_rate_percent=MIN_HIT_RATE_PERCENT,
                    min_enabled_cache_attempts=MIN_ENABLED_ATTEMPTS,
                    min_enabled_cache_hits=MIN_ENABLED_HITS,
                    min_trace_hit_rate_percent=OUT_OF_RANGE_PERCENT,
                    min_trace_estimated_speedup=MIN_TRACE_ESTIMATED_SPEEDUP,
                    min_trace_metadata_fraction_percent=(
                        MIN_TRACE_METADATA_FRACTION_PERCENT
                    ),
                )
            )

    def test_main_reads_files_and_writes_bundle_evidence(self) -> None:
        """The CLI should persist the combined evidence summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            benchmark_path = tmp_path / "benchmark.json"
            trace_path = tmp_path / "mace_trace_evidence.json"
            output_path = tmp_path / "bundle_evidence.json"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = isodelta_evidence_bundle.main(
                    [
                        "--benchmark-report",
                        str(benchmark_path),
                        "--trace-evidence",
                        str(trace_path),
                        "--require-trace-model",
                        "MACE",
                        "--min-speedup",
                        str(MIN_SPEEDUP),
                        "--min-hit-rate-percent",
                        str(MIN_HIT_RATE_PERCENT),
                        "--min-enabled-cache-attempts",
                        str(MIN_ENABLED_ATTEMPTS),
                        "--min-enabled-cache-hits",
                        str(MIN_ENABLED_HITS),
                        "--min-trace-hit-rate-percent",
                        str(MIN_HIT_RATE_PERCENT),
                        "--min-trace-estimated-speedup",
                        str(MIN_TRACE_ESTIMATED_SPEEDUP),
                        "--min-trace-metadata-fraction-percent",
                        str(MIN_TRACE_METADATA_FRACTION_PERCENT),
                        "--min-distinct-trace-models",
                        str(isodelta_evidence_bundle.DEFAULT_MIN_DISTINCT_TRACE_MODELS),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "passed")
            self.assertEqual(
                written["bundle_schema_version"],
                EXPECTED_BUNDLE_SCHEMA_VERSION,
            )
            self.assertIn("platform", written["provenance"])
            self.assertEqual(written["required_trace_models"], ["MACE"])
            self.assertEqual(
                written["artifacts"]["benchmark_report"]["path"],
                str(benchmark_path),
            )
            self.assertEqual(
                written["artifacts"]["benchmark_report"]["sha256"],
                _sha256_file(benchmark_path),
            )
            self.assertEqual(
                written["artifacts"]["trace_evidence"][0]["path"],
                str(trace_path),
            )
            self.assertEqual(
                written["artifacts"]["trace_evidence"][0]["size_bytes"],
                trace_path.stat().st_size,
            )


if __name__ == "__main__":
    unittest.main()
