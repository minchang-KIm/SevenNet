"""Unit tests for the IsoDelta-Halo paper evidence bundle checker.

The bundle checker combines SevenNet benchmark evidence with portable MLIP trace
evidence. These tests keep that publication gate reproducible without requiring
a LAMMPS binary or an external model runtime.
"""

from __future__ import annotations

import contextlib
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
RUN_TIMEOUT_SECONDS = 3600.0
BASELINE_LOOP_TIME_SECONDS = 12.0
ISODELTA_LOOP_TIME_SECONDS = 10.0
ZERO_SAMPLE_VARIANCE_LOOP_TIME_SECONDS = 0.0
ZERO_SAMPLE_STDDEV_LOOP_TIME_SECONDS = 0.0
EXPECTED_RESULT_COUNT = 4


def _cache_summary(
    attempts: float = MIN_ENABLED_ATTEMPTS,
    hits: float = MIN_ENABLED_HITS,
    hit_rate_percent: float | None = None,
) -> dict[str, float]:
    """Create a complete IsoDelta-Halo cache summary for one run."""
    resolved_hit_rate_percent = (
        PERCENT_SCALE * hits / attempts
        if hit_rate_percent is None
        else hit_rate_percent
    )
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": resolved_hit_rate_percent,
        "miss_disabled": 0.0,
        "miss_no-cache": attempts - hits,
        "miss_neighbor-list-rebuilt": 0.0,
        "miss_shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _benchmark_report() -> dict[str, object]:
    """Create a benchmark report that proves correctness and speedup."""
    return {
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
                "cache_summary": _cache_summary(hits=0.0, hit_rate_percent=0.0),
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
                "cache_summary": _cache_summary(hits=0.0, hit_rate_percent=0.0),
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
            "miss_tag-count-changed": 0.0,
            "miss_tag-order-changed": 0.0,
            "miss_comm-topology-changed": 0.0,
            "miss_comm-list-tag-order-changed": 0.0,
        },
        "timing": {
            "metadata_fraction_percent": MIN_TRACE_METADATA_FRACTION_PERCENT,
            "estimated_average_speedup": 1.18,
            "estimated_worst_case_speedup": 1.18,
        },
        "model_agnostic_requirements": {
            "uses_ordered_graph_node_tags": True,
            "uses_edge_count_shape_guard": True,
            "uses_neighbor_rebuild_guard": True,
            "uses_comm_topology_guard": True,
            "uses_comm_list_tag_order_guard": True,
        },
    }


def _thresholds() -> object:
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

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["trace_models"], ["MACE"])
        self.assertEqual(evidence["trace_model_count"], 1)
        self.assertIn("benchmark_evidence", evidence)

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
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "passed")
            self.assertEqual(written["required_trace_models"], ["MACE"])


if __name__ == "__main__":
    unittest.main()
