"""Unit tests for the model-agnostic IsoDelta-Halo trace checker.

These tests use synthetic MLIP traces so the generality check stays independent
from SevenNet, MACE, NequIP, Allegro, and any external simulator build.
"""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
SPEC = importlib.util.spec_from_file_location("isodelta_mlip_trace", TRACE_CHECK_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_mlip_trace = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_mlip_trace
SPEC.loader.exec_module(isodelta_mlip_trace)


# Named constants keep synthetic traces readable and avoid hidden benchmark
# assumptions in the expected speedup calculations.
DEFAULT_EDGE_COUNT = 12
DEFAULT_FIRST_RECV = 100
BASELINE_STEP_TIME_SECONDS = 10.0
METADATA_BUILD_SECONDS = 2.0
STABLE_TRACE_STEP_COUNT = 4
EXPECTED_TRACE_SCHEMA_VERSION = "1.0"
EXPECTED_STABLE_HIT_RATE_PERCENT = 75.0
EXPECTED_ZERO_TRACE_COUNT_RESIDUAL = 0.0
MIN_STABLE_TRACE_SPEEDUP = 1.15
MIN_STABLE_TRACE_METADATA_FRACTION_PERCENT = 20.0
EXPECTED_STABLE_AVERAGE_SPEEDUP = 40.0 / 34.0
OUT_OF_RANGE_PERCENT = 101.0
INVALID_SPEEDUP_THRESHOLD = 0.0
FRACTIONAL_TRACE_COUNT = 1.5
INCONSISTENT_TIMING_VALUE = 99.0
FAILED_STATUS = "failed"
WHITESPACE_PADDED_MODEL = " MACE "


def _phase(
    *,
    send_rank: int = 1,
    recv_rank: int = 1,
    send_tags: list[int] | None = None,
    recv_tags: list[int] | None = None,
) -> dict[str, object]:
    """Create one synthetic communication phase with matching count fields."""
    phase_send_tags = [1, 2] if send_tags is None else send_tags
    phase_recv_tags = [5, 6] if recv_tags is None else recv_tags
    return {
        "send_rank": send_rank,
        "recv_rank": recv_rank,
        "send_count": len(phase_send_tags),
        "recv_count": len(phase_recv_tags),
        "first_recv": DEFAULT_FIRST_RECV,
        "send_tags": phase_send_tags,
        "recv_tags": phase_recv_tags,
    }


def _step(
    step_id: int,
    *,
    node_tags: list[int] | None = None,
    edge_count: int = DEFAULT_EDGE_COUNT,
    comm_phases: list[dict[str, object]] | None = None,
    neighbor_rebuilt: bool = False,
) -> dict[str, object]:
    """Create one trace step with timing fields for speedup estimates."""
    return {
        "step": step_id,
        "neighbor_list_rebuilt": neighbor_rebuilt,
        "graph_node_tags": [10, 20, 30] if node_tags is None else node_tags,
        "edge_count": edge_count,
        "comm_phases": [_phase()] if comm_phases is None else comm_phases,
        "step_time_seconds": BASELINE_STEP_TIME_SECONDS,
        "metadata_build_time_seconds": METADATA_BUILD_SECONDS,
    }


def _stable_trace(model_name: str = "MACE") -> dict[str, object]:
    """Create a trace where every step after cache warmup can be reused."""
    return {
        "model": model_name,
        "steps": [_step(index) for index in range(STABLE_TRACE_STEP_COUNT)],
    }


class IsoDeltaMlipTraceCheckTest(unittest.TestCase):
    """Check model-agnostic applicability and evidence gates."""

    def test_validate_trace_accepts_mace_like_stable_halo(self) -> None:
        """A stable MACE-like trace should pass hit-rate and speedup gates."""
        evidence = isodelta_mlip_trace.validate_trace_evidence(
            isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE")),
            isodelta_mlip_trace.TraceThresholds(
                min_hit_rate_percent=EXPECTED_STABLE_HIT_RATE_PERCENT,
                min_estimated_speedup=MIN_STABLE_TRACE_SPEEDUP,
                min_metadata_fraction_percent=MIN_STABLE_TRACE_METADATA_FRACTION_PERCENT,
            ),
        )

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["model"], "MACE")
        self.assertEqual(evidence["hits"], 3.0)
        self.assertIn(
            "miss_index-tensor-shape-changed",
            evidence["miss_breakdown"],
        )
        self.assertEqual(evidence["hit_rate_percent"], EXPECTED_STABLE_HIT_RATE_PERCENT)
        self.assertEqual(
            evidence["trace_count_residual"],
            EXPECTED_ZERO_TRACE_COUNT_RESIDUAL,
        )
        self.assertTrue(
            evidence["model_agnostic_requirements"]["uses_comm_topology_guard"]
        )
        self.assertAlmostEqual(
            evidence["timing"]["estimated_average_speedup"],
            EXPECTED_STABLE_AVERAGE_SPEEDUP,
        )

    def test_validate_trace_rejects_inconsistent_hit_rate(self) -> None:
        """Trace hit rate should match hits divided by attempts."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["hit_rate_percent"] = 99.0

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "hits / attempts",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_invalid_status(self) -> None:
        """Precomputed trace evidence should carry an evaluated or passed status."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["status"] = FAILED_STATUS

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "status",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_empty_model_label(self) -> None:
        """Trace evidence should identify the MLIP model label."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["model"] = " "

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "model",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_normalizes_model_label(self) -> None:
        """Trace evidence model labels should be trimmed before bundle matching."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["model"] = WHITESPACE_PADDED_MODEL

        validated = isodelta_mlip_trace.validate_trace_evidence(
            evidence,
            isodelta_mlip_trace.TraceThresholds(),
        )

        self.assertEqual(validated["model"], "MACE")

    def test_validate_trace_rejects_fractional_attempts(self) -> None:
        """Trace attempts should be whole reuse-decision counts."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["attempts"] = FRACTIONAL_TRACE_COUNT

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "nonnegative integer",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_fractional_hits(self) -> None:
        """Trace hits should be whole reuse-decision counts."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["hits"] = FRACTIONAL_TRACE_COUNT

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "nonnegative integer",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_nonfinite_numeric_values(self) -> None:
        """Trace evidence should not accept NaN or infinity values."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["timing"]["estimated_average_speedup"] = math.inf

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "must be finite",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(min_estimated_speedup=1.0),
            )

    def test_validate_trace_rejects_inconsistent_metadata_fraction(self) -> None:
        """Metadata fraction should match metadata time divided by baseline time."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["timing"]["metadata_fraction_percent"] = INCONSISTENT_TIMING_VALUE

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "metadata / baseline",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_inconsistent_average_speedup(self) -> None:
        """Average speedup should match baseline divided by enabled time."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["timing"]["estimated_average_speedup"] = INCONSISTENT_TIMING_VALUE

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "estimated_average_speedup",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_missing_speedup_basis(self) -> None:
        """Precomputed speedup evidence should include enabled-time basis."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["timing"]["estimated_average_enabled_seconds"] = None

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "estimated_average_enabled_seconds",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_inconsistent_worst_case_speedup(self) -> None:
        """Worst-case speedup should match baseline divided by worst enabled time."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["timing"]["estimated_worst_case_speedup"] = INCONSISTENT_TIMING_VALUE

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "estimated_worst_case_speedup",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_inconsistent_miss_breakdown(self) -> None:
        """Trace miss counters should sum to attempts minus hits."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["miss_breakdown"]["miss_shape-changed"] = 1.0

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "attempts - hits",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_negative_miss_counter(self) -> None:
        """Trace miss counters should be nonnegative counts."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["miss_breakdown"]["miss_shape-changed"] = -1.0

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "nonnegative integer",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_fractional_miss_counter(self) -> None:
        """Trace miss counters should be whole reuse-decision counts."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["miss_breakdown"]["miss_shape-changed"] = FRACTIONAL_TRACE_COUNT

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "nonnegative integer",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_missing_model_agnostic_guards(self) -> None:
        """Precomputed trace evidence should prove every required reuse guard."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        del evidence["model_agnostic_requirements"]["uses_comm_topology_guard"]

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "uses_comm_topology_guard",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_validate_trace_rejects_disabled_model_agnostic_guard(self) -> None:
        """Required reuse guard flags should be explicit true booleans."""
        evidence = isodelta_mlip_trace.evaluate_trace(_stable_trace("MACE"))
        evidence["model_agnostic_requirements"]["uses_comm_topology_guard"] = False

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "must be true",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(),
            )

    def test_trace_rejects_unstable_tag_order_for_effect_gate(self) -> None:
        """Atom tag reordering should be a miss and fail a reuse-rate gate."""
        trace = {
            "model": "NequIP",
            "steps": [
                _step(0),
                _step(1, node_tags=[10, 30, 20]),
            ],
        }
        evidence = isodelta_mlip_trace.evaluate_trace(trace)
        self.assertEqual(
            evidence["miss_breakdown"]["miss_tag-order-changed"],
            1.0,
        )
        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "hit rate",
        ):
            isodelta_mlip_trace.validate_trace_evidence(
                evidence,
                isodelta_mlip_trace.TraceThresholds(min_hit_rate_percent=1.0),
            )

    def test_trace_counts_topology_and_comm_list_order_changes(self) -> None:
        """Topology and phase-local tag changes should be diagnosed separately."""
        trace = {
            "model": "Allegro",
            "steps": [
                _step(0),
                _step(1, comm_phases=[_phase(send_rank=2)]),
                _step(2, comm_phases=[_phase(send_rank=2, send_tags=[2, 1])]),
            ],
        }
        evidence = isodelta_mlip_trace.evaluate_trace(trace)
        self.assertEqual(
            evidence["miss_breakdown"]["miss_comm-topology-changed"],
            1.0,
        )
        self.assertEqual(
            evidence["miss_breakdown"]["miss_comm-list-tag-order-changed"],
            1.0,
        )

    def test_cache_disabled_reports_disabled_misses(self) -> None:
        """Disabled-cache traces should use the same baseline miss reason."""
        evidence = isodelta_mlip_trace.evaluate_trace(
            _stable_trace("SevenNet"),
            cache_disabled=True,
        )
        self.assertEqual(evidence["hits"], 0.0)
        self.assertEqual(
            evidence["miss_breakdown"]["miss_disabled"],
            float(STABLE_TRACE_STEP_COUNT),
        )

    def test_evaluate_trace_rejects_empty_model_label(self) -> None:
        """Trace exporters should not emit an empty model label."""
        trace = _stable_trace(" ")

        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "model",
        ):
            isodelta_mlip_trace.evaluate_trace(trace)

    def test_trace_schema_documents_portable_export_fields(self) -> None:
        """The checker should expose a schema for non-SevenNet trace exporters."""
        schema = isodelta_mlip_trace.trace_schema()
        self.assertEqual(schema["schema_version"], EXPECTED_TRACE_SCHEMA_VERSION)
        self.assertIn("graph_node_tags", schema["required_step_fields"])
        self.assertIn("comm_phases", schema["required_step_fields"])
        self.assertIn("send_tags", schema["required_comm_phase_fields"])
        self.assertIn("recv_tags", schema["required_comm_phase_fields"])
        self.assertIn("estimated_worst_case_speedup", schema["timing_estimates"])

    def test_validate_thresholds_rejects_out_of_range_percent(self) -> None:
        """Trace gates should reject impossible percentage thresholds."""
        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "min_hit_rate_percent",
        ):
            isodelta_mlip_trace.validate_thresholds(
                isodelta_mlip_trace.TraceThresholds(
                    min_hit_rate_percent=OUT_OF_RANGE_PERCENT,
                )
            )

    def test_validate_thresholds_rejects_nonpositive_speedup(self) -> None:
        """Trace gates should reject nonpositive speedup thresholds."""
        with self.assertRaisesRegex(
            isodelta_mlip_trace.TraceCheckError,
            "min_estimated_speedup",
        ):
            isodelta_mlip_trace.validate_thresholds(
                isodelta_mlip_trace.TraceThresholds(
                    min_estimated_speedup=INVALID_SPEEDUP_THRESHOLD,
                )
            )

    def test_main_prints_schema_without_trace(self) -> None:
        """The CLI should let researchers inspect the trace format first."""
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = isodelta_mlip_trace.main(["--print-schema"])

        self.assertEqual(exit_code, 0)
        schema = json.loads(stdout.getvalue())
        self.assertEqual(schema["schema_version"], EXPECTED_TRACE_SCHEMA_VERSION)
        self.assertIn("required_comm_phase_fields", schema)

    def test_main_reads_json_trace_and_writes_evidence(self) -> None:
        """The CLI should validate a portable trace and persist its evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            output_path = Path(tmpdir) / "evidence.json"
            trace_path.write_text(json.dumps(_stable_trace("MACE")), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = isodelta_mlip_trace.main(
                    [
                        "--trace",
                        str(trace_path),
                        "--min-hit-rate-percent",
                        str(EXPECTED_STABLE_HIT_RATE_PERCENT),
                        "--min-estimated-speedup",
                        str(MIN_STABLE_TRACE_SPEEDUP),
                        "--min-metadata-fraction-percent",
                        str(MIN_STABLE_TRACE_METADATA_FRACTION_PERCENT),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written["model"], "MACE")
            self.assertEqual(written["status"], "passed")


if __name__ == "__main__":
    unittest.main()
