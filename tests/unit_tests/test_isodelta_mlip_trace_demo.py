"""Unit tests for the portable IsoDelta-Halo MLIP trace demo runner.

The demo runner gives researchers known-good SevenNet, MACE, NequIP, and
Allegro trace bundles without requiring those runtimes in the test environment.
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


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT_PARENT_DEPTH = 2
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
DEMO_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_mlip_trace_demo.py"
SPEC = importlib.util.spec_from_file_location("isodelta_mlip_trace_demo", DEMO_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_mlip_trace_demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_mlip_trace_demo
SPEC.loader.exec_module(isodelta_mlip_trace_demo)


EXPECTED_MODEL_COUNT = 4
EXPECTED_HIT_RATE_PERCENT = 80.0
EXPECTED_TRACE_ARTIFACT_COMMENT = (
    isodelta_mlip_trace_demo.trace_check.TRACE_ARTIFACT_COMMENT
)
EXPECTED_TRACE_EVIDENCE_REPORT_COMMENT = (
    isodelta_mlip_trace_demo.trace_check.TRACE_EVIDENCE_REPORT_COMMENT
)
EXPECTED_SUMMARY_REPORT_COMMENT = isodelta_mlip_trace_demo.SUMMARY_REPORT_COMMENT
MIN_EXPECTED_SPEEDUP = 1.05
MIN_METADATA_FRACTION_PERCENT = 10.0
ZERO_CACHE_LOOKUP_OVERHEAD_SECONDS = 0.0
DEFAULT_OUTPUT_FILE_COUNT = 9
FIRST_DEMO_STEP_INDEX = 0
FIRST_DEMO_PHASE_INDEX = 0
FIRST_GRAPH_TAG_INDEX = 0


class IsoDeltaMlipTraceDemoTest(unittest.TestCase):
    """Check that the demo runner produces portable multi-model evidence."""

    def test_run_demo_writes_trace_and_evidence_for_each_model(self) -> None:
        """Every configured model should get a passing trace evidence file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary = isodelta_mlip_trace_demo.run_demo(
                output_dir,
                isodelta_mlip_trace_demo.trace_check.TraceThresholds(
                    min_hit_rate_percent=EXPECTED_HIT_RATE_PERCENT,
                    min_estimated_speedup=MIN_EXPECTED_SPEEDUP,
                    min_metadata_fraction_percent=MIN_METADATA_FRACTION_PERCENT,
                ),
                cache_lookup_overhead_seconds=ZERO_CACHE_LOOKUP_OVERHEAD_SECONDS,
            )

            self.assertEqual(summary["status"], "passed")
            self.assertEqual(len(summary["models"]), EXPECTED_MODEL_COUNT)
            self.assertEqual(len(list(output_dir.iterdir())), DEFAULT_OUTPUT_FILE_COUNT)
            for model_entry in summary["models"]:
                trace_path = Path(model_entry["trace"])
                evidence_path = Path(model_entry["evidence"])
                self.assertTrue(trace_path.exists())
                self.assertTrue(evidence_path.exists())
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    trace["artifact_comment"],
                    EXPECTED_TRACE_ARTIFACT_COMMENT,
                )
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                self.assertEqual(evidence["status"], "passed")
                self.assertEqual(
                    evidence["report_comment"],
                    EXPECTED_TRACE_EVIDENCE_REPORT_COMMENT,
                )
                self.assertEqual(
                    evidence["hit_rate_percent"],
                    EXPECTED_HIT_RATE_PERCENT,
                )

    def test_build_demo_trace_uses_string_tags_for_mace_like_models(self) -> None:
        """The demo should exercise portable non-integer tag handling."""
        trace = isodelta_mlip_trace_demo.build_demo_trace("MACE")
        self.assertEqual(trace["artifact_comment"], EXPECTED_TRACE_ARTIFACT_COMMENT)
        first_step = trace["steps"][FIRST_DEMO_STEP_INDEX]
        self.assertIsInstance(first_step["graph_node_tags"][FIRST_GRAPH_TAG_INDEX], str)
        first_phase = first_step["comm_phases"][FIRST_DEMO_PHASE_INDEX]
        self.assertEqual(first_phase["send_count"], len(first_phase["send_tags"]))
        self.assertEqual(first_phase["recv_count"], len(first_phase["recv_tags"]))

    def test_main_prints_summary_and_writes_bundle(self) -> None:
        """The CLI should generate a complete demo bundle in the output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = isodelta_mlip_trace_demo.main(
                    [
                        "--output-dir",
                        tmpdir,
                        "--min-hit-rate-percent",
                        str(EXPECTED_HIT_RATE_PERCENT),
                        "--min-estimated-speedup",
                        str(MIN_EXPECTED_SPEEDUP),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["report_comment"], EXPECTED_SUMMARY_REPORT_COMMENT)
            summary_path = Path(tmpdir) / isodelta_mlip_trace_demo.SUMMARY_FILE_NAME
            self.assertTrue(summary_path.exists())
            persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted_summary["report_comment"],
                EXPECTED_SUMMARY_REPORT_COMMENT,
            )


if __name__ == "__main__":
    unittest.main()
