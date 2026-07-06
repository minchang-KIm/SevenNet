"""Generate portable IsoDelta-Halo MLIP trace examples and evidence.

The real portability claim should be checked with traces exported from each
target runtime. This helper gives researchers a reproducible, dependency-free
reference bundle for SevenNet-like, MACE-like, NequIP-like, and Allegro-like
halo metadata so new exporters can compare their JSON shape and evidence
fields against a known-good example.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


# The demo is intentionally small but still names every physical assumption, so
# expected speedup numbers can be traced back to the synthetic step timings.
REPO_ROOT_PARENT_DEPTH = 1
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
TRACE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
DEFAULT_OUTPUT_DIR = Path("isodelta_mlip_trace_demo")
DEMO_MODELS = ("SevenNet", "MACE", "NequIP", "Allegro")
DEMO_STRING_TAG_MODELS = ("MACE", "Allegro")
DEMO_STRING_GRAPH_TAG_SUFFIXES = ("a", "b", "c")
DEMO_PHASE_TAG_INDICES = (0, 1)
DEMO_INTEGER_GRAPH_TAGS = (10, 20, 30)
DEMO_INTEGER_SEND_TAGS = (1, 2)
DEMO_INTEGER_RECV_TAGS = (5, 6)
DEMO_STEP_COUNT = 5
DEMO_EDGE_COUNT = 24
DEMO_FIRST_RECV_OFFSET = 100
DEMO_SEND_RANK = 1
DEMO_RECV_RANK = 1
DEMO_STEP_TIME_SECONDS = 0.010
DEMO_METADATA_BUILD_SECONDS = 0.002
DEFAULT_MIN_HIT_RATE_PERCENT = 60.0
DEFAULT_MIN_ESTIMATED_SPEEDUP = 1.05
DEFAULT_MIN_METADATA_FRACTION_PERCENT = 10.0
JSON_INDENT_SPACES = 2
EVIDENCE_FILE_SUFFIX = "_isodelta_trace_evidence.json"
TRACE_FILE_SUFFIX = "_halo_trace.json"
SUMMARY_FILE_NAME = "isodelta_mlip_trace_demo_summary.json"
SUMMARY_REPORT_COMMENT = (
    "IsoDelta-Halo portable MLIP trace demo summary linking generated traces, "
    "validated evidence files, thresholds, and estimated speedups."
)
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def _load_trace_check() -> Any:
    """Load the sibling trace checker without requiring tools/ to be a package."""
    spec = importlib.util.spec_from_file_location(
        "isodelta_demo_trace_check",
        TRACE_CHECK_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace checker: {TRACE_CHECK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trace_check = _load_trace_check()


def _model_tags(model_name: str) -> tuple[list[int | str], list[int | str], list[int | str]]:
    """Return deterministic graph, send, and receive tags for one demo model."""
    if model_name in DEMO_STRING_TAG_MODELS:
        graph_tags = [
            f"{model_name.lower()}-{tag}"
            for tag in DEMO_STRING_GRAPH_TAG_SUFFIXES
        ]
        send_tags = [
            f"{model_name.lower()}-send-{index}"
            for index in DEMO_PHASE_TAG_INDICES
        ]
        recv_tags = [
            f"{model_name.lower()}-recv-{index}"
            for index in DEMO_PHASE_TAG_INDICES
        ]
        return graph_tags, send_tags, recv_tags
    return (
        list(DEMO_INTEGER_GRAPH_TAGS),
        list(DEMO_INTEGER_SEND_TAGS),
        list(DEMO_INTEGER_RECV_TAGS),
    )


def _phase(model_name: str) -> dict[str, Any]:
    """Create one halo phase whose count fields match its tag arrays."""
    _, send_tags, recv_tags = _model_tags(model_name)
    return {
        "send_rank": DEMO_SEND_RANK,
        "recv_rank": DEMO_RECV_RANK,
        "send_count": len(send_tags),
        "recv_count": len(recv_tags),
        "first_recv": DEMO_FIRST_RECV_OFFSET,
        "send_tags": send_tags,
        "recv_tags": recv_tags,
    }


def build_demo_trace(model_name: str) -> dict[str, Any]:
    """Build a stable, model-labeled trace using the portable schema."""
    graph_tags, _, _ = _model_tags(model_name)
    return {
        trace_check.GENERATED_ARTIFACT_COMMENT_KEY: (
            trace_check.TRACE_ARTIFACT_COMMENT
        ),
        "model": model_name,
        "schema_version": trace_check.TRACE_SCHEMA_VERSION,
        "steps": [
            {
                "step": step_index,
                "neighbor_list_rebuilt": False,
                "graph_node_tags": graph_tags,
                "edge_count": DEMO_EDGE_COUNT,
                "comm_phases": [_phase(model_name)],
                "step_time_seconds": DEMO_STEP_TIME_SECONDS,
                "metadata_build_time_seconds": DEMO_METADATA_BUILD_SECONDS,
            }
            for step_index in range(DEMO_STEP_COUNT)
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON so generated demo bundles are diffable."""
    path.write_text(json.dumps(payload, indent=JSON_INDENT_SPACES), encoding="utf-8")


def run_demo(
    output_dir: Path,
    thresholds: Any,
    cache_lookup_overhead_seconds: float,
) -> dict[str, Any]:
    """Generate demo traces, validate them, and return a compact summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_summaries: list[dict[str, Any]] = []
    for model_name in DEMO_MODELS:
        trace = build_demo_trace(model_name)
        evidence = trace_check.validate_trace_evidence(
            trace_check.evaluate_trace(
                trace,
                cache_lookup_overhead_seconds=cache_lookup_overhead_seconds,
            ),
            thresholds,
        )
        trace_path = output_dir / f"{model_name.lower()}{TRACE_FILE_SUFFIX}"
        evidence_path = output_dir / f"{model_name.lower()}{EVIDENCE_FILE_SUFFIX}"
        _write_json(trace_path, trace)
        _write_json(evidence_path, evidence)
        model_summaries.append(
            {
                "model": model_name,
                "trace": str(trace_path),
                "evidence": str(evidence_path),
                "hit_rate_percent": evidence["hit_rate_percent"],
                "estimated_average_speedup": evidence["timing"][
                    "estimated_average_speedup"
                ],
                "estimated_worst_case_speedup": evidence["timing"][
                    "estimated_worst_case_speedup"
                ],
            }
        )

    summary = {
        trace_check.GENERATED_REPORT_COMMENT_KEY: SUMMARY_REPORT_COMMENT,
        "status": "passed",
        "models": model_summaries,
        "thresholds": {
            "min_hit_rate_percent": thresholds.min_hit_rate_percent,
            "min_estimated_speedup": thresholds.min_estimated_speedup,
            "min_metadata_fraction_percent": thresholds.min_metadata_fraction_percent,
        },
    }
    _write_json(output_dir / SUMMARY_FILE_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and build a reusable demo evidence bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive demo traces, evidence, and summary JSON",
    )
    parser.add_argument(
        "--cache-lookup-overhead-seconds",
        type=float,
        default=trace_check.DEFAULT_CACHE_LOOKUP_OVERHEAD_SECONDS,
        help="Per-step cache lookup overhead used by the evidence estimator",
    )
    parser.add_argument(
        "--min-hit-rate-percent",
        type=float,
        default=DEFAULT_MIN_HIT_RATE_PERCENT,
        help="Minimum reusable-step hit rate required for every demo trace",
    )
    parser.add_argument(
        "--min-estimated-speedup",
        type=float,
        default=DEFAULT_MIN_ESTIMATED_SPEEDUP,
        help="Minimum average speedup required for every demo trace",
    )
    parser.add_argument(
        "--min-metadata-fraction-percent",
        type=float,
        default=DEFAULT_MIN_METADATA_FRACTION_PERCENT,
        help="Minimum baseline metadata fraction required for every demo trace",
    )
    args = parser.parse_args(argv)

    thresholds = trace_check.TraceThresholds(
        min_hit_rate_percent=args.min_hit_rate_percent,
        min_estimated_speedup=args.min_estimated_speedup,
        min_metadata_fraction_percent=args.min_metadata_fraction_percent,
    )
    try:
        summary = run_demo(
            args.output_dir,
            thresholds,
            args.cache_lookup_overhead_seconds,
        )
    except trace_check.TraceCheckError as exc:
        print(f"IsoDelta-Halo MLIP trace demo failed: {exc}")
        return EXIT_FAILURE

    print(json.dumps(summary, indent=JSON_INDENT_SPACES))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
