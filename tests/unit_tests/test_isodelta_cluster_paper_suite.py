"""Unit tests for the IsoDelta-Halo cluster paper suite runner.

The cluster runner must remain testable without a real 8-GPU node, so these
tests use synthetic but fully validated benchmark and trace evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_SUITE_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_cluster_paper_suite.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_cluster_paper_suite",
    CLUSTER_SUITE_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
isodelta_cluster_suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_cluster_suite
SPEC.loader.exec_module(isodelta_cluster_suite)


PERCENT_SCALE = 100.0
RUN_TIMEOUT_SECONDS = 3600.0
BASELINE_LOOP_TIME_SECONDS = 12.0
ISODELTA_LOOP_TIME_SECONDS = 10.0
EXPECTED_SPEEDUP = BASELINE_LOOP_TIME_SECONDS / ISODELTA_LOOP_TIME_SECONDS
EXPECTED_RESULT_COUNT = 4
ENABLED_ATTEMPTS = 10.0
ENABLED_HITS = 8.0
DISABLED_ATTEMPTS = 10.0
TRACE_ATTEMPTS = 4.0
TRACE_HITS = 3.0
TRACE_HIT_RATE_PERCENT = PERCENT_SCALE * TRACE_HITS / TRACE_ATTEMPTS
TRACE_BASELINE_SECONDS = 100.0
TRACE_METADATA_SECONDS = 20.0
TRACE_ENABLED_SECONDS = 80.0
TRACE_SPEEDUP = TRACE_BASELINE_SECONDS / TRACE_ENABLED_SECONDS
PRINT_INFO_ENV = "SEVENN_PRINT_INFO"
DISABLE_CACHE_ENV = "SEVENN_ISODELTA_HALO_DISABLE"
PROFILE_CACHE_ENV = "SEVENN_ISODELTA_HALO_PROFILE"
ENV_FLAG_ENABLED = "1"


def _cache_summary(
    *,
    attempts: float = ENABLED_ATTEMPTS,
    hits: float = ENABLED_HITS,
    miss_disabled: float = 0.0,
    miss_no_cache: float = ENABLED_ATTEMPTS - ENABLED_HITS,
) -> dict[str, float]:
    """Create a complete cache summary with all miss counters."""
    return {
        "attempts": attempts,
        "hits": hits,
        "hit_rate_percent": PERCENT_SCALE * hits / attempts,
        "summary_rank_count": 1.0,
        "miss_disabled": miss_disabled,
        "miss_no-cache": miss_no_cache,
        "miss_neighbor-list-rebuilt": 0.0,
        "miss_shape-changed": 0.0,
        "miss_index-tensor-shape-changed": 0.0,
        "miss_tag-count-changed": 0.0,
        "miss_tag-order-changed": 0.0,
        "miss_comm-topology-changed": 0.0,
        "miss_comm-list-tag-order-changed": 0.0,
    }


def _disabled_cache_summary() -> dict[str, float]:
    """Create the disabled-cache baseline summary expected by the checker."""
    return _cache_summary(
        attempts=DISABLED_ATTEMPTS,
        hits=0.0,
        miss_disabled=DISABLED_ATTEMPTS,
        miss_no_cache=0.0,
    )


def _benchmark_report() -> dict[str, object]:
    """Create a small passing paired benchmark report."""
    return {
        "provenance": {
            "report_schema_version": "isodelta-benchmark-report-v1",
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
            "speedup_vs_disabled_cache": EXPECTED_SPEEDUP,
            "cases": {
                "baseline-disabled": {
                    "mean_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": 0.0,
                    "sample_stddev_loop_time_seconds": 0.0,
                    "min_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": BASELINE_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
                "isodelta-enabled": {
                    "mean_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "sample_variance_loop_time_seconds": 0.0,
                    "sample_stddev_loop_time_seconds": 0.0,
                    "min_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "max_loop_time_seconds": ISODELTA_LOOP_TIME_SECONDS,
                    "valid_loop_time_count": 2,
                },
            },
            "final_thermo_delta_vs_disabled_cache": {
                "PotEng": {"max_abs_delta": 1.0e-9, "paired_count": 2.0},
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


def _trace_evidence(model_name: str) -> dict[str, object]:
    """Create portable trace evidence accepted by the trace checker."""
    return {
        "status": "passed",
        "model": model_name,
        "attempts": TRACE_ATTEMPTS,
        "hits": TRACE_HITS,
        "hit_rate_percent": TRACE_HIT_RATE_PERCENT,
        "miss_breakdown": {
            "miss_disabled": 0.0,
            "miss_no-cache": TRACE_ATTEMPTS - TRACE_HITS,
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
            "metadata_fraction_percent": (
                PERCENT_SCALE * TRACE_METADATA_SECONDS / TRACE_BASELINE_SECONDS
            ),
            "cache_lookup_overhead_seconds": 0.0,
            "estimated_average_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_worst_case_enabled_seconds": TRACE_ENABLED_SECONDS,
            "estimated_average_speedup": TRACE_SPEEDUP,
            "estimated_worst_case_speedup": TRACE_SPEEDUP,
        },
        "model_agnostic_requirements": {
            "uses_ordered_graph_node_tags": True,
            "uses_edge_count_shape_guard": True,
            "uses_neighbor_rebuild_guard": True,
            "uses_comm_topology_guard": True,
            "uses_comm_list_tag_order_guard": True,
        },
    }


def _external_timing_report(model_name: str) -> dict[str, object]:
    """Create an external-pair timing report for a non-SevenNet runtime."""
    baseline_times = [BASELINE_LOOP_TIME_SECONDS - 1.0, BASELINE_LOOP_TIME_SECONDS + 1.0]
    enabled_times = [ISODELTA_LOOP_TIME_SECONDS - 1.0, ISODELTA_LOOP_TIME_SECONDS + 1.0]
    sample_variance = 2.0
    sample_stddev = sample_variance ** 0.5
    return {
        "schema_version": "isodelta-external-pair-timing-v1",
        "case_name": f"{model_name.lower()}-existing",
        "model": model_name,
        "repeat_count": 2,
        "disabled_success_count": 2,
        "enabled_success_count": 2,
        "baseline_times_seconds": baseline_times,
        "enabled_times_seconds": enabled_times,
        "baseline_mean_seconds": BASELINE_LOOP_TIME_SECONDS,
        "enabled_mean_seconds": ISODELTA_LOOP_TIME_SECONDS,
        "baseline_sample_variance_seconds": sample_variance,
        "enabled_sample_variance_seconds": sample_variance,
        "baseline_sample_stddev_seconds": sample_stddev,
        "enabled_sample_stddev_seconds": sample_stddev,
        "speedup_vs_disabled_cache": EXPECTED_SPEEDUP,
        "commands": [],
    }


class IsoDeltaClusterPaperSuiteTest(unittest.TestCase):
    """Check manifest validation and paper artifact generation."""

    def test_write_template_creates_commented_three_model_manifest(self) -> None:
        """The template should be editable and include the required models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "suite.toml"
            isodelta_cluster_suite.write_template(template_path)
            template = template_path.read_text(encoding="utf-8")

        self.assertTrue(template.lstrip().startswith("#"))
        self.assertIn('required_models = ["SevenNet", "MACE", "NequIP"]', template)
        self.assertIn('kind = "sevennet_lammps"', template)
        self.assertIn('kind = "external_pair"', template)

    def test_manifest_validation_requires_all_default_foundation_models(self) -> None:
        """A paper manifest should not silently omit MACE or NequIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "missing-models"

[[cases]]
name = "sevennet-only"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "missing required model cases",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_rejects_unknown_artifact_required_by(self) -> None:
        """Artifact model references should not hide spelling mistakes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "bad-artifact-reference"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "data.ext"
required_by = ["TypoModel"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "required_by names unknown case models",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_download_artifact_copies_file_url_and_checks_sha256(self) -> None:
        """Artifact downloads should verify immutable paper inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.bin"
            destination_path = Path(tmpdir) / "downloaded.bin"
            source_path.write_bytes(b"isodelta artifact")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            artifact = isodelta_cluster_suite.ArtifactConfig(
                name="local-artifact",
                path=destination_path,
                url=source_path.as_uri(),
                sha256=digest,
            )

            record = isodelta_cluster_suite.download_artifact(artifact)

        self.assertTrue(record["downloaded"])
        self.assertEqual(record["sha256"], digest)

    def test_download_artifact_skips_optional_missing_without_url(self) -> None:
        """Optional artifacts may be absent but should be recorded explicitly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = isodelta_cluster_suite.ArtifactConfig(
                name="optional-note",
                path=Path(tmpdir) / "optional.txt",
                required=False,
            )

            record = isodelta_cluster_suite.download_artifact(artifact)

        self.assertFalse(record["downloaded"])
        self.assertTrue(record["skipped_optional_missing"])

    def test_skip_downloads_rejects_missing_required_artifact(self) -> None:
        """A cluster run should not start when required inputs are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "missing-required-artifact"
required_models = ["SevenNet"]

[[artifacts]]
name = "required-dataset"
path = "{(root / "missing_dataset.bin").as_posix()}"
required = true

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
artifacts = ["required-dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "required artifacts are missing",
            ):
                isodelta_cluster_suite.run_suite(
                    config,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )

    def test_plan_only_writes_preflight_manifest_audit(self) -> None:
        """Researchers should inspect planned commands before using GPU time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"cluster input")
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            plan_path = root / "planned_run.json"
            manifest_path.write_text(
                f"""
[suite]
name = "plan-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "{(root / "missing_dataset.bin").as_posix()}"
url = "{source_path.as_uri()}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet-plan"
model = "SevenNet"
kind = "trace_only"
trace_input = "trace.json"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--plan-only",
                    "--plan-output",
                    str(plan_path),
                    "--skip-gpu-check",
                ]
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        self.assertEqual(exit_code, 0)
        self.assertFalse(plan["gpu_check_planned"])
        self.assertTrue(plan["artifacts"][0]["will_download"])
        self.assertTrue(plan["artifacts"][0]["missing_required"])
        self.assertEqual(
            plan["suite"]["manifest"]["sha256"],
            manifest_digest,
        )
        self.assertEqual(plan["cases"][0]["model"], "SevenNet")
        self.assertIn("trace_evidence", plan["cases"][0]["expected_outputs"])
        self.assertIn("speedup_by_case.svg", plan["paper_outputs"]["speedup_svg"])

    def test_external_timing_report_rejects_inconsistent_speedup(self) -> None:
        """External MACE/NequIP timing rows should be internally auditable."""
        report = _external_timing_report("NequIP")
        report["speedup_vs_disabled_cache"] = EXPECTED_SPEEDUP + 0.5
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "must match baseline / enabled seconds",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_external_timing_report_rejects_inconsistent_raw_mean(self) -> None:
        """External timing reports should keep raw samples and means aligned."""
        report = _external_timing_report("NequIP")
        report["baseline_mean_seconds"] = BASELINE_LOOP_TIME_SECONDS + 0.25
        case = isodelta_cluster_suite.CaseConfig(
            name="nequip-existing",
            model="NequIP",
            kind="external_pair",
            disabled_command="run baseline",
            enabled_command="run enabled",
            repeat_count=2,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "baseline_mean_seconds must match raw timing samples",
        ):
            isodelta_cluster_suite.validate_external_timing_report(report, case)

    def test_collect_only_writes_tables_correlations_and_svg_figures(self) -> None:
        """Existing evidence should become paper tables, correlations, and graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "sevennet_benchmark.json"
            sevennet_trace_path = root / "sevennet_trace.json"
            mace_trace_path = root / "mace_trace.json"
            nequip_trace_path = root / "nequip_trace.json"
            nequip_timing_path = root / "nequip_timing.json"
            output_dir = root / "paper_outputs"
            benchmark_path.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            mace_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            nequip_trace_path.write_text(json.dumps(_trace_evidence("NequIP")), encoding="utf-8")
            nequip_timing_path.write_text(json.dumps(_external_timing_report("NequIP")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "collect-only-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
min_speedup = 1.1
min_hit_rate_percent = 75.0
min_trace_hit_rate_percent = 50.0
required_models = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "sevennet_lammps"
lammps_command = "lmp"
input = "in.sevennet"
benchmark_report = "{benchmark_path.as_posix()}"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]
min_enabled_cache_attempts = 10
min_enabled_cache_hits = 8

[[cases]]
name = "mace-existing"
model = "MACE"
kind = "trace_only"
trace_evidence = ["{mace_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "external_pair"
disabled_command = "python -c print('baseline')"
enabled_command = "python -c print('enabled')"
repeat_count = 2
external_timing_report = "{nequip_timing_path.as_posix()}"
trace_evidence = ["{nequip_trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                collect_only=True,
                skip_downloads=True,
                skip_gpu_check=True,
            )

            summary_path = output_dir / "isodelta_cluster_paper_summary.json"
            case_summary_csv = output_dir / "tables" / "case_summary.csv"
            correlation_csv = output_dir / "tables" / "correlation.csv"
            speedup_svg = output_dir / "figures" / "speedup_by_case.svg"
            manifest_snapshot = output_dir / "isodelta_cluster_suite_manifest.toml"

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            case_summary_exists = case_summary_csv.exists()
            correlation_exists = correlation_csv.exists()
            speedup_svg_exists = speedup_svg.exists()
            manifest_snapshot_exists = manifest_snapshot.exists()
            case_summary_text = case_summary_csv.read_text(encoding="utf-8")
            speedup_svg_text = speedup_svg.read_text(encoding="utf-8")
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        self.assertEqual(exit_code, 0)
        self.assertTrue(case_summary_exists)
        self.assertTrue(correlation_exists)
        self.assertTrue(speedup_svg_exists)
        self.assertTrue(manifest_snapshot_exists)
        self.assertEqual(
            summary["suite"]["manifest"]["sha256"],
            manifest_digest,
        )
        self.assertEqual(len(summary["cases"]), 3)
        self.assertEqual(summary["suite_evidence"]["distinct_trace_model_count"], 3)
        self.assertIn("speedup_vs_disabled_cache", case_summary_text)
        self.assertIn("baseline_sample_variance_seconds", case_summary_text)
        self.assertIn("enabled_sample_stddev_seconds", case_summary_text)
        self.assertIn("<svg", speedup_svg_text)

    def test_collect_only_rejects_insufficient_distinct_trace_models(self) -> None:
        """Suite-level gates should count model labels inside trace evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sevennet_trace_path = root / "sevennet_trace.json"
            mace_trace_path = root / "mace_trace.json"
            mislabeled_nequip_trace_path = root / "nequip_trace.json"
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            mace_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            mislabeled_nequip_trace_path.write_text(json.dumps(_trace_evidence("MACE")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "weak-distinct-traces"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
min_trace_count = 3
min_distinct_trace_models = 3

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]

[[cases]]
name = "mace-existing"
model = "MACE"
kind = "trace_only"
trace_evidence = ["{mace_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "trace_only"
trace_evidence = ["{mislabeled_nequip_trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "distinct trace model count",
            ):
                isodelta_cluster_suite.run_suite(
                    config,
                    collect_only=True,
                    skip_downloads=True,
                    skip_gpu_check=True,
                )


if __name__ == "__main__":
    unittest.main()
