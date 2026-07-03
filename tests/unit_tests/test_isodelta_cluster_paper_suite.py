"""Unit tests for the IsoDelta-Halo cluster paper suite runner.

The cluster runner must remain testable without a real 8-GPU node, so these
tests use synthetic but fully validated benchmark and trace evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
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
PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE = 7
PIPELINE_MIN_SPEEDUP = 1.05
PIPELINE_MIN_SPEEDUP_LOWER_BOUND = 1.0
PAPER_REPEAT_COUNT = isodelta_cluster_suite.FINAL_PAPER_MIN_REPEAT_COUNT
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

    def test_verify_output_bundle_accepts_relocated_bundle_paths(self) -> None:
        """Bundle verification should survive archiving to a different directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_output_dir = root / "paper_outputs"
            logs_dir = original_output_dir / "logs"
            tables_dir = original_output_dir / "tables"
            logs_dir.mkdir(parents=True)
            tables_dir.mkdir(parents=True)
            table_path = tables_dir / "case_summary.csv"
            stdout_path = logs_dir / "case.stdout"
            missing_stderr_path = logs_dir / "case.stderr"
            table_path.write_text("case,speedup\nsevennet,1.2\n", encoding="utf-8")
            stdout_path.write_text("completed\n", encoding="utf-8")
            summary_path = original_output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary_path.write_text(
                json.dumps(
                    {
                        "suite": {"output_dir": str(original_output_dir)},
                        "artifact_fingerprints": {
                            "case_summary_csv": isodelta_cluster_suite.generated_artifact_record(
                                table_path
                            )
                        },
                        "command_log_fingerprints": [
                            {
                                "name": "case",
                                "stdout": isodelta_cluster_suite.optional_file_fingerprint(
                                    stdout_path
                                ),
                                "stderr": isodelta_cluster_suite.optional_file_fingerprint(
                                    missing_stderr_path
                                ),
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            relocated_output_dir = root / "archived_outputs"
            shutil.copytree(original_output_dir, relocated_output_dir)
            shutil.rmtree(original_output_dir)
            verification = isodelta_cluster_suite.verify_output_bundle(relocated_output_dir)

        self.assertEqual(verification["status"], "passed")
        self.assertEqual(verification["verified_artifact_count"], 1)
        self.assertEqual(verification["verified_command_log_count"], 2)

    def test_write_template_creates_commented_three_model_manifest(self) -> None:
        """The template should be editable and include the required models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "suite.toml"
            isodelta_cluster_suite.write_template(template_path)
            template = template_path.read_text(encoding="utf-8")

        self.assertTrue(template.lstrip().startswith("#"))
        self.assertIn('required_models = ["SevenNet", "MACE", "NequIP"]', template)
        self.assertIn("require_artifact_sha256 = true", template)
        self.assertIn('kind = "sevennet_lammps"', template)
        self.assertIn('kind = "external_pair"', template)
        self.assertIn('preflight_command = \'python -c "import mace"\'', template)

    def test_write_slurm_script_creates_commented_preflight_first_launcher(self) -> None:
        """The SLURM wrapper should submit reproducible preflight evidence first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            output_dir = root / "paper outputs"
            slurm_path = root / "run_isodelta.sbatch"
            manifest_path.write_text(
                f"""
[suite]
name = "slurm-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["trace.json"]
""",
                encoding="utf-8",
            )
            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--write-slurm-script",
                    str(slurm_path),
                    "--skip-downloads",
                    "--keep-going",
                    "--reuse-passed",
                    "--slurm-job-name",
                    "paper suite",
                    "--slurm-cpus-per-task",
                    "12",
                ]
            )
            script = slurm_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertTrue(script.startswith("#!/usr/bin/env bash"))
        self.assertIn("# IsoDelta-Halo cluster paper suite launcher.", script)
        self.assertIn("#SBATCH --job-name=paper_suite", script)
        self.assertIn("#SBATCH --gres=gpu:8", script)
        self.assertIn("#SBATCH --cpus-per-task=12", script)
        self.assertIn("COMMON_ARGS=(--manifest \"$MANIFEST_PATH\")", script)
        self.assertIn("PREFLIGHT_OUTPUT=", script)
        self.assertIn("--preflight-only --preflight-output \"$PREFLIGHT_OUTPUT\"", script)
        self.assertIn("--plan-only --plan-output \"$PLAN_OUTPUT\"", script)
        self.assertIn("COMMON_ARGS+=(--skip-downloads)", script)
        self.assertIn("COMMON_ARGS+=(--keep-going)", script)
        self.assertIn("COMMON_ARGS+=(--reuse-passed)", script)
        self.assertIn("# Run SevenNet, MACE, NequIP, and any extra manifest cases.", script)

    def test_reuse_passed_skips_existing_valid_case_outputs(self) -> None:
        """Validated outputs should be reusable after an interrupted cluster run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_path = root / "sevennet_trace.json"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            manifest_path.write_text(
                f"""
[suite]
name = "reuse-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-existing"
model = "SevenNet"
kind = "external_pair"
disabled_command = "should-not-run-disabled"
enabled_command = "should-not-run-enabled"
repeat_count = 2
trace_evidence = ["{trace_path.as_posix()}"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            timing_report_path = isodelta_cluster_suite._external_timing_report_path(
                config,
                config.cases[0],
            )
            timing_report_path.parent.mkdir(parents=True, exist_ok=True)
            timing_report_path.write_text(
                json.dumps(_external_timing_report("SevenNet")),
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.run_suite(
                config,
                skip_downloads=True,
                skip_gpu_check=True,
                reuse_passed=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["cases"][0]["status"], "reused")
        self.assertEqual(summary["suite_evidence"]["passed_models"], ["SevenNet"])
        self.assertEqual(summary["commands"], [])

    def test_preflight_failure_skips_expensive_case_commands(self) -> None:
        """A failed case preflight should stop the paired timing loop early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace_path = root / "sevennet_trace.json"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            python_bin = Path(sys.executable).as_posix()
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-pass"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{trace_path.as_posix()}"]

[[cases]]
name = "mace-preflight"
model = "MACE"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(3)"'
disabled_command = "should-not-run-disabled"
enabled_command = "should-not-run-enabled"
repeat_count = 1
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                skip_downloads=True,
                skip_gpu_check=True,
                keep_going=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            command_names = [record["name"] for record in summary["commands"]]
            preflight_log = (
                output_dir
                / "cases"
                / "mace-preflight"
                / "logs"
                / "preflight.stderr.log"
            )
            preflight_log_exists = preflight_log.exists()
            preflight_log_digest = hashlib.sha256(preflight_log.read_bytes()).hexdigest()
            preflight_fingerprint = next(
                record
                for record in summary["command_log_fingerprints"]
                if record["name"] == "mace-preflight:preflight"
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("mace-preflight:preflight", command_names)
        self.assertFalse(any(":disabled:" in name for name in command_names))
        self.assertFalse(any(":enabled:" in name for name in command_names))
        self.assertTrue(preflight_log_exists)
        self.assertTrue(preflight_fingerprint["stderr"]["exists"])
        self.assertEqual(preflight_fingerprint["stderr"]["sha256"], preflight_log_digest)
        self.assertTrue(
            any(case["status"].startswith("failed:") for case in summary["cases"])
        )

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

    def test_manifest_validation_requires_sha256_for_required_artifacts_when_enabled(self) -> None:
        """Final paper manifests can require immutable digests for all inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "strict-artifacts"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "data.ext"
required = true
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "required artifact needs sha256",
        ):
            isodelta_cluster_suite.validate_suite_config(config)

    def test_manifest_validation_rejects_malformed_artifact_sha256(self) -> None:
        """Digest fields should be full SHA-256 hex strings, not placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "suite.toml"
            manifest_path.write_text(
                """
[suite]
name = "bad-sha"
required_models = ["SevenNet"]

[[artifacts]]
name = "dataset"
path = "data.ext"
sha256 = "replace-with-real-sha256"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

        with self.assertRaisesRegex(
            isodelta_cluster_suite.ClusterSuiteError,
            "64-character hexadecimal SHA-256",
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

    def test_prepare_artifacts_downloads_and_writes_audit_report(self) -> None:
        """Artifact preparation should finish before any GPU case is launched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.bin"
            destination_path = root / "inputs" / "prepared.bin"
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            source_path.write_bytes(b"prepared artifact bytes")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                f"""
[suite]
name = "prepare-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "{destination_path.as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["sevennet_trace.json"]
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--prepare-artifacts"]
            )
            report_path = output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            report = json.loads(report_path.read_text(encoding="utf-8"))
            destination_exists = destination_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(destination_exists)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["artifacts"][0]["actual_sha256"], digest)
        self.assertEqual(report["artifacts"][0]["size_bytes"], len(b"prepared artifact bytes"))

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
                    "--reuse-passed",
                ]
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        self.assertEqual(exit_code, 0)
        self.assertFalse(plan["gpu_check_planned"])
        self.assertTrue(plan["modes"]["reuse_passed"])
        self.assertTrue(plan["cases"][0]["reuse"]["enabled"])
        self.assertTrue(plan["artifacts"][0]["will_download"])
        self.assertTrue(plan["artifacts"][0]["missing_required"])
        self.assertFalse(plan["suite"]["require_artifact_sha256"])
        self.assertFalse(plan["artifacts"][0]["has_sha256"])
        self.assertEqual(
            plan["suite"]["manifest"]["sha256"],
            manifest_digest,
        )
        self.assertEqual(plan["cases"][0]["model"], "SevenNet")
        self.assertIn("trace_evidence", plan["cases"][0]["expected_outputs"])
        self.assertIn("environment_snapshot.json", plan["paper_outputs"]["environment_snapshot"])
        self.assertIn("speedup_by_case.svg", plan["paper_outputs"]["speedup_svg"])

    def test_preflight_only_downloads_artifacts_and_runs_case_checks(self) -> None:
        """Preflight-only mode should verify inputs and model launch commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            target_path = root / "downloaded-data.bin"
            source_path.write_bytes(b"cluster preflight input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            preflight_path = root / "preflight_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
require_artifact_sha256 = true

[[artifacts]]
name = "dataset"
path = "{target_path.as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet"]

[[cases]]
name = "sevennet-preflight"
model = "SevenNet"
kind = "trace_only"
preflight_command = '"{python_bin}" -c "print(12345)"'
trace_input = "trace.json"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--preflight-only",
                    "--preflight-output",
                    str(preflight_path),
                    "--skip-gpu-check",
                ]
            )
            report = json.loads(preflight_path.read_text(encoding="utf-8"))
            stdout_path = Path(report["case_preflights"][0]["stdout_path"])
            stdout_text = stdout_path.read_text(encoding="utf-8")
            target_exists = target_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(target_exists)
        self.assertEqual(report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED)
        self.assertEqual(
            report["preflight_report_schema_version"],
            isodelta_cluster_suite.PREFLIGHT_REPORT_SCHEMA_VERSION,
        )
        self.assertTrue(report["downloads"][0]["downloaded"])
        self.assertEqual(
            report["case_preflights"][0]["status"],
            isodelta_cluster_suite.PREFLIGHT_STATUS_PASSED,
        )
        self.assertIn("12345", stdout_text)
        self.assertEqual(len(report["command_log_fingerprints"]), 1)
        self.assertIn(
            isodelta_cluster_suite.PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME,
            report["environment_snapshot"],
        )
        self.assertGreater(report["environment_snapshot_fingerprint"]["size_bytes"], 0)
        self.assertEqual(
            len(report["environment_snapshot_fingerprint"]["sha256"]),
            isodelta_cluster_suite.SHA256_HEX_LENGTH,
        )

    def test_preflight_only_reports_failed_case_check(self) -> None:
        """A nonzero case preflight should fail before expensive model runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            manifest_path = root / "suite.toml"
            output_dir = root / "paper_outputs"
            preflight_path = root / "preflight_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "preflight-failure-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]

[[cases]]
name = "sevennet-preflight-fail"
model = "SevenNet"
kind = "trace_only"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(7)"'
trace_input = "trace.json"
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--preflight-only",
                    "--preflight-output",
                    str(preflight_path),
                    "--skip-gpu-check",
                ]
            )
            report = json.loads(preflight_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_FAILED)
        self.assertEqual(report["case_preflights"][0]["returncode"], 7)
        self.assertEqual(report["failures"][0]["stage"], "case_preflight")

    def test_pipeline_runs_all_paper_stages_and_verifies_bundle(self) -> None:
        """Pipeline mode should chain readiness, preflight, run, and verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-pipeline"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "print(12345)"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = 3
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-suite"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_trace_count = 3
min_distinct_trace_models = 3
min_speedup = 1.05
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            for case in config.cases:
                timing_path = isodelta_cluster_suite._external_timing_report_path(
                    config,
                    case,
                )
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing_path.write_text(
                    json.dumps(
                        {
                            "schema_version": isodelta_cluster_suite.EXTERNAL_TIMING_SCHEMA_VERSION,
                            "case_name": case.name,
                            "model": case.model,
                            "repeat_count": 3,
                            "disabled_success_count": 3,
                            "enabled_success_count": 3,
                            "baseline_times_seconds": [BASELINE_LOOP_TIME_SECONDS] * 3,
                            "enabled_times_seconds": [ISODELTA_LOOP_TIME_SECONDS] * 3,
                            "baseline_mean_seconds": BASELINE_LOOP_TIME_SECONDS,
                            "enabled_mean_seconds": ISODELTA_LOOP_TIME_SECONDS,
                            "baseline_sample_variance_seconds": 0.0,
                            "enabled_sample_variance_seconds": 0.0,
                            "baseline_sample_stddev_seconds": 0.0,
                            "enabled_sample_stddev_seconds": 0.0,
                            "speedup_vs_disabled_cache": EXPECTED_SPEEDUP,
                            "commands": [],
                        }
                    ),
                    encoding="utf-8",
                )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pipeline",
                    "--pipeline-report",
                    str(pipeline_report_path),
                    "--skip-gpu-check",
                    "--reuse-passed",
                ]
            )
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            summary_path = output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]

        self.assertEqual(exit_code, 0)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_PASSED)
        self.assertEqual(
            stage_names,
            [
                "readiness",
                "prepare_artifacts",
                "preflight",
                "plan",
                "run_suite",
                "verify_output_bundle",
            ],
        )
        self.assertEqual(verification["status"], "passed")
        self.assertIn("preflight_report", summary["artifact_fingerprints"])
        self.assertIn("preflight_environment_snapshot", summary["artifact_fingerprints"])
        self.assertIn("run_plan", summary["artifact_fingerprints"])
        self.assertNotIn("pipeline_report", summary["artifact_fingerprints"])

    def test_pipeline_stops_when_readiness_fails(self) -> None:
        """Pipeline mode should not prepare inputs or run cases after a failed gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-readiness-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-trace"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["missing_sevennet_trace.json"]

[[cases]]
name = "mace-trace"
model = "MACE"
kind = "trace_only"
trace_evidence = ["missing_mace_trace.json"]

[[cases]]
name = "nequip-trace"
model = "NequIP"
kind = "trace_only"
trace_evidence = ["missing_nequip_trace.json"]
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pipeline",
                    "--pipeline-report",
                    str(pipeline_report_path),
                    "--skip-gpu-check",
                ]
            )
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            readiness_path = output_dir / isodelta_cluster_suite.READINESS_REPORT_NAME
            stopped_before_prepare = not (
                output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            ).exists()
            stopped_before_preflight = not (
                output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            ).exists()
            stopped_before_plan = not (
                output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            ).exists()
            stopped_before_summary = not (
                output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            ).exists()
            readiness_exists = readiness_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual([stage["name"] for stage in pipeline_report["stages"]], ["readiness"])
        self.assertTrue(readiness_exists)
        self.assertTrue(stopped_before_prepare)
        self.assertTrue(stopped_before_preflight)
        self.assertTrue(stopped_before_plan)
        self.assertTrue(stopped_before_summary)

    def test_pipeline_stops_when_preflight_fails(self) -> None:
        """A failed model preflight should stop before plan, run, and summary stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline preflight failure input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-preflight-fails"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                preflight_exit_code = (
                    PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE
                    if model_name == "MACE"
                    else isodelta_cluster_suite.SUCCESS_RETURN_CODE
                )
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit({preflight_exit_code})"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = {PAPER_REPEAT_COUNT}
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-preflight-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = {PAPER_REPEAT_COUNT}
min_trace_count = {PAPER_REPEAT_COUNT}
min_distinct_trace_models = {len(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS)}
min_speedup = {PIPELINE_MIN_SPEEDUP}
min_speedup_95ci_lower_bound = {PIPELINE_MIN_SPEEDUP_LOWER_BOUND}

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )

            exit_code = isodelta_cluster_suite.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pipeline",
                    "--pipeline-report",
                    str(pipeline_report_path),
                    "--skip-gpu-check",
                ]
            )
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            preflight_path = output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]
            stopped_before_plan = not (
                output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            ).exists()
            stopped_before_summary = not (
                output_dir / isodelta_cluster_suite.SUMMARY_REPORT_NAME
            ).exists()
            artifact_report_exists = (
                output_dir / isodelta_cluster_suite.ARTIFACT_PREPARATION_REPORT_NAME
            ).exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual(stage_names, ["readiness", "prepare_artifacts", "preflight"])
        self.assertEqual(preflight_report["status"], isodelta_cluster_suite.PREFLIGHT_STATUS_FAILED)
        self.assertEqual(preflight_report["failures"][0]["stage"], "case_preflight")
        self.assertEqual(
            preflight_report["case_preflights"][1]["returncode"],
            PIPELINE_PREFLIGHT_FAILURE_RETURN_CODE,
        )
        self.assertTrue(artifact_report_exists)
        self.assertTrue(stopped_before_plan)
        self.assertTrue(stopped_before_summary)

    def test_pipeline_reports_bundle_verification_failure(self) -> None:
        """A final fingerprint mismatch should fail the pipeline after run output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            python_bin = Path(sys.executable).as_posix()
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"pipeline verification failure input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output_dir = root / "paper_outputs"
            manifest_path = root / "suite.toml"
            pipeline_report_path = root / "pipeline_report.json"
            case_blocks = []
            for model_name in ("SevenNet", "MACE", "NequIP"):
                case_name = f"{model_name.lower()}-verification-fails"
                trace_path = root / f"{model_name.lower()}_trace.json"
                trace_path.write_text(json.dumps(_trace_evidence(model_name)), encoding="utf-8")
                case_blocks.append(
                    f"""
[[cases]]
name = "{case_name}"
model = "{model_name}"
kind = "external_pair"
preflight_command = '"{python_bin}" -c "import sys; sys.exit(0)"'
disabled_command = "unused-disabled"
enabled_command = "unused-enabled"
repeat_count = {PAPER_REPEAT_COUNT}
trace_evidence = ["{trace_path.as_posix()}"]
required_trace_models = ["{model_name}"]
artifacts = ["dataset"]
"""
                )
            manifest_path.write_text(
                f"""
[suite]
name = "pipeline-verification-fails"
output_dir = "{output_dir.as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = {PAPER_REPEAT_COUNT}
min_trace_count = {PAPER_REPEAT_COUNT}
min_distinct_trace_models = {len(isodelta_cluster_suite.FINAL_PAPER_REQUIRED_MODELS)}
min_speedup = {PIPELINE_MIN_SPEEDUP}
min_speedup_95ci_lower_bound = {PIPELINE_MIN_SPEEDUP_LOWER_BOUND}

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

{''.join(case_blocks)}
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            for case in config.cases:
                timing_path = isodelta_cluster_suite._external_timing_report_path(
                    config,
                    case,
                )
                timing_path.parent.mkdir(parents=True, exist_ok=True)
                timing_path.write_text(
                    json.dumps(
                        {
                            "schema_version": isodelta_cluster_suite.EXTERNAL_TIMING_SCHEMA_VERSION,
                            "case_name": case.name,
                            "model": case.model,
                            "repeat_count": PAPER_REPEAT_COUNT,
                            "disabled_success_count": PAPER_REPEAT_COUNT,
                            "enabled_success_count": PAPER_REPEAT_COUNT,
                            "baseline_times_seconds": [BASELINE_LOOP_TIME_SECONDS]
                            * PAPER_REPEAT_COUNT,
                            "enabled_times_seconds": [ISODELTA_LOOP_TIME_SECONDS]
                            * PAPER_REPEAT_COUNT,
                            "baseline_mean_seconds": BASELINE_LOOP_TIME_SECONDS,
                            "enabled_mean_seconds": ISODELTA_LOOP_TIME_SECONDS,
                            "baseline_sample_variance_seconds": 0.0,
                            "enabled_sample_variance_seconds": 0.0,
                            "baseline_sample_stddev_seconds": 0.0,
                            "enabled_sample_stddev_seconds": 0.0,
                            "speedup_vs_disabled_cache": EXPECTED_SPEEDUP,
                            "commands": [],
                        }
                    ),
                    encoding="utf-8",
                )

            original_verify_output_bundle = isodelta_cluster_suite.verify_output_bundle

            def corrupt_case_summary_before_verify(bundle_or_summary_path: Path) -> dict[str, object]:
                """Mutate a generated table so the real bundle verifier must reject it."""
                case_summary_path = output_dir / "tables" / "case_summary.csv"
                case_summary_path.write_text(
                    case_summary_path.read_text(encoding="utf-8") + "\ncorrupted-row\n",
                    encoding="utf-8",
                )
                return original_verify_output_bundle(bundle_or_summary_path)

            isodelta_cluster_suite.verify_output_bundle = corrupt_case_summary_before_verify
            try:
                exit_code = isodelta_cluster_suite.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--pipeline",
                        "--pipeline-report",
                        str(pipeline_report_path),
                        "--skip-gpu-check",
                        "--reuse-passed",
                    ]
                )
            finally:
                isodelta_cluster_suite.verify_output_bundle = original_verify_output_bundle
            pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
            stage_names = [stage["name"] for stage in pipeline_report["stages"]]
            verification_stage = pipeline_report["stages"][-1]

        self.assertEqual(exit_code, 1)
        self.assertEqual(pipeline_report["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertEqual(
            stage_names,
            [
                "readiness",
                "prepare_artifacts",
                "preflight",
                "plan",
                "run_suite",
                "verify_output_bundle",
            ],
        )
        self.assertEqual(verification_stage["status"], isodelta_cluster_suite.PIPELINE_STATUS_FAILED)
        self.assertIn("SHA-256 mismatch", verification_stage["detail"])

    def test_readiness_check_accepts_strict_three_model_paired_manifest(self) -> None:
        """A final paper manifest should prove strict input and model coverage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source-data.bin"
            source_path.write_bytes(b"strict cluster input")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "ready-suite"
output_dir = "{(root / "paper_outputs").as_posix()}"
expected_gpus = 8
required_models = ["SevenNet", "MACE", "NequIP"]
require_artifact_sha256 = true
repeat_count = 3
min_speedup_95ci_lower_bound = 1.0

[[artifacts]]
name = "dataset"
path = "{(root / "downloaded.bin").as_posix()}"
url = "{source_path.as_uri()}"
sha256 = "{digest}"
required_by = ["SevenNet", "MACE", "NequIP"]

[[cases]]
name = "sevennet-ready"
model = "SevenNet"
kind = "sevennet_lammps"
preflight_command = 'python -c "import sevenn"'
lammps_command = "mpiexec -n 8 lmp"
input = "inputs/in.sevennet"
artifacts = ["dataset"]

[[cases]]
name = "mace-ready"
model = "MACE"
kind = "external_pair"
preflight_command = 'python -c "import mace"'
disabled_command = "python run_mace.py --mode baseline"
enabled_command = "python run_mace.py --mode isodelta"
artifacts = ["dataset"]

[[cases]]
name = "nequip-ready"
model = "NequIP"
kind = "external_pair"
preflight_command = 'python -c "import nequip"'
disabled_command = "python run_nequip.py --mode baseline"
enabled_command = "python run_nequip.py --mode isodelta"
artifacts = ["dataset"]
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--readiness-check"]
            )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        self.assertEqual(exit_code, 0)

    def test_readiness_check_rejects_unedited_template_manifest(self) -> None:
        """A generated template must be filled in before paper execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "suite.toml"
            isodelta_cluster_suite.write_template(manifest_path)
            config = isodelta_cluster_suite.load_manifest(manifest_path)

            report = isodelta_cluster_suite.build_readiness_report(config)
            exit_code = isodelta_cluster_suite.main(
                ["--manifest", str(manifest_path), "--readiness-check"]
            )
            failed_checks = {
                check["name"] for check in report["checks"] if not check["passed"]
            }

        self.assertEqual(report["status"], "failed")
        self.assertEqual(exit_code, 1)
        self.assertIn("manifest_schema", failed_checks)
        self.assertIn("artifact_template_markers_removed", failed_checks)

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

    def test_speedup_lower_bound_gate_rejects_uncertain_case(self) -> None:
        """A mean speedup should fail when its conservative CI bound is weak."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sevennet_trace_path = root / "sevennet_trace.json"
            nequip_trace_path = root / "nequip_trace.json"
            nequip_timing_path = root / "nequip_timing.json"
            output_dir = root / "paper_outputs"
            sevennet_trace_path.write_text(json.dumps(_trace_evidence("SevenNet")), encoding="utf-8")
            nequip_trace_path.write_text(json.dumps(_trace_evidence("NequIP")), encoding="utf-8")
            nequip_timing_path.write_text(json.dumps(_external_timing_report("NequIP")), encoding="utf-8")
            manifest_path = root / "suite.toml"
            manifest_path.write_text(
                f"""
[suite]
name = "lower-bound-suite"
output_dir = "{output_dir.as_posix()}"
required_models = ["SevenNet"]
min_trace_count = 1
min_distinct_trace_models = 1

[[cases]]
name = "sevennet-pass"
model = "SevenNet"
kind = "trace_only"
trace_evidence = ["{sevennet_trace_path.as_posix()}"]

[[cases]]
name = "nequip-existing"
model = "NequIP"
kind = "external_pair"
disabled_command = "baseline"
enabled_command = "enabled"
repeat_count = 2
external_timing_report = "{nequip_timing_path.as_posix()}"
trace_evidence = ["{nequip_trace_path.as_posix()}"]
min_speedup_95ci_lower_bound = 1.1
""",
                encoding="utf-8",
            )
            config = isodelta_cluster_suite.load_manifest(manifest_path)
            exit_code = isodelta_cluster_suite.run_suite(
                config,
                collect_only=True,
                skip_downloads=True,
                skip_gpu_check=True,
                keep_going=True,
            )
            summary = json.loads(
                (output_dir / "isodelta_cluster_paper_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            nequip_case = next(case for case in summary["cases"] if case["model"] == "NequIP")

        self.assertEqual(exit_code, 1)
        self.assertIn("speedup 95% CI lower bound", nequip_case["status"])
        self.assertLess(nequip_case["speedup_95ci_lower_bound"], 1.1)

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
            output_dir.mkdir(parents=True)
            preflight_report_path = output_dir / isodelta_cluster_suite.PREFLIGHT_REPORT_NAME
            plan_path = output_dir / isodelta_cluster_suite.PLAN_REPORT_NAME
            preflight_report_path.write_text(
                json.dumps({"status": "passed", "kind": "preflight"}),
                encoding="utf-8",
            )
            plan_path.write_text(
                json.dumps({"status": "planned", "kind": "plan"}),
                encoding="utf-8",
            )
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
            environment_snapshot = output_dir / "environment_snapshot.json"
            case_summary_csv = output_dir / "tables" / "case_summary.csv"
            correlation_csv = output_dir / "tables" / "correlation.csv"
            speedup_svg = output_dir / "figures" / "speedup_by_case.svg"
            manifest_snapshot = output_dir / "isodelta_cluster_suite_manifest.toml"

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            environment_payload = json.loads(environment_snapshot.read_text(encoding="utf-8"))
            environment_snapshot_exists = environment_snapshot.exists()
            case_summary_exists = case_summary_csv.exists()
            correlation_exists = correlation_csv.exists()
            speedup_svg_exists = speedup_svg.exists()
            manifest_snapshot_exists = manifest_snapshot.exists()
            case_summary_text = case_summary_csv.read_text(encoding="utf-8")
            speedup_svg_text = speedup_svg.read_text(encoding="utf-8")
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            environment_digest = hashlib.sha256(environment_snapshot.read_bytes()).hexdigest()
            case_summary_digest = hashlib.sha256(case_summary_csv.read_bytes()).hexdigest()
            manifest_snapshot_digest = hashlib.sha256(manifest_snapshot.read_bytes()).hexdigest()
            preflight_digest = hashlib.sha256(preflight_report_path.read_bytes()).hexdigest()
            plan_digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            environment_size = environment_snapshot.stat().st_size
            case_summary_size = case_summary_csv.stat().st_size
            manifest_snapshot_size = manifest_snapshot.stat().st_size
            preflight_size = preflight_report_path.stat().st_size
            plan_size = plan_path.stat().st_size
            nequip_case = next(
                case for case in summary["cases"] if case["model"] == "NequIP"
            )
            verification = isodelta_cluster_suite.verify_output_bundle(output_dir)
            case_summary_csv.write_text(case_summary_text + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                isodelta_cluster_suite.ClusterSuiteError,
                "SHA-256 mismatch",
            ):
                isodelta_cluster_suite.verify_output_bundle(summary_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(verification["status"], "passed")
        self.assertGreaterEqual(verification["verified_artifact_count"], 1)
        self.assertTrue(environment_snapshot_exists)
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
        self.assertIn("baseline_mean_95ci_half_width_seconds", case_summary_text)
        self.assertIn("speedup_95ci_lower_bound", case_summary_text)
        self.assertEqual(nequip_case["baseline_timing_count"], 2)
        self.assertAlmostEqual(
            nequip_case["baseline_mean_95ci_half_width_seconds"],
            isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER,
        )
        self.assertAlmostEqual(
            nequip_case["enabled_mean_95ci_half_width_seconds"],
            isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER,
        )
        self.assertAlmostEqual(
            nequip_case["speedup_95ci_lower_bound"],
            (
                BASELINE_LOOP_TIME_SECONDS
                - isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            )
            / (
                ISODELTA_LOOP_TIME_SECONDS
                + isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            ),
        )
        self.assertAlmostEqual(
            nequip_case["speedup_95ci_upper_bound"],
            (
                BASELINE_LOOP_TIME_SECONDS
                + isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            )
            / (
                ISODELTA_LOOP_TIME_SECONDS
                - isodelta_cluster_suite.NORMAL_APPROX_95_CI_MULTIPLIER
            ),
        )
        self.assertIn("<svg", speedup_svg_text)
        self.assertEqual(
            environment_payload["snapshot_schema_version"],
            isodelta_cluster_suite.ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        )
        self.assertIn("torch", environment_payload["package_versions"])
        self.assertIn("CUDA_VISIBLE_DEVICES", environment_payload["selected_environment"])
        self.assertEqual(
            summary["artifact_fingerprints"]["environment_snapshot"]["sha256"],
            environment_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["environment_snapshot"]["size_bytes"],
            environment_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["case_summary_csv"]["sha256"],
            case_summary_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["case_summary_csv"]["size_bytes"],
            case_summary_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["manifest_snapshot"]["sha256"],
            manifest_snapshot_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["manifest_snapshot"]["size_bytes"],
            manifest_snapshot_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["preflight_report"]["sha256"],
            preflight_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["preflight_report"]["size_bytes"],
            preflight_size,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["run_plan"]["sha256"],
            plan_digest,
        )
        self.assertEqual(
            summary["artifact_fingerprints"]["run_plan"]["size_bytes"],
            plan_size,
        )

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
