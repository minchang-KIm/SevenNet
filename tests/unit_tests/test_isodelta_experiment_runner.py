"""Unit tests for the IsoDelta-Halo end-to-end experiment driver.

The experiment driver is mostly command orchestration, so these tests use fake
subprocess runners to verify command order, failure behavior, and report output
without launching LAMMPS.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_experiment.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_experiment",
    EXPERIMENT_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
isodelta_experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_experiment
SPEC.loader.exec_module(isodelta_experiment)


EXPECTED_EXPERIMENT_REPORT_SCHEMA_VERSION = "isodelta-experiment-report-v1"
EXPECTED_EXPERIMENT_REPORT_COMMENT = (
    "IsoDelta-Halo experiment driver report recording launched benchmark, trace, "
    "and evidence-bundle commands, output paths, return codes, and run provenance."
)
EXPECTED_FINGERPRINT_ALGORITHM = "sha256"
EMPTY_SHA256_HEXDIGEST = hashlib.sha256(b"").hexdigest()
MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY = 2
EMPTY_TRACE_EVIDENCE_COUNT = 0


class IsoDeltaExperimentRunnerTest(unittest.TestCase):
    """Check the experiment runner without depending on a LAMMPS binary."""

    def test_build_experiment_commands_preserves_gate_order(self) -> None:
        """The driver should run prereq, binary, benchmark, then report gate."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="mpiexec -n 2 lmp",
            input_path=Path("in.sevenn"),
            output_dir=Path("out"),
            repeat_count=5,
            lammps_root=Path("lammps"),
            require_torch=True,
            min_speedup=1.05,
            min_hit_rate_percent=50.0,
            min_enabled_cache_attempts=4,
            min_enabled_cache_hits=2,
            benchmark_timeout_seconds=120.0,
        )
        commands = isodelta_experiment.build_experiment_commands(config)
        self.assertEqual(
            [command.name for command in commands],
            ["prerequisites", "binary-smoke", "paired-benchmark", "report-gate"],
        )
        self.assertIn("--require-torch", commands[0].argv)
        self.assertIn("mpiexec -n 2 lmp", commands[1].argv)
        self.assertIn("--repeat", commands[2].argv)
        self.assertIn("5", commands[2].argv)
        self.assertIn("--run-timeout-seconds", commands[2].argv)
        self.assertIn("120.0", commands[2].argv)
        self.assertIn("--min-speedup", commands[3].argv)
        self.assertIn("--min-enabled-cache-attempts", commands[3].argv)
        self.assertIn("4", commands[3].argv)
        self.assertIn("--min-enabled-cache-hits", commands[3].argv)
        self.assertIn("2", commands[3].argv)
        self.assertIn(str(config.benchmark_report_path()), commands[3].argv)
        self.assertIn("--ablation-mode", commands[2].argv)
        self.assertIn(isodelta_experiment.ABLATION_MODE_PAIRED, commands[2].argv)

    def test_build_experiment_commands_supports_one_sided_ablation(self) -> None:
        """One-sided ablation should run raw timing without publishable gates."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            output_dir=Path("out"),
            ablation_mode=isodelta_experiment.ABLATION_MODE_ENABLED_ONLY,
        )
        commands = isodelta_experiment.build_experiment_commands(config)

        self.assertEqual(
            [command.name for command in commands],
            ["prerequisites", "binary-smoke", "ablation-benchmark"],
        )
        self.assertIn("--ablation-mode", commands[-1].argv)
        self.assertIn(isodelta_experiment.ABLATION_MODE_ENABLED_ONLY, commands[-1].argv)
        self.assertNotIn("report-gate", [command.name for command in commands])

    def test_validate_config_rejects_speedup_gate_for_one_sided_ablation(self) -> None:
        """Speedup thresholds need both baseline and enabled timings."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            ablation_mode=isodelta_experiment.ABLATION_MODE_BASELINE_ONLY,
            min_speedup=1.1,
        )

        with self.assertRaisesRegex(ValueError, "paired ablation_mode"):
            isodelta_experiment.validate_config(config)

    def test_build_experiment_commands_can_append_bundle_gate(self) -> None:
        """Trace evidence options should add a final evidence-bundle gate."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            output_dir=Path("out"),
            min_speedup=1.05,
            trace_evidence_paths=(
                Path("mace_trace_evidence.json"),
                Path("nequip_trace_evidence.json"),
            ),
            required_trace_models=("MACE", "NequIP"),
            min_distinct_trace_models=MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY,
            min_trace_hit_rate_percent=50.0,
            min_trace_estimated_speedup=1.05,
            min_trace_metadata_fraction_percent=5.0,
        )
        commands = isodelta_experiment.build_experiment_commands(config)

        self.assertEqual(commands[-1].name, "evidence-bundle")
        self.assertEqual(len(commands), 5)
        self.assertIn("check_isodelta_evidence_bundle.py", commands[-1].argv[1])
        self.assertIn("--trace-evidence", commands[-1].argv)
        self.assertIn("mace_trace_evidence.json", commands[-1].argv)
        self.assertIn("nequip_trace_evidence.json", commands[-1].argv)
        self.assertIn("--require-trace-model", commands[-1].argv)
        self.assertIn("MACE", commands[-1].argv)
        self.assertIn("NequIP", commands[-1].argv)
        self.assertIn("--min-distinct-trace-models", commands[-1].argv)
        self.assertIn(str(MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY), commands[-1].argv)
        self.assertIn("--min-trace-estimated-speedup", commands[-1].argv)
        self.assertIn(str(config.bundle_evidence_report_path()), commands[-1].argv)

    def test_validate_config_rejects_zero_repeat_count(self) -> None:
        """The experiment driver should not build empty paired benchmarks."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            repeat_count=0,
        )
        with self.assertRaisesRegex(ValueError, "repeat_count"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_nonpositive_binary_timeout(self) -> None:
        """A binary smoke test timeout must be positive to be meaningful."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            binary_timeout_seconds=0.0,
        )
        with self.assertRaisesRegex(ValueError, "binary_timeout_seconds"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_nonpositive_benchmark_timeout(self) -> None:
        """Each paired benchmark run should have a positive timeout."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            benchmark_timeout_seconds=0.0,
        )
        with self.assertRaisesRegex(ValueError, "benchmark_timeout_seconds"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_impossible_cache_gate(self) -> None:
        """Minimum cache hits cannot exceed the minimum attempts threshold."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            min_enabled_cache_attempts=1,
            min_enabled_cache_hits=2,
        )
        with self.assertRaisesRegex(ValueError, "min_enabled_cache_hits"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_zero_distinct_model_gate(self) -> None:
        """The experiment bundle gate should require a positive model count."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            min_distinct_trace_models=0,
        )
        with self.assertRaisesRegex(ValueError, "min_distinct_trace_models"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_trace_threshold_without_evidence(self) -> None:
        """Trace-specific thresholds should not be silently ignored."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            min_distinct_trace_models=MIN_DISTINCT_TRACE_MODELS_FOR_PORTABILITY,
        )
        self.assertEqual(len(config.trace_evidence_paths), EMPTY_TRACE_EVIDENCE_COUNT)
        with self.assertRaisesRegex(ValueError, "trace_evidence_paths"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_duplicate_trace_evidence_paths(self) -> None:
        """One trace evidence artifact should not be scheduled twice."""
        trace_path = Path("mace_trace_evidence.json")
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            trace_evidence_paths=(trace_path, trace_path),
        )
        with self.assertRaisesRegex(ValueError, "duplicate trace evidence paths"):
            isodelta_experiment.validate_config(config)

    def test_validate_config_rejects_duplicate_required_trace_models(self) -> None:
        """Duplicate required model labels should fail before benchmark launch."""
        config = isodelta_experiment.ExperimentConfig(
            lammps_command="lmp",
            input_path=Path("in.sevenn"),
            trace_evidence_paths=(
                Path("first_mace_trace_evidence.json"),
                Path("second_mace_trace_evidence.json"),
            ),
            required_trace_models=("MACE", "MACE"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate required_trace_models"):
            isodelta_experiment.validate_config(config)

    def test_run_experiment_stops_on_first_failed_stage(self) -> None:
        """A failing gate should write a partial report and skip later stages."""
        executed: list[str] = []

        def fake_runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            stage_name = "binary-smoke" if "check_isodelta_lammps_binary.py" in argv[1] else "other"
            executed.append(stage_name)
            return subprocess.CompletedProcess(
                args=argv,
                returncode=2 if stage_name == "binary-smoke" else 0,
                stdout=f"{stage_name} stdout",
                stderr=f"{stage_name} stderr",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = isodelta_experiment.ExperimentConfig(
                lammps_command="lmp",
                input_path=Path("in.sevenn"),
                output_dir=Path(tmpdir),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = isodelta_experiment.run_experiment(
                    config,
                    runner=fake_runner,
                )
            report = json.loads(config.experiment_report_path().read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(executed, ["other", "binary-smoke"])
        self.assertEqual(report["report_comment"], EXPECTED_EXPERIMENT_REPORT_COMMENT)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failed_stage"], "binary-smoke")
        self.assertEqual(len(report["commands"]), 2)
        failed_command = report["commands"][1]
        self.assertEqual(failed_command["stdout_fingerprint"]["exists"], True)
        self.assertEqual(
            failed_command["stdout_fingerprint"]["algorithm"],
            EXPECTED_FINGERPRINT_ALGORITHM,
        )
        self.assertEqual(
            failed_command["stdout_fingerprint"]["sha256"],
            hashlib.sha256(b"binary-smoke stdout").hexdigest(),
        )
        self.assertEqual(failed_command["stdout_fingerprint"]["byte_size"], 19)
        self.assertEqual(
            failed_command["stderr_fingerprint"]["sha256"],
            hashlib.sha256(b"binary-smoke stderr").hexdigest(),
        )
        self.assertEqual(
            report["provenance"]["report_schema_version"],
            EXPECTED_EXPERIMENT_REPORT_SCHEMA_VERSION,
        )

    def test_run_experiment_writes_success_report(self) -> None:
        """A fully passing run should write all command results."""
        executed: list[list[str]] = []

        def fake_runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            executed.append(argv)
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="ok",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = isodelta_experiment.ExperimentConfig(
                lammps_command="lmp",
                input_path=Path("in.sevenn"),
                output_dir=Path(tmpdir),
                min_speedup=1.1,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = isodelta_experiment.run_experiment(
                    config,
                    runner=fake_runner,
                )
            report = json.loads(config.experiment_report_path().read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(executed), 4)
        self.assertEqual(report["report_comment"], EXPECTED_EXPERIMENT_REPORT_COMMENT)
        self.assertTrue(report["ok"])
        self.assertIsNone(report["failed_stage"])
        self.assertEqual(len(report["commands"]), 4)
        self.assertEqual(report["benchmark_report"], str(config.benchmark_report_path()))
        for command in report["commands"]:
            self.assertEqual(command["stdout_fingerprint"]["exists"], True)
            self.assertEqual(command["stderr_fingerprint"]["exists"], True)
            self.assertEqual(
                command["stdout_fingerprint"]["algorithm"],
                EXPECTED_FINGERPRINT_ALGORITHM,
            )
            self.assertEqual(
                command["stdout_fingerprint"]["sha256"],
                hashlib.sha256(b"ok").hexdigest(),
            )
            self.assertEqual(command["stdout_fingerprint"]["byte_size"], 2)
            self.assertEqual(
                command["stderr_fingerprint"]["sha256"],
                EMPTY_SHA256_HEXDIGEST,
            )
            self.assertEqual(command["stderr_fingerprint"]["byte_size"], 0)
        self.assertIn("git_commit", report["provenance"])
        self.assertIn("python_executable", report["provenance"])

    def test_run_experiment_writes_bundle_report_path_when_requested(self) -> None:
        """The success report should expose the optional bundle evidence output."""
        executed: list[list[str]] = []

        def fake_runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            executed.append(argv)
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="ok",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = isodelta_experiment.ExperimentConfig(
                lammps_command="lmp",
                input_path=Path("in.sevenn"),
                output_dir=Path(tmpdir),
                trace_evidence_paths=(Path("mace_trace_evidence.json"),),
                required_trace_models=("MACE",),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = isodelta_experiment.run_experiment(
                    config,
                    runner=fake_runner,
                )
            report = json.loads(config.experiment_report_path().read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(executed), 5)
        self.assertEqual(report["report_comment"], EXPECTED_EXPERIMENT_REPORT_COMMENT)
        self.assertEqual(
            report["bundle_evidence_report"],
            str(config.bundle_evidence_report_path()),
        )
        self.assertEqual(
            report["config"]["trace_evidence_paths"],
            ["mace_trace_evidence.json"],
        )


if __name__ == "__main__":
    unittest.main()
