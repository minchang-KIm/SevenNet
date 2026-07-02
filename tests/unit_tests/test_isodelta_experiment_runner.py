"""Unit tests for the IsoDelta-Halo end-to-end experiment driver.

The experiment driver is mostly command orchestration, so these tests use fake
subprocess runners to verify command order, failure behavior, and report output
without launching LAMMPS.
"""

from __future__ import annotations

import importlib.util
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
        self.assertIn("--min-speedup", commands[3].argv)
        self.assertIn(str(config.benchmark_report_path()), commands[3].argv)

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
            exit_code = isodelta_experiment.run_experiment(config, runner=fake_runner)
            report = json.loads(config.experiment_report_path().read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(executed, ["other", "binary-smoke"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["failed_stage"], "binary-smoke")
        self.assertEqual(len(report["commands"]), 2)

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
            exit_code = isodelta_experiment.run_experiment(config, runner=fake_runner)
            report = json.loads(config.experiment_report_path().read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(executed), 4)
        self.assertTrue(report["ok"])
        self.assertIsNone(report["failed_stage"])
        self.assertEqual(len(report["commands"]), 4)
        self.assertEqual(report["benchmark_report"], str(config.benchmark_report_path()))


if __name__ == "__main__":
    unittest.main()
