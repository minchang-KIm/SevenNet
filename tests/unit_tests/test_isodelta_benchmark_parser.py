"""Unit tests for the IsoDelta-Halo benchmark log parser.

The benchmark runner must stay testable without launching LAMMPS, so these
tests cover parsing and summary behavior with small representative log strings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


# Load the script by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "run_isodelta_lammps_benchmark.py"
SPEC = importlib.util.spec_from_file_location("isodelta_benchmark", BENCHMARK_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_benchmark = importlib.util.module_from_spec(SPEC)
# Dataclasses inspect sys.modules while processing annotations, so a path-loaded
# module must be registered before exec_module runs.
sys.modules[SPEC.name] = isodelta_benchmark
SPEC.loader.exec_module(isodelta_benchmark)


class IsoDeltaBenchmarkParserTest(unittest.TestCase):
    """Check that profiling logs become stable numeric report fields."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark script source for CLI behavior checks."""
        cls.source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    def test_parse_loop_time_from_lammps_log(self) -> None:
        """LAMMPS loop-time lines should parse into seconds."""
        log_text = "Loop time of 12.3456 on 4 procs for 100 steps with 4096 atoms"
        self.assertEqual(isodelta_benchmark.parse_loop_time(log_text), 12.3456)

    def test_parse_cache_summary_counters(self) -> None:
        """IsoDelta-Halo summary lines should parse all numeric key-value pairs."""
        log_text = (
            "0 IsoDelta-Halo summary: attempts=10 hits=8 "
            "hit_rate_percent=80 miss_no-cache=1 miss_shape-changed=1"
        )
        parsed = isodelta_benchmark.parse_cache_summary(log_text)
        self.assertEqual(parsed["attempts"], 10.0)
        self.assertEqual(parsed["hits"], 8.0)
        self.assertEqual(parsed["hit_rate_percent"], 80.0)
        self.assertEqual(parsed["miss_no-cache"], 1.0)
        self.assertEqual(parsed["miss_shape-changed"], 1.0)

    def test_parse_run_output_uses_both_streams(self) -> None:
        """MPI wrappers may split loop time and profiling summary across streams."""
        stdout_text = "Loop time of 9.5 on 2 procs for 40 steps with 512 atoms"
        stderr_text = "0 IsoDelta-Halo summary: attempts=5 hits=4 hit_rate_percent=80"
        loop_time, summary = isodelta_benchmark.parse_run_output(stdout_text, stderr_text)
        self.assertEqual(loop_time, 9.5)
        self.assertEqual(summary["attempts"], 5.0)
        self.assertEqual(summary["hits"], 4.0)

    def test_build_command_appends_input_flag(self) -> None:
        """The runner should build commands without shell-specific quoting."""
        command = isodelta_benchmark._build_command("mpiexec -n 2 lmp", Path("in.test"))
        self.assertEqual(command, ["mpiexec", "-n", "2", "lmp", "-in", "in.test"])

    def test_work_dir_defaults_to_input_directory(self) -> None:
        """Relative files in LAMMPS inputs should resolve beside the input file."""
        self.assertIn('--work-dir', self.source)
        self.assertIn('work_dir = args.work_dir.resolve() if args.work_dir else input_path.parent', self.source)
        self.assertIn('"work_dir": str(work_dir)', self.source)


if __name__ == "__main__":
    unittest.main()
