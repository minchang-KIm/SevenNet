"""Unit tests for the IsoDelta-Halo benchmark log parser.

The benchmark runner must stay testable without launching LAMMPS, so these
tests cover parsing and summary behavior with small representative log strings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
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


MPI_RANK_COUNT = 2.0
AGGREGATED_ATTEMPTS = 30.0
AGGREGATED_HITS = 18.0
AGGREGATED_HIT_RATE_PERCENT = 60.0
AGGREGATED_NO_CACHE_MISSES = 3.0
AGGREGATED_SHAPE_CHANGED_MISSES = 3.0
EXPECTED_BENCHMARK_REPORT_SCHEMA_VERSION = "isodelta-benchmark-report-v1"


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
        self.assertEqual(parsed["summary_rank_count"], 1.0)
        self.assertEqual(parsed["miss_no-cache"], 1.0)
        self.assertEqual(parsed["miss_shape-changed"], 1.0)

    def test_parse_cache_summary_aggregates_mpi_rank_counters(self) -> None:
        """Multiple rank summaries should aggregate counters and recompute rate."""
        log_text = "\n".join(
            (
                "0 IsoDelta-Halo summary: attempts=10 hits=8 "
                "hit_rate_percent=80 miss_no-cache=1 miss_shape-changed=1",
                "1 IsoDelta-Halo summary: attempts=20 hits=10 "
                "hit_rate_percent=50 miss_no-cache=2 miss_shape-changed=2",
            )
        )
        parsed = isodelta_benchmark.parse_cache_summary(log_text)
        self.assertEqual(parsed["summary_rank_count"], MPI_RANK_COUNT)
        self.assertEqual(parsed["attempts"], AGGREGATED_ATTEMPTS)
        self.assertEqual(parsed["hits"], AGGREGATED_HITS)
        self.assertEqual(parsed["hit_rate_percent"], AGGREGATED_HIT_RATE_PERCENT)
        self.assertEqual(parsed["miss_no-cache"], AGGREGATED_NO_CACHE_MISSES)
        self.assertEqual(
            parsed["miss_shape-changed"],
            AGGREGATED_SHAPE_CHANGED_MISSES,
        )

    def test_parse_run_output_uses_both_streams(self) -> None:
        """MPI wrappers may split loop time and profiling summary across streams."""
        stdout_text = "\n".join(
            (
                "Step Temp PotEng TotEng",
                "0 300 -10.0 -9.0",
                "40 310 -10.5 -9.2",
                "Loop time of 9.5 on 2 procs for 40 steps with 512 atoms",
            )
        )
        stderr_text = "0 IsoDelta-Halo summary: attempts=5 hits=4 hit_rate_percent=80"
        loop_time, summary, thermo = isodelta_benchmark.parse_run_output(
            stdout_text,
            stderr_text,
        )
        self.assertEqual(loop_time, 9.5)
        self.assertEqual(summary["attempts"], 5.0)
        self.assertEqual(summary["hits"], 4.0)
        self.assertEqual(thermo["Step"], 40.0)
        self.assertEqual(thermo["PotEng"], -10.5)
        self.assertEqual(thermo["TotEng"], -9.2)

    def test_summarize_final_thermo_deltas(self) -> None:
        """Paired runs should expose output-difference evidence in the report."""
        baseline = isodelta_benchmark.BenchmarkResult(
            case=isodelta_benchmark.BASELINE_CASE,
            repeat_index=0,
            returncode=0,
            loop_time_seconds=12.0,
            cache_summary={},
            final_thermo_observables={"Step": 100.0, "PotEng": -20.0},
            stdout_path="baseline.out",
            stderr_path="baseline.err",
        )
        enabled = isodelta_benchmark.BenchmarkResult(
            case=isodelta_benchmark.ISODELTA_CASE,
            repeat_index=0,
            returncode=0,
            loop_time_seconds=10.0,
            cache_summary={"hit_rate_percent": 90.0},
            final_thermo_observables={"Step": 100.0, "PotEng": -19.999},
            stdout_path="enabled.out",
            stderr_path="enabled.err",
        )
        summary = isodelta_benchmark._summarize([baseline, enabled])
        self.assertEqual(summary["speedup_vs_disabled_cache"], 1.2)
        self.assertAlmostEqual(
            summary["final_thermo_delta_vs_disabled_cache"]["PotEng"][
                "max_abs_delta"
            ],
            0.001,
        )
        self.assertIsNone(
            summary["cases"][isodelta_benchmark.BASELINE_CASE][
                "sample_variance_loop_time_seconds"
            ]
        )

    def test_summarize_loop_time_repeat_statistics(self) -> None:
        """Repeated timings should include variance fields for paper tables."""
        results = [
            isodelta_benchmark.BenchmarkResult(
                case=isodelta_benchmark.BASELINE_CASE,
                repeat_index=0,
                returncode=0,
                loop_time_seconds=10.0,
                cache_summary={},
                final_thermo_observables={},
                stdout_path="baseline0.out",
                stderr_path="baseline0.err",
            ),
            isodelta_benchmark.BenchmarkResult(
                case=isodelta_benchmark.BASELINE_CASE,
                repeat_index=1,
                returncode=0,
                loop_time_seconds=14.0,
                cache_summary={},
                final_thermo_observables={},
                stdout_path="baseline1.out",
                stderr_path="baseline1.err",
            ),
            isodelta_benchmark.BenchmarkResult(
                case=isodelta_benchmark.ISODELTA_CASE,
                repeat_index=0,
                returncode=0,
                loop_time_seconds=5.0,
                cache_summary={},
                final_thermo_observables={},
                stdout_path="enabled0.out",
                stderr_path="enabled0.err",
            ),
            isodelta_benchmark.BenchmarkResult(
                case=isodelta_benchmark.ISODELTA_CASE,
                repeat_index=1,
                returncode=0,
                loop_time_seconds=7.0,
                cache_summary={},
                final_thermo_observables={},
                stdout_path="enabled1.out",
                stderr_path="enabled1.err",
            ),
        ]
        summary = isodelta_benchmark._summarize(results)
        baseline_summary = summary["cases"][isodelta_benchmark.BASELINE_CASE]
        enabled_summary = summary["cases"][isodelta_benchmark.ISODELTA_CASE]
        self.assertEqual(baseline_summary["mean_loop_time_seconds"], 12.0)
        self.assertEqual(baseline_summary["sample_variance_loop_time_seconds"], 8.0)
        self.assertAlmostEqual(
            baseline_summary["sample_stddev_loop_time_seconds"],
            8.0 ** 0.5,
        )
        self.assertEqual(baseline_summary["min_loop_time_seconds"], 10.0)
        self.assertEqual(baseline_summary["max_loop_time_seconds"], 14.0)
        self.assertEqual(enabled_summary["mean_loop_time_seconds"], 6.0)
        self.assertEqual(summary["speedup_vs_disabled_cache"], 2.0)

    def test_build_command_appends_input_flag(self) -> None:
        """The runner should build commands without shell-specific quoting."""
        command = isodelta_benchmark._build_command("mpiexec -n 2 lmp", Path("in.test"))
        self.assertEqual(command, ["mpiexec", "-n", "2", "lmp", "-in", "in.test"])

    def test_validate_benchmark_options_rejects_empty_repeat_set(self) -> None:
        """A benchmark with zero pairs cannot support a performance claim."""
        with self.assertRaisesRegex(ValueError, "repeat_count"):
            isodelta_benchmark.validate_benchmark_options(
                0,
                isodelta_benchmark.DEFAULT_RUN_TIMEOUT_SECONDS,
            )

    def test_validate_benchmark_options_rejects_nonpositive_timeout(self) -> None:
        """Each LAMMPS benchmark run should have a positive timeout."""
        with self.assertRaisesRegex(ValueError, "run_timeout_seconds"):
            isodelta_benchmark.validate_benchmark_options(1, 0.0)

    def test_run_case_records_timeout_as_failed_result(self) -> None:
        """Timeouts should leave raw logs and fail like other bad runs."""
        original_run = isodelta_benchmark.subprocess.run

        def fake_run(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(
                cmd=["lmp"],
                timeout=1.0,
                output="partial stdout",
                stderr="partial stderr",
            )

        try:
            isodelta_benchmark.subprocess.run = fake_run
            with tempfile.TemporaryDirectory() as tmpdir:
                result = isodelta_benchmark._run_case(
                    command=["lmp", "-in", "in.test"],
                    case=isodelta_benchmark.BENCHMARK_CASES[1],
                    repeat_index=0,
                    work_dir=Path(tmpdir),
                    output_dir=Path(tmpdir),
                    keep_going=True,
                    run_timeout_seconds=1.0,
                )
                stderr_text = Path(result.stderr_path).read_text(encoding="utf-8")
                stdout_text = Path(result.stdout_path).read_text(encoding="utf-8")
        finally:
            isodelta_benchmark.subprocess.run = original_run

        self.assertEqual(result.returncode, isodelta_benchmark.TIMEOUT_RETURN_CODE)
        self.assertIn("partial stdout", stdout_text)
        self.assertIn(isodelta_benchmark.TIMEOUT_DETAIL_PREFIX, stderr_text)

    def test_collect_run_provenance_records_git_and_runtime_context(self) -> None:
        """Benchmark reports should carry enough context for audit trails."""
        provenance = isodelta_benchmark.collect_run_provenance()
        self.assertEqual(
            provenance["report_schema_version"],
            EXPECTED_BENCHMARK_REPORT_SCHEMA_VERSION,
        )
        self.assertIn("git_commit", provenance)
        self.assertIn("git_branch", provenance)
        self.assertIn("git_dirty", provenance)
        self.assertIn("python_executable", provenance)
        self.assertIn("platform", provenance)
        self.assertIn(
            isodelta_benchmark.ISODELTA_CASE,
            provenance["case_environment_overrides"],
        )

    def test_work_dir_defaults_to_input_directory(self) -> None:
        """Relative files in LAMMPS inputs should resolve beside the input file."""
        self.assertIn('--work-dir', self.source)
        self.assertIn('work_dir = args.work_dir.resolve() if args.work_dir else input_path.parent', self.source)
        self.assertIn('"work_dir": str(work_dir)', self.source)


if __name__ == "__main__":
    unittest.main()
