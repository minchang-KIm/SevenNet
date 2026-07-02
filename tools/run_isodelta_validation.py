"""Run the complete lightweight validation suite for IsoDelta-Halo changes.

The real LAMMPS/LibTorch build is still required for full runtime validation,
but this script keeps every dependency-free check in one reproducible command.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


# Each command is kept as an argument list so Windows, Linux, and CI shells do
# not reinterpret paths or quoting differently.
REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_COMMANDS = (
    (sys.executable, "tools/run_isodelta_static_checks.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_halo_static.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_benchmark_report_check.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_benchmark_parser.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_build_prereqs.py"),
    (sys.executable, "tests/unit_tests/test_isodelta_lammps_binary_check.py"),
    (
        sys.executable,
        "-m",
        "py_compile",
        "tools/check_isodelta_build_prereqs.py",
        "tools/check_isodelta_benchmark_report.py",
        "tools/check_isodelta_lammps_binary.py",
        "tools/run_isodelta_lammps_benchmark.py",
        "tools/run_isodelta_static_checks.py",
        "tools/run_isodelta_validation.py",
        "tests/unit_tests/test_isodelta_benchmark_report_check.py",
        "tests/unit_tests/test_isodelta_benchmark_parser.py",
        "tests/unit_tests/test_isodelta_build_prereqs.py",
        "tests/unit_tests/test_isodelta_lammps_binary_check.py",
        "tests/unit_tests/test_isodelta_halo_static.py",
    ),
    ("git", "diff", "--check"),
)


def _run(command: tuple[str, ...]) -> None:
    """Run one validation command and stop immediately on the first failure."""
    print(f"[IsoDelta-Halo validation] {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    """Execute all lightweight checks in the same order before every commit."""
    for command in VALIDATION_COMMANDS:
        _run(command)
    print("[IsoDelta-Halo validation] all checks passed")


if __name__ == "__main__":
    main()
