"""Unit tests for the IsoDelta-Halo LAMMPS binary smoke checker.

The tests avoid requiring a LAMMPS build by checking command construction and
help-text parsing with representative strings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_lammps_binary.py"
SPEC = importlib.util.spec_from_file_location("isodelta_binary_check", CHECK_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_binary_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_binary_check
SPEC.loader.exec_module(isodelta_binary_check)


class IsoDeltaLammpsBinaryCheckTest(unittest.TestCase):
    """Check smoke-test behavior without launching LAMMPS."""

    def test_build_help_command_appends_help_flag(self) -> None:
        """The help flag should be appended after a possibly compound command."""
        command = isodelta_binary_check.build_help_command("mpiexec -n 1 lmp")
        self.assertEqual(command, ["mpiexec", "-n", "1", "lmp", "-h"])

    def test_parse_pair_style_available(self) -> None:
        """The checker should detect the patched pair style in help output."""
        help_text = "Pair styles:\n  eam lj/cut e3gnn/parallel\n"
        self.assertTrue(isodelta_binary_check.parse_pair_style_available(help_text))

    def test_parse_pair_style_missing(self) -> None:
        """Missing pair style registration should be reported as unavailable."""
        help_text = "Pair styles:\n  eam lj/cut\n"
        self.assertFalse(isodelta_binary_check.parse_pair_style_available(help_text))


if __name__ == "__main__":
    unittest.main()
