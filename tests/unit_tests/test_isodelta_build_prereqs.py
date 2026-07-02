"""Unit tests for the IsoDelta-Halo build prerequisite checker.

These tests exercise parser and repository checks without requiring a real
LAMMPS checkout or a local LibTorch installation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is not an import package in SevenNet.
REPO_ROOT = Path(__file__).resolve().parents[2]
PREREQ_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_build_prereqs.py"
SPEC = importlib.util.spec_from_file_location("isodelta_prereqs", PREREQ_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
isodelta_prereqs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = isodelta_prereqs
SPEC.loader.exec_module(isodelta_prereqs)


class IsoDeltaBuildPrereqTest(unittest.TestCase):
    """Check that build prerequisite failures are visible before compilation."""

    def test_parse_lammps_version(self) -> None:
        """The checker should parse the canonical LAMMPS version define."""
        version_text = '#define LAMMPS_VERSION "2 Aug 2023"\n'
        self.assertEqual(
            isodelta_prereqs.parse_lammps_version(version_text),
            "2 Aug 2023",
        )

    def test_pair_source_check_uses_expected_file_list(self) -> None:
        """The current repository should contain every source copied by the patch."""
        results = isodelta_prereqs.check_pair_sources(REPO_ROOT)
        failed = [result for result in results if not result.ok]
        self.assertEqual(failed, [])
        checked_names = {result.name for result in results}
        self.assertIn("pair-source:pair_e3gnn_parallel.cpp", checked_names)
        self.assertIn("pair-source:comm_brick.cpp", checked_names)

    def test_lammps_root_version_mismatch_is_reported(self) -> None:
        """A wrong LAMMPS version should be a visible failed check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cmake").mkdir()
            (root / "src").mkdir()
            (root / "src" / "version.h").write_text(
                '#define LAMMPS_VERSION "1 Jan 2000"\n',
                encoding="utf-8",
            )
            results = isodelta_prereqs.check_lammps_root(root)
        version_results = [result for result in results if result.name == "lammps-version"]
        self.assertEqual(len(version_results), 1)
        self.assertFalse(version_results[0].ok)
        self.assertIn("2 Aug 2023", version_results[0].detail)


if __name__ == "__main__":
    unittest.main()
