"""Unit tests for the IsoDelta-Halo local goal-readiness audit.

The audit is used as a lightweight completion-evidence gate, so these tests use
small temporary source trees to verify both passing and failing checks without
depending on the full repository layout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


# Load the tool by path because tools/ is intentionally not a Python package.
REPO_ROOT_PARENT_DEPTH = 2
REPO_ROOT = Path(__file__).resolve().parents[REPO_ROOT_PARENT_DEPTH]
GOAL_READINESS_SCRIPT = REPO_ROOT / "tools" / "check_isodelta_goal_readiness.py"
SPEC = importlib.util.spec_from_file_location(
    "isodelta_goal_readiness",
    GOAL_READINESS_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
goal_readiness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal_readiness
SPEC.loader.exec_module(goal_readiness)


class IsoDeltaGoalReadinessTest(unittest.TestCase):
    """Check source-tree readiness reports for required files and comments."""

    def test_report_path_is_optional_to_keep_validation_tree_clean(self) -> None:
        """Default CLI parsing should avoid writing generated reports to git root."""
        parsed = goal_readiness.parse_args([])

        self.assertIsNone(parsed.report_path)

    def test_explicit_report_path_is_preserved_for_audits(self) -> None:
        """An explicit report path should still be available for sync evidence."""
        parsed = goal_readiness.parse_args(["--report-path", "readiness.json"])

        self.assertEqual(parsed.report_path, Path("readiness.json"))

    def test_expected_branch_names_codex_work_branch(self) -> None:
        """The local completion audit should document the active work branch."""
        self.assertEqual(
            goal_readiness.EXPECTED_BRANCH,
            "codex/isodelta-halo-runtime",
        )

    def test_goal_readiness_report_passes_for_required_snippets_and_prefixes(self) -> None:
        """A source tree with required snippets and comments should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = root / "tools" / "script.py"
            doc_path = root / "docs" / "guide.md"
            script_path.parent.mkdir(parents=True)
            doc_path.parent.mkdir(parents=True)
            script_path.write_text('"""Tool comment."""\nFEATURE = "ready"\n', encoding="utf-8")
            doc_path.write_text("<!-- guide comment -->\nready docs\n", encoding="utf-8")

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={
                    "tools/script.py": ("FEATURE", "ready"),
                    "docs/guide.md": ("ready docs",),
                },
                forbidden_file_snippets={},
                comment_prefix_requirements={
                    "tools/script.py": '"""',
                    "docs/guide.md": "<!--",
                },
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_PASSED)
        self.assertTrue(all(record["passed"] for record in report["checks"]))

    def test_goal_readiness_report_fails_for_missing_snippet(self) -> None:
        """A missing snippet should make the readiness report fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = root / "tools" / "script.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text('"""Tool comment."""\n', encoding="utf-8")

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={"tools/script.py": ("missing-feature",)},
                forbidden_file_snippets={},
                comment_prefix_requirements={"tools/script.py": '"""'},
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_FAILED)
        self.assertFalse(all(record["passed"] for record in report["checks"]))

    def test_goal_readiness_accepts_isodelta_python_headers(self) -> None:
        """Every IsoDelta Python file with an explanatory header should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool_path = root / "tools" / "run_isodelta_example.py"
            test_path = root / "tests" / "unit_tests" / "test_isodelta_example.py"
            tool_path.parent.mkdir(parents=True)
            test_path.parent.mkdir(parents=True)
            tool_path.write_text('"""Example tool header."""\nVALUE = 1\n', encoding="utf-8")
            test_path.write_text('"""Example test header."""\nVALUE = 1\n', encoding="utf-8")

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={},
                forbidden_file_snippets={},
                comment_prefix_requirements={},
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_PASSED)
        header_checks = [
            record
            for record in report["checks"]
            if record["name"].startswith("isodelta_python_header:")
        ]
        self.assertEqual(len(header_checks), 2)
        self.assertTrue(all(record["passed"] for record in header_checks))

    def test_goal_readiness_rejects_isodelta_python_without_header(self) -> None:
        """A new IsoDelta Python file must not skip its file-level comment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool_path = root / "tools" / "run_isodelta_missing_header.py"
            tool_path.parent.mkdir(parents=True)
            tool_path.write_text("VALUE = 1\n", encoding="utf-8")

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={},
                forbidden_file_snippets={},
                comment_prefix_requirements={},
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_FAILED)
        failed_checks = [
            record["name"] for record in report["checks"] if not record["passed"]
        ]
        self.assertIn(
            "isodelta_python_header:tools/run_isodelta_missing_header.py",
            failed_checks,
        )

    def test_goal_readiness_rejects_forbidden_production_marker(self) -> None:
        """Production IsoDelta files should not carry temporary-work markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool_path = root / "tools" / "run_isodelta_marker.py"
            tool_path.parent.mkdir(parents=True)
            tool_path.write_text(
                '"""Example tool header."""\n# TODO remove before paper run\nVALUE = 1\n',
                encoding="utf-8",
            )

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={},
                forbidden_file_snippets={},
                comment_prefix_requirements={},
                implementation_marker_glob_patterns=("tools/*isodelta*.py",),
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_FAILED)
        failed_checks = [
            record
            for record in report["checks"]
            if record["name"]
            == "forbidden_implementation_marker:tools/run_isodelta_marker.py"
        ]
        self.assertEqual(len(failed_checks), 1)
        self.assertIn("TODO", failed_checks[0]["detail"])

    def test_goal_readiness_ignores_marker_string_literals(self) -> None:
        """Static check tools may mention marker text as data without failing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool_path = root / "tools" / "run_isodelta_marker_check.py"
            tool_path.parent.mkdir(parents=True)
            tool_path.write_text(
                '"""Example tool header."""\nMARKER = "TODO"\nVALUE = 1\n',
                encoding="utf-8",
            )

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={},
                forbidden_file_snippets={},
                comment_prefix_requirements={},
                implementation_marker_glob_patterns=("tools/*isodelta*.py",),
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_PASSED)
        marker_checks = [
            record
            for record in report["checks"]
            if record["name"]
            == "forbidden_implementation_marker:tools/run_isodelta_marker_check.py"
        ]
        self.assertEqual(len(marker_checks), 1)
        self.assertTrue(marker_checks[0]["passed"])

    def test_goal_readiness_rejects_forbidden_file_snippet(self) -> None:
        """Known magic-number snippets should not return to production files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "sevenn" / "pair_e3gnn" / "comm_brick.cpp"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("int all[6];\n", encoding="utf-8")

            report = goal_readiness.build_goal_readiness_report(
                root=root,
                required_file_snippets={},
                forbidden_file_snippets={
                    "sevenn/pair_e3gnn/comm_brick.cpp": ("int all[6]",),
                },
                comment_prefix_requirements={},
                isodelta_python_glob_patterns=(),
                implementation_marker_glob_patterns=(),
            )

        self.assertEqual(report["status"], goal_readiness.STATUS_FAILED)
        failed_checks = [
            record
            for record in report["checks"]
            if record["name"] == "forbidden_file:sevenn/pair_e3gnn/comm_brick.cpp"
        ]
        self.assertEqual(len(failed_checks), 1)
        self.assertIn("int all[6]", failed_checks[0]["detail"])


if __name__ == "__main__":
    unittest.main()
