"""Check prerequisites before building LAMMPS with IsoDelta-Halo sources.

The script separates environment problems from source-level problems before a
long LAMMPS/LibTorch build starts. It intentionally uses only the standard
library unless the caller explicitly asks to verify PyTorch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import re
import sys


# These files are copied by sevenn/pair_e3gnn/patch_lammps.sh, so missing any
# one of them means the build cannot represent the current branch.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION_KEY = "report_schema_version"
PREREQ_REPORT_SCHEMA_VERSION = "isodelta-build-prerequisite-report-v1"
GENERATED_REPORT_COMMENT_KEY = "report_comment"
PREREQ_REPORT_COMMENT = (
    "IsoDelta-Halo build prerequisite report recording source-file, LAMMPS tree, "
    "version, and optional torch checks before compiling patched LAMMPS."
)
EXPECTED_LAMMPS_VERSION = "2 Aug 2023"
LAMMPS_VERSION_RE = re.compile(r'#define\s+LAMMPS_VERSION\s+"(?P<version>[^"]+)"')
REQUIRED_PAIR_SOURCE_FILES = (
    "pair_e3gnn.cpp",
    "pair_e3gnn.h",
    "pair_e3gnn_parallel.cpp",
    "pair_e3gnn_parallel.h",
    "pair_e3gnn_oeq_autograd.cpp",
    "comm_brick.cpp",
    "comm_brick.h",
)


@dataclass(frozen=True)
class CheckResult:
    """Represent one prerequisite check in a machine-readable report."""

    name: str
    ok: bool
    detail: str


def parse_lammps_version(version_header_text: str) -> str | None:
    """Extract the LAMMPS version string from src/version.h contents."""
    match = LAMMPS_VERSION_RE.search(version_header_text)
    if match is None:
        return None
    return match.group("version")


def check_pair_sources(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    """Verify that all pair-style sources needed by patch_lammps.sh exist."""
    source_dir = repo_root / "sevenn" / "pair_e3gnn"
    results: list[CheckResult] = []
    for filename in REQUIRED_PAIR_SOURCE_FILES:
        path = source_dir / filename
        results.append(
            CheckResult(
                name=f"pair-source:{filename}",
                ok=path.is_file(),
                detail=str(path),
            )
        )
    return results


def check_lammps_root(lammps_root: Path) -> list[CheckResult]:
    """Verify the directory shape and version expected by the patch script."""
    resolved_root = lammps_root.resolve()
    version_header = resolved_root / "src" / "version.h"
    results = [
        CheckResult("lammps-root", resolved_root.is_dir(), str(resolved_root)),
        CheckResult("lammps-cmake-dir", (resolved_root / "cmake").is_dir(), str(resolved_root / "cmake")),
        CheckResult("lammps-src-dir", (resolved_root / "src").is_dir(), str(resolved_root / "src")),
        CheckResult("lammps-version-header", version_header.is_file(), str(version_header)),
    ]

    if version_header.is_file():
        detected_version = parse_lammps_version(version_header.read_text(encoding="utf-8"))
        results.append(
            CheckResult(
                name="lammps-version",
                ok=detected_version == EXPECTED_LAMMPS_VERSION,
                detail=f"detected={detected_version!r}, expected={EXPECTED_LAMMPS_VERSION!r}",
            )
        )
    return results


def check_torch_import() -> list[CheckResult]:
    """Verify that Python can import torch and report its CMake prefix path."""
    try:
        torch = importlib.import_module("torch")
        prefix_path = getattr(torch.utils, "cmake_prefix_path", "")
        return [
            CheckResult("python-torch-import", True, str(getattr(torch, "__version__", "unknown"))),
            CheckResult("python-torch-cmake-prefix", bool(prefix_path), str(prefix_path)),
        ]
    except Exception as exc:  # pragma: no cover - depends on local environment.
        return [CheckResult("python-torch-import", False, repr(exc))]


def collect_checks(
    lammps_root: Path | None,
    require_torch: bool,
    repo_root: Path = REPO_ROOT,
) -> list[CheckResult]:
    """Collect all requested checks in a stable order."""
    results = check_pair_sources(repo_root)
    if lammps_root is not None:
        results.extend(check_lammps_root(lammps_root))
    if require_torch:
        results.extend(check_torch_import())
    return results


def build_prereq_report(results: list[CheckResult]) -> dict[str, object]:
    """Return the self-describing JSON payload printed by the checker."""
    return {
        REPORT_SCHEMA_VERSION_KEY: PREREQ_REPORT_SCHEMA_VERSION,
        GENERATED_REPORT_COMMENT_KEY: PREREQ_REPORT_COMMENT,
        "ok": all(result.ok for result in results),
        "checks": [asdict(result) for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    """Run prerequisite checks and return nonzero when a required check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lammps-root",
        type=Path,
        help="Optional LAMMPS source root to validate before patching",
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Import torch and report torch.utils.cmake_prefix_path",
    )
    args = parser.parse_args(argv)

    results = collect_checks(args.lammps_root, args.require_torch)
    report = build_prereq_report(results)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
