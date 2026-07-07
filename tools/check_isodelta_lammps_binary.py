"""Smoke-check a LAMMPS binary patched with IsoDelta-Halo pair styles.

The checker runs the binary help command and verifies that `e3gnn/parallel`
appears in the registered style list before longer benchmarks are attempted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import shlex
import subprocess
import sys


# Keep all externally meaningful literals named so the smoke test is easy to
# audit when a LAMMPS build fails on a remote node.
HELP_FLAG = "-h"
PAIR_STYLE_NAME = "e3gnn/parallel"
REPORT_SCHEMA_VERSION_KEY = "report_schema_version"
BINARY_SMOKE_REPORT_SCHEMA_VERSION = "isodelta-lammps-binary-smoke-report-v1"
GENERATED_REPORT_COMMENT_KEY = "report_comment"
BINARY_SMOKE_REPORT_COMMENT = (
    "IsoDelta-Halo LAMMPS binary smoke report recording help-command execution "
    "and patched e3gnn/parallel pair-style registration evidence."
)
SUCCESS_RETURN_CODE = 0
DEFAULT_TIMEOUT_SECONDS = 60.0
MIN_POSITIVE_TIMEOUT_SECONDS = 0.0


@dataclass(frozen=True)
class BinaryCheckResult:
    """Represent one binary smoke-check result in JSON output."""

    name: str
    ok: bool
    detail: str


def _split_lammps_command(lammps_command: str) -> list[str]:
    """Split a user command and reject empty command lines."""
    command_tokens = shlex.split(lammps_command)
    if not command_tokens:
        raise ValueError("lammps_command must not be empty")
    return command_tokens


def validate_binary_check_options(
    lammps_command: str,
    timeout_seconds: float,
) -> None:
    """Reject smoke-check settings that cannot run a meaningful command."""
    _split_lammps_command(lammps_command)
    if not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite")
    if timeout_seconds <= MIN_POSITIVE_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds must be positive")


def build_help_command(lammps_command: str) -> list[str]:
    """Append the LAMMPS help flag without depending on shell quoting."""
    return [*_split_lammps_command(lammps_command), HELP_FLAG]


def parse_pair_style_available(help_text: str) -> bool:
    """Return true when the LAMMPS help text lists the patched pair style."""
    return PAIR_STYLE_NAME in help_text


def check_lammps_binary(
    lammps_command: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[BinaryCheckResult]:
    """Run the LAMMPS help command and check for the parallel pair style."""
    validate_binary_check_options(lammps_command, timeout_seconds)
    command = build_help_command(lammps_command)
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return [BinaryCheckResult("lammps-help-command", False, repr(exc))]
    except subprocess.TimeoutExpired as exc:
        return [BinaryCheckResult("lammps-help-command", False, repr(exc))]

    combined_output = completed.stdout + "\n" + completed.stderr
    return [
        BinaryCheckResult(
            "lammps-help-command",
            completed.returncode == SUCCESS_RETURN_CODE,
            f"returncode={completed.returncode}, command={command}",
        ),
        BinaryCheckResult(
            "pair-style:e3gnn/parallel",
            parse_pair_style_available(combined_output),
            f"searched={PAIR_STYLE_NAME!r}",
        ),
    ]


def build_binary_check_report(results: list[BinaryCheckResult]) -> dict[str, object]:
    """Return the self-describing JSON payload printed by the smoke checker."""
    return {
        REPORT_SCHEMA_VERSION_KEY: BINARY_SMOKE_REPORT_SCHEMA_VERSION,
        GENERATED_REPORT_COMMENT_KEY: BINARY_SMOKE_REPORT_COMMENT,
        "ok": all(result.ok for result in results),
        "checks": [asdict(result) for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the binary smoke check and return nonzero when it fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-command", required=True, help="Example: lmp")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for the help command",
    )
    args = parser.parse_args(argv)
    try:
        validate_binary_check_options(args.lammps_command, args.timeout_seconds)
    except ValueError as exc:
        parser.error(str(exc))

    results = check_lammps_binary(args.lammps_command, args.timeout_seconds)
    report = build_binary_check_report(results)
    print(json.dumps(report, indent=2))
    return SUCCESS_RETURN_CODE if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
