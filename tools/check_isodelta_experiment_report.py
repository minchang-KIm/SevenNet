"""Verify an IsoDelta-Halo experiment driver report and its log fingerprints.

The experiment driver writes command stdout/stderr logs plus a top-level JSON
report. This checker reopens that report after archiving or transfer and
confirms that every recorded log fingerprint still matches the filesystem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import run_isodelta_experiment as experiment_driver


# Report field names and schema strings are constants so the checker and tests
# can detect accidental contract drift.
EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION = "isodelta-experiment-report-check-v1"
GENERATED_REPORT_COMMENT_KEY = "report_comment"
EXPERIMENT_REPORT_CHECK_COMMENT = (
    "IsoDelta-Halo experiment report verification evidence recording driver "
    "report schema, comment, command log fingerprints, and command-result checks."
)
EXPERIMENT_REPORT_SCHEMA_VERSION_KEY = "report_schema_version"
EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION_KEY = "experiment_report_check_schema_version"
STATUS_KEY = "status"
PASSED_STATUS = "passed"
FAILED_STATUS = "failed"
PROVENANCE_KEY = "provenance"
COMMANDS_KEY = "commands"
ARGV_KEY = "argv"
RETURNCODE_KEY = "returncode"
STDOUT_PATH_KEY = "stdout_path"
STDERR_PATH_KEY = "stderr_path"
STDOUT_FINGERPRINT_KEY = "stdout_fingerprint"
STDERR_FINGERPRINT_KEY = "stderr_fingerprint"
EXISTS_KEY = "exists"
ALGORITHM_KEY = "algorithm"
SHA256_KEY = "sha256"
BYTE_SIZE_KEY = "byte_size"
COMMAND_LOG_SIDECAR_FINGERPRINT_KEY = experiment_driver.COMMAND_LOG_SIDECAR_FINGERPRINT_KEY
COMMAND_LOG_SIDECAR_PATH_KEY = experiment_driver.COMMAND_LOG_SIDECAR_PATH_KEY
STREAMS_PER_COMMAND = 2
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1


def _require(condition: bool, message: str) -> None:
    """Raise a validation error with a precise report location."""
    if not condition:
        raise ValueError(message)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    """Require a JSON object and return it with a typed shape."""
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _as_sequence(value: Any, label: str) -> list[Any]:
    """Require a JSON array and return it with a typed shape."""
    _require(isinstance(value, list), f"{label} must be an array")
    return value


def _as_string(value: Any, label: str) -> str:
    """Require a non-empty string field."""
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value


def _as_int(value: Any, label: str) -> int:
    """Require an integer without accepting JSON booleans as counts."""
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return value


def _as_bool(value: Any, label: str) -> bool:
    """Require a JSON boolean field."""
    _require(isinstance(value, bool), f"{label} must be a boolean")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a report file and require the top-level value to be an object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _as_mapping(payload, str(path))


def _resolve_log_path(path_text: str) -> Path:
    """Resolve a recorded log path the same way the experiment driver writes it."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return experiment_driver.REPO_ROOT / path


def _check_report_comment(report: dict[str, Any]) -> str:
    """Require the experiment report to describe its evidence purpose."""
    report_comment = _as_string(
        report.get(GENERATED_REPORT_COMMENT_KEY),
        GENERATED_REPORT_COMMENT_KEY,
    )
    _require(
        report_comment == experiment_driver.EXPERIMENT_REPORT_COMMENT,
        f"{GENERATED_REPORT_COMMENT_KEY} must describe IsoDelta-Halo experiment evidence",
    )
    return report_comment


def _check_provenance(report: dict[str, Any]) -> str:
    """Require the driver report schema version inside provenance."""
    provenance = _as_mapping(report.get(PROVENANCE_KEY), PROVENANCE_KEY)
    schema_version = _as_string(
        provenance.get(EXPERIMENT_REPORT_SCHEMA_VERSION_KEY),
        f"{PROVENANCE_KEY}.{EXPERIMENT_REPORT_SCHEMA_VERSION_KEY}",
    )
    _require(
        schema_version == experiment_driver.EXPERIMENT_REPORT_SCHEMA_VERSION,
        (
            f"{PROVENANCE_KEY}.{EXPERIMENT_REPORT_SCHEMA_VERSION_KEY} must be "
            f"{experiment_driver.EXPERIMENT_REPORT_SCHEMA_VERSION}"
        ),
    )
    return schema_version


def _check_fingerprint(
    command: dict[str, Any],
    path_key: str,
    fingerprint_key: str,
    label: str,
    *,
    command_name: str,
    stream_name: str,
) -> None:
    """Require a command log fingerprint to match the current filesystem."""
    path_text = _as_string(command.get(path_key), f"{label}.{path_key}")
    expected = _as_mapping(command.get(fingerprint_key), f"{label}.{fingerprint_key}")
    recorded_log_path = Path(path_text)
    log_path = _resolve_log_path(path_text)
    actual = experiment_driver.command_log_fingerprint(
        log_path,
        command_name=command_name,
        stream_name=stream_name,
    )
    for key in (EXISTS_KEY, ALGORITHM_KEY, SHA256_KEY, BYTE_SIZE_KEY):
        if key == EXISTS_KEY:
            expected_value = _as_bool(expected.get(key), f"{label}.{fingerprint_key}.{key}")
        elif key == BYTE_SIZE_KEY:
            expected_value = _as_int(expected.get(key), f"{label}.{fingerprint_key}.{key}")
        else:
            expected_value = _as_string(expected.get(key), f"{label}.{fingerprint_key}.{key}")
        _require(
            expected_value == actual[key],
            f"{label}.{fingerprint_key}.{key} does not match {path_text}",
        )
    expected_sidecar_path = _as_string(
        expected.get(COMMAND_LOG_SIDECAR_PATH_KEY),
        f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_PATH_KEY}",
    )
    actual_sidecar_path = str(experiment_driver.command_log_sidecar_path(recorded_log_path))
    _require(
        expected_sidecar_path == actual_sidecar_path,
        f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_PATH_KEY} does not match {path_text}",
    )
    expected_sidecar = _as_mapping(
        expected.get(COMMAND_LOG_SIDECAR_FINGERPRINT_KEY),
        f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_FINGERPRINT_KEY}",
    )
    actual_sidecar = experiment_driver.file_fingerprint(_resolve_log_path(expected_sidecar_path))
    for key in (EXISTS_KEY, ALGORITHM_KEY, SHA256_KEY, BYTE_SIZE_KEY):
        if key == EXISTS_KEY:
            expected_value = _as_bool(
                expected_sidecar.get(key),
                f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_FINGERPRINT_KEY}.{key}",
            )
        elif key == BYTE_SIZE_KEY:
            expected_value = _as_int(
                expected_sidecar.get(key),
                f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_FINGERPRINT_KEY}.{key}",
            )
        else:
            expected_value = _as_string(
                expected_sidecar.get(key),
                f"{label}.{fingerprint_key}.{COMMAND_LOG_SIDECAR_FINGERPRINT_KEY}.{key}",
            )
        _require(
            expected_value == actual_sidecar[key],
            (
                f"{label}.{fingerprint_key}."
                f"{COMMAND_LOG_SIDECAR_FINGERPRINT_KEY}.{key} does not match "
                f"{expected_sidecar_path}"
            ),
        )
    _check_command_log_sidecar(
        sidecar_path=_resolve_log_path(expected_sidecar_path),
        expected_log_path=path_text,
        expected_log_fingerprint=expected,
        command_name=command_name,
        stream_name=stream_name,
        label=f"{label}.{fingerprint_key}",
    )


def _check_command_log_sidecar(
    *,
    sidecar_path: Path,
    expected_log_path: str,
    expected_log_fingerprint: dict[str, Any],
    command_name: str,
    stream_name: str,
    label: str,
) -> None:
    """Require a sidecar JSON file to describe the matching raw log stream."""
    payload = _load_json_object(sidecar_path)
    schema_version = _as_string(
        payload.get("experiment_log_sidecar_schema_version"),
        f"{label}.sidecar.experiment_log_sidecar_schema_version",
    )
    _require(
        schema_version == experiment_driver.EXPERIMENT_LOG_SIDECAR_SCHEMA_VERSION,
        f"{label}.sidecar schema version does not match",
    )
    report_comment = _as_string(
        payload.get(GENERATED_REPORT_COMMENT_KEY),
        f"{label}.sidecar.{GENERATED_REPORT_COMMENT_KEY}",
    )
    _require(
        report_comment == experiment_driver.EXPERIMENT_LOG_SIDECAR_COMMENT,
        f"{label}.sidecar.{GENERATED_REPORT_COMMENT_KEY} must describe command-log evidence",
    )
    _require(
        _as_string(payload.get("command_name"), f"{label}.sidecar.command_name")
        == command_name,
        f"{label}.sidecar.command_name does not match",
    )
    _require(
        _as_string(payload.get("stream"), f"{label}.sidecar.stream") == stream_name,
        f"{label}.sidecar.stream does not match",
    )
    _require(
        _as_string(payload.get("log_path"), f"{label}.sidecar.log_path")
        == expected_log_path,
        f"{label}.sidecar.log_path does not match",
    )
    sidecar_log_fingerprint = _as_mapping(
        payload.get("log_fingerprint"),
        f"{label}.sidecar.log_fingerprint",
    )
    for key in (EXISTS_KEY, ALGORITHM_KEY, SHA256_KEY, BYTE_SIZE_KEY):
        _require(
            sidecar_log_fingerprint.get(key) == expected_log_fingerprint.get(key),
            f"{label}.sidecar.log_fingerprint.{key} does not match",
        )


def _check_command(command: Any, index: int) -> None:
    """Validate one experiment command record."""
    label = f"{COMMANDS_KEY}[{index}]"
    command_record = _as_mapping(command, label)
    command_name = _as_string(command_record.get("name"), f"{label}.name")
    _as_sequence(command_record.get(ARGV_KEY), f"{label}.{ARGV_KEY}")
    _as_int(command_record.get(RETURNCODE_KEY), f"{label}.{RETURNCODE_KEY}")
    _check_fingerprint(
        command_record,
        STDOUT_PATH_KEY,
        STDOUT_FINGERPRINT_KEY,
        label,
        command_name=command_name,
        stream_name="stdout",
    )
    _check_fingerprint(
        command_record,
        STDERR_PATH_KEY,
        STDERR_FINGERPRINT_KEY,
        label,
        command_name=command_name,
        stream_name="stderr",
    )


def validate_experiment_report(report_path: Path) -> dict[str, object]:
    """Validate an experiment report and return a compact evidence summary."""
    report = _load_json_object(report_path)
    report_comment = _check_report_comment(report)
    schema_version = _check_provenance(report)
    commands = _as_sequence(report.get(COMMANDS_KEY), COMMANDS_KEY)
    _require(commands, f"{COMMANDS_KEY} must not be empty")
    for index, command in enumerate(commands):
        _check_command(command, index)
    return {
        EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION_KEY: EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION,
        GENERATED_REPORT_COMMENT_KEY: EXPERIMENT_REPORT_CHECK_COMMENT,
        STATUS_KEY: PASSED_STATUS,
        "experiment_report": str(report_path),
        "experiment_report_comment": report_comment,
        "experiment_report_schema_version": schema_version,
        "checked_command_count": len(commands),
        "checked_log_fingerprint_count": len(commands) * STREAMS_PER_COMMAND,
    }


def _failure_summary(report_path: Path, detail: str) -> dict[str, object]:
    """Build a machine-readable failure report for CLI callers."""
    return {
        EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION_KEY: EXPERIMENT_REPORT_CHECK_SCHEMA_VERSION,
        GENERATED_REPORT_COMMENT_KEY: EXPERIMENT_REPORT_CHECK_COMMENT,
        STATUS_KEY: FAILED_STATUS,
        "experiment_report": str(report_path),
        "detail": detail,
    }


def _write_evidence(output_path: Path, evidence: dict[str, object]) -> None:
    """Write verification evidence as a self-describing JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for report verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Experiment report JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON evidence file written after verification",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify one experiment report and print JSON evidence."""
    args = parse_args(argv)
    try:
        evidence = validate_experiment_report(args.report)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        evidence = _failure_summary(args.report, str(exc))
        if args.output is not None:
            _write_evidence(args.output, evidence)
        print(json.dumps(evidence, indent=2))
        return FAILURE_RETURN_CODE
    if args.output is not None:
        _write_evidence(args.output, evidence)
    print(json.dumps(evidence, indent=2))
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
