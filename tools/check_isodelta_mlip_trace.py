"""Evaluate model-agnostic MLIP traces for IsoDelta-Halo applicability.

The SevenNet implementation proves the runtime path, but paper reviewers also
need evidence that the idea is not tied to one model class. This checker reads
a compact JSON trace containing graph and halo metadata from any distributed
MLIP runtime and applies the same conservative reuse rules used by the C++
cache.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


# JSON field names are centralized so the schema stays stable across trace
# exporters written for SevenNet, NequIP, MACE, Allegro, or another MLIP.
MODEL_KEY = "model"
SCHEMA_VERSION_KEY = "schema_version"
STEPS_KEY = "steps"
STEP_ID_KEY = "step"
NEIGHBOR_REBUILT_KEY = "neighbor_list_rebuilt"
NODE_TAGS_KEY = "graph_node_tags"
EDGE_COUNT_KEY = "edge_count"
COMM_PHASES_KEY = "comm_phases"
SEND_RANK_KEY = "send_rank"
RECV_RANK_KEY = "recv_rank"
SEND_COUNT_KEY = "send_count"
RECV_COUNT_KEY = "recv_count"
FIRST_RECV_KEY = "first_recv"
SEND_TAGS_KEY = "send_tags"
RECV_TAGS_KEY = "recv_tags"
STEP_TIME_SECONDS_KEY = "step_time_seconds"
METADATA_BUILD_SECONDS_KEY = "metadata_build_time_seconds"
TRACE_SCHEMA_VERSION = "1.0"
SCHEMA_REQUIRED_STEP_FIELDS = (
    NEIGHBOR_REBUILT_KEY,
    NODE_TAGS_KEY,
    EDGE_COUNT_KEY,
    COMM_PHASES_KEY,
)
SCHEMA_REQUIRED_PHASE_FIELDS = (
    SEND_RANK_KEY,
    RECV_RANK_KEY,
    SEND_COUNT_KEY,
    RECV_COUNT_KEY,
    FIRST_RECV_KEY,
    SEND_TAGS_KEY,
    RECV_TAGS_KEY,
)
SCHEMA_OPTIONAL_STEP_FIELDS = (
    STEP_ID_KEY,
    STEP_TIME_SECONDS_KEY,
    METADATA_BUILD_SECONDS_KEY,
)

MISS_DISABLED = "disabled"
MISS_NO_CACHE = "no-cache"
MISS_NEIGHBOR_REBUILT = "neighbor-list-rebuilt"
MISS_SHAPE_CHANGED = "shape-changed"
MISS_TAG_COUNT_CHANGED = "tag-count-changed"
MISS_TAG_ORDER_CHANGED = "tag-order-changed"
MISS_COMM_TOPOLOGY_CHANGED = "comm-topology-changed"
MISS_COMM_LIST_TAG_ORDER_CHANGED = "comm-list-tag-order-changed"
HIT_DECISION = "hit"
MISS_REASON_PREFIX = "miss_"
ATTEMPTS_KEY = "attempts"
HITS_KEY = "hits"
HIT_RATE_PERCENT_KEY = "hit_rate_percent"
MISS_BREAKDOWN_KEY = "miss_breakdown"
TRACE_COUNT_RESIDUAL_KEY = "trace_count_residual"
MODEL_AGNOSTIC_REQUIREMENTS_KEY = "model_agnostic_requirements"
TIMING_KEY = "timing"
TIMING_BASELINE_SECONDS_KEY = "baseline_step_time_seconds"
TIMING_METADATA_SECONDS_KEY = "metadata_build_time_seconds"
TIMING_METADATA_FRACTION_KEY = "metadata_fraction_percent"
TIMING_LOOKUP_OVERHEAD_SECONDS_KEY = "cache_lookup_overhead_seconds"
TIMING_AVERAGE_ENABLED_SECONDS_KEY = "estimated_average_enabled_seconds"
TIMING_WORST_CASE_ENABLED_SECONDS_KEY = "estimated_worst_case_enabled_seconds"
TIMING_AVERAGE_SPEEDUP_KEY = "estimated_average_speedup"
TIMING_WORST_CASE_SPEEDUP_KEY = "estimated_worst_case_speedup"
REQUIRED_MODEL_AGNOSTIC_REQUIREMENTS = (
    "uses_ordered_graph_node_tags",
    "uses_edge_count_shape_guard",
    "uses_neighbor_rebuild_guard",
    "uses_comm_topology_guard",
    "uses_comm_list_tag_order_guard",
)
MISS_REASONS = (
    MISS_DISABLED,
    MISS_NO_CACHE,
    MISS_NEIGHBOR_REBUILT,
    MISS_SHAPE_CHANGED,
    MISS_TAG_COUNT_CHANGED,
    MISS_TAG_ORDER_CHANGED,
    MISS_COMM_TOPOLOGY_CHANGED,
    MISS_COMM_LIST_TAG_ORDER_CHANGED,
)

PERCENT_SCALE = 100.0
IDENTITY_SPEEDUP = 1.0
DEFAULT_CACHE_LOOKUP_OVERHEAD_SECONDS = 0.0
DEFAULT_MIN_HIT_RATE_PERCENT = 0.0
DEFAULT_MIN_METADATA_FRACTION_PERCENT = 0.0
MIN_NONNEGATIVE_VALUE = 0.0
MIN_REQUIRED_TRACE_ATTEMPTS = 1.0
MIN_PERCENT_VALUE = 0.0
MAX_PERCENT_VALUE = PERCENT_SCALE
MIN_POSITIVE_SPEEDUP = 0.0
TRACE_COUNT_TOLERANCE = 1.0e-9
TIMING_ABSOLUTE_TOLERANCE_SECONDS = 1.0e-12
TIMING_RELATIVE_TOLERANCE = 1.0e-9
TIMING_PERCENT_TOLERANCE = 1.0e-9


class TraceCheckError(ValueError):
    """Raised when a trace is invalid or fails the requested evidence gates."""


JsonTag = int | str


@dataclass(frozen=True)
class CommPhaseSignature:
    """Store one halo communication phase in model-independent form."""

    send_rank: int
    recv_rank: int
    send_count: int
    recv_count: int
    first_recv: int
    send_tags: tuple[JsonTag, ...]
    recv_tags: tuple[JsonTag, ...]

    @property
    def topology(self) -> tuple[int, int, int, int, int]:
        """Return the routing fields that must match before reuse."""
        return (
            self.send_rank,
            self.recv_rank,
            self.send_count,
            self.recv_count,
            self.first_recv,
        )

    @property
    def list_tag_order(self) -> tuple[tuple[JsonTag, ...], tuple[JsonTag, ...]]:
        """Return the phase-local send and receive tag order."""
        return (self.send_tags, self.recv_tags)


@dataclass(frozen=True)
class StepSignature:
    """Store the graph and halo metadata that define a reusable step."""

    graph_node_tags: tuple[JsonTag, ...]
    edge_count: int
    comm_phases: tuple[CommPhaseSignature, ...]

    @property
    def comm_topology(self) -> tuple[tuple[int, int, int, int, int], ...]:
        """Return the communication topology without atom-list tag order."""
        return tuple(phase.topology for phase in self.comm_phases)

    @property
    def comm_list_tag_order(
        self,
    ) -> tuple[tuple[tuple[JsonTag, ...], tuple[JsonTag, ...]], ...]:
        """Return the sendlist and recv-segment tag order for every phase."""
        return tuple(phase.list_tag_order for phase in self.comm_phases)


@dataclass(frozen=True)
class StepRecord:
    """Store one parsed trace step and optional timing evidence."""

    step_id: int | str
    neighbor_list_rebuilt: bool
    signature: StepSignature
    step_time_seconds: float | None
    metadata_build_time_seconds: float | None


@dataclass(frozen=True)
class TraceThresholds:
    """Store gates used to decide whether a trace supports a paper claim."""

    min_hit_rate_percent: float = DEFAULT_MIN_HIT_RATE_PERCENT
    min_estimated_speedup: float | None = None
    min_metadata_fraction_percent: float = DEFAULT_MIN_METADATA_FRACTION_PERCENT


def load_trace(path: Path) -> dict[str, Any]:
    """Read one MLIP halo trace JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "trace root must be a JSON object")
    return payload


def trace_schema() -> dict[str, Any]:
    """Return a compact schema description for portable MLIP trace exporters."""
    return {
        SCHEMA_VERSION_KEY: TRACE_SCHEMA_VERSION,
        "root": {
            MODEL_KEY: "optional string label such as SevenNet, NequIP, MACE, or Allegro",
            STEPS_KEY: "non-empty array of per-MD-step metadata records",
        },
        "required_step_fields": list(SCHEMA_REQUIRED_STEP_FIELDS),
        "optional_step_fields": list(SCHEMA_OPTIONAL_STEP_FIELDS),
        "required_comm_phase_fields": list(SCHEMA_REQUIRED_PHASE_FIELDS),
        "field_types": {
            STEP_ID_KEY: "integer or string",
            NEIGHBOR_REBUILT_KEY: "boolean",
            NODE_TAGS_KEY: "ordered array of integer or string atom tags",
            EDGE_COUNT_KEY: "nonnegative integer",
            COMM_PHASES_KEY: "array of communication phase objects",
            SEND_RANK_KEY: "nonnegative integer MPI rank",
            RECV_RANK_KEY: "nonnegative integer MPI rank",
            SEND_COUNT_KEY: "nonnegative integer equal to len(send_tags)",
            RECV_COUNT_KEY: "nonnegative integer equal to len(recv_tags)",
            FIRST_RECV_KEY: "nonnegative integer receive segment offset",
            SEND_TAGS_KEY: "ordered array of integer or string atom tags",
            RECV_TAGS_KEY: "ordered array of integer or string atom tags",
            STEP_TIME_SECONDS_KEY: "optional nonnegative baseline step time",
            METADATA_BUILD_SECONDS_KEY: "optional nonnegative metadata build time",
        },
        "reuse_guards": [
            "neighbor-list rebuild state",
            "graph node tag count",
            "edge count",
            "graph node tag order",
            "communication topology",
            "phase-local send and receive tag order",
        ],
        "timing_estimates": [
            TIMING_METADATA_FRACTION_KEY,
            TIMING_AVERAGE_SPEEDUP_KEY,
            TIMING_WORST_CASE_SPEEDUP_KEY,
        ],
    }


def _require(condition: bool, message: str) -> None:
    """Raise a compact checker error when a schema or evidence rule fails."""
    if not condition:
        raise TraceCheckError(message)


def _validate_percent(value: float, field_name: str) -> None:
    """Require a percentage threshold to stay in the physical range."""
    _require(
        MIN_PERCENT_VALUE <= value <= MAX_PERCENT_VALUE,
        f"{field_name} must be between {MIN_PERCENT_VALUE:g} and {MAX_PERCENT_VALUE:g}",
    )


def validate_thresholds(thresholds: TraceThresholds) -> None:
    """Reject trace acceptance criteria that cannot support a claim."""
    _validate_percent(thresholds.min_hit_rate_percent, "min_hit_rate_percent")
    _validate_percent(
        thresholds.min_metadata_fraction_percent,
        "min_metadata_fraction_percent",
    )
    if thresholds.min_estimated_speedup is not None:
        _require(
            thresholds.min_estimated_speedup > MIN_POSITIVE_SPEEDUP,
            "min_estimated_speedup must be positive when provided",
        )


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Return a JSON object field with a descriptive error on mismatch."""
    _require(isinstance(value, dict), f"{field_name} must be a JSON object")
    return value


def _as_sequence(value: Any, field_name: str) -> list[Any]:
    """Return a JSON array field with a descriptive error on mismatch."""
    _require(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _as_bool(value: Any, field_name: str) -> bool:
    """Return a JSON boolean field without treating numbers as booleans."""
    _require(isinstance(value, bool), f"{field_name} must be a boolean")
    return value


def _as_number(value: Any, field_name: str) -> float:
    """Return a numeric JSON field without accepting booleans."""
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    numeric_value = float(value)
    _require(math.isfinite(numeric_value), f"{field_name} must be finite")
    return numeric_value


def _as_nonnegative_int(value: Any, field_name: str) -> int:
    """Return a nonnegative integer field for counts, ranks, and offsets."""
    numeric_value = _as_number(value, field_name)
    int_value = int(numeric_value)
    _require(
        numeric_value == int_value and numeric_value >= MIN_NONNEGATIVE_VALUE,
        f"{field_name} must be a nonnegative integer",
    )
    return int_value


def _as_optional_timing_number(value: Any, field_name: str) -> float | None:
    """Return an optional timing evidence number."""
    if value is None:
        return None
    return _as_number(value, field_name)


def _as_optional_nonnegative_timing_number(
    value: Any,
    field_name: str,
) -> float | None:
    """Return an optional timing evidence number that cannot be negative."""
    numeric_value = _as_optional_timing_number(value, field_name)
    if numeric_value is None:
        return None
    _require(
        numeric_value >= MIN_NONNEGATIVE_VALUE,
        f"{field_name} must be nonnegative",
    )
    return numeric_value


def _is_close(
    observed: float,
    expected: float,
    absolute_tolerance: float = TIMING_ABSOLUTE_TOLERANCE_SECONDS,
    relative_tolerance: float = TIMING_RELATIVE_TOLERANCE,
) -> bool:
    """Return whether two timing evidence values agree within named tolerance."""
    tolerance = max(
        absolute_tolerance,
        relative_tolerance * max(abs(observed), abs(expected)),
    )
    return abs(observed - expected) <= tolerance


def _as_optional_nonnegative_number(
    mapping: dict[str, Any],
    key: str,
    field_name: str,
) -> float | None:
    """Return an optional nonnegative timing field."""
    if key not in mapping:
        return None
    value = _as_number(mapping[key], field_name)
    _require(value >= MIN_NONNEGATIVE_VALUE, f"{field_name} must be nonnegative")
    return value


def _as_tag(value: Any, field_name: str) -> JsonTag:
    """Return a stable atom tag representation used for ordering checks."""
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, str),
        f"{field_name} must be an integer or string tag",
    )
    return value


def _as_tag_tuple(value: Any, field_name: str) -> tuple[JsonTag, ...]:
    """Return an immutable tag sequence so exact order comparisons are cheap."""
    return tuple(
        _as_tag(tag, f"{field_name}[{index}]")
        for index, tag in enumerate(_as_sequence(value, field_name))
    )


def _parse_phase(value: Any, field_name: str) -> CommPhaseSignature:
    """Parse one communication phase and verify count fields match tag arrays."""
    phase = _as_mapping(value, field_name)
    send_tags = _as_tag_tuple(phase.get(SEND_TAGS_KEY), f"{field_name}.{SEND_TAGS_KEY}")
    recv_tags = _as_tag_tuple(phase.get(RECV_TAGS_KEY), f"{field_name}.{RECV_TAGS_KEY}")
    send_count = _as_nonnegative_int(
        phase.get(SEND_COUNT_KEY),
        f"{field_name}.{SEND_COUNT_KEY}",
    )
    recv_count = _as_nonnegative_int(
        phase.get(RECV_COUNT_KEY),
        f"{field_name}.{RECV_COUNT_KEY}",
    )
    _require(
        send_count == len(send_tags),
        f"{field_name}.{SEND_COUNT_KEY} must match send tag count",
    )
    _require(
        recv_count == len(recv_tags),
        f"{field_name}.{RECV_COUNT_KEY} must match receive tag count",
    )
    return CommPhaseSignature(
        send_rank=_as_nonnegative_int(
            phase.get(SEND_RANK_KEY),
            f"{field_name}.{SEND_RANK_KEY}",
        ),
        recv_rank=_as_nonnegative_int(
            phase.get(RECV_RANK_KEY),
            f"{field_name}.{RECV_RANK_KEY}",
        ),
        send_count=send_count,
        recv_count=recv_count,
        first_recv=_as_nonnegative_int(
            phase.get(FIRST_RECV_KEY),
            f"{field_name}.{FIRST_RECV_KEY}",
        ),
        send_tags=send_tags,
        recv_tags=recv_tags,
    )


def _parse_step(value: Any, field_name: str) -> StepRecord:
    """Parse one trace step into the signature used by the reuse decision."""
    step = _as_mapping(value, field_name)
    comm_phases = tuple(
        _parse_phase(phase, f"{field_name}.{COMM_PHASES_KEY}[{index}]")
        for index, phase in enumerate(_as_sequence(step.get(COMM_PHASES_KEY), f"{field_name}.{COMM_PHASES_KEY}"))
    )
    return StepRecord(
        step_id=step.get(STEP_ID_KEY, field_name),
        neighbor_list_rebuilt=_as_bool(
            step.get(NEIGHBOR_REBUILT_KEY),
            f"{field_name}.{NEIGHBOR_REBUILT_KEY}",
        ),
        signature=StepSignature(
            graph_node_tags=_as_tag_tuple(
                step.get(NODE_TAGS_KEY),
                f"{field_name}.{NODE_TAGS_KEY}",
            ),
            edge_count=_as_nonnegative_int(
                step.get(EDGE_COUNT_KEY),
                f"{field_name}.{EDGE_COUNT_KEY}",
            ),
            comm_phases=comm_phases,
        ),
        step_time_seconds=_as_optional_nonnegative_number(
            step,
            STEP_TIME_SECONDS_KEY,
            f"{field_name}.{STEP_TIME_SECONDS_KEY}",
        ),
        metadata_build_time_seconds=_as_optional_nonnegative_number(
            step,
            METADATA_BUILD_SECONDS_KEY,
            f"{field_name}.{METADATA_BUILD_SECONDS_KEY}",
        ),
    )


def _classify_reuse(
    previous_signature: StepSignature | None,
    current_step: StepRecord,
    cache_disabled: bool,
) -> str | None:
    """Return a miss reason or None when metadata reuse is safe."""
    if cache_disabled:
        return MISS_DISABLED
    if previous_signature is None:
        return MISS_NO_CACHE
    current_signature = current_step.signature
    if current_step.neighbor_list_rebuilt:
        return MISS_NEIGHBOR_REBUILT
    if len(current_signature.graph_node_tags) != len(previous_signature.graph_node_tags):
        return MISS_TAG_COUNT_CHANGED
    if current_signature.edge_count != previous_signature.edge_count:
        return MISS_SHAPE_CHANGED
    if current_signature.graph_node_tags != previous_signature.graph_node_tags:
        return MISS_TAG_ORDER_CHANGED
    if current_signature.comm_topology != previous_signature.comm_topology:
        return MISS_COMM_TOPOLOGY_CHANGED
    if current_signature.comm_list_tag_order != previous_signature.comm_list_tag_order:
        return MISS_COMM_LIST_TAG_ORDER_CHANGED
    return None


def _speedup_or_none(baseline_seconds: float, enabled_seconds: float) -> float | None:
    """Return a speedup when the estimated enabled runtime is physically valid."""
    if enabled_seconds <= MIN_NONNEGATIVE_VALUE:
        return None
    return baseline_seconds / enabled_seconds


def _timing_summary(
    records: list[StepRecord],
    hit_flags: list[bool],
    cache_lookup_overhead_seconds: float,
) -> dict[str, float | None]:
    """Estimate lower and average speedups from optional trace timings."""
    step_times = [record.step_time_seconds for record in records]
    metadata_times = [record.metadata_build_time_seconds for record in records]
    baseline_seconds = sum(step_times) if all(value is not None for value in step_times) else None
    metadata_seconds = (
        sum(metadata_times) if all(value is not None for value in metadata_times) else None
    )
    hit_metadata_times = [
        metadata_time
        for metadata_time, is_hit in zip(metadata_times, hit_flags)
        if is_hit and metadata_time is not None
    ]
    lookup_overhead_seconds = cache_lookup_overhead_seconds * len(records)

    metadata_fraction_percent = None
    if baseline_seconds is not None and metadata_seconds is not None and baseline_seconds:
        metadata_fraction_percent = (
            PERCENT_SCALE * metadata_seconds / baseline_seconds
        )

    estimated_average_speedup = None
    estimated_worst_case_speedup = None
    estimated_average_enabled_seconds = None
    estimated_worst_case_enabled_seconds = None
    if baseline_seconds is not None and len(hit_metadata_times) == sum(hit_flags):
        average_saved_seconds = sum(hit_metadata_times)
        worst_case_saved_seconds = (
            min(hit_metadata_times) * len(hit_metadata_times)
            if hit_metadata_times
            else MIN_NONNEGATIVE_VALUE
        )
        estimated_average_enabled_seconds = (
            baseline_seconds - average_saved_seconds + lookup_overhead_seconds
        )
        estimated_worst_case_enabled_seconds = (
            baseline_seconds - worst_case_saved_seconds + lookup_overhead_seconds
        )
        estimated_average_speedup = _speedup_or_none(
            baseline_seconds,
            estimated_average_enabled_seconds,
        )
        estimated_worst_case_speedup = _speedup_or_none(
            baseline_seconds,
            estimated_worst_case_enabled_seconds,
        )

    return {
        TIMING_BASELINE_SECONDS_KEY: baseline_seconds,
        TIMING_METADATA_SECONDS_KEY: metadata_seconds,
        TIMING_METADATA_FRACTION_KEY: metadata_fraction_percent,
        TIMING_LOOKUP_OVERHEAD_SECONDS_KEY: lookup_overhead_seconds,
        TIMING_AVERAGE_ENABLED_SECONDS_KEY: estimated_average_enabled_seconds,
        TIMING_WORST_CASE_ENABLED_SECONDS_KEY: estimated_worst_case_enabled_seconds,
        TIMING_AVERAGE_SPEEDUP_KEY: estimated_average_speedup,
        TIMING_WORST_CASE_SPEEDUP_KEY: estimated_worst_case_speedup,
    }


def evaluate_trace(
    trace: dict[str, Any],
    *,
    cache_disabled: bool = False,
    cache_lookup_overhead_seconds: float = DEFAULT_CACHE_LOOKUP_OVERHEAD_SECONDS,
) -> dict[str, Any]:
    """Apply IsoDelta-Halo reuse rules to one model-agnostic MLIP trace."""
    _require(
        cache_lookup_overhead_seconds >= MIN_NONNEGATIVE_VALUE,
        "cache lookup overhead must be nonnegative",
    )
    model_name = trace.get(MODEL_KEY, "unknown-mlip")
    _require(isinstance(model_name, str), f"{MODEL_KEY} must be a string when present")
    step_values = _as_sequence(trace.get(STEPS_KEY), STEPS_KEY)
    _require(step_values, f"{STEPS_KEY} must not be empty")

    records = [
        _parse_step(step, f"{STEPS_KEY}[{index}]")
        for index, step in enumerate(step_values)
    ]
    miss_counts = {f"{MISS_REASON_PREFIX}{reason}": 0.0 for reason in MISS_REASONS}
    previous_signature: StepSignature | None = None
    hit_flags: list[bool] = []
    decisions: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        miss_reason = _classify_reuse(previous_signature, record, cache_disabled)
        is_hit = miss_reason is None
        hit_flags.append(is_hit)
        decision = HIT_DECISION if is_hit else miss_reason
        if miss_reason is not None:
            miss_counts[f"{MISS_REASON_PREFIX}{miss_reason}"] += 1.0
        decisions.append(
            {
                "step_index": index,
                "step": record.step_id,
                "decision": decision,
            }
        )
        if not cache_disabled:
            previous_signature = record.signature

    attempts = float(len(records))
    hits = float(sum(hit_flags))
    hit_rate_percent = PERCENT_SCALE * hits / attempts
    return {
        "status": "evaluated",
        "model": model_name,
        ATTEMPTS_KEY: attempts,
        HITS_KEY: hits,
        HIT_RATE_PERCENT_KEY: hit_rate_percent,
        MISS_BREAKDOWN_KEY: miss_counts,
        "decisions": decisions,
        "timing": _timing_summary(
            records,
            hit_flags,
            cache_lookup_overhead_seconds,
        ),
        MODEL_AGNOSTIC_REQUIREMENTS_KEY: {
            requirement: True
            for requirement in REQUIRED_MODEL_AGNOSTIC_REQUIREMENTS
        },
    }


def _timing_field_name(key: str) -> str:
    """Return the dotted evidence path for one timing field."""
    return f"{TIMING_KEY}.{key}"


def _require_timing_basis(
    value_key: str,
    required_value: float | None,
    required_key: str,
) -> float:
    """Require a derived timing field to include its numerical basis."""
    _require(
        required_value is not None,
        f"{_timing_field_name(value_key)} requires {_timing_field_name(required_key)}",
    )
    return required_value


def _check_trace_timing(
    timing: dict[str, Any],
    thresholds: TraceThresholds,
) -> None:
    """Validate optional timing evidence and recompute derived speedups."""
    baseline_seconds = _as_optional_nonnegative_timing_number(
        timing.get(TIMING_BASELINE_SECONDS_KEY),
        _timing_field_name(TIMING_BASELINE_SECONDS_KEY),
    )
    metadata_seconds = _as_optional_nonnegative_timing_number(
        timing.get(TIMING_METADATA_SECONDS_KEY),
        _timing_field_name(TIMING_METADATA_SECONDS_KEY),
    )
    metadata_fraction = _as_optional_nonnegative_timing_number(
        timing.get(TIMING_METADATA_FRACTION_KEY),
        _timing_field_name(TIMING_METADATA_FRACTION_KEY),
    )
    _as_optional_nonnegative_timing_number(
        timing.get(TIMING_LOOKUP_OVERHEAD_SECONDS_KEY),
        _timing_field_name(TIMING_LOOKUP_OVERHEAD_SECONDS_KEY),
    )
    average_enabled_seconds = _as_optional_nonnegative_timing_number(
        timing.get(TIMING_AVERAGE_ENABLED_SECONDS_KEY),
        _timing_field_name(TIMING_AVERAGE_ENABLED_SECONDS_KEY),
    )
    worst_case_enabled_seconds = _as_optional_nonnegative_timing_number(
        timing.get(TIMING_WORST_CASE_ENABLED_SECONDS_KEY),
        _timing_field_name(TIMING_WORST_CASE_ENABLED_SECONDS_KEY),
    )
    average_speedup = _as_optional_timing_number(
        timing.get(TIMING_AVERAGE_SPEEDUP_KEY),
        _timing_field_name(TIMING_AVERAGE_SPEEDUP_KEY),
    )
    worst_case_speedup = _as_optional_timing_number(
        timing.get(TIMING_WORST_CASE_SPEEDUP_KEY),
        _timing_field_name(TIMING_WORST_CASE_SPEEDUP_KEY),
    )

    if metadata_fraction is not None:
        _validate_percent(
            metadata_fraction,
            _timing_field_name(TIMING_METADATA_FRACTION_KEY),
        )
        baseline_for_fraction = _require_timing_basis(
            TIMING_METADATA_FRACTION_KEY,
            baseline_seconds,
            TIMING_BASELINE_SECONDS_KEY,
        )
        metadata_for_fraction = _require_timing_basis(
            TIMING_METADATA_FRACTION_KEY,
            metadata_seconds,
            TIMING_METADATA_SECONDS_KEY,
        )
        _require(
            baseline_for_fraction > MIN_NONNEGATIVE_VALUE,
            (
                f"{_timing_field_name(TIMING_BASELINE_SECONDS_KEY)} must be "
                "positive when checking "
                f"{_timing_field_name(TIMING_METADATA_FRACTION_KEY)}"
            ),
        )
        expected_fraction = (
            PERCENT_SCALE * metadata_for_fraction / baseline_for_fraction
        )
        _require(
            _is_close(
                metadata_fraction,
                expected_fraction,
                absolute_tolerance=TIMING_PERCENT_TOLERANCE,
            ),
            (
                f"{_timing_field_name(TIMING_METADATA_FRACTION_KEY)} "
                "must match metadata / baseline"
            ),
        )

    for speedup_key, enabled_key, speedup_value in (
        (
            TIMING_AVERAGE_SPEEDUP_KEY,
            TIMING_AVERAGE_ENABLED_SECONDS_KEY,
            average_speedup,
        ),
        (
            TIMING_WORST_CASE_SPEEDUP_KEY,
            TIMING_WORST_CASE_ENABLED_SECONDS_KEY,
            worst_case_speedup,
        ),
    ):
        if speedup_value is None:
            continue
        _require(
            speedup_value > MIN_POSITIVE_SPEEDUP,
            f"{_timing_field_name(speedup_key)} must be positive",
        )
        baseline_for_speedup = _require_timing_basis(
            speedup_key,
            baseline_seconds,
            TIMING_BASELINE_SECONDS_KEY,
        )
        enabled_seconds = _require_timing_basis(
            speedup_key,
            (
                average_enabled_seconds
                if enabled_key == TIMING_AVERAGE_ENABLED_SECONDS_KEY
                else worst_case_enabled_seconds
            ),
            enabled_key,
        )
        _require(
            enabled_seconds > MIN_NONNEGATIVE_VALUE,
            (
                f"{_timing_field_name(enabled_key)} must be positive when "
                f"checking {_timing_field_name(speedup_key)}"
            ),
        )
        expected_speedup = baseline_for_speedup / enabled_seconds
        _require(
            _is_close(speedup_value, expected_speedup),
            f"{_timing_field_name(speedup_key)} must match baseline / enabled seconds",
        )

    if thresholds.min_metadata_fraction_percent > DEFAULT_MIN_METADATA_FRACTION_PERCENT:
        _require(
            metadata_fraction is not None,
            f"{_timing_field_name(TIMING_METADATA_FRACTION_KEY)} must be numeric",
        )
        _require(
            metadata_fraction >= thresholds.min_metadata_fraction_percent,
            (
                f"metadata fraction {metadata_fraction:g}% is below "
                f"{thresholds.min_metadata_fraction_percent:g}%"
            ),
        )
    if thresholds.min_estimated_speedup is not None:
        _require(
            average_speedup is not None,
            f"{_timing_field_name(TIMING_AVERAGE_SPEEDUP_KEY)} must be numeric",
        )
        _require(
            average_speedup >= thresholds.min_estimated_speedup,
            (
                f"estimated speedup {average_speedup:g} is below "
                f"{thresholds.min_estimated_speedup:g}"
            ),
        )


def validate_trace_evidence(
    evidence: dict[str, Any],
    thresholds: TraceThresholds,
) -> dict[str, Any]:
    """Gate evaluated trace evidence before using it as a generality claim."""
    validate_thresholds(thresholds)
    attempts = _as_nonnegative_int(evidence.get(ATTEMPTS_KEY), ATTEMPTS_KEY)
    hits = _as_nonnegative_int(evidence.get(HITS_KEY), HITS_KEY)
    hit_rate_percent = _as_number(
        evidence.get(HIT_RATE_PERCENT_KEY),
        HIT_RATE_PERCENT_KEY,
    )
    _require(
        attempts >= MIN_REQUIRED_TRACE_ATTEMPTS,
        f"{ATTEMPTS_KEY} must be at least one",
    )
    _require(hits <= attempts, f"{HITS_KEY} cannot exceed {ATTEMPTS_KEY}")
    _validate_percent(hit_rate_percent, HIT_RATE_PERCENT_KEY)
    expected_hit_rate = PERCENT_SCALE * hits / attempts
    _require(
        abs(hit_rate_percent - expected_hit_rate) <= TRACE_COUNT_TOLERANCE,
        f"{HIT_RATE_PERCENT_KEY} must match hits / attempts",
    )
    _require(
        hit_rate_percent >= thresholds.min_hit_rate_percent,
        (
            f"hit rate {hit_rate_percent:g}% is below "
            f"{thresholds.min_hit_rate_percent:g}%"
        ),
    )
    miss_breakdown = _as_mapping(evidence.get(MISS_BREAKDOWN_KEY), MISS_BREAKDOWN_KEY)
    miss_count_sum = 0.0
    for miss_reason in MISS_REASONS:
        miss_key = f"{MISS_REASON_PREFIX}{miss_reason}"
        miss_count = _as_nonnegative_int(
            miss_breakdown.get(miss_key),
            f"{MISS_BREAKDOWN_KEY}.{miss_key}",
        )
        miss_count_sum += miss_count
    trace_count_residual = abs(miss_count_sum - (attempts - hits))
    _require(
        trace_count_residual <= TRACE_COUNT_TOLERANCE,
        f"{MISS_BREAKDOWN_KEY} counters must match attempts - hits",
    )
    model_agnostic_requirements = _as_mapping(
        evidence.get(MODEL_AGNOSTIC_REQUIREMENTS_KEY),
        MODEL_AGNOSTIC_REQUIREMENTS_KEY,
    )
    for requirement in REQUIRED_MODEL_AGNOSTIC_REQUIREMENTS:
        requirement_value = _as_bool(
            model_agnostic_requirements.get(requirement),
            f"{MODEL_AGNOSTIC_REQUIREMENTS_KEY}.{requirement}",
        )
        _require(
            requirement_value,
            f"{MODEL_AGNOSTIC_REQUIREMENTS_KEY}.{requirement} must be true",
        )
    timing = _as_mapping(evidence.get(TIMING_KEY), TIMING_KEY)
    _check_trace_timing(timing, thresholds)
    evidence["status"] = "passed"
    evidence["thresholds"] = asdict(thresholds)
    evidence[TRACE_COUNT_RESIDUAL_KEY] = trace_count_residual
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, evaluate one trace, and print JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the portable trace schema and exit without evaluating a trace",
    )
    parser.add_argument(
        "--cache-disabled",
        action="store_true",
        help="Simulate the disabled-cache baseline on the same trace",
    )
    parser.add_argument(
        "--cache-lookup-overhead-seconds",
        type=float,
        default=DEFAULT_CACHE_LOOKUP_OVERHEAD_SECONDS,
        help="Per-step cache lookup overhead used in speedup estimates",
    )
    parser.add_argument(
        "--min-hit-rate-percent",
        type=float,
        default=DEFAULT_MIN_HIT_RATE_PERCENT,
        help="Minimum reusable-step hit rate required for the trace",
    )
    parser.add_argument(
        "--min-estimated-speedup",
        type=float,
        help="Optional minimum average-case speedup estimated from timing fields",
    )
    parser.add_argument(
        "--min-metadata-fraction-percent",
        type=float,
        default=DEFAULT_MIN_METADATA_FRACTION_PERCENT,
        help="Minimum baseline runtime share spent building reusable metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the evaluated evidence JSON",
    )
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(trace_schema(), indent=2))
        return 0
    if args.trace is None:
        parser.error("--trace is required unless --print-schema is used")

    thresholds = TraceThresholds(
        min_hit_rate_percent=args.min_hit_rate_percent,
        min_estimated_speedup=args.min_estimated_speedup,
        min_metadata_fraction_percent=args.min_metadata_fraction_percent,
    )
    try:
        evidence = validate_trace_evidence(
            evaluate_trace(
                load_trace(args.trace),
                cache_disabled=args.cache_disabled,
                cache_lookup_overhead_seconds=args.cache_lookup_overhead_seconds,
            ),
            thresholds,
        )
    except TraceCheckError as exc:
        print(f"IsoDelta-Halo MLIP trace check failed: {exc}")
        return 1

    output_text = json.dumps(evidence, indent=2)
    if args.output is not None:
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
