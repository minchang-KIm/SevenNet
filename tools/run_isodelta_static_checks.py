"""Run dependency-free static checks for the IsoDelta-Halo implementation.

This script is intentionally limited to the Python standard library so it can
run on developer machines before the heavier LAMMPS/LibTorch build exists.
"""

from __future__ import annotations

from pathlib import Path


# Keep every static check anchored at the repository root so the script works
# from both CI-like shells and ad-hoc local invocations.
REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "pair_e3gnn_parallel.cpp"
HEADER_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "pair_e3gnn_parallel.h"
COMM_BRICK_CPP_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "comm_brick.cpp"
COMM_BRICK_HEADER_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "comm_brick.h"
BENCHMARK_PATH = REPO_ROOT / "tools" / "run_isodelta_lammps_benchmark.py"
EXPERIMENT_PATH = REPO_ROOT / "tools" / "run_isodelta_experiment.py"
CLUSTER_SUITE_PATH = REPO_ROOT / "tools" / "run_isodelta_cluster_paper_suite.py"
REPORT_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
EVIDENCE_BUNDLE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_evidence_bundle.py"
GOAL_READINESS_PATH = REPO_ROOT / "tools" / "check_isodelta_goal_readiness.py"
PREREQ_PATH = REPO_ROOT / "tools" / "check_isodelta_build_prereqs.py"
BINARY_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_lammps_binary.py"
MLIP_TRACE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
MLIP_TRACE_DEMO_PATH = REPO_ROOT / "tools" / "run_isodelta_mlip_trace_demo.py"
VALIDATION_RUNNER_PATH = REPO_ROOT / "tools" / "run_isodelta_validation.py"
SYNC_GATE_PATH = REPO_ROOT / "tools" / "run_isodelta_sync_gate.py"
PATCH_SCRIPT_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "patch_lammps.sh"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "isodelta-halo.yml"
DOC_PATH = REPO_ROOT / "docs" / "source" / "user_guide" / "isodelta_halo.md"
DOC_INDEX_PATH = REPO_ROOT / "docs" / "source" / "user_guide" / "index.rst"
CLUSTER_SUITE_TEST_PATH = REPO_ROOT / "tests" / "unit_tests" / "test_isodelta_cluster_paper_suite.py"
BENCHMARK_REPORT_TEST_PATH = (
    REPO_ROOT / "tests" / "unit_tests" / "test_isodelta_benchmark_report_check.py"
)
GOAL_READINESS_TEST_PATH = REPO_ROOT / "tests" / "unit_tests" / "test_isodelta_goal_readiness.py"
SYNC_GATE_TEST_PATH = REPO_ROOT / "tests" / "unit_tests" / "test_isodelta_sync_gate.py"
VALIDATION_RUNNER_TEST_PATH = REPO_ROOT / "tests" / "unit_tests" / "test_isodelta_validation_runner.py"


def _read(path: Path) -> str:
    """Read a source file with an explicit encoding for reproducible checks."""
    return path.read_text(encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    """Raise an assertion-style error while keeping script output concise."""
    if not condition:
        raise SystemExit(f"IsoDelta-Halo static check failed: {message}")


def main() -> None:
    """Validate that the cache remains metadata-only and conservatively gated."""
    cpp = _read(CPP_PATH)
    header = _read(HEADER_PATH)
    comm_brick_cpp = _read(COMM_BRICK_CPP_PATH)
    comm_brick_header = _read(COMM_BRICK_HEADER_PATH)
    benchmark = _read(BENCHMARK_PATH)
    experiment = _read(EXPERIMENT_PATH)
    cluster_suite = _read(CLUSTER_SUITE_PATH)
    report_check = _read(REPORT_CHECK_PATH)
    evidence_bundle_check = _read(EVIDENCE_BUNDLE_CHECK_PATH)
    goal_readiness = _read(GOAL_READINESS_PATH)
    prereq = _read(PREREQ_PATH)
    binary_check = _read(BINARY_CHECK_PATH)
    mlip_trace_check = _read(MLIP_TRACE_CHECK_PATH)
    mlip_trace_demo = _read(MLIP_TRACE_DEMO_PATH)
    validation_runner = _read(VALIDATION_RUNNER_PATH)
    sync_gate = _read(SYNC_GATE_PATH)
    patch_script = _read(PATCH_SCRIPT_PATH)
    workflow = _read(WORKFLOW_PATH)
    doc = _read(DOC_PATH)
    doc_index = _read(DOC_INDEX_PATH)
    cluster_suite_test = _read(CLUSTER_SUITE_TEST_PATH)
    benchmark_report_test = _read(BENCHMARK_REPORT_TEST_PATH)
    goal_readiness_test = _read(GOAL_READINESS_TEST_PATH)
    sync_gate_test = _read(SYNC_GATE_TEST_PATH)
    validation_runner_test = _read(VALIDATION_RUNNER_TEST_PATH)
    combined = cpp + "\n" + header + "\n" + comm_brick_cpp + "\n" + comm_brick_header

    _require(
        "TODO" not in combined and "temporary" not in combined.lower(),
        "IsoDelta-Halo pair/comm sources must not carry temporary implementation markers",
    )
    _require("kCommPhaseCount = 6" in header, "named comm phase count missing")
    _require(
        "kCommCacheMissReasonCount = 9" in header,
        "cache miss reason count must include index tensor shape changes",
    )
    _require("[6]" not in cpp, "raw six-phase array/magic count remains in cpp")
    _require("[6]" not in header, "raw six-phase array/magic count remains in header")
    for include_name in (
        "<algorithm>",
        "<cstring>",
        "<iostream>",
        "<list>",
        "<map>",
        "<set>",
    ):
        _require(
            f"#include {include_name}" in cpp,
            f"pair_e3gnn_parallel.cpp must explicitly include {include_name}",
        )
    _require(
        "std::vector<long> &upmap = comm_index_unpack_forward[comm_phase];" in cpp,
        "CUDA unpack tensor creation must not copy the unpack vector first",
    )
    _require(
        "make_owned_index_tensor" in cpp
        and ".clone()\n      .to(target_device)" in cpp
        and "kEmptyIndexTensorLength" in cpp
        and "kIndexTensorRank" in cpp
        and "kIndexTensorLengthDimension" in cpp
        and "index_tensor_matches_vector" in cpp
        and "cached_comm_tensors_match_vectors" in combined
        and "if (!cached_comm_tensors_match_vectors())" in cpp
        and "kIndexTensorShapeChanged" in header
        and "index-tensor-shape-changed" in cpp
        and "record_comm_cache_miss(CommCacheMissReason::kIndexTensorShapeChanged);" in cpp
        and "if (index_map.empty())" in cpp
        and "torch::empty({kEmptyIndexTensorLength}, INTEGER_TYPE)" in cpp
        and "torch::from_blob(idx_map_forward.data()" not in cpp
        and "torch::from_blob(upmap.data()" not in cpp
        and "torch::from_blob(idx_map_reverse.data()" not in cpp,
        "cached index tensors must own memory instead of borrowing step vectors",
    )
    _require(
        "kIsoDeltaHaloCommBrickRequiredError" in cpp
        and "if (comm_brick == nullptr)" in cpp
        and "error->all(FLERR, kIsoDeltaHaloCommBrickRequiredError)" in cpp,
        "comm_preprocess must fail clearly when CommBrick is unavailable",
    )
    _require(
        "notify_proc_ids(" in cpp
        and "int active_phase_count" in cpp
        and "kNoActiveCommPhases = 0" in header
        and "bounded_active_phase_count" in cpp
        and "std::min(std::max(active_phase_count, kNoActiveCommPhases)" in cpp
        and "iswap < bounded_active_phase_count" in cpp
        and "active_phase ? sendproc[iswap] : kInactiveCommPhaseValue" in cpp
        and "active_phase ? recvproc[iswap] : kInactiveCommPhaseValue" in cpp,
        "proc id notification must initialize inactive comm phases",
    )
    _require(
        "kE3GnnCommPhaseLimit = 6" in comm_brick_cpp,
        "CommBrick e3gnn phase limit must be named",
    )
    _require(
        "nswap > 6" not in comm_brick_cpp,
        "CommBrick e3gnn path must not compare against a raw phase limit",
    )
    _require(
        "kE3GnnCommPhaseLimitError" in comm_brick_cpp,
        "CommBrick e3gnn phase-limit error must be named",
    )
    _require(
        "pair->notify_proc_ids(sendproc, recvproc, nswap)" in comm_brick_cpp,
        "CommBrick must pass the active phase count to PairE3GNNParallel",
    )
    for accessor_name in (
        "e3gnn_nswap",
        "e3gnn_sendnum",
        "e3gnn_recvnum",
        "e3gnn_sendproc",
        "e3gnn_recvproc",
        "e3gnn_firstrecv",
        "e3gnn_sendlist_atom",
    ):
        _require(
            accessor_name in comm_brick_header and f"CommBrick::{accessor_name}" in comm_brick_cpp,
            f"CommBrick topology accessor missing: {accessor_name}",
        )
    _require(
        "comm_topology_matches_cache" in combined
        and "store_comm_topology_signature" in combined
        and "comm-topology-changed" in cpp,
        "cache reuse must compare and report communication topology signatures",
    )
    _require(
        "comm_list_tags_match_cache" in combined
        and "store_comm_list_tag_signature" in combined
        and "comm-list-tag-order-changed" in cpp,
        "cache reuse must compare and report send/recv list tag signatures",
    )

    _require(
        "try_reuse_comm_preprocess_cache" in combined,
        "cache reuse function is missing",
    )
    _require(
        "store_comm_preprocess_cache" in combined,
        "cache store function is missing",
    )
    _require(
        "current_comm_topology_is_cacheable" in combined
        and "if (!current_comm_topology_is_cacheable())" in cpp
        and "current_nswap >= kNoActiveCommPhases" in cpp
        and "current_nswap <= kCommPhaseCount" in cpp,
        "cache store must reject uncapturable communication topologies",
    )
    _require(
        "invalidate_comm_preprocess_cache" in combined
        and "comm_cache_valid = false;" in cpp
        and "comm_cache_nswap = kInactiveCommPhaseValue;" in cpp
        and "comm_cache_graph_tags.clear();" in cpp
        and "comm_cache_index_pack_forward_tensor[comm_phase] = torch::Tensor();" in cpp
        and cpp.count("invalidate_comm_preprocess_cache();") >= 7,
        "cache miss paths must invalidate stale IsoDelta-Halo metadata",
    )
    _require(
        "clear_comm_preprocess_work" in combined,
        "per-step cache work cleanup helper is missing",
    )
    _require(
        "neighbor->ago <= kNeighborListJustBuiltAgo" in cpp,
        "cache is not gated by LAMMPS neighbor-list age",
    )
    _require(
        "comm_preprocess();\n    if (iso_delta_halo_enabled)" in cpp,
        "cache miss path must rebuild before any optional cache store",
    )
    _require(
        "if (iso_delta_halo_enabled) {\n      store_comm_preprocess_cache" in cpp,
        "cache store must be skipped when IsoDelta-Halo is disabled",
    )
    _require(
        'kIsoDeltaHaloDisableEnv = "SEVENN_ISODELTA_HALO_DISABLE"' in cpp,
        "cache disable environment variable is not named in the cpp file",
    )
    _require(
        'kIsoDeltaHaloProfileEnv = "SEVENN_ISODELTA_HALO_PROFILE"' in cpp,
        "cache profiling environment variable is not named in the cpp file",
    )
    _require(
        "iso_delta_halo_env_flag_is_enabled" in cpp
        and "normalize_iso_delta_halo_env_flag_value" in cpp
        and "std::isspace(static_cast<unsigned char>(*begin))" in cpp
        and "std::isspace(static_cast<unsigned char>(*last_character))" in cpp
        and 'kIsoDeltaHaloEnvFlagValueZero = "0"' in cpp
        and 'kIsoDeltaHaloEnvFlagValueFalse = "false"' in cpp
        and 'kIsoDeltaHaloEnvFlagValueNo = "no"' in cpp
        and 'kIsoDeltaHaloEnvFlagValueOff = "off"' in cpp
        and "!iso_delta_halo_env_flag_is_enabled(kIsoDeltaHaloDisableEnv)"
        in cpp
        and "iso_delta_halo_env_flag_is_enabled(kIsoDeltaHaloProfileEnv)"
        in cpp
        and "std::getenv(kIsoDeltaHaloDisableEnv) == nullptr" not in cpp
        and "std::getenv(kIsoDeltaHaloProfileEnv) != nullptr" not in cpp,
        "cache runtime flags must parse explicit false values",
    )
    _require(
        "static constexpr const char *" not in header,
        "header must not require out-of-class string pointer definitions",
    )
    _require(
        "kIsoDeltaHaloPercentScale" in cpp,
        "cache hit-rate scale must be a named cpp-local constant",
    )
    _require(
        "kBytesPerMebibyte" in cpp
        and "kFloatElementBytes" in cpp
        and "(1024 * 1024)" not in cpp
        and "x_dim * n * 4" not in cpp,
        "profiling byte-size calculations must use named constants",
    )
    _require(
        "MEM use after backward(MiB)" in cpp
        and "send size(MiB)" in cpp
        and "send size(MB)" not in cpp,
        "profiling memory labels must match MiB calculations",
    )
    _require(
        "comm_cache_attempts++" in cpp
        and "comm_cache_hits++" in cpp
        and "comm_cache_miss_reason_is_valid" in combined
        and "reason_index >= 0" in cpp
        and "reason_index < kCommCacheMissReasonCount" in cpp
        and "if (!comm_cache_miss_reason_is_valid(reason))" in cpp,
        "cache counters or miss-reason bounds guard are missing",
    )
    _require(
        "print_comm_cache_summary" in combined and "hit_rate_percent" in cpp,
        "cache profiling summary is missing",
    )

    forbidden_cache_terms = (
        "force_cache",
        "message_cache",
        "embedding_cache",
        "geometry_cache",
        "edge_vec_cache",
    )
    for term in forbidden_cache_terms:
        _require(term not in combined, f"forbidden value cache term found: {term}")

    for line in combined.splitlines():
        if "comm_cache" in line:
            lowered = line.lower()
            _require("edge_vec" not in lowered, "edge vectors must not be cached")
            _require("force" not in lowered, "forces must not be cached")
            _require("message" not in lowered, "messages must not be cached")
            _require("embedding" not in lowered, "embeddings must not be cached")

    _require(
        "extra_graph_idx_map[list_i] = graph_size + extra_graph_idx_map.size();"
        in cpp,
        "pack-forward extra graph map must be keyed by list_i",
    )
    _require(
        doc.lstrip().startswith("<!--"),
        "IsoDelta-Halo guide must start with a generated-file comment",
    )
    _require(
        "run_isodelta_lammps_benchmark.py" in doc,
        "IsoDelta-Halo guide must document the benchmark runner",
    )
    _require(
        "EXPECTED_LAMMPS_VERSION" in prereq
        and "check_lammps_root" in prereq
        and "check_torch_import" in prereq,
        "build prerequisite checker must cover LAMMPS and torch checks",
    )
    _require(
        'REQUIRED_LAMMPS_VERSION="2 Aug 2023"' in patch_script
        and "REQUIRED_LAMMPS_VERSION" in patch_script
        and "pair_e3gnn_oeq_autograd.cpp" in patch_script,
        "LAMMPS patch script must name the version and copy the oEq bridge",
    )
    _require(
        "TODO" not in patch_script and "Example required version" not in patch_script,
        "LAMMPS patch script must not contain temporary implementation notes",
    )
    _require(
        workflow.lstrip().startswith("#")
        and "Run IsoDelta-Halo validation" in workflow
        and "python tools/run_isodelta_validation.py" in workflow
        and "--report-path isodelta_validation_report.json" in workflow
        and "actions/upload-artifact@v4" in workflow
        and "sevenn/pair_e3gnn/**" in workflow
        and "tools/run_isodelta_*.py" in workflow,
        "IsoDelta-Halo workflow must run the lightweight validation gate",
    )
    _require(
        "VALIDATION_REPORT_SCHEMA_VERSION" in validation_runner
        and "--report-path" in validation_runner
        and "--expected-branch" in validation_runner
        and "_validation_commands" in validation_runner
        and "elapsed_seconds" in validation_runner
        and "stdout_tail" in validation_runner
        and "stderr_tail" in validation_runner
        and "git_status_short" in validation_runner
        and "VALIDATION_COMMAND_REQUIRED_FIELDS" in validation_runner_test
        and "test_expected_branch_reaches_goal_readiness_command" in validation_runner_test
        and "test_run_validation_writes_sync_gate_compatible_command_record"
        in validation_runner_test
        and "test_isodelta_validation_runner.py" in validation_runner,
        "validation runner must emit auditable sync reports",
    )
    _require(
        "SYNC_REPORT_SCHEMA_VERSION" in sync_gate
        and "EXPECTED_VALIDATION_REPORT_SCHEMA_VERSION" in sync_gate
        and "PUSH_AUTH_ENVIRONMENT" in sync_gate
        and "GIT_TERMINAL_PROMPT" in sync_gate
        and "run_sync" in sync_gate
        and "--expected-branch" in sync_gate
        and "_sync_git_provenance" in sync_gate
        and "git_provenance" in sync_gate
        and "STATUS_REMOTE_VERIFICATION_FAILED" in sync_gate
        and "REMOTE_REF_VERIFICATION_KEY" in sync_gate
        and "REMOTE_REF_VERIFY_COMMAND_NAME" in sync_gate
        and "VALIDATION_REPORT_FINGERPRINT_KEY" in sync_gate
        and "VALIDATION_REPORT_SUMMARY_KEY" in sync_gate
        and "SYNC_COMMAND_SUMMARY_KEY" in sync_gate
        and "_sync_command_summary" in sync_gate
        and "PUSH_BRANCH_PRECONDITION_COMMAND_NAME" in sync_gate
        and "_branch_precondition_failure_record" in sync_gate
        and "STATUS_VALIDATION_REPORT_MISSING" in sync_gate
        and "STATUS_VALIDATION_REPORT_INVALID" in sync_gate
        and "_validation_report_summary" in sync_gate
        and "command_failure_count" in sync_gate
        and "VALIDATION_REPORT_COMMAND_REQUIRED_FIELDS" in sync_gate
        and "command_missing_field_count" in sync_gate
        and "_validation_command_record_has_valid_shape" in sync_gate
        and "command_invalid_field_count" in sync_gate
        and "validation report commands are missing required fields" in sync_gate
        and "validation report commands have invalid field values" in sync_gate
        and "validation report commands include nonzero returncodes" in sync_gate
        and "STATUS_DIRTY_WORKTREE" in sync_gate
        and "WORKTREE_STATUS_KEY" in sync_gate
        and "_parse_status_short" in sync_gate
        and "_worktree_status" in sync_gate
        and "--require-clean-worktree" in sync_gate
        and "_file_fingerprint" in sync_gate
        and "_remote_ref_verify_command" in sync_gate
        and "_verify_remote_ref" in sync_gate
        and "--push-failure-bundle" in sync_gate
        and "_write_push_failure_bundle" in sync_gate
        and "PUSH_FAILURE_BUNDLE_VERIFY_COMMAND_NAME" in sync_gate
        and "_bundle_verify_command" in sync_gate
        and "PUSH_FAILURE_BUNDLE_KEY" in sync_gate
        and "BUNDLE_HASH_READ_CHUNK_BYTES" in sync_gate
        and "_bundle_file_fingerprint" in sync_gate
        and "verify_returncode" in sync_gate
        and "hashlib.sha256" in sync_gate
        and "STATUS_PUSH_FAILED" in sync_gate
        and "PUSH_FAILURE_REASON_AUTH_PROMPT_DISABLED" in sync_gate
        and "_classify_push_failure" in sync_gate
        and '"push_failure": _classify_push_failure(push_record)' in sync_gate
        and "test_validation_command_enforces_target_branch" in sync_gate_test
        and "test_run_sync_records_git_provenance_for_push_target" in sync_gate_test
        and "test_run_sync_fails_when_remote_ref_does_not_match" in sync_gate_test
        and "test_run_sync_rejects_missing_validation_report" in sync_gate_test
        and "test_run_sync_rejects_failed_validation_report_json" in sync_gate_test
        and "test_run_sync_rejects_failed_validation_report_command" in sync_gate_test
        and "test_run_sync_can_require_clean_worktree" in sync_gate_test
        and "WORKTREE_STATUS_KEY" in sync_gate_test
        and "VALIDATION_REPORT_FINGERPRINT_KEY" in sync_gate_test
        and "VALIDATION_REPORT_SUMMARY_KEY" in sync_gate_test
        and "REMOTE_REF_VERIFY_COMMAND_NAME" in sync_gate_test
        and "SYNC_COMMAND_REQUIRED_FIELDS" in sync_gate_test
        and "assert_sync_command_record_shape" in sync_gate_test
        and "test_run_sync_records_replayable_branch_precondition_failure"
        in sync_gate_test
        and "test_run_sync_can_write_bundle_after_push_failure" in sync_gate_test
        and "PUSH_FAILURE_BUNDLE_VERIFY_COMMAND_NAME" in sync_gate_test
        and "hashlib.sha256(b\"bundle\").hexdigest()" in sync_gate_test
        and "test_run_sync_classifies_noninteractive_auth_push_failure" in sync_gate_test
        and "test_run_sync_rejects_incomplete_validation_command_record" in sync_gate_test
        and "test_run_sync_rejects_invalid_validation_command_record_values"
        in sync_gate_test
        and "test_isodelta_sync_gate.py" in validation_runner,
        "sync gate must validate before recording git push attempts",
    )
    _require(
        "GOAL_READINESS_SCHEMA_VERSION" in goal_readiness
        and "REQUIRED_FILE_SNIPPETS" in goal_readiness
        and "COMMENT_PREFIX_REQUIREMENTS" in goal_readiness
        and 'EXPECTED_BRANCH = "codex/isodelta-halo-runtime"' in goal_readiness
        and "ISODELTA_PYTHON_GLOB_PATTERNS" in goal_readiness
        and "_audit_isodelta_python_headers" in goal_readiness
        and "build_goal_readiness_report" in goal_readiness
        and "default=None" in goal_readiness
        and '"goal_readiness_report": str(args.report_path) if args.report_path is not None else None'
        in goal_readiness
        and "test_isodelta_goal_readiness.py" in validation_runner
        and "test_goal_readiness_rejects_isodelta_python_without_header"
        in goal_readiness_test
        and "check_isodelta_goal_readiness.py" in validation_runner,
        "goal readiness audit must be part of the lightweight validation gate",
    )
    _require(
        "check_isodelta_build_prereqs.py" in doc,
        "IsoDelta-Halo guide must document the build prerequisite checker",
    )
    _require(
        "PAIR_STYLE_NAME = \"e3gnn/parallel\"" in binary_check
        and "parse_pair_style_available" in binary_check,
        "LAMMPS binary smoke checker must verify e3gnn/parallel registration",
    )
    _require(
        "validate_binary_check_options" in binary_check
        and "MIN_POSITIVE_TIMEOUT_SECONDS" in binary_check
        and "timeout_seconds must be positive" in binary_check
        and "lammps_command must not be empty" in binary_check,
        "LAMMPS binary smoke checker must reject meaningless options",
    )
    _require(
        "check_isodelta_lammps_binary.py" in doc
        and "`--timeout-seconds` must be positive" in doc,
        "IsoDelta-Halo guide must document the LAMMPS binary smoke checker",
    )
    _require(
        "MODEL_KEY = \"model\"" in mlip_trace_check
        and "STATUS_KEY = \"status\"" in mlip_trace_check
        and "EVALUATED_STATUS" in mlip_trace_check
        and "_as_nonempty_string(evidence.get(STATUS_KEY)" in mlip_trace_check
        and "_as_nonempty_string(evidence.get(MODEL_KEY)" in mlip_trace_check
        and "evidence[MODEL_KEY] = model_name" in mlip_trace_check
        and "MISS_COMM_TOPOLOGY_CHANGED" in mlip_trace_check
        and "MISS_COMM_LIST_TAG_ORDER_CHANGED" in mlip_trace_check
        and "MISS_INDEX_TENSOR_SHAPE_CHANGED" in mlip_trace_check
        and "TraceThresholds" in mlip_trace_check
        and "validate_thresholds" in mlip_trace_check
        and "MAX_PERCENT_VALUE" in mlip_trace_check
        and "TRACE_COUNT_TOLERANCE" in mlip_trace_check
        and "TRACE_COUNT_RESIDUAL_KEY" in mlip_trace_check
        and "MODEL_AGNOSTIC_REQUIREMENTS_KEY" in mlip_trace_check
        and "REQUIRED_MODEL_AGNOSTIC_REQUIREMENTS" in mlip_trace_check
        and "_check_trace_timing" in mlip_trace_check
        and "TIMING_AVERAGE_ENABLED_SECONDS_KEY" in mlip_trace_check
        and "metadata / baseline" in mlip_trace_check
        and "baseline / enabled seconds" in mlip_trace_check
        and "must be true" in mlip_trace_check
        and "math.isfinite" in mlip_trace_check
        and "attempts = _as_nonnegative_int" in mlip_trace_check
        and "hits = _as_nonnegative_int" in mlip_trace_check
        and "miss_count = _as_nonnegative_int" in mlip_trace_check
        and "must be a nonnegative integer" in mlip_trace_check
        and "must match hits / attempts" in mlip_trace_check
        and "must match attempts - hits" in mlip_trace_check
        and "trace_schema" in mlip_trace_check
        and "--print-schema" in mlip_trace_check
        and "estimated_average_speedup" in mlip_trace_check
        and "estimated_worst_case_speedup" in mlip_trace_check,
        "MLIP trace checker must expose model-agnostic reuse and speedup evidence",
    )
    _require(
        "DEMO_MODELS = (\"SevenNet\", \"MACE\", \"NequIP\", \"Allegro\")" in mlip_trace_demo
        and "check_isodelta_mlip_trace.py" in mlip_trace_demo
        and "build_demo_trace" in mlip_trace_demo
        and "run_demo" in mlip_trace_demo
        and "TraceThresholds" in mlip_trace_demo
        and "estimated_average_speedup" in mlip_trace_demo
        and "estimated_worst_case_speedup" in mlip_trace_demo,
        "MLIP trace demo must generate validated multi-model portability evidence",
    )
    _require(
        "check_isodelta_benchmark_report.py" in evidence_bundle_check
        and "check_isodelta_mlip_trace.py" in evidence_bundle_check
        and "validate_bundle" in evidence_bundle_check
        and "validate_thresholds" in evidence_bundle_check
        and "_validate_unique_trace_evidence_paths" in evidence_bundle_check
        and "_validate_distinct_model_count" in evidence_bundle_check
        and "_artifact_record" in evidence_bundle_check
        and "collect_run_provenance" in evidence_bundle_check
        and "BUNDLE_SCHEMA_VERSION" in evidence_bundle_check
        and "BUNDLE_SCHEMA_VERSION_KEY" in evidence_bundle_check
        and "PROVENANCE_KEY" in evidence_bundle_check
        and "git_dirty" in evidence_bundle_check
        and "platform.platform" in evidence_bundle_check
        and "hashlib.sha256" in evidence_bundle_check
        and "HASH_READ_CHUNK_BYTES" in evidence_bundle_check
        and "ARTIFACTS_KEY" in evidence_bundle_check
        and "ARTIFACT_SHA256_KEY" in evidence_bundle_check
        and "ARTIFACT_SIZE_BYTES_KEY" in evidence_bundle_check
        and "_validate_required_model_names" in evidence_bundle_check
        and "duplicate trace evidence paths" in evidence_bundle_check
        and "duplicate required trace models" in evidence_bundle_check
        and "min_distinct_trace_models" in evidence_bundle_check
        and "--min-distinct-trace-models" in evidence_bundle_check
        and "MIN_REQUIRED_TRACE_COUNT" in evidence_bundle_check
        and "MAX_PERCENT_VALUE" in evidence_bundle_check
        and "--require-trace-model" in evidence_bundle_check
        and "min_trace_estimated_speedup" in evidence_bundle_check,
        "evidence bundle checker must gate benchmark and trace evidence together",
    )
    _require(
        "check_isodelta_mlip_trace.py" in doc
        and "run_isodelta_mlip_trace_demo.py" in doc
        and "graph_node_tags" in doc
        and "comm_phases" in doc
        and "SevenNet, MACE, NequIP, and Allegro" in doc
        and "--print-schema" in doc
        and "estimated_average_speedup" in doc
        and "MACE" in doc
        and "NequIP" in doc
        and "Allegro" in doc,
        "IsoDelta-Halo guide must document the model-agnostic trace checker",
    )
    _require(
        "hit_rate_percent" in doc
        and "hits / attempts" in doc
        and "attempts - hits" in doc,
        "IsoDelta-Halo guide must document trace evidence counter consistency",
    )
    _require(
        "whole nonnegative reuse-decision counts" in doc,
        "IsoDelta-Halo guide must document trace count integer validation",
    )
    _require(
        "`status` as `evaluated` or `passed`" in doc
        and "non-empty `model` label" in doc,
        "IsoDelta-Halo guide must document trace model/status validation",
    )
    _require(
        "Model labels are stripped" in doc
        and "normalized value" in doc
        and "bundle gate" in doc,
        "IsoDelta-Halo guide must document trace model label normalization",
    )
    _require(
        "When timing fields are present" in doc
        and "`metadata_fraction_percent`" in doc
        and "baseline" in doc
        and "enabled seconds" in doc
        and "timing basis" in doc,
        "IsoDelta-Halo guide must document trace timing consistency validation",
    )
    _require(
        "model_agnostic_requirements" in doc
        and "ordered graph node tags" in doc
        and "communication list" in doc,
        "IsoDelta-Halo guide must document trace reuse guard evidence",
    )
    _require(
        "check_isodelta_evidence_bundle.py" in doc
        and "bundle_evidence.json" in doc
        and "--require-trace-model" in doc,
        "IsoDelta-Halo guide must document the evidence bundle checker",
    )
    _require(
        "duplicate trace evidence paths" in doc
        and "duplicate" in doc
        and "empty required model labels" in doc
        and "min_trace_count" in doc,
        "IsoDelta-Halo guide must document bundle input validation",
    )
    _require(
        "--min-distinct-trace-models" in doc
        and "multi-model portability evidence" in doc
        and "same MLIP label" in doc,
        "IsoDelta-Halo guide must document distinct trace model gating",
    )
    _require(
        "SHA-256 digest" in doc
        and "byte" in doc
        and "benchmark report and every trace evidence file" in doc,
        "IsoDelta-Halo guide must document evidence artifact fingerprints",
    )
    _require(
        "`bundle_schema_version`" in doc
        and "`provenance` object" in doc
        and "Git, Python, and platform" in doc,
        "IsoDelta-Halo guide must document bundle evidence provenance",
    )
    _require(
        "parse_final_thermo_observables" in benchmark
        and "final_thermo_delta_vs_disabled_cache" in benchmark,
        "benchmark runner must report final thermo consistency deltas",
    )
    _require(
        "SUMMARY_RANK_COUNT_KEY" in benchmark
        and "summary_rank_count" in benchmark
        and "summary.get(key, 0.0) + float" in benchmark
        and "PERCENT_SCALE * hits / attempts" in benchmark,
        "benchmark runner must aggregate cache summaries across MPI ranks",
    )
    _require(
        "FLOAT_PATTERN" in benchmark
        and "SUMMARY_VALUE_RE" in benchmark
        and "FLOAT_TOKEN_RE = re.compile(FLOAT_PATTERN)" in benchmark,
        "benchmark runner must parse decimal and scientific numeric logs consistently",
    )
    _require(
        "REPORT_SCHEMA_VERSION" in benchmark
        and "collect_run_provenance" in benchmark
        and "git_commit" in benchmark
        and "case_environment_overrides" in benchmark
        and '"provenance": collect_run_provenance(args.ablation_mode, benchmark_cases)' in benchmark,
        "benchmark runner must include report provenance metadata",
    )
    _require(
        "sample_variance_loop_time_seconds" in benchmark
        and "sample_stddev_loop_time_seconds" in benchmark
        and "MIN_SAMPLE_VARIANCE_COUNT" in benchmark,
        "benchmark runner must report repeat variance statistics",
    )
    _require(
        "MIN_REPEAT_COUNT" in benchmark
        and "validate_benchmark_options" in benchmark
        and "repeat_count must be at least" in benchmark
        and "lammps_command must not be empty" in benchmark
        and "parser.error(str(exc))" in benchmark,
        "benchmark runner must reject empty repeat sets and commands",
    )
    _require(
        "DEFAULT_RUN_TIMEOUT_SECONDS" in benchmark
        and "TIMEOUT_RETURN_CODE" in benchmark
        and "TIMEOUT_DETAIL_PREFIX" in benchmark
        and "timeout=run_timeout_seconds" in benchmark
        and "--run-timeout-seconds" in benchmark,
        "benchmark runner must bound each LAMMPS benchmark run",
    )
    _require(
        "ABLATION_MODE_CHOICES" in benchmark
        and "--ablation-mode" in benchmark
        and "benchmark_cases_for_ablation_mode" in benchmark
        and '"benchmark_cases": [case.name for case in benchmark_cases]' in benchmark,
        "benchmark runner must expose one-sided ablation runtime options",
    )
    _require(
        "check_isodelta_build_prereqs.py" in experiment
        and "check_isodelta_lammps_binary.py" in experiment
        and "run_isodelta_lammps_benchmark.py" in experiment
        and "check_isodelta_benchmark_report.py" in experiment
        and "check_isodelta_evidence_bundle.py" in experiment
        and "--min-enabled-cache-attempts" in experiment
        and "--min-enabled-cache-hits" in experiment
        and "--trace-evidence" in experiment
        and "--require-trace-model" in experiment
        and "--min-distinct-trace-models" in experiment
        and "should_run_bundle_gate" in experiment
        and "DEFAULT_MIN_DISTINCT_TRACE_MODELS" in experiment
        and "min_distinct_trace_models must be at least one" in experiment
        and "trace_evidence_paths count must be at least min_trace_count" in experiment
        and "duplicate trace evidence paths" in experiment
        and "duplicate required_trace_models" in experiment
        and "bundle_evidence.json" in experiment
        and "isodelta_experiment_report.json" in experiment,
        "experiment driver must connect all runtime validation stages",
    )
    _require(
        "ABLATION_MODE_CHOICES" in experiment
        and "--ablation-mode" in experiment
        and "ablation-benchmark" in experiment
        and "should_run_publishable_pair_gates" in experiment
        and "min_speedup requires paired ablation_mode" in experiment,
        "experiment driver must expose ablation mode without weakening paired gates",
    )
    _require(
        "validate_config" in experiment
        and "MIN_POSITIVE_TIMEOUT_SECONDS" in experiment
        and "binary_timeout_seconds must be positive" in experiment
        and "benchmark_timeout_seconds must be positive" in experiment
        and "--run-timeout-seconds" in experiment
        and "min_enabled_cache_hits cannot exceed min_enabled_cache_attempts" in experiment
        and "required_trace_models must not include empty names" in experiment,
        "experiment driver must reject meaningless experiment options",
    )
    _require(
        "EXPERIMENT_REPORT_SCHEMA_VERSION" in experiment
        and "collect_run_provenance" in experiment
        and "git_dirty" in experiment
        and '"provenance": collect_run_provenance()' in experiment,
        "experiment driver must include report provenance metadata",
    )
    _require(
        "DEFAULT_MAX_ABS_THERMO_DELTA" in report_check
        and "validate_report" in report_check
        and "validate_thresholds" in report_check
        and "PROVENANCE_KEY" in report_check
        and "_check_provenance" in report_check
        and "EXPECTED_REPORT_SCHEMA_VERSION" in report_check
        and "CASE_ENVIRONMENT_OVERRIDES_KEY" in report_check
        and "REQUIRED_CASE_ENVIRONMENT_OVERRIDES" in report_check
        and "SEVENN_ISODELTA_HALO_DISABLE" in report_check
        and "ENV_FLAG_FALSE_VALUES" in report_check
        and "env_flag_is_enabled" in report_check
        and "must be absent " in report_check
        and "or false for enabled case" in report_check
        and "test_validate_report_accepts_false_disable_env_in_enabled_case"
        in benchmark_report_test
        and "RUN_TIMEOUT_SECONDS_KEY" in report_check
        and "_check_run_timeout" in report_check
        and "SUMMARY_RUNS_KEY" in report_check
        and "_check_result_count" in report_check
        and "result_count" in report_check
        and "REPEAT_INDEX_KEY" in report_check
        and "_check_paired_runs" in report_check
        and "paired_repeat_count" in report_check
        and "LOOP_TIME_SECONDS_KEY" in report_check
        and "_check_timing_summary" in report_check
        and "timing_speedup_residual" in report_check
        and "must match mean loop times" in report_check
        and "SAMPLE_VARIANCE_LOOP_TIME_KEY" in report_check
        and "SAMPLE_STDDEV_LOOP_TIME_KEY" in report_check
        and "MIN_LOOP_TIME_KEY" in report_check
        and "MAX_LOOP_TIME_KEY" in report_check
        and "_require_optional_none" in report_check
        and "MAX_ABS_DELTA_KEY} must be nonnegative" in report_check
        and "PAIRED_COUNT_KEY} must be a positive integer" in report_check
        and "MIN_REQUIRED_PAIRED_THERMO_COUNT" in report_check
        and "MAX_PERCENT_VALUE" in report_check
        and "math.isfinite" in report_check
        and "min_speedup" in report_check,
        "benchmark report checker must gate thermo consistency and effect",
    )
    _require(
        "DEFAULT_MIN_ENABLED_CACHE_ATTEMPTS" in report_check
        and "DEFAULT_MIN_ENABLED_CACHE_HITS" in report_check
        and "min_enabled_cache_hits" in report_check,
        "benchmark report checker must gate cache activity evidence",
    )
    _require(
        "CACHE_HIT_RATE_TOLERANCE_PERCENT" in report_check
        and "hits <= attempts" in report_check
        and "must match hits / attempts" in report_check,
        "benchmark report checker must validate cache hit-rate consistency",
    )
    _require(
        "CACHE_COUNT_TOLERANCE" in report_check
        and "miss_count_sum" in report_check
        and "max_cache_count_residual" in report_check
        and "_as_nonnegative_count" in report_check
        and "MIN_REQUIRED_CACHE_ATTEMPTS" in report_check
        and "SUMMARY_RANK_COUNT_KEY" in report_check
        and "min_cache_summary_rank_count" in report_check
        and "must be a positive integer" in report_check
        and "must be a nonnegative integer" in report_check
        and "must match attempts - hits" in report_check,
        "benchmark report checker must validate miss-counter consistency",
    )
    _require(
        "DISABLED_CACHE_EXPECTED_HITS" in report_check
        and "baseline-disabled" in report_check
        and "miss_disabled" in report_check
        and "must match attempts" in report_check,
        "benchmark report checker must validate disabled baseline cache evidence",
    )
    _require(
        "REQUIRED_CACHE_MISS_KEYS" in report_check
        and "miss_comm-list-tag-order-changed" in report_check
        and "miss_index-tensor-shape-changed" in report_check
        and "verified_cache_miss_key_count" in report_check,
        "benchmark report checker must require miss breakdown coverage",
    )
    _require(
        "final_thermo_delta_vs_disabled_cache" in doc
        and "final_thermo_observables" in doc,
        "IsoDelta-Halo guide must document final thermo consistency fields",
    )
    _require(
        "provenance.report_schema_version" in doc
        and "provenance.git_commit" in doc
        and "case_environment_overrides" in doc,
        "IsoDelta-Halo guide must document report provenance fields",
    )
    _require(
        "report checker requires `provenance.report_schema_version`" in doc
        and "provenance.git_branch" in doc
        and "provenance.git_dirty" in doc,
        "IsoDelta-Halo guide must document provenance validation",
    )
    _require(
        "SEVENN_PRINT_INFO=1" in doc
        and "SEVENN_ISODELTA_HALO_DISABLE=1" in doc
        and "no disable flag" in doc,
        "IsoDelta-Halo guide must document case runtime env provenance",
    )
    _require(
        "--ablation-mode baseline-disabled" in doc
        and "--ablation-mode isodelta-enabled" in doc
        and "--ablation-mode paired" in doc
        and "one-sided ablation" in doc,
        "IsoDelta-Halo guide must document ablation runtime options",
    )
    _require(
        "run_timeout_seconds" in doc
        and "report checker validates" in doc,
        "IsoDelta-Halo guide must document report timeout evidence",
    )
    _require(
        "results.*.repeat_index" in doc
        and "paired_repeat_count" in doc
        and "exactly one `baseline-disabled`" in doc,
        "IsoDelta-Halo guide must document paired repeat evidence",
    )
    _require(
        "results.*.loop_time_seconds" in doc
        and "recomputes case mean loop times" in doc
        and "min/max and sample variance/stddev" in doc
        and "summary.speedup_vs_disabled_cache" in doc,
        "IsoDelta-Halo guide must document timing consistency evidence",
    )
    _require(
        "summary.runs" in doc
        and "number of `results` rows" in doc,
        "IsoDelta-Halo guide must document result count consistency",
    )
    _require(
        "summary_rank_count" in doc
        and "recomputes" in doc
        and "aggregated hits and attempts" in doc,
        "IsoDelta-Halo guide must document MPI cache-summary aggregation",
    )
    _require(
        "`summary_rank_count` to be a positive integer" in doc,
        "IsoDelta-Halo guide must document summary rank-count validation",
    )
    _require(
        "scientific notation" in doc
        and "loop-time" in doc
        and "cache evidence" in doc,
        "IsoDelta-Halo guide must document numeric parser coverage",
    )
    _require(
        "attempts - hits" in doc
        and "miss breakdown table" in doc,
        "IsoDelta-Halo guide must document miss-counter consistency",
    )
    _require(
        "CUDA-aware MPI runs" in doc
        and "cached index tensors" in doc
        and "`miss_index-tensor-shape-changed`" in doc,
        "IsoDelta-Halo guide must document CUDA index tensor reuse guard",
    )
    _require(
        "whole nonnegative profiling counts" in doc
        and "at least one attempt" in doc,
        "IsoDelta-Halo guide must document cache counter count validation",
    )
    _require(
        "miss_disabled` equal to `attempts`" in doc
        and "zero hit rate" in doc,
        "IsoDelta-Halo guide must document disabled baseline cache evidence",
    )
    _require(
        "sample_variance_loop_time_seconds" in doc
        and "sample_stddev_loop_time_seconds" in doc,
        "IsoDelta-Halo guide must document repeat variance fields",
    )
    _require(
        "`--repeat` must be at least 1" in doc
        and "`--binary-timeout-seconds` must be positive" in doc,
        "IsoDelta-Halo guide must document experiment option sanity checks",
    )
    _require(
        "`--run-timeout-seconds`" in doc
        and "`--benchmark-timeout-seconds` must be positive" in doc,
        "IsoDelta-Halo guide must document benchmark run timeouts",
    )
    _require(
        "check_isodelta_benchmark_report.py" in doc
        and "--max-abs-thermo-delta" in doc
        and "--min-speedup" in doc,
        "IsoDelta-Halo guide must document the benchmark report checker",
    )
    _require(
        "--min-enabled-cache-attempts" in doc
        and "--min-enabled-cache-hits" in doc,
        "IsoDelta-Halo guide must document cache activity gates",
    )
    _require(
        "run_isodelta_experiment.py" in doc
        and "isodelta_experiment_report.json" in doc,
        "IsoDelta-Halo guide must document the end-to-end experiment driver",
    )
    _require(
        "Trace-specific gates" in doc
        and "no trace evidence files are supplied" in doc
        and "trace evidence path is duplicated" in doc
        and "required trace model label is empty or duplicated" in doc,
        "IsoDelta-Halo guide must document experiment bundle fail-fast checks",
    )
    _require(
        "SUPPORTED_CASE_KINDS = frozenset((\"sevennet_lammps\", \"external_pair\", \"trace_only\"))"
        in cluster_suite
        and "DEFAULT_EXPECTED_GPU_COUNT = 8" in cluster_suite
        and "DEFAULT_REQUIRED_MODELS = (\"SevenNet\", \"MACE\", \"NequIP\")"
        in cluster_suite
        and "DEFAULT_REQUIRE_ARTIFACT_SHA256" in cluster_suite
        and "FINAL_PAPER_REQUIRED_MODELS" in cluster_suite
        and "FINAL_PAPER_PAIRED_CASE_KINDS" in cluster_suite
        and "FINAL_PAPER_MIN_REPEAT_COUNT" in cluster_suite
        and "FINAL_PAPER_READINESS_CHECK_NAMES" in cluster_suite
        and "UNRESOLVED_TEMPLATE_MARKERS" in cluster_suite
        and "READINESS_SCHEMA_VERSION" in cluster_suite
        and "ARTIFACT_PREPARATION_SCHEMA_VERSION" in cluster_suite
        and "PREFLIGHT_REPORT_SCHEMA_VERSION" in cluster_suite
        and "PIPELINE_REPORT_SCHEMA_VERSION" in cluster_suite
        and "ARTIFACT_PREPARATION_REPORT_NAME" in cluster_suite
        and "PREFLIGHT_REPORT_NAME" in cluster_suite
        and "PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME" in cluster_suite
        and "PIPELINE_REPORT_NAME" in cluster_suite
        and "STAGE_REPORT_FINGERPRINTS_KEY" in cluster_suite
        and "OUTPUT_BUNDLE_VERIFICATION_KEY" in cluster_suite
        and "_pipeline_stage_report_fingerprints" in cluster_suite
        and "SHA256_HEX_LENGTH" in cluster_suite
        and "SHA256_HEX_PATTERN" in cluster_suite
        and "require_artifact_sha256" in cluster_suite
        and "build_readiness_report" in cluster_suite
        and "--readiness-check" in cluster_suite
        and "prepare_artifacts" in cluster_suite
        and "--prepare-artifacts" in cluster_suite
        and "run_preflight_only" in cluster_suite
        and "--preflight-only" in cluster_suite
        and "--preflight-output" in cluster_suite
        and "run_pipeline" in cluster_suite
        and "--pipeline" in cluster_suite
        and "--pipeline-report" in cluster_suite
        and "verify_pipeline_report" in cluster_suite
        and "--verify-pipeline-report" in cluster_suite
        and "PREFLIGHT_OUTPUT" in cluster_suite
        and "validate_gpu_count" in cluster_suite
        and "download_artifact" in cluster_suite
        and "DOWNLOAD_PROGRESS_INTERVAL_BYTES" in cluster_suite
        and "_update_download_progress" in cluster_suite
        and "_emit_download_progress" in cluster_suite
        and "download_progress" in cluster_suite
        and "validate_required_artifacts_available" in cluster_suite
        and "skipped_optional_missing" in cluster_suite
        and "skip_downloads_would_fail" in cluster_suite
        and "sha256_file" in cluster_suite
        and "write_speedup_svg" in cluster_suite
        and "build_correlation_rows" in cluster_suite
        and "validate_suite_evidence" in cluster_suite
        and "suite_evidence" in cluster_suite
        and "build_run_plan" in cluster_suite
        and "write_run_plan" in cluster_suite
        and "PLAN_REPORT_NAME" in cluster_suite
        and "MANIFEST_SNAPSHOT_NAME" in cluster_suite
        and "DEFAULT_SLURM_JOB_NAME" in cluster_suite
        and "write_slurm_script" in cluster_suite
        and "--write-slurm-script" in cluster_suite
        and "#SBATCH --gres=gpu:" in cluster_suite
        and "COMMON_ARGS" in cluster_suite
        and "PIPELINE_OUTPUT" in cluster_suite
        and "--pipeline-report" in cluster_suite
        and ' --verify-pipeline-report "$PIPELINE_OUTPUT"' in cluster_suite
        and "Re-open the finished pipeline report" in cluster_suite
        and "--write-slurm-script cannot be combined with --collect-only" in cluster_suite
        and "DEFAULT_PREFLIGHT_TIMEOUT_SECONDS" in cluster_suite
        and "preflight_command" in cluster_suite
        and "preflight_env" in cluster_suite
        and "preflight_timeout_seconds" in cluster_suite
        and "MODE_CONTROL_ENV_KEYS" in cluster_suite
        and "COMMAND_ENV_SNAPSHOT_KEYS" in cluster_suite
        and "command_environment_snapshot" in cluster_suite
        and "tracked_env" in cluster_suite
        and "cwd=str(cwd)" in cluster_suite
        and "MODE_CONTROLS_KEY" in cluster_suite
        and "case_mode_control_record" in cluster_suite
        and "ENV_FLAG_FALSE_VALUES" in cluster_suite
        and "ENV_FLAG_FALSE_VALUES_KEY" in cluster_suite
        and "_as_env_value" in cluster_suite
        and "env_flag_is_enabled" in cluster_suite
        and "_validate_external_timing_mode_controls" in cluster_suite
        and "external_pair_mode_controls" in cluster_suite
        and '"case_mode_controls"' in cluster_suite
        and "run_case_preflight" in cluster_suite
        and "preflight.stdout.log" in cluster_suite
        and "test_preflight_only_downloads_artifacts_and_runs_case_checks" in cluster_suite_test
        and "test_download_artifact_prints_terminal_progress_when_requested"
        in cluster_suite_test
        and "test_preflight_only_reports_failed_case_check" in cluster_suite_test
        and "test_pipeline_stops_when_readiness_fails" in cluster_suite_test
        and "test_pipeline_stops_when_preflight_fails" in cluster_suite_test
        and "test_pipeline_reports_bundle_verification_failure" in cluster_suite_test
        and "--verify-pipeline-report" in cluster_suite_test
        and "Re-open the finished pipeline report" in cluster_suite_test
        and "test_manifest_validation_rejects_enabled_external_pair_disable_env" in cluster_suite_test
        and "test_manifest_validation_accepts_false_enabled_disable_env"
        in cluster_suite_test
        and "test_manifest_validation_accepts_empty_enabled_disable_env"
        in cluster_suite_test
        and "ENV_FLAG_FALSE_VALUES_KEY" in cluster_suite_test
        and "test_external_timing_report_rejects_mismatched_mode_controls" in cluster_suite_test
        and "test_external_timing_report_requires_repeat_command_records" in cluster_suite_test
        and "test_external_timing_report_rejects_failed_command_record" in cluster_suite_test
        and "test_external_timing_report_rejects_mutated_command_log" in cluster_suite_test
        and "test_command_records_include_cwd_and_tracked_environment" in cluster_suite_test
        and "test_verify_output_bundle_rejects_mismatched_command_fingerprints" in cluster_suite_test
        and "test_verify_output_bundle_rejects_mutated_source_evidence" in cluster_suite_test
        and "test_verify_output_bundle_rejects_mutated_external_command_log" in cluster_suite_test
        and "test_verify_output_bundle_rejects_semantically_invalid_svg_artifact"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_speedup_svg_label_drift"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_scatter_svg_point_count_drift"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_mismatched_artifact_index_path"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_case_summary_value_drift"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_correlation_value_drift"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_command_timing_value_drift"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_repeat_timing_value_drift"
        in cluster_suite_test
        and "test_run_suite_verifies_output_bundle_after_writing" in cluster_suite_test
        and "test_write_slurm_script_rejects_collect_only_pipeline_launcher" in cluster_suite_test
        and "CASE_STATUS_REUSED" in cluster_suite
        and "PASSING_CASE_STATUSES" in cluster_suite
        and "try_reuse_case_outputs" in cluster_suite
        and "--reuse-passed" in cluster_suite
        and "ENVIRONMENT_SNAPSHOT_NAME" in cluster_suite
        and "ENVIRONMENT_PACKAGE_NAMES" in cluster_suite
        and "ENVIRONMENT_VARIABLE_NAMES" in cluster_suite
        and "collect_environment_snapshot" in cluster_suite
        and "write_environment_snapshot" in cluster_suite
        and "environment_snapshot.json" in cluster_suite
        and "EVIDENCE_FINGERPRINTS_KEY" in cluster_suite
        and "evidence_fingerprints" in cluster_suite
        and "_require_evidence_fingerprint_matches" in cluster_suite
        and "verified_evidence_file_count" in cluster_suite
        and "command_log_fingerprints" in cluster_suite
        and "optional_file_fingerprint" in cluster_suite
        and "verify_output_bundle" in cluster_suite
        and "--verify-output-bundle" in cluster_suite
        and "SUMMARY_REPORT_NAME" in cluster_suite
        and "_require_command_record_alignment" in cluster_suite
        and "verified_command_record_count" in cluster_suite
        and "must align by name" in cluster_suite
        and '"stdout": optional_file_fingerprint' in cluster_suite
        and '"stderr": optional_file_fingerprint' in cluster_suite
        and "manifest_record" in cluster_suite
        and "generated_artifact_record" in cluster_suite
        and "artifact_fingerprints" in cluster_suite
        and "optional_artifact_paths" in cluster_suite
        and '"preflight_report": config.output_dir / PREFLIGHT_REPORT_NAME' in cluster_suite
        and '"preflight_environment_snapshot": config.output_dir / PREFLIGHT_ENVIRONMENT_SNAPSHOT_NAME'
        in cluster_suite
        and '"run_plan": config.output_dir / PLAN_REPORT_NAME' in cluster_suite
        and "write_manifest_snapshot" in cluster_suite
        and "EXTERNAL_TIMING_SCHEMA_VERSION" in cluster_suite
        and "validate_external_timing_report" in cluster_suite
        and "_validate_external_timing_command_records" in cluster_suite
        and "_require_external_command_log_fingerprints" in cluster_suite
        and "COMMAND_LOG_FINGERPRINTS_KEY" in cluster_suite
        and "_require_external_timing_reports_from_summary" in cluster_suite
        and "verified_external_command_log_count" in cluster_suite
        and "REQUIRED_PAPER_ARTIFACT_NAMES" in cluster_suite
        and "_require_artifact_index_alignment" in cluster_suite
        and "verified_artifact_index_count" in cluster_suite
        and "_require_case_summary_cell_values" in cluster_suite
        and "_summary_correlations_by_metric_pair" in cluster_suite
        and "_format_csv_value" in cluster_suite
        and "PAPER_COMMAND_TIMING_COLUMNS" in cluster_suite
        and "_command_timing_rows" in cluster_suite
        and "_require_command_timing_csv" in cluster_suite
        and "_require_command_timing_markdown" in cluster_suite
        and "PAPER_REPEAT_TIMING_COLUMNS" in cluster_suite
        and "_repeat_timing_rows" in cluster_suite
        and "_repeat_timing_rows_from_summary" in cluster_suite
        and "_require_repeat_timing_csv" in cluster_suite
        and "_require_repeat_timing_markdown" in cluster_suite
        and "_require_paper_artifact_semantics" in cluster_suite
        and "verified_paper_artifact_semantic_count" in cluster_suite
        and "_require_svg_document" in cluster_suite
        and "_require_speedup_svg_semantics" in cluster_suite
        and "_require_scatter_svg_semantics" in cluster_suite
        and "_svg_element_count" in cluster_suite
        and "verify_output" in cluster_suite
        and "verifying output bundle" in cluster_suite
        and "CORRELATION_METRIC_PAIRS" in cluster_suite
        and "EXTERNAL_DISABLED_COMMAND_LABEL" in cluster_suite
        and "EXTERNAL_ENABLED_COMMAND_LABEL" in cluster_suite
        and "BASELINE_TIMES_SECONDS_KEY" in cluster_suite
        and "ENABLED_TIMES_SECONDS_KEY" in cluster_suite
        and "BASELINE_SAMPLE_VARIANCE_SECONDS_KEY" in cluster_suite
        and "_sample_variance" in cluster_suite
        and "_sample_stddev" in cluster_suite
        and "NORMAL_APPROX_95_CI_MULTIPLIER" in cluster_suite
        and "_mean_ci_half_width" in cluster_suite
        and "baseline_mean_95ci_half_width_seconds" in cluster_suite
        and "_speedup_ci_bounds" in cluster_suite
        and "speedup_95ci_lower_bound" in cluster_suite
        and "min_speedup_95ci_lower_bound" in cluster_suite
        and "validate_case_summary_thresholds" in cluster_suite
        and "must match baseline / enabled seconds" in cluster_suite
        and "must match raw timing samples" in cluster_suite
        and "--plan-only" in cluster_suite
        and "--plan-output" in cluster_suite
        and "case_summary.csv" in cluster_suite
        and "correlation.csv" in cluster_suite
        and "command_timing.csv" in cluster_suite
        and "command_timing.md" in cluster_suite
        and "repeat_timing.csv" in cluster_suite
        and "repeat_timing.md" in cluster_suite
        and "speedup_by_case.svg" in cluster_suite
        and "write_template" in cluster_suite
        and "ABLATION_MODE_CHOICES" in cluster_suite
        and "ABLATION_OVERRIDE_CASE_KINDS" in cluster_suite
        and "--ablation-mode-override" in cluster_suite
        and "_apply_ablation_mode_override" in cluster_suite
        and "_runtime_override_comment" in cluster_suite
        and "CLI runtime overrides" in cluster_suite
        and "runtime_overrides" in cluster_suite
        and "_has_one_sided_ablation_case" in cluster_suite
        and "one-sided ablation suite" in cluster_suite
        and "PIPELINE_REPORT_PASSED_STATUS_ERROR" in cluster_suite
        and "pipeline report status must be 'passed'" in cluster_suite
        and "PIPELINE_DRY_RUN_PASSED_ERROR" in cluster_suite
        and "PIPELINE_GPU_CHECK_SKIPPED_ERROR" in cluster_suite
        and "PIPELINE_GPU_MISMATCH_ALLOWED_ERROR" in cluster_suite
        and "PIPELINE_PREFLIGHT_GPU_CHECK_REQUIRED_ERROR" in cluster_suite
        and "PIPELINE_PREFLIGHT_GPU_COUNT_ERROR" in cluster_suite
        and "PIPELINE_ARTIFACT_PREPARATION_ARTIFACTS_ERROR" in cluster_suite
        and "PIPELINE_PLAN_MODES_ERROR" in cluster_suite
        and "PIPELINE_PLAN_CASES_ERROR" in cluster_suite
        and "PIPELINE_PLAN_REQUIRED_PAPER_OUTPUT_KEYS" in cluster_suite
        and "PAPER_ARTIFACT_COMMENTS" in cluster_suite
        and "GENERATED_ARTIFACT_COMMENT_KEY" in cluster_suite
        and "RUN_PLAN_REPORT_COMMENT" in cluster_suite
        and "PIPELINE_READINESS_CHECKS_ERROR" in cluster_suite
        and "PIPELINE_SUITE_REQUIRED_MODELS_ERROR" in cluster_suite
        and "PIPELINE_UNSUPPORTED_RUNTIME_OVERRIDE_ERROR" in cluster_suite
        and "PIPELINE_ONE_SIDED_RUNTIME_OVERRIDE_ERROR" in cluster_suite
        and "PIPELINE_ALLOWED_RUNTIME_OVERRIDE_KEYS" in cluster_suite
        and "PIPELINE_REQUIRED_MODE_KEYS" in cluster_suite
        and "REQUIRED_PIPELINE_STAGE_NAMES" in cluster_suite
        and "PIPELINE_REQUIRED_STAGES_ERROR" in cluster_suite
        and "PIPELINE_BUNDLE_VERIFICATION_REQUIRED_ERROR" in cluster_suite
        and "_as_json_bool" in cluster_suite
        and "_require_pipeline_report_modes" in cluster_suite
        and "_require_pipeline_readiness_report" in cluster_suite
        and "_require_pipeline_artifact_preparation_report" in cluster_suite
        and "_require_pipeline_plan_report" in cluster_suite
        and "_require_csv_artifact_comment" in cluster_suite
        and "_require_markdown_artifact_comment" in cluster_suite
        and "_svg_desc_content" in cluster_suite
        and "_require_pipeline_preflight_gpu_check" in cluster_suite
        and "_require_pipeline_runtime_overrides" in cluster_suite
        and "_require_pipeline_suite_metadata" in cluster_suite
        and "_require_pipeline_success_stages" in cluster_suite
        and "_resolve_present_fingerprint_path" in cluster_suite
        and "# CLI runtime overrides: ablation_mode=isodelta-enabled." in cluster_suite_test
        and "test_cli_ablation_override_is_recorded_in_run_plan" in cluster_suite_test
        and "test_verify_pipeline_report_rejects_failed_pipeline_status"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_dry_run_passed_mode"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_skipped_gpu_check_mode"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_allowed_gpu_mismatch_mode"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_failed_readiness_check"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_failed_artifact_preparation"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_gpu_skipped_run_plan"
        in cluster_suite_test
        and "test_verify_output_bundle_rejects_missing_generated_file_comment"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_preflight_skipped_gpu_check"
        in cluster_suite_test
        and "test_verify_pipeline_report_requires_final_paper_suite_models"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_unsupported_runtime_override"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_one_sided_runtime_override"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_shallow_passed_report"
        in cluster_suite_test
        and "test_verify_pipeline_report_requires_success_stage_fingerprints"
        in cluster_suite_test
        and "test_verify_pipeline_report_rejects_absent_success_stage_fingerprint"
        in cluster_suite_test
        and "test_verify_pipeline_report_requires_passed_bundle_verification"
        in cluster_suite_test
        and "TIMING_MODES_KEY" in cluster_suite
        and "_external_timing_modes_for_ablation" in cluster_suite
        and "_external_pair_command_requirement_errors" in cluster_suite
        and "_validate_one_sided_benchmark_report" in cluster_suite
        and "external_pair_final_paper_ablation_mode" in cluster_suite
        and "sevennet_final_paper_ablation_mode" in cluster_suite
        and "test_external_pair_enabled_ablation_omits_disabled_command"
        in cluster_suite_test
        and "test_external_pair_paired_mode_still_requires_both_commands"
        in cluster_suite_test
        and "required_models = [\"SevenNet\", \"MACE\", \"NequIP\"]" in cluster_suite,
        "cluster paper suite must orchestrate 8-GPU multi-model paper artifacts",
    )
    _require(
        "run_isodelta_cluster_paper_suite.py" in doc
        and "8-GPU cluster" in doc
        and "TOML manifest" in doc
        and "SevenNet, MACE, and NequIP" in doc
        and "`require_artifact_sha256 = true`" in doc
        and "64-character SHA-256 digest" in doc
        and "`--readiness-check`" in doc
        and "`--prepare-artifacts`" in doc
        and "`--preflight-only`" in doc
        and "`preflight_report.json`" in doc
        and "`download_progress`" in doc
        and "byte-level download progress" in doc
        and "`--pipeline`" in doc
        and "`pipeline_report.json`" in doc
        and "`--verify-pipeline-report`" in doc
        and "publication-success gate" in doc
        and "failed or planned `pipeline_report.json` exits" in doc
        and "full final-paper stage sequence" in doc
        and "non-skipped stage" in doc
        and "present report files" in doc
        and "skipped artifact-preparation" in doc
        and "pipeline `modes` as booleans" in doc
        and "`dry_run = true`" in doc
        and "`skip_gpu_check = true`" in doc
        and "`allow_gpu_mismatch = true`" in doc
        and "SevenNet/MACE/NequIP required-model scope" in doc
        and "well-formed manifest fingerprint" in doc
        and "Runtime overrides are limited to known keys" in doc
        and "`ablation_mode = \"paired\"`" in doc
        and "one-sided ablation override" in doc
        and "reopens the fingerprinted readiness report" in doc
        and "every final-paper readiness check" in doc
        and "present and passed" in doc
        and "reopens the fingerprinted artifact preparation report" in doc
        and "`dry_run = false`" in doc
        and "missing_required_artifacts" in doc
        and "required artifact SHA-256 digest" in doc
        and "reopens the fingerprinted run plan report" in doc
        and "final-paper execution modes" in doc
        and "checks `gpu_check_planned`" in doc
        and "required model case plans" in doc
        and "Generated paper artifacts are self-describing" in doc
        and "`artifact_comment` or `report_comment`" in doc
        and "Every generated table must keep its explanatory comment" in doc
        and "expected `<desc>` description" in doc
        and "reopens the fingerprinted preflight report" in doc
        and "checks `gpu_check`" in doc
        and "requested GPU count was actually detected" in doc
        and "changing only" in doc
        and "stage report fingerprints" in doc
        and "final bundle verification counts" in doc
        and "`artifact_preparation_report.json`" in doc
        and "does not probe GPUs" in doc
        and "uses `trace_only` instead of paired" in doc
        and "`min_speedup_95ci_lower_bound` gates" in doc
        and "case_summary.csv" in doc
        and "correlation.csv" in doc
        and "command_timing.csv" in doc
        and "repeat_timing.csv" in doc
        and "speedup_by_case.svg" in doc
        and "`ablation_mode = \"paired\"`" in doc
        and "`--ablation-mode-override`" in doc
        and "`suite.runtime_overrides`" in doc
        and "one-sided SevenNet" in doc
        and "external_pair ablation" in doc
        and "`baseline-disabled` needs `disabled_command`" in doc
        and "`isodelta-enabled` needs `enabled_command`" in doc
        and "SHA-256" in doc
        and "`--collect-only`" in doc,
        "IsoDelta-Halo guide must document the cluster paper suite",
    )
    _require(
        "--plan-only" in doc
        and "preflight plan" in doc
        and "The plan JSON records artifact existence and download intent" in doc
        and "expected benchmark/trace/timing outputs" in doc,
        "IsoDelta-Halo guide must document cluster preflight planning",
    )
    _require(
        "--write-slurm-script" in doc
        and "SLURM cluster" in doc
        and "`sbatch` file requests `expected_gpus`" in doc
        and "runs `--preflight-only`" in doc
        and "COMMON_ARGS" in doc
        and "CLI runtime overrides" in doc
        and "`--pipeline` path" in doc
        and "`--pipeline-report`" in doc
        and "`--verify-pipeline-report \"$PIPELINE_OUTPUT\"`" in doc
        and "one-sided ablation suite" in doc
        and "`--verify-output-bundle`" in doc
        and "stage fingerprints and final bundle verification" in doc
        and "`--collect-only` is not accepted" in doc
        and "PYTHON_BIN" in doc
        and "SUITE_RUNNER" in doc,
        "IsoDelta-Halo guide must document SLURM cluster launch generation",
    )
    _require(
        "--reuse-passed" in doc
        and "interrupted cluster job" in doc
        and "Reused rows are marked as `reused`" in doc
        and "current validation gates" in doc
        and "normal command" in doc
        and "execution path" in doc,
        "IsoDelta-Halo guide must document partial rerun reuse mode",
    )
    _require(
        "`preflight_command`" in doc
        and "preflight.stdout.log" in doc
        and "preflight.stderr.log" in doc
        and "`preflight_env`" in doc
        and "`preflight_timeout_seconds`" in doc,
        "IsoDelta-Halo guide must document cluster case preflight checks",
    )
    _require(
        "`disabled_env`" in doc
        and "`enabled_env`" in doc
        and "`SEVENN_ISODELTA_HALO_DISABLE`" in doc
        and "`case_mode_controls`" in doc
        and "external_pair mode controls" in doc
        and "truthy value" in doc
        and "empty string" in doc
        and "`env_flag_false_values`" in doc
        and "false-value parser list" in doc
        and "cache-enabled evidence" in doc
        and "matching the C++" in doc
        and "runtime parser" in doc,
        "IsoDelta-Halo guide must document external-pair mode controls",
    )
    _require(
        "external timing report" in doc
        and "`mode_controls`" in doc
        and "`env_flag_false_values`" in doc
        and "must match the manifest" in doc,
        "IsoDelta-Halo guide must document timing-report mode-control validation",
    )
    _require(
        "manifest SHA-256 digest" in doc
        and "isodelta_cluster_suite_manifest.toml" in doc
        and "snapshot path" in doc,
        "IsoDelta-Halo guide must document cluster manifest fingerprints",
    )
    _require(
        "`artifact_fingerprints`" in doc
        and "SHA-256 digest and byte size" in doc
        and "generated" in doc
        and "The sibling `artifacts` index must carry the same artifact names and paths" in doc
        and "human-facing path index points to a different file" in doc
        and "environment snapshot" in doc
        and "SVG figure" in doc
        and "manifest snapshot" in doc,
        "IsoDelta-Halo guide must document generated paper artifact fingerprints",
    )
    _require(
        "`preflight_report.json`" in doc
        and "`preflight_environment_snapshot.json`" in doc
        and "`isodelta_cluster_paper_plan.json`" in doc
        and "auxiliary pre-run artifacts are fingerprinted too" in doc
        and "records that absence explicitly" in doc,
        "IsoDelta-Halo guide must document optional pre-run artifact fingerprints",
    )
    _require(
        "`evidence_fingerprints`" in doc
        and "benchmark" in doc
        and "external timing report" in doc
        and "trace evidence files" in doc
        and "source-evidence SHA-256 digests" in doc
        and "input evidence was edited or lost" in doc,
        "IsoDelta-Halo guide must document source evidence fingerprints",
    )
    _require(
        "full pipeline" in doc
        and "executes the final-paper readiness gate" in doc
        and "stops before downloads" in doc
        and "preflight gate fails" in doc
        and "stops before plan, timing runs, or summary generation" in doc
        and "commands, timing runs, or summary generation" in doc
        and "output bundle verification fails" in doc
        and "fingerprint mismatch" in doc
        and "verifies the output bundle fingerprints" in doc
        and "Use `--reuse-passed`" in doc
        and "`--pipeline` after an interrupted run" in doc,
        "IsoDelta-Halo guide must document the one-command paper pipeline",
    )
    _require(
        "`command_log_fingerprints`" in doc
        and "stdout/stderr log path" in doc
        and "existence flag" in doc
        and "SHA-256 digest" in doc
        and "working directory" in doc
        and "`tracked_env`" in doc
        and "CUDA" in doc
        and "SLURM" in doc
        and "same command names, return codes, and stdout/stderr paths" in doc
        and "command" in doc
        and "records" in doc,
        "IsoDelta-Halo guide must document command log fingerprints",
    )
    _require(
        "--verify-output-bundle" in doc
        and "isodelta_cluster_paper_summary.json" in doc
        and "fails if any recorded artifact or" in doc
        and "command log fingerprint" in doc,
        "IsoDelta-Halo guide must document output bundle verification",
    )
    _require(
        "semantic paper-artifact checks" in doc
        and "`case_summary.csv` and `case_summary.md` must contain exactly one row per" in doc
        and "same cell values as `summary[\"cases\"]`" in doc
        and "`correlation.csv` must contain the configured metric-pair rows" in doc
        and "same values as" in doc
        and "`summary[\"correlations\"]`" in doc
        and "`command_timing.csv`/`command_timing.md` must" in doc
        and "`summary[\"commands\"]`" in doc
        and "`repeat_timing.csv`/`repeat_timing.md` must" in doc
        and "source timing evidence" in doc
        and "Each SVG figure must parse as an SVG document" in doc
        and "speedup chart must include every measured-speedup" in doc
        and "scatter plots must contain the same" in doc
        and "`environment_snapshot.json` must carry the expected snapshot schema" in doc
        and "matching hash" in doc,
        "IsoDelta-Halo guide must document semantic paper artifact verification",
    )
    _require(
        "`--verify-output-bundle` also reopens the archived" in doc
        and "nested `command_log_fingerprints`" in doc
        and "MACE/NequIP stdout/stderr logs cannot drift silently" in doc,
        "IsoDelta-Halo guide must document nested external timing log verification",
    )
    _require(
        "normal suite run also reopens the completed" in doc
        and "bundle verification failure returns a nonzero exit code" in doc,
        "IsoDelta-Halo guide must document automatic run-suite bundle verification",
    )
    _require(
        "`environment_snapshot.json`" in doc
        and "Git/Python provenance" in doc
        and "CUDA/SLURM environment variables" in doc
        and "package versions" in doc
        and "`nvidia-smi` GPU" in doc,
        "IsoDelta-Halo guide must document cluster environment snapshots",
    )
    _require(
        "external timing report" in doc
        and "repeat success counts" in doc
        and "`commands` array" in doc
        and "`case:disabled:N` and `case:enabled:N` command record" in doc
        and "tracked environment matching the manifest mode controls" in doc
        and "`command_log_fingerprints` array fingerprints every external" in doc
        and "suite validation rechecks those hashes" in doc
        and "`baseline_mean_seconds` and `enabled_mean_seconds`" in doc
        and "`speedup_vs_disabled_cache` to match baseline divided by enabled seconds" in doc,
        "IsoDelta-Halo guide must document external timing report validation",
    )
    _require(
        "`baseline_times_seconds` and `enabled_times_seconds`" in doc
        and "sample variance/stddev" in doc
        and "95% CI half-widths" in doc
        and "speedup_95ci_lower_bound" in doc
        and "min_speedup_95ci_lower_bound" in doc
        and "fail any case" in doc
        and "survives uncertainty" in doc
        and "summary table" in doc,
        "IsoDelta-Halo guide must document external timing repeat statistics",
    )
    _require(
        "min_distinct_trace_models" in doc
        and "`model` labels inside" in doc
        and "those trace files" in doc
        and "duplicated MACE evidence" in doc
        and "Artifact `required_by`" in doc,
        "IsoDelta-Halo guide must document suite-level cluster evidence gates",
    )
    _require(
        "Required artifacts must already exist or be downloadable" in doc
        and "`--skip-downloads`" in doc
        and "required artifact is missing" in doc
        and "`required = false`" in doc
        and "skipped rather than silently" in doc,
        "IsoDelta-Halo guide must document cluster artifact availability gates",
    )
    _require(
        "SEVENN_ISODELTA_HALO_DISABLE" in doc
        and "SEVENN_ISODELTA_HALO_PROFILE" in doc,
        "IsoDelta-Halo guide must document runtime controls",
    )
    _require(
        "boolean-style parsing" in doc
        and "Surrounding" in doc
        and "whitespace is ignored before parsing" in doc
        and "`0`" in doc
        and "`false`" in doc
        and "`no`" in doc
        and "`off`" in doc
        and "`SEVENN_ISODELTA_HALO_DISABLE=0`" in doc
        and "keeps the cache enabled" in doc
        and '`SEVENN_ISODELTA_HALO_DISABLE=" off "` keeps the cache enabled'
        in doc
        and "`SEVENN_ISODELTA_HALO_PROFILE=0` keeps profiling off" in doc,
        "IsoDelta-Halo guide must document explicit false runtime controls",
    )
    _require(
        "IsoDelta-Halo lightweight validation" in doc
        and ".github/workflows/isodelta-halo.yml" in doc
        and "`isodelta_validation_report.json`" in doc
        and "check_isodelta_goal_readiness.py" in doc
        and "`isodelta_goal_readiness_report.json`" in doc
        and "codex/isodelta-halo-runtime" in doc
        and "`--expected-branch`" in doc
        and "validate one checkout branch" in doc
        and "`tools/*isodelta*.py`" in doc
        and "`tests/unit_tests/test_isodelta*.py`" in doc
        and "completion-readiness audit" in doc
        and "run_isodelta_sync_gate.py" in doc
        and "`isodelta_sync_report.json`" in doc
        and "`validation_report_fingerprint`" in doc
        and "`validation_report_summary`" in doc
        and "`sync_command_summary`" in doc
        and "`push_branch_precondition`" in doc
        and "`validation_report_missing`" in doc
        and "`validation_report_invalid`" in doc
        and "expected schema version" in doc
        and "required validation command fields" in doc
        and "valid types and values" in doc
        and "zero failed validation command return codes" in doc
        and "`worktree_status`" in doc
        and "`--require-clean-worktree`" in doc
        and "`dirty_worktree`" in doc
        and "`git_provenance` fields" in doc
        and "`git ls-remote --heads`" in doc
        and "`remote_ref_verification`" in doc
        and "`remote_verification_failed`" in doc
        and "`--push-failure-bundle`" in doc
        and "`push_failure_bundle`" in doc
        and "`git bundle verify` `verify_returncode`" in doc
        and "SHA-256 digest" in doc
        and "byte size" in doc
        and "non-interactive `git push -u`" in doc
        and "uploads the same JSON validation report" in doc,
        "IsoDelta-Halo guide must document the CI validation workflow",
    )
    for miss_key in (
        "miss_disabled",
        "miss_no-cache",
        "miss_neighbor-list-rebuilt",
        "miss_shape-changed",
        "miss_index-tensor-shape-changed",
        "miss_tag-count-changed",
        "miss_tag-order-changed",
        "miss_comm-topology-changed",
        "miss_comm-list-tag-order-changed",
    ):
        _require(
            miss_key in doc,
            f"IsoDelta-Halo guide must document cache miss key: {miss_key}",
        )
    _require(
        "sendlist tag order" in doc,
        "IsoDelta-Halo guide must document tag-order guards",
    )
    _require(
        "isodelta_halo" in doc_index,
        "IsoDelta-Halo guide must be linked from the user guide index",
    )

    print("IsoDelta-Halo static checks passed.")


if __name__ == "__main__":
    main()
