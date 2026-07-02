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
REPORT_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_benchmark_report.py"
EVIDENCE_BUNDLE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_evidence_bundle.py"
PREREQ_PATH = REPO_ROOT / "tools" / "check_isodelta_build_prereqs.py"
BINARY_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_lammps_binary.py"
MLIP_TRACE_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_mlip_trace.py"
MLIP_TRACE_DEMO_PATH = REPO_ROOT / "tools" / "run_isodelta_mlip_trace_demo.py"
PATCH_SCRIPT_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "patch_lammps.sh"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "isodelta-halo.yml"
DOC_PATH = REPO_ROOT / "docs" / "source" / "user_guide" / "isodelta_halo.md"
DOC_INDEX_PATH = REPO_ROOT / "docs" / "source" / "user_guide" / "index.rst"


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
    report_check = _read(REPORT_CHECK_PATH)
    evidence_bundle_check = _read(EVIDENCE_BUNDLE_CHECK_PATH)
    prereq = _read(PREREQ_PATH)
    binary_check = _read(BINARY_CHECK_PATH)
    mlip_trace_check = _read(MLIP_TRACE_CHECK_PATH)
    mlip_trace_demo = _read(MLIP_TRACE_DEMO_PATH)
    patch_script = _read(PATCH_SCRIPT_PATH)
    workflow = _read(WORKFLOW_PATH)
    doc = _read(DOC_PATH)
    doc_index = _read(DOC_INDEX_PATH)
    combined = cpp + "\n" + header + "\n" + comm_brick_cpp + "\n" + comm_brick_header

    _require(
        "TODO" not in combined and "temporary" not in combined.lower(),
        "IsoDelta-Halo pair/comm sources must not carry temporary implementation markers",
    )
    _require("kCommPhaseCount = 6" in header, "named comm phase count missing")
    _require(
        "kCommCacheMissReasonCount = 8" in header,
        "cache miss reason count must include comm list tag changes",
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
        "comm_cache_attempts++" in cpp and "comm_cache_hits++" in cpp,
        "cache hit/attempt counters are missing",
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
        and "sevenn/pair_e3gnn/**" in workflow
        and "tools/run_isodelta_*.py" in workflow,
        "IsoDelta-Halo workflow must run the lightweight validation gate",
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
        "check_isodelta_lammps_binary.py" in doc,
        "IsoDelta-Halo guide must document the LAMMPS binary smoke checker",
    )
    _require(
        "MODEL_KEY = \"model\"" in mlip_trace_check
        and "MISS_COMM_TOPOLOGY_CHANGED" in mlip_trace_check
        and "MISS_COMM_LIST_TAG_ORDER_CHANGED" in mlip_trace_check
        and "TraceThresholds" in mlip_trace_check
        and "validate_thresholds" in mlip_trace_check
        and "MAX_PERCENT_VALUE" in mlip_trace_check
        and "TRACE_COUNT_TOLERANCE" in mlip_trace_check
        and "TRACE_COUNT_RESIDUAL_KEY" in mlip_trace_check
        and "MODEL_AGNOSTIC_REQUIREMENTS_KEY" in mlip_trace_check
        and "REQUIRED_MODEL_AGNOSTIC_REQUIREMENTS" in mlip_trace_check
        and "must be true" in mlip_trace_check
        and "math.isfinite" in mlip_trace_check
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
        "REPORT_SCHEMA_VERSION" in benchmark
        and "collect_run_provenance" in benchmark
        and "git_commit" in benchmark
        and "case_environment_overrides" in benchmark
        and '"provenance": collect_run_provenance()' in benchmark,
        "benchmark runner must include report provenance metadata",
    )
    _require(
        "sample_variance_loop_time_seconds" in benchmark
        and "sample_stddev_loop_time_seconds" in benchmark
        and "MIN_SAMPLE_VARIANCE_COUNT" in benchmark,
        "benchmark runner must report repeat variance statistics",
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
        and "bundle_evidence.json" in experiment
        and "isodelta_experiment_report.json" in experiment,
        "experiment driver must connect all runtime validation stages",
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
        and "must match attempts - hits" in report_check,
        "benchmark report checker must validate miss-counter consistency",
    )
    _require(
        "REQUIRED_CACHE_MISS_KEYS" in report_check
        and "miss_comm-list-tag-order-changed" in report_check
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
        "summary_rank_count" in doc
        and "recomputes" in doc
        and "aggregated hits and attempts" in doc,
        "IsoDelta-Halo guide must document MPI cache-summary aggregation",
    )
    _require(
        "attempts - hits" in doc
        and "miss breakdown table" in doc,
        "IsoDelta-Halo guide must document miss-counter consistency",
    )
    _require(
        "sample_variance_loop_time_seconds" in doc
        and "sample_stddev_loop_time_seconds" in doc,
        "IsoDelta-Halo guide must document repeat variance fields",
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
        "SEVENN_ISODELTA_HALO_DISABLE" in doc
        and "SEVENN_ISODELTA_HALO_PROFILE" in doc,
        "IsoDelta-Halo guide must document runtime controls",
    )
    _require(
        "IsoDelta-Halo lightweight validation" in doc
        and ".github/workflows/isodelta-halo.yml" in doc,
        "IsoDelta-Halo guide must document the CI validation workflow",
    )
    for miss_key in (
        "miss_disabled",
        "miss_no-cache",
        "miss_neighbor-list-rebuilt",
        "miss_shape-changed",
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
