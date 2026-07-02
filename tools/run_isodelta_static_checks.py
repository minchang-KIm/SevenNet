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
PREREQ_PATH = REPO_ROOT / "tools" / "check_isodelta_build_prereqs.py"
BINARY_CHECK_PATH = REPO_ROOT / "tools" / "check_isodelta_lammps_binary.py"
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
    prereq = _read(PREREQ_PATH)
    binary_check = _read(BINARY_CHECK_PATH)
    doc = _read(DOC_PATH)
    doc_index = _read(DOC_INDEX_PATH)
    combined = cpp + "\n" + header + "\n" + comm_brick_cpp + "\n" + comm_brick_header

    _require("kCommPhaseCount = 6" in header, "named comm phase count missing")
    _require(
        "kCommCacheMissReasonCount = 8" in header,
        "cache miss reason count must include comm list tag changes",
    )
    _require("[6]" not in cpp, "raw six-phase array/magic count remains in cpp")
    _require("[6]" not in header, "raw six-phase array/magic count remains in header")
    for include_name in ("<cstring>", "<iostream>", "<list>", "<map>", "<set>"):
        _require(
            f"#include {include_name}" in cpp,
            f"pair_e3gnn_parallel.cpp must explicitly include {include_name}",
        )
    _require(
        "std::vector<long> &upmap = comm_index_unpack_forward[comm_phase];" in cpp,
        "CUDA unpack tensor creation must not copy the unpack vector first",
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
        "parse_final_thermo_observables" in benchmark
        and "final_thermo_delta_vs_disabled_cache" in benchmark,
        "benchmark runner must report final thermo consistency deltas",
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
        and "isodelta_experiment_report.json" in experiment,
        "experiment driver must connect all runtime validation stages",
    )
    _require(
        "DEFAULT_MAX_ABS_THERMO_DELTA" in report_check
        and "validate_report" in report_check
        and "min_speedup" in report_check,
        "benchmark report checker must gate thermo consistency and effect",
    )
    _require(
        "final_thermo_delta_vs_disabled_cache" in doc
        and "final_thermo_observables" in doc,
        "IsoDelta-Halo guide must document final thermo consistency fields",
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
        "miss_comm-topology-changed" in doc
        and "miss_comm-list-tag-order-changed" in doc
        and "sendlist tag order" in doc,
        "IsoDelta-Halo guide must document topology and tag-order guards",
    )
    _require(
        "isodelta_halo" in doc_index,
        "IsoDelta-Halo guide must be linked from the user guide index",
    )

    print("IsoDelta-Halo static checks passed.")


if __name__ == "__main__":
    main()
