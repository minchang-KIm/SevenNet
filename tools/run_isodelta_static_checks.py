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
    doc = _read(DOC_PATH)
    doc_index = _read(DOC_INDEX_PATH)
    combined = cpp + "\n" + header

    _require("kCommPhaseCount = 6" in header, "named comm phase count missing")
    _require("[6]" not in cpp, "raw six-phase array/magic count remains in cpp")
    _require("[6]" not in header, "raw six-phase array/magic count remains in header")

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
        "SEVENN_ISODELTA_HALO_DISABLE" in doc
        and "SEVENN_ISODELTA_HALO_PROFILE" in doc,
        "IsoDelta-Halo guide must document runtime controls",
    )
    _require(
        "isodelta_halo" in doc_index,
        "IsoDelta-Halo guide must be linked from the user guide index",
    )

    print("IsoDelta-Halo static checks passed.")


if __name__ == "__main__":
    main()
