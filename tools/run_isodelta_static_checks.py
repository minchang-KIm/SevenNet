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
        "comm_preprocess();\n    store_comm_preprocess_cache" in cpp,
        "cache miss path does not rebuild then store metadata",
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

    print("IsoDelta-Halo static checks passed.")


if __name__ == "__main__":
    main()
