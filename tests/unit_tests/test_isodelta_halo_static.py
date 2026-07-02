"""Static regression tests for the IsoDelta-Halo communication metadata cache.

These tests avoid optional SevenNet runtime dependencies so they can run in a
minimal checkout while still guarding the most important implementation rules.
"""

from __future__ import annotations

from pathlib import Path
import unittest


# Resolve from this test file instead of relying on the current working
# directory, which differs between local shells and automated runners.
REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "pair_e3gnn_parallel.cpp"
HEADER_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "pair_e3gnn_parallel.h"
COMM_BRICK_CPP_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "comm_brick.cpp"
COMM_BRICK_HEADER_PATH = REPO_ROOT / "sevenn" / "pair_e3gnn" / "comm_brick.h"


class IsoDeltaHaloStaticTest(unittest.TestCase):
    """Check that the first implementation stays metadata-only and conservative."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load source text once so individual tests stay focused."""
        cls.cpp = CPP_PATH.read_text(encoding="utf-8")
        cls.header = HEADER_PATH.read_text(encoding="utf-8")
        cls.comm_brick_cpp = COMM_BRICK_CPP_PATH.read_text(encoding="utf-8")
        cls.comm_brick_header = COMM_BRICK_HEADER_PATH.read_text(encoding="utf-8")
        cls.combined = (
            cls.cpp
            + "\n"
            + cls.header
            + "\n"
            + cls.comm_brick_cpp
            + "\n"
            + cls.comm_brick_header
        )

    def test_named_comm_phase_count_replaces_raw_six(self) -> None:
        """The LAMMPS six-phase detail should be a named constant."""
        self.assertIn("kCommPhaseCount = 6", self.header)
        self.assertIn("kE3GnnCommPhaseLimit = 6", self.comm_brick_cpp)
        self.assertIn("kE3GnnCommPhaseLimitError", self.comm_brick_cpp)
        self.assertNotIn("[6]", self.cpp)
        self.assertNotIn("[6]", self.header)
        self.assertNotIn("nswap > 6", self.comm_brick_cpp)

    def test_cache_reuses_only_communication_metadata(self) -> None:
        """Force, message, embedding, and edge-geometry values must not be cached."""
        for term in (
            "force_cache",
            "message_cache",
            "embedding_cache",
            "geometry_cache",
            "edge_vec_cache",
        ):
            self.assertNotIn(term, self.combined)

        for line in self.combined.splitlines():
            if "comm_cache" not in line:
                continue
            lowered = line.lower()
            self.assertNotIn("edge_vec", lowered)
            self.assertNotIn("force", lowered)
            self.assertNotIn("message", lowered)
            self.assertNotIn("embedding", lowered)

    def test_cache_is_conservatively_gated(self) -> None:
        """Reuse should only happen after neighbor-list rebuild checks pass."""
        self.assertIn("try_reuse_comm_preprocess_cache", self.combined)
        self.assertIn("neighbor->ago <= kNeighborListJustBuiltAgo", self.cpp)
        self.assertIn("comm_cache_graph_tags", self.combined)
        self.assertIn("tag[atom_idx] != comm_cache_graph_tags[graph_idx]", self.cpp)
        self.assertIn("comm_topology_matches_cache", self.combined)
        self.assertIn("store_comm_topology_signature", self.combined)
        self.assertIn("comm-topology-changed", self.cpp)

    def test_cache_miss_rebuilds_and_stores_metadata(self) -> None:
        """A miss must rebuild first and store only when the cache is enabled."""
        self.assertIn(
            "comm_preprocess();\n    if (iso_delta_halo_enabled)",
            self.cpp,
        )
        self.assertIn(
            "if (iso_delta_halo_enabled) {\n      store_comm_preprocess_cache",
            self.cpp,
        )
        self.assertIn("clear_comm_preprocess_work();", self.cpp)

    def test_pack_forward_extra_map_uses_atom_index_key(self) -> None:
        """The extra graph map should use the communicated atom index, not loop i."""
        self.assertIn(
            "extra_graph_idx_map[list_i] = graph_size + extra_graph_idx_map.size();",
            self.cpp,
        )

    def test_cache_can_be_profiled_and_disabled(self) -> None:
        """Runtime toggles should expose fair baseline and profiling experiments."""
        self.assertIn("SEVENN_ISODELTA_HALO_DISABLE", self.cpp)
        self.assertIn("SEVENN_ISODELTA_HALO_PROFILE", self.cpp)
        self.assertNotIn("static constexpr const char *", self.header)
        self.assertIn("kIsoDeltaHaloPercentScale", self.cpp)
        self.assertIn("comm_cache_attempts++", self.cpp)
        self.assertIn("comm_cache_hits++", self.cpp)
        self.assertIn("record_comm_cache_miss", self.combined)
        self.assertIn("hit_rate_percent", self.cpp)

    def test_comm_brick_exposes_read_only_topology_accessors(self) -> None:
        """The pair cache should compare current CommBrick topology before reuse."""
        for accessor_name in (
            "e3gnn_nswap",
            "e3gnn_sendnum",
            "e3gnn_recvnum",
            "e3gnn_sendproc",
            "e3gnn_recvproc",
            "e3gnn_firstrecv",
        ):
            self.assertIn(accessor_name, self.comm_brick_header)
            self.assertIn(f"CommBrick::{accessor_name}", self.comm_brick_cpp)


if __name__ == "__main__":
    unittest.main()
