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

    def test_sources_do_not_contain_temporary_markers(self) -> None:
        """The implementation should not carry TODO or temporary-work markers."""
        self.assertNotIn("TODO", self.combined)
        self.assertNotIn("temporary", self.combined.lower())

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
        self.assertIn("kCommCacheMissReasonCount = 9", self.header)
        self.assertIn("neighbor->ago <= kNeighborListJustBuiltAgo", self.cpp)
        self.assertIn("comm_cache_graph_tags", self.combined)
        self.assertIn("tag[atom_idx] != comm_cache_graph_tags[graph_idx]", self.cpp)
        self.assertIn("comm_topology_matches_cache", self.combined)
        self.assertIn("store_comm_topology_signature", self.combined)
        self.assertIn("comm-topology-changed", self.cpp)
        self.assertIn("comm_list_tags_match_cache", self.combined)
        self.assertIn("store_comm_list_tag_signature", self.combined)
        self.assertIn("comm-list-tag-order-changed", self.cpp)
        self.assertIn("index-tensor-shape-changed", self.cpp)

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

    def test_cache_store_requires_cacheable_topology(self) -> None:
        """A rebuilt cache should be marked valid only for captured topology."""
        self.assertIn("current_comm_topology_is_cacheable", self.combined)
        self.assertIn("if (!current_comm_topology_is_cacheable())", self.cpp)
        self.assertIn("current_nswap >= kNoActiveCommPhases", self.cpp)
        self.assertIn("current_nswap <= kCommPhaseCount", self.cpp)

    def test_cache_misses_invalidate_stale_metadata(self) -> None:
        """A failed reuse attempt should not leave stale metadata marked valid."""
        self.assertIn("invalidate_comm_preprocess_cache", self.combined)
        self.assertIn("comm_cache_valid = false;", self.cpp)
        self.assertIn("comm_cache_graph_tags.clear();", self.cpp)
        self.assertIn("comm_cache_nswap = kInactiveCommPhaseValue;", self.cpp)
        self.assertIn(
            "comm_cache_index_pack_forward_tensor[comm_phase] = torch::Tensor();",
            self.cpp,
        )
        self.assertGreaterEqual(
            self.cpp.count("invalidate_comm_preprocess_cache();"),
            7,
        )

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
        self.assertIn("kBytesPerMebibyte", self.cpp)
        self.assertIn("kFloatElementBytes", self.cpp)
        self.assertNotIn("(1024 * 1024)", self.cpp)
        self.assertNotIn("x_dim * n * 4", self.cpp)
        self.assertIn("MEM use after backward(MiB)", self.cpp)
        self.assertIn("send size(MiB)", self.cpp)
        self.assertNotIn("send size(MB)", self.cpp)
        self.assertIn("comm_cache_attempts++", self.cpp)
        self.assertIn("comm_cache_hits++", self.cpp)
        self.assertIn("record_comm_cache_miss", self.combined)
        self.assertIn("hit_rate_percent", self.cpp)

    def test_cached_index_tensors_own_their_memory(self) -> None:
        """Cached tensor views should not borrow vectors cleared on later steps."""
        self.assertIn("make_owned_index_tensor", self.cpp)
        self.assertIn(".clone()\n      .to(target_device)", self.cpp)
        self.assertIn("kEmptyIndexTensorLength", self.cpp)
        self.assertIn("if (index_map.empty())", self.cpp)
        self.assertIn("torch::empty({kEmptyIndexTensorLength}, INTEGER_TYPE)", self.cpp)
        self.assertNotIn("torch::from_blob(idx_map_forward.data()", self.cpp)
        self.assertNotIn("torch::from_blob(upmap.data()", self.cpp)
        self.assertNotIn("torch::from_blob(idx_map_reverse.data()", self.cpp)

    def test_cached_index_tensors_match_cached_vectors(self) -> None:
        """CUDA index tensor reuse should be gated by cached vector lengths."""
        self.assertIn("cached_comm_tensors_match_vectors", self.combined)
        self.assertIn("index_tensor_matches_vector", self.cpp)
        self.assertIn("kIndexTensorRank", self.cpp)
        self.assertIn("kIndexTensorLengthDimension", self.cpp)
        self.assertIn("index_tensor.defined()", self.cpp)
        self.assertIn("comm_cache_index_pack_forward_tensor", self.cpp)
        self.assertIn("comm_cache_index_unpack_forward_tensor", self.cpp)
        self.assertIn("comm_cache_index_unpack_reverse_tensor", self.cpp)
        self.assertIn(
            "if (!cached_comm_tensors_match_vectors())",
            self.cpp,
        )
        self.assertIn(
            "record_comm_cache_miss(CommCacheMissReason::kIndexTensorShapeChanged);",
            self.cpp,
        )

    def test_comm_preprocess_requires_comm_brick(self) -> None:
        """The parallel pair style should fail clearly without CommBrick."""
        self.assertIn("kIsoDeltaHaloCommBrickRequiredError", self.cpp)
        self.assertIn("if (comm_brick == nullptr)", self.cpp)
        self.assertIn(
            "error->all(FLERR, kIsoDeltaHaloCommBrickRequiredError)",
            self.cpp,
        )

    def test_comm_brick_exposes_read_only_topology_accessors(self) -> None:
        """The pair cache should compare current CommBrick topology before reuse."""
        for accessor_name in (
            "e3gnn_nswap",
            "e3gnn_sendnum",
            "e3gnn_recvnum",
            "e3gnn_sendproc",
            "e3gnn_recvproc",
            "e3gnn_firstrecv",
            "e3gnn_sendlist_atom",
        ):
            self.assertIn(accessor_name, self.comm_brick_header)
            self.assertIn(f"CommBrick::{accessor_name}", self.comm_brick_cpp)

    def test_inactive_comm_phases_are_initialized(self) -> None:
        """Inactive phases should not read uninitialized CommBrick proc ids."""
        self.assertIn("notify_proc_ids(", self.cpp)
        self.assertIn("int active_phase_count", self.cpp)
        self.assertIn("kNoActiveCommPhases = 0", self.header)
        self.assertIn("bounded_active_phase_count", self.cpp)
        self.assertIn(
            "std::min(std::max(active_phase_count, kNoActiveCommPhases)",
            self.cpp,
        )
        self.assertIn("iswap < bounded_active_phase_count", self.cpp)
        self.assertIn(
            "active_phase ? sendproc[iswap] : kInactiveCommPhaseValue",
            self.cpp,
        )
        self.assertIn(
            "active_phase ? recvproc[iswap] : kInactiveCommPhaseValue",
            self.cpp,
        )
        self.assertIn(
            "pair->notify_proc_ids(sendproc, recvproc, nswap)",
            self.comm_brick_cpp,
        )


if __name__ == "__main__":
    unittest.main()
