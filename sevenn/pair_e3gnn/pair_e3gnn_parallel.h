/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
http://lammps.sandia.gov, Sandia National Laboratories
Steve Plimpton, sjplimp@sandia.gov

Copyright (2003) Sandia Corporation.  Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
certain rights in this software.  This software is distributed under
the GNU General Public License.

See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(e3gnn/parallel, PairE3GNNParallel)

#else

#ifndef LMP_PAIR_E3GNN_PARALLEL
#define LMP_PAIR_E3GNN_PARALLEL

#include "pair.h"

#include <array>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {
class PairE3GNNParallel : public Pair {
private:
  // LAMMPS brick communication uses six directional phases: x/y/z forward
  // and reverse sweeps. Keeping this named avoids hidden phase-count coupling.
  static constexpr int kCommPhaseCount = 6;
  static constexpr int kNeighborListJustBuiltAgo = 0;
  static constexpr int kNoActiveCommPhases = 0;

  enum class CommCacheMissReason {
    kDisabled = 0,
    kNoCache,
    kNeighborListRebuilt,
    kShapeChanged,
    kIndexTensorShapeChanged,
    kTagCountChanged,
    kTagOrderChanged,
    kCommTopologyChanged,
    kCommListTagOrderChanged,
  };
  static constexpr int kCommCacheMissReasonCount = 9;
  static_assert(
      static_cast<int>(CommCacheMissReason::kCommListTagOrderChanged) + 1 ==
          kCommCacheMissReasonCount,
      "CommCacheMissReason count must match the enum entries.");
  static constexpr int kInactiveCommPhaseValue = -1;

  double cutoff;
  double cutoff_square;
  std::vector<torch::jit::Module> model_list;
  torch::Device device = torch::kCPU;
  torch::Device device_comm = torch::kCPU;
  torch::Device get_cuda_device();
  bool use_cuda_mpi;

  // Communication state is rebuilt for each MD step unless IsoDelta-Halo
  // safely restores the metadata that only depends on stable halo topology.
  int x_dim; // to determine per atom data size
  int graph_size;
  torch::Tensor x_comm; // x_local + x_ghost + x_comm_extra

  void comm_preprocess();
  bool comm_preprocess_done = false;

  // Per-step communication index maps populated by comm_preprocess().
  std::unordered_map<int, long> extra_graph_idx_map;
  // To use scatter, store long instead of int
  // array of vector
  std::vector<long> comm_index_pack_forward[kCommPhaseCount];
  std::vector<long> comm_index_unpack_forward[kCommPhaseCount];
  std::vector<long> comm_index_unpack_reverse[kCommPhaseCount];

  // its size is kCommPhaseCount and initialized at comm_preprocess()
  torch::Tensor comm_index_pack_forward_tensor[kCommPhaseCount];
  torch::Tensor comm_index_unpack_forward_tensor[kCommPhaseCount];
  torch::Tensor comm_index_unpack_reverse_tensor[kCommPhaseCount];

  // IsoDelta-Halo cache: only communication metadata and CUDA index tensors are
  // reused. Edge vectors, embeddings, messages, energies, and forces are still
  // recomputed every timestep by the original SevenNet path.
  bool comm_cache_valid = false;
  bool iso_delta_halo_enabled = true;
  bool iso_delta_halo_profile = false;
  long long comm_cache_attempts = 0;
  long long comm_cache_hits = 0;
  std::array<long long, kCommCacheMissReasonCount> comm_cache_misses = {};
  int comm_cache_nlocal = 0;
  int comm_cache_ghost_node_num = 0;
  int comm_cache_graph_size = 0;
  int comm_cache_nedges = 0;
  int comm_cache_nswap = 0;
  std::vector<tagint> comm_cache_graph_tags;
  std::array<int, kCommPhaseCount> comm_cache_sendnum = {};
  std::array<int, kCommPhaseCount> comm_cache_recvnum = {};
  std::array<int, kCommPhaseCount> comm_cache_sendproc = {};
  std::array<int, kCommPhaseCount> comm_cache_recvproc = {};
  std::array<int, kCommPhaseCount> comm_cache_firstrecv = {};
  std::vector<tagint> comm_cache_sendlist_tags[kCommPhaseCount];
  std::vector<tagint> comm_cache_recvlist_tags[kCommPhaseCount];
  std::unordered_map<int, long> comm_cache_extra_graph_idx_map;
  std::vector<long> comm_cache_index_pack_forward[kCommPhaseCount];
  std::vector<long> comm_cache_index_unpack_forward[kCommPhaseCount];
  std::vector<long> comm_cache_index_unpack_reverse[kCommPhaseCount];
  torch::Tensor comm_cache_index_pack_forward_tensor[kCommPhaseCount];
  torch::Tensor comm_cache_index_unpack_forward_tensor[kCommPhaseCount];
  torch::Tensor comm_cache_index_unpack_reverse_tensor[kCommPhaseCount];

  bool try_reuse_comm_preprocess_cache(int, int, int, const int *);
  void store_comm_preprocess_cache(int, int, int, const int *);
  void clear_comm_preprocess_work();
  void invalidate_comm_preprocess_cache();
  bool current_comm_topology_is_cacheable() const;
  bool comm_topology_matches_cache() const;
  bool cached_comm_tensors_match_vectors() const;
  void store_comm_topology_signature();
  bool comm_list_tags_match_cache() const;
  void store_comm_list_tag_signature();
  void record_comm_cache_miss(CommCacheMissReason);
  static bool comm_cache_miss_reason_is_valid(CommCacheMissReason);
  static const char *comm_cache_miss_reason_name(CommCacheMissReason);
  void print_comm_cache_summary() const;
  void validate_comm_phase(int comm_phase) const;

  // Per-step graph-index lookup borrowed from compute() while CommBrick fills
  // communication maps; clear_comm_preprocess_work() closes this borrow.
  int *tag_to_graph_idx_ptr = nullptr;

  int sendproc[kCommPhaseCount];
  int recvproc[kCommPhaseCount];

public:
  PairE3GNNParallel(class LAMMPS *);
  ~PairE3GNNParallel();

  // LAMMPS invokes this Pair interface; CommBrick calls the GNN-specific
  // communication helpers below during the halo exchange phases.
  void compute(int, int) override;
  void settings(int, char **) override;
  // read Atom type string from input script & related coeff
  void coeff(int, char **) override;
  void allocate();

  void pack_forward_init(int n, int *list, int comm_phase);
  void unpack_forward_init(int n, int first, int comm_phase);

  int pack_forward_comm_gnn(float *buf, int comm_phase);
  void unpack_forward_comm_gnn(float *buf, int comm_phase);
  int pack_reverse_comm_gnn(float *buf, int comm_phase);
  void unpack_reverse_comm_gnn(float *buf, int comm_phase);

  void init_style() override;
  double init_one(int, int) override;

  int get_x_dim();
  bool use_cuda_mpi_();
  bool is_comm_preprocess_done();
  void notify_proc_ids(const int *sendproc, const int *recvproc, int);

  bool print_info = false;
  int world_rank;
};

class DeviceBuffManager {
private:
  DeviceBuffManager() {}
  DeviceBuffManager(const DeviceBuffManager &);
  DeviceBuffManager &operator=(const DeviceBuffManager &);

  float *buf_send_device = nullptr;
  float *buf_recv_device = nullptr;
  int send_buf_size = 0;
  int recv_buf_size = 0;

public:
  static DeviceBuffManager &getInstance();
  void get_buffer(int, int, float *&, float *&, class Error *);

  ~DeviceBuffManager();
};
} // namespace LAMMPS_NS

#endif
#endif
