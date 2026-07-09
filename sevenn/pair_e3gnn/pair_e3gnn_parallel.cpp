/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Yutack Park (SNU)
------------------------------------------------------------------------- */

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include "atom.h"
#include "comm.h"
#include "comm_brick.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
// #include "nvToolsExt.h"

#include "pair_e3gnn_parallel.h"
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <string>

#ifdef OMPI_MPI_H
#include "mpi-ext.h" //This should be included after mpi.h which is included in pair.h
#endif

using namespace LAMMPS_NS;

// Undefined reference; body in pair_e3gnn_oeq_autograd.cpp to be linked
extern void pair_e3gnn_oeq_register_autograd();

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

namespace {
// Keep profiling toggles local to this translation unit to avoid C++14 static
// data-member definitions while still naming every experiment-facing literal.
constexpr const char *kIsoDeltaHaloDisableEnv = "SEVENN_ISODELTA_HALO_DISABLE";
constexpr const char *kIsoDeltaHaloProfileEnv = "SEVENN_ISODELTA_HALO_PROFILE";
constexpr const char *kIsoDeltaHaloEnvFlagValueEmpty = "";
constexpr const char *kIsoDeltaHaloEnvFlagValueZero = "0";
constexpr const char *kIsoDeltaHaloEnvFlagValueFalse = "false";
constexpr const char *kIsoDeltaHaloEnvFlagValueNo = "no";
constexpr const char *kIsoDeltaHaloEnvFlagValueOff = "off";
constexpr const char *kIsoDeltaHaloCommBrickRequiredError =
    "IsoDelta-Halo e3gnn/parallel requires LAMMPS CommBrick communication";
constexpr const char *kIsoDeltaHaloGraphIndexRequiredError =
    "IsoDelta-Halo graph index map must be active before communication preprocessing";
constexpr const char *kIsoDeltaHaloCommPhaseRangeError =
    "IsoDelta-Halo communication phase index is out of range";
constexpr const char *kCudaSendBufferAllocationError =
    "PairE3GNNParallel: CUDA send buffer allocation failed";
constexpr const char *kCudaRecvBufferAllocationError =
    "PairE3GNNParallel: CUDA receive buffer allocation failed";
constexpr const char *kCudaPackForwardMemcpyError =
    "PairE3GNNParallel: CUDA pack-forward buffer copy failed";
constexpr const char *kCudaPackReverseMemcpyError =
    "PairE3GNNParallel: CUDA pack-reverse buffer copy failed";
constexpr const char *kE3GnnPayloadElementCountError =
    "PairE3GNNParallel: communication payload element count is out of range";
constexpr double kIsoDeltaHaloPercentScale = 100.0;
constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;
constexpr double kFloatElementBytes = static_cast<double>(sizeof(float));
constexpr int kMinimumFeatureWidth = 1;
constexpr int kMinimumPayloadAtomCount = 0;
constexpr int kSpatialDimension = 3;
constexpr int kXCoordinate = 0;
constexpr int kYCoordinate = 1;
constexpr int kZCoordinate = 2;
constexpr int kVoigtStressComponentCount = 6;
constexpr int kVoigtXX = 0;
constexpr int kVoigtYY = 1;
constexpr int kVoigtZZ = 2;
constexpr int kVoigtXY = 3;
constexpr int kVoigtYZ = 4;
constexpr int kVoigtZX = 5;
constexpr int kAtomTagIndexBase = 1;
constexpr int kInvalidGraphIndex = -1;
constexpr long long kEmptyIndexTensorLength = 0;
constexpr int kIndexTensorRank = 1;
constexpr int kIndexTensorLengthDimension = 0;

torch::Tensor make_owned_index_tensor(std::vector<long> &index_map,
                                      const torch::Device &target_device) {
  // from_blob borrows vector memory on CPU, so clone before moving to the
  // target device. The cache must outlive per-step vector cleanup.
  if (index_map.empty()) {
    return torch::empty({kEmptyIndexTensorLength}, INTEGER_TYPE)
        .to(target_device);
  }
  return torch::from_blob(
             index_map.data(),
             {static_cast<long long>(index_map.size())},
             INTEGER_TYPE)
      .clone()
      .to(target_device);
}

bool index_tensor_matches_vector(const torch::Tensor &index_tensor,
                                 const std::vector<long> &index_map) {
  return index_tensor.defined() && index_tensor.dim() == kIndexTensorRank &&
         index_tensor.size(kIndexTensorLengthDimension) ==
             static_cast<long long>(index_map.size());
}

void check_cuda_status(cudaError_t cuda_err, const char *context,
                       Error *error) {
  if (cuda_err == cudaSuccess) {
    return;
  }
  error->all(FLERR, std::string(context) + ": " +
                         std::string(cudaGetErrorString(cuda_err)));
}

int checked_e3gnn_payload_element_count(int feature_width, int atom_count,
                                        Error *error) {
  if (feature_width < kMinimumFeatureWidth ||
      atom_count < kMinimumPayloadAtomCount) {
    error->all(FLERR, kE3GnnPayloadElementCountError);
  }

  const long long element_count =
      static_cast<long long>(feature_width) * static_cast<long long>(atom_count);
  if (element_count > std::numeric_limits<int>::max()) {
    error->all(FLERR, kE3GnnPayloadElementCountError);
  }
  return static_cast<int>(element_count);
}

size_t checked_e3gnn_payload_byte_count(int payload_element_count) {
  return static_cast<size_t>(payload_element_count) * sizeof(float);
}

std::string normalize_iso_delta_halo_env_flag_value(const char *value) {
  if (value == nullptr) {
    return std::string();
  }
  const std::string raw_value(value);
  auto begin = raw_value.begin();
  auto end = raw_value.end();
  while (begin != end &&
         std::isspace(static_cast<unsigned char>(*begin)) != 0) {
    ++begin;
  }
  while (end != begin) {
    auto last_character = end;
    --last_character;
    if (std::isspace(static_cast<unsigned char>(*last_character)) == 0) {
      break;
    }
    end = last_character;
  }

  std::string normalized_value;
  for (auto character = begin; character != end; ++character) {
    normalized_value.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(*character))));
  }
  return normalized_value;
}

bool iso_delta_halo_env_flag_is_enabled(const char *env_name) {
  const char *value = std::getenv(env_name);
  if (value == nullptr) {
    return false;
  }
  const std::string normalized_value =
      normalize_iso_delta_halo_env_flag_value(value);
  return normalized_value != kIsoDeltaHaloEnvFlagValueEmpty &&
         normalized_value != kIsoDeltaHaloEnvFlagValueZero &&
         normalized_value != kIsoDeltaHaloEnvFlagValueFalse &&
         normalized_value != kIsoDeltaHaloEnvFlagValueNo &&
         normalized_value != kIsoDeltaHaloEnvFlagValueOff;
}
} // namespace

DeviceBuffManager &DeviceBuffManager::getInstance() {
  static DeviceBuffManager instance;
  return instance;
}

void DeviceBuffManager::get_buffer(int send_size, int recv_size,
                                   float *&buf_send_ptr, float *&buf_recv_ptr,
                                   Error *error) {
  if (send_size > send_buf_size) {
    cudaFree(buf_send_device);
    cudaError_t cuda_err =
        cudaMalloc(&buf_send_device, send_size * sizeof(float));
    check_cuda_status(cuda_err, kCudaSendBufferAllocationError, error);
    send_buf_size = send_size;
  }
  if (recv_size > recv_buf_size) {
    cudaFree(buf_recv_device);
    cudaError_t cuda_err =
        cudaMalloc(&buf_recv_device, recv_size * sizeof(float));
    check_cuda_status(cuda_err, kCudaRecvBufferAllocationError, error);
    recv_buf_size = recv_size;
  }
  buf_send_ptr = buf_send_device;
  buf_recv_ptr = buf_recv_device;
}

DeviceBuffManager::~DeviceBuffManager() {
  cudaFree(buf_send_device);
  cudaFree(buf_recv_device);
}

PairE3GNNParallel::PairE3GNNParallel(LAMMPS *lmp) : Pair(lmp) {
  // constructor

  const char *print_flag = std::getenv("SEVENN_PRINT_INFO");
  const char *print_both_flag = std::getenv("SEVENN_PRINT_BOTH_INFO");
  iso_delta_halo_enabled =
      !iso_delta_halo_env_flag_is_enabled(kIsoDeltaHaloDisableEnv);
  iso_delta_halo_profile =
      iso_delta_halo_env_flag_is_enabled(kIsoDeltaHaloProfileEnv);
  if (print_flag) {
    world_rank = comm->me;
    std::cout << "process rank: " << world_rank << " initialized" << std::endl;
    print_info = (world_rank == 0) || print_both_flag;
  }

  std::string device_name;
  const bool use_gpu = torch::cuda::is_available();

  comm_forward = 0;
  comm_reverse = 0;

  // OpenMPI detection
#ifdef OMPI_MPI_H
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    use_cuda_mpi = true;
  } else {
    use_cuda_mpi = false;
  }
#else
  use_cuda_mpi = false;
#endif
#else
  use_cuda_mpi = false;
#endif
  // use_cuda_mpi = use_gpu && use_cuda_mpi;
  // if (use_cuda_mpi) {
  if (use_gpu) {
    device = get_cuda_device();
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (std::getenv("OFF_E3GNN_PARALLEL_CUDA_MPI")) {
      use_cuda_mpi = false;
  }

  if (lmp->screen) {
    if (use_gpu && !use_cuda_mpi) {
      device_comm = torch::kCPU;
      fprintf(lmp->screen,
              "cuda-aware mpi not found, communicate via host device\n");
    } else {
      device_comm = device;
    }
    fprintf(lmp->screen, "PairE3GNNParallel using device : %s\n",
            device_name.c_str());
    fprintf(lmp->screen, "PairE3GNNParallel cuda-aware mpi: %s\n",
            use_cuda_mpi ? "True" : "False");
  }
  if (lmp->logfile) {
    if (use_gpu && !use_cuda_mpi) {
      device_comm = torch::kCPU;
      fprintf(lmp->logfile,
              "cuda-aware mpi not found, communicate via host device\n");
    } else {
      device_comm = device;
    }
    fprintf(lmp->logfile, "PairE3GNNParallel using device : %s\n",
            device_name.c_str());
    fprintf(lmp->logfile, "PairE3GNNParallel cuda-aware mpi: %s\n",
            use_cuda_mpi ? "True" : "False");
  }

  if (print_info) {
    std::cout << world_rank << " IsoDelta-Halo metadata cache: "
              << (iso_delta_halo_enabled ? "enabled" : "disabled")
              << std::endl;
    if (iso_delta_halo_profile) {
      std::cout << world_rank << " IsoDelta-Halo profiling enabled by "
                << kIsoDeltaHaloProfileEnv << std::endl;
    }
  }
}

torch::Device PairE3GNNParallel::get_cuda_device() {
  char *cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
  int num_gpus;
  int idx;
  int rank = comm->me;
  num_gpus = torch::cuda::device_count();
  idx = rank % num_gpus;
  if (print_info)
    std::cout << world_rank << " Available # of GPUs found: " << num_gpus
              << std::endl;
  cudaError_t cuda_err = cudaSetDevice(idx);
  if (cuda_err != cudaSuccess) {
    std::cerr << "E3GNN: Failed to set CUDA device: "
              << cudaGetErrorString(cuda_err) << std::endl;
  }
  return torch::Device(torch::kCUDA, idx);
}

PairE3GNNParallel::~PairE3GNNParallel() {
  print_comm_cache_summary();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
  }
}

int PairE3GNNParallel::get_x_dim() { return x_dim; }

bool PairE3GNNParallel::use_cuda_mpi_() { return use_cuda_mpi; }

bool PairE3GNNParallel::is_comm_preprocess_done() {
  return comm_preprocess_done;
}

void PairE3GNNParallel::compute(int eflag, int vflag) {
  /*
     Graph build on cpu
  */
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;
  if (vflag_atom) {
    error->all(FLERR, "atomic stress is not supported\n");
  }

  if (atom->tag_consecutive() == 0) {
    error->all(FLERR, "Pair e3gnn requires consecutive atom IDs");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = list->inum; // same as nlocal
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *ilist = list->ilist;
  int inum = list->inum;

  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    error->all(FLERR, "e3gnn/parallel: comm style should be brick & from "
                      "modified code of comm_brick");
  }

  bigint natoms = atom->natoms;

  // tag ignore PBC
  tagint *tag = atom->tag;

  // Store graph_idx from local to known ghost atoms inside the cutoff. The tag
  // table is heap-backed because paper runs may contain millions of atoms.
  std::vector<int> tag_to_graph_idx(
      static_cast<size_t>(natoms) + kAtomTagIndexBase, kInvalidGraphIndex);

  // to access tag_to_graph_idx from comm
  tag_to_graph_idx_ptr = tag_to_graph_idx.data();

  int graph_indexer = nlocal;
  std::vector<int> graph_index_to_i(static_cast<size_t>(ntotal));

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list
  const int nedges_upper_bound =
      std::accumulate(numneigh, numneigh + nlocal, 0);

  std::vector<long> node_type;
  std::vector<long> node_type_ghost;

  std::vector<float> edge_vec_storage(
      static_cast<size_t>(nedges_upper_bound) * kSpatialDimension);
  std::vector<long> edge_idx_src(static_cast<size_t>(nedges_upper_bound));
  std::vector<long> edge_idx_dst(static_cast<size_t>(nedges_upper_bound));

  int nedges = 0;
  for (int ii = 0; ii < inum; ii++) {
    // populate tag_to_graph_idx of local atoms
    const int i = ilist[ii];
    const tagint itag = tag[i];
    const int itype = type[i];
    tag_to_graph_idx[static_cast<size_t>(itag)] = ii;
    graph_index_to_i[ii] = i;
    node_type.push_back(map[itype]);
  }

  // loop over neighbors, build graph
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int i_graph_idx = ii;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      const tagint jtag = tag[j];
      const int jtype = type[j];
      // we have to calculate Rij to check cutoff in lammps side
      const double delij[kSpatialDimension] = {
          x[j][kXCoordinate] - x[i][kXCoordinate],
          x[j][kYCoordinate] - x[i][kYCoordinate],
          x[j][kZCoordinate] - x[i][kZCoordinate]};
      const double Rij = delij[kXCoordinate] * delij[kXCoordinate] +
                         delij[kYCoordinate] * delij[kYCoordinate] +
                         delij[kZCoordinate] * delij[kZCoordinate];

      int j_graph_idx;
      if (Rij < cutoff_square) {
        // if given j is not local atom and inside cutoff
        if (tag_to_graph_idx[static_cast<size_t>(jtag)] == kInvalidGraphIndex) {
          // if j is ghost atom inside cutoff but first seen
          tag_to_graph_idx[static_cast<size_t>(jtag)] = graph_indexer;
          graph_index_to_i[graph_indexer] = j;
          node_type_ghost.push_back(map[jtype]);
          graph_indexer++;
        }

        j_graph_idx = tag_to_graph_idx[static_cast<size_t>(jtag)];
        edge_idx_src[nedges] = i_graph_idx;
        edge_idx_dst[nedges] = j_graph_idx;
        const size_t edge_offset =
            static_cast<size_t>(nedges) * kSpatialDimension;
        edge_vec_storage[edge_offset + kXCoordinate] =
            static_cast<float>(delij[kXCoordinate]);
        edge_vec_storage[edge_offset + kYCoordinate] =
            static_cast<float>(delij[kYCoordinate]);
        edge_vec_storage[edge_offset + kZCoordinate] =
            static_cast<float>(delij[kZCoordinate]);
        nedges++;
      }
    } // j loop end
  }   // i loop end

  // member variable
  graph_size = graph_indexer;
  const int ghost_node_num = graph_size - nlocal;

  // convert data to Tensor
  auto inp_node_type = torch::from_blob(node_type.data(), nlocal, INTEGER_TYPE);
  auto inp_node_type_ghost =
      torch::from_blob(node_type_ghost.data(), ghost_node_num, INTEGER_TYPE);

  long num_nodes[1] = {long(nlocal)};
  auto inp_num_atoms = torch::from_blob(num_nodes, {1}, INTEGER_TYPE);

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src.data(), {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst.data(), {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  auto inp_edge_vec = torch::from_blob(
      edge_vec_storage.data(), {nedges, kSpatialDimension}, FLOAT_TYPE);
  if (print_info) {
    std::cout << world_rank << " Nlocal: " << nlocal << std::endl;
    std::cout << world_rank << " Graph_size: " << graph_size << std::endl;
    std::cout << world_rank << " Ghost_node_num: " << ghost_node_num
              << std::endl;
    std::cout << world_rank << " Nedges: " << nedges << "\n" << std::endl;
  }

  // r_original requires grad True
  inp_edge_vec.set_requires_grad(true);

  torch::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("x_ghost", inp_node_type_ghost.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("edge_vec", inp_edge_vec.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("nlocal", inp_num_atoms.to(torch::kCPU));

  std::list<std::vector<torch::Tensor>> wrt_tensors;
  wrt_tensors.push_back({input_dict.at("edge_vec")});

  auto model_part = model_list.front();

  auto output = model_part.forward({input_dict}).toGenericDict();

  if (!try_reuse_comm_preprocess_cache(nlocal, ghost_node_num, nedges,
                                       graph_index_to_i.data())) {
    comm_preprocess();
    if (iso_delta_halo_enabled) {
      store_comm_preprocess_cache(nlocal, ghost_node_num, nedges,
                                  graph_index_to_i.data());
    }
  }

  // extra_graph_idx_map is set from comm_preprocess();
  // last one is for trash values. See pack_forward_init
  const int extra_size =
      ghost_node_num + static_cast<int>(extra_graph_idx_map.size()) + 1;
  torch::Tensor x_local;
  torch::Tensor x_ghost;

  for (auto it = model_list.begin(); it != model_list.end(); ++it) {
    if (it == model_list.begin())
      continue;
    model_part = *it;

    x_local = output.at("x").toTensor().detach().to(device);
    x_dim = x_local.size(1); // length of per atom vector(node feature)

    auto ghost_and_extra_x = torch::zeros({ghost_node_num + extra_size, x_dim},
                                          FLOAT_TYPE.device(device));
    x_comm = torch::cat({x_local, ghost_and_extra_x}, 0).to(device_comm);
    comm_brick->forward_comm(this); // populate x_ghost by communication

    // What we got from forward_comm (node feature of ghosts)
    x_ghost = torch::split_with_sizes(
        x_comm, {nlocal, ghost_node_num, extra_size}, 0)[1];
    x_ghost.set_requires_grad(true);

    // prepare next input (output > next input)
    output.insert_or_assign("x_ghost", x_ghost.to(device));
    // make another edge_vec to discriminate grad calculation with other
    // edge_vecs(maybe redundant?)
    output.insert_or_assign("edge_vec",
                            output.at("edge_vec").toTensor().clone());

    // save tensors for backprop
    wrt_tensors.push_back({output.at("edge_vec").toTensor(),
                           output.at("x").toTensor(),
                           output.at("self_cont_tmp").toTensor(),
                           output.at("x_ghost").toTensor()});

    output = model_part.forward({output}).toGenericDict();
  }
  torch::Tensor energy_tensor =
      output.at("inferred_total_energy").toTensor().squeeze();

  torch::Tensor dE_dr =
      torch::zeros({nedges, kSpatialDimension},
                   FLOAT_TYPE.device(device)); // create on device
  torch::Tensor x_local_save; // holds grad info of x_local (it loses its grad
                              // when sends to CPU)
  torch::Tensor self_conn_grads;
  std::vector<torch::Tensor> grads;
  std::vector<torch::Tensor> of_tensor;

  // self_conn_grads is usually sparse because the energy head uses scalar
  // channels, but it is still passed through autograd to preserve semantics.
  for (auto rit = wrt_tensors.rbegin(); rit != wrt_tensors.rend(); ++rit) {
    // edge_vec, x, x_ghost order
    auto wrt_tensor = *rit;
    if (rit == wrt_tensors.rbegin()) {
      grads = torch::autograd::grad({energy_tensor}, wrt_tensor);
    } else {
      x_local_save.copy_(x_local);
      //                            of         wrt         grads_output
      grads = torch::autograd::grad(of_tensor, wrt_tensor,
                                    {x_local_save, self_conn_grads});
    }

    dE_dr = dE_dr + grads.at(0); // accumulate force
    if (std::distance(rit, wrt_tensors.rend()) == 1)
      continue; // if last iteration

    of_tensor.clear();
    of_tensor.push_back(wrt_tensor[1]); // x
    of_tensor.push_back(wrt_tensor[2]); // self_cont_tmp

    x_local_save = grads.at(1);      // for grads_output
    x_local = x_local_save.detach(); // grad_outputs & communication
    x_dim = x_local.size(1);

    self_conn_grads = grads.at(2); // no communication, for grads_output

    x_ghost = grads.at(3).detach(); // yes communication, not for grads_output

    auto extra_x = torch::zeros({extra_size, x_dim}, FLOAT_TYPE.device(device));
    x_comm = torch::cat({x_local, x_ghost, extra_x}, 0).to(device_comm);

    comm_brick->reverse_comm(this); // completes x_local

    // now x_local is complete (dE_dx), become next grads_output(with
    // self_conn_grads)
    x_local = torch::split_with_sizes(
        x_comm, {nlocal, ghost_node_num, extra_size}, 0)[0];
  }

  // postprocessing
  if (print_info) {
    size_t free, tot;
    cudaMemGetInfo(&free, &tot);
    std::cout << world_rank << " MEM use after backward(MiB)" << std::endl;
    double Mfree = static_cast<double>(free) / kBytesPerMebibyte;
    double Mtot = static_cast<double>(tot) / kBytesPerMebibyte;
    std::cout << world_rank << " Total: " << Mtot << std::endl;
    std::cout << world_rank << " Free: " << Mfree << std::endl;
    std::cout << world_rank << " Used: " << Mtot - Mfree << std::endl;
    double Mused = Mtot - Mfree;
    std::cout << world_rank << " Used/Nedges: " << Mused / nedges << std::endl;
    std::cout << world_rank << " Used/Nlocal: " << Mused / nlocal << std::endl;
    std::cout << world_rank << " Used/GraphSize: " << Mused / graph_size << "\n"
              << std::endl;
  }
  eng_vdwl += energy_tensor.item<float>(); // accumulate energy

  dE_dr = dE_dr.to(torch::kCPU);
  torch::Tensor force_tensor =
      torch::zeros({graph_indexer, kSpatialDimension});

  auto _edge_idx_src_tensor =
      edge_idx_src_tensor.repeat_interleave(kSpatialDimension)
          .view({nedges, kSpatialDimension});
  auto _edge_idx_dst_tensor =
      edge_idx_dst_tensor.repeat_interleave(kSpatialDimension)
          .view({nedges, kSpatialDimension});

  force_tensor.scatter_reduce_(0, _edge_idx_src_tensor, dE_dr, "sum");
  force_tensor.scatter_reduce_(0, _edge_idx_dst_tensor, torch::neg(dE_dr),
                               "sum");

  auto forces = force_tensor.accessor<float, 2>();

  for (int graph_idx = 0; graph_idx < graph_indexer; graph_idx++) {
    int i = graph_index_to_i[graph_idx];
    f[i][kXCoordinate] += forces[graph_idx][kXCoordinate];
    f[i][kYCoordinate] += forces[graph_idx][kYCoordinate];
    f[i][kZCoordinate] += forces[graph_idx][kZCoordinate];
  }

  if (vflag) {
    auto diag = inp_edge_vec * dE_dr;
    auto s12 = inp_edge_vec.select(1, kXCoordinate) *
               dE_dr.select(1, kYCoordinate);
    auto s23 = inp_edge_vec.select(1, kYCoordinate) *
               dE_dr.select(1, kZCoordinate);
    auto s31 = inp_edge_vec.select(1, kZCoordinate) *
               dE_dr.select(1, kXCoordinate);
    std::vector<torch::Tensor> voigt_list = {
        diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)};
    auto voigt = torch::cat(voigt_list, 1);

    torch::Tensor per_atom_stress_tensor =
        torch::zeros({graph_indexer, kVoigtStressComponentCount});
    auto _edge_idx_dst6_tensor =
        edge_idx_dst_tensor.repeat_interleave(kVoigtStressComponentCount)
            .view({nedges, kVoigtStressComponentCount});
    per_atom_stress_tensor.scatter_reduce_(0, _edge_idx_dst6_tensor, voigt,
                                           "sum");
    auto virial_stress_tensor =
        torch::neg(torch::sum(per_atom_stress_tensor, 0));
    auto virial_stress = virial_stress_tensor.accessor<float, 1>();

    virial[kVoigtXX] += virial_stress[kVoigtXX];
    virial[kVoigtYY] += virial_stress[kVoigtYY];
    virial[kVoigtZZ] += virial_stress[kVoigtZZ];
    virial[kVoigtXY] += virial_stress[kVoigtXY];
    virial[kVoigtYZ] += virial_stress[kVoigtZX];
    virial[kVoigtZX] += virial_stress[kVoigtYZ];
  }

  if (eflag_atom) {
    torch::Tensor atomic_energy_tensor =
        output.at("atomic_energy").toTensor().cpu().view({nlocal});
    auto atomic_energy = atomic_energy_tensor.accessor<float, 1>();
    for (int graph_idx = 0; graph_idx < nlocal; graph_idx++) {
      int i = graph_index_to_i[graph_idx];
      eatom[i] += atomic_energy[graph_idx];
    }
  }

  clear_comm_preprocess_work();
}

// allocate arrays (called from coeff)
void PairE3GNNParallel::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNNParallel::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNNParallel::coeff(int narg, char **arg) {
  if (allocated) {
    error->all(FLERR, "pair_e3gnn coeff called twice");
  }
  allocate();

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR,
               "e3gnn: first and second input of pair_coeff should be '*'");
  }
  // expected input : pair_coeff * * pot.pth type_name1 type_name2 ...

  std::unordered_map<std::string, std::string> meta_dict = {
      {"chemical_symbols_to_index", ""},
      {"cutoff", ""},
      {"num_species", ""},
      {"model_type", ""},
      {"version", ""},
      {"dtype", ""},
      {"time", ""},
      {"flashTP", "version mismatch"},
      {"oeq", "version mismatch"},
      {"comm_size", ""}};

  // model loading from input
  int n_model = std::stoi(arg[2]);
  int chem_arg_i = 4;
  std::vector<std::string> model_fnames;
  if (std::filesystem::exists(arg[3])) {
    if (std::filesystem::is_directory(arg[3])) {
      auto headf = std::string(arg[3]);
      for (int i = 0; i < n_model; i++) {
        auto stri = std::to_string(i);
        model_fnames.push_back(headf + "/deployed_parallel_" + stri + ".pt");
      }
    } else if (std::filesystem::is_regular_file(arg[3])) {
      for (int i = 3; i < n_model + 3; i++) {
        model_fnames.push_back(std::string(arg[i]));
      }
      chem_arg_i = n_model + 3;
    } else {
      error->all(FLERR, "No such file or directory:" + std::string(arg[3]));
    }
  }

  for (const auto &modelf : model_fnames) {
    if (!std::filesystem::is_regular_file(modelf)) {
      error->all(FLERR, "Expected this is a regular file:" + modelf);
    }
    model_list.push_back(torch::jit::load(modelf, device, meta_dict));
  }

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);

  // maximum possible size of per atom x before last convolution
  int comm_size = std::stod(meta_dict["comm_size"]);

  // to initialize buffer size for communication
  comm_forward = comm_size;
  comm_reverse = comm_size;

  cutoff_square = cutoff * cutoff;

  // to make torch::autograd::grad() works
  if (meta_dict["oeq"] == "yes") {
    pair_e3gnn_oeq_register_autograd();
  }

  if (meta_dict["model_type"].compare("E3_equivariant_model") != 0) {
    error->all(FLERR, "given model type is not E3_equivariant_model");
  }

  std::string chem_str = meta_dict["chemical_symbols_to_index"];
  int ntypes = atom->ntypes;

  auto delim = " ";
  char *tok = std::strtok(const_cast<char *>(chem_str.c_str()), delim);
  std::vector<std::string> chem_vec;
  while (tok != nullptr) {
    chem_vec.push_back(std::string(tok));
    tok = std::strtok(nullptr, delim);
  }

  // what if unknown chemical specie is in arg? should I abort? is there any use
  // case for that?
  bool found_flag = false;
  int n_chem = narg - chem_arg_i;
  for (int i = 0; i < n_chem; i++) {
    found_flag = false;
    for (int j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i + chem_arg_i]) == 0) {
        map[i + 1] = j; // store from 1, (not 0)
        found_flag = true;
        if (lmp->logfile) {
          fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                  arg[i + chem_arg_i], i + 1);
          break;
        }
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given or the number of "
                        "potential files is not consistent");
    }
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      if ((map[i] >= 0) && (map[j] >= 0)) {
        setflag[i][j] = 1;
        cutsq[i][j] = cutoff * cutoff;
      }
    }
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "from sevenn version '%s' ",
            meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model, deployed: %s\n",
            meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
    fprintf(lmp->logfile, "FlashTP: %s\n",
            meta_dict["flashTP"].c_str());
    fprintf(lmp->logfile, "OEQ: %s\n",
            meta_dict["oeq"].c_str());
  }
}

// init specific to this pair
void PairE3GNNParallel::init_style() {
  // full neighbor list & newton on
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style e3gnn/parallel requires newton pair on");
  }
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNNParallel::init_one(int i, int j) { return cutoff; }

void PairE3GNNParallel::notify_proc_ids(
    const int *sendproc, const int *recvproc, int active_phase_count) {
  const int bounded_active_phase_count =
      std::min(std::max(active_phase_count, kNoActiveCommPhases),
               kCommPhaseCount);
  for (int iswap = 0; iswap < kCommPhaseCount; iswap++) {
    const bool active_phase = iswap < bounded_active_phase_count;
    this->sendproc[iswap] =
        active_phase ? sendproc[iswap] : kInactiveCommPhaseValue;
    this->recvproc[iswap] =
        active_phase ? recvproc[iswap] : kInactiveCommPhaseValue;
  }
}

void PairE3GNNParallel::record_comm_cache_miss(CommCacheMissReason reason) {
  if (!comm_cache_miss_reason_is_valid(reason)) {
    return;
  }
  const size_t reason_index = static_cast<size_t>(reason);
  comm_cache_misses[reason_index]++;
}

bool PairE3GNNParallel::comm_cache_miss_reason_is_valid(
    CommCacheMissReason reason) {
  const int reason_index = static_cast<int>(reason);
  return reason_index >= 0 && reason_index < kCommCacheMissReasonCount;
}

const char *PairE3GNNParallel::comm_cache_miss_reason_name(
    CommCacheMissReason reason) {
  switch (reason) {
  case CommCacheMissReason::kDisabled:
    return "disabled";
  case CommCacheMissReason::kNoCache:
    return "no-cache";
  case CommCacheMissReason::kNeighborListRebuilt:
    return "neighbor-list-rebuilt";
  case CommCacheMissReason::kShapeChanged:
    return "shape-changed";
  case CommCacheMissReason::kIndexTensorShapeChanged:
    return "index-tensor-shape-changed";
  case CommCacheMissReason::kTagCountChanged:
    return "tag-count-changed";
  case CommCacheMissReason::kTagOrderChanged:
    return "tag-order-changed";
  case CommCacheMissReason::kCommTopologyChanged:
    return "comm-topology-changed";
  case CommCacheMissReason::kCommListTagOrderChanged:
    return "comm-list-tag-order-changed";
  }
  return "unknown";
}

void PairE3GNNParallel::print_comm_cache_summary() const {
  if (!iso_delta_halo_profile || !print_info) {
    return;
  }

  const double hit_percent =
      comm_cache_attempts == 0
          ? 0.0
          : kIsoDeltaHaloPercentScale * static_cast<double>(comm_cache_hits) /
                static_cast<double>(comm_cache_attempts);
  std::cout << world_rank << " IsoDelta-Halo summary: attempts="
            << comm_cache_attempts << " hits=" << comm_cache_hits
            << " hit_rate_percent=" << hit_percent;

  for (int reason_index = 0; reason_index < kCommCacheMissReasonCount;
       reason_index++) {
    const auto reason = static_cast<CommCacheMissReason>(reason_index);
    std::cout << " miss_" << comm_cache_miss_reason_name(reason) << "="
              << comm_cache_misses[reason_index];
  }
  std::cout << std::endl;
}

void PairE3GNNParallel::validate_comm_phase(int comm_phase) const {
  if (comm_phase < kNoActiveCommPhases || comm_phase >= kCommPhaseCount) {
    error->all(FLERR, kIsoDeltaHaloCommPhaseRangeError);
  }
}

bool PairE3GNNParallel::try_reuse_comm_preprocess_cache(
    int nlocal, int ghost_node_num, int nedges, const int *graph_index_to_i) {
  comm_cache_attempts++;
  if (!iso_delta_halo_enabled) {
    record_comm_cache_miss(CommCacheMissReason::kDisabled);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (!comm_cache_valid) {
    record_comm_cache_miss(CommCacheMissReason::kNoCache);
    return false;
  }
  if (neighbor->ago <= kNeighborListJustBuiltAgo) {
    record_comm_cache_miss(CommCacheMissReason::kNeighborListRebuilt);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (nlocal != comm_cache_nlocal ||
      ghost_node_num != comm_cache_ghost_node_num ||
      graph_size != comm_cache_graph_size || nedges != comm_cache_nedges) {
    record_comm_cache_miss(CommCacheMissReason::kShapeChanged);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (comm_cache_graph_tags.size() != static_cast<size_t>(graph_size)) {
    record_comm_cache_miss(CommCacheMissReason::kTagCountChanged);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (!comm_topology_matches_cache()) {
    record_comm_cache_miss(CommCacheMissReason::kCommTopologyChanged);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (!comm_list_tags_match_cache()) {
    record_comm_cache_miss(CommCacheMissReason::kCommListTagOrderChanged);
    invalidate_comm_preprocess_cache();
    return false;
  }
  if (!cached_comm_tensors_match_vectors()) {
    record_comm_cache_miss(CommCacheMissReason::kIndexTensorShapeChanged);
    invalidate_comm_preprocess_cache();
    return false;
  }

  tagint *tag = atom->tag;
  for (int graph_idx = 0; graph_idx < graph_size; graph_idx++) {
    const int atom_idx = graph_index_to_i[graph_idx];
    if (tag[atom_idx] != comm_cache_graph_tags[graph_idx]) {
      record_comm_cache_miss(CommCacheMissReason::kTagOrderChanged);
      invalidate_comm_preprocess_cache();
      return false;
    }
  }

  extra_graph_idx_map = comm_cache_extra_graph_idx_map;
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    comm_index_pack_forward[comm_phase] =
        comm_cache_index_pack_forward[comm_phase];
    comm_index_unpack_forward[comm_phase] =
        comm_cache_index_unpack_forward[comm_phase];
    comm_index_unpack_reverse[comm_phase] =
        comm_cache_index_unpack_reverse[comm_phase];
    comm_index_pack_forward_tensor[comm_phase] =
        comm_cache_index_pack_forward_tensor[comm_phase];
    comm_index_unpack_forward_tensor[comm_phase] =
        comm_cache_index_unpack_forward_tensor[comm_phase];
    comm_index_unpack_reverse_tensor[comm_phase] =
        comm_cache_index_unpack_reverse_tensor[comm_phase];
  }

  comm_preprocess_done = true;
  comm_cache_hits++;
  if (print_info) {
    std::cout << world_rank
              << " IsoDelta-Halo: reused communication metadata cache"
              << std::endl;
  }
  return true;
}

bool PairE3GNNParallel::cached_comm_tensors_match_vectors() const {
  if (!use_cuda_mpi) {
    return true;
  }

  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    if (!index_tensor_matches_vector(
            comm_cache_index_pack_forward_tensor[comm_phase],
            comm_cache_index_pack_forward[comm_phase]) ||
        !index_tensor_matches_vector(
            comm_cache_index_unpack_forward_tensor[comm_phase],
            comm_cache_index_unpack_forward[comm_phase]) ||
        !index_tensor_matches_vector(
            comm_cache_index_unpack_reverse_tensor[comm_phase],
            comm_cache_index_unpack_reverse[comm_phase])) {
      return false;
    }
  }
  return true;
}

void PairE3GNNParallel::store_comm_preprocess_cache(
    int nlocal, int ghost_node_num, int nedges, const int *graph_index_to_i) {
  if (!current_comm_topology_is_cacheable()) {
    invalidate_comm_preprocess_cache();
    return;
  }

  comm_cache_nlocal = nlocal;
  comm_cache_ghost_node_num = ghost_node_num;
  comm_cache_graph_size = graph_size;
  comm_cache_nedges = nedges;

  tagint *tag = atom->tag;
  comm_cache_graph_tags.clear();
  comm_cache_graph_tags.reserve(graph_size);
  for (int graph_idx = 0; graph_idx < graph_size; graph_idx++) {
    const int atom_idx = graph_index_to_i[graph_idx];
    comm_cache_graph_tags.push_back(tag[atom_idx]);
  }

  store_comm_topology_signature();
  store_comm_list_tag_signature();
  comm_cache_extra_graph_idx_map = extra_graph_idx_map;
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    comm_cache_index_pack_forward[comm_phase] =
        comm_index_pack_forward[comm_phase];
    comm_cache_index_unpack_forward[comm_phase] =
        comm_index_unpack_forward[comm_phase];
    comm_cache_index_unpack_reverse[comm_phase] =
        comm_index_unpack_reverse[comm_phase];
    comm_cache_index_pack_forward_tensor[comm_phase] =
        comm_index_pack_forward_tensor[comm_phase];
    comm_cache_index_unpack_forward_tensor[comm_phase] =
        comm_index_unpack_forward_tensor[comm_phase];
    comm_cache_index_unpack_reverse_tensor[comm_phase] =
        comm_index_unpack_reverse_tensor[comm_phase];
  }

  comm_cache_valid = true;
}

void PairE3GNNParallel::clear_comm_preprocess_work() {
  comm_preprocess_done = false;
  tag_to_graph_idx_ptr = nullptr;
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    comm_index_pack_forward[comm_phase].clear();
    comm_index_unpack_forward[comm_phase].clear();
    comm_index_unpack_reverse[comm_phase].clear();
  }

  extra_graph_idx_map.clear();
}

void PairE3GNNParallel::invalidate_comm_preprocess_cache() {
  comm_cache_valid = false;
  comm_cache_nswap = kInactiveCommPhaseValue;
  comm_cache_sendnum.fill(kInactiveCommPhaseValue);
  comm_cache_recvnum.fill(kInactiveCommPhaseValue);
  comm_cache_sendproc.fill(kInactiveCommPhaseValue);
  comm_cache_recvproc.fill(kInactiveCommPhaseValue);
  comm_cache_firstrecv.fill(kInactiveCommPhaseValue);
  comm_cache_graph_tags.clear();
  comm_cache_extra_graph_idx_map.clear();
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    comm_cache_sendlist_tags[comm_phase].clear();
    comm_cache_recvlist_tags[comm_phase].clear();
    comm_cache_index_pack_forward[comm_phase].clear();
    comm_cache_index_unpack_forward[comm_phase].clear();
    comm_cache_index_unpack_reverse[comm_phase].clear();
    comm_cache_index_pack_forward_tensor[comm_phase] = torch::Tensor();
    comm_cache_index_unpack_forward_tensor[comm_phase] = torch::Tensor();
    comm_cache_index_unpack_reverse_tensor[comm_phase] = torch::Tensor();
  }
}

bool PairE3GNNParallel::current_comm_topology_is_cacheable() const {
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    return false;
  }

  const int current_nswap = comm_brick->e3gnn_nswap();
  return current_nswap >= kNoActiveCommPhases &&
         current_nswap <= kCommPhaseCount;
}

bool PairE3GNNParallel::comm_topology_matches_cache() const {
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    return false;
  }

  const int current_nswap = comm_brick->e3gnn_nswap();
  if (current_nswap != comm_cache_nswap ||
      current_nswap > kCommPhaseCount) {
    return false;
  }
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    const bool active_phase = comm_phase < current_nswap;
    const int current_sendnum = active_phase
                                    ? comm_brick->e3gnn_sendnum(comm_phase)
                                    : kInactiveCommPhaseValue;
    const int current_recvnum = active_phase
                                    ? comm_brick->e3gnn_recvnum(comm_phase)
                                    : kInactiveCommPhaseValue;
    const int current_sendproc = active_phase
                                     ? comm_brick->e3gnn_sendproc(comm_phase)
                                     : kInactiveCommPhaseValue;
    const int current_recvproc = active_phase
                                     ? comm_brick->e3gnn_recvproc(comm_phase)
                                     : kInactiveCommPhaseValue;
    const int current_firstrecv = active_phase
                                      ? comm_brick->e3gnn_firstrecv(comm_phase)
                                      : kInactiveCommPhaseValue;

    if (current_sendnum != comm_cache_sendnum[comm_phase] ||
        current_recvnum != comm_cache_recvnum[comm_phase] ||
        current_sendproc != comm_cache_sendproc[comm_phase] ||
        current_recvproc != comm_cache_recvproc[comm_phase] ||
        current_firstrecv != comm_cache_firstrecv[comm_phase]) {
      return false;
    }
  }
  return true;
}

bool PairE3GNNParallel::comm_list_tags_match_cache() const {
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    return false;
  }

  const int current_nswap = comm_brick->e3gnn_nswap();
  if (current_nswap > kCommPhaseCount) {
    return false;
  }

  tagint *tag = atom->tag;
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    const bool active_phase = comm_phase < current_nswap;
    const int current_sendnum =
        active_phase ? comm_brick->e3gnn_sendnum(comm_phase) : 0;
    const int current_recvnum =
        active_phase ? comm_brick->e3gnn_recvnum(comm_phase) : 0;
    if (comm_cache_sendlist_tags[comm_phase].size() !=
            static_cast<size_t>(current_sendnum) ||
        comm_cache_recvlist_tags[comm_phase].size() !=
            static_cast<size_t>(current_recvnum)) {
      return false;
    }

    for (int index = 0; index < current_sendnum; index++) {
      const int atom_idx = comm_brick->e3gnn_sendlist_atom(comm_phase, index);
      if (tag[atom_idx] != comm_cache_sendlist_tags[comm_phase][index]) {
        return false;
      }
    }

    const int firstrecv =
        active_phase ? comm_brick->e3gnn_firstrecv(comm_phase) : 0;
    for (int index = 0; index < current_recvnum; index++) {
      if (tag[firstrecv + index] !=
          comm_cache_recvlist_tags[comm_phase][index]) {
        return false;
      }
    }
  }
  return true;
}

void PairE3GNNParallel::store_comm_topology_signature() {
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  comm_cache_nswap =
      comm_brick == nullptr ? kInactiveCommPhaseValue : comm_brick->e3gnn_nswap();

  comm_cache_sendnum.fill(kInactiveCommPhaseValue);
  comm_cache_recvnum.fill(kInactiveCommPhaseValue);
  comm_cache_sendproc.fill(kInactiveCommPhaseValue);
  comm_cache_recvproc.fill(kInactiveCommPhaseValue);
  comm_cache_firstrecv.fill(kInactiveCommPhaseValue);

  if (comm_brick == nullptr || comm_cache_nswap > kCommPhaseCount) {
    return;
  }
  for (int comm_phase = 0; comm_phase < comm_cache_nswap; comm_phase++) {
    comm_cache_sendnum[comm_phase] = comm_brick->e3gnn_sendnum(comm_phase);
    comm_cache_recvnum[comm_phase] = comm_brick->e3gnn_recvnum(comm_phase);
    comm_cache_sendproc[comm_phase] = comm_brick->e3gnn_sendproc(comm_phase);
    comm_cache_recvproc[comm_phase] = comm_brick->e3gnn_recvproc(comm_phase);
    comm_cache_firstrecv[comm_phase] = comm_brick->e3gnn_firstrecv(comm_phase);
  }
}

void PairE3GNNParallel::store_comm_list_tag_signature() {
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    comm_cache_sendlist_tags[comm_phase].clear();
    comm_cache_recvlist_tags[comm_phase].clear();
  }

  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    return;
  }

  const int current_nswap = comm_brick->e3gnn_nswap();
  if (current_nswap > kCommPhaseCount) {
    return;
  }

  tagint *tag = atom->tag;
  for (int comm_phase = 0; comm_phase < current_nswap; comm_phase++) {
    const int current_sendnum = comm_brick->e3gnn_sendnum(comm_phase);
    comm_cache_sendlist_tags[comm_phase].reserve(current_sendnum);
    for (int index = 0; index < current_sendnum; index++) {
      const int atom_idx = comm_brick->e3gnn_sendlist_atom(comm_phase, index);
      comm_cache_sendlist_tags[comm_phase].push_back(tag[atom_idx]);
    }

    const int current_recvnum = comm_brick->e3gnn_recvnum(comm_phase);
    const int firstrecv = comm_brick->e3gnn_firstrecv(comm_phase);
    comm_cache_recvlist_tags[comm_phase].reserve(current_recvnum);
    for (int index = 0; index < current_recvnum; index++) {
      comm_cache_recvlist_tags[comm_phase].push_back(tag[firstrecv + index]);
    }
  }
}

void PairE3GNNParallel::comm_preprocess() {
  assert(!comm_preprocess_done);
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    error->all(FLERR, kIsoDeltaHaloCommBrickRequiredError);
  }

  // fake lammps communication call to preprocess index
  // gives complete comm_index_pack, unpack_forward, and extra_graph_idx_map
  comm_brick->forward_comm(this);

  std::map<int, std::set<int>> already_met_map;
  for (int comm_phase = 0; comm_phase < kCommPhaseCount; comm_phase++) {
    const int n = comm_index_pack_forward[comm_phase].size();
    int sproc = this->sendproc[comm_phase];
    if (already_met_map.count(sproc) == 0) {
      already_met_map.insert({sproc, std::set<int>()});
    }

    // for unpack_reverse, Ignore duplicated index by 'already_met'
    std::vector<long> &idx_map_forward = comm_index_pack_forward[comm_phase];
    std::vector<long> &idx_map_reverse = comm_index_unpack_reverse[comm_phase];
    std::set<int>& already_met = already_met_map[sproc];
    // the last index of x_comm is used to trash unnecessary values
    const int trash_index =
        graph_size + static_cast<int>(extra_graph_idx_map.size()); //+ 1;
    for (int i = 0; i < n; i++) {
      const int idx = idx_map_forward[i];
      if (idx < graph_size) {
        if (already_met.count(idx) == 1) {
          idx_map_reverse.push_back(trash_index);
        } else {
          idx_map_reverse.push_back(idx);
          already_met.insert(idx);
        }
      } else {
        idx_map_reverse.push_back(idx);
      }
    }

    if (use_cuda_mpi) {
      comm_index_pack_forward_tensor[comm_phase] =
          make_owned_index_tensor(idx_map_forward, device);

      std::vector<long> &upmap = comm_index_unpack_forward[comm_phase];
      comm_index_unpack_forward_tensor[comm_phase] =
          make_owned_index_tensor(upmap, device);
      comm_index_unpack_reverse_tensor[comm_phase] =
          make_owned_index_tensor(idx_map_reverse, device);
    }
  }
  comm_preprocess_done = true;
}

// called from comm_brick if comm_preprocess_done is false
void PairE3GNNParallel::pack_forward_init(int n, int *list_send,
                                          int comm_phase) {
  validate_comm_phase(comm_phase);
  if (tag_to_graph_idx_ptr == nullptr) {
    error->all(FLERR, kIsoDeltaHaloGraphIndexRequiredError);
  }
  std::vector<long> &idx_map = comm_index_pack_forward[comm_phase];

  idx_map.reserve(n);

  int i, j;
  int nlocal = list->inum;
  tagint *tag = atom->tag;

  for (i = 0; i < n; i++) {
    int list_i = list_send[i];
    int graph_idx = tag_to_graph_idx_ptr[tag[list_i]];

    if (graph_idx != kInvalidGraphIndex) {
      // known atom (local atom + ghost atom inside cutoff)
      idx_map.push_back(graph_idx);
    } else {
      // unknown atom, these are not used in computation in this process
      // instead, this process is used to hand over these atoms to other proecss
      // hold them in continuous manner for flexible tensor operations later
      if (extra_graph_idx_map.find(list_i) != extra_graph_idx_map.end()) {
        idx_map.push_back(extra_graph_idx_map[list_i]);
      } else {
        // unknown atom at pack forward, ghost atom outside cutoff?
        extra_graph_idx_map[list_i] = graph_size + extra_graph_idx_map.size();
        idx_map.push_back(extra_graph_idx_map[list_i]);
      }
    }
  }
}

// called from comm_brick if comm_preprocess_done is false
void PairE3GNNParallel::unpack_forward_init(int n, int first, int comm_phase) {
  validate_comm_phase(comm_phase);
  if (tag_to_graph_idx_ptr == nullptr) {
    error->all(FLERR, kIsoDeltaHaloGraphIndexRequiredError);
  }
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];

  idx_map.reserve(n);

  int i, j, last;
  last = first + n;
  int nlocal = list->inum;
  tagint *tag = atom->tag;

  for (i = first; i < last; i++) {
    int graph_idx = tag_to_graph_idx_ptr[tag[i]];
    if (graph_idx != kInvalidGraphIndex) {
      idx_map.push_back(graph_idx);
    } else {
      extra_graph_idx_map[i] = graph_size + extra_graph_idx_map.size();
      idx_map.push_back(extra_graph_idx_map[i]); // same as list_i in pack
    }
  }
}

int PairE3GNNParallel::pack_forward_comm_gnn(float *buf, int comm_phase) {
  validate_comm_phase(comm_phase);
  std::vector<long> &idx_map = comm_index_pack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());
  const int payload_element_count =
      checked_e3gnn_payload_element_count(x_dim, n, error);
  const size_t payload_byte_count =
      checked_e3gnn_payload_byte_count(payload_element_count);
  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_pack_forward_tensor[comm_phase];
    auto selected = x_comm.index_select(0, idx_map_tensor); // its size is x_dim * n
    cudaError_t cuda_err =
        cudaMemcpy(buf, selected.data_ptr<float>(), payload_byte_count,
                   cudaMemcpyDeviceToDevice);
    check_cuda_status(cuda_err, kCudaPackForwardMemcpyError, error);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *from = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        buf[m++] = from[j];
      }
    }
  }
  if (print_info) {
    std::cout << world_rank << " comm_phase: " << comm_phase << std::endl;
    std::cout << world_rank << " pack_forward x_dim: " << x_dim << std::endl;
    std::cout << world_rank << " pack_forward n: " << n << std::endl;
    std::cout << world_rank << " pack_forward x_dim*n: " << payload_element_count
              << std::endl;
    double Msend = static_cast<double>(x_dim) * static_cast<double>(n) *
                   kFloatElementBytes / kBytesPerMebibyte;
    std::cout << world_rank << " send size(MiB): " << Msend << "\n" << std::endl;
  }
  return payload_element_count;
}

void PairE3GNNParallel::unpack_forward_comm_gnn(float *buf, int comm_phase) {
  validate_comm_phase(comm_phase);
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());
  checked_e3gnn_payload_element_count(x_dim, n, error);

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_forward_tensor[comm_phase];
    auto buf_tensor =
        torch::from_blob(buf, {n, x_dim}, FLOAT_TYPE.device(device));
    x_comm.scatter_(0, idx_map_tensor.repeat_interleave(x_dim).view({n, x_dim}),
                    buf_tensor);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *to = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        to[j] = buf[m++];
      }
    }
  }
}

int PairE3GNNParallel::pack_reverse_comm_gnn(float *buf, int comm_phase) {
  validate_comm_phase(comm_phase);
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());
  const int payload_element_count =
      checked_e3gnn_payload_element_count(x_dim, n, error);
  const size_t payload_byte_count =
      checked_e3gnn_payload_byte_count(payload_element_count);

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_forward_tensor[comm_phase];
    auto selected = x_comm.index_select(0, idx_map_tensor);
    cudaError_t cuda_err =
        cudaMemcpy(buf, selected.data_ptr<float>(), payload_byte_count,
                   cudaMemcpyDeviceToDevice);
    check_cuda_status(cuda_err, kCudaPackReverseMemcpyError, error);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *from = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        buf[m++] = from[j];
      }
    }
  }
  if (print_info) {
    std::cout << world_rank << " comm_phase: " << comm_phase << std::endl;
    std::cout << world_rank << " pack_reverse x_dim: " << x_dim << std::endl;
    std::cout << world_rank << " pack_reverse n: " << n << std::endl;
    std::cout << world_rank << " pack_reverse x_dim*n: " << payload_element_count
              << std::endl;
    double Msend = static_cast<double>(x_dim) * static_cast<double>(n) *
                   kFloatElementBytes / kBytesPerMebibyte;
  }
  return payload_element_count;
}

void PairE3GNNParallel::unpack_reverse_comm_gnn(float *buf, int comm_phase) {
  validate_comm_phase(comm_phase);
  std::vector<long> &idx_map = comm_index_unpack_reverse[comm_phase];
  const int n = static_cast<int>(idx_map.size());
  checked_e3gnn_payload_element_count(x_dim, n, error);

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_reverse_tensor[comm_phase];
    auto buf_tensor =
        torch::from_blob(buf, {n, x_dim}, FLOAT_TYPE.device(device));
    x_comm.scatter_(0, idx_map_tensor.repeat_interleave(x_dim).view({n, x_dim}),
                    buf_tensor, "add");
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      if (idx == kInvalidGraphIndex) {
        m += x_dim;
        continue;
      }
      float *to = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        to[j] += buf[m++];
      }
    }
  }
}
