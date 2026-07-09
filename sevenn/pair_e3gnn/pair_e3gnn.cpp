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

#include <ATen/ops/from_blob.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include "pair_e3gnn.h"

using namespace LAMMPS_NS;

// Undefined reference; body in pair_e3gnn_oeq_autograd.cpp to be linked
extern void pair_e3gnn_oeq_register_autograd();

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

namespace {
constexpr const char *kSerialPairCoeffArgumentError =
    "PairE3GNN: pair_coeff arguments are invalid";
constexpr const char *kSerialPairCoeffNumericMetadataError =
    "PairE3GNN: deployed model numeric metadata is invalid";
constexpr const char *kSerialPairCoeffSpeciesMetadataError =
    "PairE3GNN: deployed model species metadata is invalid";
constexpr int kSerialPairCoeffWildcardFirstIndex = 0;
constexpr int kSerialPairCoeffWildcardSecondIndex = 1;
constexpr int kSerialPairCoeffModelPathIndex = 2;
constexpr int kSerialPairCoeffSpeciesStartIndex = 3;
constexpr int kMinimumSerialPairCoeffArgumentCount = 4;
constexpr int kMinimumSerialSpeciesCount = 1;
constexpr int kSerialFirstLammpsAtomType = 1;

void validate_serial_pair_coeff_minimum_args(int arg_count, Error *error) {
  if (arg_count < kMinimumSerialPairCoeffArgumentCount) {
    error->all(FLERR, kSerialPairCoeffArgumentError);
  }
}

double checked_serial_positive_double_metadata(const std::string &value,
                                               Error *error) {
  size_t parsed_length = 0;
  double parsed_value = 0.0;
  try {
    parsed_value = std::stod(value, &parsed_length);
  } catch (const std::exception &) {
    error->all(FLERR, kSerialPairCoeffNumericMetadataError);
  }
  if (parsed_length != value.size() || !std::isfinite(parsed_value) ||
      parsed_value <= 0.0) {
    error->all(FLERR, kSerialPairCoeffNumericMetadataError);
  }
  return parsed_value;
}

int checked_serial_positive_int_metadata(const std::string &value,
                                         Error *error) {
  const double parsed_value =
      checked_serial_positive_double_metadata(value, error);
  if (parsed_value > static_cast<double>(std::numeric_limits<int>::max())) {
    error->all(FLERR, kSerialPairCoeffNumericMetadataError);
  }
  const int parsed_int = static_cast<int>(parsed_value);
  if (static_cast<double>(parsed_int) != parsed_value) {
    error->all(FLERR, kSerialPairCoeffNumericMetadataError);
  }
  return parsed_int;
}

std::vector<std::string> parse_serial_chemical_symbol_tokens(
    const std::string &chemical_symbols, Error *error) {
  std::istringstream symbol_stream(chemical_symbols);
  std::vector<std::string> symbols;
  std::string symbol;
  while (symbol_stream >> symbol) {
    symbols.push_back(symbol);
  }
  if (symbols.empty()) {
    error->all(FLERR, kSerialPairCoeffSpeciesMetadataError);
  }
  return symbols;
}

void validate_serial_deployed_species_metadata(int deployed_species_count,
                                               size_t parsed_symbol_count,
                                               Error *error) {
  if (static_cast<size_t>(deployed_species_count) != parsed_symbol_count) {
    error->all(FLERR, kSerialPairCoeffSpeciesMetadataError);
  }
}

void validate_serial_pair_coeff_species_count(int pair_coeff_species_count,
                                              int lammps_atom_type_count,
                                              Error *error) {
  if (pair_coeff_species_count < kMinimumSerialSpeciesCount ||
      pair_coeff_species_count != lammps_atom_type_count) {
    error->all(FLERR, kSerialPairCoeffArgumentError);
  }
}
} // namespace

PairE3GNN::PairE3GNN(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  const char *print_flag = std::getenv("SEVENN_PRINT_INFO");
  if (print_flag)
    print_info = true;

  std::string device_name;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNN using device : %s\n", device_name.c_str());
  }
}

PairE3GNN::~PairE3GNN() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    // memory->destroy(elements);
  }
}

void PairE3GNN::compute(int eflag, int vflag) {
  // compute
  /*
     This compute function is ispired/modified from stress branch of pair-nequip
     https://github.com/mir-group/pair_nequip
  */
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  // remove consecutive guard, addressed by
  // tag_map: LAMMPS tag -> graph node index
  // graph_index_to_i: graph node index -> LAMMPS local index
  // if (atom->tag_consecutive() == 0) {
  //   error->all(FLERR, "Pair e3gnn requires consecutive atom IDs");
  // }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = list->inum; // same as nlocal, Why? is it different from atom->nlocal?
  int *ilist = list->ilist;
  int inum = list->inum;

  // tag ignore PBC
  tagint *tag = atom->tag;

  std::unordered_map<tagint, int> tag_map;
  std::vector<int> graph_index_to_i(nlocal);

  long num_atoms[1] = {nlocal};

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list

  int bound;
  if (this->nedges_bound == -1) {
    bound = std::accumulate(numneigh, numneigh + nlocal, 0);
  } else {
    bound = this->nedges_bound;
  }
  const int nedges_upper_bound = bound;

  std::vector<long> node_type;

  float edge_vec[nedges_upper_bound][3];
  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  for (int ii = 0; ii < inum; ii++) {
    // populate tag_map of local atoms
    const int i = ilist[ii];
    const int itag = tag[i];
    const int itype = type[i];
    tag_map[itag] = ii;
    graph_index_to_i[ii] = i;
    node_type.push_back(map[itype]);
  }

  int nedges = 0;
  // loop over neighbors, build graph
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int i_graph_idx = ii;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const int jtag = tag[j];
      j &= NEIGHMASK;

      const auto found = tag_map.find(jtag);
      if (found == tag_map.end()) continue;
      const int j_graph_idx = found->second;

      // we have to calculate Rij to check cutoff in lammps side
      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1],
                               x[j][2] - x[i][2]};
      const double Rij =
          delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      if (Rij < cutoff_square) {
        // if given j is not inside cutoff
        if (nedges >= nedges_upper_bound) {
          error->all(FLERR, "nedges exceeded nedges_upper_bound");
        }
        edge_idx_src[nedges] = i_graph_idx;
        edge_idx_dst[nedges] = j_graph_idx;
        edge_vec[nedges][0] = delij[0];
        edge_vec[nedges][1] = delij[1];
        edge_vec[nedges][2] = delij[2];
        nedges++;
      }
    } // j loop end
  }   // i loop end

  // convert data to Tensor
  auto inp_node_type = torch::from_blob(node_type.data(), nlocal, INTEGER_TYPE);
  auto inp_num_atoms = torch::from_blob(num_atoms, {1}, INTEGER_TYPE);

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src, {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst, {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  auto inp_edge_vec = torch::from_blob(edge_vec, {nedges, 3}, FLOAT_TYPE);
  if (print_info) {
    std::cout << " Nlocal: " << nlocal << std::endl;
    std::cout << " Nedges: " << nedges << "\n" << std::endl;
  }

  auto edge_vec_device = inp_edge_vec.to(device);
  edge_vec_device.set_requires_grad(true);

  torch::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("edge_vec", edge_vec_device);
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("nlocal", inp_num_atoms.to(torch::kCPU));

  std::vector<torch::IValue> input(1, input_dict);
  auto output = model.forward(input).toGenericDict();

  torch::Tensor energy_tensor =
      output.at("inferred_total_energy").toTensor().squeeze();

  // dE_dr
  auto grads = torch::autograd::grad({energy_tensor}, {edge_vec_device});
  torch::Tensor dE_dr = grads[0].to(torch::kCPU);

  eng_vdwl += energy_tensor.detach().to(torch::kCPU).item<float>();
  torch::Tensor force_tensor = torch::zeros({nlocal, 3});

  auto _edge_idx_src_tensor =
      edge_idx_src_tensor.repeat_interleave(3).view({nedges, 3});
  auto _edge_idx_dst_tensor =
      edge_idx_dst_tensor.repeat_interleave(3).view({nedges, 3});

  force_tensor.scatter_reduce_(0, _edge_idx_src_tensor, dE_dr, "sum");
  force_tensor.scatter_reduce_(0, _edge_idx_dst_tensor, torch::neg(dE_dr),
                               "sum");

  auto forces = force_tensor.accessor<float, 2>();

  for (int graph_idx = 0; graph_idx < nlocal; graph_idx++) {
    int i = graph_index_to_i[graph_idx];
    f[i][0] += forces[graph_idx][0];
    f[i][1] += forces[graph_idx][1];
    f[i][2] += forces[graph_idx][2];
  }

  // Virial stress from edge contributions
  if (vflag) {
    auto diag = inp_edge_vec * dE_dr;
    auto s12 = inp_edge_vec.select(1, 0) * dE_dr.select(1, 1);
    auto s23 = inp_edge_vec.select(1, 1) * dE_dr.select(1, 2);
    auto s31 = inp_edge_vec.select(1, 2) * dE_dr.select(1, 0);
    std::vector<torch::Tensor> voigt_list = {
        diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)};
    auto voigt = torch::cat(voigt_list, 1);

    torch::Tensor per_atom_stress_tensor = torch::zeros({nlocal, 6});
    auto _edge_idx_dst6_tensor =
        edge_idx_dst_tensor.repeat_interleave(6).view({nedges, 6});
    per_atom_stress_tensor.scatter_reduce_(0, _edge_idx_dst6_tensor, voigt,
                                           "sum");

    auto virial_stress_tensor =
        torch::neg(torch::sum(per_atom_stress_tensor, 0));
    auto virial_stress = virial_stress_tensor.accessor<float, 1>();

    virial[0] += virial_stress[0];
    virial[1] += virial_stress[1];
    virial[2] += virial_stress[2];
    virial[3] += virial_stress[3];
    virial[4] += virial_stress[5];
    virial[5] += virial_stress[4];

    if (vflag_atom) {
      auto per_atom_stress = per_atom_stress_tensor.accessor<float, 2>();

      for (int gi = 0; gi < nlocal; gi++) {
        const int i = graph_index_to_i[gi];
        vatom[i][0] += -per_atom_stress[gi][0];
        vatom[i][1] += -per_atom_stress[gi][1];
        vatom[i][2] += -per_atom_stress[gi][2];
        vatom[i][3] += -per_atom_stress[gi][3];
        vatom[i][4] += -per_atom_stress[gi][5];
        vatom[i][5] += -per_atom_stress[gi][4];
      }
    }
  }

  if (eflag_atom) {
    torch::Tensor atomic_energy_tensor =
        output.at("atomic_energy").toTensor().to(torch::kCPU).view({nlocal});
    auto atomic_energy = atomic_energy_tensor.accessor<float, 1>();
    for (int gi = 0; gi < nlocal; gi++) {
      const int i = graph_index_to_i[gi];
      eatom[i] += atomic_energy[gi];
    }
  }

  // if it was the first MD step
  if (this->nedges_bound == -1) {
    this->nedges_bound = nedges * 1.2;
  } // else if the nedges is too small, increase the bound
  else if (nedges > this->nedges_bound / 1.2) {
    this->nedges_bound = nedges * 1.2;
  }
}

// allocate arrays (called from coeff)
void PairE3GNN::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNN::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNN::coeff(int narg, char **arg) {

  if (allocated) {
    error->all(FLERR, "pair_e3gnn coeff called twice");
  }
  validate_serial_pair_coeff_minimum_args(narg, error);
  allocate();

  if (strcmp(arg[kSerialPairCoeffWildcardFirstIndex], "*") != 0 ||
      strcmp(arg[kSerialPairCoeffWildcardSecondIndex], "*") != 0) {
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
      {"flashTP", "version mismatch"},
      {"oeq", "version mismatch"},
      {"time", ""}};

  // model loading from input
  try {
    model = torch::jit::load(std::string(arg[kSerialPairCoeffModelPathIndex]),
                             device, meta_dict);
  } catch (const c10::Error &e) {
    error->all(FLERR, "error loading the model, check the path of the model");
  }
  // model = torch::jit::freeze(model); model is already freezed

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  // thing about dynamic recompile as tensor shape varies, this is default
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = checked_serial_positive_double_metadata(meta_dict["cutoff"], error);
  cutoff_square = cutoff * cutoff;

  // to make torch::autograd::grad() works
  if (meta_dict["oeq"] == "yes") {
    pair_e3gnn_oeq_register_autograd();
  }

  if (meta_dict["model_type"].compare("E3_equivariant_model") != 0) {
    error->all(FLERR, "given model type is not E3_equivariant_model");
  }

  const std::vector<std::string> chem_vec =
      parse_serial_chemical_symbol_tokens(
          meta_dict["chemical_symbols_to_index"], error);
  const int deployed_species_count =
      checked_serial_positive_int_metadata(meta_dict["num_species"], error);
  validate_serial_deployed_species_metadata(deployed_species_count,
                                            chem_vec.size(), error);
  int ntypes = atom->ntypes;

  bool found_flag = false;
  const int n_chem = narg - kSerialPairCoeffSpeciesStartIndex;
  validate_serial_pair_coeff_species_count(n_chem, ntypes, error);
  for (int i = 0; i < n_chem; i++) {
    found_flag = false;
    for (size_t j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i + kSerialPairCoeffSpeciesStartIndex]) ==
          0) {
        const int lammps_atom_type = i + kSerialFirstLammpsAtomType;
        map[lammps_atom_type] = static_cast<int>(j);
        found_flag = true;
        if (lmp->logfile) {
          fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                  arg[i + kSerialPairCoeffSpeciesStartIndex],
                  lammps_atom_type);
        }
        break;
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given");
    }
  }

  for (int i = kSerialFirstLammpsAtomType; i <= ntypes; i++) {
    for (int j = kSerialFirstLammpsAtomType; j <= ntypes; j++) {
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
void PairE3GNN::init_style() {
  // Newton flag is irrelevant if use only one processor for simulation
  /*
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");
  }
  */

  // full neighbor list (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNN::init_one(int i, int j) { return cutoff; }
