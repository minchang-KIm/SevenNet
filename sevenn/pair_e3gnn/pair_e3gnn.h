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
PairStyle(e3gnn, PairE3GNN)

#else

#ifndef LMP_PAIR_E3GNN
#define LMP_PAIR_E3GNN

#include "pair.h"

#include <torch/torch.h>

namespace LAMMPS_NS {
class PairE3GNN : public Pair {
private:
  double cutoff;
  double cutoff_square;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  int nelements;
  bool print_info = false;
  bool pair_execution = false;
  bool topology_cache = true;
  std::string pair_execution_policy = "baseline";

  int nedges_bound = -1;
  bool pair_cache_valid = false;
  torch::Tensor cached_edge_index_cpu;
  torch::Tensor cached_cell_shift_cpu;
  torch::Tensor cached_edge_pair_map_cpu;
  torch::Tensor cached_edge_pair_reverse_cpu;
  torch::Tensor cached_pair_forward_index_cpu;
  torch::Tensor cached_pair_backward_index_cpu;
  torch::Tensor cached_pair_has_reverse_cpu;

public:
  PairE3GNN(class LAMMPS *);
  ~PairE3GNN();
  void compute(int, int);

  void settings(int, char **);
  // read Atom type string from input script & related coeff
  void coeff(int, char **);
  void allocate();

  void init_style();
  double init_one(int, int);
};
} // namespace LAMMPS_NS

#endif
#endif
