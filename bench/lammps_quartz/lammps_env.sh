#!/usr/bin/env bash

set -euo pipefail

export PATH="/home/wise/minchang/DenseMLIP/lammps_sevenn/build:/home/wise/miniconda3/bin:${PATH}"
export CUDA_HOME="/home/wise/miniconda3"
export LD_LIBRARY_PATH="/home/wise/miniconda3/lib:/home/wise/miniconda3/targets/x86_64-linux/lib:/home/wise/miniconda3/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEVENNET_LMP="/home/wise/minchang/DenseMLIP/lammps_sevenn/build/lmp"

ulimit -s unlimited
