#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/bench/lammps_quartz"
source "${BENCH_DIR}/lammps_env.sh"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${BENCH_DIR}/runs/${DATE_TAG}"
mkdir -p "${RUN_DIR}"

DATA_FILE="${DATA_FILE:-/home/wise/minchang/DenseMLIP/lammps_sevenn/examples/vashishta/data.quartz}"
REPLICATE_X="${REPLICATE_X:-8}"
REPLICATE_Y="${REPLICATE_Y:-8}"
REPLICATE_Z="${REPLICATE_Z:-8}"
TEMPERATURE="${TEMPERATURE:-300}"
TIMESTEP="${TIMESTEP:-0.002}"
WARMUP_STEPS="${WARMUP_STEPS:-110}"
MEASURE_STEPS="${MEASURE_STEPS:-100}"
PAIR_LAYERS="${PAIR_LAYERS:-5}"
MPIRUN_CMD="${MPIRUN_CMD:-mpirun -np 1}"
VARIANTS="${VARIANTS:-baseline pairaware flash combined}"

BASELINE_MODEL="${BASELINE_MODEL:-${BENCH_DIR}/models/baseline/7net0_parallel}"
PAIRAWARE_MODEL="${PAIRAWARE_MODEL:-${BENCH_DIR}/models/pairaware/7net0_parallel_pairaware}"
FLASH_MODEL="${FLASH_MODEL:-${BENCH_DIR}/models/flash/7net0_parallel_flash}"
COMBINED_MODEL="${COMBINED_MODEL:-${BENCH_DIR}/models/combined/7net0_parallel_flash_pairaware}"

run_variant() {
    local variant="$1"
    local model_dir="$2"
    local input_file="${RUN_DIR}/in_${variant}.lmp"
    local log_file="${RUN_DIR}/log_${variant}.lammps"
    local screen_file="${RUN_DIR}/screen_${variant}.txt"

    cat > "${input_file}" <<EOF
units           metal
dimension       3
boundary        p p p
box tilt        large
atom_style      atomic

read_data       ${DATA_FILE}
replicate       ${REPLICATE_X} ${REPLICATE_Y} ${REPLICATE_Z}

pair_style      e3gnn/parallel
pair_coeff      * * ${PAIR_LAYERS} ${model_dir} Si O

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        ${TIMESTEP}
velocity        all create ${TEMPERATURE} 4928459 mom yes rot yes dist gaussian
fix             int all nvt temp ${TEMPERATURE} ${TEMPERATURE} 0.2

thermo          10
thermo_style    custom step temp pe ke etotal press vol

run             ${WARMUP_STEPS}
run             ${MEASURE_STEPS}
EOF

    echo "[run] ${variant}"
    ${MPIRUN_CMD} "${SEVENNET_LMP}" -in "${input_file}" -log "${log_file}" > "${screen_file}" 2>&1
}

for variant in ${VARIANTS}; do
    case "${variant}" in
        baseline)
            run_variant "${variant}" "${BASELINE_MODEL}"
            ;;
        pairaware)
            run_variant "${variant}" "${PAIRAWARE_MODEL}"
            ;;
        flash)
            run_variant "${variant}" "${FLASH_MODEL}"
            ;;
        combined)
            run_variant "${variant}" "${COMBINED_MODEL}"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 1
            ;;
    esac
done

python "${BENCH_DIR}/parse_quartz_bench.py" "${RUN_DIR}" | tee "${RUN_DIR}/summary_stdout.txt"
echo "Run directory: ${RUN_DIR}"
