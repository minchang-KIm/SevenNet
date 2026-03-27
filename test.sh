#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="all"
OUTPUT_DIR="${SEVENN_OUTPUT_DIR:-}"
REPEAT="${SEVENN_REPEAT:-7}"
WARMUP="${SEVENN_WARMUP:-2}"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --lammps-cmd)
      export SEVENN_LAMMPS_CMD="$2"
      shift 2
      ;;
    --mpirun-cmd)
      export SEVENN_MPIRUN_CMD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP="$(date +"%Y%m%d-%H%M%S")-$$"
  OUTPUT_DIR="${ROOT_DIR}/bench/results/${TIMESTAMP}"
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "Mode: ${MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -m bench.runner \
  --mode "${MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --repeat "${REPEAT}" \
  --warmup "${WARMUP}"

echo "Artifacts written to ${OUTPUT_DIR}"
