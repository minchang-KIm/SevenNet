#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PYTHON="$SCRIPT_DIR/../.venv/bin/python"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python interpreter not found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT:-7net-0}"
DEVICE="${DEVICE:-auto}"
WARMUP="${WARMUP:-2}"
STEPS="${STEPS:-5}"
BENCH_SIZES="${BENCH_SIZES:-256 2000 20000}"
BENCH_MODES="${BENCH_MODES:-baseline pairaware flash combined}"
BENCH_ALLOW_ACCELERATOR_SKIP="${BENCH_ALLOW_ACCELERATOR_SKIP:-1}"

run_mode() {
  local mode="$1"
  local size="$2"
  shift 2
  echo "=== target_atoms=${size} args=$* ==="
  if "${PYTHON_BIN}" "${SCRIPT_DIR}/pairaware_bench.py" \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --warmup "${WARMUP}" \
    --steps "${STEPS}" \
    --target-atoms "${size}" \
    "$@"; then
    :
  else
    local status=$?
    if [[ ("${mode}" == "flash" || "${mode}" == "combined") && "${status}" -eq 2 && "${BENCH_ALLOW_ACCELERATOR_SKIP}" == "1" ]]; then
      echo "SKIP target_atoms=${size} args=$*"
    else
      echo "FAIL target_atoms=${size} args=$* status=${status}" >&2
      exit "${status}"
    fi
  fi
  echo
}

for size in ${BENCH_SIZES}; do
  for mode in ${BENCH_MODES}; do
    case "${mode}" in
      baseline)
        run_mode "${mode}" "${size}"
        ;;
      pairaware)
        run_mode "${mode}" "${size}" --enable_pairaware
        ;;
      flash)
        run_mode "${mode}" "${size}" --enable_flash
        ;;
      combined)
        run_mode "${mode}" "${size}" --enable_flash --enable_pairaware
        ;;
      *)
        echo "Unknown BENCH_MODES entry: ${mode}" >&2
        exit 1
        ;;
    esac
  done
done
