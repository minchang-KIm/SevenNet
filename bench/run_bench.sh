#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-7net-0}"
DEVICE="${DEVICE:-auto}"
WARMUP="${WARMUP:-2}"
STEPS="${STEPS:-5}"
BENCH_SIZES="${BENCH_SIZES:-256 2000 20000}"
BENCH_MODES="${BENCH_MODES:-baseline pairaware flash combined}"

run_mode() {
  local size="$1"
  shift
  echo "=== target_atoms=${size} args=$* ==="
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/pairaware_bench.py" \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --warmup "${WARMUP}" \
    --steps "${STEPS}" \
    --target-atoms "${size}" \
    "$@"; then
    echo "SKIP target_atoms=${size} args=$*"
  fi
  echo
}

for size in ${BENCH_SIZES}; do
  for mode in ${BENCH_MODES}; do
    case "${mode}" in
      baseline)
        run_mode "${size}"
        ;;
      pairaware)
        run_mode "${size}" --enable_pairaware
        ;;
      flash)
        run_mode "${size}" --enable_flash
        ;;
      combined)
        run_mode "${size}" --enable_flash --enable_pairaware
        ;;
      *)
        echo "Unknown BENCH_MODES entry: ${mode}" >&2
        exit 1
        ;;
    esac
  done
done
