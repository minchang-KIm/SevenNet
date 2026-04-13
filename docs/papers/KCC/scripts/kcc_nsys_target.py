from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

import numpy as np
import torch

from kcc_common import FLASH_CASES, REFERENCE_CASE, resolve_device, supported_dataset_samples, sync_if_needed
from sevenn.calculator import SevenNetCalculator


CASES = {
    REFERENCE_CASE["case"]: REFERENCE_CASE,
    **{case["case"]: case for case in FLASH_CASES},
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(argv)

    samples = supported_dataset_samples(dataset_names=[args.dataset])
    if not samples:
        raise SystemExit(f"Dataset not benchmarkable: {args.dataset}")
    sample = samples[0]
    case = CASES[args.case]
    device = resolve_device()
    calc = SevenNetCalculator(
        model="7net-omni",
        modal=sample.modal,
        device=device,
        enable_flash=case["enable_flash"],
        enable_pair_execution=case["enable_pair_execution"],
        pair_execution_policy=case.get("pair_execution_policy"),
    )

    for _ in range(args.warmup):
        sync_if_needed(device)
        calc.calculate(sample.atoms)
        sync_if_needed(device)

    if args.loop:
        while True:
            sync_if_needed(device)
            calc.calculate(sample.atoms)
            sync_if_needed(device)

    timings = []
    for _ in range(args.repeat):
        sync_if_needed(device)
        start = time.perf_counter()
        calc.calculate(sample.atoms)
        sync_if_needed(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    payload = {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "case": args.case,
        "repeat": args.repeat,
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings, ddof=0)),
        "resolved_policy": calc.pair_execution_config["resolved_policy"],
    }
    print(json.dumps(payload, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
