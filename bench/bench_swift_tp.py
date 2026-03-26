"""
Benchmark the unfused convolution path against the experimental SWIFT-TP path.

This benchmark targets a different optimization axis than FlashTP:
it removes per-edge message materialization by fusing on-the-fly SH evaluation
with destination-centric aggregation.
"""

import argparse
import time

import torch

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from tests.unit_tests.test_flash import get_graphs, get_model_config


def _first_convolution_module(model):
    for key, module in model._modules.items():
        if key.endswith('convolution'):
            return module
    raise RuntimeError('No convolution module found')


def _time_module(module, data, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        module(data)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        module(data)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for bench_swift_tp.py')

    torch.manual_seed(777)
    ref_model = build_E3_equivariant_model(get_model_config(), parallel=False)
    torch.manual_seed(777)
    swift_cfg = get_model_config()
    swift_cfg[KEY.USE_SWIFT_TP] = True
    swift_model = build_E3_equivariant_model(swift_cfg, parallel=False)

    ref_model.to('cuda')
    swift_model.to('cuda')
    ref_model.set_is_batch_data(True)
    swift_model.set_is_batch_data(True)

    ref_data = ref_model._preprocess(get_graphs(batched=True))
    swift_data = swift_model._preprocess(get_graphs(batched=True))

    ref_module = _first_convolution_module(ref_model)
    swift_module = _first_convolution_module(swift_model)

    with torch.no_grad():
        ref_time = _time_module(ref_module, ref_data, args.warmup, args.iters)
        swift_time = _time_module(
            swift_module, swift_data, args.warmup, args.iters
        )

    print(f'reference_s={ref_time:.6f}')
    print(f'swift_s={swift_time:.6f}')
    print(f'speedup={ref_time / swift_time:.3f}')


if __name__ == '__main__':
    main()
