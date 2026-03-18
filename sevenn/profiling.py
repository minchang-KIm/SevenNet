import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class ProfileTimings:
    total_ms: float = 0.0
    geometry_ms: float = 0.0
    tensor_product_ms: float = 0.0


def synchronize_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class ModuleProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.timings = ProfileTimings()
        self._pending: Dict[str, object] = {}
        self._cuda_measurements: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self.timings = ProfileTimings()
        self._pending = {}
        self._cuda_measurements = []

    def _now(self) -> float:
        return time.perf_counter()

    def _start(self, key: str) -> None:
        if self.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending[key] = start
        else:
            self._pending[key] = self._now()

    def _stop(self, key: str, bucket: str) -> None:
        if self.device.type == 'cuda':
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self._pending.pop(key)
            assert isinstance(start, torch.cuda.Event)
            self._cuda_measurements.append((bucket, start, end))
            return

        start = self._pending.pop(key)
        assert isinstance(start, float)
        elapsed_ms = (self._now() - start) * 1000.0
        setattr(self.timings, bucket, getattr(self.timings, bucket) + elapsed_ms)

    def flush(self) -> None:
        if self.device.type != 'cuda' or len(self._cuda_measurements) == 0:
            return
        synchronize_device(self.device)
        for bucket, start, end in self._cuda_measurements:
            elapsed_ms = float(start.elapsed_time(end))
            setattr(self.timings, bucket, getattr(self.timings, bucket) + elapsed_ms)
        self._cuda_measurements = []

    def make_pre_hook(self, key: str):
        def hook(_module, _inputs):
            self._start(key)

        return hook

    def make_post_hook(self, key: str, bucket: str):
        def hook(_module, _inputs, _output):
            self._stop(key, bucket)

        return hook

    def register(self, model: torch.nn.Module):
        handles = []
        for name, module in model._modules.items():
            if name == 'edge_embedding':
                handles.append(module.register_forward_pre_hook(self.make_pre_hook(name)))
                handles.append(
                    module.register_forward_hook(
                        self.make_post_hook(name, 'geometry_ms')
                    )
                )
            elif name.endswith('_convolution'):
                handles.append(module.register_forward_pre_hook(self.make_pre_hook(name)))
                handles.append(
                    module.register_forward_hook(
                        self.make_post_hook(name, 'tensor_product_ms')
                    )
                )
        return handles
