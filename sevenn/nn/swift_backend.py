import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, SphericalHarmonics, TensorProduct, wigner_3j

_EXTENSION = None
_EXTENSION_ERROR = None


@dataclass(frozen=True)
class SwiftLaunchConfig:
    small_row_threshold: int = 8
    cta_row_threshold: int = 64
    out_tile: int = 8
    warps_per_block: int = 4
    enable_scalar_fastpath: bool = True


class _CSRCache:
    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._cache: 'OrderedDict[Tuple[int, ...], Tuple[torch.Tensor, ...]]' = (
            OrderedDict()
        )

    def get(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
        config: SwiftLaunchConfig,
    ) -> Tuple[torch.Tensor, ...]:
        key = (
            int(edge_src.data_ptr()),
            int(edge_dst.data_ptr()),
            int(edge_src.numel()),
            int(num_nodes),
            int(edge_src.device.index or -1),
            int(config.small_row_threshold),
            int(config.cta_row_threshold),
        )
        cached = self._cache.get(key)
        if cached is None:
            row_ptr, col_idx, perm = _build_csr_from_coo(
                edge_src, edge_dst, num_nodes
            )
            buckets = _bucket_rows(row_ptr, config)
            cached = (row_ptr, col_idx, perm, *buckets)
            self._cache[key] = cached
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return cached


def _extension_sources() -> Tuple[str, str]:
    base = Path(__file__).resolve().parents[1] / 'swift_tp'
    return str(base / 'swift_tp.cpp'), str(base / 'swift_tp_kernel.cu')


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError('Failed to build SWIFT-TP extension') from _EXTENSION_ERROR

    from torch.utils.cpp_extension import load

    cpp_src, cu_src = _extension_sources()
    try:
        _EXTENSION = load(
            name='sevenn_swift_tp',
            sources=[cpp_src, cu_src],
            extra_cflags=['-O3'],
            extra_cuda_cflags=[
                '-O3',
                '--expt-relaxed-constexpr',
                '--ptxas-options=-v',
            ],
            verbose=True,
        )
        return _EXTENSION
    except Exception as exc:  # pragma: no cover - depends on CUDA toolchain
        _EXTENSION_ERROR = exc
        raise


def _build_csr_from_coo(
    edge_src: torch.Tensor, edge_dst: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = edge_dst.argsort(stable=True)
    sorted_dst = edge_dst.index_select(0, perm)
    col_idx = edge_src.index_select(0, perm).to(torch.int32)
    counts = torch.bincount(sorted_dst.to(torch.int64), minlength=num_nodes)
    row_ptr = torch.zeros(
        num_nodes + 1, dtype=torch.int32, device=edge_dst.device
    )
    row_ptr[1:] = counts.cumsum(0).to(torch.int32)
    return row_ptr, col_idx, perm


def _bucket_rows(
    row_ptr: torch.Tensor, config: SwiftLaunchConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    degrees = row_ptr[1:].to(torch.int64) - row_ptr[:-1].to(torch.int64)
    row_ids = torch.arange(
        degrees.numel(), dtype=torch.int64, device=row_ptr.device
    )
    non_empty = degrees > 0
    small = row_ids[
        non_empty & (degrees <= config.small_row_threshold)
    ].to(torch.int32)
    medium = row_ids[
        non_empty
        & (degrees > config.small_row_threshold)
        & (degrees <= config.cta_row_threshold)
    ].to(torch.int32)
    high = row_ids[degrees > config.cta_row_threshold].to(torch.int32)
    return small, medium, high


def _flatten_wigner_block(l1: int, l2: int, l3: int, scale: float) -> torch.Tensor:
    block = wigner_3j(l1, l2, l3) * scale
    return block.permute(2, 0, 1).contiguous().reshape(-1).to(torch.float32)


def _build_path_metadata(
    tp: TensorProduct,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in1_slices = tp.irreps_in1.slices()
    in2_slices = tp.irreps_in2.slices()
    out_slices = tp.irreps_out.slices()

    meta_rows = []
    coeff_rows = []
    coeff_offset = 0
    weight_offset = 0

    for ins in tp.instructions:
        if ins.connection_mode != 'uvu':
            raise NotImplementedError('SWIFT-TP only supports uvu paths')
        if not ins.has_weight:
            raise NotImplementedError('SWIFT-TP expects weighted paths')

        mul_in, ir_in = tp.irreps_in1[ins.i_in1]
        mul_filter, ir_filter = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]

        if mul_filter != 1:
            raise NotImplementedError('SWIFT-TP expects 1x spherical irreps')
        if mul_in != mul_out:
            raise NotImplementedError('SWIFT-TP expects channelwise uvu paths')
        if ins.path_shape != (mul_in, 1):
            raise NotImplementedError('Unexpected path shape for uvu path')

        coeff_block = _flatten_wigner_block(
            ir_in.l, ir_filter.l, ir_out.l, ins.path_weight
        )
        meta_rows.append(
            [
                in1_slices[ins.i_in1].start,
                in2_slices[ins.i_in2].start,
                out_slices[ins.i_out].start,
                weight_offset,
                mul_in,
                ir_in.dim,
                ir_filter.dim,
                ir_out.dim,
                coeff_offset,
            ]
        )
        coeff_rows.append(coeff_block)
        coeff_offset += coeff_block.numel()
        weight_offset += mul_in

    return (
        torch.tensor(meta_rows, dtype=torch.int32),
        torch.cat(coeff_rows, dim=0),
    )


def _scatter_reference(
    num_nodes: int, edge_dst: torch.Tensor, message: torch.Tensor
) -> torch.Tensor:
    out = message.new_zeros((num_nodes, message.shape[1]))
    index = edge_dst.to(torch.long).view(-1, 1).expand_as(message)
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


class SwiftTPConvolution(nn.Module):
    """
    Forward-only experimental backend for fixed-shape SevenNet convolutions.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights: bool = False,
        internal_weights: bool = False,
    ) -> None:
        super().__init__()
        if shared_weights or internal_weights:
            raise NotImplementedError('SWIFT-TP expects external edge weights')

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.launch_config = SwiftLaunchConfig()
        self._csr_cache = _CSRCache()

        self._validate_supported_irreps()

        self.reference_tp = TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.spherical = SphericalHarmonics(
            self.irreps_in2,
            normalize=True,
            normalization='component',
            irreps_in=Irreps('1x1e'),
        )
        path_meta, cg_coeff = _build_path_metadata(self.reference_tp)
        self.register_buffer('path_meta', path_meta, persistent=False)
        self.register_buffer('cg_coeff', cg_coeff, persistent=False)
        self._debug_compare = os.getenv('SEVENN_SWIFT_DEBUG_COMPARE') == '1'

    def _validate_supported_irreps(self) -> None:
        if self.irreps_in2.lmax > 3:
            raise NotImplementedError('SWIFT-TP only supports lmax <= 3')
        if any(mul != 1 for mul, _ in self.irreps_in2):
            raise NotImplementedError('SWIFT-TP expects one SH block per degree')
        if any(ir.p != 1 for _, ir in self.irreps_in1):
            raise NotImplementedError('SWIFT-TP only supports parity=False input')
        if any(ir.p != 1 for _, ir in self.irreps_in2):
            raise NotImplementedError('SWIFT-TP only supports parity=False SH')
        if any(ir.p != 1 for _, ir in self.irreps_out):
            raise NotImplementedError('SWIFT-TP only supports parity=False out')

    def set_launch_config(self, **kwargs) -> None:
        self.launch_config = SwiftLaunchConfig(
            **{**self.launch_config.__dict__, **kwargs}
        )

    def _reference_forward(
        self,
        x: torch.Tensor,
        edge_vec: torch.Tensor,
        weight: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        sh = self.spherical(edge_vec)
        msg = self.reference_tp(
            x.index_select(0, edge_src.to(torch.long)), sh, weight
        )
        return _scatter_reference(x.shape[0], edge_dst, msg)

    def _can_use_fused(
        self, x: torch.Tensor, edge_vec: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        return (
            x.is_cuda
            and edge_vec.is_cuda
            and weight.is_cuda
            and x.dtype == torch.float32
            and edge_vec.dtype == torch.float32
            and weight.dtype == torch.float32
            and not (
                torch.is_grad_enabled()
                and (x.requires_grad or edge_vec.requires_grad or weight.requires_grad)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_vec: torch.Tensor,
        weight: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        if not self._can_use_fused(x, edge_vec, weight):
            return self._reference_forward(x, edge_vec, weight, edge_src, edge_dst)

        extension = _load_extension()

        (
            row_ptr,
            col_idx,
            perm,
            small_rows,
            medium_rows,
            high_rows,
        ) = self._csr_cache.get(
            edge_src.to(torch.int32),
            edge_dst.to(torch.int32),
            x.shape[0],
            self.launch_config,
        )

        edge_vec_sorted = edge_vec.index_select(0, perm.to(torch.long)).contiguous()
        weight_sorted = weight.index_select(0, perm.to(torch.long)).contiguous()
        out = extension.swift_forward(
            x.contiguous(),
            edge_vec_sorted,
            weight_sorted,
            row_ptr.contiguous(),
            col_idx.contiguous(),
            small_rows.contiguous(),
            medium_rows.contiguous(),
            high_rows.contiguous(),
            self.path_meta.contiguous(),
            self.cg_coeff.contiguous(),
            int(self.irreps_out.dim),
            int(self.launch_config.small_row_threshold),
            int(self.launch_config.cta_row_threshold),
            int(self.launch_config.out_tile),
            int(self.launch_config.warps_per_block),
            bool(self.launch_config.enable_scalar_fastpath),
        )

        if self._debug_compare:
            ref = self._reference_forward(x, edge_vec, weight, edge_src, edge_dst)
            if not torch.allclose(out, ref, atol=1e-5, rtol=1e-4):
                diff = (out - ref).abs().max().item()
                raise RuntimeError(
                    f'SWIFT-TP debug compare failed, max abs diff={diff:.3e}'
                )

        return out
