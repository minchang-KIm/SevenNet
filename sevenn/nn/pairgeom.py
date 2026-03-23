from typing import Optional, Tuple

import torch
import torch.nn as nn
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .edge_embedding import EdgeEmbedding


def build_spherical_harmonic_parity_sign(irreps: Irreps) -> torch.Tensor:
    signs = []
    for mul, ir in irreps:
        sign = -1.0 if ir.l % 2 else 1.0
        signs.extend([sign] * (mul * ir.dim))
    return torch.tensor(signs, dtype=torch.float32)


def apply_spherical_harmonic_parity(
    edge_attr: torch.Tensor,
    parity_sign: torch.Tensor,
    edge_is_reversed: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(edge_is_reversed):
        return edge_attr
    parity_sign = parity_sign.to(device=edge_attr.device, dtype=edge_attr.dtype)
    return torch.where(
        edge_is_reversed.unsqueeze(-1),
        edge_attr * parity_sign,
        edge_attr,
    )


def _canonicalize_pair_direction(
    edge_index: torch.Tensor,
    cell_shift: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = edge_index[0]
    dst = edge_index[1]
    edge_is_reversed = src > dst

    self_mask = src == dst
    if torch.any(self_mask):
        self_shift = cell_shift[self_mask]
        nonzero_mask = self_shift != 0
        if not torch.all(torch.any(nonzero_mask, dim=1)):
            raise ValueError(
                'pairgeom requires reverse edges; zero-shift self edges are not '
                'supported'
            )
        first_nonzero = nonzero_mask.to(torch.int64).argmax(dim=1, keepdim=True)
        first_vals = self_shift.gather(1, first_nonzero).squeeze(1)
        edge_is_reversed = edge_is_reversed.clone()
        edge_is_reversed[self_mask] = first_vals < 0

    pair_src = torch.where(edge_is_reversed, dst, src)
    pair_dst = torch.where(edge_is_reversed, src, dst)
    pair_shift = torch.where(edge_is_reversed.unsqueeze(1), -cell_shift, cell_shift)
    return pair_src, pair_dst, pair_shift, edge_is_reversed


def build_undirected_pair_mapping(
    edge_index: torch.Tensor,
    cell_shift: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('pairgeom expects edge_index with shape [2, num_edges]')

    num_edges = edge_index.shape[1]
    if cell_shift is None:
        cell_shift = torch.zeros(
            (num_edges, 3), dtype=torch.int64, device=edge_index.device
        )
    else:
        if cell_shift.ndim != 2 or cell_shift.shape[1] != 3:
            raise ValueError(
                'pairgeom expects cell_shift with shape [num_edges, 3]'
            )
        cell_shift = torch.round(cell_shift).to(
            dtype=torch.int64, device=edge_index.device
        )

    pair_src, pair_dst, pair_shift, edge_is_reversed = _canonicalize_pair_direction(
        edge_index, cell_shift
    )
    pair_keys = torch.cat(
        [
            pair_src.unsqueeze(1),
            pair_dst.unsqueeze(1),
            pair_shift,
        ],
        dim=1,
    )
    unique_keys, edge_to_pair, counts = torch.unique(
        pair_keys, dim=0, return_inverse=True, return_counts=True, sorted=True
    )

    if not torch.all(counts == 2):
        bad_count = int(counts[counts != 2][0].item())
        raise ValueError(
            'pairgeom requires exactly two directed edges per undirected pair '
            f'(found count={bad_count})'
        )

    pair_index = unique_keys[:, :2].t().contiguous()
    unique_shift = unique_keys[:, 2:].contiguous()
    return pair_index, unique_shift, edge_to_pair, edge_is_reversed


def build_pair_metadata(
    edge_index: torch.Tensor,
    cell_shift: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_index, pair_shift, edge_to_pair, edge_is_reversed = (
        build_undirected_pair_mapping(edge_index, cell_shift)
    )
    pair_owner = torch.empty(
        pair_index.shape[1], dtype=torch.int64, device=edge_index.device
    )
    if edge_to_pair.numel():
        canonical_edge_idx = torch.arange(
            edge_index.shape[1], dtype=torch.int64, device=edge_index.device
        )[~edge_is_reversed]
        pair_owner[edge_to_pair[~edge_is_reversed]] = canonical_edge_idx
    return pair_index, pair_shift, edge_to_pair, edge_is_reversed, pair_owner


def ensure_pair_metadata(
    data: AtomGraphDataType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = (
        KEY.PAIR_IDX,
        KEY.PAIR_SHIFT,
        KEY.EDGE_TO_PAIR,
        KEY.EDGE_IS_REVERSED,
        KEY.PAIR_OWNER,
    )
    if all(key in data for key in keys):
        return (
            data[KEY.PAIR_IDX],
            data[KEY.PAIR_SHIFT],
            data[KEY.EDGE_TO_PAIR],
            data[KEY.EDGE_IS_REVERSED],
            data[KEY.PAIR_OWNER],
        )

    pair_metadata = build_pair_metadata(data[KEY.EDGE_IDX], data.get(KEY.CELL_SHIFT))
    (
        data[KEY.PAIR_IDX],
        data[KEY.PAIR_SHIFT],
        data[KEY.EDGE_TO_PAIR],
        data[KEY.EDGE_IS_REVERSED],
        data[KEY.PAIR_OWNER],
    ) = pair_metadata
    return pair_metadata


class PairAwareEdgeEmbedding(EdgeEmbedding):
    def __init__(
        self,
        basis_module: nn.Module,
        cutoff_module: nn.Module,
        spherical_module: nn.Module,
    ) -> None:
        super().__init__(basis_module, cutoff_module, spherical_module)
        self.register_buffer(
            'parity_sign',
            build_spherical_harmonic_parity_sign(self.spherical.irreps_out),
            persistent=False,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if (
            KEY.PAIR_EMBEDDING in data
            and KEY.EDGE_LENGTH in data
            and KEY.EDGE_EMBEDDING in data
            and KEY.EDGE_ATTR in data
        ):
            return data

        rvec = data[KEY.EDGE_VEC]
        _, _, edge_to_pair, edge_is_reversed, pair_owner = ensure_pair_metadata(
            data
        )
        pair_vec = rvec.index_select(0, pair_owner)

        pair_r = torch.linalg.norm(pair_vec, dim=-1)
        pair_embedding = self.basis_function(pair_r) * self.cutoff_function(
            pair_r
        ).unsqueeze(-1)
        pair_attr = self.spherical(pair_vec)

        data[KEY.PAIR_EMBEDDING] = pair_embedding
        data[KEY.EDGE_LENGTH] = pair_r.index_select(0, edge_to_pair)
        data[KEY.EDGE_EMBEDDING] = pair_embedding.index_select(0, edge_to_pair)
        edge_attr = pair_attr.index_select(0, edge_to_pair)
        data[KEY.EDGE_ATTR] = apply_spherical_harmonic_parity(
            edge_attr, self.parity_sign, edge_is_reversed
        )
        return data
