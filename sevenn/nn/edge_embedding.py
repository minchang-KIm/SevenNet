import math
from typing import Tuple

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, SphericalHarmonics
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


def build_undirected_pair_mapping(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_vec: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build canonical undirected pair mapping for a directed edge list.

    Returns:
        pair_index: [2, num_pairs] canonical (min, max) node pairs
        edge_to_pair: [num_edges] pair id for each directed edge
        edge_is_reversed: [num_edges] whether edge follows max -> min
    """
    src = edge_index[0]
    dst = edge_index[1]
    pair_src = torch.minimum(src, dst)
    pair_dst = torch.maximum(src, dst)
    if src.numel() == 0:
        pair_index = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        edge_to_pair = torch.empty((0,), dtype=torch.long, device=edge_index.device)
        edge_is_reversed = torch.empty(
            (0,), dtype=torch.bool, device=edge_index.device
        )
        return pair_index, edge_to_pair, edge_is_reversed

    edge_is_reversed = src > dst
    if edge_vec is not None:
        same_node = src == dst
        if torch.any(same_node):
            vec = edge_vec
            self_reversed = (
                (vec[:, 0] < 0)
                | ((vec[:, 0] == 0) & (vec[:, 1] < 0))
                | ((vec[:, 0] == 0) & (vec[:, 1] == 0) & (vec[:, 2] < 0))
            )
            edge_is_reversed = edge_is_reversed | (same_node & self_reversed)

        canonical_vec = torch.where(
            edge_is_reversed.unsqueeze(-1), -edge_vec, edge_vec
        )
        pair_descriptor = torch.cat(
            [
                pair_src.unsqueeze(-1).to(canonical_vec.dtype),
                pair_dst.unsqueeze(-1).to(canonical_vec.dtype),
                canonical_vec,
            ],
            dim=1,
        )
        unique_descriptor, edge_to_pair = torch.unique(
            pair_descriptor, dim=0, return_inverse=True
        )
        pair_index = unique_descriptor[:, :2].transpose(0, 1).to(edge_index.dtype)
        return pair_index, edge_to_pair.to(torch.long), edge_is_reversed

    pair_key = pair_src * num_nodes + pair_dst
    sorted_key, perm = torch.sort(pair_key)
    is_new_pair = torch.ones_like(sorted_key, dtype=torch.bool)
    if sorted_key.numel() > 1:
        is_new_pair[1:] = sorted_key[1:] != sorted_key[:-1]
    pair_ids_sorted = torch.cumsum(is_new_pair.to(torch.long), dim=0) - 1
    unique_key = sorted_key[is_new_pair]
    pair_index = torch.stack(
        [unique_key // num_nodes, unique_key % num_nodes], dim=0
    )

    edge_to_pair = torch.empty_like(pair_ids_sorted)
    edge_to_pair.scatter_(0, perm, pair_ids_sorted)
    return pair_index, edge_to_pair, edge_is_reversed


def build_spherical_harmonic_parity_sign(
    irreps_out: Irreps,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Build the per-channel parity sign vector for spherical harmonics.

    Channels belonging to odd-l irreps get -1, even-l irreps get +1.
    """
    signs = []
    for mul, ir in irreps_out:
        sign = -1.0 if ir.l % 2 else 1.0
        signs.append(
            torch.full((mul * ir.dim,), sign, dtype=dtype, device=device)
        )
    if len(signs) == 0:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(signs, dim=0)


def apply_spherical_harmonic_parity(
    pair_attr: torch.Tensor,
    edge_is_reversed: torch.Tensor,
    parity_sign: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Y_l(-r) = (-1)^l Y_l(r) to reversed edges.
    """
    sign = torch.where(
        edge_is_reversed.unsqueeze(-1),
        parity_sign.unsqueeze(0),
        torch.ones_like(pair_attr),
    )
    return pair_attr * sign


@compile_mode('script')
class EdgePreprocess(nn.Module):
    """
    preprocessing pos to edge vectors and edge lengths
    currently used in sevenn/scripts/deploy for lammps serial model
    """

    def __init__(self, is_stress: bool) -> None:
        super().__init__()
        # controlled by 'AtomGraphSequential'
        self.is_stress = is_stress
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            cell = data[KEY.CELL].view(-1, 3, 3)
        else:
            cell = data[KEY.CELL].view(3, 3)
        cell_shift = data[KEY.CELL_SHIFT]
        pos = data[KEY.POS]

        batch = data[KEY.BATCH]  # for deploy, must be defined first
        if self.is_stress:
            if self._is_batch_data:
                num_batch = int(batch.max().cpu().item()) + 1
                strain = torch.zeros(
                    (num_batch, 3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.bmm(
                    pos.unsqueeze(-2), sym_strain[batch]
                ).squeeze(-2)
                cell = cell + torch.bmm(cell, sym_strain)
            else:
                strain = torch.zeros(
                    (3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.mm(pos, sym_strain)
                cell = cell + torch.mm(cell, sym_strain)

        idx_src = data[KEY.EDGE_IDX][0]
        idx_dst = data[KEY.EDGE_IDX][1]

        edge_vec = pos[idx_dst] - pos[idx_src]

        if self._is_batch_data:
            edge_vec = edge_vec + torch.einsum(
                'ni,nij->nj', cell_shift, cell[batch[idx_src]]
            )
        else:
            edge_vec = edge_vec + torch.einsum(
                'ni,ij->nj', cell_shift, cell.squeeze(0)
            )
        data[KEY.EDGE_VEC] = edge_vec
        data[KEY.EDGE_LENGTH] = torch.linalg.norm(edge_vec, dim=-1)
        return data


class BesselBasis(nn.Module):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ) -> None:
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor([
            n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)
        ])
        if trainable_coeff:
            self.coeffs = nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        ur = r.unsqueeze(-1)  # to fit dimension
        return self.prefactor * torch.sin(self.coeffs * ur) / ur


class PolynomialCutoff(nn.Module):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(
        self,
        cutoff_length: float,
        poly_cut_p_value: int = 6,
    ) -> None:
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * torch.pow(r, self.p)
            + self.coeff_p1 * torch.pow(r, self.p + 1.0)
            - self.coeff_p2 * torch.pow(r, self.p + 2.0)
        )


class XPLORCutoff(nn.Module):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(
        self,
        cutoff_length: float,
        cutoff_on: float,
    ) -> None:
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return torch.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )


@compile_mode('script')
class SphericalEncoding(nn.Module):
    def __init__(
        self,
        lmax: int,
        parity: int = -1,
        normalization: str = 'component',
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.normalization = normalization
        self.irreps_in = Irreps('1x1o') if parity == -1 else Irreps('1x1e')
        self.irreps_out = Irreps.spherical_harmonics(lmax, parity)
        self.sph = SphericalHarmonics(
            self.irreps_out,
            normalize=normalize,
            normalization=normalization,
            irreps_in=self.irreps_in,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.sph(r)


@compile_mode('script')
class EdgeEmbedding(nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        basis_module: nn.Module,
        cutoff_module: nn.Module,
        spherical_module: nn.Module,
        use_pairaware: bool = False,
    ) -> None:
        super().__init__()
        self.basis_function = basis_module
        self.cutoff_function = cutoff_module
        self.spherical = spherical_module
        self.use_pairaware = use_pairaware
        parity_sign = build_spherical_harmonic_parity_sign(
            self.spherical.irreps_out
        )
        self.register_buffer('_sh_parity_sign', parity_sign, persistent=False)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if not self.use_pairaware:
            rvec = data[KEY.EDGE_VEC]
            r = torch.linalg.norm(data[KEY.EDGE_VEC], dim=-1)
            data[KEY.EDGE_LENGTH] = r
            data[KEY.EDGE_EMBEDDING] = self.basis_function(
                r
            ) * self.cutoff_function(r).unsqueeze(-1)
            data[KEY.EDGE_ATTR] = self.spherical(rvec)
            return data

        edge_index = data[KEY.EDGE_IDX]
        edge_vec = data[KEY.EDGE_VEC]
        num_nodes = int(data[KEY.NODE_FEATURE].shape[0])

        pair_index, edge_to_pair, edge_is_reversed = build_undirected_pair_mapping(
            edge_index, num_nodes, edge_vec
        )
        num_pairs = int(pair_index.shape[1])
        pair_vec = torch.zeros(
            (num_pairs, 3), dtype=edge_vec.dtype, device=edge_vec.device
        )
        orientation = torch.where(
            edge_is_reversed.unsqueeze(-1), -1.0, 1.0
        ).to(edge_vec.dtype)
        pair_vec.index_add_(0, edge_to_pair, edge_vec * orientation)
        pair_counts = torch.bincount(edge_to_pair, minlength=num_pairs).to(
            edge_vec.dtype
        )
        pair_vec = pair_vec / pair_counts.unsqueeze(-1)

        pair_length = torch.linalg.norm(pair_vec, dim=-1)
        pair_embedding = self.basis_function(
            pair_length
        ) * self.cutoff_function(pair_length).unsqueeze(-1)
        pair_attr = self.spherical(pair_vec)

        edge_length = pair_length[edge_to_pair]
        edge_embedding = pair_embedding[edge_to_pair]
        edge_attr = pair_attr[edge_to_pair]
        edge_attr = apply_spherical_harmonic_parity(
            edge_attr,
            edge_is_reversed,
            self._sh_parity_sign.to(
                dtype=edge_attr.dtype, device=edge_attr.device
            ),
        )

        data[KEY.EDGE_LENGTH] = edge_length
        data[KEY.EDGE_EMBEDDING] = edge_embedding
        data[KEY.EDGE_ATTR] = edge_attr
        num_edges = int(edge_index.shape[1])
        data[KEY.PAIRAWARE_NUM_EDGES] = torch.tensor(
            num_edges, dtype=torch.int64, device=edge_vec.device
        )
        data[KEY.PAIRAWARE_NUM_PAIRS] = torch.tensor(
            num_pairs, dtype=torch.int64, device=edge_vec.device
        )
        data[KEY.PAIRAWARE_REUSE_FACTOR] = torch.tensor(
            0.0 if num_pairs == 0 else float(num_edges) / float(num_pairs),
            dtype=edge_vec.dtype,
            device=edge_vec.device,
        )
        return data
