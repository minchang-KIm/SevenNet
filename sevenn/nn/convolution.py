from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .pairgeom import build_spherical_harmonic_parity_sign
from .util import broadcast


def message_gather(
    node_features: torch.Tensor, edge_dst: torch.Tensor, message: torch.Tensor
) -> torch.Tensor:
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape, dtype=node_features.dtype, device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
        pairgeom_backend: str = 'disabled',
    ) -> None:
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel
        self.pairgeom_backend = pairgeom_backend

        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )

        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet

        if not lazy_layer_instantiate:
            self.instantiate()

        self._comm_size = irreps_x.dim  # used in parallel

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'
        if KEY.PAIR_EMBEDDING in data and KEY.EDGE_TO_PAIR in data:
            pair_weight = self.weight_nn(data[KEY.PAIR_EMBEDDING])
            weight = pair_weight.index_select(0, data[KEY.EDGE_TO_PAIR])
        else:
            weight = self.weight_nn(data[self.key_weight_input])

        x = data[self.key_x]

        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        message = self.convolution(x[edge_src], data[self.key_filter], weight)

        x = message_gather(x, edge_dst, message)

        x = x.div(self.denominator)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]

        data[self.key_x] = x
        return data


@compile_mode('script')
class PairFusedIrrepsConvolution(IrrepsConvolution):
    """
    Reference pair-fused convolution for serial e3nn pairgeom models.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
        pairgeom_backend: str = 'disabled',
    ) -> None:
        super().__init__(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out,
            weight_layer_input_to_hidden=weight_layer_input_to_hidden,
            weight_layer_act=weight_layer_act,
            denominator=denominator,
            train_denominator=train_denominator,
            data_key_x=data_key_x,
            data_key_filter=data_key_filter,
            data_key_weight_input=data_key_weight_input,
            data_key_edge_idx=data_key_edge_idx,
            lazy_layer_instantiate=lazy_layer_instantiate,
            is_parallel=is_parallel,
            pairgeom_backend=pairgeom_backend,
        )
        self.register_buffer(
            'pair_parity_sign',
            build_spherical_harmonic_parity_sign(irreps_filter),
            persistent=False,
        )
        self._out_dim: int = self.convolution_kwargs['irreps_out'].dim

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.is_parallel:
            raise RuntimeError(
                'PairFusedIrrepsConvolution does not support parallel models'
            )
        if KEY.PAIR_EMBEDDING not in data or KEY.EDGE_TO_PAIR not in data:
            return super().forward(data)

        # Pair-level weight reuse is still valid here, but one large TP call over
        # directed edges is faster on GPU than two half-sized TP launches.
        if self.key_filter in data:
            return IrrepsConvolution.forward(self, data)

        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'

        if KEY.PAIR_IDX not in data or KEY.PAIR_ATTR not in data:
            return super().forward(data)

        x = data[self.key_x]
        pair_index = data[KEY.PAIR_IDX]
        pair_dst = pair_index[0]
        pair_src = pair_index[1]
        pair_attr = data[KEY.PAIR_ATTR]
        pair_weight = self.weight_nn(data[KEY.PAIR_EMBEDDING])

        if pair_src.numel() == 0:
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (pair_attr.sum() + pair_weight.sum()) * 0
        else:
            parity_sign = self.pair_parity_sign.to(
                device=pair_attr.device,
                dtype=pair_attr.dtype,
            )
            edge_src = torch.cat((pair_src, pair_dst), dim=0)
            edge_dst = torch.cat((pair_dst, pair_src), dim=0)
            edge_attr = torch.cat(
                (pair_attr, pair_attr * parity_sign), dim=0
            )
            edge_weight = torch.cat((pair_weight, pair_weight), dim=0)
            message = self.convolution(
                x.index_select(0, edge_src),
                edge_attr,
                edge_weight,
            )
            x = message_gather(x, edge_dst, message)

        x = x.div(self.denominator)
        data[self.key_x] = x
        return data


@compile_mode('script')
class IrrepsScatterGatterFusedConvolution(nn.Module):
    """
    Same as above but forward
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
        pairgeom_backend: str = 'disabled',
    ) -> None:
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel
        self.pairgeom_backend = pairgeom_backend

        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )

        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = None  # must be assigned from outside
        self.weight_nn_cls = FullyConnectedNet

        if not lazy_layer_instantiate:
            self.instantiate()

        self._comm_size = irreps_x.dim  # used in parallel
        self._out_dim: int = irreps_mid.dim

    @classmethod
    def from_irreps_convolution(cls, src: IrrepsConvolution):
        """
        I'm looking for better idea
        """
        irreps_x = src.convolution_kwargs['irreps_in1']
        ret = cls(
            irreps_x, irreps_x, irreps_x, weight_layer_input_to_hidden=[1],
        )
        ret.__dict__ = deepcopy(src.__dict__)
        ret._out_dim = src.convolution_kwargs['irreps_out'].dim
        return ret

    def instantiate(self) -> None:
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        assert self.convolution_cls is not None

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'

        x = data[self.key_x]
        if KEY.PAIR_EMBEDDING in data and KEY.EDGE_TO_PAIR in data:
            pair_weight = self.weight_nn(data[KEY.PAIR_EMBEDDING])
            weight = pair_weight.index_select(0, data[KEY.EDGE_TO_PAIR])
        else:
            weight = self.weight_nn(data[self.key_weight_input])

        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        edge_filter = data[self.key_filter]

        # No edges (e.g., single isolated atom): skip the uvu_TP CUDA kernel
        if edge_src.numel() == 0:
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (edge_filter.sum() + weight.sum()) * 0  # keep in autograd graph
        else:
            x = self.convolution(
                x,
                edge_filter,
                weight,
                edge_src.to(torch.int32),
                edge_dst.to(torch.int32),
            )

        x = x.div(self.denominator)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x

        return data


@compile_mode('script')
class PairAwareFlashTPConvolution(IrrepsScatterGatterFusedConvolution):
    """
    Pair-aware adapter for directed-edge FlashTP-style convolutions.

    The underlying TP/scatter kernel still consumes directed-edge tensors.
    This wrapper only materializes directed-edge weights/filters from the
    pair-invariant contract when possible.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
        pairgeom_backend: str = 'flash',
    ) -> None:
        super().__init__(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out,
            weight_layer_input_to_hidden=weight_layer_input_to_hidden,
            weight_layer_act=weight_layer_act,
            denominator=denominator,
            train_denominator=train_denominator,
            data_key_x=data_key_x,
            data_key_filter=data_key_filter,
            data_key_weight_input=data_key_weight_input,
            data_key_edge_idx=data_key_edge_idx,
            lazy_layer_instantiate=lazy_layer_instantiate,
            is_parallel=is_parallel,
            pairgeom_backend=pairgeom_backend,
        )
        self.register_buffer(
            'pair_parity_sign',
            build_spherical_harmonic_parity_sign(irreps_filter),
            persistent=False,
        )

    @classmethod
    def from_irreps_convolution(cls, src: IrrepsConvolution):
        ret = super().from_irreps_convolution(src)
        assert isinstance(ret, PairAwareFlashTPConvolution)
        ret.register_buffer(
            'pair_parity_sign',
            build_spherical_harmonic_parity_sign(
                ret.convolution_kwargs['irreps_in2']
            ),
            persistent=False,
        )
        return ret

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'

        x = data[self.key_x]
        if KEY.PAIR_EMBEDDING in data and KEY.EDGE_TO_PAIR in data:
            pair_weight = self.weight_nn(data[KEY.PAIR_EMBEDDING])
            weight = pair_weight.index_select(0, data[KEY.EDGE_TO_PAIR])
        else:
            weight = self.weight_nn(data[self.key_weight_input])

        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        if self.key_filter in data:
            edge_filter = data[self.key_filter]
        elif (
            KEY.PAIR_ATTR in data
            and KEY.EDGE_TO_PAIR in data
            and KEY.EDGE_IS_REVERSED in data
        ):
            edge_filter = data[KEY.PAIR_ATTR].index_select(0, data[KEY.EDGE_TO_PAIR])
            parity_sign = self.pair_parity_sign.to(
                device=edge_filter.device,
                dtype=edge_filter.dtype,
            )
            edge_filter = torch.where(
                data[KEY.EDGE_IS_REVERSED].unsqueeze(-1),
                edge_filter * parity_sign,
                edge_filter,
            )
        else:
            raise KeyError(
                'PairAwareFlashTPConvolution requires edge_attr or pair_attr '
                'with pair metadata'
            )

        if edge_src.numel() == 0:
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (edge_filter.sum() + weight.sum()) * 0
        else:
            x = self.convolution(
                x,
                edge_filter,
                weight,
                edge_src.to(torch.int32),
                edge_dst.to(torch.int32),
            )

        x = x.div(self.denominator)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data
