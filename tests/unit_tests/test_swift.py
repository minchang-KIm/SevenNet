import pytest
import torch

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.convolution import IrrepsConvolution
from sevenn.nn.swift_backend import (
    SwiftTPConvolution,
    _build_csr_from_coo,
    _bucket_rows,
)
from sevenn.nn.swift_helper import is_swift_available
from tests.unit_tests.test_flash import get_graphs, get_model_config


def _first_convolution(model):
    for module in model.modules():
        if isinstance(module, IrrepsConvolution):
            return module
    raise AssertionError('IrrepsConvolution not found')


def test_swift_csr_builder():
    edge_src = torch.tensor([2, 0, 1, 3, 1], dtype=torch.int32)
    edge_dst = torch.tensor([0, 1, 0, 2, 2], dtype=torch.int32)
    row_ptr, col_idx, perm = _build_csr_from_coo(edge_src, edge_dst, num_nodes=4)

    assert row_ptr.tolist() == [0, 2, 3, 5, 5]
    assert col_idx.tolist() == [2, 1, 0, 3, 1]
    assert perm.tolist() == [0, 2, 1, 3, 4]


def test_swift_row_bucket_builder():
    row_ptr = torch.tensor([0, 0, 3, 12, 90], dtype=torch.int32)
    config = SwiftTPConvolution(
        **_first_convolution(build_E3_equivariant_model(get_model_config())).convolution_kwargs
    ).launch_config

    small, medium, high = _bucket_rows(row_ptr, config)
    assert small.tolist() == [1]
    assert medium.tolist() == [2]
    assert high.tolist() == [3]


def test_swift_metadata_matches_weight_layout():
    model = build_E3_equivariant_model(get_model_config())
    conv = _first_convolution(model)
    swift = SwiftTPConvolution(**conv.convolution_kwargs)

    assert swift.path_meta.shape[0] == len(swift.reference_tp.instructions)
    assert swift.cg_coeff.numel() > 0

    last_meta = swift.path_meta[-1]
    last_weight_offset = int(last_meta[3].item())
    last_mul = int(last_meta[4].item())
    assert last_weight_offset + last_mul == swift.reference_tp.weight_numel


@pytest.mark.skipif(not is_swift_available(), reason='swift not available')
def test_swift_matches_reference_first_convolution():
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

    with torch.no_grad():
        for key, ref_module in ref_model._modules.items():
            swift_module = swift_model._modules[key]
            ref_data = ref_module(ref_data)  # type: ignore[assignment]
            swift_data = swift_module(swift_data)  # type: ignore[assignment]
            if key.endswith('convolution'):
                assert torch.allclose(ref_data.x, swift_data.x, atol=1e-5)
                break
