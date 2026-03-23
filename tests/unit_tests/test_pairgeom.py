import copy
import importlib.util
import pathlib

import ase.io
import numpy as np
import pytest
import torch
from ase.build import bulk
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.convolution import PairFusedIrrepsConvolution
from sevenn.nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    PolynomialCutoff,
    SphericalEncoding,
)
from sevenn.nn.flash_helper import is_flash_available
from sevenn.nn.pairgeom import (
    PairAwareEdgeEmbedding,
    apply_spherical_harmonic_parity,
    build_spherical_harmonic_parity_sign,
    ensure_pair_metadata,
    build_undirected_pair_mapping,
)
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.util import chemical_species_preprocess, load_checkpoint, model_from_checkpoint

file_path = pathlib.Path(__file__).parent.resolve()
data_root = (file_path.parent / 'data').resolve()
cp_0_path = str(data_root / 'checkpoints' / 'cp_0.pth')
hfo2_path = str(data_root / 'systems' / 'hfo2.extxyz')
bench_module_path = file_path.parent.parent / 'bench' / 'pairgeom_bench.py'


def _make_pair_graph() -> AtomGraphData:
    graph = {
        KEY.NODE_FEATURE: np.array([14, 14, 14]),
        KEY.ATOMIC_NUMBERS: np.array([14, 14, 14]),
        KEY.EDGE_IDX: np.array(
            [
                [0, 1, 0, 0, 1, 2],
                [1, 0, 0, 0, 2, 1],
            ]
        ),
        KEY.EDGE_VEC: np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.3, 1.2, -0.1],
                [-0.3, -1.2, 0.1],
            ]
        ),
        KEY.CELL_SHIFT: np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
            ]
        ),
        KEY.CELL_VOLUME: np.array(1.0),
        KEY.NUM_ATOMS: np.array(3),
    }
    return AtomGraphData.from_numpy_dict(graph)


def _make_model_config():
    cutoff = 4.0
    atoms = bulk('NaCl', 'rocksalt', a=4.0) * (2, 2, 2)
    config = {
        'cutoff': cutoff,
        'channel': 32,
        'lmax': 2,
        'is_parity': False,
        'num_convolution_layer': 3,
        'self_connection_type': 'nequip',
        'interaction_type': 'nequip',
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
        'train_denominator': False,
        'shift': -10.0,
        'scale': 10.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    config.update(**chemical_species_preprocess(sorted(set(atoms.get_chemical_symbols()))))
    return config


def test_pair_mapping_correctness():
    graph = _make_pair_graph()
    pair_index, pair_shift, edge_to_pair, edge_is_reversed = build_undirected_pair_mapping(
        graph[KEY.EDGE_IDX], graph[KEY.CELL_SHIFT]
    )

    reconstructed = torch.cat(
        [
            pair_index.t().index_select(0, edge_to_pair),
            pair_shift.index_select(0, edge_to_pair),
        ],
        dim=1,
    )
    expected = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 2, 0, 1, 0],
            [1, 2, 0, 1, 0],
        ],
        dtype=torch.int64,
    )
    expected_reversed = torch.tensor([False, True, False, True, False, True])

    assert torch.equal(reconstructed, expected)
    assert torch.equal(edge_is_reversed.cpu(), expected_reversed)


def test_spherical_harmonic_parity_matches_baseline():
    spherical = SphericalEncoding(lmax=3, parity=-1, normalize=True)
    vec = torch.tensor([[0.3, 1.2, -0.4], [1.1, -0.2, 0.5]], dtype=torch.float32)
    baseline = spherical(vec)
    reversed_baseline = spherical(-vec)
    parity_sign = build_spherical_harmonic_parity_sign(spherical.irreps_out)
    actual = apply_spherical_harmonic_parity(
        baseline,
        parity_sign,
        torch.tensor([True, True]),
    )
    assert torch.allclose(actual, reversed_baseline, atol=1e-6, rtol=1e-6)


def test_pairaware_edge_embedding_matches_baseline():
    graph = _make_pair_graph()
    basis = BesselBasis(cutoff_length=4.0, bessel_basis_num=8, trainable_coeff=False)
    cutoff = PolynomialCutoff(cutoff_length=4.0, poly_cut_p_value=6)
    spherical = SphericalEncoding(lmax=2, parity=-1, normalize=True)

    baseline = EdgeEmbedding(copy.deepcopy(basis), copy.deepcopy(cutoff), copy.deepcopy(spherical))
    pairaware = PairAwareEdgeEmbedding(
        copy.deepcopy(basis), copy.deepcopy(cutoff), copy.deepcopy(spherical)
    )

    out_baseline = baseline(graph.clone())
    out_pairaware = pairaware(graph.clone())

    assert torch.allclose(
        out_baseline[KEY.EDGE_LENGTH], out_pairaware[KEY.EDGE_LENGTH], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_baseline[KEY.EDGE_EMBEDDING],
        out_pairaware[KEY.EDGE_EMBEDDING],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out_baseline[KEY.EDGE_ATTR], out_pairaware[KEY.EDGE_ATTR], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_pairaware[KEY.PAIR_ATTR]
        .index_select(0, out_pairaware[KEY.EDGE_TO_PAIR])[
            ~out_pairaware[KEY.EDGE_IS_REVERSED]
        ],
        out_baseline[KEY.EDGE_ATTR][~out_pairaware[KEY.EDGE_IS_REVERSED]],
        atol=1e-6,
        rtol=1e-6,
    )


def test_pairaware_edge_embedding_uses_precomputed_metadata():
    graph = _make_pair_graph()
    graph_cached = _make_pair_graph()
    ensure_pair_metadata(graph_cached)
    basis = BesselBasis(cutoff_length=4.0, bessel_basis_num=8, trainable_coeff=False)
    cutoff = PolynomialCutoff(cutoff_length=4.0, poly_cut_p_value=6)
    spherical = SphericalEncoding(lmax=2, parity=-1, normalize=True)
    pairaware = PairAwareEdgeEmbedding(basis, cutoff, spherical)

    out_runtime = pairaware(graph.clone())
    out_cached = pairaware(graph_cached.clone())

    assert torch.allclose(
        out_runtime[KEY.EDGE_LENGTH], out_cached[KEY.EDGE_LENGTH], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_runtime[KEY.EDGE_EMBEDDING],
        out_cached[KEY.EDGE_EMBEDDING],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out_runtime[KEY.PAIR_EMBEDDING],
        out_cached[KEY.PAIR_EMBEDDING],
        atol=1e-6,
        rtol=1e-6,
    )


def test_pair_metadata_batch_offsets():
    graph_a = _make_pair_graph()
    graph_b = _make_pair_graph()
    ensure_pair_metadata(graph_a)
    ensure_pair_metadata(graph_b)

    batch = next(iter(DataLoader([graph_a, graph_b], batch_size=2)))
    restored = batch.to_data_list()

    assert len(restored) == 2
    for graph in restored:
        assert torch.equal(graph[KEY.EDGE_TO_PAIR], graph_a[KEY.EDGE_TO_PAIR])
        assert torch.equal(graph[KEY.PAIR_OWNER], graph_a[KEY.PAIR_OWNER])


def test_pairgeom_full_model_matches_baseline():
    cp = load_checkpoint(cp_0_path)
    atoms = ase.io.read(hfo2_path)
    graph = AtomGraphData.from_numpy_dict(
        dl.unlabeled_atoms_to_graph(atoms, cp.config[KEY.CUTOFF], with_shift=True)
    )
    graph[KEY.BATCH] = torch.zeros([0])

    baseline, _ = model_from_checkpoint(cp_0_path, enable_pairgeom=False)
    pairgeom, _ = model_from_checkpoint(cp_0_path, enable_pairgeom=True)
    baseline.set_is_batch_data(False)
    pairgeom.set_is_batch_data(False)

    out_baseline = baseline(graph.clone())
    out_pairgeom = pairgeom(graph.clone())

    assert torch.allclose(
        out_baseline[KEY.PRED_TOTAL_ENERGY],
        out_pairgeom[KEY.PRED_TOTAL_ENERGY],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        out_baseline[KEY.PRED_FORCE],
        out_pairgeom[KEY.PRED_FORCE],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        out_baseline[KEY.PRED_STRESS],
        out_pairgeom[KEY.PRED_STRESS],
        atol=1e-6,
        rtol=1e-5,
    )


def test_pairgeom_serial_model_uses_pair_fused_convolution():
    config = _make_model_config()
    config[KEY.USE_PAIRGEOM] = True
    model = build_E3_equivariant_model(config, parallel=False)
    assert isinstance(model, AtomGraphSequential)

    conv_modules = [
        module
        for name, module in model._modules.items()
        if name.endswith('_convolution')
    ]
    assert conv_modules
    assert all(
        isinstance(module, PairFusedIrrepsConvolution) for module in conv_modules
    )


def test_pairgeom_accelerated_model_skips_pair_fused_convolution():
    config = _make_model_config()
    config[KEY.USE_PAIRGEOM] = True
    config[KEY.USE_FLASH_TP] = True
    model = build_E3_equivariant_model(config, parallel=False)
    assert isinstance(model, AtomGraphSequential)

    conv_modules = [
        module
        for name, module in model._modules.items()
        if name.endswith('_convolution')
    ]
    assert conv_modules
    assert not any(
        isinstance(module, PairFusedIrrepsConvolution) for module in conv_modules
    )


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
def test_pairgeom_flash_matches_flash_baseline():
    atoms = bulk('NaCl', 'rocksalt', a=4.0) * (2, 2, 2)
    graph = AtomGraphData.from_numpy_dict(
        dl.unlabeled_atoms_to_graph(atoms, 4.0, with_shift=True)
    ).to('cuda')

    config = _make_model_config()
    config_flash = copy.deepcopy(config)
    config_flash[KEY.USE_FLASH_TP] = True

    config_pairgeom_flash = copy.deepcopy(config_flash)
    config_pairgeom_flash[KEY.USE_PAIRGEOM] = True

    torch.manual_seed(123)
    model_flash = build_E3_equivariant_model(config_flash, parallel=False)
    torch.manual_seed(123)
    model_pairgeom_flash = build_E3_equivariant_model(
        config_pairgeom_flash, parallel=False
    )

    assert isinstance(model_flash, AtomGraphSequential)
    assert isinstance(model_pairgeom_flash, AtomGraphSequential)

    model_flash.to('cuda')
    model_pairgeom_flash.to('cuda')
    model_flash.set_is_batch_data(False)
    model_pairgeom_flash.set_is_batch_data(False)

    out_flash = model_flash(graph.clone())
    out_pairgeom_flash = model_pairgeom_flash(graph.clone())

    assert torch.allclose(
        out_flash[KEY.PRED_TOTAL_ENERGY],
        out_pairgeom_flash[KEY.PRED_TOTAL_ENERGY],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        out_flash[KEY.PRED_FORCE],
        out_pairgeom_flash[KEY.PRED_FORCE],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        out_flash[KEY.PRED_STRESS],
        out_pairgeom_flash[KEY.PRED_STRESS],
        atol=1e-5,
        rtol=1e-5,
    )


def test_pairgeom_benchmark_smoke():
    spec = importlib.util.spec_from_file_location('pairgeom_bench', bench_module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [cp_0_path, hfo2_path, '--mode', 'all', '--steps', '1', '--warmup', '0']
    )
    results = module.run_suite(args)

    mode_names = {result['effective_mode'] for result in results}
    assert 'baseline' in mode_names
    assert 'pairgeom' in mode_names
    pairgeom_result = next(result for result in results if result['effective_mode'] == 'pairgeom')
    assert pairgeom_result['geometry_reuse_factor'] >= 1.9
