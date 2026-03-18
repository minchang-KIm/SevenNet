import pytest
import torch

pytest.importorskip('e3nn.o3')

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.edge_embedding import (
    BesselBasis,
    EdgeEmbedding,
    PolynomialCutoff,
    SphericalEncoding,
    apply_spherical_harmonic_parity,
    build_spherical_harmonic_parity_sign,
    build_undirected_pair_mapping,
)
from sevenn.util import chemical_species_preprocess

cutoff = 4.0


def _make_edge_embedding(use_pairaware: bool) -> EdgeEmbedding:
    cutoff = 4.0
    basis = BesselBasis(
        cutoff_length=cutoff,
        bessel_basis_num=4,
        trainable_coeff=False,
    )
    env = PolynomialCutoff(cutoff_length=cutoff)
    sph = SphericalEncoding(lmax=3, parity=-1, normalize=True)
    return EdgeEmbedding(
        basis_module=basis,
        cutoff_module=env,
        spherical_module=sph,
        use_pairaware=use_pairaware,
    )


def _make_graph():
    edge_index = torch.tensor(
        [
            [2, 0, 1, 0, 2, 1],
            [0, 2, 0, 1, 1, 2],
        ],
        dtype=torch.long,
    )
    edge_vec = torch.tensor(
        [
            [-0.6, -0.9, -0.5],
            [0.6, 0.9, 0.5],
            [-0.4, 0.1, -0.3],
            [0.4, -0.1, 0.3],
            [-1.2, -0.2, 0.7],
            [1.2, 0.2, -0.7],
        ],
        dtype=torch.float32,
    )
    return {
        KEY.EDGE_IDX: edge_index,
        KEY.EDGE_VEC: edge_vec,
        KEY.NODE_FEATURE: torch.zeros((3, 1), dtype=torch.float32),
    }


def test_pair_mapping_correctness():
    edge_index = torch.tensor(
        [
            [2, 0, 1, 0, 2, 1],
            [0, 2, 0, 1, 1, 2],
        ],
        dtype=torch.long,
    )

    pair_index, edge_to_pair, edge_is_reversed = build_undirected_pair_mapping(
        edge_index, num_nodes=3
    )

    expected_pair_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )
    expected_edge_to_pair = torch.tensor([1, 1, 0, 0, 2, 2], dtype=torch.long)
    expected_edge_is_reversed = torch.tensor(
        [True, False, True, False, True, False], dtype=torch.bool
    )

    assert torch.equal(pair_index, expected_pair_index)
    assert torch.equal(edge_to_pair, expected_edge_to_pair)
    assert torch.equal(edge_is_reversed, expected_edge_is_reversed)
    assert torch.equal(
        torch.bincount(edge_to_pair, minlength=pair_index.shape[1]),
        torch.full((pair_index.shape[1],), 2, dtype=torch.long),
    )


@pytest.mark.parametrize('lmax', [2, 3])
def test_spherical_harmonic_parity_matches_baseline(lmax):
    spherical = SphericalEncoding(lmax=lmax, parity=-1, normalize=True)
    parity_sign = build_spherical_harmonic_parity_sign(spherical.irreps_out)

    pair_vec = torch.tensor([[0.3, -0.4, 0.5]], dtype=torch.float32)
    baseline = spherical(torch.cat([pair_vec, -pair_vec], dim=0))
    pair_attr = spherical(pair_vec)

    edge_attr = apply_spherical_harmonic_parity(
        pair_attr.repeat(2, 1),
        torch.tensor([False, True], dtype=torch.bool),
        parity_sign,
    )

    assert torch.allclose(edge_attr, baseline, atol=1e-6, rtol=1e-6)


def test_pairaware_edge_embedding_matches_baseline():
    baseline = _make_edge_embedding(use_pairaware=False)
    pairaware = _make_edge_embedding(use_pairaware=True)

    data_baseline = _make_graph()
    data_pairaware = _make_graph()

    out_baseline = baseline(data_baseline)
    out_pairaware = pairaware(data_pairaware)

    assert torch.allclose(
        out_baseline[KEY.EDGE_LENGTH], out_pairaware[KEY.EDGE_LENGTH], atol=1e-6
    )
    assert torch.allclose(
        out_baseline[KEY.EDGE_EMBEDDING],
        out_pairaware[KEY.EDGE_EMBEDDING],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out_baseline[KEY.EDGE_ATTR],
        out_pairaware[KEY.EDGE_ATTR],
        atol=1e-6,
        rtol=1e-6,
    )


def _make_model_config():
    config = {
        KEY.CUTOFF: cutoff,
        KEY.NODE_FEATURE_MULTIPLICITY: 4,
        KEY.RADIAL_BASIS: {KEY.RADIAL_BASIS_NAME: 'bessel'},
        KEY.CUTOFF_FUNCTION: {KEY.CUTOFF_FUNCTION_NAME: 'poly_cut'},
        KEY.INTERACTION_TYPE: 'nequip',
        KEY.LMAX: 2,
        KEY.IS_PARITY: True,
        KEY.NUM_CONVOLUTION: 2,
        KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [16, 16],
        KEY.ACTIVATION_RADIAL: 'silu',
        KEY.ACTIVATION_SCARLAR: {'e': 'silu', 'o': 'tanh'},
        KEY.ACTIVATION_GATE: {'e': 'silu', 'o': 'tanh'},
        KEY.CONV_DENOMINATOR: 8.0,
        KEY.TRAIN_DENOMINTAOR: False,
        KEY.SELF_CONNECTION_TYPE: 'nequip',
        KEY.SHIFT: -1.0,
        KEY.SCALE: 1.0,
        KEY.TRAIN_SHIFT_SCALE: False,
        KEY.IRREPS_MANUAL: False,
        KEY.LMAX_EDGE: -1,
        KEY.LMAX_NODE: -1,
        KEY.READOUT_AS_FCN: False,
        KEY.USE_BIAS_IN_LINEAR: False,
        KEY._NORMALIZE_SPH: True,
    }
    config.update(chemical_species_preprocess(['Hf', 'O']))
    return config


def test_pairaware_full_model_matches_baseline():
    atoms = dl.ase_reader('tests/data/systems/hfo2.extxyz')[0]
    graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(atoms, cutoff))

    torch.manual_seed(1234)
    model_baseline = build_E3_equivariant_model(_make_model_config())
    torch.manual_seed(1234)
    model_pairaware = build_E3_equivariant_model(
        {**_make_model_config(), KEY.USE_PAIRAWARE: True}
    )

    model_baseline.set_is_batch_data(False)
    model_pairaware.set_is_batch_data(False)

    out_baseline = model_baseline(graph.clone())
    out_pairaware = model_pairaware(graph.clone())

    assert torch.allclose(
        out_baseline[KEY.PRED_TOTAL_ENERGY],
        out_pairaware[KEY.PRED_TOTAL_ENERGY],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out_baseline[KEY.PRED_FORCE],
        out_pairaware[KEY.PRED_FORCE],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        out_baseline[KEY.PRED_STRESS],
        out_pairaware[KEY.PRED_STRESS],
        atol=1e-5,
        rtol=1e-5,
    )
