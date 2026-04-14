import argparse
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from ase.build import bulk
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
import sevenn.pair_runtime as pair_runtime
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.main.sevenn_get_model import add_args as add_get_model_args
from sevenn.main.sevenn_inference import add_args as add_inference_args
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.flash_helper import is_flash_available
from sevenn.scripts import deploy as deploy_mod
from sevenn.util import chemical_species_preprocess, load_checkpoint


def _manual_graph(edge_vec_scale: float = 1.0) -> AtomGraphData:
    graph = AtomGraphData.from_numpy_dict(
        {
            KEY.ATOMIC_NUMBERS: np.array([1, 2, 3], dtype=np.int64),
            KEY.POS: np.zeros((3, 3), dtype=np.float32),
            KEY.EDGE_IDX: np.array([[0, 1, 0], [1, 0, 2]], dtype=np.int64),
            KEY.EDGE_VEC: np.array(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            * edge_vec_scale,
            KEY.NUM_ATOMS: np.array(3, dtype=np.int64),
        }
    )
    return graph


def _pair_cfg(**overwrite):
    cfg = {
        'use': True,
        'policy': 'full',
        'fuse_reduction': True,
        'use_topology_cache': True,
        'distributed_schedule': 'auto',
        'backend_policy': 'prefer_common',
    }
    cfg.update(overwrite)
    return cfg


def _baseline_cfg():
    return pair_runtime.normalize_pair_execution_config({'use': False})


def _model_cfg():
    cfg = {
        'cutoff': 4.5,
        'channel': 4,
        'lmax': 1,
        'is_parity': False,
        'num_convolution_layer': 2,
        'self_connection_type': 'nequip',
        'interaction_type': 'nequip',
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [8],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 4.0,
        'train_denominator': False,
        'shift': -1.0,
        'scale': 1.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    cfg.update(chemical_species_preprocess(['Na', 'Cl']))
    return cfg


def _flash_checkpoint_cfg():
    cfg = _model_cfg()
    cfg.update(
        {
            'channel': 32,
            'lmax': 2,
            'num_convolution_layer': 3,
            'weight_nn_hidden_neurons': [64, 64],
            'conv_denominator': 30.0,
        }
    )
    return cfg


def test_pair_metadata_builder_and_cache_reuse():
    graph = _manual_graph()
    meta = pair_runtime.build_pair_metadata(graph[KEY.EDGE_IDX], graph[KEY.EDGE_VEC])

    assert meta[KEY.EDGE_PAIR_MAP].tolist() == [0, 0, 1]
    assert meta[KEY.EDGE_PAIR_REVERSE].tolist() == [False, True, False]
    assert meta[KEY.PAIR_EDGE_FORWARD_INDEX].tolist() == [0, 2]
    assert meta[KEY.PAIR_EDGE_BACKWARD_INDEX].tolist() == [1, 2]
    assert meta[KEY.PAIR_EDGE_HAS_REVERSE].tolist() == [True, False]
    assert meta[KEY.PAIR_EDGE_VEC].shape == (2, 3)
    assert meta[KEY.PAIR_TOPOLOGY_SIGNATURE].shape == (2,)

    cache_state = {}
    cfg = _pair_cfg()
    graph1 = _manual_graph(edge_vec_scale=1.0)
    graph1, cache_state = pair_runtime.prepare_pair_metadata(
        graph1, cfg, cache_state=cache_state, num_atoms=3
    )

    graph2 = _manual_graph(edge_vec_scale=2.0)
    graph2, cache_state = pair_runtime.prepare_pair_metadata(
        graph2, cfg, cache_state=cache_state, num_atoms=3
    )

    assert graph1[KEY.PAIR_EDGE_VEC].shape == graph2[KEY.PAIR_EDGE_VEC].shape
    assert torch.allclose(
        graph2[KEY.PAIR_EDGE_VEC],
        graph2[KEY.EDGE_VEC].index_select(0, graph2[KEY.PAIR_EDGE_FORWARD_INDEX]),
    )
    assert torch.equal(cache_state[KEY.PAIR_TOPOLOGY_SIGNATURE], graph2[KEY.PAIR_TOPOLOGY_SIGNATURE].cpu())


def test_pair_metadata_builder_with_cell_shift_vectorized():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.int64)
    edge_vec = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cell_shift = torch.zeros((3, 3), dtype=torch.float32)

    meta = pair_runtime.build_pair_metadata(
        edge_index,
        edge_vec,
        cell_shift=cell_shift,
    )

    assert meta[KEY.EDGE_PAIR_MAP].tolist() == [0, 0, 1]
    assert meta[KEY.EDGE_PAIR_REVERSE].tolist() == [False, True, False]
    assert meta[KEY.PAIR_EDGE_FORWARD_INDEX].tolist() == [0, 2]
    assert meta[KEY.PAIR_EDGE_BACKWARD_INDEX].tolist() == [1, 2]
    assert meta[KEY.PAIR_EDGE_HAS_REVERSE].tolist() == [True, False]
    assert torch.allclose(
        meta[KEY.PAIR_EDGE_VEC],
        edge_vec.index_select(0, meta[KEY.PAIR_EDGE_FORWARD_INDEX]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda not available')
def test_pair_metadata_builder_matches_cpu_on_cuda():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.int64)
    edge_vec = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cell_shift = torch.zeros((3, 3), dtype=torch.float32)

    cpu_meta = pair_runtime.build_pair_metadata(
        edge_index,
        edge_vec,
        cell_shift=cell_shift,
    )
    gpu_meta = pair_runtime.build_pair_metadata(
        edge_index.cuda(),
        edge_vec.cuda(),
        cell_shift=cell_shift.cuda(),
    )

    for key in (
        KEY.EDGE_PAIR_MAP,
        KEY.EDGE_PAIR_REVERSE,
        KEY.PAIR_EDGE_FORWARD_INDEX,
        KEY.PAIR_EDGE_BACKWARD_INDEX,
        KEY.PAIR_EDGE_HAS_REVERSE,
        KEY.PAIR_EDGE_VEC,
    ):
        assert torch.equal(cpu_meta[key].cpu(), gpu_meta[key].cpu())


def test_pair_metadata_batches_with_offsets():
    cfg = _pair_cfg()
    g1 = pair_runtime.ensure_pair_metadata_graph(_manual_graph(), cfg)
    g2 = pair_runtime.ensure_pair_metadata_graph(_manual_graph(), cfg)
    batch = Collater([g1, g2])([g1, g2])

    assert batch[KEY.EDGE_PAIR_MAP].tolist() == [0, 0, 1, 2, 2, 3]
    assert batch[KEY.PAIR_EDGE_FORWARD_INDEX].tolist() == [0, 2, 3, 5]
    assert batch[KEY.PAIR_EDGE_BACKWARD_INDEX].tolist() == [1, 2, 4, 5]
    assert batch[KEY.PAIR_EDGE_HAS_REVERSE].tolist() == [True, False, True, False]
    assert batch[KEY.PAIR_EDGE_VEC].shape == (4, 3)


def test_pair_execution_matches_baseline_cpu():
    atoms = bulk('NaCl', 'rocksalt', a=4.0)
    cfg = _model_cfg()
    cfg[KEY.PAIR_EXECUTION_CONFIG] = _baseline_cfg()
    pair_cfg = deepcopy(cfg)
    pair_cfg[KEY.PAIR_EXECUTION_CONFIG] = _pair_cfg()

    torch.manual_seed(1)
    baseline = build_E3_equivariant_model(cfg)
    torch.manual_seed(1)
    pair_model = build_E3_equivariant_model(pair_cfg)
    baseline.eval()
    pair_model.eval()
    baseline.set_is_batch_data(False)
    pair_model.set_is_batch_data(False)

    graph = AtomGraphData.from_numpy_dict(
        dl.unlabeled_atoms_to_graph(atoms, cfg['cutoff'], with_shift=True)
    )
    graph, _ = pair_runtime.prepare_pair_metadata(graph, pair_cfg[KEY.PAIR_EXECUTION_CONFIG])

    out_base = baseline(graph.clone())
    out_pair = pair_model(graph.clone())
    for key in (
        KEY.PRED_TOTAL_ENERGY,
        KEY.ATOMIC_ENERGY,
        KEY.PRED_FORCE,
        KEY.PRED_STRESS,
    ):
        assert torch.allclose(out_base[key], out_pair[key], atol=1e-6, rtol=1e-6)


def test_pair_cli_flags_and_overrides():
    inf_parser = argparse.ArgumentParser()
    add_inference_args(inf_parser)
    inf_args = inf_parser.parse_args(
        [
            'checkpoint.pth',
            'target.xyz',
            '--enable_pair_execution',
            '--pair_execution_policy',
            'geometry_only',
            '--disable_topology_cache',
        ]
    )
    overrides = pair_runtime.pair_execution_overrides_from_args(inf_args)
    assert overrides == {
        'enable_pair_execution': True,
        'pair_execution_policy': 'geometry_only',
        'disable_topology_cache': True,
    }

    get_model_parser = argparse.ArgumentParser()
    add_get_model_args(get_model_parser)
    get_model_args = get_model_parser.parse_args(
        ['checkpoint.pth', '--enable_pair_execution', '--pair_execution_policy', 'full']
    )
    get_model_overrides = pair_runtime.pair_execution_overrides_from_args(
        get_model_args
    )
    assert get_model_overrides == {
        'enable_pair_execution': True,
        'pair_execution_policy': 'full',
        'disable_topology_cache': None,
    }


def test_calculator_torchscript_extra_files_pair_metadata(monkeypatch):
    loaded = {}

    class DummyScript:
        def to(self, device):
            loaded['device'] = device
            return self

        def eval(self):
            loaded['eval'] = True
            return self

    def fake_load(model_path, _extra_files, map_location=None):
        _extra_files['chemical_symbols_to_index'] = b'Na Cl'
        _extra_files['cutoff'] = b'4.5'
        _extra_files['num_species'] = b'2'
        _extra_files['model_type'] = b'E3_equivariant_model'
        _extra_files['version'] = b'0.0.0'
        _extra_files['dtype'] = b'single'
        _extra_files['time'] = b'2026-03-27'
        _extra_files['pair_execution'] = b'yes'
        _extra_files['pair_execution_policy'] = b'full'
        _extra_files['topology_cache'] = b'no'
        _extra_files['distributed_schedule'] = b'pair'
        loaded['path'] = model_path
        return DummyScript()

    monkeypatch.setattr('sevenn.calculator.torch.jit.load', fake_load)
    calc = SevenNetCalculator(
        'dummy.pt',
        file_type='torchscript',
        enable_pair_execution=True,
        pair_execution_policy='full',
        disable_topology_cache=True,
    )

    assert loaded['path'] == 'dummy.pt'
    assert loaded['eval'] is True
    assert calc.pair_execution_config['resolved_policy'] == 'full'
    assert calc.pair_execution_config['use_topology_cache'] is False


def test_deploy_extra_files_pair_metadata(monkeypatch, tmp_path):
    cfg = _model_cfg()
    cfg[KEY.PAIR_EXECUTION_CONFIG] = _baseline_cfg()
    model = build_E3_equivariant_model(cfg)
    fake_cp = SimpleNamespace(
        config=deepcopy(cfg),
        build_model=lambda **kwargs: model,
    )
    captured = {}

    monkeypatch.setattr('sevenn.scripts.deploy.load_checkpoint', lambda path: fake_cp)
    monkeypatch.setattr(
        'sevenn.scripts.deploy.e3nn.util.jit.script',
        lambda module: module,
    )
    monkeypatch.setattr('sevenn.scripts.deploy.torch.jit.freeze', lambda module: module)

    def fake_save(scripted, fname, _extra_files=None):
        captured['fname'] = fname
        captured['extra_files'] = dict(_extra_files or {})

    monkeypatch.setattr('sevenn.scripts.deploy.torch.jit.save', fake_save)

    out = tmp_path / 'deployed_serial'
    deploy_mod.deploy(
        'unused.pth',
        str(out),
        enable_pair_execution=True,
        pair_execution_policy='full',
        disable_topology_cache=True,
    )

    assert captured['fname'].endswith('.pt')
    assert captured['extra_files']['pair_execution'] == 'yes'
    assert captured['extra_files']['pair_execution_policy'] == 'full'
    assert captured['extra_files']['topology_cache'] == 'no'
    assert captured['extra_files']['distributed_schedule'] == 'auto'


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
def test_checkpoint_backend_override_updates_pair_policy(tmp_path):
    cfg = _flash_checkpoint_cfg()
    cfg[KEY.PAIR_EXECUTION_CONFIG] = _baseline_cfg()
    cfg['version'] = '0.0.0'
    model = build_E3_equivariant_model(cfg)
    cp_path = tmp_path / 'flash_pair_cp.pth'
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, cp_path)

    cp = load_checkpoint(str(cp_path))
    model = cp.build_model(enable_flash=True, enable_pair_execution=True)
    assert model.pair_execution_config['resolved_policy'] == 'geometry_only'


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
def test_calculator_checkpoint_override_updates_pair_policy(tmp_path):
    cfg = _flash_checkpoint_cfg()
    cfg[KEY.PAIR_EXECUTION_CONFIG] = _baseline_cfg()
    cfg['version'] = '0.0.0'
    model = build_E3_equivariant_model(cfg)
    cp_path = tmp_path / 'flash_pair_calc_cp.pth'
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, cp_path)

    calc = SevenNetCalculator(
        str(cp_path),
        enable_flash=True,
        enable_pair_execution=True,
    )
    assert calc.pair_execution_config['resolved_policy'] == 'geometry_only'
