from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import sevenn._keys as KEY

_PAIR_SIGNATURE_INTS = 2


def normalize_pair_execution_config(
    config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    cfg = deepcopy(config or {})
    defaults = {
        'use': False,
        'policy': 'auto',
        'fuse_reduction': True,
        'use_topology_cache': True,
        'distributed_schedule': 'auto',
        'backend_policy': 'auto',
    }
    defaults.update(cfg)
    return defaults


def resolve_pair_execution_config(
    config: Dict[str, Any],
    *,
    enable_pair_execution: Optional[bool] = None,
    pair_execution_policy: Optional[str] = None,
    disable_topology_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    pair_cfg = normalize_pair_execution_config(config.get(KEY.PAIR_EXECUTION_CONFIG))
    if enable_pair_execution is not None:
        pair_cfg['use'] = enable_pair_execution
    if pair_execution_policy is not None:
        pair_cfg['policy'] = pair_execution_policy
    if disable_topology_cache:
        pair_cfg['use_topology_cache'] = False

    requested = pair_cfg['policy']
    if not pair_cfg['use']:
        resolved = 'baseline'
    elif requested == 'baseline':
        resolved = 'baseline'
    elif requested == 'geometry_only':
        resolved = 'geometry_only'
    elif requested == 'full':
        resolved = 'full'
    else:
        prefer_common = pair_cfg['backend_policy'] == 'prefer_common'
        uses_accelerator = any(
            [
                config.get(KEY.USE_FLASH_TP, False),
                config.get(KEY.USE_OEQ, False),
                config.get(KEY.CUEQUIVARIANCE_CONFIG, {}).get('use', False),
            ]
        )
        if uses_accelerator and not prefer_common:
            resolved = 'geometry_only'
        elif pair_cfg.get('fuse_reduction', True):
            resolved = 'full'
        else:
            resolved = 'geometry_only'
    pair_cfg['resolved_policy'] = resolved
    return pair_cfg


def pair_execution_enabled(config: Dict[str, Any]) -> bool:
    return resolve_pair_execution_config(config)['resolved_policy'] != 'baseline'


def add_pair_execution_args(parser) -> None:
    parser.add_argument(
        '--enable_pair_execution',
        action='store_true',
        help='enable pair-execution runtime optimizations',
    )
    parser.add_argument(
        '--pair_execution_policy',
        type=str,
        choices=['auto', 'full', 'geometry_only', 'baseline'],
        default=None,
        help='override pair-execution policy',
    )
    parser.add_argument(
        '--disable_topology_cache',
        action='store_true',
        help='disable persistent topology cache for pair execution',
    )


def pair_execution_overrides_from_args(args) -> Dict[str, Any]:
    return {
        'enable_pair_execution': True
        if getattr(args, 'enable_pair_execution', False)
        else None,
        'pair_execution_policy': getattr(args, 'pair_execution_policy', None),
        'disable_topology_cache': True
        if getattr(args, 'disable_topology_cache', False)
        else None,
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    arr = tensor.detach().cpu().contiguous().numpy()
    return arr.tobytes()


def build_topology_signature_tensor(
    edge_index: torch.Tensor,
    cell_shift: Optional[torch.Tensor] = None,
    *,
    nlocal: Optional[int] = None,
    num_atoms: Optional[int] = None,
) -> torch.Tensor:
    hasher = hashlib.blake2b(digest_size=8 * _PAIR_SIGNATURE_INTS)
    hasher.update(np.asarray(edge_index.shape, dtype=np.int64).tobytes())
    hasher.update(_tensor_bytes(edge_index.to(torch.int64)))
    if cell_shift is not None:
        hasher.update(np.asarray(cell_shift.shape, dtype=np.int64).tobytes())
        hasher.update(_tensor_bytes(torch.round(cell_shift).to(torch.int64)))
    if nlocal is not None:
        hasher.update(np.asarray([nlocal], dtype=np.int64).tobytes())
    if num_atoms is not None:
        hasher.update(np.asarray([num_atoms], dtype=np.int64).tobytes())
    return torch.from_numpy(np.frombuffer(hasher.digest(), dtype=np.int64).copy())


def _shift_tuple(cell_shift: Optional[torch.Tensor], index: int) -> Optional[Tuple[int, ...]]:
    if cell_shift is None:
        return None
    shift = torch.round(cell_shift[index]).to(torch.int64).detach().cpu().tolist()
    return tuple(int(v) for v in shift)


def _vec_tuple(edge_vec: torch.Tensor, index: int) -> Tuple[float, float, float]:
    vec = edge_vec[index].detach().cpu().tolist()
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _make_edge_key(
    dst: int,
    src: int,
    *,
    cell_shift: Optional[Tuple[int, ...]] = None,
    edge_vec: Optional[Tuple[float, float, float]] = None,
):
    if cell_shift is not None:
        return ('shift', dst, src, *cell_shift)
    assert edge_vec is not None
    return ('vec', dst, src, edge_vec[0], edge_vec[1], edge_vec[2])


def _reverse_key(key):
    if key[0] == 'shift':
        _, dst, src, sx, sy, sz = key
        return ('shift', src, dst, -sx, -sy, -sz)
    _, dst, src, vx, vy, vz = key
    return ('vec', src, dst, -vx, -vy, -vz)


def build_pair_metadata(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    *,
    cell_shift: Optional[torch.Tensor] = None,
    nlocal: Optional[int] = None,
    num_atoms: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    edge_index_cpu = edge_index.detach().cpu().to(torch.int64)
    edge_vec_cpu = edge_vec.detach().cpu().to(torch.float32)
    cell_shift_cpu = (
        cell_shift.detach().cpu().to(torch.float32) if cell_shift is not None else None
    )
    num_edges = int(edge_index_cpu.shape[1])

    keys = []
    reverse_lookup = {}
    for edge_i in range(num_edges):
        dst = int(edge_index_cpu[0, edge_i].item())
        src = int(edge_index_cpu[1, edge_i].item())
        key = _make_edge_key(
            dst,
            src,
            cell_shift=_shift_tuple(cell_shift_cpu, edge_i),
            edge_vec=None if cell_shift_cpu is not None else _vec_tuple(edge_vec_cpu, edge_i),
        )
        keys.append(key)
        reverse_lookup[key] = edge_i

    pair_map = [-1] * num_edges
    pair_reverse = [False] * num_edges
    pair_forward_index = []
    pair_backward_index = []
    pair_has_reverse = []

    for edge_i, key in enumerate(keys):
        if pair_map[edge_i] != -1:
            continue

        pair_i = len(pair_forward_index)
        pair_map[edge_i] = pair_i
        pair_reverse[edge_i] = False
        pair_forward_index.append(edge_i)

        reverse_edge_i = reverse_lookup.get(_reverse_key(key))
        if reverse_edge_i is not None and reverse_edge_i != edge_i:
            if pair_map[reverse_edge_i] == -1:
                pair_map[reverse_edge_i] = pair_i
                pair_reverse[reverse_edge_i] = True
            if pair_map[reverse_edge_i] == pair_i:
                pair_backward_index.append(reverse_edge_i)
                pair_has_reverse.append(True)
                continue

        pair_backward_index.append(edge_i)
        pair_has_reverse.append(False)

    device = edge_index.device
    meta = {
        KEY.EDGE_PAIR_MAP: torch.tensor(pair_map, dtype=torch.int64, device=device),
        KEY.EDGE_PAIR_REVERSE: torch.tensor(
            pair_reverse, dtype=torch.bool, device=device
        ),
        KEY.PAIR_EDGE_FORWARD_INDEX: torch.tensor(
            pair_forward_index, dtype=torch.int64, device=device
        ),
        KEY.PAIR_EDGE_BACKWARD_INDEX: torch.tensor(
            pair_backward_index, dtype=torch.int64, device=device
        ),
        KEY.PAIR_EDGE_HAS_REVERSE: torch.tensor(
            pair_has_reverse, dtype=torch.bool, device=device
        ),
        KEY.PAIR_TOPOLOGY_SIGNATURE: build_topology_signature_tensor(
            edge_index_cpu,
            cell_shift_cpu,
            nlocal=nlocal,
            num_atoms=num_atoms,
        ).to(device),
    }
    meta[KEY.PAIR_EDGE_VEC] = edge_vec.index_select(0, meta[KEY.PAIR_EDGE_FORWARD_INDEX])
    return meta


def _copy_cached_pair_metadata(
    data: Dict[str, torch.Tensor],
    cached: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    for key in (
        KEY.EDGE_PAIR_MAP,
        KEY.EDGE_PAIR_REVERSE,
        KEY.PAIR_EDGE_FORWARD_INDEX,
        KEY.PAIR_EDGE_BACKWARD_INDEX,
        KEY.PAIR_EDGE_HAS_REVERSE,
        KEY.PAIR_TOPOLOGY_SIGNATURE,
    ):
        data[key] = cached[key].to(data[KEY.EDGE_IDX].device)
    data[KEY.PAIR_EDGE_VEC] = data[KEY.EDGE_VEC].index_select(
        0, data[KEY.PAIR_EDGE_FORWARD_INDEX]
    )
    return data


def prepare_pair_metadata(
    data: Dict[str, torch.Tensor],
    pair_cfg: Optional[Dict[str, Any]],
    *,
    cache_state: Optional[Dict[str, Any]] = None,
    nlocal: Optional[int] = None,
    num_atoms: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    pair_cfg = normalize_pair_execution_config(pair_cfg)
    if not pair_cfg.get('use', False):
        return data, cache_state
    if pair_cfg.get('resolved_policy', pair_cfg.get('policy')) == 'baseline':
        return data, cache_state
    if (
        KEY.EDGE_PAIR_MAP in data
        and KEY.EDGE_PAIR_REVERSE in data
        and KEY.PAIR_EDGE_VEC in data
        and KEY.PAIR_EDGE_FORWARD_INDEX in data
        and KEY.PAIR_EDGE_BACKWARD_INDEX in data
    ):
        return data, cache_state

    cell_shift = data.get(KEY.CELL_SHIFT)
    signature = build_topology_signature_tensor(
        data[KEY.EDGE_IDX],
        cell_shift,
        nlocal=nlocal,
        num_atoms=num_atoms,
    ).cpu()
    if (
        cache_state is not None
        and pair_cfg.get('use_topology_cache', True)
        and KEY.PAIR_TOPOLOGY_SIGNATURE in cache_state
        and torch.equal(cache_state[KEY.PAIR_TOPOLOGY_SIGNATURE], signature)
    ):
        return _copy_cached_pair_metadata(data, cache_state), cache_state

    meta = build_pair_metadata(
        data[KEY.EDGE_IDX],
        data[KEY.EDGE_VEC],
        cell_shift=cell_shift,
        nlocal=nlocal,
        num_atoms=num_atoms,
    )
    data.update(meta)

    if cache_state is not None and pair_cfg.get('use_topology_cache', True):
        cache_state.clear()
        cache_state.update(
            {
                KEY.EDGE_PAIR_MAP: data[KEY.EDGE_PAIR_MAP].detach().cpu(),
                KEY.EDGE_PAIR_REVERSE: data[KEY.EDGE_PAIR_REVERSE].detach().cpu(),
                KEY.PAIR_EDGE_FORWARD_INDEX: data[
                    KEY.PAIR_EDGE_FORWARD_INDEX
                ].detach().cpu(),
                KEY.PAIR_EDGE_BACKWARD_INDEX: data[
                    KEY.PAIR_EDGE_BACKWARD_INDEX
                ].detach().cpu(),
                KEY.PAIR_EDGE_HAS_REVERSE: data[
                    KEY.PAIR_EDGE_HAS_REVERSE
                ].detach().cpu(),
                KEY.PAIR_TOPOLOGY_SIGNATURE: signature,
            }
        )
    return data, cache_state


def ensure_pair_metadata_graph(
    graph: Dict[str, torch.Tensor], pair_cfg: Optional[Dict[str, Any]]
):
    graph, _ = prepare_pair_metadata(graph, pair_cfg)
    return graph
