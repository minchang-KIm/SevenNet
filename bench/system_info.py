from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _run_command(command: list[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            'available': True,
            'returncode': proc.returncode,
            'stdout': proc.stdout.strip(),
            'stderr': proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {'available': False, 'reason': 'not found'}
    except PermissionError:
        return {'available': False, 'reason': 'permission denied'}
    except subprocess.TimeoutExpired:
        return {'available': False, 'reason': 'timeout'}


def _git_info(repo_root: Path) -> Dict[str, Any]:
    git = shutil.which('git')
    if git is None:
        return {'available': False, 'reason': 'git not found'}
    base = [git, '-C', str(repo_root)]
    sha = _run_command(base + ['rev-parse', 'HEAD'])
    branch = _run_command(base + ['rev-parse', '--abbrev-ref', 'HEAD'])
    dirty = _run_command(base + ['status', '--short'])
    return {
        'available': True,
        'sha': sha.get('stdout', ''),
        'branch': branch.get('stdout', ''),
        'dirty': bool(dirty.get('stdout', '')),
    }


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
        'devices': [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info['devices'].append(
                {
                    'index': idx,
                    'name': props.name,
                    'total_memory_gb': round(
                        props.total_memory / (1024**3), 3
                    ),
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor,
                }
            )
    return info


def _env_subset() -> Dict[str, str]:
    keep_prefixes = ('SLURM_', 'OMPI_', 'PMI_')
    keep_exact = {
        'CUDA_VISIBLE_DEVICES',
        'HOSTNAME',
        'SEVENN_LAMMPS_CMD',
        'SEVENN_MPIRUN_CMD',
        'SEVENN_OUTPUT_DIR',
        'SEVENN_BACKENDS',
        'SEVENN_REPEAT',
        'SEVENN_NUM_GPUS',
    }
    subset = {}
    for key, value in os.environ.items():
        if key in keep_exact or key.startswith(keep_prefixes):
            subset[key] = value
    return subset


def collect_system_info(repo_root: Path) -> Dict[str, Any]:
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version,
        'cwd': os.getcwd(),
        'repo_root': str(repo_root),
        'git': _git_info(repo_root),
        'torch': _torch_info(),
        'commands': {
            'nvidia_smi': _run_command(['nvidia-smi', '-L']),
            'mpirun': _run_command(['mpirun', '--version']),
            'srun': _run_command(['srun', '--version']),
            'scontrol': _run_command(['scontrol', '--version']),
        },
        'environment': _env_subset(),
    }


def write_system_info(repo_root: Path, output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = collect_system_info(repo_root)
    output_path.write_text(json.dumps(info, indent=2, sort_keys=True))
    return info
