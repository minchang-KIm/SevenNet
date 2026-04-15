from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from ase import Atoms
from ase.io import iread, write
from ase.calculators.calculator import all_changes

import sevenn._keys as KEY
import sevenn.nn.convolution as conv_mod
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[4]
KCC_ROOT = REPO_ROOT / "docs" / "papers" / "KCC"
OUTPUT_ROOT = KCC_ROOT / "lmax_baseline_sweep"
DATA_ROOT = OUTPUT_ROOT / "data" / "rmd17_azobenzene"
RUN_ROOT = OUTPUT_ROOT / "runs"
METRIC_ROOT = OUTPUT_ROOT / "metrics"
FIG_ROOT = OUTPUT_ROOT / "figures"
REPORT_ROOT = OUTPUT_ROOT / "reports"

SOURCE_PARQUETS = [
    REPO_ROOT / "datasets" / "raw" / "hf" / "rmd17" / "co" / "co_0.parquet",
    REPO_ROOT / "datasets" / "raw" / "hf" / "rmd17" / "co" / "co_1.parquet",
]

TARGET_NAME = "rmd17_azobenzene"
L_VALUES = list(range(1, 9))
RANDOM_SEED = 7
TRAIN_COUNT = 512
VALID_COUNT = 128
TEST_COUNT = 128
TOTAL_COUNT = TRAIN_COUNT + VALID_COUNT + TEST_COUNT

CUTOFF = 5.0
EPOCHS = 15
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENCY_WARMUP = 20
LATENCY_REPEAT = 30
PROFILE_WARMUP = 5
PROFILE_REPEAT = 20

PLOT_COLORS = {
    "energy_rmse": "#0072B2",
    "force_rmse": "#D55E00",
    "energy_mae": "#009E73",
    "force_mae": "#CC79A7",
    "latency": "#4D4D4D",
    "sh": "#E69F00",
    "tp": "#3B4C63",
    "force_output": "#7A5195",
}

WALL_TIME_PATTERN = re.compile(r"Total wall time:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _ensure_dirs() -> None:
    for path in [DATA_ROOT, RUN_ROOT, METRIC_ROOT, FIG_ROOT, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def _sync() -> None:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run(cmd: Sequence[str], cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), check=True, env=env)


def _to_numeric_array(value: Any, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != object:
        return np.asarray(arr, dtype=dtype)
    return np.asarray([np.asarray(v, dtype=dtype) for v in value], dtype=dtype)


def _to_bool_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray([bool(v) for v in value], dtype=bool)
    return np.asarray(arr, dtype=bool)


def _normalize_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(v) for v in arr.tolist()]


def _row_to_atoms(row: Dict[str, Any]) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(row["atomic_numbers"], dtype=np.int64),
        positions=_to_numeric_array(row["positions"], dtype=np.float64),
        cell=_to_numeric_array(row["cell"], dtype=np.float64),
        pbc=_to_bool_array(row["pbc"]),
    )
    atoms.info["y_energy"] = float(row["energy"])
    atoms.arrays["y_force"] = _to_numeric_array(row["atomic_forces"], dtype=np.float64)
    atoms.info["configuration_id"] = str(row["configuration_id"])
    atoms.info["source_name"] = TARGET_NAME
    return atoms


def _iter_matching_rows(limit: int) -> list[Atoms]:
    matches: list[Atoms] = []
    columns = [
        "configuration_id",
        "names",
        "energy",
        "atomic_forces",
        "positions",
        "atomic_numbers",
        "cell",
        "pbc",
    ]
    for parquet_path in SOURCE_PARQUETS:
        parquet = pq.ParquetFile(parquet_path)
        for row_group_idx in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_idx, columns=columns)
            frame = table.to_pandas()
            for _, row in frame.iterrows():
                names = _normalize_names(row["names"])
                if TARGET_NAME not in names:
                    continue
                matches.append(_row_to_atoms(row.to_dict()))
                if len(matches) >= limit:
                    return matches
    return matches


def prepare_dataset_splits() -> Dict[str, Path]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_ROOT / "split_manifest.csv"
    split_paths = {
        "train": DATA_ROOT / "train.extxyz",
        "valid": DATA_ROOT / "valid.extxyz",
        "test": DATA_ROOT / "test.extxyz",
    }
    if manifest_path.exists() and all(path.exists() for path in split_paths.values()):
        return split_paths

    rng = random.Random(RANDOM_SEED)
    atoms_list = _iter_matching_rows(TOTAL_COUNT)
    if len(atoms_list) < TOTAL_COUNT:
        raise RuntimeError(
            f"Only {len(atoms_list)} matching structures found for {TARGET_NAME}, "
            f"need {TOTAL_COUNT}"
        )
    rng.shuffle(atoms_list)
    train_atoms = atoms_list[:TRAIN_COUNT]
    valid_atoms = atoms_list[TRAIN_COUNT : TRAIN_COUNT + VALID_COUNT]
    test_atoms = atoms_list[TRAIN_COUNT + VALID_COUNT : TOTAL_COUNT]

    write(split_paths["train"], train_atoms, format="extxyz")
    write(split_paths["valid"], valid_atoms, format="extxyz")
    write(split_paths["test"], test_atoms, format="extxyz")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "count", "avg_atoms", "min_atoms", "max_atoms", "seed"],
        )
        writer.writeheader()
        for split_name, split_atoms in [
            ("train", train_atoms),
            ("valid", valid_atoms),
            ("test", test_atoms),
        ]:
            natoms = np.asarray([len(at) for at in split_atoms], dtype=np.int64)
            writer.writerow(
                {
                    "split": split_name,
                    "count": len(split_atoms),
                    "avg_atoms": float(natoms.mean()),
                    "min_atoms": int(natoms.min()),
                    "max_atoms": int(natoms.max()),
                    "seed": RANDOM_SEED,
                }
            )
    return split_paths


def _base_config(lmax: int, split_paths: Dict[str, Path]) -> Dict[str, Any]:
    return {
        "model": {
            "chemical_species": ["C", "H", "N"],
            "cutoff": CUTOFF,
            "channel": 16,
            "lmax": lmax,
            "num_convolution_layer": 3,
            "weight_nn_hidden_neurons": [64, 64],
            "radial_basis": {
                "radial_basis_name": "bessel",
                "bessel_basis_num": 8,
            },
            "cutoff_function": {
                "cutoff_function_name": "poly_cut",
                "poly_cut_p_value": 6,
            },
            "act_gate": {"e": "silu", "o": "tanh"},
            "act_scalar": {"e": "silu", "o": "tanh"},
            "is_parity": False,
            "self_connection_type": "nequip",
            "conv_denominator": 20.0,
            "train_denominator": False,
            "train_shift_scale": False,
        },
        "train": {
            "random_seed": RANDOM_SEED,
            "is_train_stress": False,
            "epoch": EPOCHS,
            "optimizer": "adam",
            "optim_param": {"lr": 0.003},
            "scheduler": "exponentiallr",
            "scheduler_param": {"gamma": 0.98},
            "force_loss_weight": 0.1,
            "per_epoch": 1,
            "error_record": [
                ["Energy", "RMSE"],
                ["Force", "RMSE"],
                ["Energy", "MAE"],
                ["Force", "MAE"],
                ["TotalLoss", "None"],
            ],
        },
        "data": {
            "batch_size": BATCH_SIZE,
            "shift": "per_atom_energy_mean",
            "scale": "force_rms",
            "data_format_args": {
                "energy_key": "y_energy",
                "force_key": "y_force",
            },
            "load_trainset_path": [str(split_paths["train"])],
            "load_validset_path": [str(split_paths["valid"])],
            "load_testset_path": [str(split_paths["test"])],
        },
    }


def train_all_lmax(split_paths: Dict[str, Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for lmax in L_VALUES:
        run_dir = RUN_ROOT / f"lmax_{lmax}"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_yaml = run_dir / "input.yaml"
        if not input_yaml.exists():
            with input_yaml.open("w") as f:
                yaml.safe_dump(_base_config(lmax, split_paths), f, sort_keys=False)
        checkpoint = run_dir / "checkpoint_best.pth"
        if not checkpoint.exists():
            cmd = [
                sys.executable,
                "-m",
                "sevenn.main.sevenn",
                "train",
                str(input_yaml),
                "-w",
                str(run_dir),
                "-s",
            ]
            _run(cmd)
        run_dirs.append(run_dir)
    return run_dirs


def _parse_training_wall_time_seconds(run_dir: Path) -> float:
    log_path = run_dir / "log.sevenn"
    if not log_path.exists():
        return math.nan
    for line in reversed(log_path.read_text().splitlines()):
        m = WALL_TIME_PATTERN.search(line)
        if m:
            hours = int(m.group(1))
            minutes = int(m.group(2))
            seconds = float(m.group(3))
            return hours * 3600.0 + minutes * 60.0 + seconds
    return math.nan


def run_test_inference(run_dir: Path, test_path: Path) -> Path:
    out_dir = run_dir / "inference_results"
    if out_dir.exists() and (out_dir / "errors.txt").exists():
        return out_dir
    cmd = [
        sys.executable,
        "-m",
        "sevenn.main.sevenn",
        "inference",
        str(run_dir / "checkpoint_best.pth"),
        str(test_path),
        "--output",
        str(out_dir),
        "--device",
        DEVICE,
        "--batch",
        str(BATCH_SIZE),
        "--kwargs",
        "energy_key=y_energy",
        "force_key=y_force",
    ]
    _run(cmd)
    return out_dir


def _normalize_error_key(raw_key: str) -> str:
    return raw_key.split("(", 1)[0].strip()


def _measure_ms(fn: Callable[[], None], *, warmup: int, repeat: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    ser = pd.Series(times)
    return {
        "mean_ms": float(ser.mean()),
        "std_ms": float(ser.std(ddof=1)),
        "median_ms": float(ser.median()),
        "p95_ms": float(ser.quantile(0.95)),
    }


def measure_step_latency(run_dir: Path, representative_atoms: Atoms) -> Dict[str, float]:
    calc = SevenNetCalculator(model=str(run_dir / "checkpoint_best.pth"), device=DEVICE)

    def one_step() -> None:
        calc.calculate(
            representative_atoms,
            properties=["energy", "forces"],
            system_changes=all_changes,
        )

    result = _measure_ms(one_step, warmup=LATENCY_WARMUP, repeat=LATENCY_REPEAT)
    result["num_edges"] = int(calc.results["num_edges"])
    return result


class StageTimer:
    def __init__(self) -> None:
        self.times_ms: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.times_ms.clear()
        self.calls.clear()

    @contextmanager
    def section(self, key: str) -> Iterable[None]:
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.times_ms[key] += (time.perf_counter() - t0) * 1000.0
            self.calls[key] += 1


def _patch_method(
    obj: Any, attr: str, wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Callable[[], None]:
    original = getattr(obj, attr)
    setattr(obj, attr, wrapper_factory(original))

    def restore() -> None:
        setattr(obj, attr, original)

    return restore


def profile_model_stages(run_dir: Path, representative_atoms: Atoms) -> pd.DataFrame:
    model, _ = model_from_checkpoint(str(run_dir / "checkpoint_best.pth"))
    device = torch.device(DEVICE)
    model.to(device)
    model.set_is_batch_data(False)
    model.eval()

    raw_graph = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(representative_atoms, CUTOFF, with_shift=False)
    )
    timer = StageTimer()
    restore_stack: list[Callable[[], None]] = []

    def timed_call(key: str, fn: Callable[..., Any], *args, **kwargs):
        with timer.section(key):
            return fn(*args, **kwargs)

    edge_module = model._modules["edge_embedding"]
    restore_stack.append(
        _patch_method(
            edge_module,
            "forward",
            lambda _original: lambda data: _edge_forward_profile(
                edge_module,
                data,
                timed_call,
            ),
        )
    )

    for name, module in model._modules.items():
        if not name.endswith("_convolution"):
            continue
        restore_stack.append(
            _patch_method(
                module,
                "forward",
                lambda _original, name=name, module=module: lambda data: _conv_forward_profile(
                    name,
                    module,
                    data,
                    timed_call,
                ),
            )
        )

    force_module = model._modules.get("force_output")
    if force_module is not None:
        restore_stack.append(
            _patch_method(
                force_module,
                "forward",
                lambda original: lambda data: _force_forward_profile(
                    original,
                    data,
                    timed_call,
                ),
            )
        )

    rows: list[Dict[str, Any]] = []
    try:
        for rep in range(PROFILE_REPEAT):
            timer.reset()
            data = raw_graph.clone().to(device)
            with timer.section("model_total_ms"):
                _ = model(data)
            for stage, value in timer.times_ms.items():
                rows.append(
                    {
                        "lmax": int(run_dir.name.split("_")[-1]),
                        "repeat": rep,
                        "stage": stage,
                        "time_ms": value,
                    }
                )
    finally:
        for restore in reversed(restore_stack):
            restore()
    return pd.DataFrame(rows)


def _edge_forward_profile(
    module: Any,
    data: AtomGraphData,
    timed_call: Callable[..., Any],
) -> AtomGraphData:
    rvec = data[KEY.EDGE_VEC]
    r = timed_call("edge_length_norm_ms", torch.linalg.norm, rvec, dim=-1)
    data[KEY.EDGE_LENGTH] = r
    basis = timed_call("radial_basis_ms", module.basis_function, r)
    cutoff = timed_call("cutoff_ms", module.cutoff_function, r)
    data[KEY.EDGE_EMBEDDING] = timed_call(
        "radial_combine_ms", torch.mul, basis, cutoff.unsqueeze(-1)
    )
    data[KEY.EDGE_ATTR] = timed_call("spherical_harmonics_ms", module.spherical, rvec)
    return data


def _conv_forward_profile(
    name: str,
    module: Any,
    data: AtomGraphData,
    timed_call: Callable[..., Any],
) -> AtomGraphData:
    x = data[module.key_x]
    if module.is_parallel:
        x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

    edge_index = data[module.key_edge_idx]
    edge_src = edge_index[1]
    edge_dst = edge_index[0]
    weight = timed_call(f"{name}.weight_nn_ms", module.weight_nn, data[module.key_weight_input])
    x_src = timed_call(f"{name}.src_gather_ms", x.index_select, 0, edge_src)
    msg = timed_call(
        f"{name}.message_tp_ms",
        module.convolution,
        x_src,
        data[module.key_filter],
        weight,
    )
    out = timed_call(
        f"{name}.aggregation_ms",
        conv_mod.message_gather,
        x,
        edge_dst,
        msg,
    )
    out = timed_call(f"{name}.denominator_ms", torch.div, out, module.denominator)
    if module.is_parallel:
        out = timed_call(
            f"{name}.parallel_split_ms",
            lambda tensor: torch.tensor_split(tensor, data[KEY.NLOCAL])[0],
            out,
        )
    data[module.key_x] = out
    return data


def _force_forward_profile(
    original: Callable[..., Any],
    data: AtomGraphData,
    timed_call: Callable[..., Any],
) -> AtomGraphData:
    return timed_call("top_force_output_ms", original, data)


def collect_run_metrics(run_dirs: Sequence[Path], split_paths: Dict[str, Path]) -> None:
    test_atoms_all = list(iread(split_paths["test"], index=":", format="extxyz"))
    representative_atoms = max(test_atoms_all, key=len).copy()

    summary_rows: list[Dict[str, Any]] = []
    history_rows: list[Dict[str, Any]] = []
    latency_rows: list[Dict[str, Any]] = []
    stage_rows: list[Dict[str, Any]] = []
    inference_rows: list[Dict[str, Any]] = []

    for run_dir in run_dirs:
        lmax = int(run_dir.name.split("_")[-1])
        inference_dir = run_test_inference(run_dir, split_paths["test"])

        lc = pd.read_csv(run_dir / "lc.csv")
        lc["lmax"] = lmax
        history_rows.extend(lc.to_dict(orient="records"))

        error_dct: Dict[str, float] = {}
        with (inference_dir / "errors.txt").open() as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.strip().split(":", 1)
                error_dct[_normalize_error_key(key.strip())] = float(val.strip())

        latency = measure_step_latency(run_dir, representative_atoms)
        latency_rows.append({"lmax": lmax, **latency})

        stage_df = profile_model_stages(run_dir, representative_atoms)
        stage_rows.extend(stage_df.to_dict(orient="records"))

        run_cfg = yaml.safe_load((run_dir / "input.yaml").read_text())
        model_for_params, _ = model_from_checkpoint(str(run_dir / "checkpoint_best.pth"))
        trainable_params = int(sum(p.numel() for p in model_for_params.parameters() if p.requires_grad))

        valid_force = float(lc["validset_Force_RMSE"].min()) if "validset_Force_RMSE" in lc else math.nan
        valid_energy = float(lc["validset_Energy_RMSE"].min()) if "validset_Energy_RMSE" in lc else math.nan
        best_epoch = int(lc.loc[lc["validset_Force_RMSE"].idxmin(), "epoch"]) if "validset_Force_RMSE" in lc else int(lc["epoch"].iloc[-1])

        summary_rows.append(
            {
                "lmax": lmax,
                "epochs": EPOCHS,
                "train_count": TRAIN_COUNT,
                "valid_count": VALID_COUNT,
                "test_count": TEST_COUNT,
                "channel": run_cfg["model"]["channel"],
                "num_convolution_layer": run_cfg["model"]["num_convolution_layer"],
                "trainable_params": trainable_params,
                "training_wall_time_s": _parse_training_wall_time_seconds(run_dir),
                "best_valid_energy_rmse": valid_energy,
                "best_valid_force_rmse": valid_force,
                "best_epoch_by_valid_force": best_epoch,
                "test_energy_rmse": error_dct.get("Energy_RMSE"),
                "test_force_rmse": error_dct.get("Force_RMSE"),
                "test_energy_mae": error_dct.get("Energy_MAE"),
                "test_force_mae": error_dct.get("Force_MAE"),
                "step_force_mean_ms": latency["mean_ms"],
                "step_force_std_ms": latency["std_ms"],
                "step_force_median_ms": latency["median_ms"],
                "step_force_p95_ms": latency["p95_ms"],
                "num_edges_representative": latency["num_edges"],
            }
        )
        inference_rows.append({"lmax": lmax, **error_dct})

    summary_df = pd.DataFrame(summary_rows).sort_values("lmax")
    history_df = pd.DataFrame(history_rows)
    latency_df = pd.DataFrame(latency_rows).sort_values("lmax")
    stage_df = pd.DataFrame(stage_rows)
    inference_df = pd.DataFrame(inference_rows).sort_values("lmax")

    summary_df.to_csv(METRIC_ROOT / "lmax_sweep_summary.csv", index=False)
    history_df.to_csv(METRIC_ROOT / "lmax_training_history.csv", index=False)
    latency_df.to_csv(METRIC_ROOT / "lmax_latency_summary.csv", index=False)
    stage_df.to_csv(METRIC_ROOT / "lmax_stage_profile_raw.csv", index=False)
    inference_df.to_csv(METRIC_ROOT / "lmax_inference_errors.csv", index=False)

    stage_mean = (
        stage_df.groupby(["lmax", "stage"], as_index=False)
        .agg(mean_ms=("time_ms", "mean"), std_ms=("time_ms", "std"))
        .sort_values(["lmax", "stage"])
    )
    stage_mean.to_csv(METRIC_ROOT / "lmax_stage_profile_summary.csv", index=False)

    _save_figures(summary_df, history_df, stage_mean)
    _write_report(summary_df, stage_mean)


def _save_figures(summary_df: pd.DataFrame, history_df: pd.DataFrame, stage_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        summary_df["lmax"],
        summary_df["test_energy_rmse"],
        marker="o",
        color=PLOT_COLORS["energy_rmse"],
        label="Energy RMSE",
    )
    ax1.set_xlabel("lmax")
    ax1.set_ylabel("Energy RMSE (eV)", color=PLOT_COLORS["energy_rmse"])
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["lmax"],
        summary_df["test_force_rmse"],
        marker="s",
        color=PLOT_COLORS["force_rmse"],
        label="Force RMSE",
    )
    ax2.set_ylabel("Force RMSE (eV/A)", color=PLOT_COLORS["force_rmse"])
    ax1.set_title("Test Accuracy vs lmax (rMD17 azobenzene baseline)")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "lmax_accuracy_rmse.png", dpi=300)
    fig.savefig(FIG_ROOT / "lmax_accuracy_rmse.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary_df["lmax"],
        summary_df["step_force_mean_ms"],
        yerr=summary_df["step_force_std_ms"],
        marker="o",
        color=PLOT_COLORS["latency"],
        capsize=4,
    )
    ax.set_xlabel("lmax")
    ax.set_ylabel("Step latency (ms)")
    ax.set_title("Energy+Force Step Latency vs lmax")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "lmax_step_latency.png", dpi=300)
    fig.savefig(FIG_ROOT / "lmax_step_latency.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary_df["step_force_mean_ms"],
        summary_df["test_force_rmse"],
        marker="o",
        color=PLOT_COLORS["force_rmse"],
    )
    for _, row in summary_df.iterrows():
        ax.annotate(int(row["lmax"]), (row["step_force_mean_ms"], row["test_force_rmse"]))
    ax.set_xlabel("Step latency (ms)")
    ax.set_ylabel("Force RMSE (eV/A)")
    ax.set_title("Accuracy-Latency Frontier")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "lmax_accuracy_latency_frontier.png", dpi=300)
    fig.savefig(FIG_ROOT / "lmax_accuracy_latency_frontier.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary_df["lmax"],
        summary_df["trainable_params"],
        marker="o",
        color="#009E73",
    )
    ax.set_xlabel("lmax")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Model Size vs lmax")
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "lmax_trainable_params.png", dpi=300)
    fig.savefig(FIG_ROOT / "lmax_trainable_params.svg")
    plt.close(fig)

    if summary_df["training_wall_time_s"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            summary_df["lmax"],
            summary_df["training_wall_time_s"] / 60.0,
            marker="o",
            color="#8C564B",
        )
        ax.set_xlabel("lmax")
        ax.set_ylabel("Training wall time (min)")
        ax.set_title("Training Time vs lmax")
        fig.tight_layout()
        fig.savefig(FIG_ROOT / "lmax_training_wall_time.png", dpi=300)
        fig.savefig(FIG_ROOT / "lmax_training_wall_time.svg")
        plt.close(fig)

    if "validset_Force_RMSE" in history_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for lmax in L_VALUES:
            part = history_df[history_df["lmax"] == lmax]
            ax.plot(part["epoch"], part["validset_Force_RMSE"], label=f"l={lmax}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Force RMSE")
        ax.set_title("Validation Force RMSE by Epoch")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_ROOT / "lmax_valid_force_curve.png", dpi=300)
        fig.savefig(FIG_ROOT / "lmax_valid_force_curve.svg")
        plt.close(fig)

    stage_interest = [
        "spherical_harmonics_ms",
        "radial_basis_ms",
        "cutoff_ms",
        "top_force_output_ms",
        "model_total_ms",
    ]
    pivot = (
        stage_df[stage_df["stage"].isin(stage_interest)]
        .pivot(index="lmax", columns="stage", values="mean_ms")
        .fillna(0.0)
        .sort_index()
    )
    if not pivot.empty:
        other = pivot["model_total_ms"].copy()
        for stage in ["spherical_harmonics_ms", "radial_basis_ms", "cutoff_ms", "top_force_output_ms"]:
            if stage in pivot.columns:
                other -= pivot[stage]
        plot_df = pd.DataFrame(
            {
                "spherical_harmonics_ms": pivot.get("spherical_harmonics_ms", 0.0),
                "radial_basis_ms": pivot.get("radial_basis_ms", 0.0),
                "cutoff_ms": pivot.get("cutoff_ms", 0.0),
                "top_force_output_ms": pivot.get("top_force_output_ms", 0.0),
                "other_ms": other.clip(lower=0.0),
            }
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(plot_df))
        color_map = {
            "spherical_harmonics_ms": PLOT_COLORS["sh"],
            "radial_basis_ms": "#56B4E9",
            "cutoff_ms": "#009E73",
            "top_force_output_ms": PLOT_COLORS["force_output"],
            "other_ms": "#B0B0B0",
        }
        for col in plot_df.columns:
            ax.bar(plot_df.index, plot_df[col], bottom=bottom, label=col, color=color_map[col])
            bottom += plot_df[col].to_numpy()
        ax.set_xlabel("lmax")
        ax.set_ylabel("Mean time (ms)")
        ax.set_title("Representative Model Stage Profile vs lmax")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_ROOT / "lmax_stage_profile.png", dpi=300)
        fig.savefig(FIG_ROOT / "lmax_stage_profile.svg")
        plt.close(fig)


def _write_report(summary_df: pd.DataFrame, stage_df: pd.DataFrame) -> None:
    best_force = summary_df.loc[summary_df["test_force_rmse"].idxmin()]
    best_energy = summary_df.loc[summary_df["test_energy_rmse"].idxmin()]
    fastest = summary_df.loc[summary_df["step_force_mean_ms"].idxmin()]
    slowest = summary_df.loc[summary_df["step_force_mean_ms"].idxmax()]
    largest = summary_df.loc[summary_df["lmax"].idxmax()]
    smallest = summary_df.loc[summary_df["lmax"].idxmin()]

    force_gain = (
        (smallest["test_force_rmse"] - best_force["test_force_rmse"])
        / smallest["test_force_rmse"]
        * 100.0
    )
    latency_growth = slowest["step_force_mean_ms"] / fastest["step_force_mean_ms"]
    train_growth = largest["training_wall_time_s"] / smallest["training_wall_time_s"]
    param_growth = largest["trainable_params"] / smallest["trainable_params"]

    stage_pivot = (
        stage_df.pivot(index="lmax", columns="stage", values="mean_ms")
        .sort_index()
        .fillna(0.0)
    )
    sh_small = float(stage_pivot.loc[smallest["lmax"], "spherical_harmonics_ms"])
    sh_large = float(stage_pivot.loc[largest["lmax"], "spherical_harmonics_ms"])
    total_small = float(stage_pivot.loc[smallest["lmax"], "model_total_ms"])
    total_large = float(stage_pivot.loc[largest["lmax"], "model_total_ms"])
    force_small = float(stage_pivot.loc[smallest["lmax"], "top_force_output_ms"])
    force_large = float(stage_pivot.loc[largest["lmax"], "top_force_output_ms"])

    report = f"""# lmax baseline sweep note

대상은 `rMD17 azobenzene` 고정 분할이며, 같은 baseline 구조에서 `lmax=1..8`만 바꿔 학습/평가했다.

## 핵심 질문

- `lmax`는 데이터셋이 자동으로 정하는 값인가?
- `lmax`를 올리면 정확도가 항상 좋아지는가?
- 그 대가로 시간은 얼마나 늘어나는가?

## 실험 설정

- dataset: `{TARGET_NAME}`
- split: train {TRAIN_COUNT}, valid {VALID_COUNT}, test {TEST_COUNT}
- cutoff: {CUTOFF}
- epoch: {EPOCHS}
- batch size: {BATCH_SIZE}
- device: {DEVICE}
- latency repeat: warmup {LATENCY_WARMUP}, repeat {LATENCY_REPEAT}
- profile repeat: warmup {PROFILE_WARMUP}, repeat {PROFILE_REPEAT}

## 대표 결과

- 최저 test force RMSE: lmax={int(best_force['lmax'])}, {best_force['test_force_rmse']:.6f} eV/A
- 최저 test energy RMSE: lmax={int(best_energy['lmax'])}, {best_energy['test_energy_rmse']:.6f} eV
- 최저 step latency: lmax={int(fastest['lmax'])}, {fastest['step_force_mean_ms']:.3f} ms
- 최대 학습 시간: {summary_df['training_wall_time_s'].max() / 60.0:.2f} min

## 숫자로 보는 추세

- `lmax=1 -> 7`에서 test force RMSE는 `{smallest['test_force_rmse']:.6f} -> {best_force['test_force_rmse']:.6f} eV/A`로 줄어들어 약 `{force_gain:.1f}%` 개선됐다.
- 하지만 `lmax=8`에서는 force RMSE가 다시 `{largest['test_force_rmse']:.6f} eV/A`로 올라가, 더 큰 `lmax`가 항상 더 정확한 것은 아니었다.
- step latency는 `lmax=1 -> 8`에서 `{fastest['step_force_mean_ms']:.3f} -> {slowest['step_force_mean_ms']:.3f} ms`로 약 `{latency_growth:.2f}x` 증가했다.
- 학습 시간은 `lmax=1 -> 8`에서 `{smallest['training_wall_time_s']:.2f} -> {largest['training_wall_time_s']:.2f} s`로 약 `{train_growth:.2f}x` 증가했다.
- trainable parameter는 `lmax=1 -> 8`에서 `{int(smallest['trainable_params'])} -> {int(largest['trainable_params'])}`로 약 `{param_growth:.2f}x` 증가했다.

## 해석 메모

- `lmax`는 데이터셋이 정해주는 값이 아니라 모델 설계자가 고르는 하이퍼파라미터다.
- 다만 어떤 데이터셋에서는 높은 각도 표현이 필요해 더 큰 `lmax`가 유리할 수 있고, 어떤 데이터셋에서는 그렇지 않을 수 있다.
- 따라서 실제로는 `정확도-시간 절충`을 실험으로 확인해야 한다.
- 이번 고정 조건 실험에서는 `rMD17 azobenzene`에 대해 `lmax=7`이 가장 좋은 힘 정확도를 보였고, `lmax=8`은 더 비싸지만 오히려 성능이 약간 나빠졌다.
- 따라서 `해상도가 높으면 정확도도 무조건 높아진다`고 쓰면 안 되고, `적절한 lmax가 존재한다`고 해석하는 것이 맞다.
- 대표 구조 기준에서 SH 절대 시간은 `lmax=1 -> 8`에서 `{sh_small:.4f} -> {sh_large:.4f} ms`로 크게 늘었다.
- 그러나 SH 비중은 전체 model time의 `{(sh_small/total_small)*100.0:.2f}% -> {(sh_large/total_large)*100.0:.2f}%` 수준에 머물렀다.
- 반면 force output 비중은 `{(force_small/total_small)*100.0:.2f}% -> {(force_large/total_large)*100.0:.2f}%`로 더 컸다.
- 즉 높은 `lmax`의 비용은 SH 하나만이 아니라 TP와 force backward가 함께 커지는 구조로 이해해야 한다.

## 저장된 파일

- summary: `metrics/lmax_sweep_summary.csv`
- training history: `metrics/lmax_training_history.csv`
- latency: `metrics/lmax_latency_summary.csv`
- stage profile: `metrics/lmax_stage_profile_summary.csv`
- figures: `figures/*.png`
"""
    (REPORT_ROOT / "lmax_baseline_sweep_report.md").write_text(report)


def main() -> None:
    _ensure_dirs()
    split_paths = prepare_dataset_splits()
    run_dirs = train_all_lmax(split_paths)
    collect_run_metrics(run_dirs, split_paths)


if __name__ == "__main__":
    main()
