from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import tarfile
import tempfile
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import ase.io
import h5py
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from ase import Atoms

from local_pair_size_profile import (
    CASE_BASELINE,
    CASE_PAIR,
    benchmark_sample,
    build_graph_and_pair_features,
    profile_sample,
)
from sevenn.calculator import SevenNetCalculator


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets"
RAW_ROOT = DATASET_ROOT / "raw"
INVENTORY = DATASET_ROOT / "inventory.csv"


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    category: str
    source_kind: str
    local_path: Path
    status: str


@dataclass(frozen=True)
class SupportedDataset:
    name: str
    loader: str
    modal: str
    root: Path
    size_column: str
    source_kind: str


def _load_inventory() -> list[DatasetRecord]:
    rows = list(csv.DictReader(INVENTORY.open()))
    return [
        DatasetRecord(
            name=row["name"],
            category=row["category"],
            source_kind=row["source_kind"],
            local_path=(
                Path(row["local_path"])
                if Path(row["local_path"]).is_absolute()
                else (REPO_ROOT / Path(row["local_path"]))
            ),
            status=row["status"],
        )
        for row in rows
    ]


def _decode_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            return ast.literal_eval(stripped)
    return value


def _as_numeric_array(value: Any, *, dtype=np.float64) -> np.ndarray:
    value = _decode_value(value)
    arr = np.asarray(value)
    if arr.dtype != object:
        return np.asarray(arr, dtype=dtype)
    return np.asarray([np.asarray(v, dtype=dtype) for v in value], dtype=dtype)


def _as_bool_array(value: Any) -> np.ndarray:
    value = _decode_value(value)
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray([bool(v) for v in value], dtype=bool)
    return np.asarray(arr, dtype=bool)


def _zero_cell() -> np.ndarray:
    return np.zeros((3, 3), dtype=np.float64)


def _false_pbc() -> np.ndarray:
    return np.zeros(3, dtype=bool)


def _modal_for_dataset(name: str) -> str | None:
    if name == "qm9_hf":
        return "qcml"
    if name.startswith("md22_") or name in {"ani1x", "ani1ccx", "spice_2023", "rmd17", "iso17"}:
        return "spice"
    if name.startswith("omat24_") or name.startswith("salex_"):
        return "omat24"
    if name == "mptrj":
        return "mpa"
    if name.startswith("omol25_"):
        return "omol25_low"
    if name.startswith("oc20_"):
        return "oc20"
    if name.startswith("oc22_"):
        return "oc22"
    if name == "wbm_initial":
        return "matpes_pbe"
    if name == "phonondb_pbe":
        return "matpes_r2scan"
    return None


def _first_parquet_schema(root: Path) -> tuple[Path, list[str]] | None:
    for path in sorted(root.rglob("*.parquet")):
        try:
            return path, pq.ParquetFile(path).read_row_group(0).schema.names
        except Exception:
            continue
    return None


def _candidate_parquet_paths(root: Path, required_columns: Iterable[str] | None = None) -> list[Path]:
    paths = sorted(root.rglob("*.parquet"))
    usable: list[Path] = []
    required = set(required_columns or [])
    for path in paths:
        try:
            names = set(pq.ParquetFile(path).read_row_group(0).schema.names)
        except Exception:
            continue
        if required and not required.issubset(names):
            continue
        usable.append(path)
    return usable


def _detect_supported_dataset(record: DatasetRecord) -> tuple[SupportedDataset | None, str]:
    if record.status == "gated":
        return None, "gated dataset"
    if not record.local_path.exists():
        return None, "missing on disk"

    if record.source_kind == "url":
        modal = _modal_for_dataset(record.name)
        if modal is None:
            return None, "no modal mapping"
        path = record.local_path
        if "extxyz" in path.name:
            return (
                SupportedDataset(
                    name=record.name,
                    loader="extxyz_file",
                    modal=modal,
                    root=path,
                    size_column="natoms",
                    source_kind=record.source_kind,
                ),
                "",
            )
        if path.suffixes[-2:] == [".tar", ".gz"]:
            return (
                SupportedDataset(
                    name=record.name,
                    loader="tar_aselmdb",
                    modal=modal,
                    root=path,
                    size_column="natoms",
                    source_kind=record.source_kind,
                ),
                "",
            )
        return None, "unsupported url format"

    if record.source_kind != "hf":
        return None, f"unsupported source_kind={record.source_kind}"

    if record.name == "ani1ccx":
        modal = _modal_for_dataset(record.name)
        if modal is None:
            return None, "no modal mapping"
        tarballs = sorted(record.local_path.rglob("*.tar.gz"))
        if tarballs:
            return (
                SupportedDataset(
                    name=record.name,
                    loader="ani1ccx_h5",
                    modal=modal,
                    root=record.local_path,
                    size_column="natoms",
                    source_kind=record.source_kind,
                ),
                "",
            )

    schema_info = _first_parquet_schema(record.local_path)
    if schema_info is None:
        return None, "no parquet files found"
    _, cols = schema_info
    colset = set(cols)

    modal = _modal_for_dataset(record.name)

    if {"atomic_numbers", "positions", "cell", "pbc", "nsites"}.issubset(colset):
        if modal is None:
            return None, "no modal mapping"
        return (
            SupportedDataset(
                name=record.name,
                loader="colabfit",
                modal=modal,
                root=record.local_path,
                size_column="nsites",
                source_kind=record.source_kind,
            ),
            "",
        )
    if {"numbers", "positions", "cell", "pbc", "num_atoms"}.issubset(colset):
        if modal is None:
            return None, "no modal mapping"
        return (
            SupportedDataset(
                name=record.name,
                loader="nimashoghi",
                modal=modal,
                root=record.local_path,
                size_column="num_atoms",
                source_kind=record.source_kind,
            ),
            "",
        )
    if {"atomic_numbers", "pos", "cell", "num_atoms"}.issubset(colset):
        if modal is None:
            return None, "no modal mapping"
        return (
            SupportedDataset(
                name=record.name,
                loader="oc20_like",
                modal=modal,
                root=record.local_path,
                size_column="num_atoms",
                source_kind=record.source_kind,
            ),
            "",
        )
    if {"atomic_numbers", "pos", "natoms"}.issubset(colset):
        if modal is None:
            return None, "no modal mapping"
        return (
            SupportedDataset(
                name=record.name,
                loader="qm9_like",
                modal=modal,
                root=record.local_path,
                size_column="natoms",
                source_kind=record.source_kind,
            ),
            "",
        )
    if {"edge_index", "node_feat", "num_nodes"}.issubset(colset):
        return None, "graph-only parquet; atomic positions unavailable"
    return None, "unrecognized parquet schema"


def _top_candidates_in_parquet(
    path: Path, size_column: str, top_k: int
) -> list[tuple[int, int, int, Path]]:
    parquet = pq.ParquetFile(path)
    heap: list[tuple[int, int, int, Path]] = []
    for row_group_idx in range(parquet.num_row_groups):
        column = parquet.read_row_group(row_group_idx, columns=[size_column]).column(0)
        for row_idx, value in enumerate(column.to_pylist()):
            size = int(value)
            item = (size, row_group_idx, row_idx, path)
            if len(heap) < top_k:
                heap.append(item)
                heap.sort(key=lambda x: x[0])
            elif size > heap[0][0]:
                heap[0] = item
                heap.sort(key=lambda x: x[0])
    return sorted(heap, key=lambda item: item[0], reverse=True)


def _rows_from_parquet(
    path: Path,
    candidates: Sequence[tuple[int, int, int, Path]],
    columns: Sequence[str],
) -> list[pd.Series]:
    parquet = pq.ParquetFile(path)
    grouped: dict[int, list[int]] = defaultdict(list)
    for _, row_group_idx, row_idx, _ in candidates:
        grouped[row_group_idx].append(row_idx)

    rows: list[pd.Series] = []
    for row_group_idx, row_indices in grouped.items():
        table = parquet.read_row_group(row_group_idx, columns=list(columns))
        frame = table.to_pandas()
        for row_idx in row_indices:
            rows.append(frame.iloc[row_idx])
    return rows


def _atoms_from_row(row: pd.Series, loader: str) -> tuple[str, int, Atoms]:
    if loader == "colabfit":
        raw_name = row.get("names")
        if isinstance(raw_name, np.ndarray) and len(raw_name) > 0:
            sample_id = str(raw_name[0])
        else:
            sample_id = str(row.get("configuration_id"))
        natoms = int(row["nsites"])
        atoms = Atoms(
            numbers=np.asarray(_decode_value(row["atomic_numbers"]), dtype=np.int64),
            positions=_as_numeric_array(row["positions"]),
            cell=_as_numeric_array(row["cell"]),
            pbc=_as_bool_array(row["pbc"]),
        )
        return sample_id, natoms, atoms

    if loader == "nimashoghi":
        sample_id = str(row.get("filename") or row.get("mp_id"))
        natoms = int(row["num_atoms"])
        atoms = Atoms(
            numbers=np.asarray(_decode_value(row["numbers"]), dtype=np.int64),
            positions=_as_numeric_array(row["positions"]),
            cell=_as_numeric_array(row["cell"]),
            pbc=_as_bool_array(row["pbc"]),
        )
        return sample_id, natoms, atoms

    if loader == "qm9_like":
        sample_id = str(row.get("id"))
        natoms = int(row["natoms"])
        atoms = Atoms(
            numbers=np.asarray(_decode_value(row["atomic_numbers"]), dtype=np.int64),
            positions=_as_numeric_array(row["pos"]),
            cell=_zero_cell(),
            pbc=_false_pbc(),
        )
        return sample_id, natoms, atoms

    if loader == "oc20_like":
        sample_id = str(row.get("fid") or row.get("sid"))
        natoms = int(row["num_atoms"])
        cell = _as_numeric_array(row["cell"])
        atoms = Atoms(
            numbers=np.asarray(_decode_value(row["atomic_numbers"]), dtype=np.int64),
            positions=_as_numeric_array(row["pos"]),
            cell=cell,
            pbc=np.asarray([True, True, True], dtype=bool) if np.any(cell) else _false_pbc(),
        )
        return sample_id, natoms, atoms

    raise ValueError(f"unsupported loader={loader}")


def _load_topk_parquet_samples(spec: SupportedDataset, top_k: int) -> list[dict[str, Any]]:
    if spec.loader == "colabfit":
        columns = (
            "configuration_id",
            "names",
            "nsites",
            "atomic_numbers",
            "positions",
            "cell",
            "pbc",
        )
    elif spec.loader == "nimashoghi":
        columns = (
            "mp_id",
            "filename",
            "num_atoms",
            "numbers",
            "positions",
            "cell",
            "pbc",
        )
    elif spec.loader == "qm9_like":
        columns = ("id", "natoms", "atomic_numbers", "pos")
    elif spec.loader == "oc20_like":
        columns = ("sid", "fid", "num_atoms", "atomic_numbers", "pos", "cell")
    else:
        raise ValueError(spec.loader)

    paths = _candidate_parquet_paths(spec.root, columns)
    if not paths:
        raise FileNotFoundError(f"no usable parquet files under {spec.root}")

    candidates: list[tuple[int, int, int, Path]] = []
    for path in paths:
        candidates.extend(_top_candidates_in_parquet(path, spec.size_column, top_k))
    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:top_k]

    by_path: dict[Path, list[tuple[int, int, int, Path]]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate[3]].append(candidate)

    selected: list[dict[str, Any]] = []
    for path, path_candidates in by_path.items():
        for row in _rows_from_parquet(path, path_candidates, columns):
            sample_id, natoms, atoms = _atoms_from_row(row, spec.loader)
            selected.append({"sample_id": sample_id, "natoms": natoms, "atoms": atoms})
    return sorted(selected, key=lambda item: item["natoms"], reverse=True)[:top_k]


def _extract_extxyz_from_zip(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    target = path.with_suffix("")
    if target.exists():
        return target
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.endswith(".extxyz")]
        if not members:
            raise ValueError(f"no extxyz in {path}")
        with target.open("wb") as dst:
            for member in members:
                with archive.open(member) as src:
                    dst.write(src.read())
    return target


def _load_topk_extxyz_samples(spec: SupportedDataset, top_k: int) -> list[dict[str, Any]]:
    source = _extract_extxyz_from_zip(spec.root)
    heap: list[tuple[int, int, Atoms]] = []
    for idx, atoms in enumerate(ase.io.iread(source, index=":", format="extxyz")):
        natoms = len(atoms)
        item = (natoms, idx, atoms)
        if len(heap) < top_k:
            heap.append(item)
            heap.sort(key=lambda x: x[0])
        elif natoms > heap[0][0]:
            heap[0] = item
            heap.sort(key=lambda x: x[0])
    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    return [
        {"sample_id": f"{source.stem}:{idx}", "natoms": natoms, "atoms": atoms}
        for natoms, idx, atoms in selected
    ]


def _heap_push_topk(heap: list[tuple[int, str, Atoms]], item: tuple[int, str, Atoms], top_k: int) -> None:
    natoms = item[0]
    if len(heap) < top_k:
        heap.append(item)
        heap.sort(key=lambda x: x[0])
    elif natoms > heap[0][0]:
        heap[0] = item
        heap.sort(key=lambda x: x[0])


def _decode_aselmdb_value(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray__" in value:
        shape, dtype, flat = value["__ndarray__"]
        return np.asarray(flat, dtype=np.dtype(dtype)).reshape(shape)
    if isinstance(value, dict):
        return {key: _decode_aselmdb_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_aselmdb_value(item) for item in value]
    return value


def _atoms_from_aselmdb_payload(payload: dict[str, Any]) -> Atoms:
    payload = _decode_aselmdb_value(payload)
    cell = payload.get("cell", _zero_cell())
    pbc = payload.get("pbc", _false_pbc())
    return Atoms(
        numbers=np.asarray(payload["numbers"], dtype=np.int64),
        positions=np.asarray(payload["positions"], dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=np.asarray(pbc, dtype=bool),
    )


def _best_samples_from_aselmdb_file(path: Path, sample_prefix: str, top_k: int) -> list[dict[str, Any]]:
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    heap: list[tuple[int, str, Atoms]] = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            if not value:
                continue
            try:
                payload = json.loads(zlib.decompress(value))
            except Exception:
                continue
            if not isinstance(payload, dict) or "numbers" not in payload or "positions" not in payload:
                continue
            atoms = _atoms_from_aselmdb_payload(payload)
            sample_id = f"{sample_prefix}:{key.decode('utf-8', errors='ignore')}"
            _heap_push_topk(heap, (len(atoms), sample_id, atoms), top_k)
    env.close()
    return [
        {"sample_id": sample_id, "natoms": natoms, "atoms": atoms}
        for natoms, sample_id, atoms in sorted(heap, key=lambda x: x[0], reverse=True)
    ]


def _load_topk_tar_aselmdb_samples(spec: SupportedDataset, top_k: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with tarfile.open(spec.root, "r:gz") as archive, tempfile.TemporaryDirectory() as tmpdir:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".aselmdb") and not member.name.endswith(".aselmdb-lock")
        ]
        if not members:
            raise FileNotFoundError(f"no .aselmdb files in {spec.root}")
        member_budget = min(len(members), max(top_k * 4, 4))
        for member in sorted(members, key=lambda item: item.size, reverse=True)[:member_budget]:
            archive.extract(member, path=tmpdir)
            extracted = Path(tmpdir) / member.name
            candidates.extend(_best_samples_from_aselmdb_file(extracted, extracted.stem, top_k))
    return sorted(candidates, key=lambda item: item["natoms"], reverse=True)[:top_k]


def _load_topk_ani1ccx_samples(spec: SupportedDataset, top_k: int) -> list[dict[str, Any]]:
    tarballs = sorted(spec.root.rglob("*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(f"no ani1ccx tarball under {spec.root}")
    src = tarballs[0]
    heap: list[tuple[int, str, Atoms]] = []
    with tarfile.open(src, "r:gz") as archive, tempfile.TemporaryDirectory() as tmpdir:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".h5")]
        if not members:
            raise FileNotFoundError(f"no .h5 file inside {src}")
        member = members[0]
        archive.extract(member, path=tmpdir)
        extracted = Path(tmpdir) / member.name
        with h5py.File(extracted, "r") as handle:
            for group_name in handle.keys():
                group = handle[group_name]
                if "species" not in group or "coordinates" not in group:
                    continue
                natoms = int(group["species"].shape[1])
                species = np.asarray(group["species"][0], dtype=np.int64)
                positions = np.asarray(group["coordinates"][0], dtype=np.float64)
                atoms = Atoms(numbers=species, positions=positions, cell=_zero_cell(), pbc=_false_pbc())
                _heap_push_topk(heap, (natoms, f"{group_name}:0", atoms), top_k)
    return [
        {"sample_id": sample_id, "natoms": natoms, "atoms": atoms}
        for natoms, sample_id, atoms in sorted(heap, key=lambda x: x[0], reverse=True)
    ]


def load_topk_samples(spec: SupportedDataset, top_k: int) -> list[dict[str, Any]]:
    if spec.loader == "extxyz_file":
        return _load_topk_extxyz_samples(spec, top_k)
    if spec.loader == "tar_aselmdb":
        return _load_topk_tar_aselmdb_samples(spec, top_k)
    if spec.loader == "ani1ccx_h5":
        return _load_topk_ani1ccx_samples(spec, top_k)
    return _load_topk_parquet_samples(spec, top_k)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_grouped_latency(df: pd.DataFrame, value_column: str, out_path: Path, title: str) -> None:
    pivot = df.pivot(index="dataset", columns="case", values=value_column).sort_index()
    fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.45), 5.5))
    x = np.arange(len(pivot.index))
    width = 0.38
    ax.bar(x - width / 2, pivot["e3nn_baseline"], width, label="e3nn_baseline")
    ax.bar(x + width / 2, pivot["e3nn_pair_full"], width, label="e3nn_pair_full")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("ms")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_speedup(df: pd.DataFrame, out_path: Path) -> None:
    ordered = df.sort_values("steady_speedup_baseline_over_pair", ascending=False)
    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.45), 5.5))
    colors = ["#3a7d44" if v >= 1.0 else "#c0563d" for v in ordered["steady_speedup_baseline_over_pair"]]
    ax.bar(ordered["dataset"], ordered["steady_speedup_baseline_over_pair"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("baseline / pair")
    ax.set_title("Steady-state speedup by dataset")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, feature: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(df[feature], df["steady_speedup_baseline_over_pair"])
    for _, row in df.iterrows():
        ax.annotate(row["dataset"], (row[feature], row["steady_speedup_baseline_over_pair"]), fontsize=7, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("baseline / pair")
    ax.set_title(f"Speedup vs {feature}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_support_status(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["status"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=["#3a7d44", "#c0563d", "#7a7a7a"])
    ax.set_ylabel("datasets")
    ax.set_title("Dataset support status")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_profile_stacked(rows: pd.DataFrame, out_path: Path) -> None:
    stage_cols = [
        "graph_build_ms",
        "pair_metadata_ms",
        "device_transfer_ms",
        "top_input_embedding_ms",
        "top_edge_embedding_ms",
        "top_interaction_other_ms",
        "top_convolution_blocks_ms",
        "top_readout_ms",
        "top_force_output_ms",
    ]
    df = rows[["dataset", "case"] + stage_cols].copy()
    df["label"] = df["dataset"] + ":" + df["case"]
    bottom = np.zeros(len(df), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in stage_cols:
        values = df[col].fillna(0.0).to_numpy(dtype=np.float64)
        ax.bar(df["label"], values, bottom=bottom, label=col)
        bottom += values
    ax.set_ylabel("ms")
    ax.set_title("Top datasets stage breakdown")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _summary_lines(
    dataset_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# All Public Local Pair Benchmark",
        "",
        f"- Total inventory entries: `{len(dataset_rows)}`",
        f"- Supported and benchmarked: `{sum(r['status'] == 'benchmarked' for r in dataset_rows)}`",
        f"- Skipped: `{sum(r['status'] == 'skipped' for r in dataset_rows)}`",
        "",
    ]
    if sample_rows:
        sample_frame = pd.DataFrame(sample_rows)
        agg = (
            sample_frame.groupby("dataset", as_index=False)
            .agg(
                modal=("modal", "first"),
                natoms=("natoms", "max"),
                num_edges=("num_edges", "max"),
                baseline_steady_median_ms=("baseline_steady_median_ms", "median"),
                pair_steady_median_ms=("pair_steady_median_ms", "median"),
                steady_speedup_baseline_over_pair=("steady_speedup_baseline_over_pair", "median"),
                max_abs_force_diff_vs_baseline=("max_abs_force_diff_vs_baseline", "max"),
                abs_energy_diff_vs_baseline=("abs_energy_diff_vs_baseline", "max"),
            )
            .sort_values("steady_speedup_baseline_over_pair", ascending=False)
        )
        best = agg.iloc[0]
        worst = agg.iloc[-1]
        lines.extend(
            [
                f"- Best steady-state speedup: `{best['dataset']}` at `{best['steady_speedup_baseline_over_pair']:.3f}x`.",
                f"- Worst steady-state speedup: `{worst['dataset']}` at `{worst['steady_speedup_baseline_over_pair']:.3f}x`.",
                f"- Worst force delta vs baseline: `{agg['max_abs_force_diff_vs_baseline'].max():.3e}` eV/A.",
                f"- Worst energy delta vs baseline: `{agg['abs_energy_diff_vs_baseline'].max():.3e}` eV.",
                "",
            ]
        )
    if skipped_rows:
        skip_counts = Counter(row["skip_reason"] for row in skipped_rows)
        lines.append("## Skip Reasons")
        lines.append("")
        for reason, count in skip_counts.most_common():
            lines.append(f"- `{reason}`: `{count}` datasets")
        lines.append("")
    if profile_rows:
        lines.extend(
            [
                "## Additional Outputs",
                "",
                "- `metrics/profiles.csv`",
                "- `plots/stage_breakdown_top.png`",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            "- `metrics/datasets.csv`",
            "- `metrics/samples.csv`",
            "- `metrics/aggregated.csv`",
            "- `metrics/skipped.csv`",
            "- `plots/steady_state_latency_all.png`",
            "- `plots/cold_latency_all.png`",
            "- `plots/speedup_by_dataset.png`",
            "- `plots/speedup_vs_natoms_all.png`",
            "- `plots/speedup_vs_num_edges_all.png`",
            "- `plots/support_status.png`",
        ]
    )
    return lines


def _analysis_text(
    sample_frame: pd.DataFrame,
    dataset_frame: pd.DataFrame,
    skipped_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# All Public Local Pair Benchmark Analysis",
        "",
        f"Run date: `{pd.Timestamp.now().isoformat()}`",
        "",
        "## Scope",
        "",
        "- Benchmarked all locally available public datasets that can be converted back to ASE `Atoms` directly from the downloaded cache.",
        "- Compared `e3nn_baseline` vs `e3nn_pair_full` with `7net-omni`.",
        "- Logged unsupported datasets explicitly instead of silently dropping them.",
        "",
    ]
    if not sample_frame.empty:
        corr_nat = float(sample_frame["natoms"].corr(sample_frame["steady_speedup_baseline_over_pair"], method="spearman"))
        corr_edges = float(sample_frame["num_edges"].corr(sample_frame["steady_speedup_baseline_over_pair"], method="spearman"))
        lines.extend(
            [
                "## Main Findings",
                "",
                f"- Spearman(speedup, natoms): `{corr_nat:.3f}`",
                f"- Spearman(speedup, num_edges): `{corr_edges:.3f}`",
                "",
            ]
        )
        agg = (
            sample_frame.groupby("dataset", as_index=False)
            .agg(
                modal=("modal", "first"),
                natoms=("natoms", "max"),
                num_edges=("num_edges", "max"),
                avg_neighbors_directed=("avg_neighbors_directed", "max"),
                steady_speedup_baseline_over_pair=("steady_speedup_baseline_over_pair", "median"),
                baseline_steady_median_ms=("baseline_steady_median_ms", "median"),
                pair_steady_median_ms=("pair_steady_median_ms", "median"),
            )
            .sort_values("steady_speedup_baseline_over_pair", ascending=False)
        )
        lines.extend(["### Best Wins", ""])
        for _, row in agg.head(8).iterrows():
            lines.append(
                f"- `{row['dataset']}`: natoms=`{int(row['natoms'])}`, edges=`{int(row['num_edges'])}`, "
                f"avg_neighbors=`{row['avg_neighbors_directed']:.1f}`, "
                f"speedup=`{row['steady_speedup_baseline_over_pair']:.3f}x`."
            )
        lines.extend(["", "### Largest Losses", ""])
        for _, row in agg.tail(8).sort_values("steady_speedup_baseline_over_pair").iterrows():
            lines.append(
                f"- `{row['dataset']}`: natoms=`{int(row['natoms'])}`, edges=`{int(row['num_edges'])}`, "
                f"avg_neighbors=`{row['avg_neighbors_directed']:.1f}`, "
                f"speedup=`{row['steady_speedup_baseline_over_pair']:.3f}x`."
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Current pair execution reduces geometry/SH/weight work, but TP/scatter rows remain edge-major.",
                "- Therefore benefits should track edge load and repeated geometric work more than raw atom count alone.",
                "- Remaining skips are dominated by graph-only mirrors, missing raw files, or gated access rather than public download availability.",
                "",
            ]
        )
    if skipped_rows:
        lines.extend(["## Skipped Datasets", ""])
        for row in skipped_rows:
            lines.append(f"- `{row['dataset']}`: {row['skip_reason']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile-top-n", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", help="optional subset of dataset names")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    inventory = _load_inventory()
    selected_names = set(args.datasets) if args.datasets else None
    if selected_names is not None:
        inventory = [record for record in inventory if record.name in selected_names]
    dataset_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    supported_specs: list[SupportedDataset] = []
    for record in inventory:
        spec, reason = _detect_supported_dataset(record)
        if spec is None:
            dataset_rows.append(
                {
                    "dataset": record.name,
                    "category": record.category,
                    "status": "skipped",
                    "source_kind": record.source_kind,
                    "modal": "",
                    "loader": "",
                    "skip_reason": reason,
                    "local_path": str(record.local_path),
                }
            )
            skipped_rows.append({"dataset": record.name, "skip_reason": reason})
        else:
            dataset_rows.append(
                {
                    "dataset": record.name,
                    "category": record.category,
                    "status": "supported",
                    "source_kind": record.source_kind,
                    "modal": spec.modal,
                    "loader": spec.loader,
                    "skip_reason": "",
                    "local_path": str(record.local_path),
                }
            )
            supported_specs.append(spec)

    for spec in supported_specs:
        print(f"[dataset] {spec.name} loader={spec.loader} modal={spec.modal}", flush=True)
        try:
            samples = load_topk_samples(spec, top_k=args.top_k)
        except Exception as exc:
            for row in dataset_rows:
                if row["dataset"] == spec.name:
                    row["status"] = "skipped"
                    row["skip_reason"] = f"sample loading failed: {exc}"
            skipped_rows.append({"dataset": spec.name, "skip_reason": f"sample loading failed: {exc}"})
            print(f"[skip] {spec.name}: {exc}", flush=True)
            continue

        if not samples:
            for row in dataset_rows:
                if row["dataset"] == spec.name:
                    row["status"] = "skipped"
                    row["skip_reason"] = "no samples selected"
            skipped_rows.append({"dataset": spec.name, "skip_reason": "no samples selected"})
            print(f"[skip] {spec.name}: no samples", flush=True)
            continue

        feature_calc = SevenNetCalculator(
            model="7net-omni",
            modal=spec.modal,
            device="cpu",
            enable_flash=False,
            enable_pair_execution=True,
            pair_execution_policy="full",
        )
        cutoff = feature_calc.cutoff
        del feature_calc
        gc.collect()

        for row in dataset_rows:
            if row["dataset"] == spec.name:
                row["status"] = "benchmarked"
                row["num_samples"] = len(samples)
                row["max_atoms"] = max(sample["natoms"] for sample in samples)

        for sample in samples:
            print(f"[sample] {spec.name} :: {sample['sample_id']} :: natoms={sample['natoms']}", flush=True)
            try:
                baseline = benchmark_sample(sample["atoms"], modal=spec.modal, case=CASE_BASELINE, repeat=args.repeat)
                pair = benchmark_sample(sample["atoms"], modal=spec.modal, case=CASE_PAIR, repeat=args.repeat)
                feats = build_graph_and_pair_features(
                    sample["atoms"], cutoff=cutoff, pair_enabled=True, policy="full"
                )
                sample_rows.append(
                    {
                        "dataset": spec.name,
                        "modal": spec.modal,
                        "loader": spec.loader,
                        "sample_id": sample["sample_id"],
                        "natoms": sample["natoms"],
                        **feats,
                        "baseline_cold_ms": baseline["cold_ms"],
                        "pair_cold_ms": pair["cold_ms"],
                        "baseline_steady_median_ms": baseline["steady_median_ms"],
                        "pair_steady_median_ms": pair["steady_median_ms"],
                        "baseline_steady_p95_ms": baseline["steady_p95_ms"],
                        "pair_steady_p95_ms": pair["steady_p95_ms"],
                        "steady_speedup_baseline_over_pair": baseline["steady_median_ms"] / pair["steady_median_ms"],
                        "cold_ratio_pair_over_baseline": pair["cold_ms"] / baseline["cold_ms"],
                        "max_abs_force_diff_vs_baseline": float(np.max(np.abs(pair["forces"] - baseline["forces"]))),
                        "abs_energy_diff_vs_baseline": abs(pair["energy"] - baseline["energy"]),
                    }
                )
                print(
                    f"[result] {spec.name}: steady {baseline['steady_median_ms']:.2f} -> {pair['steady_median_ms']:.2f} ms",
                    flush=True,
                )
            except Exception as exc:
                skipped_rows.append({"dataset": spec.name, "skip_reason": f"benchmark failed: {exc}"})
                print(f"[error] {spec.name} / {sample['sample_id']}: {exc}", flush=True)

    sample_frame = pd.DataFrame(sample_rows)
    dataset_frame = pd.DataFrame(dataset_rows)
    profile_rows: list[dict[str, Any]] = []
    if not sample_frame.empty and args.profile_top_n > 0:
        reps = sample_frame.sort_values("natoms", ascending=False)["dataset"].drop_duplicates().head(args.profile_top_n)
        for dataset in reps.tolist():
            subset = sample_frame[sample_frame["dataset"] == dataset]
            if subset.empty:
                continue
            spec = next(spec for spec in supported_specs if spec.name == dataset)
            sample = load_topk_samples(spec, top_k=1)[0]
            for case in (CASE_BASELINE, CASE_PAIR):
                summary, _ = profile_sample(sample["atoms"], modal=spec.modal, case=case)
                profile_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample["sample_id"],
                        "case": case["case"],
                        "natoms": sample["natoms"],
                        **summary,
                    }
                )
    profile_frame = pd.DataFrame(profile_rows)

    _write_csv(metrics_dir / "datasets.csv", dataset_rows)
    _write_csv(metrics_dir / "samples.csv", sample_rows)
    _write_csv(metrics_dir / "skipped.csv", skipped_rows)
    if not sample_frame.empty:
        agg_frame = (
            sample_frame.groupby("dataset", as_index=False)
            .agg(
                modal=("modal", "first"),
                natoms=("natoms", "max"),
                num_edges=("num_edges", "max"),
                avg_neighbors_directed=("avg_neighbors_directed", "max"),
                baseline_cold_ms=("baseline_cold_ms", "median"),
                pair_cold_ms=("pair_cold_ms", "median"),
                baseline_steady_median_ms=("baseline_steady_median_ms", "median"),
                pair_steady_median_ms=("pair_steady_median_ms", "median"),
                steady_speedup_baseline_over_pair=("steady_speedup_baseline_over_pair", "median"),
                max_abs_force_diff_vs_baseline=("max_abs_force_diff_vs_baseline", "max"),
                abs_energy_diff_vs_baseline=("abs_energy_diff_vs_baseline", "max"),
            )
            .sort_values("steady_speedup_baseline_over_pair", ascending=False)
        )
        _write_csv(metrics_dir / "aggregated.csv", agg_frame.to_dict(orient="records"))

        latency_long = pd.concat(
            [
                agg_frame.assign(case="e3nn_baseline", steady_ms=agg_frame["baseline_steady_median_ms"], cold_ms=agg_frame["baseline_cold_ms"]),
                agg_frame.assign(case="e3nn_pair_full", steady_ms=agg_frame["pair_steady_median_ms"], cold_ms=agg_frame["pair_cold_ms"]),
            ],
            ignore_index=True,
        )
        _plot_grouped_latency(latency_long, "steady_ms", plots_dir / "steady_state_latency_all.png", "Steady-state latency across all supported datasets")
        _plot_grouped_latency(latency_long, "cold_ms", plots_dir / "cold_latency_all.png", "Cold latency across all supported datasets")
        _plot_speedup(agg_frame, plots_dir / "speedup_by_dataset.png")
        _plot_scatter(agg_frame, "natoms", plots_dir / "speedup_vs_natoms_all.png")
        _plot_scatter(agg_frame, "num_edges", plots_dir / "speedup_vs_num_edges_all.png")
    else:
        _write_csv(metrics_dir / "aggregated.csv", [])
        agg_frame = pd.DataFrame()

    _plot_support_status(dataset_frame, plots_dir / "support_status.png")
    if not profile_frame.empty:
        _write_csv(metrics_dir / "profiles.csv", profile_rows)
        _plot_profile_stacked(profile_frame, plots_dir / "stage_breakdown_top.png")
    else:
        _write_csv(metrics_dir / "profiles.csv", [])

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(_summary_lines(dataset_rows, sample_rows, skipped_rows, profile_rows)) + "\n")
    analysis_path = output_dir / "analysis.md"
    analysis_path.write_text(_analysis_text(sample_frame, dataset_frame, skipped_rows))

    env = {
        "repo_root": str(REPO_ROOT),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "supported_modalities": ["omat24", "mpa", "omol25_low", "omol25_high", "matpes_pbe", "matpes_r2scan", "mp_r2scan", "oc20", "oc22", "spice", "qcml", "odac23", "pet_mad"],
    }
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
