from __future__ import annotations

import csv
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass

import ase.calculators.lammps
import ase.io.lammpsdata
from ase.build import bulk, surface


ROOT = pathlib.Path("/home/wise/minchang/DenseMLIP/SevenNet")
OUT_ROOT = ROOT / "docs/papers/KCC/lammps_pair_metadata_bench"
MODEL_DIR = OUT_ROOT / "models"
METRICS_DIR = OUT_ROOT / "metrics"
REPORTS_DIR = OUT_ROOT / "reports"
LAMMPS_BIN = pathlib.Path("/home/wise/minchang/DenseMLIP/lammps_sevenn/build/lmp")
CONDA_ROOT = pathlib.Path("/home/wise/miniconda3")
TORCH_LIB_DIR = (
    CONDA_ROOT / "lib/python3.12/site-packages/torch/lib"
)

PROFILE_RE = re.compile(
    r"\[sevenn_lammps_profile\]\s+mode=(?P<mode>\S+)"
    r"(?:\s+rank=(?P<rank>\d+))?"
    r"(?:\s+builder=(?P<builder>\S+))?"
    r"\s+policy=(?P<policy>\S+)\s+nedges=(?P<nedges>\d+)\s+npairs=(?P<npairs>\d+)"
    r"\s+pair_metadata_total_ms=(?P<pair_metadata_total_ms>[0-9.]+)"
    r"\s+pair_metadata_build_ms=(?P<pair_metadata_build_ms>[0-9.]+)"
    r"\s+model_inference_ms=(?P<model_inference_ms>[0-9.]+)"
    r"\s+compute_total_ms=(?P<compute_total_ms>[0-9.]+)"
)

LMP_TEMPLATE = """\
units            metal
boundary         __BOUNDARY__
read_data        __LMP_STCT__

mass * 1.0

pair_style       e3gnn
pair_coeff       * * __POTENTIAL__ __ELEMENTS__

timestep         0.002
compute pa all pe/atom
thermo           1
fix 1 all nve
thermo_style     custom step tpcpu pe ke vol pxx pyy pzz pxy pxz pyz press temp
dump             mydump all custom 1 __FORCE_DUMP_PATH__ id type element c_pa x y z fx fy fz
dump_modify      mydump sort id element __ELEMENTS__
run 0
"""


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str
    replicate: tuple[int, int, int]


SYSTEMS = [
    SystemSpec("bulk_small", "bulk", (2, 2, 2)),
    SystemSpec("bulk_large", "bulk", (6, 6, 3)),
]


CASES = [
    {
        "case": "baseline",
        "potential": MODEL_DIR / "baseline_serial.pt",
        "legacy_pair_build": False,
    },
    {
        "case": "geometry_only_legacy",
        "potential": MODEL_DIR / "geometry_only_serial.pt",
        "legacy_pair_build": True,
    },
    {
        "case": "geometry_only_upstream",
        "potential": MODEL_DIR / "geometry_only_serial.pt",
        "legacy_pair_build": False,
    },
]

REPEATS = 30


def hfo2_bulk(replicate=(2, 2, 2), a=4.0):
    atoms = bulk("HfO", "rocksalt", a, orthorhombic=True)
    atoms = atoms * replicate
    atoms.rattle(stdev=0.10)
    return atoms


def hf_surface(replicate=(3, 3, 1), layers=4, vacuum=0.5):
    atoms = surface("Al", (1, 0, 0), layers=layers, vacuum=vacuum)
    atoms.set_atomic_numbers([72] * len(atoms))
    atoms = atoms * replicate
    atoms.rattle(stdev=0.10)
    return atoms


def build_system(spec: SystemSpec):
    if spec.kind == "bulk":
        return hfo2_bulk(replicate=spec.replicate)
    if spec.kind == "surface":
        return hf_surface(replicate=spec.replicate)
    raise ValueError(spec.kind)


def write_lammps_inputs(atoms, workdir: pathlib.Path, potential: pathlib.Path):
    pbc = atoms.get_pbc()
    pbc_str = " ".join("p" if x else "f" for x in pbc)
    chem = list(dict.fromkeys(atoms.get_chemical_symbols()))
    prism = ase.calculators.lammps.coordinatetransform.Prism(
        atoms.get_cell(), pbc=pbc
    )
    lmp_stct = workdir / "lammps_structure"
    ase.io.lammpsdata.write_lammps_data(
        lmp_stct, atoms, prismobj=prism, specorder=chem
    )

    force_dump = workdir / "force.dump"
    content = (
        LMP_TEMPLATE.replace("__BOUNDARY__", pbc_str)
        .replace("__LMP_STCT__", str(lmp_stct.resolve()))
        .replace("__POTENTIAL__", str(potential.resolve()))
        .replace("__ELEMENTS__", " ".join(chem))
        .replace("__FORCE_DUMP_PATH__", str(force_dump.resolve()))
    )
    input_path = workdir / "in.lmp"
    input_path.write_text(content)
    return input_path


def run_once(system: SystemSpec, case: dict, repeat_idx: int):
    atoms = build_system(system)
    with tempfile.TemporaryDirectory(prefix="kcc_lammps_pair_bench_") as tmp:
        workdir = pathlib.Path(tmp)
        input_path = write_lammps_inputs(atoms, workdir, case["potential"])
        env = os.environ.copy()
        env["SEVENN_LAMMPS_PROFILE"] = "1"
        ld_parts = [
            str(CONDA_ROOT / "lib"),
            str(TORCH_LIB_DIR),
        ]
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        if case["legacy_pair_build"]:
            env["SEVENN_LAMMPS_LEGACY_PAIR_BUILD"] = "1"
        else:
            env.pop("SEVENN_LAMMPS_LEGACY_PAIR_BUILD", None)

        cmd = [str(LAMMPS_BIN), "-in", str(input_path), "-log", str(workdir / "log.lammps")]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        text = (res.stdout or "") + "\n" + (res.stderr or "")
        match = PROFILE_RE.search(text)
        if not match:
            raise RuntimeError(
                f"profile line not found for {system.name}/{case['case']} repeat {repeat_idx}\n{text}"
            )
        row = match.groupdict()
        out = {
            "system": system.name,
            "case": case["case"],
            "repeat": repeat_idx,
            "mode": row.get("mode") or "",
            "builder": row.get("builder") or "",
            "policy": row.get("policy") or "",
            "rank": int(row["rank"]) if row.get("rank") else 0,
            "nedges": int(row["nedges"]),
            "npairs": int(row["npairs"]),
            "pair_metadata_total_ms": float(row["pair_metadata_total_ms"]),
            "pair_metadata_build_ms": float(row["pair_metadata_build_ms"]),
            "model_inference_ms": float(row["model_inference_ms"]),
            "compute_total_ms": float(row["compute_total_ms"]),
        }
        return out


def summarize(rows: list[dict]):
    summary = []
    keys = [
        "pair_metadata_total_ms",
        "pair_metadata_build_ms",
        "model_inference_ms",
        "compute_total_ms",
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["case"]), []).append(row)

    for (system, case), grp in grouped.items():
        out = {
            "system": system,
            "case": case,
            "n": len(grp),
            "mode": grp[0]["mode"],
            "builder": grp[0]["builder"],
            "policy": grp[0]["policy"],
            "nedges_mean": statistics.mean(x["nedges"] for x in grp),
            "npairs_mean": statistics.mean(x["npairs"] for x in grp),
        }
        for key in keys:
            values = [x[key] for x in grp]
            out[f"{key}_mean"] = statistics.mean(values)
            out[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return sorted(summary, key=lambda x: (x["system"], x["case"]))


def write_csv(path: pathlib.Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: pathlib.Path, summary_rows: list[dict]):
    grouped: dict[str, dict[str, dict]] = {}
    for row in summary_rows:
        grouped.setdefault(row["system"], {})[row["case"]] = row

    lines = [
        "# LAMMPS Pair Metadata Bench",
        "",
        f"- LAMMPS binary: `{LAMMPS_BIN}`",
        f"- repeats: `{REPEATS}`",
        "- cases: `baseline`, `geometry_only_legacy`, `geometry_only_upstream`",
        "- note: current environment exposes a single GPU, so this report covers serial `pair_style e3gnn` only",
        "",
    ]

    for system, cases in grouped.items():
        lines.append(f"## {system}")
        lines.append("")
        base = cases.get("baseline")
        legacy = cases.get("geometry_only_legacy")
        upstream = cases.get("geometry_only_upstream")
        if base:
            lines.append(
                f"- baseline total: `{base['compute_total_ms_mean']:.3f} ± {base['compute_total_ms_std']:.3f} ms`"
            )
        if legacy:
            lines.append(
                f"- geometry_only legacy pair_metadata: `{legacy['pair_metadata_total_ms_mean']:.3f} ± {legacy['pair_metadata_total_ms_std']:.3f} ms`"
            )
            lines.append(
                f"- geometry_only legacy total: `{legacy['compute_total_ms_mean']:.3f} ± {legacy['compute_total_ms_std']:.3f} ms`"
            )
        if upstream:
            lines.append(
                f"- geometry_only upstream pair_metadata: `{upstream['pair_metadata_total_ms_mean']:.3f} ± {upstream['pair_metadata_total_ms_std']:.3f} ms`"
            )
            lines.append(
                f"- geometry_only upstream total: `{upstream['compute_total_ms_mean']:.3f} ± {upstream['compute_total_ms_std']:.3f} ms`"
            )
        if legacy and upstream:
            meta_speedup = legacy["pair_metadata_total_ms_mean"] / upstream["pair_metadata_total_ms_mean"]
            total_speedup = legacy["compute_total_ms_mean"] / upstream["compute_total_ms_mean"]
            lines.append(f"- pair_metadata reduction factor: `{meta_speedup:.3f}x`")
            lines.append(f"- geometry_only total reduction factor: `{total_speedup:.3f}x`")
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    if not LAMMPS_BIN.is_file():
        raise FileNotFoundError(LAMMPS_BIN)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for system in SYSTEMS:
        for case in CASES:
            for repeat_idx in range(REPEATS):
                rows.append(run_once(system, case, repeat_idx))
    summary_rows = summarize(rows)
    write_csv(METRICS_DIR / "lammps_pair_metadata_raw.csv", rows)
    write_csv(METRICS_DIR / "lammps_pair_metadata_summary.csv", summary_rows)
    write_report(REPORTS_DIR / "lammps_pair_metadata_report.md", summary_rows)


if __name__ == "__main__":
    main()
