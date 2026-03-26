#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import pathlib
import re


LOOP_RE = re.compile(
    r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps with\s+(\d+)\s+atoms"
)
PERF_RE = re.compile(r"Performance:\s+([0-9.eE+-]+)\s+ns/day,\s+([0-9.eE+-]+)\s+hours/ns,\s+([0-9.eE+-]+)\s+timesteps/s")
def parse_log(path: pathlib.Path) -> dict[str, float | int | str]:
    text = path.read_text()

    loops = LOOP_RE.findall(text)
    if not loops:
        raise RuntimeError(f"No loop summary found in {path}")
    loop_time, procs, steps, atoms = loops[-1]

    perf = PERF_RE.findall(text)
    if not perf:
        raise RuntimeError(f"No performance summary found in {path}")
    ns_per_day, hours_per_ns, timesteps_per_s = perf[-1]

    loop_time_f = float(loop_time)
    steps_i = int(steps)
    atoms_i = int(atoms)
    step_ms = loop_time_f / steps_i * 1000.0
    atoms_per_s = atoms_i * float(timesteps_per_s)

    return {
        "variant": path.stem.replace("log_", ""),
        "atoms": atoms_i,
        "steps": steps_i,
        "procs": int(procs),
        "loop_time_s": loop_time_f,
        "step_ms": step_ms,
        "timesteps_per_s": float(timesteps_per_s),
        "atoms_per_s": atoms_per_s,
        "ns_per_day": float(ns_per_day),
        "hours_per_ns": float(hours_per_ns),
        "log_path": str(path),
    }


def format_table(rows: list[dict[str, float | int | str]]) -> str:
    baseline = next((r for r in rows if r["variant"] == "baseline"), None)
    headers = [
        "variant",
        "atoms",
        "step_ms",
        "timesteps_per_s",
        "atoms_per_s",
        "ns_per_day",
        "vs_baseline_step",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if baseline is None:
            delta = "0.00%"
        else:
            delta_val = (float(baseline["step_ms"]) - float(row["step_ms"])) / float(baseline["step_ms"]) * 100.0
            delta = f"{delta_val:+.2f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["atoms"]),
                    f"{float(row['step_ms']):.3f}",
                    f"{float(row['timesteps_per_s']):.3f}",
                    f"{float(row['atoms_per_s']):.3f}",
                    f"{float(row['ns_per_day']):.3f}",
                    delta,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=pathlib.Path)
    args = parser.parse_args()

    log_paths = sorted(args.log_dir.glob("log_*.lammps"))
    if not log_paths:
        raise SystemExit(f"No logs found in {args.log_dir}")

    rows = [parse_log(path) for path in log_paths]
    order = {"baseline": 0, "pairaware": 1, "flash": 2, "combined": 3}
    rows.sort(key=lambda row: order.get(str(row["variant"]), 99))

    csv_path = args.log_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "atoms",
                "steps",
                "procs",
                "loop_time_s",
                "step_ms",
                "timesteps_per_s",
                "atoms_per_s",
                "ns_per_day",
                "hours_per_ns",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    table = format_table(rows)
    md_path = args.log_dir / "summary.md"
    md_path.write_text(table)
    print(table, end="")
    print(f"CSV: {csv_path}")
    print(f"MD: {md_path}")


if __name__ == "__main__":
    main()
