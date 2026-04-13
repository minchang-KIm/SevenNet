from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def write_csv() -> pd.DataFrame:
    rows = [
        {
            "dataset": "qm9_hf",
            "group": "small molecular",
            "natoms": 29,
            "baseline_stable_ms": 28.58796447981149,
            "pair_stable_ms": 47.14385146507993,
        },
        {
            "dataset": "iso17",
            "group": "small molecular",
            "natoms": 19,
            "baseline_stable_ms": 29.043069458566606,
            "pair_stable_ms": 47.231388511136174,
        },
        {
            "dataset": "salex_train_official",
            "group": "periodic medium",
            "natoms": 132,
            "baseline_stable_ms": 151.56103752087802,
            "pair_stable_ms": 147.3670310806483,
        },
        {
            "dataset": "oc20_s2ef_train_20m",
            "group": "periodic medium",
            "natoms": 225,
            "baseline_stable_ms": 125.32425503013656,
            "pair_stable_ms": 118.60258149681613,
        },
        {
            "dataset": "omat24_1m_official",
            "group": "periodic large",
            "natoms": 160,
            "baseline_stable_ms": 232.0935784955509,
            "pair_stable_ms": 228.9856020361185,
        },
        {
            "dataset": "mptrj",
            "group": "periodic large",
            "natoms": 444,
            "baseline_stable_ms": 424.73339347634465,
            "pair_stable_ms": 419.9460480012931,
        },
    ]
    df = pd.DataFrame(rows)
    df["speedup_baseline_over_pair"] = (
        df["baseline_stable_ms"] / df["pair_stable_ms"]
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ASSETS / "rechecked_representative_cases.csv", index=False)
    return df


def plot_rechecked_speedup(df: pd.DataFrame) -> None:
    ordered = df.sort_values("speedup_baseline_over_pair", ascending=False)
    colors = [
        "#2f855a" if value > 1.0 else "#c0563d"
        for value in ordered["speedup_baseline_over_pair"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(ordered["dataset"], ordered["speedup_baseline_over_pair"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Baseline / Pair")
    ax.set_title("Stable speedup after discarding warm-up repeats")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(ASSETS / "rechecked_representative_speedup.png", dpi=220)
    plt.close(fig)


def plot_rechecked_latency(df: pd.DataFrame) -> None:
    ordered = df.sort_values("natoms")
    x = range(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(
        [i - width / 2 for i in x],
        ordered["baseline_stable_ms"],
        width,
        label="baseline",
        color="#4c78a8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        ordered["pair_stable_ms"],
        width,
        label="pair",
        color="#f58518",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["dataset"], rotation=30)
    ax.set_ylabel("Stable median latency (ms)")
    ax.set_title("Representative stable latency recheck")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "rechecked_representative_latency.png", dpi=220)
    plt.close(fig)


def plot_qm9_warmup_artifact() -> None:
    baseline = [292.39, 371.14, 29.77, 28.67, 28.53, 28.61, 28.56, 28.56, 28.54, 28.54]
    pair = [403.56, 76.07, 47.26, 47.01, 47.08, 47.47, 47.2, 46.94, 46.98, 46.92]
    repeats = list(range(1, len(baseline) + 1))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(repeats, baseline, marker="o", label="baseline", color="#4c78a8")
    ax.plot(repeats, pair, marker="o", label="pair", color="#f58518")
    ax.axvspan(1, 2, color="#dddddd", alpha=0.5)
    ax.text(1.15, max(baseline + pair) * 0.92, "warm-up dominated", fontsize=9)
    ax.set_xlabel("Repeated call index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Warm-up artifact on qm9_hf")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ASSETS / "qm9_warmup_artifact.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df = write_csv()
    plot_rechecked_speedup(df)
    plot_rechecked_latency(df)
    plot_qm9_warmup_artifact()


if __name__ == "__main__":
    main()
