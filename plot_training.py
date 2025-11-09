#!/usr/bin/env python
"""Parse ProtoPNet train.log and generate training curves."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAT_LABELS = {
    "time": ("time_sec", float),
    "cross ent": ("cross_entropy", float),
    "cluster": ("cluster_cost", float),
    "separation": ("separation_cost", float),
    "diversity": ("diversity_cost", float),
    "avg separation": ("avg_separation_cost", float),
    "accu": ("accuracy", lambda s: float(s.rstrip("%"))),
    "l1": ("l1_norm", float),
    "avg proto dist": ("avg_proto_dist", float),
}

PHASE_LABELS = {"last layer": "last_only", "warm": "warm", "joint": "joint"}


def parse_log(log_path: Path) -> pd.DataFrame:
    epoch_re = re.compile(r"^epoch:\s*(\d+)")
    iteration_re = re.compile(r"^iteration:\s*(\d+)")

    records = []
    current = None
    current_epoch = None
    current_iteration = None
    current_phase = None

    with log_path.open("r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("epoch:"):
                match = epoch_re.match(stripped)
                if match:
                    current_epoch = int(match.group(1))
                continue

            if stripped.startswith("iteration:"):
                match = iteration_re.match(stripped)
                if match:
                    current_iteration = int(match.group(1))
                continue

            if stripped in PHASE_LABELS:
                current_phase = PHASE_LABELS[stripped]
                continue

            if stripped == "train" or stripped == "test":
                current = {
                    "mode": stripped,
                    "epoch": current_epoch,
                    "iteration": current_iteration,
                    "phase": current_phase,
                }
                continue

            if stripped.startswith("--------------------------------"):
                current_iteration = None
                current = None
                continue

            if current is None:
                continue

            if stripped.startswith(tuple(STAT_LABELS.keys())):
                label, value_str = stripped.split(":", 1)
                label = label.strip()
                value_str = value_str.strip()
                key, caster = STAT_LABELS[label]
                try:
                    current[key] = caster(value_str)
                except ValueError:
                    current[key] = np.nan

                if label == "avg proto dist":
                    # finalize record when we reach the last statistic
                    records.append(current)
                    current = None

    df = pd.DataFrame.from_records(records)
    df.sort_values(["epoch", "mode"], inplace=True)
    return df


def plot_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, group in df.dropna(subset=["accuracy"]).groupby("mode"):
        ax.plot(group["epoch"], group["accuracy"], marker="o", linewidth=1.2, label=mode)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Test Accuracy")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=200)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ProtoPNet training progress from train.log.")
    parser.add_argument("--log", required=True, help="Path to train.log")
    parser.add_argument("--output", default="training_figures", help="Directory for plots and CSV")
    parser.add_argument("--export-json", action="store_true", help="Save parsed metrics as JSON alongside CSV")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_log(log_path)
    if df.empty:
        raise RuntimeError(f"No records parsed from {log_path}")

    df.to_csv(out_dir / "training_metrics.csv", index=False)
    if args.export_json:
        (out_dir / "training_metrics.json").write_text(df.to_json(orient="records", indent=2))

    plot_accuracy(df, out_dir)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
