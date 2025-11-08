#!/usr/bin/env python
"""Compute prototype-part alignment distances and plot their distribution."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms


def load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a saved ProtoPNet checkpoint on the requested device."""
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def load_part_locations(dataset_root: str | Path, img_size: int):
    """Return train dataset, the image-id array, and scaled part locations."""
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
    train_set = datasets.ImageFolder(str(train_dir), transform)

    img_sizes = {}
    img_ids = []
    for path, _ in train_set.samples:
        img_id = int(Path(path).stem)
        with Image.open(path) as img:
            width, height = img.size
        img_sizes[img_id] = (width, height)
        img_ids.append(img_id)

    part_locs = pd.read_csv(dataset_root / "part_locs.csv")
    size_df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in img_sizes.items()],
        columns=["image_id", "width", "height"],
    )
    part_locs = part_locs.merge(size_df, on="image_id")
    part_locs["x"] = (part_locs["x"] * img_size / part_locs["width"]).round().astype(int)
    part_locs["y"] = (part_locs["y"] * img_size / part_locs["height"]).round().astype(int)
    part_locs = part_locs[["image_id", "part_name", "x", "y"]]
    return train_set, np.array(img_ids), part_locs


def load_prototype_info(model_path: str | Path) -> np.ndarray:
    """Load bounding-box metadata (bb.npy) associated with the checkpoint."""
    model_path = Path(model_path).resolve()
    match = re.search(r"\d+", model_path.stem)
    if not match:
        raise ValueError("Cannot infer epoch number from checkpoint name.")
    epoch = match.group(0)
    proto_dir = model_path.parent.parent / "img" / f"epoch-{epoch}"
    return np.load(proto_dir / "bb.npy")


def compute_ppa(
    proto_info: np.ndarray,
    train_img_ids: np.ndarray,
    part_locs: pd.DataFrame,
    radius: float,
):
    """Compute distances between prototypes and annotated parts."""
    hits = []
    rows = []
    grouped = part_locs.groupby("image_id")

    for proto_idx, row in enumerate(proto_info.astype(int)):
        dataset_idx, h0, h1, w0, w1, proto_cls = row
        image_id = train_img_ids[dataset_idx]
        parts = grouped.get_group(image_id)
        cx = 0.5 * (w0 + w1)
        cy = 0.5 * (h0 + h1)
        dx = parts["x"] - cx
        dy = parts["y"] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        best_idx = dist.idxmin()
        best = parts.loc[best_idx]
        distance = dist.loc[best_idx]
        rows.append(
            dict(
                prototype=proto_idx,
                image_id=image_id,
                part=best["part_name"],
                distance=distance,
                cls=int(proto_cls),
            )
        )
        hits.append(distance <= radius)

    ppa = pd.DataFrame(rows)
    return ppa, float(np.mean(hits))


def plot_ppa_distribution(
    df: pd.DataFrame,
    hit_rate: float,
    out_dir: Path,
    bins: int = 40,
) -> Path:
    """Plot histogram of prototype-to-part distances and save the figure."""
    distances = df["distance"].dropna().to_numpy()
    median_distance = float(np.median(distances)) if distances.size else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distances, bins=bins, color="#4B85F5", alpha=0.85, edgecolor="white")

    if np.isfinite(median_distance):
        ax.axvline(
            median_distance,
            color="#E4572E",
            linestyle="--",
            linewidth=2,
            label=f"median={median_distance:.1f}",
        )

    ax.set_xlabel("Prototype-to-part distance (pixels)")
    ax.set_ylabel("Prototypes")
    ax.set_title(f"PPA distance distribution · hit rate={hit_rate:.3f}")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "ppa_distance_hist.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the PPA distance distribution plot for a ProtoPNet checkpoint."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved ProtoPNet checkpoint (torch.save output).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root containing the train split and part_locs.csv.",
    )
    parser.add_argument(
        "--radius",
        required=True,
        type=float,
        help="Radius threshold (in pixels) used for computing the hit rate.",
    )
    parser.add_argument(
        "--output",
        default="figures",
        help="Directory where the histogram figure and CSV table are saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins to use for the histogram (default: 40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    _, train_img_ids, part_locs = load_part_locations(args.dataset, model.img_size)
    proto_info = load_prototype_info(args.model)
    ppa_table, hit_rate = compute_ppa(proto_info, train_img_ids, part_locs, args.radius)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ppa_table.csv"
    ppa_table.to_csv(csv_path, index=False)
    figure_path = plot_ppa_distribution(ppa_table, hit_rate, output_dir, bins=args.bins)

    print(f"Saved PPA table to {csv_path}")
    print(f"Saved histogram to {figure_path}")
    print(f"PPA hit rate (radius={args.radius}): {hit_rate:.3f}")


if __name__ == "__main__":
    main()