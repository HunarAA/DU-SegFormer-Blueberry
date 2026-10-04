from pathlib import Path
import csv
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ============================================================
# MAKE FIGURE 1 - STRICT SQUARE VERSION
# ============================================================
#
# IMPORTANT:
# - Only 1440 x 1440 source images are allowed.
# - The script STOPS if a selected image is not 1440 x 1440.
# - Existing make_figure1.png / .pdf are deleted before saving.
# - No vertical row labels are used, avoiding the large left margin.
#
# Output:
#   figures/make_figure1.png
#   figures/make_figure1.pdf
# ============================================================

SCRIPT_VERSION = "MAKE_FIGURE1_V4_SQUARE_ONLY"

PROJECT_DIR = Path(r"E:\Papers\AgriVision Blueberry")
DATASET_DIR = PROJECT_DIR / "DB-1"
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"

OUTPUT_DIR = PROJECT_DIR / "figures"
OUTPUT_PNG = OUTPUT_DIR / "make_figure1.png"
OUTPUT_PDF = OUTPUT_DIR / "make_figure1.pdf"

NUM_EXAMPLES = 3
SOURCE_SPLIT = "test"
RANDOM_SEED = 42

FIG_DPI = 600
PANEL_LABEL_FONT = 13


def load_records():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def camera_id(filename):
    match = re.search(
        r"Camera\s*(\d+)",
        filename,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def get_size(filename):
    with Image.open(IMAGE_DIR / filename) as img:
        return img.size


def choose_examples():
    records = [
        row
        for row in load_records()
        if row["split"].strip() == SOURCE_SPLIT
    ]

    # STRICT: keep only square 1440 x 1440 images.
    square_records = [
        row
        for row in records
        if get_size(row["image"]) == (1440, 1440)
    ]

    if len(square_records) < NUM_EXAMPLES:
        raise RuntimeError(
            f"Only {len(square_records)} square 1440x1440 images "
            f"were found in split '{SOURCE_SPLIT}'."
        )

    # Prefer different cameras.
    by_camera = {}

    for row in square_records:
        cam = camera_id(row["image"])

        if cam is not None:
            by_camera.setdefault(cam, []).append(row)

    available_cameras = sorted(by_camera.keys())

    if len(available_cameras) < NUM_EXAMPLES:
        raise RuntimeError(
            f"Only {len(available_cameras)} cameras contain 1440x1440 "
            f"images in the '{SOURCE_SPLIT}' split. Need {NUM_EXAMPLES}."
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(available_cameras)

    selected = []

    for cam in available_cameras[:NUM_EXAMPLES]:
        choices = sorted(
            by_camera[cam],
            key=lambda row: row["image"],
        )
        selected.append(rng.choice(choices))

    # Final hard check.
    for row in selected:
        size = get_size(row["image"])

        if size != (1440, 1440):
            raise RuntimeError(
                f"Non-square image selected unexpectedly: "
                f"{row['image']} -> {size}"
            )

    return selected


def load_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def load_binary_mask(path):
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))

    return (arr > 127).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old figure files first so there is no confusion
    # about whether the newly generated figure was actually saved.
    for old_output in [OUTPUT_PNG, OUTPUT_PDF]:
        if old_output.exists():
            old_output.unlink()

    print("=" * 72)
    print(SCRIPT_VERSION)
    print("=" * 72)

    selected = choose_examples()

    print("\nSelected examples:")

    for i, row in enumerate(selected):
        size = get_size(row["image"])
        cam = camera_id(row["image"])

        print(
            f"  ({chr(ord('a') + i)}) "
            f"{row['image']} | Camera {cam} | "
            f"{size[0]}x{size[1]}"
        )

    print(
        "\nAll selected images passed the strict 1440x1440 check."
    )

    # Equal subplot dimensions because all selected images are square.
    fig, axes = plt.subplots(
        2,
        NUM_EXAMPLES,
        figsize=(10.4, 6.8),
    )

    for col, row in enumerate(selected):
        image = load_rgb(
            IMAGE_DIR / row["image"]
        )

        mask = load_binary_mask(
            MASK_DIR / row["mask"]
        )

        axes[0, col].imshow(image)
        axes[0, col].axis("off")
        axes[0, col].set_title(
            f"({chr(ord('a') + col)})",
            fontsize=PANEL_LABEL_FONT,
            pad=5,
        )

        axes[1, col].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, col].axis("off")

    # No row labels: the caption explains top/bottom rows.
    # This also removes the large empty band on the left.
    plt.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.95,
        bottom=0.005,
        wspace=0.025,
        hspace=0.035,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    if not OUTPUT_PNG.exists():
        raise RuntimeError(
            "PNG output was not created."
        )

    print("\nNEW FIGURE SAVED:")
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print("\nFinished successfully.")


if __name__ == "__main__":
    main()
