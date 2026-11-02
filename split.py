from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import re

# ============================================================
# CONFIGURATION
# ============================================================

from paths import PROJECT_DIR, DATASET_DIR
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"

SEED = 42

# Keep False so the frozen splits.csv used in the experiments
# is never overwritten accidentally.
OVERWRITE_SPLIT = False

PATTERN = re.compile(
    r"Camera\s+(\d+)\s+Video\s+\((\d+)\)_(\d+)",
    re.IGNORECASE,
)


# ============================================================
# READ DATASET
# ============================================================

def build_records():
    images = {
        p.stem: p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}
    }

    records = []

    for name, path in images.items():
        match = PATTERN.fullmatch(name)

        if not match:
            raise ValueError(f"Cannot parse filename: {name}")

        camera = int(match.group(1))
        video = int(match.group(2))
        frame = int(match.group(3))

        mask_path = MASK_DIR / f"{name}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        records.append({
            "name": name,
            "image": path.name,
            "mask": mask_path.name,
            "camera": camera,
            "video": video,
            "frame": frame,
        })

    records.sort(
        key=lambda r: (
            r["camera"],
            r["video"],
            r["frame"],
        )
    )

    return records


# ============================================================
# VIDEO-DISJOINT 70/10/20 SPLIT
# ============================================================

def make_camera_split(video_groups, seed):
    rng = random.Random(seed)
    videos = list(video_groups.keys())

    total_images = sum(
        len(video_groups[v])
        for v in videos
    )

    targets = {
        "train": total_images * 0.70,
        "val": total_images * 0.10,
        "test": total_images * 0.20,
    }

    best_assignment = None
    best_score = float("inf")

    for _ in range(10000):
        buckets = defaultdict(list)

        for video in videos:
            buckets[len(video_groups[video])].append(video)

        ordered = []

        for size in sorted(buckets, reverse=True):
            same_size = buckets[size].copy()
            rng.shuffle(same_size)
            ordered.extend(same_size)

        counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        assignment = {}

        for video in ordered:
            n = len(video_groups[video])

            deficits = {
                split: (targets[split] - counts[split]) / max(targets[split], 1)
                for split in targets
            }

            split = max(deficits, key=deficits.get)
            assignment[video] = split
            counts[split] += n

        score = sum(
            abs(counts[s] - targets[s])
            for s in targets
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    return best_assignment


def generate_split(records):
    groups_by_camera = defaultdict(lambda: defaultdict(list))

    for record in records:
        groups_by_camera[record["camera"]][record["video"]].append(record)

    split_lookup = {}

    for camera in sorted(groups_by_camera):
        assignment = make_camera_split(
            groups_by_camera[camera],
            seed=SEED + camera,
        )

        for video, split in assignment.items():
            split_lookup[(camera, video)] = split

    for record in records:
        record["split"] = split_lookup[
            (record["camera"], record["video"])
        ]

    return records


# ============================================================
# SAVE / LOAD
# ============================================================

def save_split(records):
    with open(
        SPLIT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "image",
                "mask",
                "camera",
                "video",
                "frame",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def load_existing_split():
    records = []

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            records.append({
                "name": row["name"],
                "image": row["image"],
                "mask": row["mask"],
                "camera": int(row["camera"]),
                "video": int(row["video"]),
                "frame": int(row["frame"]),
                "split": row["split"].strip(),
            })

    return records


# ============================================================
# VERIFY
# ============================================================

def verify(records):
    print("=" * 70)
    print("AGRIVISION DB-1 VIDEO-DISJOINT SPLIT CHECK")
    print("=" * 70)

    counts = Counter(
        r["split"]
        for r in records
    )

    total = len(records)

    for split in ["train", "val", "test"]:
        n = counts[split]
        print(
            f"{split:5s}: "
            f"{n:4d} images "
            f"({100*n/total:.2f}%)"
        )

    print("\nCAMERA DISTRIBUTION")
    print("-" * 70)

    for camera in sorted(set(r["camera"] for r in records)):
        subset = [
            r for r in records
            if r["camera"] == camera
        ]

        c = Counter(
            r["split"]
            for r in subset
        )

        print(
            f"Camera {camera}: "
            f"Train={c['train']}, "
            f"Val={c['val']}, "
            f"Test={c['test']}"
        )

    video_splits = defaultdict(set)

    for record in records:
        video_splits[
            (record["camera"], record["video"])
        ].add(record["split"])

    leaking = {
        group: splits
        for group, splits in video_splits.items()
        if len(splits) > 1
    }

    print("\nVIDEO LEAKAGE CHECK")
    print("-" * 70)
    print(f"Video groups: {len(video_splits)}")
    print(f"Groups appearing in multiple splits: {len(leaking)}")

    if leaking:
        raise RuntimeError("Video leakage detected.")

    print("PASS: No source video crosses train/val/test.")


# ============================================================
# MAIN
# ============================================================

def main():
    if SPLIT_FILE.exists() and not OVERWRITE_SPLIT:
        print(f"Using existing frozen split:\n{SPLIT_FILE}\n")
        verify(load_existing_split())
        return

    records = build_records()
    records = generate_split(records)
    save_split(records)

    print(f"New split saved:\n{SPLIT_FILE}\n")
    verify(records)


if __name__ == "__main__":
    main()
