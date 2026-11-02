"""
Leakage study: how much do video-derived frames inflate segmentation results?

Model and recipe: SegFormer-B0 64-ch decoder, final recipe (r5: AdamW +
warmup/poly, CE + Dice, augmentation, 100 epochs). Only the data split changes.

PART A - controlled leakage (same test images, same training-set size)
    Test videos with >= 2 frames are split frame-by-frame (alternating):
      E = evaluation frames (odd positions), S = sibling frames (even positions).
    Disjoint condition : the existing r5 models (trained on the normal 833
                         training images, seeds 42/7/123) evaluated on E.
                         -> no new training needed.
    Leaky condition    : training set = 833 normal training images minus |S|
                         randomly removed + the sibling frames S (still 833
                         images in total), seeds 42/7/123, evaluated on E.
    The only difference is whether training contains other frames from the
    same videos as the evaluation images.

PART B - common practice (random image-level split)
    Three random 833/122/240 image-level partitions (ignoring videos),
    trained with seed 42, each tested on its own random test set, compared
    with the video-disjoint result (80.21 +/- 0.11).

Usage:
    python train_leakage.py
6 training runs (about 5 hours). Stop with Ctrl+C at any time; rerun to resume.
Results: runs/leakage_summary.csv and runs/leakage_splits.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv
import random
import statistics
from collections import defaultdict

import torch

import train_recipe as recipe
import train_repeats_resume as base
import model_ablation as abl


FINAL = {"loss": "ce_dice", "augment": True, "build": abl.build_segformer_dec64}
TRAIN_SEEDS = [42, 7, 123]
RANDOM_PARTITIONS = [1, 2, 3]
DISJOINT_RUN = "recipe_r5_adamw_cedice_aug_seed{}"   # existing final-recipe models

recipe.RECIPES = {"leaky_siblings": FINAL,
                  **{f"random_split_p{p}": FINAL for p in RANDOM_PARTITIONS}}
recipe.SUMMARY_FILE = recipe.RUNS_DIR / "leakage_summary.csv"
recipe.build_for = lambda cfg: cfg["build"]


# ------------------------------------------------------------
# Splits
# ------------------------------------------------------------

def load_all():
    with open(base.SPLIT_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(r):
    return (r["camera"], r["video"])


def controlled_split(rows):
    """Returns records for the leaky condition and the shared evaluation set E."""
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test_videos = defaultdict(list)
    for r in rows:
        if r["split"] == "test":
            test_videos[key(r)].append(r)

    E, S = [], []
    for frames in test_videos.values():
        if len(frames) < 2:
            continue
        frames = sorted(frames, key=lambda r: int(r["frame"]))
        E += frames[0::2]
        S += frames[1::2]

    rng = random.Random(0)
    kept = train.copy()
    rng.shuffle(kept)
    kept = kept[len(S):]

    leaky = {"train": kept + S, "val": val, "test": E}
    disjoint_eval = {"train": train, "val": val, "test": E}
    return leaky, disjoint_eval, E, S


def random_split(rows, partition):
    shuffled = rows.copy()
    random.Random(partition).shuffle(shuffled)
    records = {"train": shuffled[:833], "val": shuffled[833:955], "test": shuffled[955:]}
    train_videos = {key(r) for r in records["train"]}
    shared = sum(key(r) in train_videos for r in records["test"])
    return records, shared / len(records["test"]) * 100


def save_split_file(leaky, randoms):
    path = recipe.RUNS_DIR / "leakage_splits.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "split", "name"])
        for split, recs in leaky.items():
            for r in recs:
                w.writerow(["leaky_siblings", split, r["name"]])
        for p, recs in randoms.items():
            for split, rr in recs.items():
                for r in rr:
                    w.writerow([f"random_split_p{p}", split, r["name"]])


# ------------------------------------------------------------
# Disjoint models evaluated on E (no training)
# ------------------------------------------------------------

def evaluate_disjoint(disjoint_eval, done):
    for seed in TRAIN_SEEDS:
        if ("disjoint_on_E", seed) in done:
            continue
        ckpt = recipe.RUNS_DIR / DISJOINT_RUN.format(seed) / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"\nEvaluating existing video-disjoint model (seed {seed}) on the shared evaluation frames...")
        model = abl.build_segformer_dec64(model_name="nvidia/mit-b0", num_classes=2)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
        model.to(recipe.DEVICE).eval()
        result = recipe.final_evaluation(model, disjoint_eval)
        recipe.append_summary({"recipe": "disjoint_on_E", "seed": seed, "loss": "ce_dice", "augment": True,
                               "best_epoch": "", "best_val_iou": "", **result, "runtime_minutes": 0})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def print_summary(n_e=None, n_s=None, shared=None):
    if not recipe.SUMMARY_FILE.exists():
        return
    with open(recipe.SUMMARY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def vals(prefix, col):
        return [float(r[col]) for r in rows if r["recipe"].startswith(prefix)]

    def ms(v):
        if not v:
            return "-"
        return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.2f} (n=1)"

    print("\n" + "=" * 78)
    print("LEAKAGE STUDY (SegFormer-B0 64-ch, final recipe)")
    print("=" * 78)
    for r in rows:
        print(f"{r['recipe']:<20}{r['seed']:>5}  IoU@0.5 {float(r['test_iou_t05']):6.2f}  "
              f"per-image IoU {float(r['test_per_image_iou']):6.2f}")

    print("\nA) Same evaluation frames" + (f" (E = {n_e} frames; {n_s} sibling frames)" if n_e else ""))
    d, l = vals("disjoint_on_E", "test_iou_t05"), vals("leaky_siblings", "test_iou_t05")
    print(f"   Video-disjoint training : {ms(d)}")
    print(f"   Leaky training          : {ms(l)}")
    if d and l:
        print(f"   Inflation from leakage  : {statistics.mean(l) - statistics.mean(d):+.2f} IoU")

    print("\nB) Random image-level split vs video-disjoint split")
    rs = vals("random_split", "test_iou_t05")
    print(f"   Random splits           : {ms(rs)}")
    print("   Video-disjoint (3 seeds): 80.21 +/- 0.11")
    if rs:
        print(f"   Apparent inflation      : {statistics.mean(rs) - 80.21:+.2f} IoU")
    if shared:
        print("   Random-test images with a frame of the same video in training: "
              + ", ".join(f"p{p} {v:.1f}%" for p, v in shared.items()))


def main():
    recipe.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_all()

    leaky, disjoint_eval, E, S = controlled_split(rows)
    randoms, shared = {}, {}
    for p in RANDOM_PARTITIONS:
        randoms[p], shared[p] = random_split(rows, p)
    save_split_file(leaky, randoms)

    print("=" * 78)
    print("LEAKAGE STUDY")
    print("=" * 78)
    print(f"Part A: evaluation frames E = {len(E)}, sibling frames S = {len(S)}, "
          f"leaky training set = {len(leaky['train'])} images")
    print("Part B: random-test images sharing a video with training: "
          + ", ".join(f"p{p} {v:.1f}%" for p, v in shared.items()))

    done = recipe.completed_runs()
    evaluate_disjoint(disjoint_eval, done)

    jobs = [("leaky_siblings", s, leaky) for s in TRAIN_SEEDS]
    jobs += [(f"random_split_p{p}", 42, randoms[p]) for p in RANDOM_PARTITIONS]
    for name, seed, records in jobs:
        if (name, seed) in recipe.completed_runs():
            print(f"SKIPPING (done): {name} seed {seed}")
            continue
        recipe.run(name, seed, records)
        print_summary(len(E), len(S), shared)

    print_summary(len(E), len(S), shared)
    print("\nLEAKAGE STUDY COMPLETE")


if __name__ == "__main__":
    main()
