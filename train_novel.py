"""
Novelty screening: learned upsampling (DySample) in the SegFormer-B0 64-ch decoder.

Uses EXACTLY the final recipe from train_recipe.py (r5: AdamW + warmup/poly,
CE + Dice, augmentation, 100 epochs, same split and evaluation).
Only the model changes. Seed 42 first, compared with r5 seed 42 (80.33 IoU).

    n1_dysample       DySample replaces bilinear upsampling inside the decoder
    n2_dysample_half  n1 + DySample x2 before the classifier (predict at 1/2 res)
    n3_bilinear_half  control: bilinear x2 before the classifier (predict at 1/2 res)

Usage:
    python train_novel.py
Stop with Ctrl+C at any time; rerun to resume. Results: runs/novel_summary.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv
import statistics

import train_recipe as recipe
import model_novel as mn


FINAL = {"loss": "ce_dice", "augment": True}

NOVEL = {
    "n1_dysample":      {**FINAL, "build": mn.build_dysample},
    "n2_dysample_half": {**FINAL, "build": mn.build_dysample_half},
    "n3_bilinear_half": {**FINAL, "build": mn.build_bilinear_half},
}

recipe.RECIPES = NOVEL
recipe.EXPERIMENTS = (
    # seed 42 screening - already finished, skipped automatically
    [(name, 42) for name in NOVEL]
    # confirmation: n2 first (the novelty claim), then the control, then n1
    + [("n2_dysample_half", 7), ("n2_dysample_half", 123),
       ("n3_bilinear_half", 7), ("n3_bilinear_half", 123),
       ("n1_dysample", 7), ("n1_dysample", 123)]
)
recipe.SUMMARY_FILE = recipe.RUNS_DIR / "novel_summary.csv"
recipe.build_for = lambda cfg: cfg["build"]


def print_summary():
    if not recipe.SUMMARY_FILE.exists():
        return
    with open(recipe.SUMMARY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 78)
    print("NOVELTY SCREENING (final recipe, SegFormer-B0 64-ch decoder)")
    print("=" * 78)
    print(f"{'Variant':<20}{'Seed':>5}{'IoU@0.5':>9}{'Thr':>7}{'ImgIoU':>8}{'<1%':>7}{'1-3%':>7}")
    for r in rows:
        print(f"{r['recipe']:<20}{r['seed']:>5}{float(r['test_iou_t05']):>9.2f}{float(r['val_best_threshold']):>7.3f}"
              f"{float(r['test_per_image_iou']):>8.2f}{float(r['img_iou_lt1']):>7.2f}{float(r['img_iou_1_3']):>7.2f}")

    multi = {}
    for r in rows:
        multi.setdefault(r["recipe"], []).append(float(r["test_iou_t05"]))
    means = [(k, v) for k, v in multi.items() if len(v) > 1]
    if means:
        print("\nMean +/- SD across seeds:")
        for k, v in means:
            print(f"  {k:<20}{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}  ({len(v)} seeds)")

    print("\nTarget, baseline 64-ch with final recipe:")
    print("  seed 42: IoU 80.33 | ImgIoU 73.93 | <1% 50.84 | 1-3% 63.22")
    print("  3 seeds: IoU 80.21 +/- 0.11   (clear win: about 80.7 or higher)")


recipe.print_summary = print_summary


if __name__ == "__main__":
    recipe.main()
