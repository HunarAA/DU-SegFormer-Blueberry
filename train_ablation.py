"""
Step 1 ablation runs: 4 variants x 3 seeds = 12 runs.

This script reuses the EXACT training code from train_repeats_resume.py
(same data split, letterboxing, loss, optimizer, AMP, early stopping,
checkpoint selection and resume support). Only the list of models and the
summary file are changed, so the ablation is directly comparable with
your published SegFormer-B0 and SFAF results.

Order of runs: all four variants with seed 42 first, then seed 7, then seed 123.
You can stop after the first four runs to see the trend early.
If the laptop crashes or you stop it, just run the script again:
finished runs are skipped and an interrupted run resumes from its last epoch.

Usage:
    python model_ablation.py      (once, quick check)
    python train_ablation.py
"""

import csv
import statistics

import train_repeats_resume as base
import model_ablation as abl


# ------------------------------------------------------------
# Ablation configuration
# ------------------------------------------------------------

ABLATION_MODELS = {
    # name in summary     builder                        run-folder prefix
    "segformer_dec64":   (abl.build_segformer_dec64,   "ablation_segformer_b0_dec64"),
    "sfaf_add":          (abl.build_sfaf_add,          "ablation_sfaf_add"),
    "sfaf_concat":       (abl.build_sfaf_concat,       "ablation_sfaf_concat"),
    "sfaf_noreinject":   (abl.build_sfaf_noreinject,   "ablation_sfaf_noreinject"),
}

SEEDS = [42, 7, 123]

EXPERIMENTS = [(name, seed) for seed in SEEDS for name in ABLATION_MODELS]


def get_experiment(experiment, seed):
    builder, prefix = ABLATION_MODELS[experiment]
    return builder, f"{prefix}_seed{seed}"


# Point the shared training code at the ablation models and a separate summary
# file, so repeat_runs_summary.csv from the paper is never touched.
base.get_experiment = get_experiment
base.EXPERIMENTS = EXPERIMENTS
base.SUMMARY_FILE = base.RUNS_DIR / "ablation_summary.csv"


# ------------------------------------------------------------
# Speed-up: in-memory cache of preprocessed images
# ------------------------------------------------------------
# The original dataset re-reads every full-resolution image (1440x1440 / 4K)
# from disk and re-letterboxes it in every epoch. Here each image is loaded and
# letterboxed ONCE with the same letterbox() function, stored as uint8 in RAM
# (about 2 GB for all 1,195 images), and reused by all epochs and all 12 runs.
# The tensors given to the model are bit-identical to the original pipeline,
# so results stay directly comparable with the paper's runs.

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize

_CACHE = {}


def _cached_getitem(self, index):
    record = self.records[index]
    key = record["image"]
    item = _CACHE.get(key)

    if item is None:
        with Image.open(base.IMAGE_DIR / record["image"]) as img:
            image = img.convert("RGB")
        with Image.open(base.MASK_DIR / record["mask"]) as m:
            mask = m.convert("L")
        image, mask = base.letterbox(image, mask, base.IMAGE_SIZE)
        item = (np.asarray(image, dtype=np.uint8).copy(), mask)
        _CACHE[key] = item

    image_array, mask = item
    image = normalize(to_tensor(image_array), mean=base.MEAN, std=base.STD)
    return image, torch.from_numpy(mask.astype(np.int64))


base.BlueberryDataset.__getitem__ = _cached_getitem


# ------------------------------------------------------------
# Fix for resuming: RNG states must be CPU ByteTensors
# ------------------------------------------------------------
# train_repeats_resume.py loads last_state.pt with map_location="cuda", which
# moves the saved random-number states to the GPU; torch.set_rng_state() then
# fails. Move them back to the CPU before restoring.

import random


def _restore_rng_state(state, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


base.restore_rng_state = _restore_rng_state


def print_summary():
    if not base.SUMMARY_FILE.exists():
        return

    results = {}
    with open(base.SUMMARY_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.setdefault(row["experiment"], []).append(row)

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (mean +/- sample SD over completed seeds)")
    print("=" * 78)
    print(f"{'Variant':<18}{'Seeds':>6}{'Val IoU (%)':>18}{'Test IoU (%)':>18}{'Params (M)':>12}")

    for name in ABLATION_MODELS:
        rows = results.get(name, [])
        if not rows:
            continue
        val = [float(r["best_val_iou"]) * 100 for r in rows]
        test = [float(r["test_iou"]) * 100 for r in rows]
        params = float(rows[0]["parameters_m"])

        def fmt(v):
            return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

        print(f"{name:<18}{len(rows):>6}{fmt(val):>18}{fmt(test):>18}{params:>12.3f}")

    print("\nReference from the paper (3 seeds):")
    print("  SegFormer-B0 (256-ch)  Test IoU 75.19 +/- 0.37   3.715 M")
    print("  SFAF (gate)            Test IoU 74.20 +/- 1.16   3.394 M")


def main():
    base.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SFAF ABLATION RUNS (Step 1)")
    print("=" * 78)
    print(f"Summary file: {base.SUMMARY_FILE}")
    print("Planned runs:")
    for name, seed in EXPERIMENTS:
        print(f"  {name:<18} seed {seed}")

    records = base.load_splits()
    print(f"\nTrain={len(records['train'])}, Val={len(records['val'])}, Test={len(records['test'])}")

    completed = base.read_completed_runs()

    for name, seed in EXPERIMENTS:
        if (name, seed) in completed:
            print(f"\nSKIPPING (already done): {name} seed {seed}")
            continue
        base.run_full_experiment(name, seed, records)
        print_summary()

    print_summary()
    print("\nALL ABLATION RUNS COMPLETE")


if __name__ == "__main__":
    main()
