"""
Phase 1 diagnostics - no training, uses existing seed-42 checkpoints.

1. MACs/FLOPs recount with PyTorch's built-in counter (torch.utils.flop_counter),
   reported both as GMACs (convention used by most segmentation papers) and
   GFLOPs (= 2 x MACs).
2. Decision-threshold check: the paper uses argmax (= threshold 0.5 on the
   blueberry probability). The best threshold is chosen on VALIDATION only,
   then applied to TEST. A large gain means the weighted loss makes the model
   over-predict blueberries.
3. Test IoU broken down by blueberry coverage (<1%, 1-3%, 3-7%, >=7%):
   number of images, share of all test blueberry pixels, pooled IoU and
   mean per-image IoU, plus overall per-image mean IoU.

Usage:
    python diagnostics.py
Results: printed, and saved to runs/diagnostics_*.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from tqdm.auto import tqdm

import train_repeats_resume as base
import model_ablation as abl
from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


DEVICE = base.DEVICE
RUNS_DIR = base.RUNS_DIR
IGNORE = base.IGNORE_INDEX
BINS = 200  # probability bins for the threshold sweep

MODELS = [
    ("SegFormer-B0 (256-ch)", build_baseline, "baseline_segformer_b0"),
    ("SegFormer-B0 (64-ch)", abl.build_segformer_dec64, "ablation_segformer_b0_dec64_seed42"),
    ("SFAF (gate)", build_sfaf, "proposed_sfaf_b0"),
]

GROUPS = [("<1%", 0.0, 0.01), ("1-3%", 0.01, 0.03), ("3-7%", 0.03, 0.07), (">=7%", 0.07, 1.01)]


def load_model(builder, folder):
    model = builder(model_name="nvidia/mit-b0", num_classes=2)
    ckpt = torch.load(RUNS_DIR / folder / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.to(DEVICE).eval()


def count_macs(model):
    x = torch.randn(1, 3, 640, 640, device=DEVICE)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        logits = model(pixel_values=x).logits
        F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    flops = counter.get_total_flops()
    return flops / 2 / 1e9, flops / 1e9


def collect(model, records):
    """Per-image histograms of blueberry probability, split by true class."""
    loader = DataLoader(base.BlueberryDataset(records), batch_size=4, shuffle=False, num_workers=0)
    fg_hists, bg_hists = [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(pixel_values=images).logits
            logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.softmax(logits, dim=1)[:, 1]
            idx = (prob * BINS).long().clamp_(0, BINS - 1)
            for i in range(images.shape[0]):
                lab, b = labels[i], idx[i]
                fg_hists.append(torch.bincount(b[lab == 1], minlength=BINS).cpu().numpy())
                bg_hists.append(torch.bincount(b[lab == 0], minlength=BINS).cpu().numpy())
    return np.stack(fg_hists).astype(np.int64), np.stack(bg_hists).astype(np.int64)


def counts_at(fg, bg, t_bin):
    """TP/FP/FN when predicting blueberry for bins >= t_bin (works on 1-D or 2-D arrays)."""
    tp = fg[..., t_bin:].sum(-1)
    fn = fg[..., :t_bin].sum(-1)
    fp = bg[..., t_bin:].sum(-1)
    return tp, fp, fn


def iou(tp, fp, fn):
    return tp / np.maximum(tp + fp + fn, 1)


def main():
    records = base.load_splits()
    thresholds = np.arange(1, BINS) / BINS
    default_bin = BINS // 2  # 0.5 = argmax for 2 classes

    summary, groups_out = [], []

    for label, builder, folder in MODELS:
        print("\n" + "=" * 72)
        print(label)
        print("=" * 72)
        model = load_model(builder, folder)

        gmacs, gflops = count_macs(model)
        print(f"Compute: {gmacs:.2f} GMACs = {gflops:.2f} GFLOPs (2 x MACs)")

        val_fg, val_bg = collect(model, records["val"])
        test_fg, test_bg = collect(model, records["test"])

        # Threshold chosen on validation only (pooled IoU)
        vfg, vbg = val_fg.sum(0), val_bg.sum(0)
        val_ious = [iou(*counts_at(vfg, vbg, b)) for b in range(1, BINS)]
        best_bin = int(np.argmax(val_ious)) + 1
        best_t = best_bin / BINS

        tfg, tbg = test_fg.sum(0), test_bg.sum(0)
        test_iou_05 = iou(*counts_at(tfg, tbg, default_bin))
        test_iou_best = iou(*counts_at(tfg, tbg, best_bin))
        print(f"Val-selected threshold: {best_t:.3f}")
        print(f"Test IoU at 0.5: {test_iou_05 * 100:.2f}%   at {best_t:.3f}: {test_iou_best * 100:.2f}%"
              f"   (change {100 * (test_iou_best - test_iou_05):+.2f})")

        # Per-image results and coverage groups (threshold 0.5)
        fg_px = test_fg.sum(1)
        valid_px = fg_px + test_bg.sum(1)
        coverage = fg_px / np.maximum(valid_px, 1)
        tp, fp, fn = counts_at(test_fg, test_bg, default_bin)
        per_img = iou(tp, fp, fn)
        per_img_mean = per_img.mean()
        print(f"Per-image mean IoU (0.5): {per_img_mean * 100:.2f}%   pooled: {test_iou_05 * 100:.2f}%")

        print(f"{'Coverage':<9}{'N':>5}{'% of berry px':>15}{'Pooled IoU':>12}{'Mean img IoU':>14}")
        total_fg = fg_px.sum()
        for name, lo, hi in GROUPS:
            m = (coverage >= lo) & (coverage < hi)
            share = fg_px[m].sum() / total_fg * 100
            pooled = iou(tp[m].sum(), fp[m].sum(), fn[m].sum()) * 100
            mean_img = per_img[m].mean() * 100 if m.any() else float("nan")
            print(f"{name:<9}{m.sum():>5}{share:>15.2f}{pooled:>12.2f}{mean_img:>14.2f}")
            groups_out.append({"model": label, "coverage": name, "n_images": int(m.sum()),
                               "share_of_berry_pixels_pct": share, "pooled_iou": pooled,
                               "mean_image_iou": mean_img})

        summary.append({"model": label, "gmacs": gmacs, "gflops_2x": gflops,
                        "val_best_threshold": best_t, "test_iou_t05": test_iou_05 * 100,
                        "test_iou_tbest": test_iou_best * 100,
                        "test_per_image_mean_iou": per_img_mean * 100})

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    for name, rows in [("diagnostics_summary.csv", summary), ("diagnostics_coverage.csv", groups_out)]:
        with open(RUNS_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Model':<24}{'GMACs':>7}{'GFLOPs':>8}{'Best t':>8}{'IoU@0.5':>9}{'IoU@best':>10}{'ImgIoU':>8}")
    for r in summary:
        print(f"{r['model']:<24}{r['gmacs']:>7.2f}{r['gflops_2x']:>8.2f}{r['val_best_threshold']:>8.3f}"
              f"{r['test_iou_t05']:>9.2f}{r['test_iou_tbest']:>10.2f}{r['test_per_image_mean_iou']:>8.2f}")
    print(f"\nSaved: {RUNS_DIR / 'diagnostics_summary.csv'} and diagnostics_coverage.csv")


if __name__ == "__main__":
    main()
