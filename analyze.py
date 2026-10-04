from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train import BlueberryDataset, load_splits, DEVICE, IGNORE_INDEX, MEAN, STD
from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\Papers\AgriVision Blueberry")
MODEL_NAME = "nvidia/mit-b0"
SPLIT_TO_ANALYZE = "test"

BASELINE_CHECKPOINT = (
    PROJECT_DIR / "runs" / "baseline_segformer_b0" / "best_model.pt"
)
SFAF_CHECKPOINT = (
    PROJECT_DIR / "runs" / "proposed_sfaf_b0" / "best_model.pt"
)

OUTPUT_CSV = PROJECT_DIR / "runs" / "test_baseline_vs_sfaf.csv"
OUTPUT_IMAGE = PROJECT_DIR / "runs" / "test_baseline_vs_sfaf_cases.png"

BOUNDARY_TOLERANCE = 2
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
NUM_CASES = 4


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(builder, checkpoint_path, label):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found:\n{checkpoint_path}")

    print(f"Loading {label}...")

    model = builder(model_name=MODEL_NAME, num_classes=2)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"  Best epoch: {checkpoint['epoch']}")

    if isinstance(checkpoint, dict) and "val_metrics" in checkpoint:
        vm = checkpoint["val_metrics"]
        if "iou" in vm:
            print(f"  Saved Val IoU: {vm['iou'] * 100:.2f}%")

    return model


# ============================================================
# METRICS
# ============================================================

def confusion_counts(prediction, target):
    valid = target != IGNORE_INDEX
    pred = prediction[valid]
    gt = target[valid]

    tp = ((pred == 1) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()

    valid_pixels = valid.sum().item()
    foreground_pixels = (gt == 1).sum().item()

    return tp, tn, fp, fn, valid_pixels, foreground_pixels


def metrics_from_counts(tp, tn, fp, fn):
    eps = 1e-8

    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
    }


def get_boundary(mask):
    mask = mask.float()[None, None]

    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)

    return (dilated - eroded) > 0


def boundary_f1(prediction, target, tolerance=2):
    valid = target != IGNORE_INDEX

    rows = torch.where(valid.any(dim=1))[0]
    cols = torch.where(valid.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return 0.0

    y1, y2 = rows[0], rows[-1] + 1
    x1, x2 = cols[0], cols[-1] + 1

    pred = (prediction[y1:y2, x1:x2] == 1).float()
    gt = (target[y1:y2, x1:x2] == 1).float()

    pred_boundary = get_boundary(pred)
    gt_boundary = get_boundary(gt)

    kernel_size = 2 * tolerance + 1

    pred_dilated = F.max_pool2d(
        pred_boundary.float(), kernel_size, stride=1, padding=tolerance
    ) > 0

    gt_dilated = F.max_pool2d(
        gt_boundary.float(), kernel_size, stride=1, padding=tolerance
    ) > 0

    matched_pred = (pred_boundary & gt_dilated).sum().item()
    matched_gt = (gt_boundary & pred_dilated).sum().item()

    pred_total = pred_boundary.sum().item()
    gt_total = gt_boundary.sum().item()

    eps = 1e-8
    precision = matched_pred / (pred_total + eps)
    recall = matched_gt / (gt_total + eps)

    return 2 * precision * recall / (precision + recall + eps)


# ============================================================
# HELPERS
# ============================================================

def coverage_group(foreground_percent):
    if foreground_percent < 1:
        return "<1%"
    if foreground_percent < 3:
        return "1-3%"
    if foreground_percent < 7:
        return "3-7%"
    return ">=7%"


def restore_image(image_tensor):
    image = image_tensor.detach().cpu().clone()
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def predict(model, images, target_size):
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda"),
        ):
            output = model(pixel_values=images)
            logits = F.interpolate(
                output.logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

    return torch.argmax(logits, dim=1)


# ============================================================
# VIDEO-GROUPED BOOTSTRAP
# ============================================================

def grouped_bootstrap_difference(rows, baseline_key, sfaf_key):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["camera"], row["video"])].append(row)

    groups = list(grouped.keys())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)

    for i in range(BOOTSTRAP_ITERATIONS):
        sampled_indices = rng.choice(
            len(groups),
            size=len(groups),
            replace=True,
        )

        base_values = []
        sfaf_values = []

        for index in sampled_indices:
            group = groups[index]
            for row in grouped[group]:
                base_values.append(row[baseline_key])
                sfaf_values.append(row[sfaf_key])

        differences[i] = np.mean(sfaf_values) - np.mean(base_values)

    return {
        "lower": np.percentile(differences, 2.5),
        "upper": np.percentile(differences, 97.5),
        "prob_positive": np.mean(differences > 0),
        "groups": len(groups),
    }


# ============================================================
# REPORTING
# ============================================================

def print_per_image_summary(rows):
    print("\n" + "=" * 96)
    print("PER-IMAGE TEST SUMMARY")
    print("=" * 96)

    for label, prefix in [("Baseline", "baseline"), ("SFAF", "sfaf")]:
        iou = np.array([r[f"{prefix}_iou"] for r in rows])
        dice = np.array([r[f"{prefix}_dice"] for r in rows])
        bf1 = np.array([r[f"{prefix}_boundary_f1"] for r in rows])

        print(
            f"{label:<10} "
            f"Mean IoU={iou.mean():6.2f}% | "
            f"Median IoU={np.median(iou):6.2f}% | "
            f"Mean Dice={dice.mean():6.2f}% | "
            f"Mean BF1={bf1.mean():6.2f}%"
        )

    delta_iou = np.array([r["delta_iou"] for r in rows])
    delta_dice = np.array([r["delta_dice"] for r in rows])
    delta_bf1 = np.array([r["delta_boundary_f1"] for r in rows])

    print("\nSFAF - Baseline")
    print(f"Mean IoU change:  {delta_iou.mean():+.2f} points")
    print(f"Mean Dice change: {delta_dice.mean():+.2f} points")
    print(f"Mean BF1 change:  {delta_bf1.mean():+.2f} points")
    print(
        "Images improved/worsened/tied: "
        f"{int((delta_iou > 0).sum())}/"
        f"{int((delta_iou < 0).sum())}/"
        f"{int((delta_iou == 0).sum())}"
    )


def print_pooled_summary(pooled):
    print("\n" + "=" * 96)
    print("PIXEL-POOLED TEST METRICS")
    print("=" * 96)

    for label, prefix in [("Baseline", "baseline"), ("SFAF", "sfaf")]:
        c = pooled[prefix]
        m = metrics_from_counts(c["tp"], c["tn"], c["fp"], c["fn"])

        print(
            f"{label:<10} "
            f"IoU={m['iou'] * 100:6.2f}% | "
            f"Dice={m['dice'] * 100:6.2f}% | "
            f"Pre={m['precision'] * 100:6.2f}% | "
            f"Sen={m['sensitivity'] * 100:6.2f}% | "
            f"Spe={m['specificity'] * 100:6.2f}%"
        )


def print_group_summary(rows, key, values, title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(
        f"{'Group':<12}{'N':>5}{'Base IoU':>12}{'SFAF IoU':>12}"
        f"{'Delta':>10}{'Base BF1':>12}{'SFAF BF1':>12}{'Delta':>10}"
    )
    print("-" * 96)

    for value in values:
        subset = [r for r in rows if r[key] == value]
        if not subset:
            continue

        base_iou = np.mean([r["baseline_iou"] for r in subset])
        sfaf_iou = np.mean([r["sfaf_iou"] for r in subset])
        base_bf1 = np.mean([r["baseline_boundary_f1"] for r in subset])
        sfaf_bf1 = np.mean([r["sfaf_boundary_f1"] for r in subset])

        print(
            f"{str(value):<12}{len(subset):>5}"
            f"{base_iou:>11.2f}%{sfaf_iou:>11.2f}%{sfaf_iou-base_iou:>+9.2f}"
            f"{base_bf1:>11.2f}%{sfaf_bf1:>11.2f}%{sfaf_bf1-base_bf1:>+9.2f}"
        )


def print_video_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["camera"], row["video"])].append(row)

    differences = []

    for items in grouped.values():
        base = np.mean([r["baseline_iou"] for r in items])
        sfaf = np.mean([r["sfaf_iou"] for r in items])
        differences.append(sfaf - base)

    differences = np.array(differences)

    print("\n" + "=" * 96)
    print("SOURCE-VIDEO LEVEL SUMMARY")
    print("=" * 96)
    print(f"Video groups: {len(differences)}")
    print(f"Mean video-level IoU change:   {differences.mean():+.2f} points")
    print(f"Median video-level IoU change: {np.median(differences):+.2f} points")
    print(
        "Video groups improved/worsened: "
        f"{int((differences > 0).sum())}/{int((differences < 0).sum())}"
    )


# ============================================================
# VISUAL CASES
# ============================================================

def save_case_figure(display_items):
    improved = sorted(
        display_items,
        key=lambda x: x["row"]["delta_iou"],
        reverse=True,
    )[:NUM_CASES]

    worsened = sorted(
        display_items,
        key=lambda x: x["row"]["delta_iou"],
    )[:NUM_CASES]

    selected = [("Improved", x) for x in improved] + [
        ("Worsened", x) for x in worsened
    ]

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(14, 3 * len(selected)),
    )

    for i, (category, item) in enumerate(selected):
        row = item["row"]
        target = item["target"]
        valid = target != IGNORE_INDEX

        gt = np.where(valid, target, 0)
        base = np.where(valid, item["baseline"], 0)
        sfaf = np.where(valid, item["sfaf"], 0)

        axes[i, 0].imshow(item["image"])
        axes[i, 0].set_title(
            f"{category}: {row['name']}\nFG={row['foreground_percent']:.2f}%"
        )

        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Baseline {row['baseline_iou']:.1f}%")

        axes[i, 3].imshow(sfaf, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title(
            f"SFAF {row['sfaf_iou']:.1f}% ({row['delta_iou']:+.1f})"
        )

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 96)
    print("FINAL TEST ANALYSIS: BASELINE vs SFAF")
    print("=" * 96)
    print(f"Device: {DEVICE}")

    records = load_splits()
    test_records = records[SPLIT_TO_ANALYZE]

    print(f"Test images: {len(test_records)}")

    baseline = load_model(
        build_baseline,
        BASELINE_CHECKPOINT,
        "Baseline",
    )

    sfaf = load_model(
        build_sfaf,
        SFAF_CHECKPOINT,
        "SFAF",
    )

    dataset = BlueberryDataset(test_records)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    display_items = []

    pooled = {
        "baseline": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "sfaf": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }

    print("\nRunning final descriptive test analysis...")

    for index, (images, labels) in tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Test analysis",
        unit="image",
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        target = labels[0]

        baseline_pred = predict(
            baseline,
            images,
            labels.shape[-2:],
        )[0]

        sfaf_pred = predict(
            sfaf,
            images,
            labels.shape[-2:],
        )[0]

        b = confusion_counts(baseline_pred, target)
        s = confusion_counts(sfaf_pred, target)

        baseline_metrics = metrics_from_counts(*b[:4])
        sfaf_metrics = metrics_from_counts(*s[:4])

        baseline_bf1 = boundary_f1(
            baseline_pred,
            target,
            BOUNDARY_TOLERANCE,
        )

        sfaf_bf1 = boundary_f1(
            sfaf_pred,
            target,
            BOUNDARY_TOLERANCE,
        )

        foreground_percent = b[5] / max(b[4], 1) * 100
        source = test_records[index]

        row = {
            "name": source["name"],
            "camera": int(source["camera"]),
            "video": int(source["video"]),
            "frame": int(source["frame"]),
            "foreground_percent": foreground_percent,
            "coverage_group": coverage_group(foreground_percent),
            "baseline_iou": baseline_metrics["iou"] * 100,
            "baseline_dice": baseline_metrics["dice"] * 100,
            "baseline_precision": baseline_metrics["precision"] * 100,
            "baseline_sensitivity": baseline_metrics["sensitivity"] * 100,
            "baseline_specificity": baseline_metrics["specificity"] * 100,
            "baseline_boundary_f1": baseline_bf1 * 100,
            "sfaf_iou": sfaf_metrics["iou"] * 100,
            "sfaf_dice": sfaf_metrics["dice"] * 100,
            "sfaf_precision": sfaf_metrics["precision"] * 100,
            "sfaf_sensitivity": sfaf_metrics["sensitivity"] * 100,
            "sfaf_specificity": sfaf_metrics["specificity"] * 100,
            "sfaf_boundary_f1": sfaf_bf1 * 100,
        }

        row["delta_iou"] = row["sfaf_iou"] - row["baseline_iou"]
        row["delta_dice"] = row["sfaf_dice"] - row["baseline_dice"]
        row["delta_boundary_f1"] = (
            row["sfaf_boundary_f1"] - row["baseline_boundary_f1"]
        )

        rows.append(row)

        display_items.append(
            {
                "row": row,
                "image": restore_image(images[0]),
                "target": target.detach().cpu().numpy(),
                "baseline": baseline_pred.detach().cpu().numpy(),
                "sfaf": sfaf_pred.detach().cpu().numpy(),
            }
        )

        for key, counts in [("baseline", b), ("sfaf", s)]:
            pooled[key]["tp"] += counts[0]
            pooled[key]["tn"] += counts[1]
            pooled[key]["fp"] += counts[2]
            pooled[key]["fn"] += counts[3]

    # Save detailed per-image results.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print_pooled_summary(pooled)
    print_per_image_summary(rows)

    print_group_summary(
        rows,
        "coverage_group",
        ["<1%", "1-3%", "3-7%", ">=7%"],
        "PERFORMANCE BY BLUEBERRY FOREGROUND COVERAGE",
    )

    print_group_summary(
        rows,
        "camera",
        [1, 2, 3, 4],
        "PERFORMANCE BY CAMERA",
    )

    print_video_summary(rows)

    iou_ci = grouped_bootstrap_difference(
        rows,
        "baseline_iou",
        "sfaf_iou",
    )

    dice_ci = grouped_bootstrap_difference(
        rows,
        "baseline_dice",
        "sfaf_dice",
    )

    bf1_ci = grouped_bootstrap_difference(
        rows,
        "baseline_boundary_f1",
        "sfaf_boundary_f1",
    )

    print("\n" + "=" * 96)
    print("VIDEO-GROUPED BOOTSTRAP: SFAF - BASELINE")
    print("=" * 96)
    print(f"Source video groups: {iou_ci['groups']}")
    print(
        f"IoU  95% CI: [{iou_ci['lower']:+.2f}, {iou_ci['upper']:+.2f}] points "
        f"| P(Delta>0)={iou_ci['prob_positive']:.3f}"
    )
    print(
        f"Dice 95% CI: [{dice_ci['lower']:+.2f}, {dice_ci['upper']:+.2f}] points "
        f"| P(Delta>0)={dice_ci['prob_positive']:.3f}"
    )
    print(
        f"BF1  95% CI: [{bf1_ci['lower']:+.2f}, {bf1_ci['upper']:+.2f}] points "
        f"| P(Delta>0)={bf1_ci['prob_positive']:.3f}"
    )

    save_case_figure(display_items)

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_IMAGE)

    print("\nIMPORTANT:")
    print("This is final descriptive test analysis.")
    print("Do not use these test findings to redesign or tune SFAF.")


if __name__ == "__main__":
    main()
