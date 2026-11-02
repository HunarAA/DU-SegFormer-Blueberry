"""
Step 3: training-recipe experiment (loss, optimizer/schedule, augmentation).

Model: SegFormer-B0 with a 64-channel decoder (the ablation winner).
Same video-disjoint split, 640 x 640 letterboxing and ImageNet normalization
as the paper. Images are cached in RAM after the first epoch (identical inputs).

Recipes (seed 42 first, for screening):
    r1_adamw_wce        AdamW + warmup + poly LR, paper's dynamic weighted CE
    r2_adamw_ce         same, plain (unweighted) cross-entropy
    r3_adamw_cedice     same, cross-entropy + Dice
    r4_adamw_ce_aug     r2 + augmentation
    r5_adamw_cedice_aug r3 + augmentation

Optimizer (official SegFormer recipe): AdamW, lr 6e-5 encoder / 6e-4 decoder,
weight decay 0.01 (none on norm/bias), 2-epoch linear warmup, then poly decay
(power 1.0) to 0 over 100 epochs. No early stopping, so the schedule completes;
the best validation-IoU checkpoint is still used for testing.

Augmentation: horizontal flip, random scale 0.75-1.5 with 640 crop/pad
(padding ignored), brightness/contrast/saturation jitter.

Each run is evaluated at the end with:
    test IoU at 0.5 (argmax), test IoU at the validation-chosen threshold,
    per-image mean IoU, and IoU by berry-coverage group.

Usage:
    python train_recipe.py
Stop with Ctrl+C at any time; rerun to resume. Results: runs/recipe_summary.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import train_repeats_resume as base
import model_ablation as abl
import diagnostics as diag


# ------------------------------------------------------------
# Experiments
# ------------------------------------------------------------

RECIPES = {
    "r1_adamw_wce":        {"loss": "wce",     "augment": False},
    "r2_adamw_ce":         {"loss": "ce",      "augment": False},
    "r3_adamw_cedice":     {"loss": "ce_dice", "augment": False},
    "r4_adamw_ce_aug":     {"loss": "ce",      "augment": True},
    "r5_adamw_cedice_aug": {"loss": "ce_dice", "augment": True},
    # Confirmation: final recipe (r5) with the standard 256-channel decoder
    "r5_dec256":           {"loss": "ce_dice", "augment": True, "decoder": 256},
}

EXPERIMENTS = (
    # Screening (seed 42) - already finished, skipped automatically
    [(name, 42) for name in ["r1_adamw_wce", "r2_adamw_ce", "r3_adamw_cedice",
                             "r4_adamw_ce_aug", "r5_adamw_cedice_aug"]]
    # Confirmation: final recipe, 3 seeds, 64- and 256-channel decoders
    + [("r5_adamw_cedice_aug", 7), ("r5_adamw_cedice_aug", 123),
       ("r5_dec256", 42), ("r5_dec256", 7), ("r5_dec256", 123)]
)

BUILD_MODEL = abl.build_segformer_dec64


def build_for(cfg):
    if cfg.get("decoder") == 256:
        from model_baseline import build_model as build_baseline
        return build_baseline
    return BUILD_MODEL

EPOCHS = 100
WARMUP_EPOCHS = 2
BATCH_SIZE = 4
ENCODER_LR = 6e-5
HEAD_LR_MULT = 10.0
WEIGHT_DECAY = 0.01

SCALE_RANGE = (0.75, 1.5)

DEVICE = base.DEVICE
USE_AMP = base.USE_AMP
IGNORE = base.IGNORE_INDEX
SIZE = base.IMAGE_SIZE
RUNS_DIR = base.RUNS_DIR
SUMMARY_FILE = RUNS_DIR / "recipe_summary.csv"
MEAN = torch.tensor(base.MEAN).view(3, 1, 1)
STD = torch.tensor(base.STD).view(3, 1, 1)


# ------------------------------------------------------------
# Data (cached letterboxed images, optional augmentation)
# ------------------------------------------------------------

_CACHE = {}


def load_cached(record):
    key = record["image"]
    if key not in _CACHE:
        with Image.open(base.IMAGE_DIR / record["image"]) as img:
            image = img.convert("RGB")
        with Image.open(base.MASK_DIR / record["mask"]) as m:
            mask = m.convert("L")
        image, mask = base.letterbox(image, mask, SIZE)
        _CACHE[key] = (
            torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1),
            torch.from_numpy(mask.astype(np.int64)),
        )
    return _CACHE[key]


def augment(image, mask):
    """image: float CHW in [0,1]; mask: HW int64 with 255 = ignore."""
    if random.random() < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)

    scale = random.uniform(*SCALE_RANGE)
    new = int(round(SIZE * scale))
    image = F.interpolate(image[None], size=(new, new), mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(mask[None, None].float(), size=(new, new), mode="nearest")[0, 0].long()

    if new >= SIZE:
        top, left = random.randint(0, new - SIZE), random.randint(0, new - SIZE)
        image = image[:, top:top + SIZE, left:left + SIZE]
        mask = mask[top:top + SIZE, left:left + SIZE]
    else:
        pad_t, pad_l = random.randint(0, SIZE - new), random.randint(0, SIZE - new)
        canvas = torch.zeros(3, SIZE, SIZE)
        mcanvas = torch.full((SIZE, SIZE), IGNORE, dtype=torch.long)
        canvas[:, pad_t:pad_t + new, pad_l:pad_l + new] = image
        mcanvas[pad_t:pad_t + new, pad_l:pad_l + new] = mask
        image, mask = canvas, mcanvas

    # photometric jitter
    b = random.uniform(0.8, 1.2)
    c = random.uniform(0.8, 1.2)
    s = random.uniform(0.8, 1.2)
    image = image * b
    mean = image.mean()
    image = (image - mean) * c + mean
    gray = image.mean(0, keepdim=True)
    image = (image - gray) * s + gray
    return image.clamp(0, 1), mask


class RecipeDataset(Dataset):
    def __init__(self, records, do_augment):
        self.records = records
        self.do_augment = do_augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_u8, mask = load_cached(self.records[index])
        image = image_u8.float() / 255.0
        if self.do_augment:
            image, mask = augment(image, mask.clone())
        return (image - MEAN) / STD, mask


# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

def dice_loss(logits, labels):
    valid = labels != IGNORE
    prob = torch.softmax(logits.float(), dim=1)[:, 1][valid]
    target = (labels[valid] == 1).float()
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + 1.0) / (prob.sum() + target.sum() + 1.0)


def compute_loss(kind, logits, labels):
    if kind == "wce":
        return base.dynamic_weighted_ce(logits, labels)
    ce = F.cross_entropy(logits, labels, ignore_index=IGNORE)
    if kind == "ce":
        return ce
    if kind == "ce_dice":
        return ce + dice_loss(logits, labels)
    raise ValueError(kind)


# ------------------------------------------------------------
# Optimizer and schedule
# ------------------------------------------------------------

def make_optimizer(model):
    groups = {"enc_decay": [], "enc_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if name.startswith("decode_head") else "enc"
        decay = "nodecay" if p.ndim <= 1 else "decay"
        groups[f"{part}_{decay}"].append(p)

    head_lr = ENCODER_LR * HEAD_LR_MULT
    return torch.optim.AdamW([
        {"params": groups["enc_decay"], "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
        {"params": groups["enc_nodecay"], "lr": ENCODER_LR, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": WEIGHT_DECAY},
        {"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.999))


def make_scheduler(optimizer, steps_per_epoch):
    total = EPOCHS * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch

    def factor(step):
        if step < warm:
            return 1e-3 + (1 - 1e-3) * step / warm
        return max(0.0, 1.0 - (step - warm) / max(1, total - warm)) ** 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ------------------------------------------------------------
# Train / evaluate
# ------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, scaler, loss_kind, desc):
    model.train()
    total, n = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = compute_loss(loss_kind, logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def val_iou(model, loader):
    model.eval()
    metrics = base.SegmentationMetrics()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(pixel_values=images).logits
            logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            metrics.update(logits.argmax(1), labels)
    return metrics.compute()["iou"]


def final_evaluation(model, records):
    """Pooled IoU at 0.5 and at the val-chosen threshold, per-image IoU, coverage groups."""
    val_fg, val_bg = diag.collect(model, records["val"])
    test_fg, test_bg = diag.collect(model, records["test"])
    B = diag.BINS

    vfg, vbg = val_fg.sum(0), val_bg.sum(0)
    best_bin = int(np.argmax([diag.iou(*diag.counts_at(vfg, vbg, b)) for b in range(1, B)])) + 1

    tfg, tbg = test_fg.sum(0), test_bg.sum(0)
    tp, fp, fn = diag.counts_at(tfg, tbg, B // 2)
    result = {
        "test_iou_t05": diag.iou(tp, fp, fn) * 100,
        "test_dice_t05": 2 * tp / max(2 * tp + fp + fn, 1) * 100,
        "test_precision_t05": tp / max(tp + fp, 1) * 100,
        "test_sensitivity_t05": tp / max(tp + fn, 1) * 100,
        "val_best_threshold": best_bin / B,
        "test_iou_tbest": diag.iou(*diag.counts_at(tfg, tbg, best_bin)) * 100,
    }

    fg_px = test_fg.sum(1)
    coverage = fg_px / np.maximum(fg_px + test_bg.sum(1), 1)
    tp_i, fp_i, fn_i = diag.counts_at(test_fg, test_bg, B // 2)
    per_img = diag.iou(tp_i, fp_i, fn_i)
    result["test_per_image_iou"] = per_img.mean() * 100
    for name, lo, hi in diag.GROUPS:
        m = (coverage >= lo) & (coverage < hi)
        key = name.replace("<", "lt").replace(">=", "ge").replace("%", "").replace("-", "_")
        result[f"img_iou_{key}"] = per_img[m].mean() * 100 if m.any() else float("nan")
    return result


# ------------------------------------------------------------
# Run management (resume support)
# ------------------------------------------------------------

def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    generator.set_state(state["loader_generator"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def completed_runs():
    if not SUMMARY_FILE.exists():
        return set()
    with open(SUMMARY_FILE, encoding="utf-8") as f:
        return {(r["recipe"], int(r["seed"])) for r in csv.DictReader(f)}


def append_summary(row):
    exists = SUMMARY_FILE.exists()
    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


def run(recipe_name, seed, records):
    cfg = RECIPES[recipe_name]
    base.set_seed(seed)
    run_dir = RUNS_DIR / f"recipe_{recipe_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, state_path = run_dir / "best_model.pt", run_dir / "last_state.pt"

    print("\n" + "=" * 78)
    print(f"RECIPE {recipe_name} | loss={cfg['loss']} | augment={cfg['augment']} | seed {seed}")
    print("=" * 78)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(RecipeDataset(records["train"], cfg["augment"]), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=USE_AMP, generator=generator)
    val_loader = DataLoader(RecipeDataset(records["val"], False), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=USE_AMP)

    model = build_for(cfg)(model_name="nvidia/mit-b0", num_classes=2).to(DEVICE)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    start_epoch, best_iou, best_epoch, history, elapsed = 1, -1.0, 0, [], 0.0
    if state_path.exists():
        s = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        start_epoch, best_iou, best_epoch = s["epoch"] + 1, s["best_iou"], s["best_epoch"]
        history, elapsed = s["history"], s["elapsed"]
        restore_rng(s["rng"], generator)
        print(f"Resuming after epoch {s['epoch']} (best val IoU {best_iou * 100:.2f}% at epoch {best_epoch})")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        te = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, cfg["loss"],
                           f"{recipe_name} s{seed} E{epoch:03d}")
        v = val_iou(model, val_loader)
        history.append({"epoch": epoch, "train_loss": loss, "val_iou": v,
                        "encoder_lr": optimizer.param_groups[0]["lr"]})

        msg = f"Epoch {epoch:03d}/{EPOCHS} ({(time.time() - te) / 60:.1f} min) loss={loss:.4f} val IoU={v * 100:.2f}%"
        if v > best_iou:
            best_iou, best_epoch = v, epoch
            base.atomic_torch_save({"model_state_dict": model.state_dict(), "epoch": epoch,
                                    "recipe": recipe_name, "seed": seed}, best_path)
            msg += "  BEST"
        print(msg)

        base.atomic_torch_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_iou": best_iou, "best_epoch": best_epoch, "history": history,
            "elapsed": elapsed + time.time() - t0, "rng": base.get_rng_state(generator),
        }, state_path)

    base.save_history(history, run_dir / "history.csv")

    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"])
    model.eval()
    result = final_evaluation(model, records)

    row = {"recipe": recipe_name, "seed": seed, "loss": cfg["loss"], "augment": cfg["augment"],
           "best_epoch": best_epoch, "best_val_iou": best_iou * 100, **result,
           "runtime_minutes": (elapsed + time.time() - t0) / 60}
    append_summary(row)

    print(f"\nBest epoch {best_epoch} | Test IoU@0.5 {result['test_iou_t05']:.2f} | "
          f"IoU@val-threshold({result['val_best_threshold']:.3f}) {result['test_iou_tbest']:.2f} | "
          f"per-image IoU {result['test_per_image_iou']:.2f}")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_summary():
    if not SUMMARY_FILE.exists():
        return
    with open(SUMMARY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("\n" + "=" * 78)
    print("RECIPE SUMMARY (SegFormer-B0, 64-ch decoder)")
    print("=" * 78)
    print(f"{'Recipe':<22}{'Seed':>5}{'IoU@0.5':>9}{'Thr':>7}{'IoU@thr':>9}{'ImgIoU':>8}{'<1%':>7}{'1-3%':>7}")
    for r in rows:
        print(f"{r['recipe']:<22}{r['seed']:>5}{float(r['test_iou_t05']):>9.2f}{float(r['val_best_threshold']):>7.3f}"
              f"{float(r['test_iou_tbest']):>9.2f}{float(r['test_per_image_iou']):>8.2f}"
              f"{float(r['img_iou_lt1']):>7.2f}{float(r['img_iou_1_3']):>7.2f}")
    print("\nReference, paper recipe (64-ch, seed 42): IoU@0.5 73.81 | IoU@thr 78.42 | ImgIoU 67.28 | <1% 38.11 | 1-3% 54.78")

    # Mean +/- SD of the final recipe across seeds
    import statistics
    labels = {"r5_adamw_cedice_aug": "Final recipe, 64-ch", "r5_dec256": "Final recipe, 256-ch"}
    lines = []
    for key, label in labels.items():
        sel = [r for r in rows if r["recipe"] == key]
        if not sel:
            continue

        def ms(col):
            v = [float(r[col]) for r in sel]
            return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

        lines.append(f"{label:<22}{len(sel):>6}{ms('test_iou_t05'):>17}{ms('test_per_image_iou'):>17}")
    if lines:
        print("\n" + "=" * 78)
        print("FINAL RECIPE ACROSS SEEDS")
        print("=" * 78)
        print(f"{'Model':<22}{'Seeds':>6}{'IoU@0.5':>17}{'Per-image IoU':>17}")
        print("\n".join(lines))
        print("Paper recipe (3 seeds): 64-ch 74.18 +/- 0.34 | 256-ch 75.19 +/- 0.37")


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    records = base.load_splits()
    print(f"Train={len(records['train'])}, Val={len(records['val'])}, Test={len(records['test'])}")
    done = completed_runs()
    for name, seed in EXPERIMENTS:
        if (name, seed) in done:
            print(f"SKIPPING (done): {name} seed {seed}")
            continue
        run(name, seed, records)
        print_summary()
    print_summary()
    print("\nALL RECIPE RUNS COMPLETE")


if __name__ == "__main__":
    main()
