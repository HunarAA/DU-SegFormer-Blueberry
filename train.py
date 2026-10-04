from pathlib import Path
import csv
import random
import time

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, normalize


# ============================================================
# EXPERIMENT SELECTION
# ============================================================

# Options:
#   "baseline"
#   "sfaf"
#   "sfaf_aux"
EXPERIMENT = "sfaf_aux"

MODEL_NAME = "nvidia/mit-b0"

if EXPERIMENT == "baseline":
    from model_baseline import build_model
    RUN_NAME = "baseline_segformer_b0"
    AUX_LOSS_WEIGHT = 0.0

elif EXPERIMENT == "sfaf":
    from model_sfaf import build_model
    RUN_NAME = "proposed_sfaf_b0"
    AUX_LOSS_WEIGHT = 0.0

elif EXPERIMENT == "sfaf_aux":
    from model_sfaf_aux import build_model
    RUN_NAME = "proposed_sfaf_aux_b0"
    AUX_LOSS_WEIGHT = 0.30

else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"E:\Papers\AgriVision Blueberry")
DATASET_DIR = PROJECT_DIR / "DB-1"
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"


# ============================================================
# SMOKE / FULL TRAINING
# ============================================================

# True  -> 10 epochs, validation only, no test evaluation.
# False -> 100 epochs, final test once after model selection.
SMOKE_TEST = False
SMOKE_EPOCHS = 10

if SMOKE_TEST:
    MAX_EPOCHS = SMOKE_EPOCHS
    RUN_FINAL_TEST = False
    RUN_DIR = PROJECT_DIR / "runs" / f"{RUN_NAME}_smoke"
else:
    MAX_EPOCHS = 100
    RUN_FINAL_TEST = True
    RUN_DIR = PROJECT_DIR / "runs" / RUN_NAME


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 640
BATCH_SIZE = 4
PATIENCE = 15
LEARNING_RATE = 1e-4
SEED = 42
NUM_WORKERS = 0
IGNORE_INDEX = 255

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_AMP = DEVICE.type == "cuda"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PREPROCESSING
# ============================================================

def letterbox(image, mask, size):
    width, height = image.size
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = image.resize((new_width, new_height), BILINEAR)
    mask = mask.resize((new_width, new_height), NEAREST)

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    image_canvas = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0),
    )
    image_canvas.paste(image, (left, top))

    mask_array = (np.array(mask) > 127).astype(np.uint8)

    mask_canvas = np.full(
        (size, size),
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    mask_canvas[
        top:top + new_height,
        left:left + new_width,
    ] = mask_array

    return image_canvas, mask_canvas


class BlueberryDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        with Image.open(IMAGE_DIR / record["image"]) as img:
            image = img.convert("RGB")

        with Image.open(MASK_DIR / record["mask"]) as m:
            mask = m.convert("L")

        image, mask = letterbox(image, mask, IMAGE_SIZE)

        image = normalize(
            to_tensor(image),
            mean=MEAN,
            std=STD,
        )

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


# ============================================================
# SPLITS
# ============================================================

def load_splits():
    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"].strip()

            if split not in records:
                raise ValueError(f"Unknown split: {split}")

            records[split].append(row)

    return records


# ============================================================
# LOSS
# ============================================================

def dynamic_weighted_ce(logits, labels):
    valid = labels != IGNORE_INDEX
    valid_labels = labels[valid]

    counts = torch.bincount(
        valid_labels,
        minlength=2,
    ).float()

    frequencies = counts / counts.sum().clamp_min(1.0)
    inverse = 1.0 / frequencies.clamp_min(1e-6)
    weights = inverse / inverse.sum()

    return F.cross_entropy(
        logits,
        labels,
        weight=weights,
        ignore_index=IGNORE_INDEX,
    )


# ============================================================
# METRICS
# ============================================================

class SegmentationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, predictions, targets):
        valid = targets != IGNORE_INDEX
        predictions = predictions.detach()[valid]
        targets = targets.detach()[valid]

        self.tp += (((predictions == 1) & (targets == 1)).sum().item())
        self.tn += (((predictions == 0) & (targets == 0)).sum().item())
        self.fp += (((predictions == 1) & (targets == 0)).sum().item())
        self.fn += (((predictions == 0) & (targets == 1)).sum().item())

    def compute(self):
        eps = 1e-8
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn

        return {
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "sensitivity": tp / (tp + fn + eps),
            "specificity": tn / (tn + fp + eps),
        }


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None,
    scaler=None,
    description="",
):
    training = optimizer is not None
    model.train() if training else model.eval()

    metrics = SegmentationMetrics()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):
                outputs = model(pixel_values=images)

                logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                main_loss = dynamic_weighted_ce(logits, labels)
                aux_loss = torch.tensor(0.0, device=DEVICE)

                aux_logits = getattr(outputs, "aux_logits", None)

                if (
                    training
                    and aux_logits is not None
                    and AUX_LOSS_WEIGHT > 0
                ):
                    aux_logits = F.interpolate(
                        aux_logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    aux_loss = dynamic_weighted_ce(
                        aux_logits,
                        labels,
                    )

                loss = (
                    main_loss + AUX_LOSS_WEIGHT * aux_loss
                    if training
                    else main_loss
                )

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = torch.argmax(logits, dim=1)
        metrics.update(predictions, labels)

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_batches += 1

        current = metrics.compute()
        postfix = {
            "loss": f"{total_loss / total_batches:.4f}",
            "IoU": f"{current['iou'] * 100:.1f}%",
        }

        if training and AUX_LOSS_WEIGHT > 0:
            postfix["aux"] = f"{total_aux_loss / total_batches:.4f}"

        if torch.cuda.is_available():
            postfix["GPU"] = (
                f"{torch.cuda.memory_allocated() / 1024**3:.1f}G"
            )

        progress.set_postfix(postfix)

    results = metrics.compute()
    denom = max(total_batches, 1)

    results["loss"] = total_loss / denom
    results["main_loss"] = total_main_loss / denom
    results["aux_loss"] = total_aux_loss / denom

    return results


# ============================================================
# HELPERS
# ============================================================

def print_metrics(prefix, m):
    print(
        f"{prefix:<6} "
        f"Loss={m['loss']:.4f} | "
        f"IoU={m['iou']*100:.2f}% | "
        f"Dice={m['dice']*100:.2f}% | "
        f"Pre={m['precision']*100:.2f}% | "
        f"Sen={m['sensitivity']*100:.2f}% | "
        f"Spe={m['specificity']*100:.2f}%"
    )


def save_history(history, path):
    if not history:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    print("AGRIVISION BLUEBERRY EXPERIMENT")
    print("=" * 72)
    print(f"Experiment:    {EXPERIMENT}")
    print(f"Device:        {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(
            f"GPU VRAM:      "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Image size:    {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Epochs:        {MAX_EPOCHS}")
    print(f"Patience:      {PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Aux weight:    {AUX_LOSS_WEIGHT}")
    print(f"Smoke test:    {SMOKE_TEST}")
    print("Split:         video-disjoint")

    records = load_splits()

    print()
    print(f"Train: {len(records['train'])}")
    print(f"Val:   {len(records['val'])}")
    print(f"Test:  {len(records['test'])}")

    train_dataset = BlueberryDataset(records["train"])
    val_dataset = BlueberryDataset(records["val"])
    test_dataset = BlueberryDataset(records["test"])

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )

    print("\nLoading model...")

    model = build_model(
        model_name=MODEL_NAME,
        num_classes=2,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters:    {total_params / 1e6:.3f} M")
    print(f"Trainable:     {trainable_params / 1e6:.3f} M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    checkpoint_path = RUN_DIR / "best_model.pt"
    history_path = RUN_DIR / "history.csv"

    best_iou = -1.0
    best_epoch = 0
    no_improvement = 0
    history = []
    overall_start = time.time()

    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            description=f"Epoch {epoch:03d}/{MAX_EPOCHS} Train",
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            description=f"Epoch {epoch:03d}/{MAX_EPOCHS} Val",
        )

        epoch_time = time.time() - epoch_start

        print()
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} "
            f"({epoch_time / 60:.1f} min)"
        )
        print_metrics("Train", train_metrics)

        if AUX_LOSS_WEIGHT > 0:
            print(
                f"       MainLoss={train_metrics['main_loss']:.4f} | "
                f"AuxLoss={train_metrics['aux_loss']:.4f}"
            )

        print_metrics("Val", val_metrics)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_main_loss": train_metrics["main_loss"],
            "train_aux_loss": train_metrics["aux_loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_precision": train_metrics["precision"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
        })

        save_history(history, history_path)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "experiment": EXPERIMENT,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "aux_loss_weight": AUX_LOSS_WEIGHT,
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "seed": SEED,
                },
                checkpoint_path,
            )

            print(
                f"BEST MODEL SAVED "
                f"(Val IoU={best_iou*100:.2f}%)"
            )

        else:
            no_improvement += 1
            print(
                f"No improvement: "
                f"{no_improvement}/{PATIENCE}"
            )

        elapsed = time.time() - overall_start
        average_epoch = elapsed / epoch
        estimated = average_epoch * (MAX_EPOCHS - epoch)

        print(f"Elapsed: {elapsed / 60:.1f} min")
        print(
            f"Approx. remaining without early stopping: "
            f"{estimated / 60:.1f} min"
        )

        if no_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    if not RUN_FINAL_TEST:
        print("\n" + "=" * 72)
        print("SMOKE TEST COMPLETE")
        print("=" * 72)
        print(f"Best epoch: {best_epoch}")
        print(f"Best Val IoU: {best_iou*100:.2f}%")

        if torch.cuda.is_available():
            peak_memory = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )
            print(f"Peak GPU memory: {peak_memory:.2f} GB")

        print("\nTest set was NOT used.")
        print(f"\nSmoke results saved in:\n{RUN_DIR}")
        print("\nFor full training change:")
        print("SMOKE_TEST = False")
        return

    print("\n" + "=" * 72)
    print("FINAL TEST - BEST VALIDATION MODEL")
    print("=" * 72)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        description="Final Test",
    )

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val IoU: {best_iou*100:.2f}%")
    print_metrics("Test", test_metrics)

    total_time = time.time() - overall_start
    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    if torch.cuda.is_available():
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )
        print(f"Peak GPU memory: {peak_memory:.2f} GB")

    print("\nSaved:")
    print(checkpoint_path)
    print(history_path)
    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
