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

from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


# ============================================================
# FINAL REPEATABILITY EXPERIMENT WITH RESUME SUPPORT
# ============================================================
#
# Existing seed-42 runs remain untouched:
#   runs/baseline_segformer_b0/
#   runs/proposed_sfaf_b0/
#
# This script adds:
#   baseline seeds 7 and 123
#   SFAF seeds 7 and 123
#
# Each run is stored separately.
# Training state is saved after every completed epoch and can resume automatically.
# ============================================================

from paths import PROJECT_DIR, DATASET_DIR
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"
RUNS_DIR = PROJECT_DIR / "runs"
SUMMARY_FILE = RUNS_DIR / "repeat_runs_summary.csv"

# Initial protocol (V2 of the study): SegFormer-B0 (256-ch) and SFAF, seeds 42/7/123.
EXPERIMENTS = [
    ("baseline", 42),
    ("sfaf", 42),
    ("baseline", 7),
    ("baseline", 123),
    ("sfaf", 7),
    ("sfaf", 123),
]

SKIP_COMPLETED = True

# Resume controls
RESUME_ENABLED = True
RESUME_FILE_NAME = "last_state.pt"

MODEL_NAME = "nvidia/mit-b0"
IMAGE_SIZE = 640
BATCH_SIZE = 4
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state(loader_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    loader_generator.set_state(state["loader_generator"])

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(obj, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    temp_path.replace(path)


def letterbox(image, mask, size):
    width, height = image.size
    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = image.resize(
        (new_width, new_height),
        BILINEAR,
    )

    mask = mask.resize(
        (new_width, new_height),
        NEAREST,
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    image_canvas = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0),
    )

    image_canvas.paste(
        image,
        (left, top),
    )

    mask_array = (
        np.array(mask) > 127
    ).astype(np.uint8)

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

        image_path = IMAGE_DIR / record["image"]
        mask_path = MASK_DIR / record["mask"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        with Image.open(mask_path) as m:
            mask = m.convert("L")

        image, mask = letterbox(
            image,
            mask,
            IMAGE_SIZE,
        )

        image = to_tensor(image)
        image = normalize(
            image,
            mean=MEAN,
            std=STD,
        )

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


def load_splits():
    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"].strip()

            if split not in records:
                raise ValueError(
                    f"Unknown split: {split}"
                )

            records[split].append(row)

    return records


def dynamic_weighted_ce(logits, labels):
    valid = labels != IGNORE_INDEX
    valid_labels = labels[valid]

    counts = torch.bincount(
        valid_labels,
        minlength=2,
    ).float()

    frequencies = (
        counts
        / counts.sum().clamp_min(1.0)
    )

    inverse = (
        1.0
        / frequencies.clamp_min(1e-6)
    )

    weights = inverse / inverse.sum()

    return F.cross_entropy(
        logits,
        labels,
        weight=weights,
        ignore_index=IGNORE_INDEX,
    )


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

        self.tp += (
            ((predictions == 1) & (targets == 1))
            .sum()
            .item()
        )

        self.tn += (
            ((predictions == 0) & (targets == 0))
            .sum()
            .item()
        )

        self.fp += (
            ((predictions == 1) & (targets == 0))
            .sum()
            .item()
        )

        self.fn += (
            ((predictions == 0) & (targets == 1))
            .sum()
            .item()
        )

    def compute(self):
        eps = 1e-8

        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        return {
            "iou":
                tp / (tp + fp + fn + eps),

            "dice":
                2 * tp / (2 * tp + fp + fn + eps),

            "precision":
                tp / (tp + fp + eps),

            "sensitivity":
                tp / (tp + fn + eps),

            "specificity":
                tn / (tn + fp + eps),
        }


def run_epoch(
    model,
    loader,
    optimizer=None,
    scaler=None,
    description=""
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    metrics = SegmentationMetrics()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for images, labels in progress:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):
                outputs = model(
                    pixel_values=images
                )

                logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = dynamic_weighted_ce(
                    logits,
                    labels,
                )

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        metrics.update(
            predictions,
            labels,
        )

        total_loss += loss.item()
        total_batches += 1

        current = metrics.compute()

        postfix = {
            "loss":
                f"{total_loss / total_batches:.4f}",

            "IoU":
                f"{current['iou'] * 100:.1f}%",
        }

        if torch.cuda.is_available():
            gpu = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            postfix["GPU"] = f"{gpu:.1f}G"

        progress.set_postfix(postfix)

    results = metrics.compute()

    results["loss"] = (
        total_loss
        / max(total_batches, 1)
    )

    return results


def print_metrics(prefix, metrics):
    print(
        f"{prefix:<6} "
        f"Loss={metrics['loss']:.4f} | "
        f"IoU={metrics['iou'] * 100:.2f}% | "
        f"Dice={metrics['dice'] * 100:.2f}% | "
        f"Pre={metrics['precision'] * 100:.2f}% | "
        f"Sen={metrics['sensitivity'] * 100:.2f}% | "
        f"Spe={metrics['specificity'] * 100:.2f}%"
    )


def save_history(history, path):
    if not history:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


SUMMARY_FIELDS = [
    "experiment",
    "seed",
    "best_epoch",
    "best_val_iou",
    "test_loss",
    "test_iou",
    "test_dice",
    "test_precision",
    "test_sensitivity",
    "test_specificity",
    "parameters_m",
    "peak_gpu_gb",
    "runtime_minutes",
    "run_dir",
]


def read_completed_runs():
    completed = set()

    if not SUMMARY_FILE.exists():
        return completed

    with open(
        SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            completed.add(
                (
                    row["experiment"],
                    int(row["seed"]),
                )
            )

    return completed


def append_summary(result):
    file_exists = SUMMARY_FILE.exists()

    with open(
        SUMMARY_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def get_experiment(experiment, seed):
    # Seed-42 runs use the folder names of the original study
    # (expected by diagnostics.py and the benchmark scripts).
    suffix = "" if seed == 42 else f"_seed{seed}"

    if experiment == "baseline":
        return (
            build_baseline,
            f"baseline_segformer_b0{suffix}",
        )

    if experiment == "sfaf":
        return (
            build_sfaf,
            f"proposed_sfaf_b0{suffix}",
        )

    raise ValueError(
        f"Unknown experiment: {experiment}"
    )


def run_full_experiment(
    experiment,
    seed,
    records
):
    set_seed(seed)

    builder, run_name = get_experiment(
        experiment,
        seed,
    )

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        run_dir / "best_model.pt"
    )

    resume_path = (
        run_dir / RESUME_FILE_NAME
    )

    history_path = (
        run_dir / "history.csv"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 78)
    print(
        f"FULL REPEAT RUN: "
        f"{experiment.upper()} | "
        f"SEED {seed}"
    )
    print("=" * 78)

    print(f"Device:        {DEVICE}")
    print(f"Image size:    {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Max epochs:    {MAX_EPOCHS}")
    print(f"Patience:      {PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Seed:          {seed}")
    print(f"Run folder:    {run_dir}")

    train_dataset = BlueberryDataset(
        records["train"]
    )

    val_dataset = BlueberryDataset(
        records["val"]
    )

    test_dataset = BlueberryDataset(
        records["test"]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

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

    model = builder(
        model_name=MODEL_NAME,
        num_classes=2,
    )

    model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{total_params / 1e6:.3f} M"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_iou = -1.0
    best_epoch = 0
    no_improvement = 0
    history = []
    start_epoch = 1
    elapsed_before_resume = 0.0

    # --------------------------------------------------------
    # AUTOMATIC RESUME
    # --------------------------------------------------------

    if RESUME_ENABLED and resume_path.exists():
        print("\nResume checkpoint found.")

        resume = torch.load(
            resume_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            resume["model_state_dict"]
        )

        optimizer.load_state_dict(
            resume["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            resume["scaler_state_dict"]
        )

        best_iou = resume["best_iou"]
        best_epoch = resume["best_epoch"]
        no_improvement = resume["no_improvement"]
        history = resume["history"]
        elapsed_before_resume = resume.get(
            "elapsed_seconds",
            0.0,
        )

        start_epoch = resume["epoch"] + 1

        restore_rng_state(
            resume["rng_state"],
            generator,
        )

        save_history(
            history,
            history_path,
        )

        print(
            f"Resuming after completed epoch {resume['epoch']}."
        )
        print(
            f"Next epoch: {start_epoch}"
        )
        print(
            f"Best Val IoU so far: "
            f"{best_iou * 100:.2f}% "
            f"(epoch {best_epoch})"
        )
        print(
            f"Early-stop counter: "
            f"{no_improvement}/{PATIENCE}"
        )

    start_time = time.time()

    # If early stopping had already been reached before an interruption,
    # do not perform extra training epochs.
    if no_improvement >= PATIENCE:
        print(
            "\nTraining had already reached early stopping. "
            "Proceeding to final test."
        )
        start_epoch = MAX_EPOCHS + 1

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            description=(
                f"{experiment} "
                f"s{seed} "
                f"E{epoch:03d} Train"
            ),
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            description=(
                f"{experiment} "
                f"s{seed} "
                f"E{epoch:03d} Val"
            ),
        )

        epoch_minutes = (
            time.time()
            - epoch_start
        ) / 60

        print()
        print(
            f"Epoch "
            f"{epoch:03d}/{MAX_EPOCHS} "
            f"({epoch_minutes:.1f} min)"
        )

        print_metrics(
            "Train",
            train_metrics,
        )

        print_metrics(
            "Val",
            val_metrics,
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                train_metrics["loss"],

            "train_iou":
                train_metrics["iou"],

            "train_dice":
                train_metrics["dice"],

            "train_precision":
                train_metrics["precision"],

            "train_sensitivity":
                train_metrics["sensitivity"],

            "train_specificity":
                train_metrics["specificity"],

            "val_loss":
                val_metrics["loss"],

            "val_iou":
                val_metrics["iou"],

            "val_dice":
                val_metrics["dice"],

            "val_precision":
                val_metrics["precision"],

            "val_sensitivity":
                val_metrics["sensitivity"],

            "val_specificity":
                val_metrics["specificity"],
        })

        save_history(
            history,
            history_path,
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            no_improvement = 0

            atomic_torch_save(
                {
                    "epoch":
                        epoch,

                    "experiment":
                        experiment,

                    "seed":
                        seed,

                    "model_state_dict":
                        model.state_dict(),

                    "val_metrics":
                        val_metrics,

                    "image_size":
                        IMAGE_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,
                },
                checkpoint_path,
            )

            print(
                f"BEST MODEL SAVED "
                f"(Val IoU="
                f"{best_iou * 100:.2f}%)"
            )

        else:
            no_improvement += 1

            print(
                f"No improvement: "
                f"{no_improvement}/"
                f"{PATIENCE}"
            )

        # Save continuation state after every COMPLETED epoch.
        if RESUME_ENABLED:
            elapsed_total = (
                elapsed_before_resume
                + time.time()
                - start_time
            )

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "no_improvement": no_improvement,
                    "history": history,
                    "rng_state": get_rng_state(generator),
                    "elapsed_seconds": elapsed_total,
                },
                resume_path,
            )

            print(
                f"Resume state saved after epoch {epoch}."
            )

        if no_improvement >= PATIENCE:
            print(
                "\nEarly stopping triggered."
            )
            break

    print("\n" + "-" * 78)
    print(
        "FINAL TEST - BEST VALIDATION CHECKPOINT"
    )
    print("-" * 78)

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
        description=(
            f"{experiment} "
            f"s{seed} Final Test"
        ),
    )

    runtime_minutes = (
        elapsed_before_resume
        + time.time()
        - start_time
    ) / 60

    if torch.cuda.is_available():
        peak_gpu_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )
    else:
        peak_gpu_gb = 0.0

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best Val IoU: "
        f"{best_iou * 100:.2f}%"
    )

    print_metrics(
        "Test",
        test_metrics,
    )

    print(
        f"Runtime: "
        f"{runtime_minutes:.1f} min"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_gpu_gb:.2f} GB"
    )

    result = {
        "experiment":
            experiment,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "best_val_iou":
            best_iou,

        "test_loss":
            test_metrics["loss"],

        "test_iou":
            test_metrics["iou"],

        "test_dice":
            test_metrics["dice"],

        "test_precision":
            test_metrics["precision"],

        "test_sensitivity":
            test_metrics["sensitivity"],

        "test_specificity":
            test_metrics["specificity"],

        "parameters_m":
            total_params / 1e6,

        "peak_gpu_gb":
            peak_gpu_gb,

        "runtime_minutes":
            runtime_minutes,

        "run_dir":
            str(run_dir),
    }

    append_summary(result)

    del model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "AGRIVISION FINAL REPEATABILITY RUNS"
    )
    print("=" * 78)

    print(
        "Architecture is frozen."
    )

    print(
        f"Automatic resume: {RESUME_ENABLED}"
    )

    print(
        "Existing seed-42 checkpoints "
        "will NOT be overwritten."
    )

    print(
        "\nAdditional runs:"
    )

    for experiment, seed in EXPERIMENTS:
        print(
            f"  {experiment:<8} seed {seed}"
        )

    print(
        "\nLoading frozen split..."
    )

    records = load_splits()

    print(
        f"Train={len(records['train'])}, "
        f"Val={len(records['val'])}, "
        f"Test={len(records['test'])}"
    )

    completed = read_completed_runs()
    results = []

    for experiment, seed in EXPERIMENTS:
        key = (
            experiment,
            seed,
        )

        if (
            SKIP_COMPLETED
            and key in completed
        ):
            print("\n" + "=" * 78)
            print(
                f"SKIPPING: "
                f"{experiment} seed {seed}"
            )
            print(
                "Already recorded in "
                "repeat_runs_summary.csv"
            )
            print("=" * 78)
            continue

        result = run_full_experiment(
            experiment,
            seed,
            records,
        )

        results.append(result)

    print("\n" + "=" * 78)
    print(
        "REPEATABILITY RUNS COMPLETE"
    )
    print("=" * 78)

    print(
        f"Summary file:\n"
        f"{SUMMARY_FILE}"
    )

    if results:
        print(
            "\nRuns completed in this execution:"
        )

        for r in results:
            print(
                f"{r['experiment']:<8} "
                f"seed={r['seed']:<3} | "
                f"Val IoU="
                f"{r['best_val_iou'] * 100:.2f}% | "
                f"Test IoU="
                f"{r['test_iou'] * 100:.2f}% | "
                f"Dice="
                f"{r['test_dice'] * 100:.2f}%"
            )

    print(
        "\nKeep the original seed-42 runs. "
        "We will combine all three seeds "
        "after these four runs finish."
    )


if __name__ == "__main__":
    main()
