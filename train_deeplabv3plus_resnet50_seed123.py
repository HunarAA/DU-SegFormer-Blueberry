from pathlib import Path
from types import SimpleNamespace
import csv
import random
import time

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet34,
    resnet50,
    ResNet34_Weights,
    ResNet50_Weights,
)
from torchvision.transforms.functional import (
    to_tensor,
    normalize,
)


# ============================================================
# AGRIVISION BLUEBERRY - STANDARD COMPARISON MODELS
# ============================================================
#
# Runs one repeat comparison baseline:
#
#   DeepLabV3+ with ImageNet-pretrained ResNet-50 encoder
#
# The experimental protocol is intentionally kept the same as
# the existing SegFormer-B0 / SFAF experiments:
#
#   - frozen splits.csv
#   - source-video-disjoint train/val/test split
#   - 640 x 640 letterbox
#   - ImageNet normalization
#   - no augmentation
#   - batch size 4
#   - Adam, lr = 1e-4
#   - dynamic inverse-frequency weighted CE
#   - max 100 epochs
#   - patience 15
#   - best validation IoU selects checkpoint
#   - seed 123
#   - AMP
#
# SFAF is NOT modified or trained by this script.
#
# Automatic resume:
#   - saves last_state.pt after every completed epoch
#   - if interrupted, rerun this same script
#   - completed runs are skipped automatically
# ============================================================


# ============================================================
# REPEAT RUN: DEEPLABV3PLUS | SEED 123
# Derived from train_standard_baselines_fixed.py
# Architecture and training protocol are unchanged.
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"E:\Papers\AgriVision Blueberry"
)

DATASET_DIR = PROJECT_DIR / "DB-1"
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"

RUNS_DIR = PROJECT_DIR / "runs"

SUMMARY_FILE = (
    RUNS_DIR
    / "standard_baselines_repeats_summary.csv"
)

MODELS_TO_RUN = [
    "deeplabv3plus",
]

SEED = 123

IMAGE_SIZE = 640
BATCH_SIZE = 4
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
IGNORE_INDEX = 255

RESUME_ENABLED = True
SKIP_COMPLETED = True
RESUME_FILE_NAME = "last_state.pt"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)

MEAN = [
    0.485,
    0.456,
    0.406,
]

STD = [
    0.229,
    0.224,
    0.225,
]


if hasattr(
    Image,
    "Resampling",
):
    BILINEAR = (
        Image.Resampling.BILINEAR
    )

    NEAREST = (
        Image.Resampling.NEAREST
    )

else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


# ============================================================
# REPRODUCIBILITY / RESUME HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )


def get_rng_state(
    loader_generator
):
    state = {
        "python":
            random.getstate(),

        "numpy":
            np.random.get_state(),

        "torch_cpu":
            torch.get_rng_state(),

        "loader_generator":
            loader_generator.get_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda
            .get_rng_state_all()
        )

    return state


def restore_rng_state(
    state,
    loader_generator
):
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    loader_generator.set_state(
        state["loader_generator"]
    )

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


def atomic_torch_save(
    obj,
    path
):
    temp_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    torch.save(
        obj,
        temp_path,
    )

    temp_path.replace(
        path
    )


# ============================================================
# INPUT PREPARATION
# ============================================================

def letterbox(
    image,
    mask,
    size
):
    width, height = image.size

    scale = min(
        size / width,
        size / height,
    )

    new_width = int(
        round(
            width * scale
        )
    )

    new_height = int(
        round(
            height * scale
        )
    )

    image = image.resize(
        (
            new_width,
            new_height,
        ),
        BILINEAR,
    )

    mask = mask.resize(
        (
            new_width,
            new_height,
        ),
        NEAREST,
    )

    left = (
        size - new_width
    ) // 2

    top = (
        size - new_height
    ) // 2

    image_canvas = Image.new(
        "RGB",
        (
            size,
            size,
        ),
        (
            0,
            0,
            0,
        ),
    )

    image_canvas.paste(
        image,
        (
            left,
            top,
        ),
    )

    mask_array = (
        np.array(mask)
        > 127
    ).astype(
        np.uint8
    )

    mask_canvas = np.full(
        (
            size,
            size,
        ),
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    mask_canvas[
        top:top + new_height,
        left:left + new_width,
    ] = mask_array

    return (
        image_canvas,
        mask_canvas,
    )


class BlueberryDataset(
    Dataset
):

    def __init__(
        self,
        records
    ):
        self.records = records

    def __len__(
        self
    ):
        return len(
            self.records
        )

    def __getitem__(
        self,
        index
    ):
        record = (
            self.records[index]
        )

        image_path = (
            IMAGE_DIR
            / record["image"]
        )

        mask_path = (
            MASK_DIR
            / record["mask"]
        )

        with Image.open(
            image_path
        ) as img:
            image = (
                img.convert("RGB")
            )

        with Image.open(
            mask_path
        ) as m:
            mask = (
                m.convert("L")
            )

        image, mask = letterbox(
            image,
            mask,
            IMAGE_SIZE,
        )

        image = to_tensor(
            image
        )

        image = normalize(
            image,
            mean=MEAN,
            std=STD,
        )

        mask = torch.from_numpy(
            mask.astype(
                np.int64
            )
        )

        return (
            image,
            mask,
        )


def load_splits():
    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            SPLIT_FILE
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f
        )

        for row in reader:
            split = (
                row["split"]
                .strip()
            )

            if split not in records:
                raise ValueError(
                    f"Unknown split: "
                    f"{split}"
                )

            records[
                split
            ].append(
                row
            )

    return records


# ============================================================
# MODEL 1: U-NET WITH RESNET-34 ENCODER
# ============================================================

class ConvBNReLU(
    nn.Sequential
):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )


class DoubleConv(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.block = (
            nn.Sequential(
                ConvBNReLU(
                    in_channels,
                    out_channels,
                ),

                ConvBNReLU(
                    out_channels,
                    out_channels,
                ),
            )
        )

    def forward(
        self,
        x
    ):
        return self.block(
            x
        )


class UNetDecoderBlock(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = DoubleConv(
            in_channels
            + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x,
        skip
    ):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.conv(
            x
        )


class UNetResNet34(
    nn.Module
):
    """
    U-Net-style encoder-decoder with an
    ImageNet-pretrained ResNet-34 encoder.
    """

    def __init__(
        self,
        num_classes=2,
    ):
        super().__init__()

        backbone = resnet34(
            weights=(
                ResNet34_Weights
                .IMAGENET1K_V1
            )
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.maxpool = (
            backbone.maxpool
        )

        self.layer1 = (
            backbone.layer1
        )

        self.layer2 = (
            backbone.layer2
        )

        self.layer3 = (
            backbone.layer3
        )

        self.layer4 = (
            backbone.layer4
        )

        # Encoder channel widths:
        # stem 64
        # layer1 64
        # layer2 128
        # layer3 256
        # layer4 512

        self.dec4 = (
            UNetDecoderBlock(
                512,
                256,
                256,
            )
        )

        self.dec3 = (
            UNetDecoderBlock(
                256,
                128,
                128,
            )
        )

        self.dec2 = (
            UNetDecoderBlock(
                128,
                64,
                64,
            )
        )

        self.dec1 = (
            UNetDecoderBlock(
                64,
                64,
                64,
            )
        )

        self.final_refine = (
            nn.Sequential(
                ConvBNReLU(
                    64,
                    32,
                ),

                nn.Dropout2d(
                    0.1
                ),

                nn.Conv2d(
                    32,
                    num_classes,
                    kernel_size=1,
                ),
            )
        )

    def forward(
        self,
        pixel_values
    ):
        input_size = (
            pixel_values.shape[-2:]
        )

        x0 = self.stem(
            pixel_values
        )

        x1 = self.layer1(
            self.maxpool(
                x0
            )
        )

        x2 = self.layer2(
            x1
        )

        x3 = self.layer3(
            x2
        )

        x4 = self.layer4(
            x3
        )

        x = self.dec4(
            x4,
            x3,
        )

        x = self.dec3(
            x,
            x2,
        )

        x = self.dec2(
            x,
            x1,
        )

        x = self.dec1(
            x,
            x0,
        )

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        logits = (
            self.final_refine(
                x
            )
        )

        return SimpleNamespace(
            logits=logits,
            aux_logits=None,
        )


# ============================================================
# MODEL 2: DEEPLABV3+ WITH RESNET-50 ENCODER
# ============================================================

class ASPPConv(
    nn.Sequential
):

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )




class SafeBatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d variant that safely handles a singleton 1x1
    activation during training.

    This is needed only for the global-image-pooling branch
    of DeepLabV3+. With 833 training images and batch size 4,
    the final training batch contains one image. After global
    average pooling, that branch becomes [1, C, 1, 1], for
    which ordinary training-mode BatchNorm cannot compute batch
    statistics.

    For that single special case, accumulated running statistics
    are used instead. All normal batches use standard BatchNorm.
    """

    def forward(self, input):
        values_per_channel = (
            input.numel()
            // input.shape[1]
        )

        if (
            self.training
            and values_per_channel == 1
        ):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )

        return super().forward(input)


class ASPPPooling(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.pool = (
            nn.AdaptiveAvgPool2d(
                1
            )
        )

        self.project = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),

                SafeBatchNorm2d(
                    out_channels
                ),

                nn.ReLU(
                    inplace=True
                ),
            )
        )

    def forward(
        self,
        x
    ):
        size = x.shape[-2:]

        y = self.pool(
            x
        )

        y = self.project(
            y
        )

        y = F.interpolate(
            y,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        return y


class ASPP(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels=256,
        rates=(
            6,
            12,
            18,
        ),
    ):
        super().__init__()

        self.branches = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            bias=False,
                        ),

                        nn.BatchNorm2d(
                            out_channels
                        ),

                        nn.ReLU(
                            inplace=True
                        ),
                    ),

                    ASPPConv(
                        in_channels,
                        out_channels,
                        rates[0],
                    ),

                    ASPPConv(
                        in_channels,
                        out_channels,
                        rates[1],
                    ),

                    ASPPConv(
                        in_channels,
                        out_channels,
                        rates[2],
                    ),

                    ASPPPooling(
                        in_channels,
                        out_channels,
                    ),
                ]
            )
        )

        self.project = (
            nn.Sequential(
                nn.Conv2d(
                    out_channels * 5,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    out_channels
                ),

                nn.ReLU(
                    inplace=True
                ),

                nn.Dropout(
                    0.5
                ),
            )
        )

    def forward(
        self,
        x
    ):
        features = [
            branch(x)
            for branch
            in self.branches
        ]

        x = torch.cat(
            features,
            dim=1,
        )

        return self.project(
            x
        )


class DeepLabV3PlusResNet50(
    nn.Module
):
    """
    DeepLabV3+ with:
      - ImageNet-pretrained ResNet-50 backbone
      - output stride 16
      - ASPP rates 6, 12, 18
      - low-level Layer-1 skip projected to 48 channels
    """

    def __init__(
        self,
        num_classes=2,
    ):
        super().__init__()

        backbone = resnet50(
            weights=(
                ResNet50_Weights
                .IMAGENET1K_V1
            ),

            # Output stride 16:
            # layer4 uses dilation instead
            # of another spatial downsample.
            replace_stride_with_dilation=[
                False,
                False,
                True,
            ],
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = (
            backbone.layer1
        )

        self.layer2 = (
            backbone.layer2
        )

        self.layer3 = (
            backbone.layer3
        )

        self.layer4 = (
            backbone.layer4
        )

        self.aspp = ASPP(
            in_channels=2048,
            out_channels=256,
            rates=(
                6,
                12,
                18,
            ),
        )

        self.low_level_projection = (
            nn.Sequential(
                nn.Conv2d(
                    256,
                    48,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    48
                ),

                nn.ReLU(
                    inplace=True
                ),
            )
        )

        self.decoder = (
            nn.Sequential(
                ConvBNReLU(
                    256 + 48,
                    256,
                ),

                ConvBNReLU(
                    256,
                    256,
                ),

                nn.Dropout2d(
                    0.1
                ),

                nn.Conv2d(
                    256,
                    num_classes,
                    kernel_size=1,
                ),
            )
        )

    def forward(
        self,
        pixel_values
    ):
        x = self.stem(
            pixel_values
        )

        low = self.layer1(
            x
        )

        x = self.layer2(
            low
        )

        x = self.layer3(
            x
        )

        high = self.layer4(
            x
        )

        x = self.aspp(
            high
        )

        low = (
            self.low_level_projection(
                low
            )
        )

        x = F.interpolate(
            x,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x,
                low,
            ],
            dim=1,
        )

        logits = self.decoder(
            x
        )

        return SimpleNamespace(
            logits=logits,
            aux_logits=None,
        )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_model(
    experiment
):
    if experiment == "unet":
        return (
            UNetResNet34(
                num_classes=2
            ),

            "U-Net + ResNet-34",

            (
                RUNS_DIR
                / f"unet_resnet34_seed{SEED}"
            ),
        )

    if experiment == "deeplabv3plus":
        return (
            DeepLabV3PlusResNet50(
                num_classes=2
            ),

            "DeepLabV3+ + ResNet-50",

            (
                RUNS_DIR
                / f"deeplabv3plus_resnet50_seed{SEED}"
            ),
        )

    raise ValueError(
        f"Unknown experiment: "
        f"{experiment}"
    )


# ============================================================
# LOSS
# ============================================================

def dynamic_weighted_ce(
    logits,
    labels
):
    valid = (
        labels
        != IGNORE_INDEX
    )

    valid_labels = (
        labels[valid]
    )

    counts = torch.bincount(
        valid_labels,
        minlength=2,
    ).float()

    frequencies = (
        counts
        / counts.sum()
        .clamp_min(
            1.0
        )
    )

    inverse = (
        1.0
        / frequencies
        .clamp_min(
            1e-6
        )
    )

    weights = (
        inverse
        / inverse.sum()
    )

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

    def __init__(
        self
    ):
        self.reset()

    def reset(
        self
    ):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        predictions,
        targets
    ):
        valid = (
            targets
            != IGNORE_INDEX
        )

        predictions = (
            predictions
            .detach()[valid]
        )

        targets = (
            targets
            .detach()[valid]
        )

        self.tp += (
            (
                (
                    predictions == 1
                )
                & (
                    targets == 1
                )
            )
            .sum()
            .item()
        )

        self.tn += (
            (
                (
                    predictions == 0
                )
                & (
                    targets == 0
                )
            )
            .sum()
            .item()
        )

        self.fp += (
            (
                (
                    predictions == 1
                )
                & (
                    targets == 0
                )
            )
            .sum()
            .item()
        )

        self.fn += (
            (
                (
                    predictions == 0
                )
                & (
                    targets == 1
                )
            )
            .sum()
            .item()
        )

    def compute(
        self
    ):
        eps = 1e-8

        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn

        return {
            "iou":
                tp
                / (
                    tp
                    + fp
                    + fn
                    + eps
                ),

            "dice":
                2 * tp
                / (
                    2 * tp
                    + fp
                    + fn
                    + eps
                ),

            "precision":
                tp
                / (
                    tp
                    + fp
                    + eps
                ),

            "sensitivity":
                tp
                / (
                    tp
                    + fn
                    + eps
                ),

            "specificity":
                tn
                / (
                    tn
                    + fp
                    + eps
                ),
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
    training = (
        optimizer
        is not None
    )

    if training:
        model.train()

    else:
        model.eval()

    metrics = (
        SegmentationMetrics()
    )

    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for (
        images,
        labels,
    ) in progress:

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

        with torch.set_grad_enabled(
            training
        ):

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

                loss = (
                    dynamic_weighted_ce(
                        logits,
                        labels,
                    )
                )

            if training:
                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        metrics.update(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
        )

        total_batches += 1

        current = (
            metrics.compute()
        )

        postfix = {
            "loss":
                f"{total_loss / total_batches:.4f}",

            "IoU":
                f"{current['iou'] * 100:.1f}%",
        }

        if torch.cuda.is_available():
            gpu = (
                torch.cuda
                .memory_allocated()
                / 1024**3
            )

            postfix["GPU"] = (
                f"{gpu:.1f}G"
            )

        progress.set_postfix(
            postfix
        )

    results = (
        metrics.compute()
    )

    results["loss"] = (
        total_loss
        / max(
            total_batches,
            1,
        )
    )

    return results


def print_metrics(
    prefix,
    metrics
):
    print(
        f"{prefix:<6} "
        f"Loss={metrics['loss']:.4f} | "
        f"IoU={metrics['iou'] * 100:.2f}% | "
        f"Dice={metrics['dice'] * 100:.2f}% | "
        f"Pre={metrics['precision'] * 100:.2f}% | "
        f"Sen={metrics['sensitivity'] * 100:.2f}% | "
        f"Spe={metrics['specificity'] * 100:.2f}%"
    )


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_history(
    history,
    path
):
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
            fieldnames=(
                history[0]
                .keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            history
        )


SUMMARY_FIELDS = [
    "experiment",
    "model_label",
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
        reader = csv.DictReader(
            f
        )

        for row in reader:
            try:
                row_seed = int(
                    row["seed"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            completed.add(
                (
                    row["experiment"],
                    row_seed,
                )
            )

    return completed


def append_summary(
    result
):
    file_exists = (
        SUMMARY_FILE.exists()
    )

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

        writer.writerow(
            result
        )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def run_full_experiment(
    experiment,
    records
):
    set_seed(
        SEED
    )

    model, model_label, run_dir = (
        build_model(
            experiment
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        run_dir
        / "best_model.pt"
    )

    resume_path = (
        run_dir
        / RESUME_FILE_NAME
    )

    history_path = (
        run_dir
        / "history.csv"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"STANDARD BASELINE: "
        f"{model_label}"
    )

    print(
        "=" * 78
    )

    print(
        f"Device:        "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU:           "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Image size:    "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Batch size:    "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max epochs:    "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Patience:      "
        f"{PATIENCE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Seed:          "
        f"{SEED}"
    )

    print(
        f"Run folder:    "
        f"{run_dir}"
    )

    print(
        "\nProtocol is fixed and "
        "matches SegFormer/SFAF."
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = (
        BlueberryDataset(
            records["train"]
        )
    )

    val_dataset = (
        BlueberryDataset(
            records["val"]
        )
    )

    test_dataset = (
        BlueberryDataset(
            records["test"]
        )
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model.to(
        DEVICE
    )

    total_params = sum(
        p.numel()
        for p
        in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParameters: "
        f"{total_params / 1e6:.3f} M"
    )

    print(
        f"Trainable:  "
        f"{trainable_params / 1e6:.3f} M"
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(
                0.9,
                0.999,
            ),
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP,
        )
    )

    best_iou = -1.0
    best_epoch = 0
    no_improvement = 0
    history = []
    start_epoch = 1
    elapsed_before_resume = 0.0

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if (
        RESUME_ENABLED
        and resume_path.exists()
    ):
        print(
            "\nResume checkpoint found."
        )

        resume = torch.load(
            resume_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            resume[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            resume[
                "optimizer_state_dict"
            ]
        )

        scaler.load_state_dict(
            resume[
                "scaler_state_dict"
            ]
        )

        best_iou = (
            resume["best_iou"]
        )

        best_epoch = (
            resume["best_epoch"]
        )

        no_improvement = (
            resume[
                "no_improvement"
            ]
        )

        history = (
            resume["history"]
        )

        elapsed_before_resume = (
            resume.get(
                "elapsed_seconds",
                0.0,
            )
        )

        start_epoch = (
            resume["epoch"]
            + 1
        )

        restore_rng_state(
            resume[
                "rng_state"
            ],
            generator,
        )

        save_history(
            history,
            history_path,
        )

        print(
            f"Resuming after "
            f"completed epoch "
            f"{resume['epoch']}."
        )

        print(
            f"Next epoch: "
            f"{start_epoch}"
        )

        print(
            f"Best Val IoU so far: "
            f"{best_iou * 100:.2f}% "
            f"(epoch {best_epoch})"
        )

        print(
            f"Early-stop counter: "
            f"{no_improvement}/"
            f"{PATIENCE}"
        )

    start_time = (
        time.time()
    )

    if (
        no_improvement
        >= PATIENCE
    ):
        print(
            "\nTraining had already "
            "reached early stopping. "
            "Proceeding to final test."
        )

        start_epoch = (
            MAX_EPOCHS + 1
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    try:
        for epoch in range(
            start_epoch,
            MAX_EPOCHS + 1,
        ):
            epoch_start = (
                time.time()
            )

            train_metrics = (
                run_epoch(
                    model,
                    train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    description=(
                        f"{experiment} "
                        f"E{epoch:03d} "
                        f"Train"
                    ),
                )
            )

            val_metrics = (
                run_epoch(
                    model,
                    val_loader,
                    description=(
                        f"{experiment} "
                        f"E{epoch:03d} "
                        f"Val"
                    ),
                )
            )

            epoch_minutes = (
                time.time()
                - epoch_start
            ) / 60

            print()

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{MAX_EPOCHS} "
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

            history.append(
                {
                    "epoch":
                        epoch,

                    "train_loss":
                        train_metrics[
                            "loss"
                        ],

                    "train_iou":
                        train_metrics[
                            "iou"
                        ],

                    "train_dice":
                        train_metrics[
                            "dice"
                        ],

                    "train_precision":
                        train_metrics[
                            "precision"
                        ],

                    "train_sensitivity":
                        train_metrics[
                            "sensitivity"
                        ],

                    "train_specificity":
                        train_metrics[
                            "specificity"
                        ],

                    "val_loss":
                        val_metrics[
                            "loss"
                        ],

                    "val_iou":
                        val_metrics[
                            "iou"
                        ],

                    "val_dice":
                        val_metrics[
                            "dice"
                        ],

                    "val_precision":
                        val_metrics[
                            "precision"
                        ],

                    "val_sensitivity":
                        val_metrics[
                            "sensitivity"
                        ],

                    "val_specificity":
                        val_metrics[
                            "specificity"
                        ],
                }
            )

            save_history(
                history,
                history_path,
            )

            if (
                val_metrics["iou"]
                > best_iou
            ):
                best_iou = (
                    val_metrics[
                        "iou"
                    ]
                )

                best_epoch = epoch
                no_improvement = 0

                atomic_torch_save(
                    {
                        "epoch":
                            epoch,

                        "experiment":
                            experiment,

                        "model_label":
                            model_label,

                        "seed":
                            SEED,

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

            if RESUME_ENABLED:
                elapsed_total = (
                    elapsed_before_resume
                    + time.time()
                    - start_time
                )

                atomic_torch_save(
                    {
                        "epoch":
                            epoch,

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "scaler_state_dict":
                            scaler.state_dict(),

                        "best_iou":
                            best_iou,

                        "best_epoch":
                            best_epoch,

                        "no_improvement":
                            no_improvement,

                        "history":
                            history,

                        "rng_state":
                            get_rng_state(
                                generator
                            ),

                        "elapsed_seconds":
                            elapsed_total,
                    },
                    resume_path,
                )

                print(
                    f"Resume state saved "
                    f"after epoch "
                    f"{epoch}."
                )

            if (
                no_improvement
                >= PATIENCE
            ):
                print(
                    "\nEarly stopping "
                    "triggered."
                )

                break

    except torch.OutOfMemoryError:
        print(
            "\nCUDA OUT OF MEMORY."
        )

        print(
            "The protocol requires "
            "batch size 4, so this "
            "script will NOT silently "
            "change the batch size."
        )

        print(
            "Send me this console "
            "output and we will choose "
            "the scientifically clean "
            "solution."
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No best validation "
            "checkpoint exists."
        )

    print(
        "\n"
        + "-" * 78
    )

    print(
        "FINAL TEST - "
        "BEST VALIDATION CHECKPOINT"
    )

    print(
        "-" * 78
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        description=(
            f"{experiment} "
            f"Final Test"
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
        f"Best epoch: "
        f"{best_epoch}"
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

        "model_label":
            model_label,

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_iou":
            best_iou,

        "test_loss":
            test_metrics[
                "loss"
            ],

        "test_iou":
            test_metrics[
                "iou"
            ],

        "test_dice":
            test_metrics[
                "dice"
            ],

        "test_precision":
            test_metrics[
                "precision"
            ],

        "test_sensitivity":
            test_metrics[
                "sensitivity"
            ],

        "test_specificity":
            test_metrics[
                "specificity"
            ],

        "parameters_m":
            total_params
            / 1e6,

        "peak_gpu_gb":
            peak_gpu_gb,

        "runtime_minutes":
            runtime_minutes,

        "run_dir":
            str(
                run_dir
            ),
    }

    append_summary(
        result
    )

    del model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 78
    )

    print(
        "AGRIVISION STANDARD "
        "COMPARISON BASELINE REPEAT - FIXED V2"
    )

    print(
        "DeepLabV3+ singleton-ASPP BatchNorm safeguard: ENABLED"
    )

    print(
        "=" * 78
    )

    print(
        "SFAF remains frozen."
    )

    print(
        f"Automatic resume: "
        f"{RESUME_ENABLED}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        "\nModel:"
    )

    print(
        "  DeepLabV3+ + "
        "ImageNet-pretrained "
        "ResNet-50"
    )

    print(
        "\nLoading frozen "
        "splits.csv..."
    )

    records = load_splits()

    print(
        f"Train="
        f"{len(records['train'])}, "
        f"Val="
        f"{len(records['val'])}, "
        f"Test="
        f"{len(records['test'])}"
    )

    print(
        "Expected frozen counts: "
        "833 / 122 / 240"
    )

    if (
        len(records["train"])
        != 833
        or len(records["val"])
        != 122
        or len(records["test"])
        != 240
    ):
        raise RuntimeError(
            "Split counts do not "
            "match the frozen "
            "protocol. Stop."
        )

    completed = (
        read_completed_runs()
    )

    results = []

    for experiment in (
        MODELS_TO_RUN
    ):
        if (
            SKIP_COMPLETED
            and (
                experiment,
                SEED,
            )
            in completed
        ):
            print(
                "\n"
                + "=" * 78
            )

            print(
                f"SKIPPING: "
                f"{experiment}"
            )

            print(
                "Already recorded in "
                "standard_baselines_repeats_summary.csv"
            )

            print(
                "=" * 78
            )

            continue

        result = (
            run_full_experiment(
                experiment,
                records,
            )
        )

        results.append(
            result
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "STANDARD BASELINE "
        "EXPERIMENTS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Summary file:\n"
        f"{SUMMARY_FILE}"
    )

    if results:
        print(
            "\nRuns completed "
            "in this execution:"
        )

        for result in results:
            print(
                f"{result['model_label']:<29} | "
                f"Val IoU="
                f"{result['best_val_iou'] * 100:.2f}% | "
                f"Test IoU="
                f"{result['test_iou'] * 100:.2f}% | "
                f"Dice="
                f"{result['test_dice'] * 100:.2f}% | "
                f"Params="
                f"{result['parameters_m']:.2f}M"
            )

    print(
        "\nDo not tune either "
        "comparison model using "
        "the test results."
    )


if __name__ == "__main__":
    main()
