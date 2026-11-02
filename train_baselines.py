"""
Comparison baselines retrained with the FINAL recipe (same as the proposed model):
AdamW (encoder 6e-5, decoder/head 6e-4, weight decay 0.01), 2-epoch warmup + poly,
CE + Dice loss, augmentation, 100 epochs, best-validation checkpoint, AMP, batch 4.

Models:
    unet_r34          U-Net + ResNet-34      (your paper's model, unchanged code)
    deeplabv3p_r50    DeepLabV3+ + ResNet-50 (your paper's model, unchanged code)
    lraspp_mbv3       LR-ASPP + MobileNetV3-Large   (torchvision, lightweight)
    deeplabv3_mbv3    DeepLabV3 + MobileNetV3-Large (torchvision, mobile)

All use ImageNet-pretrained encoders. Order: all four with seed 42 first,
then seeds 7 and 123, so a complete single-seed comparison is available early.

Usage:
    python train_baselines.py
Stop with Ctrl+C at any time; rerun to resume. Results: runs/baselines_summary.csv
Note: the MobileNetV3 ImageNet weights are downloaded once (internet needed the
first time); ResNet weights are already on your computer.
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv
import statistics
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import ASPPPooling as TVASPPPooling

import train_recipe as recipe
import train_standard_baselines_fixed as std


# ------------------------------------------------------------
# Model builders (common signature)
# ------------------------------------------------------------

class TorchvisionWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.net(pixel_values)["out"], aux_logits=None)


def _safe_aspp_pooling(net):
    """Image-pooling BatchNorm fails on a batch of 1 (833 % 4 = 1); use the
    same safeguard as your DeepLabV3+ code."""
    for module in net.modules():
        if isinstance(module, TVASPPPooling):
            for i, layer in enumerate(module):
                if isinstance(layer, nn.BatchNorm2d):
                    safe = std.SafeBatchNorm2d(layer.num_features)
                    safe.load_state_dict(layer.state_dict())
                    module[i] = safe
    return net


def build_unet(model_name=None, num_classes=2):
    return std.UNetResNet34(num_classes=num_classes)


def build_deeplabv3p(model_name=None, num_classes=2):
    return std.DeepLabV3PlusResNet50(num_classes=num_classes)


def build_lraspp(model_name=None, num_classes=2):
    net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                                    num_classes=num_classes)
    return TorchvisionWrapper(net)


def build_deeplabv3_mbv3(model_name=None, num_classes=2):
    net = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                                       num_classes=num_classes, aux_loss=False)
    return TorchvisionWrapper(_safe_aspp_pooling(net))


FINAL = {"loss": "ce_dice", "augment": True}
MODELS = {
    "base_unet_r34":       {**FINAL, "build": build_unet,           "label": "U-Net + ResNet-34"},
    "base_deeplabv3p_r50": {**FINAL, "build": build_deeplabv3p,     "label": "DeepLabV3+ + ResNet-50"},
    "base_lraspp_mbv3":    {**FINAL, "build": build_lraspp,         "label": "LR-ASPP + MobileNetV3-L"},
    "base_deeplabv3_mbv3": {**FINAL, "build": build_deeplabv3_mbv3, "label": "DeepLabV3 + MobileNetV3-L"},
}
SEEDS = [42, 7, 123]


# ------------------------------------------------------------
# Optimizer: pretrained encoder at 6e-5, everything else at 6e-4 (same rule as SegFormer)
# ------------------------------------------------------------

ENCODER_PREFIXES = ("segformer.", "encoder.", "stem.", "layer1.", "layer2.", "layer3.", "layer4.",
                    "net.backbone.")


def make_optimizer(model):
    groups = {"enc_decay": [], "enc_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "enc" if name.startswith(ENCODER_PREFIXES) else "head"
        decay = "nodecay" if p.ndim <= 1 else "decay"
        groups[f"{part}_{decay}"].append(p)
    head_lr = recipe.ENCODER_LR * recipe.HEAD_LR_MULT
    return torch.optim.AdamW([
        {"params": groups["enc_decay"], "lr": recipe.ENCODER_LR, "weight_decay": recipe.WEIGHT_DECAY},
        {"params": groups["enc_nodecay"], "lr": recipe.ENCODER_LR, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": recipe.WEIGHT_DECAY},
        {"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ], betas=(0.9, 0.999))


recipe.RECIPES = MODELS
recipe.EXPERIMENTS = [(name, seed) for seed in SEEDS for name in MODELS]
recipe.SUMMARY_FILE = recipe.RUNS_DIR / "baselines_summary.csv"
recipe.build_for = lambda cfg: cfg["build"]
recipe.make_optimizer = make_optimizer


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

REFERENCE = [
    ("SegFormer-B0 64-ch", "80.21 +/- 0.11", "73.74 +/- 0.17"),
    ("SegFormer-B0 256-ch", "80.50 +/- 0.16", "74.20 +/- 0.13"),
    ("Proposed (DySample + 1/2 res)", "80.91 +/- 0.04", "74.63 +/- 0.09"),
]


def print_summary():
    if not recipe.SUMMARY_FILE.exists():
        return
    with open(recipe.SUMMARY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 78)
    print("BASELINES WITH THE FINAL RECIPE")
    print("=" * 78)
    print(f"{'Model':<28}{'Seed':>5}{'IoU@0.5':>9}{'Thr':>7}{'ImgIoU':>8}{'<1%':>7}{'1-3%':>7}")
    for r in rows:
        label = MODELS.get(r["recipe"], {}).get("label", r["recipe"])
        print(f"{label:<28}{r['seed']:>5}{float(r['test_iou_t05']):>9.2f}{float(r['val_best_threshold']):>7.3f}"
              f"{float(r['test_per_image_iou']):>8.2f}{float(r['img_iou_lt1']):>7.2f}{float(r['img_iou_1_3']):>7.2f}")

    print(f"\n{'Model (mean +/- SD)':<32}{'Seeds':>6}{'IoU@0.5':>17}{'Per-image IoU':>17}")
    for name, cfg in MODELS.items():
        sel = [r for r in rows if r["recipe"] == name]
        if not sel:
            continue

        def ms(col):
            v = [float(r[col]) for r in sel]
            return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

        print(f"{cfg['label']:<32}{len(sel):>6}{ms('test_iou_t05'):>17}{ms('test_per_image_iou'):>17}")
    for label, iou, img in REFERENCE:
        print(f"{label:<32}{3:>6}{iou:>17}{img:>17}")


recipe.print_summary = print_summary


if __name__ == "__main__":
    recipe.main()
