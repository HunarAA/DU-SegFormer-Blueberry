"""
Qualitative figure for the revised manuscript (final-recipe, seed-42 checkpoints).

Rows: one test image per blueberry-coverage group (<1%, 1-3%, 3-7%, >=7%),
      chosen as the image closest to the group's median coverage (same rule as V2).
Columns: image | ground truth | SegFormer-B0 64-ch | Proposed | U-Net ResNet-34
Prediction panels are error maps: white = correct blueberry (TP),
red = false positive, cyan = missed blueberry (false negative).

Usage:
    python make_figure_qualitative.py
Output: runs/figure_qualitative.png (300 dpi)
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_repeats_resume as base
import model_ablation as abl
import model_novel as mn
import train_baselines as tb

MODELS = [
    ("SegFormer-B0 (64-ch)", abl.build_segformer_dec64, "recipe_r5_adamw_cedice_aug_seed42"),
    ("Proposed", mn.build_dysample_half, "recipe_n2_dysample_half_seed42"),
    ("U-Net (ResNet-34)", tb.build_unet, "recipe_base_unet_r34_seed42"),
]
GROUPS = [("<1%", 0.0, 0.01), ("1–3%", 0.01, 0.03), ("3–7%", 0.03, 0.07), ("≥7%", 0.07, 1.01)]
OUTPUT = base.RUNS_DIR / "figure_qualitative.png"
MEAN = np.array(base.MEAN).reshape(3, 1, 1)
STD = np.array(base.STD).reshape(3, 1, 1)


def load(builder, folder):
    model = builder(model_name="nvidia/mit-b0", num_classes=2)
    ckpt = torch.load(base.RUNS_DIR / folder / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.to(base.DEVICE).eval()


def predict(model, image):
    with torch.inference_mode():
        logits = model(pixel_values=image[None].to(base.DEVICE)).logits
        logits = F.interpolate(logits.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def valid_box(mask):
    ys, xs = np.where(mask != base.IGNORE_INDEX)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def error_map(pred, gt):
    out = np.zeros(gt.shape + (3,), dtype=np.float32)
    out[(pred == 1) & (gt == 1)] = (1, 1, 1)
    out[(pred == 1) & (gt == 0)] = (0.90, 0.10, 0.10)
    out[(pred == 0) & (gt == 1)] = (0.10, 0.80, 0.90)
    return out


def main():
    records = base.load_splits()["test"]
    dataset = base.BlueberryDataset(records)

    coverage = []
    for i in range(len(dataset)):
        _, m = dataset[i]
        m = m.numpy()
        valid = m != base.IGNORE_INDEX
        coverage.append((m[valid] == 1).mean())
    coverage = np.array(coverage)

    chosen = []
    for _, lo, hi in GROUPS:
        idx = np.where((coverage >= lo) & (coverage < hi))[0]
        med = np.median(coverage[idx])
        chosen.append(idx[np.argmin(np.abs(coverage[idx] - med))])

    models = [(label, load(b, f)) for label, b, f in MODELS]

    cols = 2 + len(models)
    fig, axes = plt.subplots(len(GROUPS), cols, figsize=(2.6 * cols, 2.6 * len(GROUPS)))
    titles = ["Image", "Ground truth"] + [m[0] for m in models]
    for r, (gname, _, _) in enumerate(GROUPS):
        image, mask = dataset[chosen[r]]
        gt = mask.numpy()
        y0, y1, x0, x1 = valid_box(gt)
        rgb = np.clip(image.numpy() * STD + MEAN, 0, 1).transpose(1, 2, 0)

        panels = [rgb, np.repeat((gt == 1)[..., None], 3, axis=2).astype(np.float32)]
        for _, model in models:
            panels.append(error_map(predict(model, image), gt))

        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(f"{gname}\n({coverage[chosen[r]] * 100:.2f}%)", fontsize=10)
        print(f"{gname}: {records[chosen[r]]['image']} (coverage {coverage[chosen[r]] * 100:.2f}%)")

    fig.text(0.5, 0.005, "White: correct blueberry   Red: false positive   Cyan: missed blueberry",
             ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUTPUT, dpi=300)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
