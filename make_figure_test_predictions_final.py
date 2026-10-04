from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize
import matplotlib.pyplot as plt

from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\Papers\AgriVision Blueberry")
DATASET_DIR = PROJECT_DIR / "DB-1"
IMAGE_DIR = DATASET_DIR / "Images"
MASK_DIR = DATASET_DIR / "Masks"
SPLIT_FILE = PROJECT_DIR / "splits.csv"

BASELINE_CKPT = PROJECT_DIR / "runs" / "baseline_segformer_b0" / "best_model.pt"
SFAF_CKPT = PROJECT_DIR / "runs" / "proposed_sfaf_b0" / "best_model.pt"

OUTPUT_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "figure_test_predictions.png"
OUTPUT_SVG = OUTPUT_DIR / "figure_test_predictions.svg"

MODEL_NAME = "nvidia/mit-b0"
NUM_CLASSES = 2
IMAGE_SIZE = 640

# Choose how many examples to show.
# Recommended: 4 rows (one from each foreground-coverage group)
MAX_CASES = 4

# Foreground-coverage groups used earlier in analysis
COVERAGE_BINS = [
    ("<1%", 0.00, 0.01),
    ("1-3%", 0.01, 0.03),
    ("3-7%", 0.03, 0.07),
    ("≥7%", 0.07, 1.00),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPERS
# ============================================================

def print_header():
    print("=" * 90)
    print("MAKE TEST PREDICTION FIGURE")
    print("=" * 90)
    print(f"Device: {DEVICE}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Split file: {SPLIT_FILE}")
    print(f"Baseline checkpoint: {BASELINE_CKPT}")
    print(f"SFAF checkpoint: {SFAF_CKPT}")
    print()


def find_column(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def resolve_existing_path(path_str, base_dir=None):
    p = Path(str(path_str))
    if p.exists():
        return p
    if base_dir is not None:
        p2 = base_dir / p
        if p2.exists():
            return p2
    return p


def resolve_image_mask_paths(df):
    split_col = find_column(df, ["split"])
    image_path_col = find_column(df, ["image_path", "img_path"], required=False)
    mask_path_col = find_column(df, ["mask_path"], required=False)

    image_name_col = None
    mask_name_col = None

    if image_path_col is None:
        image_name_col = find_column(df, ["image", "image_name", "filename", "file_name", "id"])
    if mask_path_col is None:
        mask_name_col = find_column(df, ["mask", "mask_name"], required=False)

    rows = []
    for _, row in df.iterrows():
        split_value = str(row[split_col]).strip().lower()

        if image_path_col is not None:
            image_path = resolve_existing_path(row[image_path_col], PROJECT_DIR)
        else:
            image_path = resolve_existing_path(row[image_name_col], IMAGE_DIR)

        if mask_path_col is not None:
            mask_path = resolve_existing_path(row[mask_path_col], PROJECT_DIR)
        else:
            if mask_name_col is not None:
                mask_path = resolve_existing_path(row[mask_name_col], MASK_DIR)
            else:
                # assume same stem in MASK_DIR
                stem = Path(image_path).stem
                possible = [
                    MASK_DIR / f"{stem}.png",
                    MASK_DIR / f"{stem}.jpg",
                    MASK_DIR / f"{stem}.jpeg",
                    MASK_DIR / f"{stem}.tif",
                    MASK_DIR / f"{stem}.tiff",
                    MASK_DIR / f"{stem}.bmp",
                ]
                found = None
                for pp in possible:
                    if pp.exists():
                        found = pp
                        break
                if found is None:
                    raise FileNotFoundError(f"Could not resolve mask for image: {image_path}")
                mask_path = found

        rows.append({
            "split": split_value,
            "image_path": image_path,
            "mask_path": mask_path,
            "image_id": image_path.stem
        })

    out_df = pd.DataFrame(rows)
    return out_df


def load_rgb_image(path):
    return Image.open(path).convert("RGB")


def load_binary_mask(path):
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    # Robust binarization
    arr = (arr > 127).astype(np.uint8)
    return arr


def letterbox_pil(image_pil, target_size=640, fill=(0, 0, 0)):
    w, h = image_pil.size
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = image_pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    meta = {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale": scale
    }
    return canvas, meta


def mask_to_letterbox(mask_arr, meta, target_size=640):
    mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8))
    resized = mask_img.resize((meta["new_w"], meta["new_h"]), Image.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(resized, (meta["pad_x"], meta["pad_y"]))
    out = (np.array(canvas) > 127).astype(np.uint8)
    return out


def undo_letterbox_mask(pred_640, meta):
    """
    pred_640: HxW uint8 mask at 640x640
    returns prediction resized back to original image size
    """
    y1 = meta["pad_y"]
    y2 = y1 + meta["new_h"]
    x1 = meta["pad_x"]
    x2 = x1 + meta["new_w"]

    crop = pred_640[y1:y2, x1:x2]
    crop_img = Image.fromarray((crop * 255).astype(np.uint8))
    orig = crop_img.resize((meta["orig_w"], meta["orig_h"]), Image.NEAREST)
    out = (np.array(orig) > 127).astype(np.uint8)
    return out


def preprocess_image(image_pil):
    boxed, meta = letterbox_pil(image_pil, target_size=IMAGE_SIZE)
    x = to_tensor(boxed)
    x = normalize(x, mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    return x.unsqueeze(0), meta


def try_build_model(builder, model_name, num_classes):
    # Robust against slight signature differences
    for kwargs in [
        {"model_name": model_name, "num_classes": num_classes},
        {"num_classes": num_classes, "model_name": model_name},
        {"num_classes": num_classes},
        {"model_name": model_name},
        {},
    ]:
        try:
            model = builder(**kwargs)
            return model
        except TypeError:
            continue
    raise RuntimeError("Could not build model with available builder signature.")


def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Remove DataParallel prefix if needed
    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state[k[len("module."):]] = v
        else:
            clean_state[k] = v

    model.load_state_dict(clean_state, strict=True)
    return ckpt


@torch.no_grad()
def predict_mask(model, image_pil):
    x, meta = preprocess_image(image_pil)
    x = x.to(DEVICE)

    out = model(x)

    # Robust output handling
    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, dict) and "logits" in out:
        logits = out["logits"]
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    # Make sure logits are 640x640 before inverse mapping
    logits = F.interpolate(
        logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    pred_640 = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    pred_orig = undo_letterbox_mask(pred_640, meta)
    return pred_orig


def compute_coverage(mask):
    return float(mask.mean())


def select_representative_cases(test_df):
    """
    Select one case from each coverage group.
    Picks the median-coverage sample within each group for stability.
    """
    records = []
    for _, row in test_df.iterrows():
        mask = load_binary_mask(row["mask_path"])
        coverage = compute_coverage(mask)
        records.append({
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "image_id": row["image_id"],
            "coverage": coverage
        })

    cov_df = pd.DataFrame(records)

    selected = []
    for label, low, high in COVERAGE_BINS:
        group = cov_df[(cov_df["coverage"] >= low) & (cov_df["coverage"] < high)].copy()

        if len(group) == 0:
            continue

        group = group.sort_values(by="coverage").reset_index(drop=True)
        mid_idx = len(group) // 2
        row = group.iloc[mid_idx].to_dict()
        row["coverage_label"] = label
        selected.append(row)

    # fallback if fewer than desired
    if len(selected) == 0:
        raise RuntimeError("No test cases could be selected.")

    return selected[:MAX_CASES]


def build_models():
    print("Loading baseline model...")
    baseline = try_build_model(build_baseline, MODEL_NAME, NUM_CLASSES)
    load_checkpoint_weights(baseline, BASELINE_CKPT, DEVICE)
    baseline.to(DEVICE).eval()

    print("Loading SFAF model...")
    sfaf = try_build_model(build_sfaf, MODEL_NAME, NUM_CLASSES)
    load_checkpoint_weights(sfaf, SFAF_CKPT, DEVICE)
    sfaf.to(DEVICE).eval()

    return baseline, sfaf


def make_figure(cases, baseline_model, sfaf_model):
    plt.rcParams["font.family"] = "Times New Roman"

    n_rows = len(cases)
    n_cols = 4  # Image, GT, Baseline, SFAF

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 3.2 * n_rows),
        squeeze=False
    )

    col_titles = ["Original image", "Ground truth", "SegFormer-B0", "SFAF"]
    for j in range(n_cols):
        axes[0, j].set_title(col_titles[j], fontsize=16)

    for i, case in enumerate(cases):
        image = load_rgb_image(case["image_path"])
        gt = load_binary_mask(case["mask_path"])

        pred_base = predict_mask(baseline_model, image)
        pred_sfaf = predict_mask(sfaf_model, image)

        image_np = np.array(image)

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_ylabel(
            case["coverage_label"],
            fontsize=14
        )

        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(pred_base, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].imshow(pred_sfaf, cmap="gray", vmin=0, vmax=1)

        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Keep the figure purely qualitative; the manuscript provides the caption.
    plt.tight_layout()

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Figure saved:")
    print(OUTPUT_PNG)
    print(OUTPUT_SVG)


def main():
    print_header()

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split file: {SPLIT_FILE}")
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {BASELINE_CKPT}")
    if not SFAF_CKPT.exists():
        raise FileNotFoundError(f"Missing SFAF checkpoint: {SFAF_CKPT}")

    df = pd.read_csv(SPLIT_FILE)
    df = resolve_image_mask_paths(df)

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test images found: {len(test_df)}")

    cases = select_representative_cases(test_df)
    print("\nSelected cases:")
    for idx, c in enumerate(cases, 1):
        print(f"{idx}. {c['image_id']} | coverage={c['coverage']*100:.2f}% | group={c['coverage_label']}")

    baseline_model, sfaf_model = build_models()
    make_figure(cases, baseline_model, sfaf_model)

    print("\nDONE.")


if __name__ == "__main__":
    main()