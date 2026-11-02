"""
Final efficiency benchmark: all models of the paper, final recipe, seed-42 checkpoints.

Same measurements as benchmark_ablation.py (CUDA events, batch 1, 640 x 640,
50 warm-up, 5 rounds x 200 iterations, forward + resize to 640).

Run with the laptop plugged in and nothing else using the GPU:
    python benchmark_final.py
Results: runs/final_efficiency.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import benchmark_ablation as bench
import model_ablation as abl
import model_novel as mn
import train_baselines as tb
from model_baseline import build_model as build_baseline

bench.MODEL_SPECS = [
    ("Proposed (DySample+1/2)", mn.build_dysample_half, "recipe_n2_dysample_half_seed42"),
    ("SegFormer-B0 64-ch", abl.build_segformer_dec64, "recipe_r5_adamw_cedice_aug_seed42"),
    ("SegFormer-B0 256-ch", build_baseline, "recipe_r5_dec256_seed42"),
    ("U-Net ResNet-34", tb.build_unet, "recipe_base_unet_r34_seed42"),
    ("DeepLabV3+ ResNet-50", tb.build_deeplabv3p, "recipe_base_deeplabv3p_r50_seed42"),
    ("LR-ASPP MobileNetV3-L", tb.build_lraspp, "recipe_base_lraspp_mbv3_seed42"),
    ("DeepLabV3 MobileNetV3-L", tb.build_deeplabv3_mbv3, "recipe_base_deeplabv3_mbv3_seed42"),
]
bench.OUTPUT_CSV = bench.RUNS_DIR / "final_efficiency.csv"

if __name__ == "__main__":
    bench.main()
