"""
Efficiency benchmark for the final-recipe models and the DySample variants.

Same measurements as benchmark_ablation.py (CUDA events, batch 1, 640 x 640,
50 warm-up, 5 rounds x 200 iterations, forward + resize to 640), using the
seed-42 checkpoints trained with the final recipe.

Run with the laptop plugged in and nothing else using the GPU:
    python benchmark_novel.py
Results: runs/novel_efficiency.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import benchmark_ablation as bench
import model_ablation as abl
import model_novel as mn
from model_baseline import build_model as build_baseline

bench.MODEL_SPECS = [
    ("SegFormer-B0 256-ch", build_baseline, "recipe_r5_dec256_seed42"),
    ("SegFormer-B0 64-ch", abl.build_segformer_dec64, "recipe_r5_adamw_cedice_aug_seed42"),
    ("DySample (n1)", mn.build_dysample, "recipe_n1_dysample_seed42"),
    ("DySample + 1/2 res (n2)", mn.build_dysample_half, "recipe_n2_dysample_half_seed42"),
    ("Bilinear + 1/2 res (n3)", mn.build_bilinear_half, "recipe_n3_bilinear_half_seed42"),
]
bench.OUTPUT_CSV = bench.RUNS_DIR / "novel_efficiency.csv"

if __name__ == "__main__":
    bench.main()
