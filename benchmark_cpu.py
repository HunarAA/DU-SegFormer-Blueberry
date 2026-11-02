"""
CPU latency benchmark (proxy for edge deployment, where compute matters more than on a GPU).

All 7 paper models, seed-42 checkpoints, float32, batch 1, 640 x 640,
forward + resize to 640. The number of CPU threads is fixed and reported.

Interleaved timing: all models are loaded first, then timed in alternating
rounds (each round times every model a few times, in a rotating order).
This prevents laptop CPU heating / turbo changes from favouring the models
measured first.

Run with the laptop plugged in and other programs closed:
    python benchmark_cpu.py
Takes about 15-20 minutes. Results: runs/cpu_latency.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_final as bf          # model list (same as the GPU benchmark)
import train_repeats_resume as base

THREADS = 4
WARMUP = 3            # per model, before timing starts
ROUNDS = 6            # interleaved rounds
ITERS_PER_ROUND = 5   # timed runs per model per round  -> 30 timed runs per model
OUTPUT_CSV = base.RUNS_DIR / "cpu_latency.csv"


def run_once(model, x):
    logits = model(pixel_values=x).logits
    F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def main():
    torch.set_num_threads(THREADS)
    print("=" * 78)
    print("CPU LATENCY BENCHMARK (float32, batch 1, 640 x 640, forward + resize)")
    print("=" * 78)
    print(f"CPU: {platform.processor() or platform.machine()} | threads: {THREADS} | "
          f"{ROUNDS} interleaved rounds x {ITERS_PER_ROUND} runs per model")

    records = base.load_splits()
    image, _ = base.BlueberryDataset(records["test"])[0]
    x = image.unsqueeze(0)

    labels, models = [], []
    for label, builder, folder in bf.bench.MODEL_SPECS:
        model = builder(model_name="nvidia/mit-b0", num_classes=2)
        ckpt = torch.load(base.RUNS_DIR / folder / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        labels.append(label)
        models.append(model.eval())

    times = {label: [] for label in labels}
    with torch.inference_mode():
        for model in models:
            for _ in range(WARMUP):
                run_once(model, x)

        n = len(models)
        for r in range(ROUNDS):
            order = [(i + r) % n for i in range(n)]
            if r % 2 == 1:
                order.reverse()
            for i in order:
                for _ in range(ITERS_PER_ROUND):
                    start = time.perf_counter()
                    run_once(models[i], x)
                    times[labels[i]].append((time.perf_counter() - start) * 1000)
            print(f"Round {r + 1}/{ROUNDS} done")

    rows = []
    for label in labels:
        t = np.asarray(times[label])
        rows.append({"model": label, "mean_ms": t.mean(), "median_ms": float(np.median(t)),
                     "sd_ms": t.std(ddof=1), "fps": 1000 / t.mean(), "threads": THREADS, "runs": len(t)})

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    ref = rows[0]["mean_ms"]
    print("\n" + "=" * 78)
    print("SUMMARY (relative to the proposed model)")
    print("=" * 78)
    print(f"{'Model':<26}{'mean ms':>9}{'median':>8}{'SD':>7}{'FPS':>7}{'x slower':>10}")
    for r in rows:
        print(f"{r['model']:<26}{r['mean_ms']:>9.1f}{r['median_ms']:>8.1f}{r['sd_ms']:>7.1f}"
              f"{r['fps']:>7.2f}{r['mean_ms'] / ref:>10.2f}")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
