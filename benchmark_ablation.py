"""
Efficiency benchmark for the decoder ablation (Step 1).

Measures, in ONE session on the same GPU, all six models:
    SegFormer-B0 (256-ch)  - paper baseline
    SFAF (gate)            - paper proposed model
    SegFormer-B0 (64-ch)   - ablation
    SFAF + addition        - ablation
    SFAF + concatenation   - ablation
    SFAF without reinjection - ablation

For each model (seed-42 best checkpoint, batch size 1, 640 x 640):
    - parameters and parameter memory
    - profiler-supported FLOPs (forward + resize)
    - peak allocated GPU memory during inference
    - latency and FPS with CUDA events (50 warm-up, 5 rounds x 200 iterations),
      both forward-only and forward + resize, exactly as in
      benchmark_latency_stable.py

Because all six are measured together, the paper's two models are re-measured
too, so every number in the table is directly comparable.

Run with the laptop plugged in and nothing else using the GPU:
    python benchmark_ablation.py
Results: runs/ablation_efficiency.csv
"""

import os
# Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.

import csv

import numpy as np
import torch
import torch.nn.functional as F

import train_repeats_resume as base
import model_ablation as abl
from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


RUNS_DIR = base.RUNS_DIR
DEVICE = base.DEVICE
OUTPUT_CSV = RUNS_DIR / "ablation_efficiency.csv"

MODEL_NAME = "nvidia/mit-b0"
WARMUP_RUNS = 50
ITERATIONS_PER_ROUND = 200
ROUNDS = 5

MODEL_SPECS = [
    ("SegFormer-B0 (256-ch)", build_baseline, "baseline_segformer_b0"),
    ("SFAF (gate)", build_sfaf, "proposed_sfaf_b0"),
    ("SegFormer-B0 (64-ch)", abl.build_segformer_dec64, "ablation_segformer_b0_dec64_seed42"),
    ("SFAF + addition", abl.build_sfaf_add, "ablation_sfaf_add_seed42"),
    ("SFAF + concatenation", abl.build_sfaf_concat, "ablation_sfaf_concat_seed42"),
    ("SFAF no reinjection", abl.build_sfaf_noreinject, "ablation_sfaf_noreinject_seed42"),
]


def load_model(builder, run_folder):
    model = builder(model_name=MODEL_NAME, num_classes=2)
    checkpoint_path = RUNS_DIR / run_folder / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def forward_only(model, x):
    return model(pixel_values=x).logits


def forward_plus_resize(model, x):
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def profiler_gflops(model, x):
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(activities=activities, with_flops=True) as prof:
            forward_plus_resize(model, x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

    flops = sum(e.flops for e in prof.key_averages() if e.flops)
    return flops / 1e9


def peak_inference_memory_gb(model, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(10):
            forward_plus_resize(model, x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def cuda_latency(model, x, function):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            function(model, x)
    torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(ROUNDS):
            for _ in range(ITERATIONS_PER_ROUND):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function(model, x)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))

    times = np.asarray(times, dtype=np.float64)
    return float(times.mean()), float(times.std(ddof=1)), float(1000.0 / times.mean())


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA.")

    torch.backends.cudnn.benchmark = True

    print("=" * 78)
    print("DECODER ABLATION - EFFICIENCY BENCHMARK")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warm-up: {WARMUP_RUNS}, rounds: {ROUNDS} x {ITERATIONS_PER_ROUND}")

    records = base.load_splits()
    image, _ = base.BlueberryDataset(records["test"])[0]
    x = image.unsqueeze(0).to(DEVICE)
    print(f"Input: {tuple(x.shape)}")

    rows = []
    for label, builder, run_folder in MODEL_SPECS:
        print("\n" + "-" * 78)
        print(label)
        print("-" * 78)

        model = load_model(builder, run_folder)
        params = sum(p.numel() for p in model.parameters())
        param_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        gflops = profiler_gflops(model, x)
        peak_gb = peak_inference_memory_gb(model, x)
        fwd_ms, fwd_sd, fwd_fps = cuda_latency(model, x, forward_only)
        tot_ms, tot_sd, tot_fps = cuda_latency(model, x, forward_plus_resize)

        row = {
            "model": label,
            "parameters_m": params / 1e6,
            "parameter_memory_mib": param_mib,
            "peak_inference_gpu_gb": peak_gb,
            "profiler_gflops": gflops,
            "forward_latency_ms": fwd_ms,
            "forward_latency_sd_ms": fwd_sd,
            "forward_fps": fwd_fps,
            "total_latency_ms": tot_ms,
            "total_latency_sd_ms": tot_sd,
            "total_fps": tot_fps,
            "checkpoint": run_folder,
        }
        rows.append(row)

        print(f"Params {params / 1e6:.3f} M | GFLOPs {gflops:.2f} | Peak GPU {peak_gb:.3f} GB")
        print(f"Forward only:     {fwd_ms:.2f} ms ({fwd_fps:.1f} FPS)")
        print(f"Forward + resize: {tot_ms:.2f} ms ({tot_fps:.1f} FPS)")

        del model
        torch.cuda.empty_cache()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("SUMMARY (forward + resize, batch 1, 640 x 640)")
    print("=" * 78)
    print(f"{'Model':<26}{'Params(M)':>10}{'GFLOPs':>9}{'PeakGB':>9}{'ms':>8}{'FPS':>8}")
    for r in rows:
        print(f"{r['model']:<26}{r['parameters_m']:>10.3f}{r['profiler_gflops']:>9.2f}"
              f"{r['peak_inference_gpu_gb']:>9.3f}{r['total_latency_ms']:>8.2f}{r['total_fps']:>8.1f}")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
