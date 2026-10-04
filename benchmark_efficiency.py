from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import (
    BlueberryDataset,
    load_splits,
    DEVICE,
)

from model_baseline import (
    build_model as build_baseline
)

from model_sfaf import (
    build_model as build_sfaf
)


# ============================================================
# CONTROLLED INFERENCE-EFFICIENCY BENCHMARK
# ============================================================
#
# Purpose:
#   Compare SegFormer-B0 and SFAF under exactly the same
#   inference conditions.
#
# Measures:
#   - Parameters
#   - Peak inference GPU memory
#   - Mean latency per image
#   - Median latency per image
#   - FPS
#   - Optional profiler FLOPs estimate
#
# Notes:
#   - Uses batch size 1.
#   - Uses the fixed TEST images only as inference inputs.
#   - No training or tuning is performed.
#   - The first warm-up passes are excluded from timing.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"E:\Papers\AgriVision Blueberry"
)

RUNS_DIR = (
    PROJECT_DIR / "runs"
)

OUTPUT_CSV = (
    RUNS_DIR
    / "efficiency_benchmark.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "nvidia/mit-b0"

BATCH_SIZE = 1

WARMUP_RUNS = 30

# Use all 240 test images for timing.
MAX_TIMED_IMAGES = None

# Repeat each image once during the main timing loop.
# Keeping this at 1 avoids unnecessarily long benchmarking.
TIMING_REPEATS = 1

# Enable a one-pass PyTorch profiler FLOPs estimate.
# This estimate includes only operations for which the
# profiler can calculate FLOPs, so treat it as supportive,
# not as a guaranteed complete theoretical FLOPs count.
ESTIMATE_FLOPS = True


MODEL_SPECS = [
    {
        "name": "baseline",
        "label": "SegFormer-B0",
        "builder": build_baseline,
        "checkpoint": (
            RUNS_DIR
            / "baseline_segformer_b0"
            / "best_model.pt"
        ),
    },
    {
        "name": "sfaf",
        "label": "SFAF",
        "builder": build_sfaf,
        "checkpoint": (
            RUNS_DIR
            / "proposed_sfaf_b0"
            / "best_model.pt"
        ),
    },
]


# ============================================================
# HELPERS
# ============================================================

def extract_state_dict(
    checkpoint
):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    return checkpoint


def load_trained_model(
    spec
):
    checkpoint_path = (
        spec["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(
        f"\nLoading {spec['label']}..."
    )

    model = spec["builder"](
        model_name=MODEL_NAME,
        num_classes=2,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        extract_state_dict(
            checkpoint
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


def parameter_count(
    model
):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


def model_parameter_memory_mb(
    model
):
    total_bytes = sum(
        p.numel()
        * p.element_size()
        for p in model.parameters()
    )

    return (
        total_bytes
        / 1024**2
    )


def forward_inference(
    model,
    images
):
    """
    Matches the actual segmentation inference path used
    in evaluation, including resizing logits back to the
    640 x 640 input size.
    """

    outputs = model(
        pixel_values=images
    )

    logits = F.interpolate(
        outputs.logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return logits


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def estimate_supported_flops(
    model,
    sample
):
    if not ESTIMATE_FLOPS:
        return None

    try:
        from torch.profiler import (
            profile,
            ProfilerActivity,
        )

        activities = [
            ProfilerActivity.CPU
        ]

        if DEVICE.type == "cuda":
            activities.append(
                ProfilerActivity.CUDA
            )

        synchronize()

        with torch.no_grad():
            with profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = forward_inference(
                    model,
                    sample
                )

                synchronize()

        supported_flops = sum(
            event.flops
            for event
            in prof.key_averages()
            if event.flops is not None
        )

        if supported_flops <= 0:
            return None

        return (
            supported_flops
            / 1e9
        )

    except Exception as error:
        print(
            "FLOPs estimation unavailable:"
        )
        print(
            f"  {type(error).__name__}: "
            f"{error}"
        )

        return None


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(
    spec,
    loader
):
    model = load_trained_model(
        spec
    )

    total_params, trainable_params = (
        parameter_count(
            model
        )
    )

    parameter_memory_mb = (
        model_parameter_memory_mb(
            model
        )
    )

    # --------------------------------------------------------
    # GET ONE REAL TEST IMAGE FOR WARM-UP
    # --------------------------------------------------------

    first_images, _ = next(
        iter(loader)
    )

    first_images = first_images.to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    print(
        f"Warm-up: "
        f"{WARMUP_RUNS} runs..."
    )

    with torch.inference_mode():
        for _ in range(
            WARMUP_RUNS
        ):
            _ = forward_inference(
                model,
                first_images
            )

    synchronize()

    # --------------------------------------------------------
    # FLOPs ESTIMATE
    # --------------------------------------------------------

    profiler_gflops = (
        estimate_supported_flops(
            model,
            first_images
        )
    )

    # --------------------------------------------------------
    # RESET PEAK MEMORY AFTER MODEL IS LOADED
    # --------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # CONTROLLED TIMING
    # --------------------------------------------------------

    print(
        "Timing inference..."
    )

    latencies_ms = []

    image_count = 0

    with torch.inference_mode():

        for images, _ in loader:

            if (
                MAX_TIMED_IMAGES
                is not None
                and image_count
                >= MAX_TIMED_IMAGES
            ):
                break

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            for _ in range(
                TIMING_REPEATS
            ):
                synchronize()

                start = (
                    time.perf_counter()
                )

                _ = forward_inference(
                    model,
                    images
                )

                synchronize()

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000

                latencies_ms.append(
                    elapsed_ms
                )

            image_count += (
                images.shape[0]
            )

    latencies_ms = np.array(
        latencies_ms,
        dtype=np.float64,
    )

    mean_latency_ms = (
        latencies_ms.mean()
    )

    median_latency_ms = (
        np.median(
            latencies_ms
        )
    )

    std_latency_ms = (
        latencies_ms.std(
            ddof=1
        )
        if len(latencies_ms) > 1
        else 0.0
    )

    fps = (
        1000.0
        / mean_latency_ms
    )

    if DEVICE.type == "cuda":
        peak_gpu_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

    else:
        peak_gpu_gb = 0.0

    result = {
        "model":
            spec["name"],

        "label":
            spec["label"],

        "parameters_m":
            total_params
            / 1e6,

        "trainable_parameters_m":
            trainable_params
            / 1e6,

        "parameter_memory_mb":
            parameter_memory_mb,

        "peak_inference_gpu_gb":
            peak_gpu_gb,

        "mean_latency_ms":
            mean_latency_ms,

        "median_latency_ms":
            median_latency_ms,

        "std_latency_ms":
            std_latency_ms,

        "fps":
            fps,

        "timed_images":
            image_count,

        "warmup_runs":
            WARMUP_RUNS,

        "batch_size":
            BATCH_SIZE,

        "profiler_supported_gflops":
            (
                profiler_gflops
                if profiler_gflops
                is not None
                else ""
            ),
    }

    print(
        "\n"
        f"{spec['label']} results"
    )

    print(
        "-" * 72
    )

    print(
        f"Parameters:             "
        f"{result['parameters_m']:.3f} M"
    )

    print(
        f"Parameter memory:       "
        f"{result['parameter_memory_mb']:.2f} MB"
    )

    print(
        f"Peak inference GPU:     "
        f"{result['peak_inference_gpu_gb']:.3f} GB"
    )

    print(
        f"Mean latency/image:     "
        f"{result['mean_latency_ms']:.2f} ms"
    )

    print(
        f"Median latency/image:   "
        f"{result['median_latency_ms']:.2f} ms"
    )

    print(
        f"Latency SD:             "
        f"{result['std_latency_ms']:.2f} ms"
    )

    print(
        f"Throughput:             "
        f"{result['fps']:.2f} FPS"
    )

    if profiler_gflops is not None:
        print(
            f"Profiler-supported FLOPs: "
            f"{profiler_gflops:.3f} GFLOPs"
        )

        print(
            "  NOTE: This is a PyTorch-profiler "
            "supported-operations estimate."
        )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=" * 72
    )

    print(
        "CONTROLLED INFERENCE EFFICIENCY BENCHMARK"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    records = load_splits()

    test_dataset = (
        BlueberryDataset(
            records["test"]
        )
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                DEVICE.type
                == "cuda"
            ),
        )
    )

    print(
        f"Test images available: "
        f"{len(test_dataset)}"
    )

    print(
        f"Timing batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Warm-up runs/model: "
        f"{WARMUP_RUNS}"
    )

    results = []

    for spec in MODEL_SPECS:
        result = benchmark_model(
            spec,
            test_loader,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # RELATIVE COMPARISON
    # --------------------------------------------------------

    baseline = results[0]
    sfaf = results[1]

    parameter_reduction = (
        100.0
        * (
            baseline[
                "parameters_m"
            ]
            - sfaf[
                "parameters_m"
            ]
        )
        / baseline[
            "parameters_m"
        ]
    )

    memory_reduction = (
        100.0
        * (
            baseline[
                "peak_inference_gpu_gb"
            ]
            - sfaf[
                "peak_inference_gpu_gb"
            ]
        )
        / max(
            baseline[
                "peak_inference_gpu_gb"
            ],
            1e-12,
        )
    )

    latency_change = (
        100.0
        * (
            sfaf[
                "mean_latency_ms"
            ]
            - baseline[
                "mean_latency_ms"
            ]
        )
        / baseline[
            "mean_latency_ms"
        ]
    )

    fps_change = (
        100.0
        * (
            sfaf["fps"]
            - baseline["fps"]
        )
        / baseline["fps"]
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "SFAF RELATIVE TO SEGFORMER-B0"
    )

    print(
        "=" * 72
    )

    print(
        f"Parameter reduction:      "
        f"{parameter_reduction:+.2f}%"
    )

    print(
        f"Peak GPU memory change:   "
        f"{-memory_reduction:+.2f}% "
        f"(negative = lower)"
    )

    print(
        f"Mean latency change:      "
        f"{latency_change:+.2f}% "
        f"(negative = faster)"
    )

    print(
        f"FPS change:               "
        f"{fps_change:+.2f}%"
    )

    if (
        baseline[
            "profiler_supported_gflops"
        ] != ""
        and sfaf[
            "profiler_supported_gflops"
        ] != ""
    ):
        flop_change = (
            100.0
            * (
                sfaf[
                    "profiler_supported_gflops"
                ]
                - baseline[
                    "profiler_supported_gflops"
                ]
            )
            / baseline[
                "profiler_supported_gflops"
            ]
        )

        print(
            f"Profiler FLOPs change:    "
            f"{flop_change:+.2f}%"
        )

    print(
        f"\nSaved:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        "\nRun this benchmark when the laptop is "
        "otherwise idle and plugged in."
    )


if __name__ == "__main__":
    main()
