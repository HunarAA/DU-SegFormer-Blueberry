from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn.functional as F

from train import (
    BlueberryDataset,
    load_splits,
    DEVICE,
)

from model_baseline import build_model as build_baseline
from model_sfaf import build_model as build_sfaf


# ============================================================
# STABLE GPU LATENCY BENCHMARK
# ============================================================
#
# Purpose:
#   Verify the inference-speed result from efficiency_benchmark.py
#   using CUDA Events and a fixed 640x640 test image.
#
# This removes:
#   - image loading time
#   - DataLoader variation
#   - CPU timer jitter
#
# It measures:
#   1) model forward only
#   2) model forward + resizing logits to 640x640
#
# Run while:
#   - laptop is plugged in
#   - no other GPU-heavy programs are running
# ============================================================


PROJECT_DIR = Path(
    r"E:\Papers\AgriVision Blueberry"
)

RUNS_DIR = PROJECT_DIR / "runs"

OUTPUT_CSV = (
    RUNS_DIR
    / "latency_stability_benchmark.csv"
)

MODEL_NAME = "nvidia/mit-b0"

WARMUP_RUNS = 50

ITERATIONS_PER_ROUND = 200

ROUNDS = 5


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


def extract_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    return checkpoint


def load_model(spec):
    model = spec["builder"](
        model_name=MODEL_NAME,
        num_classes=2,
    )

    checkpoint = torch.load(
        spec["checkpoint"],
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        extract_state_dict(checkpoint)
    )

    model.to(DEVICE)
    model.eval()

    return model


def forward_only(model, x):
    outputs = model(
        pixel_values=x
    )

    return outputs.logits


def forward_plus_resize(model, x):
    outputs = model(
        pixel_values=x
    )

    logits = F.interpolate(
        outputs.logits,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return logits


def benchmark_cuda(
    model,
    x,
    function,
    measurement_name,
):
    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = function(
                model,
                x,
            )

    torch.cuda.synchronize()

    round_results = []
    all_times = []

    # --------------------------------------------------------
    # ROUNDS
    # --------------------------------------------------------

    with torch.inference_mode():

        for round_id in range(
            1,
            ROUNDS + 1
        ):
            times_ms = []

            for _ in range(
                ITERATIONS_PER_ROUND
            ):
                start_event = (
                    torch.cuda.Event(
                        enable_timing=True
                    )
                )

                end_event = (
                    torch.cuda.Event(
                        enable_timing=True
                    )
                )

                start_event.record()

                _ = function(
                    model,
                    x,
                )

                end_event.record()

                end_event.synchronize()

                elapsed_ms = (
                    start_event.elapsed_time(
                        end_event
                    )
                )

                times_ms.append(
                    elapsed_ms
                )

            times_ms = np.asarray(
                times_ms,
                dtype=np.float64,
            )

            round_mean = float(
                times_ms.mean()
            )

            round_median = float(
                np.median(
                    times_ms
                )
            )

            round_results.append(
                {
                    "round":
                        round_id,

                    "mean_ms":
                        round_mean,

                    "median_ms":
                        round_median,
                }
            )

            all_times.extend(
                times_ms.tolist()
            )

            print(
                f"  {measurement_name} "
                f"round {round_id}: "
                f"mean={round_mean:.3f} ms, "
                f"median={round_median:.3f} ms"
            )

    all_times = np.asarray(
        all_times,
        dtype=np.float64,
    )

    return {
        "measurement":
            measurement_name,

        "mean_ms":
            float(
                all_times.mean()
            ),

        "median_ms":
            float(
                np.median(
                    all_times
                )
            ),

        "std_ms":
            float(
                all_times.std(
                    ddof=1
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    all_times,
                    95
                )
            ),

        "fps":
            float(
                1000.0
                / all_times.mean()
            ),

        "samples":
            int(
                len(all_times)
            ),

        "round_means_ms":
            ";".join(
                f"{x['mean_ms']:.6f}"
                for x
                in round_results
            ),

        "round_medians_ms":
            ";".join(
                f"{x['median_ms']:.6f}"
                for x
                in round_results
            ),
    }


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "This benchmark requires CUDA."
        )

    print(
        "=" * 72
    )

    print(
        "STABLE CUDA-EVENT LATENCY BENCHMARK"
    )

    print(
        "=" * 72
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Warm-up runs: "
        f"{WARMUP_RUNS}"
    )

    print(
        f"Rounds: {ROUNDS}"
    )

    print(
        f"Iterations/round: "
        f"{ITERATIONS_PER_ROUND}"
    )

    # Fixed input shape => let cuDNN choose efficient algorithms.
    torch.backends.cudnn.benchmark = True

    records = load_splits()

    test_dataset = (
        BlueberryDataset(
            records["test"]
        )
    )

    image, _ = test_dataset[0]

    x = image.unsqueeze(0).to(
        DEVICE
    )

    print(
        f"Input shape: "
        f"{tuple(x.shape)}"
    )

    rows = []

    for spec in MODEL_SPECS:

        print(
            "\n" + "-" * 72
        )

        print(
            spec["label"]
        )

        print(
            "-" * 72
        )

        model = load_model(
            spec
        )

        forward_result = (
            benchmark_cuda(
                model,
                x,
                forward_only,
                "forward_only",
            )
        )

        total_result = (
            benchmark_cuda(
                model,
                x,
                forward_plus_resize,
                "forward_plus_resize",
            )
        )

        for result in [
            forward_result,
            total_result,
        ]:
            rows.append(
                {
                    "model":
                        spec["name"],

                    "label":
                        spec["label"],

                    **result,
                }
            )

        del model
        torch.cuda.empty_cache()

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
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 72
    )

    for measurement in [
        "forward_only",
        "forward_plus_resize",
    ]:

        baseline = next(
            row
            for row in rows
            if (
                row["model"]
                == "baseline"
                and row[
                    "measurement"
                ]
                == measurement
            )
        )

        sfaf = next(
            row
            for row in rows
            if (
                row["model"]
                == "sfaf"
                and row[
                    "measurement"
                ]
                == measurement
            )
        )

        latency_change = (
            100.0
            * (
                sfaf["mean_ms"]
                - baseline["mean_ms"]
            )
            / baseline["mean_ms"]
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
            f"\n{measurement}"
        )

        print(
            f"  Baseline: "
            f"{baseline['mean_ms']:.3f} ms "
            f"({baseline['fps']:.2f} FPS)"
        )

        print(
            f"  SFAF:     "
            f"{sfaf['mean_ms']:.3f} ms "
            f"({sfaf['fps']:.2f} FPS)"
        )

        print(
            f"  SFAF latency change: "
            f"{latency_change:+.2f}%"
        )

        print(
            f"  SFAF FPS change:     "
            f"{fps_change:+.2f}%"
        )

    print(
        f"\nSaved:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
