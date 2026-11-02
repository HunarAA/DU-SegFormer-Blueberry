# DU-SegFormer: Learned Dynamic Upsampling for Lightweight Blueberry Segmentation

Code, fixed data splits, and result files for the paper

> **Learned Dynamic Upsampling for Lightweight Blueberry Segmentation with Bias-Corrected Training and Leakage-Aware Evaluation**
> Hunar Abubakir Ahmed, University of Raparin (manuscript under review)

DU-SegFormer is a SegFormer-B0 variant whose 64-channel decoder replaces bilinear upsampling with
learned dynamic point sampling ([DySample](https://arxiv.org/abs/2308.15085)) and predicts at 1/2
instead of 1/4 of the input resolution. The repository also contains the training-protocol study,
the comparison with six other models, and the leakage / split-variability study.

## Main results (AgriVision DB-1, source-video-disjoint test set, mean ± SD over 3 seeds)

| Model | IoU (%) | Per-image IoU (%) | Params (M) | GMACs | GPU (ms) | CPU (ms) |
|---|---|---|---|---|---|---|
| **DU-SegFormer (proposed)** | **80.91 ± 0.04** | **74.63 ± 0.08** | 3.46 | 6.64 | 11.91 | 128.6 |
| SegFormer-B0, 64-ch decoder | 80.21 ± 0.11 | 73.73 ± 0.18 | 3.37 | 6.44 | 9.67 | 114.0 |
| SegFormer-B0, 256-ch decoder | 80.50 ± 0.15 | 74.20 ± 0.13 | 3.71 | 13.05 | 13.34 | 223.0 |
| U-Net + ResNet-34 | 81.20 ± 0.16 | 74.84 ± 0.15 | 24.51 | 60.13 | 15.19 | 412.4 |
| DeepLabV3+ + ResNet-50 | 81.41 ± 0.10 | 75.17 ± 0.25 | 40.35 | 108.04 | 24.90 | 697.7 |
| LR-ASPP + MobileNetV3-L | 73.72 ± 0.08 | 65.80 ± 0.17 | 3.22 | 3.06 | 5.49 | 51.4 |
| DeepLabV3 + MobileNetV3-L | 72.20 ± 0.06 | 63.63 ± 0.19 | 11.02 | 15.34 | 7.68 | 125.4 |

640 × 640 input, batch 1. GPU: RTX 4060 Laptop (FP32). CPU: Intel Core Ultra 7 155H (FP32, 4 threads).

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.9.19, PyTorch 2.4.1 (CUDA 11.8), torchvision 0.19.1, Transformers 4.44.0 on an
8 GB laptop GPU. The ImageNet-pretrained MiT-B0 (`nvidia/mit-b0`), ResNet and MobileNetV3 weights are
downloaded automatically on first use. After that, `HF_HUB_OFFLINE=1` lets the scripts run without internet.

### Data

Download the **AgriVision DB-1** subset from Figshare
([doi:10.6084/m9.figshare.29222462](https://doi.org/10.6084/m9.figshare.29222462)) and place it as

```
DB-1/
  Images/   Camera 1 Video (1)_1.bmp, ...
  Masks/    Camera 1 Video (1)_1.png, ...
```

If the data is stored elsewhere, set `BLUEBERRY_DATA` to the folder containing `Images/` and `Masks/`
(and optionally `BLUEBERRY_ROOT` for where `splits.csv` and `runs/` live). See `paths.py`.

### Fixed split

`splits.csv` is the source-video-disjoint split used in the paper (833 / 122 / 240 images from
206 / 26 / 56 camera–video groups; no video appears in more than one subset). Run `python split.py`
to verify it against your copy of the dataset; it never overwrites the file.

## Reproducing the paper

All scripts write checkpoints and CSV summaries to `runs/`, skip runs that are already finished, and
resume an interrupted run from its last completed epoch (just run the same command again).

| Step | Command | Produces |
|---|---|---|
| 1 | `python train_recipe.py` | Protocol study (Table 3); SegFormer-B0 64-ch and 256-ch, final protocol, 3 seeds |
| 2 | `python train_novel.py` | DU-SegFormer and upsampling ablation, 3 seeds (Tables 4–6) |
| 3 | `python train_baselines.py` | U-Net, DeepLabV3+, LR-ASPP, DeepLabV3-MobileNetV3, 3 seeds (Tables 4, 6) |
| 4 | `python train_leakage.py` | Controlled leakage and random-split study (Table 8); needs step 1 |
| 5 | `python benchmark_final.py` and `python benchmark_cpu.py` | GPU and CPU efficiency (Table 7) |
| 6 | `python benchmark_novel.py` | GPU latency of the ablation variants (Table 5) |
| 7 | `python make_figure_qualitative.py` | Qualitative figure (Fig. 4) |

Steps 5–7 use the seed-42 checkpoints from steps 1–3. On an RTX 4060 Laptop GPU, one SegFormer-B0
run takes about 45–70 minutes; U-Net and DeepLabV3+ take about 1.2 and 2 hours per run.

### Initial protocol (Sections 4.1 and 4.3 of the paper)

The first version of the study used Adam with a fixed learning rate and an inverse-frequency weighted
loss. These runs are needed only to reproduce the protocol diagnosis and the preliminary fusion-decoder
results:

| Command | Produces |
|---|---|
| `python train_repeats_resume.py` | SegFormer-B0 (256-ch) and SFAF fusion decoder, seeds 42 / 7 / 123 |
| `python train_ablation.py` | 64-ch decoder and fusion variants (addition, concatenation, no reinjection) |
| `SEED=42 python train_standard_baselines_fixed.py` (also 7, 123) | U-Net and DeepLabV3+ |
| `python diagnostics.py` | Decision-threshold diagnosis and coverage breakdown |
| `python benchmark_ablation.py` | Efficiency of the fusion variants |

On Windows (Command Prompt), set the seed with `set SEED=7` before the command.

## Repository contents

| File | Purpose |
|---|---|
| `model_novel.py` | **DU-SegFormer** (DySample decoder) and the bilinear 1/2-resolution control |
| `model_ablation.py` | 64-channel SegFormer-B0 decoder and fusion-decoder variants |
| `model_baseline.py`, `model_sfaf.py` | Standard SegFormer-B0 and the SFAF fusion decoder |
| `train_repeats_resume.py` | Shared data loading, letterboxing, metrics and the initial-protocol training loop |
| `train_recipe.py` | Final training protocol (AdamW, warm-up + poly, CE + Dice, augmentation) and evaluation |
| `train_novel.py`, `train_baselines.py`, `train_leakage.py` | Experiments of the paper (final protocol) |
| `train_standard_baselines_fixed.py`, `train_ablation.py` | Initial-protocol experiments; U-Net and DeepLabV3+ definitions |
| `diagnostics.py` | Threshold calibration diagnosis and per-coverage evaluation |
| `benchmark_*.py` | Parameter, MAC, GPU-latency and CPU-latency measurements |
| `split.py`, `splits.csv`, `paths.py` | Data split and paths |
| `results/` | Result CSVs behind the tables of the paper |

## Pretrained models

Seed-42 checkpoints of the final models are provided under
[Releases](../../releases). Load them with the matching builder, for example:

```python
import torch
from model_novel import build_dysample_half      # DU-SegFormer
model = build_dysample_half()
state = torch.load("recipe_n2_dysample_half_seed42.pt", map_location="cpu")
model.load_state_dict(state["model_state_dict"])
```

## Citation

If you use this code or the split, please cite the paper (citation will be added after publication)
and the AgriVision dataset:

> Owais, M., et al. AgriVision: A benchmark dataset for advancing real-world robotic vision in densely
> fruited blueberry crop. *Scientific Data* 12, 2014 (2025).

## License

Code: MIT License (see `LICENSE`). The AgriVision dataset is distributed under its own licence and is
not included in this repository.
