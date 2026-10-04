# Efficient Blueberry Segmentation Using a Lightweight Adaptive-Fusion SegFormer Decoder

Official code for the paper:

> **Efficient Blueberry Segmentation Using a Lightweight Adaptive-Fusion SegFormer Decoder**
> Hunar A. Ahmed, Software Engineering Department, University of Raparin, Rania, Iraq

This repository contains the implementation of the **Small-Fruit Adaptive Fusion (SFAF)** decoder for SegFormer-B0, together with the baseline models, the fixed source-video-disjoint data split, and the scripts used for training, evaluation, efficiency benchmarking, and figure generation.

## Overview

SFAF keeps the ImageNet-pretrained MiT-B0 encoder of SegFormer unchanged and replaces the standard MLP decoder with a lightweight decoder that:

1. Projects the four encoder feature maps (C1–C4) to 64 channels and upsamples them to the C1 resolution.
2. Forms a **detail stream** from the shallow stages (C1, C2) and a **semantic stream** from the deeper stages (C3, C4), each processed with depthwise separable convolution.
3. Combines the two streams with a learned gate: `F = G ⊙ D + (1 − G) ⊙ S`.
4. Reinjects the projected C1 features, applies a lightweight refinement block, and outputs a two-class (blueberry / background) segmentation.

## Main Results

**Segmentation performance** on the AgriVision DB-1 test set (mean ± std over seeds 42, 7, and 123):

| Model | Params (M) | Test IoU (%) | Dice (%) |
|---|---|---|---|
| SegFormer-B0 | 3.71 | 75.19 ± 0.37 | 85.83 ± 0.24 |
| **SFAF (ours)** | **3.39** | 74.20 ± 1.16 | 85.19 ± 0.77 |
| U-Net + ResNet-34 | 24.51 | 76.79 ± 0.93 | 86.87 ± 0.59 |
| DeepLabV3+ + ResNet-50 | 40.35 | 76.77 ± 0.49 | 86.86 ± 0.32 |

**Computational efficiency** (640 × 640 input, batch size 1, NVIDIA RTX 4060 Laptop GPU):

| Model | FLOPs* (G) | Peak GPU mem. (GiB) | Latency (ms) | FPS |
|---|---|---|---|---|
| SegFormer-B0 | 26.11 | 0.260 | 13.47 | 74.24 |
| **SFAF (ours)** | **14.13** | **0.121** | **10.96** | **91.23** |

\* Profiler-supported FLOPs, including the forward pass and output resizing.

SFAF trades a small drop in accuracy for 45.89% fewer FLOPs, 53.49% lower peak GPU memory, and 22.88% higher throughput compared with SegFormer-B0.

## Repository Structure

| File | Purpose |
|---|---|
| `model_sfaf.py` | Proposed SFAF model (MiT-B0 encoder + SFAF decoder) |
| `model_baseline.py` | SegFormer-B0 baseline |
| `split.py` | Generates the source-video-disjoint train/val/test split |
| `splits.csv` | The fixed split used in all experiments |
| `train.py` | Training script for SegFormer-B0 and SFAF |
| `train_standard_baselines_fixed.py` | Training script for the U-Net and DeepLabV3+ baselines |
| `train_unet_resnet34_seed7.py`, `train_unet_resnet34_seed123.py` | U-Net + ResNet-34 seed repetitions |
| `train_deeplabv3plus_resnet50_seed7.py`, `train_deeplabv3plus_resnet50_seed123.py` | DeepLabV3+ + ResNet-50 seed repetitions |
| `train_repeats_resume.py` | Seed repetitions for SegFormer-B0 and SFAF (with resume support) |
| `analyze.py` | Per-image, coverage-group, camera, and grouped-bootstrap analysis |
| `benchmark_efficiency.py` | Parameters, parameter memory, peak GPU memory, and FLOPs |
| `benchmark_latency_stable.py` | Latency and throughput measured with CUDA events |
| `make_figure1_final.py` | Figure 1 (dataset examples) |
| `make_sfaf_architecture.py` | Figure 2 (SFAF architecture diagram) |
| `make_figure_test_predictions_final.py` | Figure 3 (qualitative predictions) |

## Environment

The experiments were run with:

- Python 3.9.19
- PyTorch 2.4.1 with CUDA 11.8
- torchvision 0.19.1
- Transformers 4.44.0

Example setup with conda:

```bash
conda create -n sfaf python=3.9.19
conda activate sfaf
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.0 numpy pandas matplotlib pillow
```

## Dataset

This study uses the **DB-1** subset of the AgriVision blueberry dataset (1,195 annotated images), which is publicly available on Figshare:

https://doi.org/10.6084/m9.figshare.29222462

Download the dataset and place the DB-1 folder in the root of this repository as `DB-1/`.

The dataset is not redistributed here. Please cite the original dataset paper if you use it (see Citation below).

### Data split

The file `splits.csv` contains the exact source-video-disjoint split used in the paper: 833 training, 122 validation, and 240 test images, with no camera–video group shared between subsets. Using this file is recommended to reproduce the reported results. `split.py` shows how the split was generated.

## Usage

### 1. Train SegFormer-B0 and SFAF

```bash
python train.py
```

Additional seeds:

```bash
python train_repeats_resume.py
```

### 2. Train the CNN baselines

```bash
python train_standard_baselines_fixed.py
python train_unet_resnet34_seed7.py
python train_unet_resnet34_seed123.py
python train_deeplabv3plus_resnet50_seed7.py
python train_deeplabv3plus_resnet50_seed123.py
```

Checkpoints and logs are saved to the `runs/` folder.

### 3. Evaluate and analyze

```bash
python analyze.py
```

### 4. Benchmark efficiency

```bash
python benchmark_efficiency.py
python benchmark_latency_stable.py
```

### 5. Reproduce the figures

```bash
python make_figure1_final.py
python make_sfaf_architecture.py
python make_figure_test_predictions_final.py
```

## Training Settings

| Setting | Value |
|---|---|
| Input size | 640 × 640 (aspect-ratio-preserving letterbox, padding ignored with label 255) |
| Normalization | ImageNet mean and std |
| Data augmentation | None |
| Optimizer | Adam, learning rate 1 × 10⁻⁴ (fixed) |
| Batch size | 4 |
| Epochs | Up to 100, early stopping with patience 15 on validation IoU |
| Loss | Cross-entropy with per-batch inverse-frequency class weights |
| Mixed precision | Enabled |
| Seeds | 42, 7, 123 |

## Citation

If you use this code, please cite:

```bibtex
@article{ahmed_sfaf_blueberry,
  title   = {Efficient Blueberry Segmentation Using a Lightweight Adaptive-Fusion SegFormer Decoder},
  author  = {Ahmed, Hunar A.},
  journal = {Under review},
  year    = {2026}
}
```

Please also cite the AgriVision dataset:

```bibtex
@article{owais2025agrivision,
  title   = {AgriVision: A benchmark dataset for advancing real-world robotic vision in densely fruited blueberry crop},
  author  = {Owais, M. and others},
  journal = {Scientific Data},
  volume  = {12},
  number  = {1},
  pages   = {2014},
  year    = {2025}
}
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Hunar A. Ahmed: hunar.abubakir@uor.edu.krd
