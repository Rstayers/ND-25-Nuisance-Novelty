<h1 align="center">Evaluating Nuisance Novelty to Expose Gaps in<br>Open Set Reliability</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/ECCV-2026-blue.svg" alt="ECCV 2026"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python 3.8+"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-1.10+-orange.svg" alt="PyTorch"></a>

</p>

<p align="center">
  <b>A dataset-agnostic framework for generating Locally Nuisanced (LN) benchmarks<br>that isolate novelty detector failure from classifier failure in Open Set Recognition.</b>
</p>

<p align="center">
  <a href="#">[Paper]</a> •
  <a href="https://www.kaggle.com/datasets/rexstayersuprick/imagenet-ln">Datasets</a> •
</p>

---


<p align="center"><i>
<b>Nuisance Novelty</b> arises when a known input is correctly classified but incorrectly rejected as unknown. Our LN framework generates benchmarks that preserve classifier accuracy while exposing detector-specific failures invisible in prior evaluations.
</i></p>

---

## Abstract

Open Set Recognition (OSR) requires models to correctly classify known samples while rejecting unknowns, typically via a closed set classifier paired with a novelty detector (or post-processor) that accepts or rejects the input. Unfortunately, current robustness benchmarks rely on global corruptions that degrade the classifier and post-processor simultaneously. This combines failure modes and obscures nuisance novelty–a state where the classifier remains accurate, yet the post-processor erroneously rejects the input–a critical failure mode in safety-sensitive domains such as autonomous driving and robotics. In contrast, we argue that OSR reliability analysis must be isolated at the post-processor level. We introduce a framework that generates targeted local perturbations using reconstruction-based competency masks, preserving classification signals while testing post-processor robustness. Evaluating 50 backbone-post-processor configurations across three datasets reveals that robustness to global corruption is orthogonal to local nuisance. Since no single ranking captures these distinct failure axes, we demonstrate that true OSR reliability requires system-level diagnostics to identify hidden vulnerabilities.

---


## Repository Structure

```
├── ln_dataset/           # LN dataset generation pipeline
│   ├── core/             # Core generation logic (AE, masks, configs)
│   └── nuisances/        # Perturbation implementations (noise, pixel, spatial, photometric)
├── bench/                # Benchmarking framework
│   ├── run_bench.py      # Main benchmark runner
│   ├── detectors.py      # 11 detector implementations
│   ├── backbones.py      # Backbone model wrappers
│   └── datasets.py       # Dataset registry and loaders
├── analysis/             # Analysis and paper output
│   ├── processing.py     # Metric aggregation engine
│   ├── plots.py          # Figure generators
│   └── tables.py         # LaTeX table generators
├── configs/              # YAML configurations
└── scripts/              # Utility scripts
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

### Quick Start

```bash
# Clone repository
git clone https://github.com/Rstayers/ND-25-Nuisance-Novelty.git
cd ND-25-Nuisance-Novelty

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate LN Dataset

```bash
python -m ln_dataset.core.generate_ln \
    --config ln_dataset/configs/imagenet.yaml \
    --data /path/to/imagenet/val \
    --imglist /path/to/val_imagenet.txt \
    --out_dir data/images_largescale/imagenet_ln
```

See [ln_dataset/README.md](ln_dataset/README.md) for full generation pipeline documentation.

### 2. Run Benchmarks

```bash
python -m bench.run_bench \
    --backbones resnet50 vit_b_16 swin_t densenet121 convnext_t \
    --detectors msp odin react ash dice knn mds vim she ebo postmax \
    --id_dataset ImageNet-Val \
    --test_datasets ImageNet-LN ImageNet-C CNS \
    --out_dir analysis/bench_results/imagenet
```

See [bench/README.md](bench/README.md) for benchmarking documentation.

### 3. Generate Analysis

```bash
python -m analysis.run_report \
    --benchmark_csvs analysis/bench_results/imagenet/bench_samples.csv \
    --benchmark_filter ImageNet-LN ImageNet-C CNS \
    --out paper_output
```

See [analysis/README.md](analysis/README.md) for analysis documentation.

---

## Applying to a New Dataset

The LN framework is dataset-agnostic. Follow these steps to apply it to your own dataset:

### Requirements

1. **Image dataset** with train/val splits
2. **Image list files** (text files with `path label` per line)
3. **Pretrained classifier backbones** for your dataset (or use ImageNet-pretrained)

### Step 1: Create Configuration File

Create a YAML config in `ln_dataset/configs/your_dataset.yaml`:

```yaml
dataset:
  name: your_dataset
  root: /path/to/your_dataset
  train_list: data/benchmark_imglist/your_dataset/train.txt
  val_list: data/benchmark_imglist/your_dataset/val.txt
  num_classes: 100  # Number of classes in your dataset

autoencoder:
  latent_dim: 256
  epochs: 50
  batch_size: 64
  learning_rate: 0.001

generation:
  target_area: 0.33      # Fraction of image to perturb
  n_sweeps: 50           # Parameter sweep granularity
  n_trials: 2            # Trials per stochastic nuisance

models:
  use_torchvision: true  # Use torchvision pretrained models
  backbones:
    - resnet50
    - vit_b_16
    - convnext_t
    - densenet121
```

### Step 2: Create Image List Files

Create text files listing images with labels:

```
# data/benchmark_imglist/your_dataset/train.txt
class_a/image001.jpg 0
class_a/image002.jpg 0
class_b/image001.jpg 1
...

# data/benchmark_imglist/your_dataset/val.txt
class_a/image101.jpg 0
class_b/image101.jpg 1
...
```

### Step 3: Train Required Components

```bash
# 1. Train autoencoder (learns image reconstruction)
python -m ln_dataset.core.train_ae --config ln_dataset/configs/your_dataset.yaml

# 2. Calibrate PaRCE (per-class reconstruction statistics)
python -m ln_dataset.core.calibrate_parce --config ln_dataset/configs/your_dataset.yaml

# 3. Calibrate bin edges (severity level thresholds)
python -m ln_dataset.core.calibrate_bins --config ln_dataset/configs/your_dataset.yaml
```

### Step 4: Generate LN Dataset

```bash
python -m ln_dataset.core.generate_ln \
    --config ln_dataset/configs/your_dataset.yaml \
    --data /path/to/your_dataset/val \
    --imglist data/benchmark_imglist/your_dataset/val.txt \
    --out_dir data/images_largescale/your_dataset_ln
```

### Step 5: Register Dataset for Benchmarking

Add your dataset to `bench/datasets.py`:

```python
DATASET_ZOO["YourDataset-LN"] = {
    "root": "data/images_largescale/your_dataset_ln",
    "imglist": "data/benchmark_imglist/your_dataset/your_dataset_ln.txt",
    "parser": parse_ln_manifest,
    "num_classes": 100,
    "is_imagenet": False
}
```

### Step 6: Run Benchmarks

```bash
python -m bench.run_bench \
    --config bench/configs/your_dataset.yaml \
    --test_datasets YourDataset-LN \
    --out_dir analysis/bench_results/your_dataset
```

### What Gets Trained vs. What's Pretrained

| Component | Training Required? | Notes |
|-----------|-------------------|-------|
| **Backbone classifiers** | No | Uses torchvision pretrained weights (IMAGENET1K_V1) |
| **Autoencoder** | Yes | Dataset-specific; learns reconstruction |
| **PaRCE calibration** | Yes | Per-class reconstruction statistics |
| **Bin edges** | Yes | Severity level thresholds |
| **Post-hoc detectors** | Some | KNN, MDS, VIM need ID features; others are zero-shot |

### Fine-Grained Classification Datasets

For fine-grained datasets (CUB-200, Stanford Cars, etc.), you may want to:

1. Use domain-specific pretrained backbones instead of ImageNet weights
2. Adjust `target_area` in config (smaller regions for fine-grained details)
3. Increase `n_sweeps` for finer-grained severity levels

---

## Metrics

| Metric | Description |
|--------|-------------|
| **CSA** | Closed-Set Accuracy: N_correct / N_known |
| **CCR** | Clean Correct Rate: correct AND accepted at threshold θ |
| **NNR** | Nuisance Novelty Rate: correct AND rejected at threshold θ |
| **CNR** | Conditional Novelty Rate: NNR / CSA |
| **OSA** | Open-Set Accuracy: (Clean_Success + rejected_unknowns) / N_total |
| **ADR** | Accuracy Degradation Rate: regression slope of OSA_Gap vs severity |

## Supported Detectors

MSP, ODIN, ReAct, ASH, DICE, KNN, MDS, VIM, SHE, EBO, PostMax

## Supported Backbones

ResNet-50, ViT-B/16, Swin-T, DenseNet-121, ConvNeXt-T

---

## References

- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. ([paper](https://arxiv.org/abs/1903.12261))
- Dünkel, O., et al. (2025). *CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts*. ([paper](https://arxiv.org/abs/2507.17651))
- Vaze, S., et al. (2022). *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* ICLR. ([paper](https://arxiv.org/abs/2110.06207))
