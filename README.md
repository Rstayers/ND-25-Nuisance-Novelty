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
cd nuisance-novelty

# Install dependencies
pip install -r requirements.txt

# Download model weights
bash scripts/download_weights.sh
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



- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. ([paper](https://arxiv.org/abs/1903.12261))
- Dünkel, O., et al. (2025). *CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts*. ([paper](https://arxiv.org/abs/2507.17651))
- Vaze, S., et al. (2022). *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* ICLR. ([paper](https://arxiv.org/abs/2110.06207))
