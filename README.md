# Nuisance Novelty

Official implementation for the ECCV 2025 paper: **"Nuisance Novelty: A Stress-Test for Open-Set Recognition Under Local Perturbations"**

## Overview

**Nuisance Novelty** is a failure mode in Open-Set Recognition (OSR) where an in-distribution image is correctly classified by a closed-set classifier but incorrectly flagged as novel/OOD by a post-hoc detector.

Current nuisance benchmarks (ImageNet-C, CNS-Bench) primarily test **closed-set robustness**—the classifier's ability to maintain accuracy under perturbations. We propose a stress-test that evaluates **open-set robustness**—the detector's ability to maintain correct accept/reject decisions when semantic content is preserved.

### Key Contributions

1. **Local Nuisance (LN) Dataset Generation**: A pipeline that creates perturbations targeting classifier-competency regions while preserving semantic information
2. **Benchmarking Framework**: Evaluation of 11 post-hoc detectors across 5 backbones on multiple datasets
3. **New Metrics**: NNR (Nuisance Novelty Rate), CNR (Conditional Novelty Rate), and ADR (Accuracy Degradation Rate)

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
git clone https://github.com/your-org/nuisance-novelty.git
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

## Citation

```bibtex
@inproceedings{nuisancenovelty2025,
    title={Nuisance Novelty: A Stress-Test for Open-Set Recognition Under Local Perturbations},
    author={[Authors]},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. ([paper](https://arxiv.org/abs/1903.12261))
- Dünkel, O., et al. (2025). *CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts*. ([paper](https://arxiv.org/abs/2507.17651))
- Vaze, S., et al. (2022). *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* ICLR. ([paper](https://arxiv.org/abs/2110.06207))
