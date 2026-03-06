# Local Nuisance (LN) Dataset Generation

This module generates Local Nuisance datasets by applying masked perturbations to competency-targeted image regions while preserving semantic content.

## Pipeline Overview

```
1. Train AE       →  Learn image reconstruction
2. Calibrate PaRCE →  Compute per-class reconstruction statistics
3. Calibrate Bins  →  Define severity level thresholds
4. Generate LN     →  Create perturbed images with severity labels
```

## Quick Start

```bash
# Full pipeline (ImageNet example)
python -m ln_dataset.core.run_pipeline --config ln_dataset/configs/imagenet.yaml
```

Or run each step individually:

```bash
# 1. Train autoencoder
python -m ln_dataset.core.train_ae --config ln_dataset/configs/imagenet.yaml

# 2. Calibrate PaRCE (per-class reconstruction statistics)
python -m ln_dataset.core.calibrate_parce --config ln_dataset/configs/imagenet.yaml

# 3. Calibrate bin edges (severity levels 1-5)
python -m ln_dataset.core.calibrate_bins --config ln_dataset/configs/imagenet.yaml

# 4. Generate LN dataset
python -m ln_dataset.core.generate_ln \
    --config ln_dataset/configs/imagenet.yaml \
    --data /path/to/imagenet/val \
    --imglist /path/to/val_imagenet.txt \
    --out_dir data/images_largescale/imagenet_ln
```

## Perturbation Types

| Nuisance | Description | Parameters |
|----------|-------------|------------|
| **Noise** | Gaussian noise injection | σ ∈ [0.05, 0.5] |
| **Pixel** | Local pixelation | block_size ∈ [4, 32] |
| **Spatial** | Elastic transform | α ∈ [50, 500] |
| **Photometric** | Brightness/contrast/saturation | factor ∈ [0.5, 2.0] |

## Key Concepts

### Competency Targeting
Perturbations are applied only to regions where the autoencoder has high reconstruction error—areas that are semantically important for classification.

### PaRCE Score
The Probabilistic Anomaly Reconstruction Competency Estimate (PaRCE) measures how "in-distribution" an image appears:

```
ρ = p_hat × Σ_c p_c × (1 - Φ((l - 2μ_c)/σ_c - z))
```

Where:
- `p_hat`: max ensemble confidence
- `l`: reconstruction MSE
- `μ_c, σ_c`: per-class reconstruction statistics
- `z`: calibrated z-score

### Severity Levels
Images are binned into 5 severity levels based on PaRCE score:
- **Level 1**: Minimal perturbation (highest PaRCE)
- **Level 5**: Maximum perturbation (lowest PaRCE, still correctly classified)

## Configuration

Example config (`ln_dataset/configs/imagenet.yaml`):

```yaml
dataset:
  name: imagenet
  root: /path/to/imagenet
  train_list: data/benchmark_imglist/imagenet/train_imagenet.txt
  val_list: data/benchmark_imglist/imagenet/val_imagenet.txt
  num_classes: 1000

autoencoder:
  latent_dim: 256
  epochs: 50
  batch_size: 64

generation:
  target_area: 0.33  # Mask 33% of image
  n_sweeps: 50       # Parameter sweep steps
  n_trials: 2        # Trials per stochastic nuisance

models:
  use_torchvision: true
  backbones: [resnet50, vit_b_16, convnext_t, densenet121]
```

## Output Format

Generated images are saved with structured filenames:

```
output_dir/
├── noise/
│   ├── level_1/
│   │   └── n01440764_00000001.JPEG
│   └── level_5/
├── pixel/
├── spatial/
└── photometric/
```

A manifest file (`manifest.txt`) is generated with metadata:

```
path label level parce nuisance
noise/level_1/n01440764_00000001.JPEG 0 1 0.85 noise
```

## Module Structure

```
ln_dataset/
├── core/
│   ├── autoencoder.py    # StandardAE model
│   ├── masks.py          # Felzenszwalb segmentation + error ranking
│   ├── generate_ln.py    # Main generation logic
│   ├── calibrate_parce.py
│   ├── calibrate_bins.py
│   ├── train_ae.py
│   └── configs.py
└── nuisances/
    ├── noise.py
    ├── pixel.py
    ├── spatial.py
    └── photometric.py
```

## Generation Statistics

After generation, statistics are saved to `generation_stats_<dataset>.json`:

```json
{
  "total_images": 50000,
  "images_processed": 48500,
  "yield_rate": 0.97,
  "by_nuisance": {
    "noise": {"level_1": 9500, "level_2": 9200, ...},
    ...
  }
}
```
