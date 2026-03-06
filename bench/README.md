# Nuisance Novelty Benchmark

Benchmarking pipeline for evaluating OOD detectors on nuisance-corrupted images.

## Quick Start

```bash
# 1. Verify setup (no data needed)
python -m bench.smoke_test --skip-backbones

# 2. Full smoke test (needs config + model weights)
python -m bench.smoke_test --config bench/configs/imagenet.yaml

# 3. Run full benchmark (config is required)
python -m bench.run_bench --config bench/configs/imagenet.yaml
```

---

## Prerequisites

### Python Environment

```bash
conda activate Nuisance  # or your environment name

# Required packages
pip install torch torchvision tqdm pandas numpy pyyaml pillow

# OpenOOD (required for postprocessors)
pip install git+https://github.com/Jingkang50/OpenOOD.git
```

### Verify OpenOOD Installation

```python
python -c "import openood; print('OpenOOD OK')"
```

---

## Directory Structure

```
project_root/
├── bench/                          # This folder
│   ├── run_bench.py               # Main benchmark script
│   ├── smoke_test.py              # Quick verification
│   ├── backbones.py               # Model loading
│   ├── detectors.py               # OOD detector factory
│   ├── datasets.py                # Dataset registry
│   ├── loader.py                  # Data loading
│   ├── metrics.py                 # Metric computation
│   ├── OSA.py                     # OSA threshold optimization
│   ├── PostMax.py                 # Custom PostMax detector
│   │
│   ├── configs/                   # Model configuration files
│   │   ├── imagenet.yaml          # ImageNet config (torchvision weights)
│   │   ├── stanford_cars.yaml     # Cars config (fine-tuned weights)
│   │   └── cub.yaml               # CUB-200 config (fine-tuned weights)
│   │
│   └── assets/                    # Model checkpoints (CUB/Cars only)
│       ├── cars_models/           # Fine-tuned Cars checkpoints
│       └── cub_models/            # Fine-tuned CUB checkpoints
│
├── data/
│   ├── images_largescale/         # Image root directory
│   │   ├── imagenet_ln_final_v2/  # Generated LN images
│   │   ├── imagenet_c/            # ImageNet-C corruption images
│   │   ├── cns/                   # CNS benchmark images
│   │   └── ...
│   │
│   └── benchmark_imglist/         # Image list files
│       ├── imagenet/
│       │   ├── val_imagenet.txt
│       │   ├── train_imagenet.txt
│       │   ├── imagenet_ln.txt
│       │   ├── test_imagenet_c.txt
│       │   └── test_openimage_o.txt
│       └── ...
│
└── analysis/
    └── bench_results/             # Output directory
        └── imagenet/
            ├── bench_summary.csv  # Aggregate metrics
            └── bench_samples.csv  # Per-sample results
```

---

## Dataset Setup

### Dataset Sources

**LN Datasets (Google Drive):**
The nuisance-corrupted LN datasets (`ImageNet-LN`, `CUB-LN`, `Cars-LN`) are available on Google Drive.
These must be downloaded manually and placed in `data/images_largescale/`:
- `imagenet_ln/` - ImageNet nuisance corruptions
- `cub_ln/` - CUB-200 nuisance corruptions
- `stanford_cars_ln/` - Stanford Cars nuisance corruptions

**Standard Datasets (OpenOOD):**
Other datasets (ImageNet-Val, ImageNet-C, OpenImage-O, etc.) will attempt to auto-download via OpenOOD.
If auto-download fails, you can download them manually from the [OpenOOD repository](https://github.com/Jingkang50/OpenOOD).

### Required Datasets

| Dataset | Purpose | Config Key | Image List |
|---------|---------|------------|------------|
| ImageNet-Val | ID validation/calibration | `val_imagenet.txt` | 50,000 images |
| ImageNet-Train | Training features (some detectors) | `train_imagenet.txt` | ~1.2M images |
| ImageNet-LN | **Test** (nuisance corruptions) | `imagenet_ln.txt` | Google Drive |
| ImageNet-C | **Test** (corruption benchmark) | `test_imagenet_c.txt` | OpenOOD |
| CNS | **Test** (style corruptions) | `cns_bench_all.txt` | OpenOOD |
| OpenImage-O-Surrogate | OOD calibration | `test_openimage_o.txt` | OpenOOD |
| OpenImage-O-Test | OOD test | `test_openimage_o.txt` | OpenOOD |

### Image List Format

Each `.txt` file has format: `relative_path label`

**Standard format:**
```
imagenet_val/n01440764/ILSVRC2012_val_00000293.JPEG 0
imagenet_val/n01440764/ILSVRC2012_val_00000543.JPEG 0
```

**LN manifest format (5 columns):**
```
blur/1/n01440764_1234.JPEG 0 1 0.85 gaussian_blur
motion_blur/3/n01440764_5678.JPEG 0 3 0.72 motion_blur
```
Format: `path label level parce nuisance`

### Configuring Paths

Edit `bench/datasets.py` to match your directory layout:

```python
# Line 171-172
DATA_ROOT_DEFAULT = "data/images_largescale"   # Where images live
LIST_ROOT_DEFAULT = "data/benchmark_imglist"   # Where .txt lists live
```

### Dataset Registry

All datasets are defined in `DATASET_ZOO` (bench/datasets.py). Each entry:

```python
"ImageNet-LN": {
    "root": f"{DATA_ROOT_DEFAULT}/imagenet_ln_final_v2",
    "imglist": f"{LIST_ROOT_DEFAULT}/imagenet/imagenet_ln.txt",
    "parser": parse_ln_manifest,
    "num_classes": 1000,
    "is_imagenet": True
},
```

---

## Backbone Configuration

### Supported Backbones (5)

| Backbone | Architecture | Feature Dim | Config Key |
|----------|--------------|-------------|------------|
| `resnet50` | ResNet-50 | 2048 | `resnet_ckpt` |
| `vit_b_16` | ViT-Base/16 | 768 | `vit_ckpt` |
| `swin_t` | Swin-Tiny | 768 | `swin_ckpt` |
| `densenet121` | DenseNet-121 | 1024 | `densenet_ckpt` |
| `convnext_t` | ConvNeXt-Tiny | 768 | `convnext_ckpt` |

### Weight Loading

Weights are controlled by config files in `bench/configs/`:

**ImageNet** (`bench/configs/imagenet.yaml`):
```yaml
models:
  use_torchvision: true  # Uses ImageNet-1K pretrained weights from torchvision
```

**CUB/Cars** (`bench/configs/cub.yaml`, `bench/configs/stanford_cars.yaml`):
```yaml
models:
  use_torchvision: false  # Requires fine-tuned checkpoints
  resnet_ckpt: "bench/assets/cub_models/resnet50_cub.pth"
  vit_ckpt: "bench/assets/cub_models/vit_b_16_cub.pth"
  # ... etc
```

### Setting Up Fine-Tuned Models (CUB/Cars)

For CUB-200 or Stanford Cars benchmarks:

1. Create the assets directory:
   ```bash
   mkdir -p bench/assets/cub_models
   mkdir -p bench/assets/cars_models
   ```

2. Copy or symlink your fine-tuned checkpoints:
   ```bash
   cp /path/to/your/resnet50_cub.pth bench/assets/cub_models/
   # ... repeat for other backbones
   ```

3. Update paths in config if needed:
   ```yaml
   # bench/configs/cub.yaml
   models:
     resnet_ckpt: "bench/assets/cub_models/resnet50_cub.pth"
   ```

---

## OOD Detectors (Postprocessors)

### Default Detectors (9)

| Detector | Requires Training Data | Score Type | Notes |
|----------|------------------------|------------|-------|
| `msp` | No | [0,1] softmax | Baseline |
| `odin` | No | Raw scores | Requires gradients |
| `react` | Yes | Raw scores | Feature rectification |
| `dice` | Yes | Raw scores | Directed sparsification |
| `knn` | Yes | Distances | k-NN in feature space |
| `mds` | Yes | Mahalanobis | Gaussian class modeling |
| `vim` | Yes | Raw scores | Virtual logit matching |
| `she` | Yes | Raw scores | Simplified Hopfield energy |
| `postmax` | Yes | Raw scores | GPD-based (custom) |

### Training Data Detectors

These detectors need `ImageNet-Train` features extracted before running:
- `react`, `dice`, `knn`, `mds`, `vim`, `she`, `postmax`

The benchmark handles this automatically via `detector.setup()`.

### All Available Detectors (12)

Additional detectors not in default list:
- `maxlogit` - Max logit score
- `ebo` - Energy-based OOD
- `ash` - Activation shaping

---

## Running the Benchmark

### Full Benchmark

All settings are controlled by the config file. Simply specify which config to use:

```bash
# ImageNet benchmark (uses torchvision pretrained weights)
python -m bench.run_bench --config bench/configs/imagenet.yaml

# CUB-200 benchmark (requires fine-tuned checkpoints)
python -m bench.run_bench --config bench/configs/cub.yaml

# Stanford Cars benchmark (requires fine-tuned checkpoints)
python -m bench.run_bench --config bench/configs/stanford_cars.yaml
```

Runs all backbones x detectors x test datasets as specified in the config.

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | **Yes** | Path to config YAML |
| `--use_amp` | No | Enable mixed precision (default: False) |
| `--force_cpu` | No | Force CPU mode (default: False) |
| `--skip_samples` | No | Skip per-sample CSV output (default: False) |

### Config File Structure

All benchmark parameters are in the config file (`bench/configs/*.yaml`):

```yaml
benchmark:
  backbones: [resnet50, vit_b_16, swin_t, densenet121, convnext_t]
  detectors: [msp, odin, react, dice, knn, mds, vim, she, postmax]
  id_dataset: "ImageNet-Val"
  train_dataset: "ImageNet-Train"
  test_datasets: [ImageNet-LN, CNS, ImageNet-C]
  calib_id_dataset: "ImageNet-Val"
  calib_ood_dataset: "OpenImage-O-Surrogate"
  test_ood_dataset: "OpenImage-O-Test"
  out_dir: "analysis/bench_results/imagenet"
  batch_size: 32
```

### Quick Test Run

Edit your config to test a subset, or create a minimal test config:

```yaml
# bench/configs/test.yaml
benchmark:
  backbones: [resnet50]
  detectors: [msp]
  test_datasets: [ImageNet-LN]
  # ... other required fields
```

```bash
python -m bench.run_bench --config bench/configs/test.yaml --skip_samples
```

---

## Output Format

### bench_summary.csv

One row per (backbone, detector, test_dataset) combination:

| Column | Description |
|--------|-------------|
| `backbone` | Model architecture |
| `detector` | OOD detector name |
| `test_dataset` | Test dataset name |
| `oosa_threshold` | Calibrated threshold |
| `OSA` | Open-Set Accuracy |
| `CSA` | Closed-Set Accuracy |
| `CCR@theta` | Correct Classification Rate @ threshold |
| `URR@theta` | Unknown Rejection Rate @ threshold |
| `NNR` | Nuisance Novelty Rate |
| `n_known` | Number of known samples |
| `n_unknown` | Number of OOD samples |
| `rejected_unknowns` | Count of rejected OOD |

### bench_samples.csv

Per-sample results (use `--skip_samples` to disable):

| Column | Description |
|--------|-------------|
| `path` | Image path |
| `label` | Ground truth label |
| `prediction` | Model prediction |
| `confidence` | Detector confidence score |
| `correct_cls` | 1 if prediction correct |
| `outcome` | Clean_Success, Nuisance_Novelty, Double_Failure, Contained_Misidentification |
| `level` | Corruption severity (0-5) |
| `nuisance` | Corruption type |
| `backbone`, `detector` | Method identifiers |

---

## Metric Definitions

```
CSA  = N_correct / N_known                    # Closed-Set Accuracy
CCR  = (Correct & Accepted) / N_known         # Correct Classification Rate
NNR  = (Correct & Rejected) / N_known         # Nuisance Novelty Rate
URR  = rejected_unknowns / N_unknown          # Unknown Rejection Rate
OSA  = (Correct&Accepted + rejected_unknowns) / (N_known + N_unknown)
```

**Outcome Categories:**
- **Clean_Success**: Correct prediction AND accepted (ideal)
- **Nuisance_Novelty**: Correct prediction BUT rejected (THE PROBLEM we measure)
- **Double_Failure**: Wrong prediction AND rejected
- **Contained_Misidentification**: Wrong prediction AND accepted

---

## Troubleshooting

### Common Errors

**1. `ModuleNotFoundError: No module named 'openood'`**
```bash
pip install git+https://github.com/Jingkang50/OpenOOD.git
```

**2. `FileNotFoundError: Checkpoint not found`**

For ImageNet, ensure config has `use_torchvision: true`:
```yaml
# bench/configs/imagenet.yaml
models:
  use_torchvision: true
```

For CUB/Cars, ensure checkpoint files exist at the paths specified in the config.

**3. `Dataset 'X' not found in DATASET_ZOO`**

Add the dataset to `bench/datasets.py` in `DATASET_ZOO` dict.

**4. `Image not found` warnings**

Check that paths in your image list match the actual directory structure.
The loader tries multiple path strategies:
- `root + relative_path`
- `absolute_path`
- `root + filename_only`

**5. CUDA out of memory**

Reduce batch size:
```bash
python -m bench.run_bench --batch_size 16
```

Or force CPU:
```bash
python -m bench.run_bench --force_cpu
```

**6. ODIN gradient errors**

ODIN requires gradients for input perturbation. This is handled automatically
in `run_bench.py`. If you see gradient-related errors, ensure you're using
the standard benchmark pipeline.

### Verification Steps

1. **Check OpenOOD:**
   ```bash
   python -c "from openood.postprocessors import BasePostprocessor; print('OK')"
   ```

2. **Check detectors:**
   ```bash
   python -m bench.smoke_test --skip-backbones
   ```

3. **Check backbones:**
   ```bash
   python -m bench.smoke_test --config bench/configs/imagenet.yaml
   ```

4. **Check single dataset:**
   ```python
   from bench.loader import get_loader
   loader = get_loader("ImageNet-Val", batch_size=4)
   batch = next(iter(loader))
   print(batch.keys())  # Should show: data, label, path, level, parce, nuisance, dataset_name
   ```

---

## Files Reference

| File | Purpose |
|------|---------|
| `run_bench.py` | Main benchmark loop |
| `smoke_test.py` | Quick verification tests |
| `backbones.py` | Model loading with OpenOOD adapter |
| `detectors.py` | OOD detector factory (wraps OpenOOD) |
| `datasets.py` | Dataset registry and parsers |
| `loader.py` | PyTorch DataLoader creation |
| `metrics.py` | Threshold computation, outcome classification |
| `OSA.py` | OSA metric and threshold optimization |
| `PostMax.py` | Custom GPD-based detector |
| `configs/*.yaml` | Model configuration files |

---

## Expected Runtime

Approximate times on RTX 3090 with batch_size=32:

| Configuration | Time |
|---------------|------|
| 1 backbone x 1 detector x 1 dataset | ~5 min |
| 1 backbone x 9 detectors x 3 datasets | ~2 hours |
| 5 backbones x 9 detectors x 3 datasets | ~10 hours |

Training-data detectors (knn, mds, vim, etc.) add ~10-15 min per backbone
for feature extraction.

---

## Contact

For issues with the benchmark pipeline, check:
1. This README
2. The smoke_test output
3. Dataset paths in `datasets.py`
