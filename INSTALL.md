# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 16GB+ RAM recommended
- ~50GB disk space for datasets

## Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/Rstayers/ND-25-Nuisance-Novelty.git
cd ND-25-Nuisance-Novelty

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate Nuisance
```

## Option 2: Pip

```bash
# Clone the repository
git clone https://github.com/Rstayers/ND-25-Nuisance-Novelty.git
cd ND-25-Nuisance-Novelty

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install OpenOOD (required for detectors)
pip install git+https://github.com/Jingkang50/OpenOOD --no-deps
```

## Model Weights

### Backbone Weights

Backbone weights (ResNet-50, ViT-B/16, Swin-T, DenseNet-121, ConvNeXt-T) are **automatically downloaded** from torchvision on first use with `IMAGENET1K_V1` weights. No manual download is required.

### Autoencoder Weights (for LN generation)

The autoencoder must be trained for your specific dataset. See [Training Your Own Weights](#training-your-own-weights) below.

---

## Training Your Own Weights

The LN generation pipeline requires dataset-specific trained components. Follow these steps in order:

### Step 1: Train the Autoencoder

The autoencoder learns to reconstruct images and is used to identify competency-critical regions.

```bash
python -m ln_dataset.core.train_ae \
    --config ln_dataset/configs/imagenet.yaml \
    --epochs 50 \
    --batch_size 64
```

**Output:** `checkpoints/ae_<dataset>.pth`

**Training time:** ~4-8 hours on a single GPU for ImageNet-scale datasets.

### Step 2: Calibrate PaRCE Statistics

Compute per-class reconstruction error statistics for the PaRCE competency score.

```bash
python -m ln_dataset.core.calibrate_parce \
    --config ln_dataset/configs/imagenet.yaml \
    --ae_weights checkpoints/ae_imagenet.pth
```

**Output:** `checkpoints/parce_calib_<dataset>.pt`

### Step 3: Calibrate Severity Bin Edges

Determine thresholds for mapping PaRCE scores to severity levels 1-5.

```bash
python -m ln_dataset.core.calibrate_bins \
    --config ln_dataset/configs/imagenet.yaml \
    --ae_weights checkpoints/ae_imagenet.pth \
    --parce_calib checkpoints/parce_calib_imagenet.pt
```

**Output:** `checkpoints/bin_edges_<dataset>.json`

### Step 4: Generate LN Dataset

With all calibration complete, generate the LN dataset:

```bash
python -m ln_dataset.core.generate_ln \
    --config ln_dataset/configs/imagenet.yaml \
    --data /path/to/dataset/val \
    --imglist data/benchmark_imglist/imagenet/val_imagenet.txt \
    --ae_weights checkpoints/ae_imagenet.pth \
    --parce_calib checkpoints/parce_calib_imagenet.pt \
    --bin_edges_json checkpoints/bin_edges_imagenet.json \
    --out_dir data/images_largescale/imagenet_ln
```

---

## Dataset Setup

### ImageNet

1. Download ImageNet ILSVRC2012 from [image-net.org](https://image-net.org/)
2. Extract to `data/images_largescale/imagenet/`
3. Structure:
   ```
   data/images_largescale/imagenet/
   ├── train/
   │   ├── n01440764/
   │   └── ...
   └── val/
       ├── n01440764/
       └── ...
   ```

### OpenImage-O (OOD calibration)

Required for threshold calibration in benchmarking. Download from the OpenOOD repository or use the provided image lists.

### ImageNet-C (optional, for comparison)

Download ImageNet-C corruptions from [Zenodo](https://zenodo.org/record/2235448).

---

## Verification

Verify installation:

```bash
# Check imports
python -c "from bench.run_bench import main; print('Bench OK')"
python -c "from analysis.run_report import main; print('Analysis OK')"
python -c "from ln_dataset.core.generate_ln import main; print('Generation OK')"

# Run a quick smoke test
python -m bench.run_bench --help
```

---

## Troubleshooting

### CUDA Issues

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### OpenOOD Import Errors

```bash
# Install without dependencies to avoid conflicts
pip install git+https://github.com/Jingkang50/OpenOOD --no-deps
```

### Memory Issues

For large datasets, reduce batch size:

```bash
python -m bench.run_bench --batch_size 16
python -m ln_dataset.core.train_ae --batch_size 32
```
