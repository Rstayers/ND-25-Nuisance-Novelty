# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 16GB+ RAM recommended
- ~50GB disk space for datasets

## Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/nuisance-novelty.git
cd nuisance-novelty

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate Nuisance
```

## Option 2: Pip

```bash
# Clone the repository
git clone https://github.com/your-org/nuisance-novelty.git
cd nuisance-novelty

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

### Autoencoder Weights (for LN generation)

Download pretrained autoencoder weights:

```bash
bash scripts/download_weights.sh
```

Or train from scratch:

```bash
python -m ln_dataset.core.train_ae --config ln_dataset/configs/imagenet.yaml
```

### Backbone Weights

Backbone weights (ResNet-50, ViT-B/16, etc.) are automatically downloaded from torchvision on first use with `IMAGENET1K_V1` weights.

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

Download OpenImage-O for OOD threshold calibration:

```bash
# Download OpenImage-O subset
python -c "from bench.datasets import setup_openimage_o; setup_openimage_o()"
```

### ImageNet-C (optional, for comparison)

Download ImageNet-C corruptions:

```bash
# From https://zenodo.org/record/2235448
wget https://zenodo.org/record/2235448/files/blur.tar
wget https://zenodo.org/record/2235448/files/digital.tar
# ... etc
```

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

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### OpenOOD Import Errors

If OpenOOD fails to import:

```bash
# Install without dependencies to avoid conflicts
pip install git+https://github.com/Jingkang50/OpenOOD --no-deps
```

### Memory Issues

For large datasets, reduce batch size:

```bash
python -m bench.run_bench --batch_size 16
```
