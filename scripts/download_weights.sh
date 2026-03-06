#!/bin/bash
# Download pretrained model weights for Nuisance Novelty

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$WEIGHTS_DIR"

echo "Downloading pretrained autoencoder weights..."

# Autoencoder weights for ImageNet
# Replace with actual download URL when available
# gdown --id <GOOGLE_DRIVE_FILE_ID> -O "$WEIGHTS_DIR/ae_imagenet.pth"

echo ""
echo "=== Weight Download Instructions ==="
echo ""
echo "Pretrained autoencoder weights are not yet publicly available."
echo "You can train the autoencoder from scratch:"
echo ""
echo "  python -m ln_dataset.core.train_ae --config ln_dataset/configs/imagenet.yaml"
echo ""
echo "Or contact the authors for pretrained weights."
echo ""
echo "Backbone weights (ResNet-50, ViT-B/16, etc.) are automatically"
echo "downloaded from torchvision on first use."
echo ""
