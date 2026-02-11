import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Assumes you have updated autoencoder.py to include StandardAE as discussed
# If not, ensure StandardAE is defined in ln_dataset.core.autoencoder
from ln_dataset.core.autoencoder import StandardAE
from ln_dataset.core.configs import load_config


# =====================================================
# 1. HELPER: RANDOM BOX MASKING (Paper Faithful)
# =====================================================
def apply_random_box_mask(img_tensor, min_cut=16, max_cut=64, num_holes=1):
    """
    Randomly cuts out rectangular holes in the image batch.

    This forces the AE to learn 'Contextual Inpainting' (hallucinating missing parts),
    which is the core mechanic required for the 'Reconstruction Loss' method
    (Approach 5) where segments are removed and predicted.

    Args:
        img_tensor (Tensor): [B, C, H, W]
    Returns:
        masked_img (Tensor): Image with blacked out regions.
    """
    B, C, H, W = img_tensor.shape

    # Create a mask initialized to 1 (keep)
    mask = torch.ones((B, 1, H, W), device=img_tensor.device)

    for i in range(B):
        for _ in range(num_holes):
            # Random hole size
            h_cut = np.random.randint(min_cut, max_cut)
            w_cut = np.random.randint(min_cut, max_cut)

            # Random location
            # Ensure we don't go out of bounds
            y = np.random.randint(0, max(1, H - h_cut))
            x = np.random.randint(0, max(1, W - w_cut))

            # Cut out (set mask to 0)
            mask[i, :, y:y + h_cut, x:x + w_cut] = 0.0

    # Apply mask: 0 where hole is, 1 where image is
    # We fill the hole with 0.0 (black) or mean (0.5).
    # Black (0.0) is standard for inpainting inputs.
    masked_img = img_tensor * mask

    return masked_img


# =====================================================
# 2. DATASET
# =====================================================
class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, root, imglist_path, transform=None, limit=None):
        self.root = root
        self.transform = transform
        self.samples = []
        if not os.path.exists(imglist_path):
            raise FileNotFoundError(f"List not found: {imglist_path}")

        with open(imglist_path, 'r') as f:
            for line in f:
                if limit is not None and len(self.samples) >= limit:
                    break
                parts = line.strip().split()
                if len(parts) >= 1:
                    self.samples.append(parts[0])

        if limit:
            print(f"Dataset limited to {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return torch.zeros(3, 224, 224)


# =====================================================
# 3. TRAINING LOOP
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, help="Path to training images root")
    parser.add_argument('--imglist', type=str, required=True, help="Path to training image list")
    parser.add_argument('--save_path', type=str, default="checkpoints/standard_ae_inpainting.pth")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config)
    img_size = cfg.image_size[0]  # e.g., 224

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Note: StandardAE usually expects [0,1]. If you use Normalize, ensure logic matches.
    ])

    # Dataset
    dataset = ImgListDataset(args.data, args.imglist, transform=transform, limit=None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model: StandardAE (Not ClassifierAware)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing StandardAE (Inpainting Mode)...")
    model = StandardAE(in_channels=3).to(device)

    # Optimizer: Train BOTH Encoder and Decoder
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Loss: MSE on pixels
    criterion = nn.MSELoss()

    scaler = GradScaler()

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        total_loss = 0

        for i, img in enumerate(pbar):
            img = img.to(device, non_blocking=True)

            # --- 1. Apply Random Mask (Paper Faithful) ---
            # We corrupt the input, but calculate loss against the ORIGINAL
            masked_input = apply_random_box_mask(img, min_cut=20, max_cut=80)

            optimizer.zero_grad()

            with autocast():
                # --- 2. Forward (Reconstruct) ---
                recon = model(masked_input)

                # --- 3. Loss (Reconstruction Error) ---
                # Paper: "The difference... is the reconstruction loss"
                # We train it to minimize this difference.
                loss = criterion(recon, img)

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

            # Debug: Save first batch of every epoch
            if i == 0:
                os.makedirs("debug_train", exist_ok=True)
                # Show: Original | Masked Input | Reconstruction
                debug_grid = torch.cat([img[:8], masked_input[:8], recon[:8]], dim=0)
                save_image(debug_grid, f"debug_train/epoch_{epoch + 1}.png", nrow=8)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()