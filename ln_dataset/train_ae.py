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
import time

# Import your model definition
from ln_dataset.core.autoencoder import SimpleConvAE


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        if not os.path.exists(imglist_path):
            raise FileNotFoundError(f"List not found: {imglist_path}")

        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    # We only need the image path, ignore labels
                    self.samples.append(parts[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            # Convert to RGB to handle greyscale images automatically
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Corrupt image {full_path}. Returning black.")
            return torch.zeros((3, 224, 224))

        if self.transform:
            img = self.transform(img)
        return img


def train_comprehensive(args):
    # 1. Setup Device & Logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Comprehensive Trainer on {device}...")

    os.makedirs(args.save_dir, exist_ok=True)
    debug_dir = os.path.join(args.save_dir, "debug_recon")
    os.makedirs(debug_dir, exist_ok=True)

    # 2. Data Preparation
    # Standard ImageNet normalization is NOT used for AE target usually,
    # but since our Classifier expects normalized, we train AE on [0,1]
    # and let the AE learn the 0-1 distribution.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 1]
    ])

    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    # Shuffle is CRITICAL for learning general features
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Dataset Loaded: {len(dataset)} images.")

    # 3. Model & Optimization
    model = SimpleConvAE().to(device)

    # Optimizer: Adam is standard for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss: MSE is the standard for 'Reconstruction Error' metric
    criterion = nn.MSELoss()

    # 4. Training Loop (Full Epoch)
    print("Starting Training (1 Full Epoch)...")
    model.train()

    start_time = time.time()
    running_loss = 0.0
    log_interval = 100
    save_interval = 1000

    # Wrap loader in tqdm for progress bar
    pbar = tqdm(loader, desc="Training AE")

    for step, img in enumerate(pbar):
        img = img.to(device)

        # Forward
        optimizer.zero_grad()
        output = model(img)

        # Loss
        loss = criterion(output, img)

        # Backward
        loss.backward()
        optimizer.step()

        # Logging
        current_loss = loss.item()
        running_loss += current_loss

        pbar.set_postfix({'loss': f"{current_loss:.6f}"})

        # Periodic Debug Visualization
        if step % save_interval == 0:
            # Concatenate Input and Recon side-by-side
            comparison = torch.cat([img[:8], output[:8]])
            save_image(comparison, os.path.join(debug_dir, f"recon_step_{step}.png"))

            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ae_latest.pth"))

    # 5. Final Save
    final_path = os.path.join(args.save_dir, "ae_competency_weights.pth")
    torch.save(model.state_dict(), final_path)

    elapsed = time.time() - start_time
    print(f"\nTraining Complete.")
    print(f"Time Elapsed: {elapsed / 60:.2f} mins")
    print(f"Final Weights Saved to: {final_path}")
    print(f"Debug images saved to: {debug_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Critical Paths
    parser.add_argument('--data', type=str, required=True, help="Path to ImageNet TRAIN set")
    parser.add_argument('--imglist', type=str, required=True, help="Path to train.txt list")
    parser.add_argument('--save_dir', type=str, default="./ln_dataset/assets", help="Where to save weights")

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    train_comprehensive(args)