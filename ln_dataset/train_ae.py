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

# NEW: Import the Classifier-Aware Model
from ln_dataset.core.autoencoder import ClassifierAwareAE


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
                    self.samples.append(parts[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Corrupt image {full_path}. Returning black.")
            return torch.zeros((3, 224, 224))

        if self.transform:
            img = self.transform(img)
        return img


def train_classifier_aware(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Classifier-Aware Trainer on {device}...")

    os.makedirs(args.save_dir, exist_ok=True)
    debug_dir = os.path.join(args.save_dir, "debug_recon_features")
    os.makedirs(debug_dir, exist_ok=True)

    # Data Prep: Input is [0, 1]. Normalization happens inside the model forward().
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 1]
    ])

    dataset = ImgListDataset(args.data, args.imglist, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Dataset: {len(dataset)} images.")

    # Initialize New Model
    print("Loading Backbones (ResNet50 + ViT-B/16)... this may trigger a download.")
    model = ClassifierAwareAE().to(device)

    # Optimizer: Only train the Decoder!
    # (The backbones are frozen in the class __init__)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    criterion = nn.MSELoss()

    print("Starting Training (1 Full Epoch)...")
    model.train()  # This puts Decoder in train mode (Backbones stay frozen)

    start_time = time.time()
    running_loss = 0.0
    save_interval = 500

    pbar = tqdm(loader, desc="Training Feature Decoder")

    for step, img in enumerate(pbar):
        img = img.to(device)

        optimizer.zero_grad()

        # Forward (Norm -> Feat Extract -> Decode)
        output = model(img)

        # Loss (Reconstruction vs Original Image)
        loss = criterion(output, img)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        if step % save_interval == 0:
            comparison = torch.cat([img[:8], output[:8]])
            save_image(comparison, os.path.join(debug_dir, f"step_{step}.png"))
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ae_latest.pth"))

    final_path = os.path.join(args.save_dir, "ae_classifier_aware_weights.pth")
    torch.save(model.state_dict(), final_path)

    elapsed = time.time() - start_time
    print(f"\nTraining Complete. Saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to ImageNet TRAIN set")
    parser.add_argument('--imglist', type=str, required=True, help="Path to train.txt list")
    parser.add_argument('--save_dir', type=str, default="./ln_dataset/assets")

    # Reduced batch size due to heavy backbones
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    train_classifier_aware(args)