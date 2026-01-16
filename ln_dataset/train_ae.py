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

# Custom AE
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
                    # Handle paths with spaces or just take the first part
                    self.samples.append(parts[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
        except:
            return torch.zeros((3, 224, 224))

        if self.transform:
            img = self.transform(img)
        return img


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Root image folder")
    parser.add_argument('--imglist', type=str, required=True, help="Path to train list")
    parser.add_argument('--save_dir', type=str, default="checkpoints")

    # NEW ARGUMENTS FOR CUSTOM WEIGHTS
    parser.add_argument('--resnet_ckpt', type=str, default=None, help="Path to custom ResNet50 .pth")
    parser.add_argument('--vit_ckpt', type=str, default=None, help="Path to custom ViT .pth")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize Model with Custom Weights ---
    model = ClassifierAwareAE(
        resnet_path=args.resnet_ckpt,
        vit_path=args.vit_ckpt
    ).to(device)

    # Transform (Resize + ToTensor)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Note: Do NOT normalize here. The AE class handles normalization internally.
        # It expects input in range [0, 1].
    ])

    dataset = ImgListDataset(args.data, args.imglist, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Training AE on {len(dataset)} images...")

    # Train Loop
    epochs = 1
    global_step = 0

    for epoch in range(epochs):
        model.decoder.train()
        pbar = tqdm(dataloader)
        running_loss = 0.0

        for img in pbar:
            img = img.to(device)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, img)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'epoch': epoch + 1, 'loss': f"{loss.item():.5f}"})

            global_step += 1
            if global_step % 500 == 0:
                # Save debug image
                debug_path = os.path.join(args.save_dir, f"epoch_{epoch}_step_{global_step}.png")
                save_image(torch.cat([img[:8], output[:8]]), debug_path)

        # Save Checkpoint per epoch
        torch.save(model.state_dict(), os.path.join(args.save_dir, "ae_latest.pth"))

    final_path = os.path.join(args.save_dir, "ae_classifier_aware_weights.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to {final_path}")


if __name__ == "__main__":
    train()