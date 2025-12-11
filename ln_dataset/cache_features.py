import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import argparse
import os
from tqdm import tqdm
from PIL import Image

# Use the dataset from your training script
# Ensure this import matches your file structure!
from ln_dataset.generate_ln import ImgListDataset


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load Backbones
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.resnet.eval()
        self.vit.eval()

        # Hooks
        self.r_feat = None
        self.v_feat = None

        # Register Hooks (ResNet Layer 3)
        self.resnet.layer3.register_forward_hook(
            lambda m, i, o: setattr(self, 'r_feat', o)
        )
        # Register Hook (ViT Last Block)
        self.vit.encoder.layers[-1].register_forward_hook(
            lambda m, i, o: setattr(self, 'v_feat', o)
        )

    def forward(self, x):
        with torch.no_grad():
            _ = self.resnet(x)
            _ = self.vit(x)

            # Process ViT Features to match Grid
            # ViT: [B, 197, 768] -> Remove CLS -> [B, 196, 768]
            v_feat = self.v_feat[:, 1:, :]
            B, N, C = v_feat.shape
            H = W = int(N ** 0.5)  # 14
            v_feat = v_feat.permute(0, 2, 1).reshape(B, C, H, W)

            # ResNet Features: [B, 1024, 14, 14]
            r_feat = self.r_feat

            # Concatenate: [B, 1792, 14, 14]
            return torch.cat([r_feat, v_feat], dim=1)


def cache_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caching features on {device}...")

    # Create Cache Directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Model
    model = FeatureExtractor().to(device)

    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Point to your ImgListDataset
    dataset = ImgListDataset(args.data, args.imglist, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # Loop
    print(f"Starting extraction for {len(dataset)} images...")

    idx_counter = 0
    for batch in tqdm(loader):
        # --- FIX: Handle [Image, Label] or just [Image] ---
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]  # Grab just the image tensor
        else:
            imgs = batch
        # --------------------------------------------------

        imgs = imgs.to(device)
        features = model(imgs)  # [B, 1792, 14, 14]

        # Save each batch
        batch_data = {
            "features": features.cpu().half(),  # FP16 saves space
            "images": imgs.cpu().half()  # FP16
        }

        torch.save(batch_data, os.path.join(args.cache_dir, f"batch_{idx_counter}.pt"))
        idx_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imglist', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    cache_features(args)