import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# ADDED 'swin_t' here
MODELS_TO_TRAIN = ['resnet50', 'densenet121', 'convnext_t', 'vit_b_16', 'swin_t']
NUM_CLASSES = 200
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4


class ImgListDataset(Dataset):
    def __init__(self, root, imglist_path, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    path = " ".join(parts[:-1])
                    label = int(parts[-1])
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root, path)
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception:
            print(f"Warning: Corrupt image {full_path}")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
        return img, label


def get_model(arch, num_classes, device):
    print(f"--> Initializing {arch}...")

    if arch == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif arch == 'convnext_t':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif arch == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # ADDED THIS BLOCK
    elif arch == 'swin_t':
        model = models.swin_t(weights='IMAGENET1K_V1')
        model.head = nn.Linear(model.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown arch {arch}")

    return model.to(device)


def train_one_model(arch, train_loader, val_loader, device, save_dir):
    save_path = os.path.join(save_dir, f"{arch}_cars.pth")

    if os.path.exists(save_path):
        print(f"[SKIP] {arch} already exists at {save_path}")
        return

    model = get_model(arch, NUM_CLASSES, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"[{arch}] Ep {epoch + 1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        epoch_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        print(f"   Ep {epoch + 1}: Train {epoch_acc:.4f} | Val {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_weights, save_path)
    print(f"--> Saved {arch} (Acc: {best_acc:.4f}) to {save_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=".", help="Root folder")
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="ln_dataset/assets/cars_models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImgListDataset(args.data_root, args.train_list, transform=train_transform)
    val_ds = ImgListDataset(args.data_root, args.val_list, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val")

    for arch in MODELS_TO_TRAIN:
        train_one_model(arch, train_loader, val_loader, device, args.out_dir)


if __name__ == "__main__":
    main()