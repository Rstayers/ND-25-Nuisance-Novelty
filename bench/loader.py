import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from bench.datasets import get_dataset_config, ensure_dataset_exists


class ConfigurableDataset(Dataset):
    def __init__(self, name, transform=None):
        self.name = name
        self.transform = transform
        self.samples = []

        # 0. Ensure dataset exists (auto-download if needed)
        ensure_dataset_exists(name)

        # 1. Fetch Config
        config = get_dataset_config(name)
        self.root = config['root']
        parser_func = config['parser']

        num_samples = config.get('num_samples', None)
        sample_offset = config.get('sample_offset', 0)
        list_path = config['imglist']

        with open(list_path, 'r') as f:
            lines = f.readlines()

        # Deterministic Sampling (Critical for OOSA Threshold Stability)
        # First shuffle with fixed seed, then apply offset and limit
        if num_samples is not None or sample_offset > 0:
            random.seed(42)
            random.shuffle(lines)

            # Apply offset (skip first N samples)
            if sample_offset > 0:
                print(f"[{name}] Skipping first {sample_offset} samples...")
                lines = lines[sample_offset:]

            # Apply limit
            if num_samples is not None and len(lines) > num_samples:
                print(f"[{name}] Taking {num_samples} samples (from {len(lines)} available)...")
                lines = lines[:num_samples]

        for line in lines:
            if not line.strip(): continue
            item = parser_func(line, self.root)
            if item['nuisance'] == 'clean_ood':
                item['nuisance'] = name
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Try multiple path resolution strategies
        possible_paths = [
            os.path.join(self.root, item['path']),  # root + relative path
            item['path'],  # absolute path as-is
            os.path.join(self.root, os.path.basename(item['path'])),  # root + filename only
        ]

        img = None
        for full_path in possible_paths:
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path).convert('RGB')
                    break
                except Exception as e:
                    continue

        if img is None:
            # Print debug info for the first few failures only
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 5:
                print(f"[WARNING] Image not found. Tried paths:")
                for p in possible_paths:
                    print(f"  - {p} (exists: {os.path.exists(p)})")
                self._error_count += 1
            elif self._error_count == 5:
                print(f"[WARNING] Suppressing further path warnings...")
                self._error_count += 1
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        return {
            'data': img,
            'label': item['label'],
            'path': item['path'],
            'level': item['level'],
            'parce': item['parce'],
            'nuisance': item['nuisance'],
            'dataset_name': self.name
        }


def get_loader(name, batch_size=64):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    dataset = ConfigurableDataset(name, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)