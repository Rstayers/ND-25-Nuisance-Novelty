import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from bench.datasets import get_dataset_config


class ConfigurableDataset(Dataset):
    def __init__(self, name, transform=None):
        self.name = name
        self.transform = transform
        self.samples = []

        # 1. Fetch Config
        config = get_dataset_config(name)
        self.root = config['root']
        parser_func = config['parser']

        # Check for sample limit
        num_samples = config.get('num_samples', None)  # <--- New Config Option

        # 2. Parse List
        if not os.path.exists(config['imglist']):
            # ... (keep existing fallback logic) ...
            pass

        list_path = config['imglist']

        # --- MODIFIED LOADING LOGIC ---
        with open(list_path, 'r') as f:
            lines = f.readlines()

        # Deterministic Sampling (Critical for OOSA Threshold Stability)
        if num_samples is not None and len(lines) > num_samples:
            print(f"[{name}] Downsampling from {len(lines)} to {num_samples} (Seed=42)...")
            random.seed(42)
            lines = random.sample(lines, num_samples)

        for line in lines:
            if not line.strip(): continue
            item = parser_func(line, self.root)
            if item['nuisance'] == 'clean_ood':
                item['nuisance'] = name
            self.samples.append(item)
        # ------------------------------

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

        # FIX: OpenOOD expects key 'data', not 'image'
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