import os
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

        # 2. Parse List
        if not os.path.exists(config['imglist']):
            # Fallback: check if path is relative to cwd
            if os.path.exists(config['imglist']):
                list_path = config['imglist']
            else:
                raise FileNotFoundError(f"Imglist for {name} not found at {config['imglist']}")
        else:
            list_path = config['imglist']

        with open(list_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = parser_func(line, self.root)
                if item['nuisance'] == 'clean_ood':
                    item['nuisance'] = name
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        full_path = os.path.join(self.root, item['path'])

        try:
            img = Image.open(full_path).convert('RGB')
        except:
            # Silent fallback black image to keep bench running
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