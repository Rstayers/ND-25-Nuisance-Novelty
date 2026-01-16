import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    use_torchvision: bool = False
    resnet_ckpt: Optional[str] = None
    vit_ckpt: Optional[str] = None
    convnext_ckpt: Optional[str] = None
    densenet_ckpt: Optional[str] = None
    swin_ckpt: Optional[str] = None  # Added Swin


@dataclass
class PathConfig:
    train_data: Optional[str] = None
    train_list: Optional[str] = None
    val_data: Optional[str] = None
    val_list: Optional[str] = None
    out_dir: Optional[str] = None
    ae_weights: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    mean: List[float]
    std: List[float]
    image_size: List[int]

    models: ModelConfig
    paths: PathConfig = field(default_factory=PathConfig)  # New section

    sticker_yaml: Optional[str] = None
    target_fracs: List[float] = field(default_factory=lambda: [0.35, 0.25, 0.20, 0.15, 0.05])


def load_config(path: str) -> DatasetConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    m_cfg = raw.get('models', {})
    model_config = ModelConfig(
        use_torchvision=m_cfg.get('use_torchvision', False),
        resnet_ckpt=m_cfg.get('resnet_ckpt', None),
        vit_ckpt=m_cfg.get('vit_ckpt', None),
        convnext_ckpt=m_cfg.get('convnext_ckpt', None),
        densenet_ckpt=m_cfg.get('densenet_ckpt', None),
        swin_ckpt=m_cfg.get('swin_ckpt', None),
    )

    p_cfg = raw.get('paths', {})
    path_config = PathConfig(
        train_data=p_cfg.get('train_data', None),
        train_list=p_cfg.get('train_list', None),
        val_data=p_cfg.get('val_data', None),
        val_list=p_cfg.get('val_list', None),
        out_dir=p_cfg.get('out_dir', None),
        ae_weights=p_cfg.get('ae_weights', None),
    )

    return DatasetConfig(
        name=raw.get('dataset', {}).get('name', 'generic'),
        num_classes=raw.get('dataset', {}).get('num_classes', 1000),
        mean=raw.get('dataset', {}).get('mean', [0.485, 0.456, 0.406]),
        std=raw.get('dataset', {}).get('std', [0.229, 0.224, 0.225]),
        image_size=raw.get('dataset', {}).get('image_size', [224, 224]),
        models=model_config,
        paths=path_config,
        sticker_yaml=raw.get('dataset', {}).get('sticker_yaml', None),
        target_fracs=raw.get('calibration', {}).get('target_fracs', [0.35, 0.25, 0.20, 0.15, 0.05])
    )