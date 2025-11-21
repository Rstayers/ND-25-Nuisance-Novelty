# psycho_bench/backbones.py

from pathlib import Path
from typing import Dict

import torch

from openood.networks.resnet50 import ResNet50
from torchvision.models import vit_b_16, ViT_B_16_Weights
from openood.networks.vit_b_16 import ViT_B_16


def load_psycho_backbones(base_dir: str = "results") -> Dict[str, str]:
    """
    Find psycho checkpoints like:
      results/30clean_70c_resnet50_openood/checkpoints/final.pth
    and map them to backbone names like:
      resnet50-psycho-30clean_70c
    """
    ckpts: Dict[str, str] = {}
    for path in Path(base_dir).glob("*_resnet50/checkpoints/final.pth"):
        # e.g. path.parts[-3] == "30clean_70c_resnet50_openood"
        variant_dir = path.parts[-3]
        ratio_tag = variant_dir.replace("_resnet50", "")  # "30clean_70c"
        backbone_name = f"resnet50-psycho-{ratio_tag}"
        ckpts[backbone_name] = str(path)
    return ckpts


def build_backbone(
    backbone: str,
    device: str,
    psycho_ckpts: Dict[str, str],
) -> torch.nn.Module:
    bb = backbone.lower()

    # auto-discovered psycho variants
    if bb.startswith("resnet50-psycho-"):
        if backbone not in psycho_ckpts:
            raise FileNotFoundError(
                f"No discovered checkpoint for {backbone} in results/*_resnet50_openood"
            )
        ckpt_path = psycho_ckpts[backbone]
        net = ResNet50(num_classes=1000)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        net.load_state_dict(state, strict=False)
        return net.to(device).eval()

    # plain ImageNet pretrained resnet50
    if bb == "resnet50":
        net = ResNet50(num_classes=1000)
        ckpt_path = (
            "./data/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt, strict=False)
        return net.to(device).eval()

    # ViT option if you want it
    if bb in ["vit", "vit_b_16"]:
        tv_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        net = ViT_B_16(num_classes=1000)
        net.load_state_dict(tv_model.state_dict(), strict=False)
        return net.to(device).eval()

    raise ValueError(f"Unsupported backbone: {backbone}")
