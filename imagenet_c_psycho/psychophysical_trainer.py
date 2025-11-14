# openood/pipelines/train_pipeline.py
import os
import math
import time
import random
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

from torchvision import models

# Your custom dataset (created earlier)
from imagenet_c_psycho.psycho_imglist_dataset import PsychoImglistDataset

# Fallback: use OpenOOD's ImglistDataset for plain splits if available
try:
    from openood.datasets.imglist_dataset import ImglistDataset
    _HAS_OPENOOD_IMGLIST = True
except Exception:
    _HAS_OPENOOD_IMGLIST = False


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class BaseTrainer:
    """Standard CE trainer (used if you ever want a non-psychophysical run)."""
    def __init__(self, net, loader: DataLoader, cfg: Dict[str, Any], device: torch.device, logger: logging.Logger):
        self.net = net
        self.loader = loader
        self.cfg = cfg
        self.device = device
        self.logger = logger

        opt = cfg.get("optimizer", {})
        lr = opt.get("lr", 0.1)
        wd = opt.get("weight_decay", 5e-4)
        mom = opt.get("momentum", 0.9)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
        self.epochs = opt.get("num_epochs", 90)

        # cosine schedule over total steps
        self.total_steps = self.epochs * max(1, len(self.loader))
        self._step = 0
        self.base_lr = lr

    def _lr(self):
        # cosine anneal to 1e-6 * base_lr
        min_lr = 1e-6
        t = self._step / max(1, self.total_steps)
        return (min_lr + 0.5 * (self.base_lr - min_lr) * (1 + math.cos(math.pi * t)))

    def train_one_epoch(self, epoch: int):
        self.net.train()
        running = 0.0
        correct = 0
        nsamp = 0

        for batch in self.loader:
            x = batch["data"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            # step lr
            lr = self._lr()
            for pg in self.optim.param_groups:
                pg["lr"] = lr
            self._step += 1

            logits = self.net(x)
            loss = F.cross_entropy(logits, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                running += float(loss) * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                nsamp += x.size(0)

        return running / max(1, nsamp), correct / max(1, nsamp)


class PsychophysicalTrainer(BaseTrainer):
    """Per-sample weighted CE: weight comes from dataset['weight'] (severity→weight)."""
    def train_one_epoch(self, epoch: int):
        self.net.train()
        running = 0.0
        correct = 0
        nsamp = 0

        for batch in self.loader:
            x = batch["data"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            w = batch.get("weight", None)
            if w is None:
                w = torch.ones(x.size(0), device=x.device, dtype=torch.float32)
            else:
                w = w.to(self.device, non_blocking=True)

            # step lr
            lr = self._lr()
            for pg in self.optim.param_groups:
                pg["lr"] = lr
            self._step += 1

            logits = self.net(x)
            ce_per = F.cross_entropy(logits, y, reduction="none")
            loss = (ce_per * w).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                running += float(loss) * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                nsamp += x.size(0)

        return running / max(1, nsamp), correct / max(1, nsamp)


class TrainPipeline:
    """
    Lean training pipeline that:
      - builds dataloader(s) from your YAML
      - builds a backbone (torchvision or timm)
      - trains with either Standard CE or Psychophysical CE
      - logs to a simple Python logger
      - saves checkpoints to save_dir
    """
    def __init__(self, cfg: Dict[str, Any], save_dir: str, logger: logging.Logger):
        self.cfg = cfg
        self.save_dir = save_dir
        self.logger = logger

        seed = cfg.get("seed", cfg.get("mark", 0)) or 0
        set_seed(int(seed))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build dataloader (train split only; val/test handled by your eval tools later)
        self.train_loader = self._build_loader(cfg["dataset"]["train"], split_name="train")

        # Build network
        self.net = self._build_network(cfg.get("network", {})).to(self.device)

        # Choose trainer
        trainer_name = cfg.get("trainer", {}).get("name", "psychophysical").lower()
        if "psychophysical" in trainer_name:
            self.trainer = PsychophysicalTrainer(self.net, self.train_loader, cfg, self.device, logger)
            self.logger.info("Using PsychophysicalTrainer (per-sample weighted CE).")
        else:
            self.trainer = BaseTrainer(self.net, self.train_loader, cfg, self.device, logger)
            self.logger.info("Using BaseTrainer (standard CE).")

        # epochs (used for progress text only; actual schedule in trainer)
        self.epochs = cfg.get("optimizer", {}).get("num_epochs", 90)

        # create ckpt dir
        self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _build_loader(self, ds_cfg: Dict[str, Any], split_name: str) -> DataLoader:
        ds_class = ds_cfg.get("dataset_class", "ImglistDataset")

        if ds_class == "PsychoImglistDataset":
            dataset = PsychoImglistDataset(ds_cfg, split_name=split_name)
        else:
            if not _HAS_OPENOOD_IMGLIST:
                raise RuntimeError("ImglistDataset not found. Please ensure openood is installed.")
            dataset = ImglistDataset(ds_cfg, split_name=split_name)

        batch_size = int(ds_cfg.get("batch_size", 128))
        shuffle = bool(ds_cfg.get("shuffle", True))
        num_workers = int(self.cfg.get("dataset", {}).get("num_workers", 8))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def _build_network(self, net_cfg: Dict[str, Any]) -> nn.Module:
        name = net_cfg.get("name", "resnet50").lower()
        num_classes = int(net_cfg.get("num_classes", 1000))
        pretrained = bool(net_cfg.get("pretrained", False))

        # torchvision fallbacks
        if name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
            return model

        if name in ["vit_b16", "vit-b-16", "vit"]:
            if not _HAS_TIMM:
                raise RuntimeError("timm not installed; required for ViT. Install timm>=1.0.")
            model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
            return model

        # generic timm path if available
        if _HAS_TIMM:
            try:
                model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
                return model
            except Exception:
                pass

        raise ValueError(f"Unknown network backbone: {name}")

    def _save_ckpt(self, tag: str, extra: Dict[str, Any]):
        path = os.path.join(self.ckpt_dir, f"{tag}.pth")
        torch.save(
            {
                "net": self.net.state_dict(),
                "extra": extra,
                "cfg": self.cfg,
            },
            path,
        )
        self.logger.info(f"Saved checkpoint: {path}")

    def run(self):
        self.logger.info(f"Starting training for {self.epochs} epochs.")
        best_loss = float("inf")
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            loss, acc = self.trainer.train_one_epoch(ep)
            dt = time.time() - t0
            self.logger.info(f"Epoch {ep:03d} | loss={loss:.4f} | acc={acc:.4f} | time={dt:.1f}s")

            if loss < best_loss:
                best_loss = loss
                self._save_ckpt("best", {"epoch": ep, "loss": loss, "acc": acc})

        self._save_ckpt("final", {"epoch": self.epochs, "loss": loss, "acc": acc})
        self.logger.info("Training complete.")
