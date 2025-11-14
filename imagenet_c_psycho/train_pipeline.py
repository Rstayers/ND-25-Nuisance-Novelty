import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models

# custom dataset
from imagenet_c_psycho.psycho_imglist_dataset import PsychoImglistDataset

# optional timm for ViT
try:
    import timm
    _HAS_TIMM = True
except:
    _HAS_TIMM = False


def _as_int(x, default=0):
    try:
        return int(x)
    except:
        return default


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ------------ LR Scheduler Builder ------------

def _build_scheduler(optimizer, cfg):
    sch = cfg.get("scheduler", {})
    name = (sch.get("name") or "multistep").lower()

    if name in ("multistep", "step"):
        milestones = sch.get("milestones", [10, 20])
        gamma = float(sch.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m) for m in milestones],
            gamma=gamma
        )
    return None


# ------------ Trainers ------------

class _BaseTrainer:
    def __init__(self, net, loader, cfg, device, logger):
        self.net = net
        self.loader = loader
        self.cfg = cfg
        self.device = device
        self.logger = logger

        opt_cfg = cfg.get("optimizer", {})
        self.epochs = _as_int(opt_cfg.get("num_epochs", 30), 30)
        lr = float(opt_cfg.get("lr", 0.001))
        wd = float(opt_cfg.get("weight_decay", 5e-4))
        mom = float(opt_cfg.get("momentum", 0.9))

        self.optim = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True
        )
        self.sched = _build_scheduler(self.optim, cfg)

    def train_one_epoch(self):
        self.net.train()
        total_loss, total_acc, n = 0, 0, 0

        for batch in self.loader:
            x = batch["data"].to(self.device)
            y = batch["label"].to(self.device)

            logits = self.net(x)
            loss = F.cross_entropy(logits, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                acc = (logits.argmax(1) == y).float().mean().item()

            bs = x.size(0)
            total_loss += float(loss) * bs
            total_acc += acc * bs
            n += bs

        return total_loss / n, total_acc / n


from tqdm import tqdm

class _PsychophysicalTrainer(_BaseTrainer):
    def train_one_epoch(self):
        self.net.train()
        total_loss, total_acc, n = 0, 0, 0

        # nice per-batch progress bar
        pbar = tqdm(self.loader, desc="train", leave=False)

        for batch in pbar:
            x = batch["data"].to(self.device)
            y = batch["label"].to(self.device)
            w = batch["weight"].to(self.device)

            logits = self.net(x)
            ce_per = F.cross_entropy(logits, y, reduction="none")
            loss = (ce_per * w).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                acc = (logits.argmax(1) == y).float().mean().item()

            bs = x.size(0)
            total_loss += float(loss) * bs
            total_acc += acc * bs
            n += bs

            # update progress bar info
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "batch_acc": f"{acc:.4f}",
                "lr": self.optim.param_groups[0]["lr"],
            })

        # at end of epoch
        return total_loss / n, total_acc / n



# ------------ Main Pipeline ------------

class TrainPipeline:
    def __init__(self, cfg, save_dir, logger):
        self.cfg = cfg
        self.save_dir = save_dir
        self.logger = logger

        set_seed(_as_int(cfg.get("seed", 0)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loaders
        ds_cfg = cfg.get("dataset", {})
        self.train_loader = self._build_loader(ds_cfg.get("train", {}), ds_cfg, "train")
        self.val_loader = self._build_loader(ds_cfg.get("val", {}), ds_cfg, "val") if "val" in ds_cfg else None

        # network
        self.net = self._build_network(cfg.get("network", {})).to(self.device)

        # trainer type
        tname = cfg.get("trainer", {}).get("name", "").lower()
        if "psychophysical" in tname:
            self.trainer = _PsychophysicalTrainer(self.net, self.train_loader, cfg, self.device, logger)
            logger.info("Using psychophysical trainer.")
        else:
            self.trainer = _BaseTrainer(self.net, self.train_loader, cfg, self.device, logger)
            logger.info("Using base trainer.")

        self.epochs = self.trainer.epochs

        # checkpoint directory
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------ Loader Builder ------
    def _build_loader(self, split_cfg, root_cfg, split_name):

        cfg = dict(split_cfg)
        if split_name == "train":
            cfg["__augment__"] = self.cfg.get("augment", {})
        else:
            cfg["__augment__"] = {}

        dataset = PsychoImglistDataset(cfg, split_name=split_name)

        batch_size = _as_int(split_cfg.get("batch_size", 128), 128)
        shuffle = bool(split_cfg.get("shuffle", split_name == "train"))
        workers = _as_int(root_cfg.get("num_workers", 8), 8)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=True, drop_last=False)

    # ------ Network Builder ------
    def _build_network(self, net_cfg):
        name = net_cfg.get("name", "resnet50").lower()
        num_classes = _as_int(net_cfg.get("num_classes", 1000), 1000)
        pretrained = bool(net_cfg.get("pretrained", False))

        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        if _HAS_TIMM:
            return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

        raise ValueError(f"Unsupported model: {name}")

    # ------ Checkpoint ------
    def _save_ckpt(self, tag, extra=None):
        path = os.path.join(self.ckpt_dir, f"{tag}.pth")
        torch.save({
            "net": self.net.state_dict(),
            "cfg": self.cfg,
            "extra": extra or {},
        }, path)
        self.logger.info(f"Saved checkpoint: {path}")

    # ------ Training Loop ------
    def run(self):
        best_val = float("inf")
        best_epoch = -1
        last_loss = 0.0

        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.trainer.train_one_epoch()

            # step scheduler once per epoch
            if self.trainer.sched is not None:
                self.trainer.sched.step()

            self.logger.info(f"Epoch {ep:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | t={time.time()-t0:.1f}s")
            last_loss = train_loss

            # ------ validation ------
            if self.val_loader is not None:
                self.net.eval()
                vloss, vacc, n = 0, 0, 0
                with torch.no_grad():
                    for b in self.val_loader:
                        x = b["data"].to(self.device)
                        y = b["label"].to(self.device)
                        logits = self.net(x)
                        loss = F.cross_entropy(logits, y)
                        bs = x.size(0)
                        vloss += float(loss) * bs
                        vacc += (logits.argmax(1) == y).float().sum().item()
                        n += bs

                vloss /= n
                vacc /= n

                self.logger.info(f"           | val_loss={vloss:.4f} | val_acc={vacc:.4f}")

                if vloss < best_val:
                    best_val = vloss
                    best_epoch = ep
                    self._save_ckpt("best", {"epoch": ep, "val_loss": vloss, "val_acc": vacc})

        self._save_ckpt("final", {"epoch": self.epochs, "train_loss": last_loss, "best_val": best_val})

        self.logger.info(f"Training complete. Best epoch = {best_epoch}, best val = {best_val:.4f}")
