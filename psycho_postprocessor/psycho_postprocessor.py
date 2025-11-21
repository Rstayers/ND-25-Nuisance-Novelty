import copy
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.datasets.imglist_dataset import ImglistDataset


class PsychoPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = getattr(config.postprocessor, "postprocessor_args", {})

        # fine-tune hyperparams
        self.ft_epochs = getattr(self.args, "ft_epochs", 1)
        self.ft_lr = getattr(self.args, "ft_lr", 5e-4)
        self.ft_weight_decay = getattr(self.args, "ft_weight_decay", 1e-4)
        self.ft_batch_size = getattr(self.args, "ft_batch_size", 32)
        self.ft_num_workers = getattr(self.args, "ft_num_workers", 4)

        # severity weighting config
        self.min_severity = getattr(self.args, "min_severity", 1)
        self.max_severity = getattr(self.args, "max_severity", 5)
        self.weight_min = getattr(self.args, "weight_min", 0.5)
        self.weight_max = getattr(self.args, "weight_max", 2.0)
        self.inverse = getattr(self.args, "inverse", False)

        # explicit ImageNet-C list + data dir
        self.psych_imglist_pth = getattr(
            self.args,
            "psych_imglist_pth",
            "./data/benchmark_imglist/imagenet/test_imagenet_c.txt",
        )
        self.psych_data_dir = getattr(
            self.args,
            "psych_data_dir",
            "./data/images_largescale/",
        )

        # debug logging
        self.log_stats = getattr(self.args, "log_stats", True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_flag = False
        self.APS_mode = False

    # ----------------- weighting: s=0 highest weight -----------------
    def _severity_to_weight(self, severity: torch.Tensor) -> torch.Tensor:
        """
        severity = 0 -> highest weight
        severity = max_severity -> lowest weight
        """
        severity = severity.float()
        s_clamped = torch.clamp(severity, 0, self.max_severity)
        t = s_clamped / max(float(self.max_severity), 1.0)

        if self.inverse:
            # inverse=True: punish corruptions more, clean less
            base = self.weight_min + (self.weight_max - self.weight_min) * t
        else:
            # normal: clean (0) -> weight_max, severe (max) -> weight_min
            base = self.weight_max - (self.weight_max - self.weight_min) * t

        return base

    def debug_print_weight_curve(self):
        sevs = torch.arange(0, self.max_severity + 1, dtype=torch.float32)
        w = self._severity_to_weight(sevs)
        print("[psycho] severity → weight mapping:")
        for s, wi in zip(sevs.tolist(), w.tolist()):
            print(f"  s={int(s)} -> w={wi:.3f}")

    # ----------------- build our own ImageNet-C loader -----------------
    def _build_psycho_loader_from_imglist(self, id_loader_dict):
        imglist_pth = self.psych_imglist_pth
        if not os.path.exists(imglist_pth):
            print(f"[psycho] WARNING: psych_imglist_pth '{imglist_pth}' not found; skipping FT.")
            return None

        # steal num_classes / preprocessors from ID dataset
        base_loader = (
            id_loader_dict.get("train")
            or id_loader_dict.get("test")
            or id_loader_dict.get("val")
            or next(iter(id_loader_dict.values()))
        )
        base_dataset = base_loader.dataset

        num_classes = getattr(base_dataset, "num_classes", 1000)
        preprocessor = getattr(base_dataset, "preprocessor", None)
        data_aux_preprocessor = getattr(
            base_dataset, "data_aux_preprocessor", preprocessor
        )

        print(f"[psycho] Loading psychophysical dataset from {imglist_pth}")
        dataset = ImglistDataset(
            name="imagenet_c_psycho",
            imglist_pth=imglist_pth,
            data_dir=self.psych_data_dir,
            num_classes=num_classes,
            preprocessor=preprocessor,
            data_aux_preprocessor=data_aux_preprocessor,
            dummy_read=False,
            dummy_size=None,
        )

        batch_size = int(self.ft_batch_size or 32)
        num_workers = int(self.ft_num_workers or 4)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader

    # ----------------- setup: fine-tune on ImageNet-C -----------------
    def setup(self, net, id_loader_dict, ood_loader_dict):
        if self.setup_flag or self.ft_epochs <= 0:
            self.setup_flag = True
            return


        psycho_loader = self._build_psycho_loader_from_imglist(id_loader_dict)
        if psycho_loader is None:
            self.setup_flag = True
            return

        net.to(self.device)
        net.train()
        # --------------------------------------------------
        # Freeze all layers except the final classifier head
        # --------------------------------------------------
        # 1) Freeze everything
        for p in net.parameters():
            p.requires_grad = False

        # 2) Try to identify the classifier head
        last_layer = None

        # Common cases: ResNet, ViT, etc.
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
            last_layer = net.fc
        elif hasattr(net, "classifier") and isinstance(net.classifier, nn.Linear):
            last_layer = net.classifier
        elif hasattr(net, "head") and isinstance(net.head, nn.Linear):
            last_layer = net.head

        # Fallback: last nn.Linear in the module tree
        if last_layer is None:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    last_layer = module

        if last_layer is None:
            print("[psycho] WARNING: could not find a Linear classifier head; "
                  "fine-tuning ALL parameters.")
            for p in net.parameters():
                p.requires_grad = True
            trainable_params = list(net.parameters())
        else:
            for p in last_layer.parameters():
                p.requires_grad = True
            trainable_params = list(last_layer.parameters())

        # Debug: how many params are we actually training?
        if self.log_stats:
            n_total = sum(p.numel() for p in net.parameters())
            n_train = sum(p.numel() for p in trainable_params)
            print(f"[psycho] Fine-tuning last layer only: "
                  f"{n_train} / {n_total} params trainable.")

        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.ft_lr,
            weight_decay=self.ft_weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        ce = nn.CrossEntropyLoss(reduction="none")

        if self.log_stats:
            self.debug_print_weight_curve()
            sev_range = self.max_severity + 1
            sev_weight_sum = torch.zeros(sev_range, dtype=torch.float64)
            sev_loss_sum = torch.zeros(sev_range, dtype=torch.float64)
            sev_count = torch.zeros(sev_range, dtype=torch.float64)

        num_samples = 0

        for epoch in range(self.ft_epochs):
            epoch_loss = 0.0
            num_samples = 0

            for batch in tqdm(
                psycho_loader,
                desc=f"[psycho] fine-tune epoch {epoch+1}/{self.ft_epochs}",
                leave=False,
            ):
                data = batch["data"].to(self.device, non_blocking=True)
                target = batch["label"].to(self.device, non_blocking=True)

                # ---------- SEVERITY PARSING ----------
                if "severity" in batch:
                    severity = batch["severity"].to(self.device)
                else:
                    names = batch.get("image_name", None)
                    if names is None:
                        # best-effort fallback
                        severity = torch.ones_like(
                            target, dtype=torch.float32, device=self.device
                        )
                    else:
                        parsed = []
                        for name in names:
                            name = str(name).replace("\\", "/")
                            parts = name.split("/")
                            sev = 1  # default if none found
                            for seg in parts:
                                # severity as folder '1'..'5' or suffix '_1'..'_5'
                                if seg.isdigit() and int(seg) in range(0, self.max_severity + 1):
                                    sev = int(seg)
                                else:
                                    m = re.search(r"_(\d)$", seg)
                                    if m:
                                        v = int(m.group(1))
                                        if v in range(0, self.max_severity + 1):
                                            sev = v
                            parsed.append(sev)
                        severity = torch.tensor(
                            parsed, dtype=torch.float32, device=self.device
                        )

                    # one-time debug print
                    if self.log_stats and epoch == 0 and num_samples == 0:
                        print("[psycho:debug] Example image_names:")
                        for nm in list(names)[:8]:
                            print("   ", nm)
                        print("[psycho:debug] Example parsed severities:")
                        print(severity[:16].detach().cpu().tolist())

                # ---------- loss + step ----------
                weights = self._severity_to_weight(severity)
                logits = net(data)
                ce_per_sample = ce(logits, target)
                loss = (ce_per_sample * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = data.size(0)
                epoch_loss += float(loss) * bs
                num_samples += bs

                if self.log_stats:
                    with torch.no_grad():
                        sev_int = severity.long().clamp(0, self.max_severity)
                        for s in range(0, self.max_severity + 1):
                            mask = sev_int == s
                            if mask.any():
                                sev_weight_sum[s] += weights[mask].sum().item()
                                sev_loss_sum[s] += ce_per_sample[mask].sum().item()
                                sev_count[s] += mask.sum().item()

            if num_samples > 0:
                print(
                    f"[psycho] epoch {epoch+1}/{self.ft_epochs} "
                    f"loss={epoch_loss / num_samples:.4f}"
                )

        if self.log_stats and sev_count.sum() > 0:
            print("\n[psycho] Severity-wise stats over fine-tuning data:")
            for s in range(0, self.max_severity + 1):
                if sev_count[s] > 0:
                    mean_w = sev_weight_sum[s] / sev_count[s]
                    mean_loss = sev_loss_sum[s] / sev_count[s]
                    print(
                        f"  severity={s}: "
                        f"n={int(sev_count[s].item())}, "
                        f"mean_weight={mean_w:.3f}, "
                        f"mean_CE={mean_loss:.3f}"
                    )

        net.eval()
        self.setup_flag = True

    # ----------------- inference: MSP on fine-tuned net -----------------
    @torch.no_grad()
    def postprocess(self, net, data):
        p = F.softmax(net(data), dim=1)
        conf, pred = torch.max(p, 1)
        return pred, conf
