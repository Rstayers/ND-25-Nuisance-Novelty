import copy
import os
import re
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.datasets.imglist_dataset import ImglistDataset


class PsychoEnergyPostprocessor(BasePostprocessor):
    """
    Training-based OOD postprocessor with psychophysical, severity-weighted loss.

    - Fine-tunes the classifier head on ImageNet-C (and optionally clean ID).
    - Uses severity-dependent weights for CE.
    - Explicitly shapes an energy-based OOD score:

        * ID (clean + corrupted) -> low energy (more negative)
        * OOD                      -> high energy

    - At inference, uses energy as the post-hoc score (conf = -energy).
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = getattr(config.postprocessor, "postprocessor_args", {})

        # ---------------- Base hyperparams ----------------
        self.APS_mode = getattr(config.postprocessor, "APS_mode", False)

        self.ft_epochs = getattr(self.args, "ft_epochs", 1)
        self.ft_lr = getattr(self.args, "ft_lr", 5e-4)
        self.ft_weight_decay = getattr(self.args, "ft_weight_decay", 0.0)
        self.ft_batch_size = getattr(self.args, "ft_batch_size", 64)
        self.ft_num_workers = getattr(self.args, "ft_num_workers", 4)

        # severity weighting
        self.min_severity = getattr(self.args, "min_severity", 1)
        self.max_severity = getattr(self.args, "max_severity", 5)
        self.weight_min = getattr(self.args, "weight_min", 0.5)
        self.weight_max = getattr(self.args, "weight_max", 2.0)
        self.inverse = getattr(self.args, "inverse", False)

        # energy-based OOD shaping
        self.energy_T = getattr(self.args, "energy_T", 1.0)

        # how strongly to shape ID vs OOD energies
        self.lambda_in = getattr(self.args, "lambda_in", 0.1)
        self.lambda_out = getattr(self.args, "lambda_out", 0.1)

        # margins for energy (remember: more negative = more ID-like)
        # ID should satisfy: E(x_id) <= id_margin  (typically negative)
        # OOD should satisfy: E(x_ood) >= ood_margin (typically >= 0)
        self.id_margin = getattr(self.args, "id_margin", -6.0)
        self.ood_margin = getattr(self.args, "ood_margin", 0.0)

        # optionally include clean ID in FT with severity=0
        self.use_clean_id = getattr(self.args, "use_clean_id", True)

        # ---------------- APS search spaces ----------------
        self.ft_epochs_list = getattr(self.args, "ft_epochs_list", [self.ft_epochs])
        self.ft_lr_list = getattr(self.args, "ft_lr_list", [self.ft_lr])
        self.ft_weight_decay_list = getattr(
            self.args, "ft_weight_decay_list", [self.ft_weight_decay]
        )
        self.weight_min_list = getattr(
            self.args, "weight_min_list", [self.weight_min]
        )
        self.weight_max_list = getattr(
            self.args, "weight_max_list", [self.weight_max]
        )
        self.inverse_list = getattr(self.args, "inverse_list", [self.inverse])

        # (We keep APS hyperparam search focused on original knobs; you can
        #  extend args_dict later to include lambda_in/out, margins, etc.)
        self.args_dict = {
            "ft_lr": self.ft_lr_list,
            "ft_epochs": self.ft_epochs_list,
            "ft_weight_decay": self.ft_weight_decay_list,
            "weight_min": self.weight_min_list,
            "weight_max": self.weight_max_list,
            "inverse": self.inverse_list,
        }

        # ImageNet-C psycho dataset paths
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

        # logging
        self.log_stats = getattr(self.args, "log_stats", True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # internal state
        self.setup_flag = False        # whether FT has run for *current* hyperparams
        self.ft_net = None             # fine-tuned copy of backbone
        self.cached_net = None         # original backbone
        self.cached_id_loader_dict: Optional[Dict[str, DataLoader]] = None
        self.cached_ood_loader_dict: Optional[Dict[str, DataLoader]] = None

    # ------------------------------------------------------------------
    # APS hooks
    # ------------------------------------------------------------------
    def set_hyperparam(self, hyperparam_dict):
        """
        Called by Evaluator.hyperparam_search() for each config.
        """
        if "ft_lr" in hyperparam_dict:
            self.ft_lr = float(hyperparam_dict["ft_lr"])
        if "ft_epochs" in hyperparam_dict:
            self.ft_epochs = int(hyperparam_dict["ft_epochs"])
        if "ft_weight_decay" in hyperparam_dict:
            self.ft_weight_decay = float(hyperparam_dict["ft_weight_decay"])
        if "weight_min" in hyperparam_dict:
            self.weight_min = float(hyperparam_dict["weight_min"])
        if "weight_max" in hyperparam_dict:
            self.weight_max = float(hyperparam_dict["weight_max"])
        if "inverse" in hyperparam_dict:
            self.inverse = bool(hyperparam_dict["inverse"])

        # reset FT state
        self.setup_flag = False
        self.ft_net = None

        # In APS mode, re-fine-tune immediately for this combo
        if self.APS_mode and self.cached_net is not None:
            self._run_finetune()

    def get_hyperparam(self):
        """
        Called by Evaluator.hyperparam_search() to record the best config.
        Must return a dict using the same keys as args_dict.
        """
        return {
            "ft_lr": self.ft_lr,
            "ft_epochs": self.ft_epochs,
            "ft_weight_decay": self.ft_weight_decay,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "inverse": self.inverse,
        }

    # ----------------- severity → weight mapping -----------------
    def _severity_to_weight(self, severity: torch.Tensor) -> torch.Tensor:
        """
        severity = 0 -> highest weight
        severity = max_severity -> lowest weight (if inverse=False)

        This encodes the psychophysical prior:
        low-severity (and clean) errors are less "forgivable" than high-severity ones.
        """
        severity = severity.float()
        s_clamped = torch.clamp(severity, 0, self.max_severity)
        t = s_clamped / max(float(self.max_severity), 1.0)

        if self.inverse:
            # corruptions heavier, clean lighter
            base = self.weight_min + (self.weight_max - self.weight_min) * t
        else:
            # clean heavier, severe lighter
            base = self.weight_max - (self.weight_max - self.weight_min) * t

        return base

    def debug_print_weight_curve(self):
        sevs = torch.arange(0, self.max_severity + 1, dtype=torch.float32)
        w = self._severity_to_weight(sevs)
        print("[psycho] severity → weight mapping:")
        for s, wi in zip(sevs.tolist(), w.tolist()):
            print(f"  s={int(s)} -> w={wi:.3f}")

    # ----------------- energy-based OOD score -----------------
    def _energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        E(x) = -T * logsumexp(logits / T)
        Lower (more negative) → more ID-like, higher → more OOD-like.
        """
        T = float(self.energy_T)
        return -T * torch.logsumexp(logits / T, dim=1)

    # ----------------- loader creation -----------------
    def _build_psycho_loader_from_imglist(self, id_loader_dict):
        imglist_pth = self.psych_imglist_pth
        if not os.path.exists(imglist_pth):
            print(f"[psycho] WARNING: psych_imglist_pth '{imglist_pth}' not found; "
                  f"skipping FT.")
            return None

        # take config from ID loader
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

    def _get_clean_id_loader(self):
        if self.cached_id_loader_dict is None:
            return None
        # prefer train loader if available
        return (
            self.cached_id_loader_dict.get("train")
            or self.cached_id_loader_dict.get("val")
            or self.cached_id_loader_dict.get("test")
        )

    def _get_ood_loader(self):
        if self.cached_ood_loader_dict is None:
            return None
        # pick "nearood" if available, otherwise first loader
        if "nearood" in self.cached_ood_loader_dict:
            return self.cached_ood_loader_dict["nearood"]
        if "farood" in self.cached_ood_loader_dict:
            return self.cached_ood_loader_dict["farood"]
        return next(iter(self.cached_ood_loader_dict.values()))

    # ------------------------------------------------------------------
    # Core FT routine (used by both APS and non-APS)
    # ------------------------------------------------------------------
    def _run_finetune(self):
        if self.ft_epochs <= 0:
            self.setup_flag = True
            return

        if self.cached_id_loader_dict is None:
            return

        psycho_loader = self._build_psycho_loader_from_imglist(
            self.cached_id_loader_dict
        )
        if psycho_loader is None:
            self.setup_flag = True
            return

        clean_loader = self._get_clean_id_loader() if self.use_clean_id else None
        ood_loader = self._get_ood_loader()

        # start from original backbone each time
        net = copy.deepcopy(self.cached_net)
        self.ft_net = net

        net.to(self.device)
        net.train()

        # freeze everything except final linear head
        for p in net.parameters():
            p.requires_grad = False

        last_layer = None
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
            last_layer = net.fc
        elif hasattr(net, "classifier") and isinstance(net.classifier, nn.Linear):
            last_layer = net.classifier
        elif hasattr(net, "head") and isinstance(net.head, nn.Linear):
            last_layer = net.head

        if last_layer is None:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    last_layer = module

        if last_layer is None:
            print("[psycho] WARNING: no Linear head found; fine-tuning all params.")
            for p in net.parameters():
                p.requires_grad = True
            trainable_params = list(net.parameters())
        else:
            for p in last_layer.parameters():
                p.requires_grad = True
            trainable_params = list(last_layer.parameters())

        if self.log_stats:
            n_total = sum(p.numel() for p in net.parameters())
            n_train = sum(p.numel() for p in trainable_params)
            print(f"[psycho] Fine-tuning last layer only: "
                  f"{n_train} / {n_total} params trainable.")
            self.debug_print_weight_curve()

        optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.ft_lr,
            weight_decay=self.ft_weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        ce = nn.CrossEntropyLoss(reduction="none")

        if self.log_stats:
            sev_range = self.max_severity + 1
            sev_weight_sum = torch.zeros(sev_range, dtype=torch.float64)
            sev_loss_sum = torch.zeros(sev_range, dtype=torch.float64)
            sev_count = torch.zeros(sev_range, dtype=torch.float64)

        for epoch in range(self.ft_epochs):
            epoch_loss = 0.0
            num_samples = 0

            if clean_loader is not None:
                clean_iter = iter(clean_loader)
            else:
                clean_iter = None

            if ood_loader is not None:
                ood_iter = iter(ood_loader)
            else:
                ood_iter = None

            for batch in tqdm(
                psycho_loader,
                desc=f"[psycho] fine-tune epoch {epoch+1}/{self.ft_epochs}",
                leave=False,
            ):
                # ----------------- ID-corrupted batch (ImageNet-C) -----------------
                data = batch["data"].to(self.device, non_blocking=True)
                target = batch["label"].to(self.device, non_blocking=True)

                # ---- severity parsing ----
                if "severity" in batch:
                    severity = batch["severity"].to(self.device)
                else:
                    names = batch.get("image_name", None)
                    if names is None:
                        severity = torch.ones_like(
                            target, dtype=torch.float32, device=self.device
                        )
                    else:
                        parsed = []
                        for name in names:
                            name = str(name).replace("\\", "/")
                            parts = name.split("/")
                            sev = 1
                            for seg in parts:
                                if seg.isdigit() and int(seg) in range(
                                    0, self.max_severity + 1
                                ):
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

                        if self.log_stats and epoch == 0 and num_samples == 0:
                            print("[psycho:debug] Example image_names:")
                            for nm in list(names)[:8]:
                                print("   ", nm)
                            print("[psycho:debug] Example parsed severities:")
                            print(severity[:16].detach().cpu().tolist())

                weights_corr = self._severity_to_weight(severity)
                logits_corr = net(data)
                ce_corr = ce(logits_corr, target)  # per-sample CE

                # energy for corrupted ID
                energy_corr = self._energy(logits_corr)
                # ID energy margin loss for corrupted ID
                id_margin_loss_corr = F.relu(energy_corr - self.id_margin)

                # ----------------- Optional clean ID batch -----------------
                if clean_iter is not None:
                    try:
                        clean_batch = next(clean_iter)
                    except StopIteration:
                        clean_iter = iter(clean_loader)
                        clean_batch = next(clean_iter)

                    clean_data = clean_batch["data"].to(self.device, non_blocking=True)
                    clean_target = clean_batch["label"].to(
                        self.device, non_blocking=True
                    )

                    logits_clean = net(clean_data)
                    ce_clean = ce(logits_clean, clean_target)

                    # treat clean as severity=0
                    sev_clean = torch.zeros_like(
                        clean_target, dtype=torch.float32, device=self.device
                    )
                    weights_clean = self._severity_to_weight(sev_clean)

                    energy_clean = self._energy(logits_clean)
                    id_margin_loss_clean = F.relu(energy_clean - self.id_margin)

                    # combine ID losses
                    ce_id = torch.cat(
                        [ce_corr * weights_corr, ce_clean * weights_clean], dim=0
                    )
                    id_margin_loss = torch.cat(
                        [
                            id_margin_loss_corr * weights_corr,
                            id_margin_loss_clean * weights_clean,
                        ],
                        dim=0,
                    )

                    id_ce_loss = ce_id.mean()
                    id_energy_loss = id_margin_loss.mean()

                    # update stats with severities for both groups
                    if self.log_stats:
                        with torch.no_grad():
                            sev_int_corr = severity.long().clamp(
                                0, self.max_severity
                            )
                            for s in range(0, self.max_severity + 1):
                                mask = sev_int_corr == s
                                if mask.any():
                                    sev_weight_sum[s] += weights_corr[mask].sum().item()
                                    sev_loss_sum[s] += ce_corr[mask].sum().item()
                                    sev_count[s] += mask.sum().item()
                            # clean severity=0 stats
                            s = 0
                            sev_weight_sum[s] += weights_clean.sum().item()
                            sev_loss_sum[s] += ce_clean.sum().item()
                            sev_count[s] += clean_data.size(0)

                    id_batch_size = data.size(0) + clean_data.size(0)

                else:
                    # no clean ID, only corrupted ID
                    id_ce_loss = (ce_corr * weights_corr).mean()
                    id_energy_loss = (id_margin_loss_corr * weights_corr).mean()
                    id_batch_size = data.size(0)

                    if self.log_stats:
                        with torch.no_grad():
                            sev_int = severity.long().clamp(0, self.max_severity)
                            for s in range(0, self.max_severity + 1):
                                mask = sev_int == s
                                if mask.any():
                                    sev_weight_sum[s] += weights_corr[mask].sum().item()
                                    sev_loss_sum[s] += ce_corr[mask].sum().item()
                                    sev_count[s] += mask.sum().item()

                # ----------------- OOD energy margin loss -----------------
                if ood_iter is not None:
                    try:
                        ood_batch = next(ood_iter)
                    except StopIteration:
                        ood_iter = iter(ood_loader)
                        ood_batch = next(ood_iter)

                    ood_data = ood_batch["data"].to(self.device, non_blocking=True)
                    logits_ood = net(ood_data)
                    energy_ood = self._energy(logits_ood)
                    # OOD should satisfy E(x_ood) >= ood_margin
                    ood_energy_loss = F.relu(self.ood_margin - energy_ood).mean()
                else:
                    ood_energy_loss = torch.tensor(
                        0.0, device=self.device, dtype=torch.float32
                    )

                # ----------------- total loss -----------------
                loss = (
                    id_ce_loss
                    + self.lambda_in * id_energy_loss
                    + self.lambda_out * ood_energy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss) * id_batch_size
                num_samples += id_batch_size

            if num_samples > 0:
                print(
                    f"[psycho] epoch {epoch+1}/{self.ft_epochs} "
                    f"loss={epoch_loss / num_samples:.4f}"
                )

        if self.log_stats and "sev_count" in locals() and sev_count.sum() > 0:
            print("\n[psycho] Severity-wise stats over fine-tuning data:")
            for s in range(0, self.max_severity + 1):
                if sev_count[s] > 0:
                    mean_w = sev_weight_sum[s] / sev_count[s]
                    mean_loss = sev_loss_sum[s] / sev_count[s]
                    print(
                        f"  severity={s}: n={int(sev_count[s].item())}, "
                        f"mean_weight={mean_w:.3f}, mean_CE={mean_loss:.3f}"
                    )

        net.eval()
        self.setup_flag = True

    # ----------------- setup: called once by Evaluator -----------------
    def setup(self, net, id_loader_dict, ood_loader_dict):
        # cache original backbone + loaders for APS refits
        if self.cached_net is None:
            self.cached_net = net
            self.cached_id_loader_dict = id_loader_dict
            self.cached_ood_loader_dict = ood_loader_dict

        # In APS mode we let hyperparam_search() drive fine-tuning
        if self.APS_mode:
            return

        # Non-APS: single FT with base config
        if self.setup_flag:
            return

        self._run_finetune()

    # ----------------- inference: energy-based post-hoc score -----------------
    @torch.no_grad()
    def postprocess(self, net, data):
        """
        Use our fine-tuned copy if we've created one; otherwise fall back.

        We use the negative energy as an "ID confidence" score:
        higher (-E) -> more ID-like; lower -> more OOD-like.
        """
        model = self.ft_net if (self.ft_net is not None and self.setup_flag) else net
        model = model.to(self.device)
        model.eval()

        data = data.to(self.device, non_blocking=True)
        logits = model(data)
        energy = self._energy(logits)          # more negative = more ID
        conf = -energy                         # more positive = more ID
        pred = logits.argmax(dim=1)
        return pred, conf
