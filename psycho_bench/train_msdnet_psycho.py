# psycho_bench/train_msdnet_psycho.py
from __future__ import annotations
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from msdnet_psycho_backbone import MSDNetPsycho
from psychophysics_loss import PsychWeightedCELoss, ExitIndexLoss
from difficulty import severity_to_difficulty, severity_to_target_exit
from bench.datasets import ImglistDataset, parse_imagenet_severity


def build_imglist_dataset(
    data_root: str,
    imglist_rel: str,
    data_dir_rel: str,
    transform,
) -> ImglistDataset:
    """
    Generic helper to build an ImglistDataset for either clean ImageNet or ImageNet-C.

    Args:
        data_root: root of the project data (e.g., 'data')
        imglist_rel: relative path to the imglist file (from data_root)
        data_dir_rel: relative path to the images root (from data_root)
        transform: torchvision transform to apply

    Returns:
        ImglistDataset instance returning (img, label, rel_path)
    """
    imglist_path = os.path.join(data_root, imglist_rel)
    data_dir = os.path.join(data_root, data_dir_rel)
    ds = ImglistDataset(imglist_path=imglist_path, data_dir=data_dir, transform=transform)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing images_largescale/ and benchmark_imglist/",
    )
    # Clean ImageNet imglist + data dir (labels are 0..999)
    parser.add_argument(
        "--clean_imglist",
        type=str,
        default="benchmark_imglist/imagenet/train_imagenet.txt",
        help="Relative path (from data_root) to clean ImageNet train imglist",
    )
    parser.add_argument(
        "--clean_data_dir",
        type=str,
        default="images_largescale/imagenet",
        help="Relative path (from data_root) to clean ImageNet images root",
    )

    # ImageNet-C imglist + data dir (labels are also 0..999, same as clean)
    parser.add_argument(
        "--imagenet_c_imglist",
        type=str,
        default="benchmark_imglist/imagenet_c_psycho/90clean_10c/test_imagenet_c_psycho.txt",
        help="Relative path (from data_root) to ImageNet-C imglist",
    )
    parser.add_argument(
        "--imagenet_c_data_dir",
        type=str,
        default="images_largescale",
        help="Relative path (from data_root) to ImageNet-C images root",
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Loss weights
    parser.add_argument("--lambda_clean", type=float, default=1.0,
                        help="Weight for clean ImageNet CE loss")
    parser.add_argument("--lambda_corr", type=float, default=1.0,
                        help="Weight for standard CE on corrupted samples")
    parser.add_argument("--lambda_psy", type=float, default=1.0,
                        help="Weight for psychophysical (difficulty + exit) losses on corrupted samples")
    parser.add_argument("--lambda_easy", type=float, default=1.0,
                        help="Strength of psych-weighting for easy corrupted samples")
    parser.add_argument("--lambda_exit", type=float, default=0.5,
                        help="Weight for exit-index loss on corrupted samples")

    parser.add_argument("--n_exits", type=int, default=5)
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/msdnet_psycho_imagenet_c_mixed.pth",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # Transforms (shared by clean and corrupted)
    # -------------------------------------------------------------
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # -------------------------------------------------------------
    # Datasets and loaders
    # -------------------------------------------------------------
    clean_ds = build_imglist_dataset(
        data_root=args.data_root,
        imglist_rel=args.clean_imglist,
        data_dir_rel=args.clean_data_dir,
        transform=transform,
    )
    imagenet_c_ds = build_imglist_dataset(
        data_root=args.data_root,
        imglist_rel=args.imagenet_c_imglist,
        data_dir_rel=args.imagenet_c_data_dir,
        transform=transform,
    )

    clean_loader = DataLoader(
        clean_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    corr_loader = DataLoader(
        imagenet_c_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    print(
        f"[INFO] Clean ImageNet samples: {len(clean_ds):,}, "
        f"ImageNet-C samples: {len(imagenet_c_ds):,}"
    )

    # -------------------------------------------------------------
    # Model + losses
    # -------------------------------------------------------------
    net = MSDNetPsycho(n_exits=args.n_exits).to(device)

    ce_standard = nn.CrossEntropyLoss()
    ce_psycho = PsychWeightedCELoss(lambda_easy=args.lambda_easy)
    exit_loss_fn = ExitIndexLoss(n_exits=args.n_exits, p_conf=1.0)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # -------------------------------------------------------------
    # Training loop: mix clean ImageNet + ImageNet-C each step
    # -------------------------------------------------------------
    for epoch in range(args.epochs):
        net.train()
        running_total = 0.0
        running_ce_clean = 0.0
        running_ce_corr = 0.0
        running_psy = 0.0
        running_exit = 0.0
        n_clean_total = 0
        n_corr_total = 0

        clean_iter = iter(clean_loader)
        corr_iter = iter(corr_loader)

        # tqdm over the clean loader to show per-batch progress in this epoch
        for step, (imgs_clean, labels_clean, _) in enumerate(
                tqdm(clean_iter, total=len(clean_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=80)
        ):
            try:
                imgs_corr, labels_corr, rel_paths_corr = next(corr_iter)
            except StopIteration:
                corr_iter = iter(corr_loader)
                imgs_corr, labels_corr, rel_paths_corr = next(corr_iter)


            imgs_clean = imgs_clean.to(device)
            labels_clean = labels_clean.to(device)

            imgs_corr = imgs_corr.to(device)
            labels_corr = labels_corr.to(device)

            # ------------ Psychophysics from ImageNet-C paths ------------
            severities = torch.tensor(
                [parse_imagenet_severity(rp) for rp in rel_paths_corr],
                device=device,
                dtype=torch.float32,
            )
            difficulties = torch.tensor(
                [severity_to_difficulty(float(s)) for s in severities],
                device=device,
            )
            exit_targets = torch.tensor(
                [severity_to_target_exit(float(s), args.n_exits) for s in severities],
                device=device,
                dtype=torch.long,
            )

            optimizer.zero_grad()

            # Forward clean
            logits_list_clean = net(imgs_clean, return_all=True)
            final_logits_clean = logits_list_clean[-1]

            # Forward corrupted
            logits_list_corr = net(imgs_corr, return_all=True)
            final_logits_corr = logits_list_corr[-1]

            # Losses
            loss_ce_clean = ce_standard(final_logits_clean, labels_clean)
            loss_ce_corr = ce_standard(final_logits_corr, labels_corr)
            loss_psy_corr = ce_psycho(final_logits_corr, labels_corr, difficulties)
            loss_exit_corr = exit_loss_fn(logits_list_corr, labels_corr, exit_targets)

            loss = (
                args.lambda_clean * loss_ce_clean
                + args.lambda_corr * loss_ce_corr
                + args.lambda_psy * (loss_psy_corr + args.lambda_exit * loss_exit_corr)
            )

            loss.backward()
            optimizer.step()

            bs_clean = imgs_clean.size(0)
            bs_corr = imgs_corr.size(0)

            running_total += loss.item() * (bs_clean + bs_corr)
            running_ce_clean += loss_ce_clean.item() * bs_clean
            running_ce_corr += loss_ce_corr.item() * bs_corr
            running_psy += loss_psy_corr.item() * bs_corr
            running_exit += loss_exit_corr.item() * bs_corr

            n_clean_total += bs_clean
            n_corr_total += bs_corr

        scheduler.step()

        denom_total = max(1, n_clean_total + n_corr_total)
        denom_corr = max(1, n_corr_total)
        denom_clean = max(1, n_clean_total)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"loss={running_total/denom_total:.4f} "
            f"ce_clean={running_ce_clean/denom_clean:.4f} "
            f"ce_corr={running_ce_corr/denom_corr:.4f} "
            f"psy_corr={running_psy/denom_corr:.4f} "
            f"exit_corr={running_exit/denom_corr:.4f}"
        )

    # -------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(net.state_dict(), args.save_path)
    print(f"[OK] Saved MSDNet-Psycho checkpoint → {args.save_path}")


if __name__ == "__main__":
    main()
