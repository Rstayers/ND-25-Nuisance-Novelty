# analysis/plots/ninco.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils_plot import ensure_dir, joinp, set_seaborn_style


def plot_ninco_ood_success_by_subset(
    df,
    out_root: str,
    fpr_focus: float = 0.05,
):
    """
    Barplot of NINCO OOD_CorrectReject rate per subset and backbone.
    """
    set_seaborn_style()
    save_dir = joinp(out_root, "ood_subsets")
    ensure_dir(save_dir)

    if "system_outcome" not in df.columns:
        print("[WARN] No 'system_outcome' column in NINCO data.")
        return

    if "fpr_target" in df.columns:
        df_plot = df[df["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if df_plot.empty:
            print(f"[WARN] No NINCO rows at FPR={fpr_focus}")
            return
    else:
        df_plot = df.copy()

    df_plot["ood_success"] = (df_plot["system_outcome"] == "OOD_CorrectReject").astype(float)

    grouped = (
        df_plot.groupby(["ood_subset", "backbone"])["ood_success"]
               .mean()
               .reset_index()
    )
    if grouped.empty:
        print("[WARN] No grouped NINCO data.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=grouped,
        x="ood_subset",
        y="ood_success",
        hue="backbone",
        ax=ax,
    )

    ax.set_xlabel("NINCO Subset")
    ax.set_ylabel("Correct Reject Rate")
    ax.set_title(f"NINCO OOD Correct Reject by Subset (FPR={fpr_focus:.2f})")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax.legend(
        title="Backbone",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    out_path = joinp(save_dir, f"ninco_subset_ood_success_FPR{fpr_focus:.2f}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[OK] Saved NINCO subset OOD success plot → {out_path}")


def plot_ninco_score_histograms(
    df,
    out_root: str,
    fpr_focus: float = 0.05,
):
    """
    Optional diagnostic: histogram of OOD scores for NINCO per backbone.
    Uses 'score' column.
    """
    set_seaborn_style()
    save_dir = joinp(out_root, "score_histograms")
    ensure_dir(save_dir)

    if "score" not in df.columns:
        print("[WARN] No 'score' column in NINCO data; skipping histograms.")
        return

    if "fpr_target" in df.columns:
        df_plot = df[df["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if df_plot.empty:
            print(f"[WARN] No NINCO rows at FPR={fpr_focus}")
            return
    else:
        df_plot = df.copy()

    backbones = sorted(df_plot["backbone"].dropna().unique())
    for bb in backbones:
        sub = df_plot[df_plot["backbone"] == bb]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(sub["score"], bins=40, kde=False, ax=ax)
        ax.set_xlabel("Detector Score")
        ax.set_ylabel("Count")
        ax.set_title(f"NINCO Score Distribution — {bb} (FPR={fpr_focus:.2f})")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        out_path = joinp(save_dir, f"ninco_scores_{bb}_FPR{fpr_focus:.2f}.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        print(f"[OK] Saved NINCO score histogram for {bb} → {out_path}")
