#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbone-aware nuisance novelty analysis (ImageNet-C, ID-only)

Highlights
- Absolute proportions only (no conditional splits).
- Backbone is included in all group-bys and output paths.
- Severity curves: 2x2 paired comparisons per (backbone, detector) at FPR=0.05.
- Head-to-head curves per backbone: full, partial, and total nuisance across severity.
- Heatmaps per (backbone, detector) ranked by mean total nuisance (full + partial).
- Stacked composition per (backbone, detector) at chosen FPR.
- Classifier accuracy vs severity per backbone at chosen FPR.

Required CSV columns:
    image_path, accept, detector, fpr_target, correct_cls
Recommended:
    backbone, dataset, corruption, severity, error_type

Usage:
    python analyze_results.py --csv /path/to/nuisance_full_set.csv --out_dir results/analysis --fpr 0.05
"""
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# -----------------------
# Utils
# -----------------------

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def joinp(*xs):
    return os.path.join(*[str(x) for x in xs])

def wilson_ci(k, n, z=1.96):
    """(Kept in case you later want CIs again)"""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def parse_from_path(path: str):
    """
    Parse corruption and severity from an ImageNet-C filepath:
      .../imagenet_c/<corruption>/<severity>/file.jpeg
    """
    parts = Path(path).parts
    idxs = [i for i, p in enumerate(parts) if str(p).lower().startswith("imagenet_c")]
    if not idxs:
        return "unknown", -1
    idx = idxs[0]
    try:
        corruption = parts[idx+1]
        severity = int(parts[idx+2])
    except Exception:
        return "unknown", -1
    return str(corruption), int(severity)

def add_corruption_severity(df: pd.DataFrame) -> pd.DataFrame:
    need_corr = "corruption" not in df.columns
    need_sev = "severity" not in df.columns
    if need_corr or need_sev:
        if "image_path" not in df.columns:
            raise ValueError("Missing 'image_path'; cannot infer corruption/severity from path.")
        corr_sev = df["image_path"].apply(lambda p: pd.Series(parse_from_path(str(p)),
                                                              index=["corruption", "severity"]))
        for col in ["corruption","severity"]:
            if col not in df.columns:
                df[col] = corr_sev[col]
    return df

def load_with_enrichment(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # standardize / coerce
    for col, typ in [("accept", int), ("correct_cls", int)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(typ)

    if "fpr_target" in df.columns:
        df["fpr_target"] = pd.to_numeric(df["fpr_target"], errors="coerce")

    if "backbone" not in df.columns:
        df["backbone"] = "unknown"

    # keep ImageNet-C rows if dataset is mixed
    if "dataset" in df.columns and "image_path" in df.columns:
        mask_c = df["dataset"].astype(str).str.lower().str.contains("imagenet") & \
                 df["image_path"].astype(str).str.lower().str.contains("imagenet_c")
        if mask_c.any():
            df = df[mask_c].copy()

    df = add_corruption_severity(df)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)
    return df

# -----------------------
# Metrics (ABSOLUTE ONLY)
# -----------------------
def aggregate_rates(df: pd.DataFrame, group_cols):
    """
    Absolute proportions across the whole dataset (no conditional splits).
    Returns counts and absolute fractions for:
      - ideal_accept               (correct == 1 & accept == 1)
      - full_nuisance              (correct == 1 & accept == 0)
      - partial_nuisance           (correct == 0 & accept == 0)
      - misclassified_accept       (correct == 0 & accept == 1)
    """
    df = df.copy()
    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, sub in grouped:
        n = len(sub)
        if n == 0:
            continue
        n_ideal   = ((sub["correct_cls"] == 1) & (sub["accept"] == 1)).sum()
        n_full    = ((sub["correct_cls"] == 1) & (sub["accept"] == 0)).sum()
        n_partial = ((sub["correct_cls"] == 0) & (sub["accept"] == 0)).sum()
        n_misacc  = ((sub["correct_cls"] == 0) & (sub["accept"] == 1)).sum()

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": n,
            "full_correct": n_ideal / n,
            "full_nuisance": n_full / n,
            "partial_nuisance": n_partial / n,
            "partial_correct": n_misacc / n,
            "n_ideal": n_ideal, "n_full": n_full, "n_partial": n_partial, "n_misacc": n_misacc
        })
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_outcomes(df: pd.DataFrame, group_cols):
    """
    For stacked composition (if error_type exists).
    """
    if "error_type" not in df.columns:
        return pd.DataFrame(), []
    one_hot = pd.get_dummies(df["error_type"])
    tmp = pd.concat([df[group_cols], one_hot], axis=1)
    agg = tmp.groupby(group_cols, dropna=False).sum(numeric_only=True).reset_index()
    outcome_cols = [c for c in agg.columns if c not in group_cols]
    agg["total"] = agg[outcome_cols].sum(axis=1).clip(lower=1)
    for c in outcome_cols:
        agg[c + "_prop"] = agg[c] / agg["total"]
    return agg, outcome_cols

# -----------------------
# Plotters (per BACKBONE)
# -----------------------
def _per_backbone_loop(df: pd.DataFrame):
    bbs = sorted(df["backbone"].dropna().unique())
    for bb in bbs:
        yield bb, df[df["backbone"] == bb].copy()

def plot_severity_pairs_2x2(df: pd.DataFrame, out_root: str, fpr_focus: float = 0.05):
    """
    For each (backbone, detector): a 2x2 grid of paired comparisons over severity at fixed FPR.
    TL: Ideal vs Full nuisance
    TR: Partial nuisance vs Misclassified accept
    BL: Misclassified accept vs Ideal
    BR: Full nuisance vs Partial
    """

    # unified palette (same as composition)
    palette = {
        "full_correct":   "#006a00",  # green
        "partial_correct":"#47ac00",  # yellow-green
        "partial_nuisance":"#ff9d00", # orange
        "full_nuisance":  "#ff0000"   # red
    }

    for backbone, bb_df in _per_backbone_loop(df):
        save_dir = joinp(out_root, "severity_curves", f"backbone={backbone}")
        ensure_dir(save_dir)

        met = aggregate_rates(bb_df, ["backbone", "detector", "fpr_target", "severity"])
        met = met[(met["severity"].between(1,5)) & (np.isclose(met["fpr_target"], fpr_focus))]
        if met.empty:
            print(f"[WARN] No rows at FPR={fpr_focus} for severity curves (backbone={backbone}).")
            continue

        for det in sorted(met["detector"].dropna().unique()):
            sub = met[met["detector"] == det].sort_values("severity")
            if sub.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            (ax1, ax2), (ax3, ax4) = axes

            # TL: Ideal vs Full nuisance
            ax1.plot(sub["severity"], sub["full_correct"], marker="^",
                     color=palette["full_correct"], label="Full Correct (Ideal Case)")
            ax1.plot(sub["severity"], sub["full_nuisance"], marker="o",
                     color=palette["full_nuisance"], label="Full nuisance (Worst Case)")
            ax1.set_title("Ideal vs Full nuisance"); ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.4)

            # TR: Partial vs Misclassified
            ax2.plot(sub["severity"], sub["partial_nuisance"], marker="s",
                     color=palette["partial_nuisance"], label="Partial nuisance (Both Layers Fail)")
            ax2.plot(sub["severity"], sub["partial_correct"], marker="v",
                     color=palette["partial_correct"], label="Correct but misclassified (Happy Accident)")
            ax2.set_title("Partial nuisance vs Misclassified accept"); ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)

            # BL: Misclassified vs Ideal
            ax3.plot(sub["severity"], sub["partial_correct"], marker="v",
                     color=palette["partial_correct"], label="Partial Correct (Happy Accident)")
            ax3.plot(sub["severity"], sub["full_correct"], marker="^",
                     color=palette["full_correct"], label="Full Correct (Ideal Case)")
            ax3.set_title("Misclassified accept vs Ideal accept"); ax3.legend(); ax3.grid(True, linestyle="--", alpha=0.4)

            # BR: Full vs Partial
            ax4.plot(sub["severity"], sub["full_nuisance"], marker="o",
                     color=palette["full_nuisance"], label="Full nuisance (Worst Case)")
            ax4.plot(sub["severity"], sub["partial_nuisance"], marker="s",
                     color=palette["partial_nuisance"], label="Partial nuisance (Both Layers Fail)")
            ax4.set_title("Full nuisance vs Partial nuisance"); ax4.legend(); ax4.grid(True, linestyle="--", alpha=0.4)

            for ax in (ax1, ax2, ax3, ax4):
                ax.set_ylabel("Proportion"); ax.set_xticks([1,2,3,4,5])
            ax3.set_xlabel("ImageNet-C Severity"); ax4.set_xlabel("ImageNet-C Severity")

            fig.suptitle(f"{backbone} — {det} — Severity outcome comparisons (FPR={fpr_focus:.2f})", fontsize=14)
            fig.tight_layout(rect=[0,0,1,0.96])
            fname = f"severity_outcomes_pairs2x2_{backbone}_{det}_FPR{fpr_focus:.2f}.png".replace("/", "_")
            fig.savefig(joinp(save_dir, fname), dpi=220)
            plt.close(fig)
def plot_head2head(df: pd.DataFrame, out_root: str, fpr_focus: float = 0.05):
    """
    Per backbone: detector comparisons across severity at fixed FPR.
    Exports three PNGs per backbone: full, partial, total nuisance.
    """
    for backbone, bb_df in _per_backbone_loop(df):
        save_dir = joinp(out_root, "head2head", f"backbone={backbone}")
        ensure_dir(save_dir)

        met = aggregate_rates(bb_df, ["backbone","detector","fpr_target","severity"])
        met = met[(met["severity"].between(1,5)) & (np.isclose(met["fpr_target"], fpr_focus))]
        if met.empty:
            print(f"[WARN] No data for head-to-head at FPR={fpr_focus} (backbone={backbone}).")
            continue

        met["total_nuisance"] = met["full_nuisance"] + met["partial_nuisance"]
        detectors = sorted(met["detector"].dropna().unique())
        colors = plt.cm.tab10.colors
        cmap = {det: colors[i % len(colors)] for i, det in enumerate(detectors)}

        def _one(metric: str, title: str, fname_suffix: str, marker: str):
            fig, ax = plt.subplots(figsize=(10,6))
            for det in detectors:
                sub = met[met["detector"] == det].sort_values("severity")
                ax.plot(sub["severity"], sub[metric], marker=marker, color=cmap[det], label=det)
            ax.set_xlabel("ImageNet-C Severity")
            ax.set_ylabel("Proportion")
            ax.set_title(f"{backbone} — {title} (FPR={fpr_focus:.2f})")
            ax.set_xticks([1,2,3,4,5])
            ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(joinp(save_dir, f"head2head_{fname_suffix}_{backbone}_FPR{fpr_focus:.2f}.png"), dpi=220)
            plt.close(fig)

        _one("full_nuisance",   "Full nuisance (correct→reject)",   "full_nuisance",   "o")
        _one("partial_nuisance","Partial nuisance (incorrect→reject)","partial_nuisance","s")
        _one("total_nuisance",  "Total nuisance (full + partial)",   "total_nuisance",  "^")

def plot_corruption_heatmaps_combined(
    df: pd.DataFrame,
    out_root: str,
    top_n: int = 20,
    fpr_focus: float = 0.05,
    palette: str = "mako",
    vmax_mode: str = "q95",
    fixed_vmin: float = 0.0,
    fixed_vmax: float = 1.0,
):
    import matplotlib.ticker as mtick
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _pretty(name: str) -> str:
        return name.replace("_", " ").replace("jpeg", "JPEG").title()

    for backbone, bb_df in _per_backbone_loop(df):
        save_dir = joinp(out_root, "heatmaps", f"backbone={backbone}")
        ensure_dir(save_dir)

        met = aggregate_rates(bb_df, ["backbone","detector","fpr_target","corruption","severity"])
        met = met[(met["severity"].between(1,5)) & (np.isclose(met["fpr_target"], fpr_focus))]
        if met.empty:
            continue




        for det in sorted(met["detector"].dropna().unique()):
            sub = met[met["detector"] == det]
            if sub.empty:
                continue

            # --- sort per detector (by partial nuisance, for example) ---
            top = (sub.groupby("corruption")["partial_nuisance"]
                   .mean().sort_values(ascending=False).head(top_n).index)

            sub = sub[sub["corruption"].isin(top)].copy()
            sub["corruption_pretty"] = sub["corruption"].map(_pretty)
            order = [_pretty(c) for c in top]
            sub["corruption_pretty"] = pd.Categorical(
                sub["corruption_pretty"], order, ordered=True
            )

            p_full = sub.pivot_table(
                index="corruption_pretty", columns="severity",
                values="full_nuisance", aggfunc="mean", observed=False
            ).reindex(index=order, columns=[1, 2, 3, 4, 5])

            p_part = sub.pivot_table(
                index="corruption_pretty", columns="severity",
                values="partial_nuisance", aggfunc="mean", observed=False
            ).reindex(index=order, columns=[1, 2, 3, 4, 5])

            # scale
            if vmax_mode == "fixed":
                vmin, vmax = fixed_vmin, fixed_vmax
            else:
                vals = np.concatenate([p_full.values.ravel(), p_part.values.ravel()])
                vals = vals[~np.isnan(vals)]
                vmax = max(0.05, float(np.quantile(vals, 0.95))) if len(vals) else 1.0
                vmin = 0.0

            # -------- Layout: almost flush --------
            fig = plt.figure(figsize=(11, max(6, 0.40 * len(order))))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.0)  # truly no space

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)

            # kill ax2's left padding
            ax2.yaxis.set_visible(False)
            ax2.spines['left'].set_visible(False)

            # also kill ax1's right spine, so the heatmaps visually touch
            ax1.spines['right'].set_visible(False)

            # Shared colorbar on the right
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="6%", pad=0.05)

            # Full nuisance
            sns.heatmap(p_full, cmap=palette + "_r", vmin=vmin, vmax=vmax,
                        ax=ax1, cbar=False, square=True,
                        linewidths=0.5, linecolor="white")

            # Partial nuisance
            hm2 = sns.heatmap(p_part, cmap=palette + "_r", vmin=vmin, vmax=vmax,
                              ax=ax2, cbar=True, cbar_ax=cax,
                              cbar_kws={"label": "Percent of samples"},
                              square=True, linewidths=0.5, linecolor="white")

            # Cosmetics
            ax1.set_title("Full nuisance")
            ax1.set_xlabel("Severity")
            ax1.set_ylabel("Corruption")
            ax1.set_xticklabels([1, 2, 3, 4, 5], rotation=0)

            ax2.set_title("Partial nuisance")
            ax2.set_xlabel("Severity")
            ax2.set_xticklabels([1, 2, 3, 4, 5], rotation=0)

            # Global title
            fig.suptitle(f"{backbone} / {det} — Nuisance heatmaps (FPR={fpr_focus:.2f})", fontsize=14)

            # percent ticks on colorbar
            cb = hm2.collections[0].colorbar
            cb.formatter = mtick.PercentFormatter(1.0)
            cb.update_ticks()

            # tighten
            fig.subplots_adjust(wspace=0.0)

            # Save
            fname = f"heatmaps_{backbone}_{det}_FPR{fpr_focus:.2f}.png".replace("/", "_")
            fig.savefig(joinp(save_dir, fname), dpi=220, bbox_inches="tight")
            plt.close(fig)



def plot_outcome_composition(df: pd.DataFrame, out_root: str, fpr_focus: float = 0.05):
    import matplotlib.ticker as mtick

    color_map = {
        "Full_Correct":    "#006a00",   # green
        "Partial_Correct": "#47ac00",   # yellow-green
        "Partial_Nuisance":"#ff9d00",   # orange
        "Full_Nuisance":   "#ff0000",   # red
    }

    for backbone, bb_df in _per_backbone_loop(df):
        save_dir = joinp(out_root, "composition", f"backbone={backbone}")
        ensure_dir(save_dir)

        comp, outcome_cols = aggregate_outcomes(bb_df, ["backbone","detector","fpr_target","severity"])
        if comp.empty:
            continue

        comp = comp[(comp["severity"].between(1,5)) & (np.isclose(comp["fpr_target"], fpr_focus))]
        if comp.empty:
            continue

        order = [c for c in ["Full_Correct","Partial_Correct","Partial_Nuisance","Full_Nuisance"] if c in outcome_cols]
        prop_cols = [f"{c}_prop" for c in order]

        for det in sorted(comp["detector"].dropna().unique()):
            sub = comp[comp["detector"] == det].sort_values("severity")
            if sub.empty:
                continue

            x = sub["severity"].values
            bottoms = np.zeros(len(sub))
            fig, ax = plt.subplots(figsize=(9,6))

            for col, prop_col in zip(order, prop_cols):
                vals = sub[prop_col].values
                ax.bar(x, vals, bottom=bottoms, label=col, color=color_map[col])
                bottoms += vals

            ax.set_xlim(0.5, 5.5)
            ax.set_xticks([1,2,3,4,5])
            ax.set_xlabel("Severity")
            ax.set_ylabel("Proportion of outcomes")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_title(f"{backbone} / {det} — Outcome composition (FPR={fpr_focus:.2f})")
            ax.grid(True, linestyle="--", alpha=0.25, axis="y")

            # Legend under plot
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.08),
                ncol=len(order), frameon=False, title="Outcome"
            )

            fig.tight_layout(rect=[0,0.05,1,1])
            fname = f"composition_{backbone}_{det}_FPR{fpr_focus:.2f}.png".replace("/", "_")
            fig.savefig(joinp(save_dir, fname), dpi=220)
            plt.close(fig)
def plot_head2head_backbones(df: pd.DataFrame, out_root: str, fpr_focus: float = 0.05):
    """
    Backbone head-to-head at fixed detector and FPR.
    For each detector, compare multiple backbones across severity on:
      - Full nuisance (correct→reject)
      - Partial nuisance (incorrect→reject)
      - Total nuisance (full + partial)
      - (Optional) Ideal accept (correct→accept)
    """
    save_root = os.path.join(out_root, "head2head_backbones")
    ensure_dir(save_root)

    met = aggregate_rates(df, ["backbone", "detector", "fpr_target", "severity"])
    met = met[(met["severity"].between(1, 5)) & (np.isclose(met["fpr_target"], fpr_focus))]
    if met.empty:
        print(f"[WARN] No data for backbone head-to-head at FPR={fpr_focus}.")
        return

    met["total_nuisance"] = met["full_nuisance"] + met["partial_nuisance"]

    detectors = sorted(met["detector"].dropna().unique())
    for det in detectors:
        sub = met[met["detector"] == det].copy()
        if sub.empty:
            continue

        save_dir = os.path.join(save_root, f"detector={det}")
        ensure_dir(save_dir)

        backbones = sorted(sub["backbone"].dropna().unique())
        colors = plt.cm.tab10.colors
        cmap = {bb: colors[i % len(colors)] for i, bb in enumerate(backbones)}

        def _one(metric: str, title: str, fname_suffix: str, marker: str):
            fig, ax = plt.subplots(figsize=(10, 6))
            for bb in backbones:
                sbb = sub[sub["backbone"] == bb].sort_values("severity")
                if sbb.empty:  # skip if detector not run for this backbone
                    continue
                ax.plot(sbb["severity"], sbb[metric], marker=marker,
                        color=cmap[bb], label=bb)
            ax.set_xlabel("ImageNet-C Severity")
            ax.set_ylabel("Proportion")
            ax.set_title(f"{det} — {title} (FPR={fpr_focus:.2f})")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Backbone")
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{fname_suffix}_{det}_FPR{fpr_focus:.2f}.png"), dpi=220)
            plt.close(fig)

        _one("full_nuisance",    "Full nuisance (Worst Case)",      "bb_h2h_full",   "o")
        _one("partial_nuisance", "Partial nuisance (Both parts fail)", "bb_h2h_partial","s")
        _one("total_nuisance",   "Total nuisance (full + partial)",     "bb_h2h_total",  "^")
        # Optional but useful:
        _one("full_correct",     "Full Correct (Ideal Case)",       "bb_h2h_ideal",  "D")

def plot_classifier_accuracy(df: pd.DataFrame, out_root: str, fpr_focus: float = 0.05):
    """
    Classifier accuracy vs severity per backbone at fixed FPR.
    (Accuracy is detector-independent, but we filter for consistency.)
    """
    for backbone, bb_df in _per_backbone_loop(df):
        save_dir = joinp(out_root, "accuracy", f"backbone={backbone}")
        ensure_dir(save_dir)

        acc_df = (bb_df.groupby(["backbone","detector","fpr_target","severity"])
                        .agg(acc=("correct_cls","mean"), n=("correct_cls","size")).reset_index())
        acc_df = acc_df[(acc_df["severity"].between(1,5)) & (np.isclose(acc_df["fpr_target"], fpr_focus))]
        if acc_df.empty:
            print(f"[WARN] No accuracy data at FPR={fpr_focus} (backbone={backbone}).")
            continue

        fig, ax = plt.subplots(figsize=(9,6))
        for det in sorted(acc_df["detector"].dropna().unique()):
            sub = acc_df[acc_df["detector"] == det].sort_values("severity")
            ax.plot(sub["severity"], sub["acc"], marker="o", label=det)
        ax.set_xlabel("ImageNet-C Severity"); ax.set_ylabel("Classifier Accuracy")
        ax.set_title(f"{backbone} — Classifier Accuracy vs Severity (FPR={fpr_focus:.2f})")
        ax.set_xticks([1,2,3,4,5]); ax.legend(title="Detector"); ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(joinp(save_dir, f"accuracy_{backbone}_FPR{fpr_focus:.2f}.png"), dpi=200)
        plt.close(fig)

        acc_df.to_csv(joinp(save_dir, f"accuracy_summary_{backbone}_FPR{fpr_focus:.2f}.csv"), index=False)

# -----------------------
# Main
# -----------------------
def main(csv_path: str, out_dir: str, fpr_focus: float):
    ensure_dir(out_dir)
    df = load_with_enrichment(csv_path)

    needed = {"image_path","accept","detector","fpr_target","correct_cls"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Plots (per-backbone)
    plot_severity_pairs_2x2(df, out_dir, fpr_focus=fpr_focus)
    plot_head2head(df, out_dir, fpr_focus=fpr_focus)
    plot_corruption_heatmaps_combined(df, out_dir, top_n=20, fpr_focus=fpr_focus)
    plot_outcome_composition(df, out_dir, fpr_focus=fpr_focus)
    plot_classifier_accuracy(df, out_dir, fpr_focus=fpr_focus)
    plot_head2head_backbones(df, out_dir, fpr_focus=fpr_focus)

    # Export summary CSV (per backbone × detector × fpr_target × severity)
    sev_summary = aggregate_rates(df, ["backbone","detector","fpr_target","severity"])
    sev_dir = joinp(out_dir, "summaries"); ensure_dir(sev_dir)
    sev_summary.to_csv(joinp(sev_dir, "severity_summary_absolute.csv"), index=False)

    print(f"[DONE] Saved plots and summaries under: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default="results/nuisance_runs/nuisance_full_set.csv", type=str, help="CSV from nuisance ImageNet-C runs (can include multiple backbones)")
    ap.add_argument("--out_dir", type=str, default="results/analysis", help="Output directory")
    ap.add_argument("--fpr", type=float, default=0.05, help="FPR to focus plots on (e.g., 0.05)")
    args = ap.parse_args()

    main(args.csv, args.out_dir, args.fpr)
