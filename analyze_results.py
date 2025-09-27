##!/usr/bin/env python3
import argparse, os
from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Utils
# -----------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def parse_from_path(path: str):
    """
    Parse corruption type and severity from an ImageNet-C filepath.
    Works for paths like .../imagenet_c/<corruption>/<severity>/file.jpeg
    """
    parts = Path(path).parts
    idxs = [i for i, p in enumerate(parts) if str(p).lower().startswith("imagenet_c")]
    if not idxs:
        return "unknown", -1
    idx = idxs[0]
    try:
        corruption = parts[idx+1]
        severity = int(parts[idx+2])
        return str(corrosion), int(severity)
    except Exception:
        return "unknown", -1

def add_corruption_severity(df: pd.DataFrame) -> pd.DataFrame:
    need_corr = "corruption" not in df.columns
    need_sev = "severity" not in df.columns
    if need_corr or need_sev:
        corr_sev = df["image_path"].apply(lambda p: pd.Series(parse_from_path(p), index=["corruption","severity"]))
        for col in ["corruption","severity"]:
            if col not in df.columns:
                df[col] = corr_sev[col]
    return df

def load_with_enrichment(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only imagenet-c rows if mixed CSVs
    if "dataset" in df.columns:
        mask_c = df["dataset"].str.lower().str.contains("imagenet") & df["image_path"].str.lower().str.contains("imagenet_c")
        if mask_c.any():
            df = df[mask_c].copy()
    df = add_corruption_severity(df)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)
    return df

# -----------------------
# Metrics (NEW definition)
# -----------------------
def aggregate_rates(df: pd.DataFrame, group_cols):
    """
    Aggregate using the new definition:
      nuisance rate = all rejections = 1 - mean(accept)
    Also computes Wilson CIs on rejection proportion.
    """
    tmp = df.groupby(group_cols).agg(
        n=("accept", "size"),
        reject_sum=("accept", lambda s: int(np.sum(1 - s.astype(int)))),
        acc=("correct_cls", "mean"),              # still useful to print
        rej_rate=("accept", lambda s: 1.0 - float(np.mean(s))),  # same as nuisance rate under new def
    ).reset_index()

    # proportions + CIs for reject (nuisance) rate
    lowers, uppers = [], []
    rates = tmp["reject_sum"] / tmp["n"].clip(lower=1)
    for k, n in zip(tmp["reject_sum"], tmp["n"]):
        lo, hi = wilson_ci(k, n)
        lowers.append(lo); uppers.append(hi)
    tmp["nn_rate"] = rates          # <- keep column name nn_rate for downstream plots
    tmp["nn_lo"] = lowers
    tmp["nn_hi"] = uppers
    return tmp

def aggregate_outcomes(df: pd.DataFrame, group_cols):
    # Keep outcome composition (can still be insightful)
    one_hot = pd.get_dummies(df["error_type"])
    tmp = pd.concat([df[group_cols], one_hot], axis=1)
    agg = tmp.groupby(group_cols).sum().reset_index()
    outcome_cols = [c for c in agg.columns if c not in group_cols]
    agg["total"] = agg[outcome_cols].sum(axis=1).clip(lower=1)
    for c in outcome_cols:
        agg[c + "_prop"] = agg[c] / agg["total"]
    return agg, outcome_cols

# -----------------------
# Plotters (all use nn_rate = reject rate)
# -----------------------
def plot_severity_curves(df: pd.DataFrame, out_dir: str):
    met = aggregate_rates(df, ["detector", "fpr_target", "severity"])
    met = met[(met["severity"] >= 1) & (met["severity"] <= 5)]
    if met.empty:
        print("[WARN] No valid severities in 1..5 for plotting.")
        return
    for det in sorted(met["detector"].unique()):
        fig, ax = plt.subplots(figsize=(9,6))
        for fpr in sorted(met["fpr_target"].unique()):
            sub = met[(met["detector"] == det) & (met["fpr_target"] == fpr)].sort_values("severity")
            ax.plot(sub["severity"], sub["nn_rate"], marker="o", label=f"FPR={fpr:.2f}")
            ax.fill_between(sub["severity"], sub["nn_lo"], sub["nn_hi"], alpha=0.2)
        ax.set_xlabel("ImageNet-C Severity")
        ax.set_ylabel("Nuisance Rate (reject rate)")
        ax.set_title(f"Nuisance vs Severity — {det}")
        ax.set_xticks([1,2,3,4,5])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Target ID FPR")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"severity_curves_{det}.png"), dpi=200)
        plt.close(fig)

def plot_corruption_heatmaps(df: pd.DataFrame, out_dir: str, top_n: int = 20):
    met = aggregate_rates(df, ["detector","fpr_target","corruption","severity"])
    met = met[(met["severity"] >= 1) & (met["severity"] <= 5)]
    if met.empty:
        print("[WARN] No valid corruption/severity combos for heatmap.")
        return
    for det in sorted(met["detector"].unique()):
        for fpr in sorted(met["fpr_target"].unique()):
            sub = met[(met["detector"] == det) & (met["fpr_target"] == fpr)].copy()
            order = sub.groupby("corruption")["nn_rate"].mean().sort_values(ascending=False).head(top_n).index
            sub = sub[sub["corruption"].isin(order)]
            sub["corruption"] = pd.Categorical(sub["corruption"], categories=order, ordered=True)
            pivot = sub.pivot_table(index="corruption", columns="severity", values="nn_rate", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(12, max(6, 0.35*len(order))))
            sns.heatmap(pivot, annot=False, cmap="Reds", cbar_kws={"label":"Nuisance rate (reject)"}, ax=ax)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Corruption")
            ax.set_title(f"Nuisance Heatmap — {det}, FPR={fpr:.2f}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"heatmap_{det}_fpr{fpr:.2f}.png"), dpi=200)
            plt.close(fig)

def plot_topk_corruptions(df: pd.DataFrame, out_dir: str, k: int = 10, severities=(1,5)):
    met = aggregate_rates(df, ["detector","fpr_target","corruption","severity"])
    met = met[met["severity"].isin(severities)]
    if met.empty:
        print("[WARN] No data for requested severities in top-k plot.")
        return
    for det in sorted(met["detector"].unique()):
        for fpr in sorted(met["fpr_target"].unique()):
            sub = met[(met["detector"] == det) & (met["fpr_target"] == fpr)]
            fig, axes = plt.subplots(1, len(severities), figsize=(14, 5), sharey=True)
            if len(severities) == 1:
                axes = [axes]
            for ax, sv in zip(axes, severities):
                ss = sub[sub["severity"] == sv].copy().sort_values("nn_rate", ascending=False).head(k)
                ax.barh(ss["corruption"], ss["nn_rate"])
                ax.invert_yaxis()
                ax.set_title(f"Severity {sv}")
                ax.set_xlabel("Nuisance rate (reject)")
            fig.suptitle(f"Top-{k} nuisance corruptions — {det}, FPR={fpr:.2f}")
            fig.tight_layout(rect=[0,0,1,0.95])
            fig.savefig(os.path.join(out_dir, f"top{k}_{det}_fpr{fpr:.2f}.png"), dpi=200)
            plt.close(fig)

def plot_outcome_composition(df: pd.DataFrame, out_dir: str):
    """
    Optional diagnostic (unchanged): stacked outcome proportions vs severity.
    Not used for nuisance rate definition, but helpful context.
    """
    grp_cols = ["detector","fpr_target","severity"]
    # Guard if error_type is missing
    if "error_type" not in df.columns:
        print("[WARN] 'error_type' not in CSV; skipping composition plot.")
        return
    comp, outcome_cols = aggregate_outcomes(df, grp_cols)
    comp = comp[(comp["severity"] >= 1) & (comp["severity"] <= 5)]
    if comp.empty:
        print("[WARN] No outcome data for composition plot.")
        return
    out_props = [c for c in comp.columns if c.endswith("_prop")]
    labels = [c.replace("_prop","") for c in out_props]
    for det in sorted(comp["detector"].unique()):
        for fpr in sorted(comp["fpr_target"].unique()):
            sub = comp[(comp["detector"] == det) & (comp["fpr_target"] == fpr)].sort_values("severity")
            fig, ax = plt.subplots(figsize=(9,6))
            bottoms = np.zeros(len(sub))
            x = sub["severity"].values
            for lbl, col in zip(labels, out_props):
                vals = sub[col].values
                ax.bar(x, vals, bottom=bottoms, label=lbl)
                bottoms += vals
            ax.set_xticks([1,2,3,4,5])
            ax.set_xlabel("Severity")
            ax.set_ylabel("Proportion")
            ax.set_title(f"Outcome composition vs severity — {det}, FPR={fpr:.2f}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"composition_{det}_fpr{fpr:.2f}.png"), dpi=200)
            plt.close(fig)

def plot_head2head(df: pd.DataFrame, out_dir: str, fpr_focus: float = 0.05, severity_focus: int = 5):
    """
    Direct detector comparisons under the new definition.
    """
    # Combined severity curve at a fixed FPR
    met = aggregate_rates(df, ["detector", "fpr_target", "severity"])
    met = met[(met["severity"].between(1,5)) & (np.isclose(met["fpr_target"], fpr_focus))]
    if not met.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        for det in sorted(met["detector"].unique()):
            sub = met[met["detector"] == det].sort_values("severity")
            ax.plot(sub["severity"], sub["nn_rate"], marker="o", label=det)
            ax.fill_between(sub["severity"], sub["nn_lo"], sub["nn_hi"], alpha=0.15)
        ax.set_xlabel("ImageNet-C Severity")
        ax.set_ylabel("Nuisance Rate (reject)")
        ax.set_title(f"Head-to-Head (FPR={fpr_focus:.2f})")
        ax.set_xticks([1,2,3,4,5])
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"head2head_severity_fpr{fpr_focus:.2f}.png"), dpi=200)
        plt.close(fig)

    # Bar chart at a fixed severity & FPR
    met2 = aggregate_rates(df, ["detector", "fpr_target", "severity"])
    met2 = met2[(met2["severity"] == severity_focus) & (np.isclose(met2["fpr_target"], fpr_focus))]
    if not met2.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(data=met2.sort_values("nn_rate", ascending=False), x="detector", y="nn_rate", ax=ax)
        ax.set_ylabel("Nuisance Rate (reject)")
        ax.set_xlabel("Detector")
        ax.set_title(f"Head-to-Head at Severity {severity_focus}, FPR={fpr_focus:.2f}")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"head2head_bar_s{severity_focus}_fpr{fpr_focus:.2f}.png"), dpi=200)
        plt.close(fig)

# -----------------------
# Main
# -----------------------
def main(csv_path: str, out_dir: str):
    ensure_dir(out_dir)
    df = load_with_enrichment(csv_path)

    # Basic sanity
    needed = {"image_path","accept","detector","fpr_target"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Plots under NEW definition (nuisance = reject)
    plot_severity_curves(df, out_dir)
    plot_corruption_heatmaps(df, out_dir, top_n=20)
    plot_topk_corruptions(df, out_dir, k=12, severities=(1,5))
    plot_outcome_composition(df, out_dir)  # optional diagnostic
    plot_head2head(df, out_dir, fpr_focus=0.05, severity_focus=5)

    # Also export severity summary CSV
    sev_summary = aggregate_rates(df, ["detector","fpr_target","severity"])
    sev_summary.to_csv(os.path.join(out_dir, "severity_summary_newdef.csv"), index=False)

    print(f"[DONE] Saved figures + severity_summary_newdef.csv to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="CSV from nuisance ImageNet-C run")
    ap.add_argument("--out_dir", type=str, default="results/analysis", help="Output directory")
    args = ap.parse_args()
    main("results/nuisance_runs/nuisance_full_set.csv", args.out_dir)
