#!/usr/bin/env python3
"""
metrics.py
-----------
Compute nuisance novelty robustness metrics for head-to-head detector evaluation.
"""

import argparse, os
import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_from_path(path: str):
    """
    Parse corruption type and severity from an ImageNet-C filepath.
    Example: .../imagenet_c/gaussian_noise/3/xxx.jpeg → ("gaussian_noise", 3)
    """
    parts = Path(path).parts
    try:
        idx = [i for i, p in enumerate(parts) if p.lower().startswith("imagenet_c")][0]
        corruption = parts[idx+1]
        severity = int(parts[idx+2])
        return corruption, severity
    except Exception:
        return "unknown", -1


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # filter ImageNet-C rows if mixed
    if "dataset" in df.columns:
        mask = df["image_path"].str.lower().str.contains("imagenet_c")
        if mask.any():
            df = df[mask].copy()
    if ("corruption" not in df.columns) or ("severity" not in df.columns):
        corr_sev = df["image_path"].apply(
            lambda p: pd.Series(parse_from_path(p), index=["corruption","severity"])
        )
        df["corruption"] = corr_sev["corruption"]
        df["severity"] = corr_sev["severity"]
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(-1).astype(int)
    df["accept"] = df["accept"].astype(int)
    df["correct_cls"] = df.get("correct_cls", 1).astype(int)
    return df


# -------------------------
# Metrics
# -------------------------
def compute_nr_at_fpr(df: pd.DataFrame) -> pd.DataFrame:
    """NR@FPR: mean accept per detector × fpr_target"""
    return df.groupby(["detector","fpr_target"]).agg(
        n=("accept","size"),
        nr=("accept","mean"),
    ).reset_index()


def compute_nr_auc(df: pd.DataFrame) -> pd.DataFrame:
    """NR-AUC: area under severity curve per detector × fpr_target"""
    sub = df[(df["severity"]>=1) & (df["severity"]<=5)]
    if sub.empty: return pd.DataFrame()
    met = sub.groupby(["detector","fpr_target","corruption","severity"])["accept"].mean().reset_index()
    out = []
    for (det,fpr,corr), grp in met.groupby(["detector","fpr_target","corruption"]):
        grp = grp.sort_values("severity")
        sev = grp["severity"].to_numpy()
        acc = grp["accept"].to_numpy()
        full_sev = np.arange(1,6)
        full_acc = np.interp(full_sev, sev, acc)
        auc = np.trapz(full_acc, full_sev) / 4.0  # normalize to [0,1]
        out.append((det,fpr,corr,auc))
    auc_df = pd.DataFrame(out, columns=["detector","fpr_target","corruption","nr_auc"])
    return auc_df.groupby(["detector","fpr_target"])["nr_auc"].mean().reset_index()


def compute_nr_worstk(df: pd.DataFrame, k_pct: float = 0.2) -> pd.DataFrame:
    """NR-Worst-K: mean accept of worst k% corruptions per detector × fpr_target"""
    sub = df[(df["severity"]>=1) & (df["severity"]<=5)]
    if sub.empty: return pd.DataFrame()
    corr_mean = sub.groupby(["detector","fpr_target","corruption"])["accept"].mean().reset_index()
    out = []
    for (det,fpr), grp in corr_mean.groupby(["detector","fpr_target"]):
        k = max(1, int(np.ceil(len(grp)*k_pct)))
        worst = grp.sort_values("accept").head(k)["accept"].mean()
        out.append((det,fpr,k_pct,worst))
    return pd.DataFrame(out, columns=["detector","fpr_target","k_pct","nr_worstk"])


def compute_task_aware(df: pd.DataFrame) -> pd.DataFrame:
    """Task-aware: usability (accept&correct) and safety (reject&incorrect)."""
    tmp = df.copy()
    tmp["accept_correct"] = tmp["accept"] * tmp["correct_cls"]
    tmp["reject_incorrect"] = (1 - tmp["accept"]) * (1 - tmp["correct_cls"])
    return tmp.groupby(["detector","fpr_target"]).agg(
        n=("accept","size"),
        ta_nr=("accept_correct","mean"),
        safety_catch=("reject_incorrect","mean"),
    ).reset_index()


# -------------------------
# Main
# -------------------------
def main(csv: str, out_dir: str):
    ensure_dir(out_dir)
    df = enrich(pd.read_csv(csv))

    nr = compute_nr_at_fpr(df)
    nr_auc = compute_nr_auc(df)
    nr_worst = compute_nr_worstk(df, k_pct=0.2)
    ta = compute_task_aware(df)

    nr.to_csv(os.path.join(out_dir, "NR_at_FPR.csv"), index=False)
    if not nr_auc.empty: nr_auc.to_csv(os.path.join(out_dir, "NR_AUC.csv"), index=False)
    if not nr_worst.empty: nr_worst.to_csv(os.path.join(out_dir, "NR_Worst20.csv"), index=False)
    ta.to_csv(os.path.join(out_dir, "TaskAware.csv"), index=False)

    print("\n[NR@FPR] mean accept per detector")
    print(nr.to_string(index=False))
    if not nr_auc.empty:
        print("\n[NR-AUC] area under severity curve")
        print(nr_auc.to_string(index=False))
    if not nr_worst.empty:
        print("\n[NR-Worst-20%] worst-case robustness")
        print(nr_worst.to_string(index=False))
    print("\n[Task-aware Usability vs Safety]")
    print(ta.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV from nuisance evaluation")
    ap.add_argument("--out_dir", default="results/metrics", help="Output dir")
    args = ap.parse_args()
    main(args.csv, args.out_dir)
