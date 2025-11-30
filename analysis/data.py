# analysis/compute.py
"""
Core computation utilities for Psycho Benchmark analysis.
Includes aggregation of rates, outcomes, and accuracy summaries.
"""

import pandas as pd
import numpy as np


# -----------------------------
# Rate and outcome aggregation
# -----------------------------
def aggregate_rates(df: pd.DataFrame, group_cols):
    """
    Aggregate proportions for the four classifier × detector combinations:
      - full_correct   : correct_cls == 1 & accept == 1
      - full_nuisance  : correct_cls == 1 & accept == 0
      - partial_nuisance: correct_cls == 0 & accept == 0
      - partial_correct : correct_cls == 0 & accept == 1
    """
    df = df.copy()
    grouped = df.groupby(group_cols, dropna=False)
    rows = []

    for keys, sub in grouped:
        n = len(sub)
        if n == 0:
            continue

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": n,
            "full_correct": ((sub["correct_cls"] == 1) & (sub["accept"] == 1)).mean(),
            "full_nuisance": ((sub["correct_cls"] == 1) & (sub["accept"] == 0)).mean(),
            "partial_nuisance": ((sub["correct_cls"] == 0) & (sub["accept"] == 0)).mean(),
            "partial_correct": ((sub["correct_cls"] == 0) & (sub["accept"] == 1)).mean(),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_outcomes(df: pd.DataFrame, group_cols):
    """
    Aggregate system_outcome column into proportions per group.
    """
    if "system_outcome" not in df.columns:
        return pd.DataFrame(), []

    one_hot = pd.get_dummies(df["system_outcome"])
    tmp = pd.concat([df[group_cols], one_hot], axis=1)
    agg = tmp.groupby(group_cols, dropna=False).sum(numeric_only=True).reset_index()

    outcome_cols = [c for c in agg.columns if c not in group_cols]
    agg["total"] = agg[outcome_cols].sum(axis=1).clip(lower=1)
    for c in outcome_cols:
        agg[c + "_prop"] = agg[c] / agg["total"]

    return agg, outcome_cols


def per_backbone(df: pd.DataFrame):
    """
    Generator yielding (backbone_name, subset_df) pairs.
    """
    for bb in sorted(df["backbone"].dropna().unique()):
        yield bb, df[df["backbone"] == bb].copy()


# -----------------------------
# CNS accuracy summary helper
# -----------------------------
def summarize_cns_accuracy(df: pd.DataFrame, fpr_focus: float = 0.05):
    """
    Compute classifier and detector accuracies across severity for CNS-Bench.

    Returns a DataFrame with:
      backbone | detector | severity | cls_acc | det_acc | n
      + one aggregated 'avg' severity row per (backbone, detector)
    """
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.contains("cns", case=False, na=False)].copy()

    if df.empty:
        print("[WARN] No CNS-Bench data available for accuracy summary.")
        return pd.DataFrame()

    if "fpr_target" in df.columns:
        df = df[np.isclose(df["fpr_target"], fpr_focus)]
        if df.empty:
            print(f"[WARN] No rows at FPR={fpr_focus}.")
            return pd.DataFrame()

    group_cols = ["backbone", "detector", "severity"]
    acc = (
        df.groupby(group_cols, dropna=False)
          .agg(
              cls_acc=("correct_cls", "mean"),
              det_acc=("accept", "mean"),
              n=("correct_cls", "size"),
          )
          .reset_index()
    )

    # Average across severities
    avg = (
        acc.groupby(["backbone", "detector"], dropna=False)
           .agg(
               cls_acc=("cls_acc", "mean"),
               det_acc=("det_acc", "mean"),
               n=("n", "sum"),
           )
           .reset_index()
    )
    avg["severity"] = "avg"

    full = pd.concat([acc, avg], ignore_index=True)

    # Sort by numeric severity if possible
    def _sev_key(x):
        try:
            return (0, float(x))
        except ValueError:
            return (1, float("inf"))

    full["severity_order"] = full["severity"].apply(_sev_key)
    full = full.sort_values(["backbone", "detector", "severity_order"]).drop(columns="severity_order")

    return full
