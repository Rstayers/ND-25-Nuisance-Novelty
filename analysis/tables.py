import os
import numpy as np
import pandas as pd


def _format_float(x):
    if pd.isna(x):
        return "-"
    return f"{x:.3f}"


def generate_detector_leaderboard(agg_df: pd.DataFrame, backbone: str, dataset: str, out_dir: str):
    """
    Detector leaderboard for one (backbone, dataset).

    Reports (macro over levels 1-5, and macro over nuisances):
      - Mean_CNR
      - Mean_NNR (unconditional nuisance novelty rate)
      - Mean_OSA_ID
      - Mean_ID_Acc
      - Mean_Rej_Rate
      - plus per-level CNR at L1/L3/L5
    """
    subset = agg_df[(agg_df["backbone"] == backbone) & (agg_df["dataset"] == dataset)].copy()
    if subset.empty:
        return

    # Focus on corrupted levels only
    subset = subset[subset["level"] > 0].copy()
    if subset.empty:
        return

    # Macro average over (nuisance, level) conditions
    per_det = (
        subset.groupby(["detector"])[["Accuracy", "OSA_ID", "CNR", "Rejection_Rate", "NNR", "CM_Rate", "DF_Rate"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Accuracy": "Mean_ID_Acc",
                "OSA_ID": "Mean_OSA_ID",
                "CNR": "Mean_CNR",
                "Rejection_Rate": "Mean_Rej_Rate",
                "NNR": "Mean_NNR",
                "CM_Rate": "Mean_CM_Rate",
                "DF_Rate": "Mean_DF_Rate",
            }
        )
    )

    # Per-level CNR (macro over nuisances at that level)
    lvl = (
        subset.groupby(["detector", "level"])[["CNR"]]
        .mean()
        .reset_index()
        .pivot(index="detector", columns="level", values="CNR")
        .reset_index()
    )
    for L in [1, 3, 5]:
        if L in lvl.columns:
            lvl = lvl.rename(columns={L: f"CNR_L{L}"})
        else:
            lvl[f"CNR_L{L}"] = np.nan
    lvl = lvl[["detector", "CNR_L1", "CNR_L3", "CNR_L5"]]

    final = per_det.merge(lvl, on="detector", how="left")

    # Formatting
    for c in [
        "Mean_ID_Acc",
        "Mean_OSA_ID",
        "Mean_CNR",
        "Mean_NNR",
        "Mean_Rej_Rate",
        "Mean_CM_Rate",
        "Mean_DF_Rate",
        "CNR_L1",
        "CNR_L3",
        "CNR_L5",
    ]:
        if c in final.columns:
            final[c] = final[c].apply(_format_float)

    fname = os.path.join(out_dir, f"leaderboard_{backbone}_{dataset}.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")


def generate_master_table(agg_df: pd.DataFrame, out_dir: str):
    """
    Master table PER DATASET (ID-only nuisance evaluation).
    Uses MACRO averaging over (nuisance, level) with level>0.

    Columns reported:
      Mean_ID_Acc
      Mean_OSA_ID
      Mean_OSA_Gap
      Mean_CNR
      Mean_NNR
      Mean_Rej_Rate
      Mean_CM_Rate
      Mean_DF_Rate

    NOTE: No OOD AUROC/FPR95 merges. Those were inconsistent with your CSV
    assumptions and your benchmark goal (ID-only nuisance novelty).
    """
    df = agg_df.copy()
    corrupted = df[df["level"] > 0].copy()
    if corrupted.empty:
        # Fallback if no level annotation: use whatever is present
        corrupted = df.copy()

    metrics = (
        corrupted.groupby(["dataset", "backbone", "detector"])[
            ["Accuracy", "OSA_ID", "OSA_Gap", "CNR", "NNR", "Rejection_Rate", "CM_Rate", "DF_Rate"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Accuracy": "Mean_ID_Acc",
                "OSA_ID": "Mean_OSA_ID",
                "OSA_Gap": "Mean_OSA_Gap",
                "CNR": "Mean_CNR",
                "NNR": "Mean_NNR",
                "Rejection_Rate": "Mean_Rej_Rate",
                "CM_Rate": "Mean_CM_Rate",
                "DF_Rate": "Mean_DF_Rate",
            }
        )
        .sort_values(["dataset", "backbone", "detector"])
    )

    # Format for report CSV
    for c in [
        "Mean_ID_Acc",
        "Mean_OSA_ID",
        "Mean_OSA_Gap",
        "Mean_CNR",
        "Mean_NNR",
        "Mean_Rej_Rate",
        "Mean_CM_Rate",
        "Mean_DF_Rate",
    ]:
        metrics[c] = metrics[c].apply(_format_float)

    fname = os.path.join(out_dir, "master_table_per_dataset.csv")
    metrics.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")
