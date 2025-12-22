import os
import numpy as np
import pandas as pd
from analysis.processing import compute_adr  # Import the new function


def _format_float(x):
    if pd.isna(x):
        return "-"
    return f"{x:.3f}"


def generate_detector_leaderboard(agg_df: pd.DataFrame, backbone: str, dataset: str, out_dir: str):
    """
    Detector leaderboard including ADR.
    """
    subset = agg_df[(agg_df["backbone"] == backbone) & (agg_df["dataset"] == dataset)].copy()
    if subset.empty:
        return

    # 1. Standard Aggregates (Levels > 0)
    corrupted = subset[subset["level"] > 0].copy()

    per_det = (
        corrupted.groupby(["detector"])[["Accuracy", "OSA_ID", "CNR", "Rejection_Rate", "NNR"]]
        .mean()
        .reset_index()
        .rename(columns={
            "Accuracy": "Mean_ID_Acc",
            "OSA_ID": "Mean_OSA_ID",
            "CNR": "Mean_CNR",
            "Rejection_Rate": "Mean_Rej_Rate",
            "NNR": "Mean_NNR",
        })
    )

    # 2. Compute ADR (Uses all levels 0-5 for slope)
    adr_df = compute_adr(subset)
    # Merge ADR: Note compute_adr returns keys [backbone, detector, dataset]
    # We only need 'detector' and 'ADR' here since we already filtered by backbone/dataset
    adr_merge = adr_df[["detector", "ADR"]]

    final = per_det.merge(adr_merge, on="detector", how="left")

    # 3. Per-level CNR
    lvl = (
        corrupted.groupby(["detector", "level"])[["CNR"]]
        .mean()
        .reset_index()
        .pivot(index="detector", columns="level", values="CNR")
        .reset_index()
    )
    for L in [1, 3, 5]:
        col_name = f"CNR_L{L}"
        if L in lvl.columns:
            lvl = lvl.rename(columns={L: col_name})
        else:
            lvl[col_name] = np.nan

    lvl = lvl[["detector"] + [c for c in lvl.columns if "CNR_L" in c]]
    final = final.merge(lvl, on="detector", how="left")

    # Formatting
    cols_to_format = [
        "Mean_ID_Acc", "Mean_OSA_ID", "Mean_CNR", "Mean_NNR",
        "Mean_Rej_Rate", "ADR", "CNR_L1", "CNR_L3", "CNR_L5"
    ]

    for c in cols_to_format:
        if c in final.columns:
            final[c] = final[c].apply(_format_float)

    # Reorder columns for readability
    head = ["detector", "ADR", "Mean_CNR", "Mean_ID_Acc", "Mean_OSA_ID"]
    tail = [c for c in final.columns if c not in head]
    final = final[head + tail]

    fname = os.path.join(out_dir, f"leaderboard_{backbone}_{dataset}.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")


def generate_master_table(agg_df: pd.DataFrame, out_dir: str):
    """
    Master table with ADR included.
    """
    df = agg_df.copy()
    corrupted = df[df["level"] > 0].copy()

    # 1. Compute Mean Metrics
    metrics = (
        corrupted.groupby(["dataset", "backbone", "detector"])[
            ["Accuracy", "OSA_ID", "OSA_Gap", "CNR", "NNR", "Rejection_Rate"]
        ]
        .mean()
        .reset_index()
        .rename(columns={
            "Accuracy": "Mean_ID_Acc",
            "OSA_ID": "Mean_OSA_ID",
            "OSA_Gap": "Mean_OSA_Gap",
            "CNR": "Mean_CNR",
            "NNR": "Mean_NNR",
            "Rejection_Rate": "Mean_Rej_Rate",
        })
    )

    # 2. Compute ADR
    adr_df = compute_adr(df)  # Pass full df to access all levels

    # Merge
    final = metrics.merge(adr_df, on=["dataset", "backbone", "detector"], how="left")

    cols_to_format = [
        "Mean_ID_Acc", "Mean_OSA_ID", "Mean_OSA_Gap",
        "Mean_CNR", "Mean_NNR", "Mean_Rej_Rate", "ADR"
    ]

    for c in cols_to_format:
        final[c] = final[c].apply(_format_float)

    # Sort
    final = final.sort_values(["dataset", "backbone", "detector"])

    fname = os.path.join(out_dir, "master_table_per_dataset.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")