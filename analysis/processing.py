import pandas as pd
import numpy as np

REQUIRED_OUTCOMES = [
    "Clean_Success",
    "Nuisance_Novelty",
    "Double_Failure",
    "Contained_Misidentification",
]


def load_and_prep(csv_path: str):
    """
    Loads the benchmark CSV. Does not assume any particular dataset naming.
    """
    df = pd.read_csv(csv_path)

    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str)

    needed = {"backbone", "detector", "dataset", "nuisance", "level", "outcome"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(list(missing))}")

    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(0).astype(int)

    return df, REQUIRED_OUTCOMES


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates row-level data and calculates key ID-only nuisance metrics.
    """
    group_cols = ["backbone", "detector", "dataset", "nuisance", "level", "outcome"]
    agg = df.groupby(group_cols).size().unstack(fill_value=0).reset_index()

    # Ensure outcome columns exist
    for col in REQUIRED_OUTCOMES:
        if col not in agg.columns:
            agg[col] = 0

    CS = agg["Clean_Success"].astype(float)
    NN = agg["Nuisance_Novelty"].astype(float)
    DF = agg["Double_Failure"].astype(float)
    CM = agg["Contained_Misidentification"].astype(float)

    agg["Total"] = CS + NN + DF + CM

    # Avoid division by zero
    denom = agg["Total"].replace(0, np.nan)

    # Accuracy ignoring rejection (semantic preservation)
    agg["Accuracy"] = (CS + NN) / denom

    # OSA on ID-only data: accepted & correct rate under operational threshold
    agg["OSA_ID"] = CS / denom

    # --- FIX: Explicitly save Correct_Total for plots.py filtering ---
    agg["Correct_Total"] = CS + NN

    # Conditional nuisance rejection among correct predictions
    correct_total_safe = agg["Correct_Total"].replace(0, np.nan)
    agg["CNR"] = NN / correct_total_safe

    # Rejection / failure mode rates
    agg["Rejection_Rate"] = (NN + DF) / denom
    agg["NNR"] = NN / denom
    agg["CM_Rate"] = CM / denom
    agg["DF_Rate"] = DF / denom
    agg["Accept_Rate"] = (CS + CM) / denom

    # Decoupling summary
    agg["OSA_Gap"] = agg["Accuracy"] - agg["OSA_ID"]

    return agg

def compute_adr(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Accuracy Divergence Rate (ADR).
    ADR = Slope of (OSA_Gap vs Level).

    Returns a DataFrame with columns: [backbone, detector, dataset, ADR]
    """
    # Filter to relevant levels (0-5) to establish slope
    # We include level 0 if available to anchor the divergence start point.
    subset = agg_df[agg_df["level"].between(0, 5)].copy()

    results = []

    # Calculate slope per (backbone, detector, dataset)
    # We aggregate over nuisances implicitly by feeding all (level, gap) points
    # for a specific detector into the regression. This gives the "Mean Trajectory" slope.
    groups = subset.groupby(["backbone", "detector", "dataset"])

    for (bb, det, ds), group in groups:
        x = group["level"].values
        y = group["OSA_Gap"].values

        if len(np.unique(x)) < 2:
            slope = np.nan
        else:
            # Linear fit (degree 1), return slope (index 0)
            slope = np.polyfit(x, y, 1)[0]

        results.append({
            "backbone": bb,
            "detector": det,
            "dataset": ds,
            "ADR": slope
        })

    return pd.DataFrame(results)