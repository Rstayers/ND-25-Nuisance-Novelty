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
    Loads the benchmark CSV. Does not assume any particular dataset naming
    (e.g., no 'ImageNet-Val' heuristics).
    """
    df = pd.read_csv(csv_path)

    # Normalize outcome dtype
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str)

    # Ensure required columns exist (best-effort; fail loudly if core columns missing)
    needed = {"backbone", "detector", "dataset", "nuisance", "level", "outcome"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(list(missing))}")

    # Ensure level is numeric
    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(0).astype(int)

    return df, REQUIRED_OUTCOMES


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates row-level data and calculates key ID-only nuisance metrics.

    Definitions (per group):
      CS = Clean_Success (correct & accepted)
      NN = Nuisance_Novelty (correct & rejected)
      DF = Double_Failure (incorrect & rejected)
      CM = Contained_Misidentification (incorrect & accepted)
      Total = CS + NN + DF + CM

    Metrics:
      Mean_ID_Acc (Accuracy)  = (CS+NN)/Total
      OSA_ID                  = CS/Total  (accepted & correct under threshold)
      CNR                     = NN/(CS+NN) (conditional rejection among correct)
      Rejection_Rate          = (NN+DF)/Total
      NNR                     = NN/Total (unconditional nuisance novelty rate)
      CM_Rate                 = CM/Total
      DF_Rate                 = DF/Total
      OSA_Gap                 = Accuracy - OSA_ID
      Accept_Rate             = (CS+CM)/Total
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

    # Conditional nuisance rejection among correct predictions
    correct_total = (CS + NN).replace(0, np.nan)
    agg["Correct_Total"] = CS + NN
    agg["CNR"] = NN / correct_total

    # Rejection / failure mode rates
    agg["Rejection_Rate"] = (NN + DF) / denom
    agg["NNR"] = NN / denom
    agg["CM_Rate"] = CM / denom
    agg["DF_Rate"] = DF / denom
    agg["Accept_Rate"] = (CS + CM) / denom

    # Decoupling summary
    agg["OSA_Gap"] = agg["Accuracy"] - agg["OSA_ID"]

    # Keep NaNs (do NOT fill CNR=0 when Correct_Total=0; that hides collapse)
    # Users can decide how to aggregate NaNs later.
    return agg
