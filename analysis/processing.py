import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_and_prep(csv_path):
    df = pd.read_csv(csv_path)
    required_outcomes = [
        "Clean_Success", "Nuisance_Novelty",
        "Double_Failure", "Contained_Misidentification"
    ]
    return df, required_outcomes


def compute_aggregates(df):
    """
    Aggregates row-level data into counts/rates per group.
    """
    cols = ['backbone', 'detector', 'dataset', 'nuisance', 'level', 'outcome']
    agg = df.groupby(cols).size().unstack(fill_value=0).reset_index()

    for col in ["Clean_Success", "Nuisance_Novelty", "Double_Failure", "Contained_Misidentification"]:
        if col not in agg.columns: agg[col] = 0

    # Total
    agg['Total'] = (agg['Clean_Success'] + agg['Nuisance_Novelty'] +
                    agg['Double_Failure'] + agg['Contained_Misidentification'])

    # Accuracy (CS + NN) / Total
    agg['Accuracy'] = (agg['Clean_Success'] + agg['Nuisance_Novelty']) / agg['Total']

    # Rejection Rate (NN + DF) / Total
    agg['Rejection_Rate'] = (agg['Nuisance_Novelty'] + agg['Double_Failure']) / agg['Total']

    # Conditional Nuisance Rate (CNR) - Formerly CRR
    # P(Rejected | Correct)
    correct_total = agg['Nuisance_Novelty'] + agg['Clean_Success']
    agg['CNR'] = agg['Nuisance_Novelty'] / correct_total
    agg['CNR'] = agg['CNR'].fillna(0.0)

    # Safety Recall (SR)
    incorrect_total = agg['Double_Failure'] + agg['Contained_Misidentification']
    agg['SR'] = agg['Double_Failure'] / incorrect_total
    agg['SR'] = agg['SR'].fillna(1.0)

    return agg


def compute_ood_metrics(raw_df):
    """
    Calculates standard OOD metrics (AUROC, FPR95) for detecting
    Nuisance (Pos) vs ID (Neg) based on Confidence.
    """
    # 1. Identify ID data (Calibration set)
    # We assume 'ImageNet-Val' or similar is the ID set.
    # Usually ID has level=0 and nuisance='clean' or 'ImageNet-Val'
    # Let's assume any dataset containing 'ImageNet-Val' or 'ID' is ID.
    id_df = raw_df[raw_df['dataset'].str.contains("ImageNet-Val", case=False, na=False)].copy()

    if id_df.empty:
        # Fallback: try finding level 0 clean data?
        # Or just return empty if no ID data found in CSV
        print("Warning: No ID data found for AUROC calculation.")
        return pd.DataFrame()

    id_df['target'] = 0  # Negative Class (ID)

    metrics = []

    # Group Nuisance Data by (Backbone, Detector, Dataset, Level)
    # We treat each Level as a separate OOD detection task
    groups = raw_df.groupby(['backbone', 'detector', 'dataset', 'level'])

    for (bb, det, ds, lvl), group in groups:
        # Skip ID data itself
        if "ImageNet-Val" in ds: continue

        # Skip Level 0 (usually clean)
        if lvl == 0: continue

        # Filter ID data for this backbone/detector
        curr_id = id_df[
            (id_df['backbone'] == bb) &
            (id_df['detector'] == det)
            ].copy()

        if curr_id.empty: continue

        # OOD Data (Positive Class)
        curr_ood = group.copy()
        curr_ood['target'] = 1

        # Merge
        combined = pd.concat([curr_id, curr_ood])
        y_true = combined['target'].values
        # Score: We want OOD to have HIGHER score.
        # Confidence is high for ID. So Score = -Confidence (or 1-Conf)
        y_score = -combined['confidence'].values

        # AUROC
        auroc = roc_auc_score(y_true, y_score)

        # FPR @ 95% TPR
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Find index where TPR >= 0.95
        idx = np.where(tpr >= 0.95)[0][0]
        fpr95 = fpr[idx]

        metrics.append({
            'backbone': bb,
            'detector': det,
            'dataset': ds,
            'level': lvl,
            'AUROC': auroc,
            'FPR95': fpr95
        })

    return pd.DataFrame(metrics)