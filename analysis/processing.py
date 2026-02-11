# analysis/processing.py
# Multi-dataset analysis for Nuisance Novelty benchmark
#
# CRITICAL PRINCIPLES:
# - Metrics computed for (backbone, detector, dataset, level) tuples
# - Can collapse over nuisance types (sum outcomes, recompute metrics)
# - Can average over severity levels for a given (backbone, detector, dataset)
# - NEVER average across backbones or detectors
#
# METRIC DEFINITIONS (must match run_bench.py exactly):
#   CSA     = N_correct / N_total                    (closed-set accuracy)
#   CCR     = Clean_Success / N_total                (correct & accepted rate)
#   NNR     = Nuisance_Novelty / N_total             (nuisance novelty prevalence)
#   CNR     = Nuisance_Novelty / N_correct           (conditional novelty rate)
#   OSA_Gap = CSA - CCR = NNR                        (known-side divergence)
#   URR     = rejected_unknowns / n_unknown          (unknown rejection rate)
#   OSA     = (Clean_Success + rejected_unknowns) / (N_total + n_unknown)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

# =============================================================================
# OUTCOME CATEGORIES
# =============================================================================

OUTCOME_COLS = [
    "Clean_Success",               # Correct AND Accepted (ideal)
    "Nuisance_Novelty",           # Correct AND Rejected (THE PROBLEM)
    "Double_Failure",              # Wrong AND Rejected
    "Contained_Misidentification", # Wrong AND Accepted
]

METRIC_LABELS = {
    "CSA": "Closed-Set Accuracy",
    "CCR": r"CCR@$\theta$",
    "NNR": "Nuisance Novelty Rate",
    "CNR": "Conditional Novelty Rate",
    "ADR": "Accuracy Divergence Rate",
    "OSA_Gap": r"CSA $-$ CCR@$\theta$",
    "OSA": "Open-Set Accuracy",
    "URR": r"URR@$\theta$",
}

METRIC_SHORT = {
    "CSA": "CSA",
    "CCR": r"CCR@$\theta$",
    "NNR": "NNR",
    "CNR": "CNR",
    "ADR": "ADR",
    "OSA_Gap": "Gap",
    "OSA": "OSA",
    "URR": "URR",
}


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_samples(csv_path: str) -> pd.DataFrame:
    """Load a single bench_samples.csv and normalize columns."""
    df = pd.read_csv(csv_path)

    if "dataset" not in df.columns:
        if "dataset_name" in df.columns:
            df["dataset"] = df["dataset_name"]
        elif "test_dataset" in df.columns:
            df["dataset"] = df["test_dataset"]

    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str)

    if "level" in df.columns:
        df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(0).astype(int)

    return df


def load_and_combine_csvs(csv_paths: List[str], filter_datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """Load CSVs and auto-detect datasets within them."""
    dfs = []

    for path in csv_paths:
        df = load_samples(path)

        if "dataset" not in df.columns:
            if "dataset_name" in df.columns:
                df["dataset"] = df["dataset_name"]
            else:
                path_lower = path.lower()
                if "imagenet" in path_lower:
                    df["dataset"] = "ImageNet-LN"
                elif "cub" in path_lower:
                    df["dataset"] = "CUB-LN"
                elif "cars" in path_lower:
                    df["dataset"] = "Cars-LN"
                else:
                    df["dataset"] = "Unknown"

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if filter_datasets:
        combined = combined[combined["dataset"].isin(filter_datasets)]

    datasets_found = combined["dataset"].unique().tolist()
    print(f"  Datasets found: {datasets_found}")

    return combined


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def enrich_with_summary(sample_df: pd.DataFrame, summary_csv: str) -> pd.DataFrame:
    """
    For old bench_samples.csv files that lack n_unknown/rejected_unknowns,
    join them from the corresponding bench_summary.csv.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Sample-level data (from bench_samples.csv).
    summary_csv : str
        Path to the corresponding bench_summary.csv.

    Returns
    -------
    pd.DataFrame
        Enriched sample_df with n_unknown and rejected_unknowns columns.
    """
    if "n_unknown" in sample_df.columns and "rejected_unknowns" in sample_df.columns:
        print("  Sample data already has OOD constants, skipping enrichment.")
        return sample_df

    summary = pd.read_csv(summary_csv)

    # Standardize column names
    if "test_dataset" in summary.columns:
        summary = summary.rename(columns={"test_dataset": "dataset"})
    if "dataset_name" in sample_df.columns and "dataset" not in sample_df.columns:
        sample_df["dataset"] = sample_df["dataset_name"]

    # Reconstruct rejected_unknowns from summary if not present
    if "rejected_unknowns" not in summary.columns:
        if "URR@theta" in summary.columns and "n_unknown" in summary.columns:
            summary["rejected_unknowns"] = (summary["URR@theta"] * summary["n_unknown"]).round().astype(int)
        else:
            print("  WARNING: Cannot enrich - summary missing URR@theta or n_unknown.")
            return sample_df

    # Build join key
    join_cols = [c for c in ["backbone", "detector", "dataset"] if c in summary.columns and c in sample_df.columns]
    if not join_cols:
        print("  WARNING: Cannot enrich - no common join columns.")
        return sample_df

    ood_info = summary[join_cols + ["n_unknown", "rejected_unknowns"]].drop_duplicates()

    enriched = sample_df.merge(ood_info, on=join_cols, how="left")

    n_enriched = enriched["n_unknown"].notna().sum()
    print(f"  Enriched {n_enriched}/{len(enriched)} samples with OOD constants.")

    return enriched


# =============================================================================
# CORE METRIC COMPUTATION
# =============================================================================

def _compute_derived_metrics(df: pd.DataFrame) -> None:
    """
    Compute ALL derived metrics in-place from outcome counts.

    Requires columns: Clean_Success, Nuisance_Novelty, Double_Failure,
                      Contained_Misidentification
    Optional columns: n_unknown, rejected_unknowns (for OSA/URR)
    """
    CS = df["Clean_Success"].astype(float)
    NN = df["Nuisance_Novelty"].astype(float)
    DF = df["Double_Failure"].astype(float)
    CM = df["Contained_Misidentification"].astype(float)

    df["N_total"] = CS + NN + DF + CM
    df["N_correct"] = CS + NN

    n_total = df["N_total"].replace(0, np.nan)
    n_correct = df["N_correct"].replace(0, np.nan)

    # --- Known-side metrics (always computable from outcomes) ---
    df["CSA"] = (CS + NN) / n_total                # Closed-Set Accuracy
    df["CCR"] = CS / n_total                        # Correct Classification Rate @ theta
    df["NNR"] = NN / n_total                        # Nuisance Novelty Rate (prevalence)
    df["CNR"] = NN / n_correct                      # Conditional Novelty Rate (on correct)
    df["OSA_Gap"] = df["CSA"] - df["CCR"]           # = NNR algebraically

    # --- Joint metrics (require OOD constants) ---
    has_ood = "n_unknown" in df.columns and "rejected_unknowns" in df.columns
    if has_ood:
        n_unknown = df["n_unknown"].astype(float)
        rej_unk = df["rejected_unknowns"].astype(float)
        df["URR"] = rej_unk / n_unknown.replace(0, np.nan)
        df["OSA"] = (CS + rej_unk) / (df["N_total"] + n_unknown)
    else:
        df["URR"] = np.nan
        df["OSA"] = np.nan


# =============================================================================
# CORE AGGREGATION
# =============================================================================

def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sample-level data into outcome counts and compute all metrics.
    Preserves (backbone, detector, dataset, nuisance, level) granularity.

    If OOD constants (n_unknown, rejected_unknowns) are present per-sample,
    they propagate through so OSA/URR can be computed at every granularity.
    """
    if "outcome" not in df.columns:
        print("Warning: No 'outcome' column. Returning as-is.")
        return df

    # --- Detect OOD constants ---
    has_ood = ("n_unknown" in df.columns and "rejected_unknowns" in df.columns)

    # --- Pivot outcomes into columns ---
    possible_groups = ["backbone", "detector", "dataset", "nuisance", "level", "outcome"]
    group_cols = [c for c in possible_groups if c in df.columns]

    agg = df.groupby(group_cols).size().unstack(fill_value=0).reset_index()

    for col in OUTCOME_COLS:
        if col not in agg.columns:
            agg[col] = 0

    # --- Propagate OOD constants ---
    # These are constant per (backbone, detector), so take first from any matching group
    if has_ood:
        ood_key_cols = [c for c in ["backbone", "detector"] if c in df.columns]
        if ood_key_cols:
            ood_constants = df.groupby(ood_key_cols).agg(
                n_unknown=("n_unknown", "first"),
                rejected_unknowns=("rejected_unknowns", "first"),
            ).reset_index()
            agg = agg.merge(ood_constants, on=ood_key_cols, how="left")
        else:
            agg["n_unknown"] = df["n_unknown"].iloc[0]
            agg["rejected_unknowns"] = df["rejected_unknowns"].iloc[0]

    # --- Compute all metrics ---
    _compute_derived_metrics(agg)

    return agg


def aggregate_over_nuisances(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Collapse over nuisance types by summing outcome counts, recompute metrics.

    group_cols MUST include 'backbone' and 'detector' to avoid cross-contamination.
    """
    outcome_cols = [c for c in OUTCOME_COLS if c in df.columns]

    if not outcome_cols:
        raise ValueError("No outcome columns found")

    # Sum outcomes; for OOD constants take first (constant per bb+det)
    agg_dict = {c: "sum" for c in outcome_cols}

    has_ood = "n_unknown" in df.columns and "rejected_unknowns" in df.columns
    if has_ood:
        agg_dict["n_unknown"] = "first"
        agg_dict["rejected_unknowns"] = "first"

    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()

    _compute_derived_metrics(grouped)

    return grouped


# =============================================================================
# METRIC ACCESSORS (preserve backbone x detector)
# =============================================================================

def get_metrics_by_severity(agg_df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """
    Get metrics per (backbone, detector, level) for a dataset.
    Aggregates over nuisance types, preserves backbone x detector.

    Returns columns: backbone, detector, dataset, level,
                     CSA, CCR, NNR, CNR, OSA_Gap, OSA, URR,
                     N_total, N_correct, + outcome counts
    """
    df = agg_df.copy()
    if dataset:
        df = df[df["dataset"] == dataset]

    if df.empty:
        return pd.DataFrame()

    group_cols = []
    if "backbone" in df.columns:
        group_cols.append("backbone")
    if "detector" in df.columns:
        group_cols.append("detector")
    if "dataset" in df.columns:
        group_cols.append("dataset")
    group_cols.append("level")

    return aggregate_over_nuisances(df, group_cols)


def get_mean_metrics(agg_df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """
    Get mean metrics per (backbone, detector) averaged over severity levels (1-5).
    Returns one row per (backbone, detector, dataset) combination.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return pd.DataFrame()

    # Only corrupted (level > 0)
    per_sev = per_sev[per_sev["level"] > 0]

    base_cols = []
    if "backbone" in per_sev.columns:
        base_cols.append("backbone")
    if "detector" in per_sev.columns:
        base_cols.append("detector")
    if "dataset" in per_sev.columns:
        base_cols.append("dataset")

    metric_cols = ["CSA", "CCR", "NNR", "CNR", "OSA_Gap", "OSA", "URR"]
    metric_cols = [c for c in metric_cols if c in per_sev.columns]

    return per_sev.groupby(base_cols)[metric_cols].mean().reset_index()


def compute_adr(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ADR for each (backbone, detector, dataset).
    ADR = slope of OSA_Gap vs severity level.
    """
    if "level" not in agg_df.columns:
        return pd.DataFrame()

    group_cols = []
    if "backbone" in agg_df.columns:
        group_cols.append("backbone")
    if "detector" in agg_df.columns:
        group_cols.append("detector")
    if "dataset" in agg_df.columns:
        group_cols.append("dataset")
    group_cols.append("level")

    per_severity = aggregate_over_nuisances(agg_df, group_cols)
    subset = per_severity[per_severity["level"].between(1, 5)].copy()

    if subset.empty:
        return pd.DataFrame()

    results = []
    base_cols = [c for c in group_cols if c != "level"]

    for keys, group in subset.groupby(base_cols):
        x = group["level"].values
        y = group["OSA_Gap"].values

        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue

        x_clean, y_clean = x[mask], y[mask]

        if len(np.unique(x_clean)) < 2:
            continue

        slope, intercept = np.polyfit(x_clean, y_clean, 1)

        result = dict(zip(base_cols, keys if isinstance(keys, tuple) else [keys]))
        result["ADR"] = slope
        results.append(result)

    return pd.DataFrame(results)