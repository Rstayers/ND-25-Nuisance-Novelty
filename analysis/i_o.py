# analysis/io.py
"""
I/O utilities for Psycho Benchmark analysis
-------------------------------------------
Handles loading and normalization of CSV data
from the Psycho Benchmark outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_with_enrichment(csv_path: str) -> pd.DataFrame:
    """
    Load a Psycho Benchmark per-sample CSV and enrich it
    with standardized column names, types, and dataset indicators.

    Ensures compatibility across CNS-Bench, ImageNet-C, and NINCO.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"[LOAD] Loaded {n_rows:,} rows from {csv_path.name}")

    # --- Normalize key columns ---
    for col in ["accept", "correct_cls"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "fpr_target" in df.columns:
        df["fpr_target"] = pd.to_numeric(df["fpr_target"], errors="coerce")

    if "backbone" not in df.columns:
        df["backbone"] = "unknown"

    # --- Unify naming conventions ---
    # CNS-Bench uses shift/scale naming; ImageNet-C uses corruption/severity
    if "shift" in df.columns and "corruption" not in df.columns:
        df["corruption"] = df["shift"].astype(str)
    if "scale" in df.columns and "severity" not in df.columns:
        df["severity"] = pd.to_numeric(df["scale"], errors="coerce")

    # Convert severity to numeric where possible
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce")

    # Rename error_type → system_outcome (for old versions)
    if "error_type" in df.columns and "system_outcome" not in df.columns:
        df = df.rename(columns={"error_type": "system_outcome"})

    # Ensure consistent dataset column
    if "dataset" not in df.columns:
        print("[WARN] No dataset column found. Inferring from file path.")
        fname = str(csv_path).lower()
        if "cns" in fname:
            df["dataset"] = "cns"
        elif "imagenet_c" in fname or "imagenetc" in fname:
            df["dataset"] = "imagenet_c"
        elif "ninco" in fname:
            df["dataset"] = "ninco"
        else:
            df["dataset"] = "unknown"

    # --- Enforce consistent types for grouping columns ---
    group_cols = [
        "dataset", "backbone", "detector",
        "fpr_target", "severity", "corruption",
    ]
    for g in group_cols:
        if g not in df.columns:
            df[g] = np.nan

    # --- Handle missing system_outcome gracefully ---
    if "system_outcome" not in df.columns:
        df["system_outcome"] = "Unknown"

    # --- Sort ---
    sort_cols = [c for c in ["dataset", "backbone", "detector", "severity"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def split_by_dataset(df: pd.DataFrame):
    """
    Split a unified Psycho Benchmark dataframe into subsets:
      - CNS-Bench
      - ImageNet-C
      - NINCO
      - Unknown
    Returns a dict of {dataset_name: sub_df}.
    """
    datasets = {}
    if "dataset" not in df.columns:
        return {"unknown": df}

    for name in ["cns", "imagenet_c", "ninco"]:
        mask = df["dataset"].astype(str).str.contains(name, case=False, na=False)
        if mask.any():
            datasets[name] = df[mask].copy()
    if not datasets:
        datasets["unknown"] = df.copy()
    return datasets
