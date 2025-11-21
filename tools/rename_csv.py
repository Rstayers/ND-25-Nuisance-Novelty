#!/usr/bin/env python3
"""
Quick utility to rename error_type labels in a nuisance benchmark CSV.

Usage:
    python update_error_types.py --csv old_results.csv --out updated_results.csv
"""

import pandas as pd
import argparse

# --- Mapping from old -> new ---
RENAME_MAP = {
    "Full_Correct": "Clean_Success",
    "Partial_Correct": "Contained_Misidentification",
    "Full_Nuisance": "Nuisance_Novelty",
    "Partial_Nuisance": "Double_Failure",
}

def main(csv_path: str, out_path: str):
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    if "error_type" not in df.columns:
        raise ValueError("No 'error_type' column found in CSV.")

    # Map to new labels
    df["error_type"] = df["error_type"].map(RENAME_MAP).fillna(df["error_type"])

    # Optional: rename the column for clarity in new analyses
    df.rename(columns={"error_type": "system_outcome"}, inplace=True)

    print(f"[INFO] Writing updated file: {out_path}")
    df.to_csv(out_path, index=False)
    print("[DONE] Successfully updated error types.")
    print(df["system_outcome"].value_counts())

if __name__ == "__main__":

   main("../results/nuisance_runs/cns_bench/combined.csv", "results/nuisance_runs/cns_bench/combined_results.csv")
