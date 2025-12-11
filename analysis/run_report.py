import argparse
import os
import pandas as pd

from analysis.processing import load_and_prep, compute_aggregates, compute_ood_metrics
from analysis.plots import (
    plot_outcomes_stack, plot_competency_cliff,
    plot_nuisance_fingerprint, plot_accuracy_degradation, plot_safety_alignment
)
from analysis.tables import (
    generate_detector_leaderboard,
    generate_transfer_table,
    generate_master_table,
    generate_acc_accept_table,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help="Path to benchmark_final.csv")
    parser.add_argument('--out_dir', default="analysis/report_output")
    args = parser.parse_args()

    dirs = {
        "root": args.out_dir,
        "stacks": os.path.join(args.out_dir, "plots", "stacks"),
        "cliffs": os.path.join(args.out_dir, "plots", "cliffs"),
        "alignment": os.path.join(args.out_dir, "plots", "alignment"),
        "tables": os.path.join(args.out_dir, "tables"),
        "raw": os.path.join(args.out_dir, "raw")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # 1. Load & Process
    print("Processing Data...")
    raw_df, _ = load_and_prep(args.csv)
    agg_df = compute_aggregates(raw_df)

    print("Calculating OOD Metrics (AUROC/FPR95)...")
    ood_df = compute_ood_metrics(raw_df)

    # Save intermediates
    agg_df.to_csv(os.path.join(dirs['raw'], "aggregated_metrics.csv"), index=False)
    if not ood_df.empty:
        ood_df.to_csv(os.path.join(dirs['raw'], "ood_metrics.csv"), index=False)

    # 2. Plots
    print("\nGenerating Visualizations...")
    datasets = agg_df['dataset'].unique()
    unique_backbones = agg_df['backbone'].unique()

    for ds_name in datasets:
        # Skip ID dataset for plots
        if "ImageNet-Val" in ds_name: continue

        print(f"  > Dataset: {ds_name}")

        # Stacks
        ds_subset = agg_df[agg_df['dataset'] == ds_name]
        combos = ds_subset[['backbone', 'detector']].drop_duplicates().values
        for bb, det in combos:
            plot_outcomes_stack(agg_df, bb, det, ds_name, dirs['stacks'])

        # Cliff, Fingerprint, Alignment
        plot_competency_cliff(agg_df, ds_name, dirs['cliffs'])
        plot_nuisance_fingerprint(agg_df, ds_name, dirs['cliffs'])  # Saved in cliffs dir
        plot_accuracy_degradation(agg_df, ds_name, dirs['cliffs'])
        plot_safety_alignment(agg_df, ds_name, dirs['alignment'])

    # 3. Tables
    print("\nGenerating Tables...")
    generate_master_table(agg_df, ood_df, dirs['tables'])

    for ds_name in datasets:
        if "ImageNet-Val" in ds_name: continue

        # --- NEW: Loop over ALL backbones for Leaderboards ---
        for bb in unique_backbones:
            generate_detector_leaderboard(agg_df, ood_df, bb, ds_name, dirs['tables'])

        # Transfer tables compare backbones, so we run them once per dataset/level
        generate_transfer_table(agg_df, 3, ds_name, dirs['tables'])
        generate_transfer_table(agg_df, 5, ds_name, dirs['tables'])
        generate_acc_accept_table(agg_df, ds_name, dirs['tables'])

    print(f"\nReport Generation Complete.")


if __name__ == "__main__":
    main()