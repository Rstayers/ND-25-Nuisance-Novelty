import argparse
import os

from analysis.processing import load_and_prep, compute_aggregates
from analysis.plots import (
    plot_CNR,
    plot_nuisance_fingerprint,
    plot_detector_leaderboard,
    plot_osa_gap,
)
from analysis.tables import generate_master_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to final_benchmark.csv")
    parser.add_argument("--out_dir", default="analysis/report_output")
    args = parser.parse_args()

    dirs = {
        "root": args.out_dir,
        "plots_OSA": os.path.join(args.out_dir, "plots", "OSA"),
        "plots_CNR": os.path.join(args.out_dir, "plots", "CNR"),
        "plots_nuisance": os.path.join(args.out_dir, "plots", "nuisance_type"),
        "plots_leaderboard": os.path.join(args.out_dir, "plots", "leaderboard"),
        "tables": os.path.join(args.out_dir, "tables"),
        "raw": os.path.join(args.out_dir, "raw"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Processing Data...")
    raw_df, _ = load_and_prep(args.csv)
    agg_df = compute_aggregates(raw_df)

    agg_path = os.path.join(dirs["raw"], "aggregated_metrics.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    print("\nGenerating Visualizations...")
    for ds_name in agg_df["dataset"].unique():
        print(f"  > Dataset: {ds_name}")
        plot_osa_gap(agg_df, ds_name, dirs["plots_OSA"])
        plot_detector_leaderboard(agg_df, ds_name, dirs["plots_leaderboard"])
        plot_CNR(agg_df, ds_name, dirs["plots_CNR"])
        plot_nuisance_fingerprint(agg_df, ds_name, dirs["plots_nuisance"])

    print("\nGenerating Tables...")
    generate_master_table(agg_df, dirs["tables"])


if __name__ == "__main__":
    main()
