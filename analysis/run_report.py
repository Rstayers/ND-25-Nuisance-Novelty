#!/usr/bin/env python3
# analysis/run_report.py

import argparse
import os

from analysis.processing import load_and_combine_csvs, compute_aggregates, load_summary
from analysis.plots import (
    plot_per_dataset_all,
    plot_benchmark_all,
    plot_cross_dataset_all,
    # OSA-NNR Landscape plots
    plot_osa_nnr_landscape,
    plot_osa_nnr_landscape_faceted,
    plot_osa_nnr_density,
    # OOD Detection metrics plots (from summary data)
    plot_ood_detection_all,
    # Q2: System-dependent nuisance novelty plots
    plot_q2_all,
)
from analysis.tables import (
    generate_dataset_summary_table,
    generate_full_results_table,
    generate_nnr_at_severity_table,
    generate_osa_at_severity_table,
    generate_adr_benchmark_table,
    generate_benchmark_summary_table,
    generate_benchmark_osa_table,
    generate_benchmark_by_severity_table,
    generate_cross_dataset_summary,
    generate_cross_dataset_osa,
    generate_cross_dataset_adr,
    generate_cross_dataset_full,
    generate_rank_comparison_table,
    generate_outcome_table,
    # OOD Detection metrics tables (from summary data)
    generate_auoscr_table,
    generate_auroc_table,
    generate_fpr95_table,
    generate_ood_metrics_summary_table,
    # COSTARR-style OSA table
    generate_costarr_style_osa_table,
    generate_costarr_benchmark_osa_tables,
    # Final paper tables
    generate_final_tables,
)
from analysis.ensemble_analysis import run_full_ensemble_analysis


def run_per_dataset(agg_df, out_dir):
    """Per-dataset figures and tables."""
    datasets = sorted(agg_df["dataset"].unique())

    for ds in datasets:
        ds_dir = os.path.join(out_dir, "figures", "per_dataset", ds.replace("-", "_"))
        os.makedirs(ds_dir, exist_ok=True)

        plot_per_dataset_all(agg_df, ds, ds_dir)

        table_dir = os.path.join(out_dir, "tables", "per_dataset")
        os.makedirs(table_dir, exist_ok=True)
        generate_dataset_summary_table(agg_df, ds, table_dir)
        generate_full_results_table(agg_df, ds, table_dir)


def run_benchmark(agg_df, out_dir, summary_df=None, sample_df=None):
    """Benchmark comparison figures and tables."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    fig_dir = os.path.join(out_dir, "figures", "benchmark")
    table_dir = os.path.join(out_dir, "tables", "benchmark")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    plot_benchmark_all(agg_df, fig_dir)

    # OSA-NNR Landscape plots
    print("\n  OSA-NNR Landscape:")
    datasets = sorted(agg_df["dataset"].unique()) if "dataset" in agg_df.columns else []
    for dataset in datasets:
        plot_osa_nnr_landscape(agg_df, dataset, fig_dir)
    plot_osa_nnr_landscape_faceted(agg_df, fig_dir)
    plot_osa_nnr_density(agg_df, fig_dir)

    # OOD Detection metrics (from summary data)
    if summary_df is not None:
        plot_ood_detection_all(summary_df, fig_dir)

    # Q2: System-dependent nuisance novelty plots (OSA @ L5)
    plot_q2_all(agg_df, fig_dir, severity=5)

    print("\n  Tables:")
    generate_nnr_at_severity_table(agg_df, table_dir, severity=5)
    generate_osa_at_severity_table(agg_df, table_dir, severity=5)
    generate_adr_benchmark_table(agg_df, table_dir)
    generate_benchmark_summary_table(agg_df, table_dir)
    generate_benchmark_osa_table(agg_df, table_dir)
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="CSA")
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="NNR")
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="OSA")
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="OSA_Gap")

    # COSTARR-style OSA table (backbone × detector × OOD dataset)
    generate_costarr_style_osa_table(agg_df, table_dir, dataset_filter="ImageNet-LN", severity=5)

    # COSTARR-style benchmark OSA tables (columns = nuisance datasets: LN, C, CNS)
    generate_costarr_benchmark_osa_tables(agg_df, table_dir, severity=5)

    # Final paper-ready tables (System Selection + Side-by-side COSTARR)
    print("\n  Final Tables:")
    generate_final_tables(agg_df, table_dir, severity=5)

    # OOD Detection metric tables (from summary data)
    if summary_df is not None:
        print("\n  OOD Detection Tables:")
        generate_auoscr_table(summary_df, table_dir)
        generate_auroc_table(summary_df, table_dir)
        generate_fpr95_table(summary_df, table_dir)
        generate_ood_metrics_summary_table(summary_df, table_dir)

    # Comprehensive Ensemble Analysis (Section 4.3 / Q2)
    if sample_df is not None:
        print("\n  Ensemble Analysis:")
        ensemble_fig_dir = os.path.join(out_dir, "figures", "benchmark", "ensemble_analysis")
        ensemble_table_dir = os.path.join(out_dir, "tables", "benchmark", "ensemble_analysis")

        run_full_ensemble_analysis(
            sample_df=sample_df,
            agg_df=agg_df,
            fig_dir=ensemble_fig_dir,
            table_dir=ensemble_table_dir,
            severity=5,
            datasets=["ImageNet-LN"],
            find_optimal=False,
            k_optimal=[3, 5]
        )


def run_cross_dataset(agg_df, out_dir, reference="ImageNet-LN"):
    """Cross-dataset comparison figures and tables."""
    print("\n" + "=" * 60)
    print("CROSS-DATASET COMPARISON")
    print("=" * 60)

    fig_dir = os.path.join(out_dir, "figures", "cross_dataset")
    table_dir = os.path.join(out_dir, "tables", "cross_dataset")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    plot_cross_dataset_all(agg_df, fig_dir, reference=reference)

    print("\n  Tables:")
    generate_cross_dataset_summary(agg_df, table_dir)
    generate_cross_dataset_osa(agg_df, table_dir)  # NEW
    generate_cross_dataset_adr(agg_df, table_dir)
    generate_cross_dataset_full(agg_df, table_dir)
    generate_rank_comparison_table(agg_df, table_dir, metric="NNR")
    generate_rank_comparison_table(agg_df, table_dir, metric="OSA")  # NEW
    generate_outcome_table(agg_df, table_dir)


def main():
    parser = argparse.ArgumentParser(description="Nuisance Novelty Analysis")

    parser.add_argument("--benchmark_csvs", nargs="+", default=None,
                        help="Paths to bench_samples.csv files")
    parser.add_argument("--benchmark_filter", nargs="+", default=None,
                        help="Filter to specific datasets")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Path to bench_summary.csv for OOD metrics (AUOSCR, AUROC, FPR@95)")
    parser.add_argument("--cross_csvs", nargs="+", default=None,
                        help="Paths to cross-dataset bench_samples.csv files")
    parser.add_argument("--cross_filter", nargs="+", default=None,
                        help="Filter to specific cross-datasets")
    parser.add_argument("--reference", default="ImageNet-LN",
                        help="Reference dataset for cross-dataset comparisons")
    parser.add_argument("--out", default="paper_output",
                        help="Output directory")
    parser.add_argument("--ensemble_only", action="store_true",
                        help="Run only ensemble analysis on LN datasets (skip all other analysis)")
    parser.add_argument("--ensemble_datasets", nargs="+", default=None,
                        help="Datasets for ensemble analysis (default: all LN datasets found)")

    args = parser.parse_args()

    if args.benchmark_csvs is None and args.cross_csvs is None:
        parser.error("Must provide --benchmark_csvs and/or --cross_csvs")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)

    print("=" * 60)
    print("NUISANCE NOVELTY ANALYSIS")
    print("=" * 60)

    # ENSEMBLE-ONLY MODE: Skip all other analysis, just run ensemble on LN datasets
    if args.ensemble_only:
        print("\n[ENSEMBLE-ONLY MODE]")

        # Load data from benchmark_csvs or cross_csvs
        csvs = args.benchmark_csvs or args.cross_csvs
        filter_datasets = args.benchmark_filter or args.cross_filter

        print(f"  Loading data from: {csvs}")
        sample_df = load_and_combine_csvs(csvs, filter_datasets)
        agg_df = compute_aggregates(sample_df)

        # Determine which datasets to analyze
        all_datasets = sample_df["dataset"].unique() if "dataset" in sample_df.columns else sample_df["dataset_name"].unique()
        if args.ensemble_datasets:
            ensemble_datasets = args.ensemble_datasets
        else:
            # Default: all LN datasets
            ensemble_datasets = [ds for ds in all_datasets if "ln" in ds.lower() or "LN" in ds]

        print(f"  Ensemble datasets: {ensemble_datasets}")

        ensemble_fig_dir = os.path.join(args.out, "figures", "benchmark", "ensemble_analysis")
        ensemble_table_dir = os.path.join(args.out, "tables", "benchmark", "ensemble_analysis")

        run_full_ensemble_analysis(
            sample_df=sample_df,
            agg_df=agg_df,
            fig_dir=ensemble_fig_dir,
            table_dir=ensemble_table_dir,
            severity=5,
            datasets=ensemble_datasets,
            find_optimal=False,
            k_optimal=[3, 5]
        )

        print("\n" + "=" * 60)
        print("ENSEMBLE ANALYSIS COMPLETE!")
        print("=" * 60)
        return

    # Load summary data if provided (for OOD metrics)
    summary_df = None
    if args.summary_csv:
        print(f"\n[LOADING] Summary data from {args.summary_csv}...")
        summary_df = load_summary(args.summary_csv)
        print(f"  Summary loaded: {len(summary_df)} rows")

    if args.benchmark_csvs:
        print("\n[LOADING] Benchmark data...")
        bench_df = load_and_combine_csvs(args.benchmark_csvs, args.benchmark_filter)
        bench_agg = compute_aggregates(bench_df)
        bench_agg.to_csv(os.path.join(args.out, "data", "benchmark_agg.csv"), index=False)

        print("\n[PER-DATASET]")
        run_per_dataset(bench_agg, args.out)
        run_benchmark(bench_agg, args.out, summary_df=summary_df, sample_df=bench_df)

    if args.cross_csvs:
        print("\n[LOADING] Cross-dataset data...")
        cross_df = load_and_combine_csvs(args.cross_csvs, args.cross_filter)
        cross_agg = compute_aggregates(cross_df)
        cross_agg.to_csv(os.path.join(args.out, "data", "cross_agg.csv"), index=False)

        if args.benchmark_csvs:
            existing = set(bench_agg["dataset"].unique())
            new_ds = [d for d in cross_agg["dataset"].unique() if d not in existing]
            if new_ds:
                print(f"\n[PER-DATASET] New: {new_ds}")
                new_agg = cross_agg[cross_agg["dataset"].isin(new_ds)]
                run_per_dataset(new_agg, args.out)
        else:
            print("\n[PER-DATASET]")
            run_per_dataset(cross_agg, args.out)

        run_cross_dataset(cross_agg, args.out, reference=args.reference)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()