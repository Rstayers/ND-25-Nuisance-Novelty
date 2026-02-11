#!/usr/bin/env python3
# analysis/run_report.py

import argparse
import os

from analysis.processing import load_and_combine_csvs, compute_aggregates
from analysis.plots import (
    plot_per_dataset_all,
    plot_benchmark_all,
    plot_cross_dataset_all,
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
)


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


def run_benchmark(agg_df, out_dir):
    """Benchmark comparison figures and tables."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    fig_dir = os.path.join(out_dir, "figures", "benchmark")
    table_dir = os.path.join(out_dir, "tables", "benchmark")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    plot_benchmark_all(agg_df, fig_dir)

    print("\n  Tables:")
    generate_nnr_at_severity_table(agg_df, table_dir, severity=5)
    generate_osa_at_severity_table(agg_df, table_dir, severity=5)  # NEW
    generate_adr_benchmark_table(agg_df, table_dir)
    generate_benchmark_summary_table(agg_df, table_dir)
    generate_benchmark_osa_table(agg_df, table_dir)  # NEW
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="CSA")
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="NNR")
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="OSA")  # NEW
    generate_benchmark_by_severity_table(agg_df, table_dir, metric="OSA_Gap")


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

    parser.add_argument("--benchmark_csvs", nargs="+", default=None)
    parser.add_argument("--benchmark_filter", nargs="+", default=None)
    parser.add_argument("--cross_csvs", nargs="+", default=None)
    parser.add_argument("--cross_filter", nargs="+", default=None)
    parser.add_argument("--reference", default="ImageNet-LN")
    parser.add_argument("--out", default="paper_output")

    args = parser.parse_args()

    if args.benchmark_csvs is None and args.cross_csvs is None:
        parser.error("Must provide --benchmark_csvs and/or --cross_csvs")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)

    print("=" * 60)
    print("NUISANCE NOVELTY ANALYSIS")
    print("=" * 60)

    if args.benchmark_csvs:
        print("\n[LOADING] Benchmark data...")
        bench_df = load_and_combine_csvs(args.benchmark_csvs, args.benchmark_filter)
        bench_agg = compute_aggregates(bench_df)
        bench_agg.to_csv(os.path.join(args.out, "data", "benchmark_agg.csv"), index=False)

        print("\n[PER-DATASET]")
        run_per_dataset(bench_agg, args.out)
        run_benchmark(bench_agg, args.out)

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