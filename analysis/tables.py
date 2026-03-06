# analysis/tables.py
# Paper-ready tables - ZERO aggregation across backbone or detector
#
# Every table shows ALL (backbone, detector) configurations.
# NEW: OSA tables alongside existing NNR/CCR tables

import os
import numpy as np
import pandas as pd
from analysis.processing import (
    get_metrics_by_severity,
    get_mean_metrics,
    compute_adr,
    OUTCOME_COLS
)


def _fmt(x, decimals=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def _save_latex(df, path, caption="", label=""):
    latex = df.to_latex(index=False, escape=False,
                        column_format="l" + "c" * (len(df.columns) - 1))
    full = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
{latex}}}%
\\end{{table}}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"  > {os.path.basename(path)}")


# =============================================================================
# PER-DATASET TABLES
# =============================================================================

def generate_dataset_summary_table(agg_df, dataset, out_dir):
    """
    Heatmap-style table: detector × backbone with NNR values.
    Generates separate tables for each OOD test dataset since all metrics
    depend on the threshold calibration which varies by OOD dataset.
    """
    mean_df = get_mean_metrics(agg_df, dataset)
    if mean_df.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df
            ood_suffix = ""

        if ood_data.empty:
            continue

        if "backbone" not in ood_data.columns or len(ood_data["backbone"].unique()) < 2:
            # Single backbone - just list detectors
            cols_to_use = ["detector", "CSA", "CCR", "NNR", "OSA_Gap"]
            if "OSA" in ood_data.columns and not ood_data["OSA"].isna().all():
                cols_to_use.insert(2, "OSA")

            result = ood_data[[c for c in cols_to_use if c in ood_data.columns]].copy()
            result = result.rename(columns={"detector": "Detector", "OSA_Gap": "Gap"})

            for col in result.columns:
                if col != "Detector":
                    result[col] = result[col].apply(lambda x: _fmt(x, 2))
            result = result.sort_values("NNR")

            safe = dataset.replace("-", "_")
            result.to_csv(os.path.join(out_dir, f"summary_{safe}{ood_suffix}.csv"), index=False)
            ood_label = f" ({ood_ds})" if ood_ds else ""
            _save_latex(result, os.path.join(out_dir, f"summary_{safe}{ood_suffix}.tex"),
                        caption=f"Results on {dataset}{ood_label}.", label=f"tab:{safe.lower()}{ood_suffix}")
        else:
            # Multiple backbones - pivot table
            pivot = ood_data.pivot(index="detector", columns="backbone", values="NNR")
            pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
            pivot = pivot.reset_index()
            pivot = pivot.rename(columns={"detector": "Detector"})

            for col in pivot.columns:
                if col != "Detector":
                    pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

            safe = dataset.replace("-", "_")
            pivot.to_csv(os.path.join(out_dir, f"NNR_matrix_{safe}{ood_suffix}.csv"), index=False)
            ood_label = f" ({ood_ds})" if ood_ds else ""
            _save_latex(pivot, os.path.join(out_dir, f"NNR_matrix_{safe}{ood_suffix}.tex"),
                        caption=f"NNR (Detector × Backbone) on {dataset}{ood_label}.",
                        label=f"tab:nnr_matrix_{safe.lower()}{ood_suffix}")


def generate_full_results_table(agg_df, dataset, out_dir):
    """
    Full table: every (backbone, detector) row with all metrics.
    Now includes OSA (averaged over OOD datasets if multiple present).
    """
    mean_df = get_mean_metrics(agg_df, dataset)
    if mean_df.empty:
        return

    # Add ADR
    adr_df = compute_adr(agg_df[agg_df["dataset"] == dataset] if "dataset" in agg_df.columns else agg_df)
    if not adr_df.empty:
        merge_cols = [c for c in ["backbone", "detector"] if c in mean_df.columns]
        mean_df = mean_df.merge(adr_df[merge_cols + ["ADR"]], on=merge_cols, how="left")

    # Build column list with OSA
    cols = ["backbone", "detector", "CSA", "CCR", "NNR", "OSA_Gap"]
    if "OSA" in mean_df.columns and not mean_df["OSA"].isna().all():
        cols.insert(3, "OSA")  # After CCR
    if "ADR" in mean_df.columns:
        cols.append("ADR")
    cols = [c for c in cols if c in mean_df.columns]

    result = mean_df[cols].copy()
    result = result.rename(columns={"backbone": "Backbone", "detector": "Detector", "OSA_Gap": "Gap"})

    for col in result.columns:
        if col not in ["Backbone", "Detector"]:
            result[col] = result[col].apply(lambda x: _fmt(x, 2))

    result = result.sort_values(["Backbone", "NNR"])

    safe = dataset.replace("-", "_")
    result.to_csv(os.path.join(out_dir, f"full_{safe}.csv"), index=False)
    _save_latex(result, os.path.join(out_dir, f"full_{safe}.tex"),
                caption=f"Full results on {dataset}.", label=f"tab:full_{safe.lower()}")


# =============================================================================
# BENCHMARK COMPARISON TABLES
# =============================================================================

def generate_nnr_at_severity_table(agg_df, out_dir, severity=5):
    """
    Table: NNR@L5 with rows=(backbone, detector), cols=datasets.
    Grouped by backbone with cmidrule separators. Bold max per row.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_data = per_sev[per_sev["level"] == severity]
    if sev_data.empty:
        print(f"  > NNR@L{severity} table skipped (no data)")
        return

    sev_data = sev_data.copy()

    pivot = sev_data.pivot_table(index=["backbone", "detector"], columns="dataset", values="NNR")

    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    cols = [c for c in desired_order if c in pivot.columns] + [c for c in pivot.columns if c not in desired_order]
    pivot = pivot[cols]

    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, f"NNR_L{severity}_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, f"NNR_L{severity}_benchmark.tex"),
        caption=f"NNR at severity {severity} across benchmarks. \\textbf{{Bold}} indicates highest NNR per configuration (row).",
        label=f"tab:nnr_l{severity}_benchmark",
        fmt=".2f"
    )


def generate_osa_at_severity_table(agg_df, out_dir, severity=5):
    """
    Table: OSA@L5 with rows=(backbone, detector), cols=datasets.
    NEW: Mirrors NNR table but for OSA.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_data = per_sev[per_sev["level"] == severity]
    if sev_data.empty or "OSA" not in sev_data.columns or sev_data["OSA"].isna().all():
        print(f"  > OSA@L{severity} table skipped (no data)")
        return

    sev_data = sev_data.copy()

    pivot = sev_data.pivot_table(index=["backbone", "detector"], columns="dataset", values="OSA")

    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    cols = [c for c in desired_order if c in pivot.columns] + [c for c in pivot.columns if c not in desired_order]
    pivot = pivot[cols]

    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, f"OSA_L{severity}_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, f"OSA_L{severity}_benchmark.tex"),
        caption=f"OSA at severity {severity} across benchmarks. \\textbf{{Bold}} indicates highest OSA per configuration (row).",
        label=f"tab:osa_l{severity}_benchmark",
        fmt=".2f",
        bold_max=True  # Higher OSA is better
    )


def generate_adr_benchmark_table(agg_df, out_dir):
    """
    Table: ADR with rows=(backbone, detector), cols=datasets.
    Grouped by backbone with cmidrule separators. Bold max per row.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR benchmark table skipped (no data)")
        return

    adr_df = adr_df.copy()

    pivot = adr_df.pivot_table(index=["backbone", "detector"], columns="dataset", values="ADR")

    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    cols = [c for c in desired_order if c in pivot.columns] + [c for c in pivot.columns if c not in desired_order]
    pivot = pivot[cols]

    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, "ADR_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, "ADR_benchmark.tex"),
        caption="Accuracy Divergence Rate (ADR) across benchmarks. \\textbf{Bold} indicates highest ADR per configuration (row).",
        label="tab:adr_benchmark",
        fmt=".3f"
    )


def _save_grouped_latex(pivot, path, caption="", label="", fmt=".2f", bold_max=True):
    """
    Save table with rows=(backbone, detector) grouped by backbone.
    Columns = datasets. Bold max (if bold_max=True) or min value per row.
    """
    datasets = list(pivot.columns)
    backbones = pivot.index.get_level_values(0).unique()

    n_ds = len(datasets)

    # Build LaTeX
    header = " & ".join(["Backbone", "Detector"] + datasets) + " \\\\"
    col_fmt = "ll" + "c" * n_ds

    rows = []
    for bb_idx, bb in enumerate(backbones):
        bb_data = pivot.loc[bb]

        for det_idx, (det, row) in enumerate(bb_data.iterrows()):
            vals = row.values.astype(float)
            valid_mask = ~np.isnan(vals)

            if valid_mask.any():
                if bold_max:
                    best_idx = np.nanargmax(vals)  # Higher is better
                else:
                    best_idx = np.nanargmin(vals)  # Lower is better

            formatted = []
            for i, v in enumerate(vals):
                if np.isnan(v):
                    formatted.append("—")
                else:
                    s = f"{v:{fmt}}"
                    if valid_mask.any() and i == best_idx:
                        s = f"\\textbf{{{s}}}"
                    formatted.append(s)

            bb_str = bb if det_idx == 0 else ""
            row_str = " & ".join([bb_str, det] + formatted) + " \\\\"
            rows.append(row_str)

        # Add midrule after each backbone group (except last)
        if bb_idx < len(backbones) - 1:
            rows.append("\\midrule")

    body = "\n".join(rows)

    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}%
\\end{{table}}
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  > {os.path.basename(path)}")


def generate_benchmark_summary_table(agg_df, out_dir):
    """
    Table: (backbone+detector) × benchmark with NNR.
    Shows every configuration's NNR on every benchmark.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values="NNR")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

        pivot.to_csv(os.path.join(out_dir, f"benchmark_NNR{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"benchmark_NNR{ood_suffix}.tex"),
                    caption=f"NNR per configuration across benchmarks ({ood_ds or 'all'}).",
                    label=f"tab:benchmark_nnr{ood_suffix.lower()}")


def generate_benchmark_osa_table(agg_df, out_dir):
    """
    Table: (backbone+detector) × benchmark with OSA.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if "OSA" not in mean_df.columns or mean_df["OSA"].isna().all():
        print("  > benchmark_OSA.csv skipped (no OSA data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values="OSA")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]  # Higher OSA at top
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

        pivot.to_csv(os.path.join(out_dir, f"benchmark_OSA{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"benchmark_OSA{ood_suffix}.tex"),
                    caption=f"OSA per configuration across benchmarks ({ood_ds or 'all'}).",
                    label=f"tab:benchmark_osa{ood_suffix.lower()}")


def generate_benchmark_by_severity_table(agg_df, out_dir, metric="NNR"):
    """
    Table: (backbone+detector) × severity for each benchmark.
    Generates SEPARATE TABLE FOR EACH DATASET and OOD test dataset.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]

    if metric not in per_sev.columns or per_sev[metric].isna().all():
        print(f"  > {metric}_by_severity tables skipped (no data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in per_sev.columns:
        ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    datasets = sorted(per_sev["dataset"].unique())

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in per_sev.columns:
            ood_data = per_sev[per_sev["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = per_sev
            ood_suffix = ""

        for dataset in datasets:
            ds_data = ood_data[ood_data["dataset"] == dataset].copy()
            if ds_data.empty:
                continue

            ds_data["Config"] = ds_data["backbone"] + " + " + ds_data["detector"]

            pivot = ds_data.pivot(index="Config", columns="level", values=metric)
            pivot.columns = [f"L{int(c)}" for c in pivot.columns]
            pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
            pivot = pivot.reset_index()

            for col in pivot.columns:
                if col != "Config":
                    pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

            safe = dataset.replace("-", "_")
            pivot.to_csv(os.path.join(out_dir, f"{metric}_by_severity_{safe}{ood_suffix}.csv"), index=False)
            _save_latex(pivot, os.path.join(out_dir, f"{metric}_by_severity_{safe}{ood_suffix}.tex"),
                        caption=f"{metric} by severity on {dataset} ({ood_ds or 'all'}).",
                        label=f"tab:{metric.lower()}_severity_{safe.lower()}{ood_suffix.lower()}")


# =============================================================================
# CROSS-DATASET TABLES
# =============================================================================

def generate_cross_dataset_summary(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with NNR.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values="NNR")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

        pivot.to_csv(os.path.join(out_dir, f"cross_dataset_NNR{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"cross_dataset_NNR{ood_suffix}.tex"),
                    caption=f"NNR per configuration across datasets ({ood_ds or 'all'}).",
                    label=f"tab:cross_nnr{ood_suffix.lower()}")


def generate_cross_dataset_osa(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with OSA.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if "OSA" not in mean_df.columns or mean_df["OSA"].isna().all():
        print("  > cross_dataset_OSA.csv skipped (no OSA data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values="OSA")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

        pivot.to_csv(os.path.join(out_dir, f"cross_dataset_OSA{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"cross_dataset_OSA{ood_suffix}.tex"),
                    caption=f"OSA per configuration across datasets ({ood_ds or 'all'}).",
                    label=f"tab:cross_osa{ood_suffix.lower()}")


def generate_cross_dataset_adr(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with ADR.
    Generates separate tables for each OOD test dataset.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR table skipped (no data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in adr_df.columns:
        ood_datasets = sorted(adr_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in adr_df.columns:
            ood_data = adr_df[adr_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = adr_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values="ADR")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 3))

        pivot.to_csv(os.path.join(out_dir, f"cross_dataset_ADR{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"cross_dataset_ADR{ood_suffix}.tex"),
                    caption=f"ADR per configuration across datasets ({ood_ds or 'all'}).",
                    label=f"tab:cross_adr{ood_suffix.lower()}")


def generate_cross_dataset_full(agg_df, out_dir):
    """
    Full table: every (backbone, detector, dataset) row.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    adr_df = compute_adr(agg_df)

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        # Merge ADR if available
        if not adr_df.empty:
            if ood_ds and "test_ood_dataset" in adr_df.columns:
                adr_filtered = adr_df[adr_df["test_ood_dataset"] == ood_ds]
            else:
                adr_filtered = adr_df
            merge_cols = [c for c in ["backbone", "detector", "dataset"] if c in ood_data.columns]
            ood_data = ood_data.merge(adr_filtered[merge_cols + ["ADR"]], on=merge_cols, how="left")

        cols = ["dataset", "backbone", "detector", "CSA", "CCR", "NNR", "OSA_Gap"]
        if "OSA" in ood_data.columns and not ood_data["OSA"].isna().all():
            cols.insert(5, "OSA")
        if "ADR" in ood_data.columns:
            cols.append("ADR")
        cols = [c for c in cols if c in ood_data.columns]

        result = ood_data[cols].copy()
        result = result.rename(columns={"dataset": "Dataset", "backbone": "Backbone",
                                        "detector": "Detector", "OSA_Gap": "Gap"})

        for col in result.columns:
            if col not in ["Dataset", "Backbone", "Detector"]:
                result[col] = result[col].apply(lambda x: _fmt(x, 2))

        result = result.sort_values(["Dataset", "Backbone", "NNR"])

        result.to_csv(os.path.join(out_dir, f"full_results{ood_suffix}.csv"), index=False)
        _save_latex(result, os.path.join(out_dir, f"full_results{ood_suffix}.tex"),
                    caption=f"Full results ({ood_ds or 'all'}).", label=f"tab:full{ood_suffix.lower()}")


def generate_rank_comparison_table(agg_df, out_dir, metric="NNR"):
    """
    Table: configuration ranks per dataset.
    Generates separate tables for each OOD test dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"] + " + " + ood_data["detector"]

        # Compute ranks per dataset
        rankings = []
        ascending = True if metric in ["NNR", "OSA_Gap", "ADR"] else False  # Lower is better for NNR

        for ds in ood_data["dataset"].unique():
            ds_data = ood_data[ood_data["dataset"] == ds].copy()
            ds_data = ds_data.sort_values(metric, ascending=ascending)
            ds_data["rank"] = range(1, len(ds_data) + 1)
            rankings.append(ds_data[["Config", "dataset", "rank"]])

        all_ranks = pd.concat(rankings, ignore_index=True)
        pivot = all_ranks.pivot(index="Config", columns="dataset", values="rank")
        pivot["Mean Rank"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("Mean Rank")
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col not in ["Config", "Mean Rank"]:
                pivot[col] = pivot[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
        pivot["Mean Rank"] = pivot["Mean Rank"].apply(lambda x: f"{x:.1f}")

        pivot.to_csv(os.path.join(out_dir, f"ranks_{metric}{ood_suffix}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"ranks_{metric}{ood_suffix}.tex"),
                    caption=f"Configuration rankings by {metric} ({ood_ds or 'all'}).",
                    label=f"tab:ranks_{metric.lower()}{ood_suffix.lower()}")


def generate_outcome_table(agg_df, out_dir):
    """
    Outcome distribution for each (backbone, detector, dataset).
    """
    if not all(c in agg_df.columns for c in OUTCOME_COLS):
        return

    corrupted = agg_df[agg_df["level"] > 0].copy()
    if corrupted.empty:
        return

    group_cols = [c for c in ["backbone", "detector", "dataset"] if c in corrupted.columns]
    grouped = corrupted.groupby(group_cols)[OUTCOME_COLS].sum().reset_index()
    grouped["N_total"] = grouped[OUTCOME_COLS].sum(axis=1)

    for col in OUTCOME_COLS:
        grouped[f"{col}_%"] = (grouped[col] / grouped["N_total"] * 100).apply(lambda x: f"{x:.1f}%")

    cols = group_cols + [f"{c}_%" for c in OUTCOME_COLS]
    result = grouped[cols].copy()

    result = result.rename(columns={
        "backbone": "Backbone", "detector": "Detector", "dataset": "Dataset",
        "Clean_Success_%": "CS%", "Nuisance_Novelty_%": "NN%",
        "Double_Failure_%": "DF%", "Contained_Misidentification_%": "CM%"
    })

    result.to_csv(os.path.join(out_dir, "outcomes.csv"), index=False)
    _save_latex(result, os.path.join(out_dir, "outcomes.tex"),
                caption="Outcome distribution.", label="tab:outcomes")


# =============================================================================
# OOD DETECTION METRIC TABLES (from summary data)
# =============================================================================

def generate_auoscr_table(summary_df, out_dir):
    """
    Table: AUOSCR with rows=(backbone, detector), cols=datasets.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "AUOSCR" not in ood_df.columns:
        print("  > AUOSCR table skipped (no data)")
        return

    # Average over OOD datasets if multiple
    if "ood_test_dataset" in ood_df.columns or "test_ood_dataset" in ood_df.columns:
        ood_df = ood_df.groupby(["backbone", "detector", "dataset"])["AUOSCR"].mean().reset_index()

    pivot = ood_df.pivot_table(index=["backbone", "detector"], columns="dataset", values="AUOSCR")
    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, "AUOSCR_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, "AUOSCR_benchmark.tex"),
        caption="AUOSCR across benchmarks. \\textbf{Bold} indicates highest AUOSCR per configuration.",
        label="tab:auoscr_benchmark",
        fmt=".3f",
        bold_max=True  # Higher is better
    )


def generate_auroc_table(summary_df, out_dir):
    """
    Table: AUROC with rows=(backbone, detector), cols=datasets.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "AUROC" not in ood_df.columns:
        print("  > AUROC table skipped (no data)")
        return

    if "ood_test_dataset" in ood_df.columns or "test_ood_dataset" in ood_df.columns:
        ood_df = ood_df.groupby(["backbone", "detector", "dataset"])["AUROC"].mean().reset_index()

    pivot = ood_df.pivot_table(index=["backbone", "detector"], columns="dataset", values="AUROC")
    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, "AUROC_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, "AUROC_benchmark.tex"),
        caption="AUROC (OOD Detection) across benchmarks. \\textbf{Bold} indicates highest AUROC per configuration.",
        label="tab:auroc_benchmark",
        fmt=".3f",
        bold_max=True  # Higher is better
    )


def generate_fpr95_table(summary_df, out_dir):
    """
    Table: FPR@95% with rows=(backbone, detector), cols=datasets.
    Lower is better, so we use bold_max=False.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "FPR@95TPR" not in ood_df.columns:
        print("  > FPR@95 table skipped (no data)")
        return

    if "ood_test_dataset" in ood_df.columns or "test_ood_dataset" in ood_df.columns:
        ood_df = ood_df.groupby(["backbone", "detector", "dataset"])["FPR@95TPR"].mean().reset_index()

    pivot = ood_df.pivot_table(index=["backbone", "detector"], columns="dataset", values="FPR@95TPR")
    pivot = pivot.sort_index()

    pivot.to_csv(os.path.join(out_dir, "FPR95_benchmark.csv"))

    _save_grouped_latex(
        pivot,
        os.path.join(out_dir, "FPR95_benchmark.tex"),
        caption="FPR@95\\% (OOD Detection) across benchmarks. \\textbf{Bold} indicates lowest FPR per configuration.",
        label="tab:fpr95_benchmark",
        fmt=".3f",
        bold_max=False  # Lower is better - need to update _save_grouped_latex
    )


def generate_ood_metrics_summary_table(summary_df, out_dir):
    """
    Combined table: All OOD metrics (AUOSCR, AUROC, FPR@95) for each (backbone, detector).
    Averaged over datasets.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    required = ["AUOSCR", "AUROC", "FPR@95TPR"]
    if ood_df.empty or not any(m in ood_df.columns for m in required):
        print("  > OOD metrics summary table skipped (no data)")
        return

    # Average over OOD datasets and test datasets
    group_cols = ["backbone", "detector"]
    available_metrics = [m for m in required if m in ood_df.columns]

    summary = ood_df.groupby(group_cols)[available_metrics].mean().reset_index()
    summary = summary.sort_values(["backbone", "detector"])

    # Format values
    for col in available_metrics:
        summary[col] = summary[col].apply(lambda x: _fmt(x, 3))

    summary.to_csv(os.path.join(out_dir, "OOD_metrics_summary.csv"), index=False)

    # Rename columns for cleaner display
    rename_map = {"backbone": "Backbone", "detector": "Detector", "FPR@95TPR": "FPR@95%"}
    summary = summary.rename(columns=rename_map)

    _save_latex(summary, os.path.join(out_dir, "OOD_metrics_summary.tex"),
                caption="OOD detection metrics averaged across datasets.",
                label="tab:ood_metrics_summary")


# =============================================================================
# COSTARR-STYLE OSA TABLE (Backbone × Detector × OOD Dataset)
# =============================================================================

def generate_costarr_style_osa_table(agg_df, out_dir, dataset_filter=None, severity=5):
    """
    COSTARR-style table: OSA at specific severity level.
    Format: Rows grouped by backbone, then detectors.
    Columns: OOD test datasets (NINCO, OpenImage-O, etc.)

    Similar to Table 1 in COSTARR paper.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated data.
    out_dir : str
        Output directory.
    dataset_filter : str, optional
        Filter to specific LN dataset (e.g., "ImageNet-LN").
    severity : int
        Severity level to report (default: 5 for most challenging).
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    # Filter to specific severity level
    per_sev = per_sev[per_sev["level"] == severity]
    if per_sev.empty:
        print(f"  > COSTARR-style OSA table skipped (no L{severity} data)")
        return

    # Filter to specific dataset if specified
    if dataset_filter:
        per_sev = per_sev[per_sev["dataset"] == dataset_filter]

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print("  > COSTARR-style OSA table skipped (no OSA data)")
        return

    # Check if we have multiple OOD datasets
    if "test_ood_dataset" not in per_sev.columns:
        print("  > COSTARR-style OSA table skipped (no test_ood_dataset column)")
        return

    ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    if len(ood_datasets) < 1:
        print("  > COSTARR-style OSA table skipped (no OOD datasets)")
        return

    backbones = sorted(per_sev["backbone"].unique())
    detectors = sorted(per_sev["detector"].unique())

    # Build pivot table: (backbone, detector) × ood_dataset
    pivot = per_sev.pivot_table(
        index=["backbone", "detector"],
        columns="test_ood_dataset",
        values="OSA"
    )

    # Sort columns
    desired_ood_order = ["NINCO", "OpenImage-O-Test", "iNat", "Textures"]
    ood_order = [o for o in desired_ood_order if o in pivot.columns] + \
                [o for o in pivot.columns if o not in desired_ood_order]
    pivot = pivot[ood_order]

    # Build LaTeX with backbone grouping (like COSTARR paper)
    n_ood = len(ood_order)
    header = " & ".join(["Arch", "Method"] + [o.replace("-", "‑").replace("_", " ") for o in ood_order]) + " \\\\"
    col_fmt = "ll" + "c" * n_ood

    rows = []
    for bb_idx, bb in enumerate(backbones):
        try:
            bb_data = pivot.loc[bb]
        except KeyError:
            continue

        for det_idx, det in enumerate(detectors):
            try:
                row = bb_data.loc[det]
            except KeyError:
                continue

            vals = row.values.astype(float)
            valid_mask = ~np.isnan(vals)

            # Bold the maximum value in this row
            if valid_mask.any():
                best_idx = np.nanargmax(vals)
            else:
                best_idx = -1

            formatted = []
            for i, v in enumerate(vals):
                if np.isnan(v):
                    formatted.append("—")
                else:
                    s = f"{v:.3f}"
                    if valid_mask.any() and i == best_idx:
                        s = f"\\textbf{{{s}}}"
                    formatted.append(s)

            bb_str = bb.replace("_", " ") if det_idx == 0 else ""
            row_str = " & ".join([bb_str, det.upper()] + formatted) + " \\\\"
            rows.append(row_str)

        # Add midrule after each backbone group (except last)
        if bb_idx < len(backbones) - 1:
            rows.append("\\midrule")

    body = "\n".join(rows)

    ds_label = dataset_filter.replace("-", "_").lower() if dataset_filter else "all"
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Operational Open-Set Accuracy (OSA) at severity level {severity}. \\textbf{{Bold}} indicates best performance per configuration.}}
\\label{{tab:costarr_osa_L{severity}_{ds_label}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}%
\\end{{table}}
"""

    fname = f"COSTARR_OSA_L{severity}"
    if dataset_filter:
        fname += f"_{dataset_filter.replace('-', '_')}"

    # Save CSV
    pivot_csv = pivot.reset_index()
    pivot_csv.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)

    # Save LaTeX
    with open(os.path.join(out_dir, f"{fname}.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  > {fname}.tex")


def generate_costarr_benchmark_osa_tables(agg_df, out_dir, severity=5):
    """
    COSTARR-style benchmark OSA tables.

    Format similar to COSTARR paper Table 1:
    - Rows grouped by backbone, then detectors
    - Columns: Baseline (ImageNet-Val) + nuisance datasets (LN, C, CNS)
    - Separate tables for each OOD test dataset (NINCO, Open-O)
    - Generates both OSA@L5 and mean OSA tables
    - Bold = highest OSA detector for each backbone

    This creates 4 tables total:
    - OSA@L5 for NINCO
    - OSA@L5 for Open-O
    - Mean OSA for NINCO
    - Mean OSA for Open-O
    """
    # === OSA@L5 Tables ===
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > COSTARR benchmark tables skipped (no data)")
        return

    sev_data = per_sev[per_sev["level"] == severity]

    if sev_data.empty or "OSA" not in sev_data.columns:
        print(f"  > COSTARR benchmark tables skipped (no L{severity} OSA data)")
        return

    if "test_ood_dataset" not in sev_data.columns:
        print("  > COSTARR benchmark tables skipped (no test_ood_dataset)")
        return

    # Get baseline data (ImageNet-Val at level 0)
    baseline_data = per_sev[(per_sev["dataset"] == "ImageNet-Val") & (per_sev["level"] == 0)]

    ood_datasets = sorted(sev_data["test_ood_dataset"].unique())
    datasets = sorted(sev_data["dataset"].unique())

    # Desired dataset order (excluding ImageNet-Val which becomes baseline)
    desired_ds_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_ds_order if d in datasets] + \
               [d for d in datasets if d not in desired_ds_order]

    for ood_ds in ood_datasets:
        ood_data = sev_data[sev_data["test_ood_dataset"] == ood_ds]
        ood_safe = ood_ds.replace("-", "_")

        # Get baseline for this OOD dataset
        if not baseline_data.empty and "test_ood_dataset" in baseline_data.columns:
            baseline_ood = baseline_data[baseline_data["test_ood_dataset"] == ood_ds]
        else:
            baseline_ood = baseline_data

        _generate_costarr_table_inner(
            ood_data, ds_order, out_dir,
            metric="OSA",
            title=f"OSA@L{severity}",
            fname=f"COSTARR_OSA_L{severity}_{ood_safe}",
            ood_label=ood_ds,
            baseline_df=baseline_ood
        )

    # === Mean OSA Tables ===
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty or "OSA" not in mean_df.columns:
        return

    datasets = sorted(mean_df["dataset"].unique())
    ds_order = [d for d in desired_ds_order if d in datasets] + \
               [d for d in datasets if d not in desired_ds_order]

    # Baseline for mean: use ImageNet-Val at level 0 (same as L5 tables)
    # Note: mean_df won't have ImageNet-Val since it only has levels 1-5
    # So we reuse baseline_data from per_sev

    for ood_ds in ood_datasets:
        if "test_ood_dataset" not in mean_df.columns:
            ood_data = mean_df
        else:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds]

        if ood_data.empty:
            continue

        # Get baseline for this OOD dataset (from level 0 data)
        if not baseline_data.empty and "test_ood_dataset" in baseline_data.columns:
            baseline_ood = baseline_data[baseline_data["test_ood_dataset"] == ood_ds]
        else:
            baseline_ood = baseline_data

        ood_safe = ood_ds.replace("-", "_")

        _generate_costarr_table_inner(
            ood_data, ds_order, out_dir,
            metric="OSA",
            title="Mean OSA",
            fname=f"COSTARR_OSA_mean_{ood_safe}",
            ood_label=ood_ds,
            baseline_df=baseline_ood
        )


def _generate_costarr_table_inner(df, ds_order, out_dir, metric, title, fname, ood_label, baseline_df=None):
    """
    Internal helper to generate a single COSTARR-style table.

    Args:
        df: DataFrame with nuisance dataset results
        ds_order: Order of dataset columns
        baseline_df: Optional DataFrame with ImageNet-Val baseline results
        Bold = highest OSA per column (best detector-backbone pair for each dataset)
    """
    if df.empty or metric not in df.columns:
        return

    backbones = sorted(df["backbone"].unique())
    detectors = sorted(df["detector"].unique())

    # Build pivot: (backbone, detector) × dataset
    pivot = df.pivot_table(
        index=["backbone", "detector"],
        columns="dataset",
        values=metric
    )

    # Filter and order columns
    ds_cols = [d for d in ds_order if d in pivot.columns]
    if not ds_cols:
        return
    pivot = pivot[ds_cols]

    # Add baseline column if available
    has_baseline = False
    if baseline_df is not None and not baseline_df.empty and metric in baseline_df.columns:
        baseline_pivot = baseline_df.pivot_table(
            index=["backbone", "detector"],
            values=metric
        )
        if not baseline_pivot.empty:
            pivot.insert(0, "Baseline", baseline_pivot[metric])
            has_baseline = True

    # Find the best detector per (backbone, column) for bolding
    # This gives 5 bolded values per column (one per backbone)
    best_det_per_bb_col = {}  # (backbone, col) -> best detector
    for bb in backbones:
        try:
            bb_data = pivot.loc[bb]
            if isinstance(bb_data, pd.Series):
                # Only one detector - it's the best by default
                for col in pivot.columns:
                    best_det_per_bb_col[(bb, col)] = bb_data.name if hasattr(bb_data, 'name') else detectors[0]
            else:
                # Multiple detectors - find best per column
                for col in bb_data.columns:
                    col_vals = bb_data[col]
                    if col_vals.notna().any():
                        best_det_per_bb_col[(bb, col)] = col_vals.idxmax()
        except KeyError:
            pass

    # Build LaTeX with backbone grouping
    all_cols = ["Baseline"] + ds_cols if has_baseline else ds_cols
    n_cols = len(all_cols)

    # Full column names mapping
    col_name_map = {
        "Baseline": "ImageNet Val",
        "ImageNet-LN": "ImageNet-LN",
        "ImageNet-C": "ImageNet-C",
        "CNS": "CNS-Bench",
    }
    col_names = [col_name_map.get(d, d) for d in all_cols]
    header = " & ".join(["Arch", "Method"] + col_names) + " \\\\"
    col_fmt = "ll" + "c" * n_cols

    rows = []
    for bb_idx, bb in enumerate(backbones):
        try:
            bb_data = pivot.loc[bb]
        except KeyError:
            continue

        for det_idx, det in enumerate(detectors):
            try:
                if isinstance(bb_data, pd.Series):
                    row = bb_data
                else:
                    row = bb_data.loc[det]
            except KeyError:
                continue

            formatted = []
            for col_idx, col in enumerate(all_cols):
                v = row.iloc[col_idx] if col_idx < len(row) else np.nan
                if np.isnan(v):
                    formatted.append("—")
                else:
                    s = f"{v:.3f}"
                    # Bold if this detector is the best for this (backbone, column)
                    if (bb, col) in best_det_per_bb_col and best_det_per_bb_col[(bb, col)] == det:
                        s = f"\\textbf{{{s}}}"
                    formatted.append(s)

            # Short backbone name (only on first row of group)
            bb_map = {
                "resnet50": "RN50",
                "vit_b_16": "ViT-B",
                "convnext_t": "CNX-T",
                "densenet121": "DN121",
                "swin_t": "Swin-T",
            }
            bb_short = bb_map.get(bb, bb.replace("_", ""))
            bb_str = bb_short if det_idx == 0 else ""
            row_str = " & ".join([bb_str, det.upper()] + formatted) + " \\\\"
            rows.append(row_str)

        # Add midrule after each backbone group (except last)
        if bb_idx < len(backbones) - 1:
            rows.append("\\midrule")

    body = "\n".join(rows)

    latex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{{title} across nuisance benchmarks (OOD: {ood_label}). \\textbf{{Bold}} = best detector per architecture.}}
\\label{{tab:{fname.lower()}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save CSV
    pivot_csv = pivot.reset_index()
    pivot_csv.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)

    # Save LaTeX
    with open(os.path.join(out_dir, f"{fname}.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  > {fname}.tex")


def _generate_severity_breakdown_table(df, ood_datasets, out_dir, dataset_name):
    """
    Generate severity breakdown table showing L1-L5 for a specific nuisance dataset.
    Side-by-side columns: NINCO (L1-L5) | Open-O (L1-L5)

    Args:
        df: DataFrame from get_metrics_by_severity (has level column)
        ood_datasets: List of OOD dataset names [NINCO, Open-O, ...]
        out_dir: Output directory
        dataset_name: Name of the nuisance dataset (e.g., "ImageNet-LN")
    """
    if df.empty or "OSA" not in df.columns:
        return

    # Filter to this dataset and levels 1-5
    ds_data = df[(df["dataset"] == dataset_name) & (df["level"] > 0) & (df["level"] <= 5)]
    if ds_data.empty:
        print(f"  > Severity breakdown for {dataset_name} skipped (no data)")
        return

    if "test_ood_dataset" not in ds_data.columns:
        print(f"  > Severity breakdown for {dataset_name} skipped (no test_ood_dataset)")
        return

    backbones = sorted(ds_data["backbone"].unique())
    detectors = sorted(ds_data["detector"].unique())

    # Get the two OOD datasets
    ood1 = ood_datasets[0]  # NINCO
    ood2 = ood_datasets[1] if len(ood_datasets) > 1 else None  # Open-O

    # Build pivots for each OOD dataset
    def build_pivot(ood_ds):
        ood_data = ds_data[ds_data["test_ood_dataset"] == ood_ds]
        pivot = ood_data.pivot_table(
            index=["backbone", "detector"],
            columns="level",
            values="OSA"
        )
        return pivot

    pivot1 = build_pivot(ood1)
    pivot2 = build_pivot(ood2) if ood2 else None

    if pivot1.empty:
        print(f"  > Severity breakdown for {dataset_name} skipped (empty pivot)")
        return

    # Find best detector per (backbone, level) for bolding
    def get_best_per_bb_level(pivot):
        best = {}
        for bb in backbones:
            try:
                bb_data = pivot.loc[bb]
                if not isinstance(bb_data, pd.Series):
                    for level in bb_data.columns:
                        if bb_data[level].notna().any():
                            best[(bb, level)] = bb_data[level].idxmax()
            except KeyError:
                pass
        return best

    best1 = get_best_per_bb_level(pivot1)
    best2 = get_best_per_bb_level(pivot2) if pivot2 is not None else {}

    # Backbone and detector names
    bb_map = {"resnet50": "RN50", "vit_b_16": "ViT-B", "convnext_t": "CNX-T", "densenet121": "DN121", "swin_t": "Swin-T"}

    # OOD short names
    ood1_short = "NINCO" if "NINCO" in ood1 else ood1.replace("-", "")[:8]
    ood2_short = "Open-O" if ood2 and "Open" in ood2 else (ood2.replace("-", "")[:8] if ood2 else "")

    # Build header
    levels = sorted([l for l in pivot1.columns if 1 <= l <= 5])
    n_levels = len(levels)
    level_headers = [f"L{int(l)}" for l in levels]

    header1 = " & ".join(level_headers)
    header2 = " & ".join(level_headers) if pivot2 is not None else ""

    # Multi-column header for OOD dataset names
    mc1 = f"\\multicolumn{{{n_levels}}}{{c}}{{{ood1_short}}}"
    mc2 = f"\\multicolumn{{{n_levels}}}{{c}}{{{ood2_short}}}" if pivot2 is not None else ""

    col_fmt = "ll" + "c" * n_levels + ("|" + "c" * n_levels if pivot2 is not None else "")

    rows = []
    for bb_idx, bb in enumerate(backbones):
        for det_idx, det in enumerate(detectors):
            # Get values from both pivots
            def get_row_vals(pivot, levels, best_dict, bb, det):
                vals = []
                try:
                    bb_data = pivot.loc[bb]
                    row = bb_data.loc[det] if not isinstance(bb_data, pd.Series) else bb_data
                    for level in levels:
                        v = row.get(level, np.nan) if isinstance(row, pd.Series) else (row[level] if level in row.index else np.nan)
                        if pd.isna(v):
                            vals.append("—")
                        else:
                            s = f"{v:.3f}"
                            if (bb, level) in best_dict and best_dict[(bb, level)] == det:
                                s = f"\\textbf{{{s}}}"
                            vals.append(s)
                except KeyError:
                    vals = ["—"] * len(levels)
                return vals

            vals1 = get_row_vals(pivot1, levels, best1, bb, det)
            vals2 = get_row_vals(pivot2, levels, best2, bb, det) if pivot2 is not None else []

            bb_str = bb_map.get(bb, bb) if det_idx == 0 else ""
            row_str = " & ".join([bb_str, det.upper()] + vals1 + vals2) + " \\\\"
            rows.append(row_str)

        if bb_idx < len(backbones) - 1:
            rows.append("\\midrule")

    body = "\n".join(rows)

    ds_safe = dataset_name.replace("-", "_")
    fname = f"severity_breakdown_{ds_safe}"

    latex = f"""\\begin{{table*}}[t]
\\centering
\\small
\\caption{{OSA by severity level (L1-L5) on {dataset_name}. Left: {ood1_short}, Right: {ood2_short}. \\textbf{{Bold}} = best detector per architecture at each severity.}}
\\label{{tab:{fname.lower()}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
 & & {mc1} & {mc2} \\\\
\\cmidrule(lr){{3-{2+n_levels}}} \\cmidrule(lr){{{3+n_levels}-{2+n_levels*2}}}
Arch & Method & {header1} & {header2} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    with open(os.path.join(out_dir, f"{fname}.tex"), "w") as f:
        f.write(latex)
    print(f"  > final_tables/{fname}.tex")

    # Also save as CSV
    csv_rows = []
    for bb in backbones:
        for det in detectors:
            row_data = {"Arch": bb_map.get(bb, bb), "Method": det.upper()}

            def get_csv_vals(pivot, levels, bb, det, prefix):
                try:
                    bb_data = pivot.loc[bb]
                    row = bb_data.loc[det] if not isinstance(bb_data, pd.Series) else bb_data
                    for level in levels:
                        v = row.get(level, np.nan) if isinstance(row, pd.Series) else (row[level] if level in row.index else np.nan)
                        col_name = f"{prefix}_L{int(level)}"
                        row_data[col_name] = f"{v:.3f}" if not pd.isna(v) else ""
                except KeyError:
                    for level in levels:
                        col_name = f"{prefix}_L{int(level)}"
                        row_data[col_name] = ""

            get_csv_vals(pivot1, levels, bb, det, ood1_short)
            if pivot2 is not None:
                get_csv_vals(pivot2, levels, bb, det, ood2_short)
            csv_rows.append(row_data)

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)
    print(f"  > final_tables/{fname}.csv")


def generate_final_tables(agg_df, out_dir, severity=5):
    """
    Generate final paper-ready tables in a dedicated folder.

    Creates:
    1. System Selection table (compact: best detector per backbone per dataset)
    2. Side-by-side COSTARR table (NINCO | Open-O columns)
    3. Severity breakdown tables (L1-L5) for each nuisance dataset
    """
    from analysis.processing import get_metrics_by_severity, get_mean_metrics

    # Create output directory
    final_dir = os.path.join(out_dir, "final_tables")
    os.makedirs(final_dir, exist_ok=True)

    # Get data
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Final tables skipped (no data)")
        return

    sev_data = per_sev[per_sev["level"] == severity]
    baseline_data = per_sev[(per_sev["dataset"] == "ImageNet-Val") & (per_sev["level"] == 0)]
    mean_df = get_mean_metrics(agg_df)

    if "test_ood_dataset" not in sev_data.columns:
        print("  > Final tables skipped (no test_ood_dataset)")
        return

    ood_datasets = sorted(sev_data["test_ood_dataset"].unique())

    # 1. System Selection tables (one per OOD dataset) for ALL severity levels L1-L5
    for ood_ds in ood_datasets:
        ood_safe = ood_ds.replace("-", "_")

        # All severity levels L1-L5
        for sev_level in range(1, 6):
            level_data = per_sev[per_sev["level"] == sev_level]
            if not level_data.empty:
                _generate_system_selection_table(
                    level_data[level_data["test_ood_dataset"] == ood_ds],
                    baseline_data[baseline_data["test_ood_dataset"] == ood_ds] if "test_ood_dataset" in baseline_data.columns else baseline_data,
                    final_dir, f"OSA@L{sev_level}", f"system_selection_L{sev_level}_{ood_safe}", ood_ds
                )

        # Mean version
        if not mean_df.empty and "OSA" in mean_df.columns:
            mean_ood = mean_df[mean_df["test_ood_dataset"] == ood_ds] if "test_ood_dataset" in mean_df.columns else mean_df
            _generate_system_selection_table(
                mean_ood, baseline_data[baseline_data["test_ood_dataset"] == ood_ds] if "test_ood_dataset" in baseline_data.columns else baseline_data,
                final_dir, "Mean OSA", f"system_selection_mean_{ood_safe}", ood_ds
            )

    # 2. Side-by-side COSTARR table (NINCO vs Open-O) for ALL severity levels L1-L5
    if len(ood_datasets) >= 2:
        for sev_level in range(1, 6):  # L1 to L5
            level_data = per_sev[per_sev["level"] == sev_level]
            if not level_data.empty:
                _generate_sidebyside_costarr_table(
                    level_data, baseline_data, ood_datasets, final_dir,
                    f"OSA@L{sev_level}", f"costarr_sidebyside_L{sev_level}"
                )

        if not mean_df.empty and "OSA" in mean_df.columns:
            _generate_sidebyside_costarr_table(
                mean_df, baseline_data, ood_datasets, final_dir,
                "Mean OSA", "costarr_sidebyside_mean", is_mean=True
            )

    # 3. Side-by-side System Selection table (NINCO vs Open-O) for ALL severity levels L1-L5
    if len(ood_datasets) >= 2:
        for sev_level in range(1, 6):  # L1 to L5
            level_data = per_sev[per_sev["level"] == sev_level]
            if not level_data.empty:
                _generate_sidebyside_system_selection_table(
                    level_data, baseline_data, ood_datasets, final_dir,
                    f"OSA@L{sev_level}", f"system_selection_sidebyside_L{sev_level}"
                )

        if not mean_df.empty and "OSA" in mean_df.columns:
            _generate_sidebyside_system_selection_table(
                mean_df, baseline_data, ood_datasets, final_dir,
                "Mean OSA", "system_selection_sidebyside_mean", is_mean=True
            )

    # 4. Severity breakdown tables (L1-L5) for each nuisance dataset
    if len(ood_datasets) >= 2:
        nuisance_datasets = ["ImageNet-LN", "ImageNet-C", "CNS"]
        available_datasets = per_sev["dataset"].unique()
        for ds_name in nuisance_datasets:
            if ds_name in available_datasets:
                _generate_severity_breakdown_table(per_sev, ood_datasets, final_dir, ds_name)


def _generate_system_selection_table(df, baseline_df, out_dir, title, fname, ood_label):
    """
    Compact table: one row per backbone, columns show best detector (score) per dataset.
    """
    if df.empty or "OSA" not in df.columns:
        return

    backbones = sorted(df["backbone"].unique())
    datasets = sorted(df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_order if d in datasets]

    # Build pivot
    pivot = df.pivot_table(index=["backbone", "detector"], columns="dataset", values="OSA")

    # Add baseline
    if baseline_df is not None and not baseline_df.empty and "OSA" in baseline_df.columns:
        baseline_pivot = baseline_df.pivot_table(index=["backbone", "detector"], values="OSA")
        if not baseline_pivot.empty:
            pivot.insert(0, "Baseline", baseline_pivot["OSA"])

    # Find best detector per (backbone, dataset)
    all_cols = ["Baseline"] + ds_order if "Baseline" in pivot.columns else ds_order

    bb_map = {
        "resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "convnext_t": "ConvNeXt-T",
        "densenet121": "DenseNet-121", "swin_t": "Swin-T",
    }
    col_map = {"Baseline": "Baseline", "ImageNet-LN": "ImageNet-LN", "ImageNet-C": "ImageNet-C", "CNS": "CNS-Bench"}

    header = " & ".join(["Backbone"] + [col_map.get(c, c) for c in all_cols]) + " \\\\"
    col_fmt = "l" + "c" * len(all_cols)

    rows = []
    for bb in backbones:
        cells = [bb_map.get(bb, bb)]
        try:
            bb_data = pivot.loc[bb]
            for col in all_cols:
                if isinstance(bb_data, pd.Series):
                    best_det, best_val = bb_data.name, bb_data.get(col, np.nan)
                else:
                    col_vals = bb_data[col]
                    if col_vals.notna().any():
                        best_det, best_val = col_vals.idxmax(), col_vals.max()
                    else:
                        best_det, best_val = "N/A", np.nan
                cells.append(f"{best_det.upper()} ({best_val:.3f})" if not pd.isna(best_val) else "—")
        except KeyError:
            cells.extend(["—"] * len(all_cols))
        rows.append(" & ".join(cells) + " \\\\")

    latex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{System Selection ({title}, OOD: {ood_label}). Best detector varies by evaluation condition.}}
\\label{{tab:{fname.lower()}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
{header}
\\midrule
{"".join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(os.path.join(out_dir, f"{fname}.tex"), "w") as f:
        f.write(latex)
    print(f"  > final_tables/{fname}.tex")


def _generate_sidebyside_costarr_table(df, baseline_df, ood_datasets, out_dir, title, fname, is_mean=False):
    """
    Side-by-side COSTARR table: left columns = NINCO, right columns = Open-O.
    """
    if df.empty or "OSA" not in df.columns:
        return

    backbones = sorted(df["backbone"].unique())
    detectors = sorted(df["detector"].unique())
    datasets = sorted(df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_order if d in datasets]

    # Get the two OOD datasets
    ood1 = ood_datasets[0]  # NINCO
    ood2 = ood_datasets[1] if len(ood_datasets) > 1 else None  # Open-O

    # Build pivots for each OOD dataset
    def build_pivot(ood_ds):
        if "test_ood_dataset" in df.columns:
            ood_data = df[df["test_ood_dataset"] == ood_ds]
        else:
            ood_data = df
        pivot = ood_data.pivot_table(index=["backbone", "detector"], columns="dataset", values="OSA")

        # Add baseline
        if baseline_df is not None and not baseline_df.empty and "OSA" in baseline_df.columns:
            if "test_ood_dataset" in baseline_df.columns:
                bl = baseline_df[baseline_df["test_ood_dataset"] == ood_ds]
            else:
                bl = baseline_df
            bl_pivot = bl.pivot_table(index=["backbone", "detector"], values="OSA")
            if not bl_pivot.empty:
                pivot.insert(0, "Val", bl_pivot["OSA"])

        return pivot

    pivot1 = build_pivot(ood1)
    pivot2 = build_pivot(ood2) if ood2 else None

    # Find best detector per (backbone, col) for bolding
    def get_best_per_bb_col(pivot):
        best = {}
        for bb in backbones:
            try:
                bb_data = pivot.loc[bb]
                if not isinstance(bb_data, pd.Series):
                    for col in bb_data.columns:
                        if bb_data[col].notna().any():
                            best[(bb, col)] = bb_data[col].idxmax()
            except KeyError:
                pass
        return best

    best1 = get_best_per_bb_col(pivot1)
    best2 = get_best_per_bb_col(pivot2) if pivot2 is not None else {}

    # Column setup
    cols1 = ["Val"] + ds_order if "Val" in pivot1.columns else ds_order
    cols2 = ["Val"] + ds_order if pivot2 is not None and "Val" in pivot2.columns else ds_order
    col_short = {"Val": "Val", "ImageNet-LN": "LN", "ImageNet-C": "C", "CNS": "CNS"}

    # Backbone and detector names
    bb_map = {"resnet50": "RN50", "vit_b_16": "ViT-B", "convnext_t": "CNX-T", "densenet121": "DN121", "swin_t": "Swin-T"}

    # Build header
    ood1_short = "NINCO" if "NINCO" in ood1 else ood1.replace("-", "")[:8]
    ood2_short = "Open-O" if ood2 and "Open" in ood2 else (ood2.replace("-", "")[:8] if ood2 else "")

    n_cols1 = len(cols1)
    n_cols2 = len(cols2) if pivot2 is not None else 0

    header1 = " & ".join([col_short.get(c, c) for c in cols1])
    header2 = " & ".join([col_short.get(c, c) for c in cols2]) if pivot2 is not None else ""

    # Multi-column header for OOD dataset names
    mc1 = f"\\multicolumn{{{n_cols1}}}{{c}}{{{ood1_short}}}"
    mc2 = f"\\multicolumn{{{n_cols2}}}{{c}}{{{ood2_short}}}" if pivot2 is not None else ""

    col_fmt = "ll" + "c" * n_cols1 + ("|" + "c" * n_cols2 if pivot2 is not None else "")

    rows = []
    for bb_idx, bb in enumerate(backbones):
        for det_idx, det in enumerate(detectors):
            # Get values from both pivots
            def get_row_vals(pivot, cols, best_dict, bb, det):
                vals = []
                try:
                    bb_data = pivot.loc[bb]
                    row = bb_data.loc[det] if not isinstance(bb_data, pd.Series) else bb_data
                    for col in cols:
                        v = row.get(col, np.nan) if isinstance(row, pd.Series) else row[col] if col in row.index else np.nan
                        if pd.isna(v):
                            vals.append("—")
                        else:
                            s = f"{v:.3f}"
                            if (bb, col) in best_dict and best_dict[(bb, col)] == det:
                                s = f"\\textbf{{{s}}}"
                            vals.append(s)
                except KeyError:
                    vals = ["—"] * len(cols)
                return vals

            vals1 = get_row_vals(pivot1, cols1, best1, bb, det)
            vals2 = get_row_vals(pivot2, cols2, best2, bb, det) if pivot2 is not None else []

            bb_str = bb_map.get(bb, bb) if det_idx == 0 else ""
            row_str = " & ".join([bb_str, det.upper()] + vals1 + vals2) + " \\\\"
            rows.append(row_str)

        if bb_idx < len(backbones) - 1:
            rows.append("\\midrule")

    body = "\n".join(rows)

    latex = f"""\\begin{{table*}}[t]
\\centering
\\small
\\caption{{{title} across nuisance benchmarks. Left: {ood1_short}, Right: {ood2_short}. \\textbf{{Bold}} = best detector per architecture.}}
\\label{{tab:{fname.lower()}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
 & & {mc1} & {mc2} \\\\
\\cmidrule(lr){{3-{2+n_cols1}}} \\cmidrule(lr){{{3+n_cols1}-{2+n_cols1+n_cols2}}}
Arch & Method & {header1} & {header2} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    with open(os.path.join(out_dir, f"{fname}.tex"), "w") as f:
        f.write(latex)
    print(f"  > final_tables/{fname}.tex")

    # Also save as CSV
    csv_rows = []
    for bb in backbones:
        for det in detectors:
            row_data = {"Arch": bb_map.get(bb, bb), "Method": det.upper()}

            def get_csv_vals(pivot, cols, bb, det, prefix):
                try:
                    bb_data = pivot.loc[bb]
                    row = bb_data.loc[det] if not isinstance(bb_data, pd.Series) else bb_data
                    for col in cols:
                        v = row.get(col, np.nan) if isinstance(row, pd.Series) else row[col] if col in row.index else np.nan
                        col_name = f"{prefix}_{col_short.get(col, col)}"
                        row_data[col_name] = f"{v:.3f}" if not pd.isna(v) else ""
                except KeyError:
                    for col in cols:
                        col_name = f"{prefix}_{col_short.get(col, col)}"
                        row_data[col_name] = ""

            get_csv_vals(pivot1, cols1, bb, det, ood1_short)
            if pivot2 is not None:
                get_csv_vals(pivot2, cols2, bb, det, ood2_short)
            csv_rows.append(row_data)

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)
    print(f"  > final_tables/{fname}.csv")


def _generate_sidebyside_system_selection_table(df, baseline_df, ood_datasets, out_dir, title, fname, is_mean=False):
    """
    Side-by-side System Selection table: NINCO columns | Open-O columns.
    Each cell shows: best_detector (score)
    Compact version with abbreviated detector names.
    """
    if df.empty or "OSA" not in df.columns:
        return

    backbones = sorted(df["backbone"].unique())
    datasets = sorted(df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_order if d in datasets]

    # Get the two OOD datasets
    ood1 = ood_datasets[0]  # NINCO
    ood2 = ood_datasets[1] if len(ood_datasets) > 1 else None  # Open-O

    # Build pivots for each OOD dataset
    def build_pivot(ood_ds):
        if "test_ood_dataset" in df.columns:
            ood_data = df[df["test_ood_dataset"] == ood_ds]
        else:
            ood_data = df
        pivot = ood_data.pivot_table(index=["backbone", "detector"], columns="dataset", values="OSA")

        # Add baseline
        if baseline_df is not None and not baseline_df.empty and "OSA" in baseline_df.columns:
            if "test_ood_dataset" in baseline_df.columns:
                bl = baseline_df[baseline_df["test_ood_dataset"] == ood_ds]
            else:
                bl = baseline_df
            bl_pivot = bl.pivot_table(index=["backbone", "detector"], values="OSA")
            if not bl_pivot.empty:
                pivot.insert(0, "Val", bl_pivot["OSA"])

        return pivot

    pivot1 = build_pivot(ood1)
    pivot2 = build_pivot(ood2) if ood2 else None

    # Column setup (include Val)
    cols1 = ["Val"] + ds_order if "Val" in pivot1.columns else ds_order
    cols2 = ["Val"] + ds_order if pivot2 is not None and "Val" in pivot2.columns else ds_order
    col_short = {"Val": "Val", "ImageNet-LN": "LN", "ImageNet-C": "C", "CNS": "CNS"}

    # Shorter backbone names
    bb_map = {
        "resnet50": "RN50", "vit_b_16": "ViT-B", "convnext_t": "CNX-T",
        "densenet121": "DN121", "swin_t": "Swin-T",
    }

    # Detector abbreviations for compactness
    det_abbrev = {
        "costarr": "CO", "dice": "DI", "knn": "KN", "mds": "MD", "msp": "MS",
        "odin": "OD", "postmax": "PM", "react": "RE", "she": "SH", "vim": "VI"
    }

    # OOD short names
    ood1_short = "NINCO" if "NINCO" in ood1 else ood1.replace("-", "")[:8]
    ood2_short = "Open-O" if ood2 and "Open" in ood2 else (ood2.replace("-", "")[:8] if ood2 else "")

    n_cols1 = len(cols1)
    n_cols2 = len(cols2) if pivot2 is not None else 0

    header1 = " & ".join([col_short.get(c, c) for c in cols1])
    header2 = " & ".join([col_short.get(c, c) for c in cols2]) if pivot2 is not None else ""

    # Multi-column header for OOD dataset names
    mc1 = f"\\multicolumn{{{n_cols1}}}{{c}}{{{ood1_short}}}"
    mc2 = f"\\multicolumn{{{n_cols2}}}{{c}}{{{ood2_short}}}" if pivot2 is not None else ""

    # Column format with center-alignment
    col_fmt = "l" + "c" * n_cols1 + ("|" + "c" * n_cols2 if pivot2 is not None else "")

    rows = []
    for bb in backbones:
        cells = [bb_map.get(bb, bb)]

        # Get best detector per column for this backbone
        def get_best_cells(pivot, cols, bb):
            cell_list = []
            try:
                bb_data = pivot.loc[bb]
                if isinstance(bb_data, pd.Series):
                    # Only one detector
                    for col in cols:
                        v = bb_data.get(col, np.nan)
                        if pd.isna(v):
                            cell_list.append("—")
                        else:
                            det_short = det_abbrev.get(bb_data.name.lower(), bb_data.name.upper()[:2])
                            cell_list.append(f"{det_short} ({v:.2f})")
                else:
                    for col in cols:
                        if col in bb_data.columns and bb_data[col].notna().any():
                            best_det = bb_data[col].idxmax()
                            best_val = bb_data[col].max()
                            det_short = det_abbrev.get(best_det.lower(), best_det.upper()[:2])
                            cell_list.append(f"{det_short} ({best_val:.2f})")
                        else:
                            cell_list.append("—")
            except KeyError:
                cell_list = ["—"] * len(cols)
            return cell_list

        cells1 = get_best_cells(pivot1, cols1, bb)
        cells2 = get_best_cells(pivot2, cols2, bb) if pivot2 is not None else []

        row_str = " & ".join(cells + cells1 + cells2) + " \\\\"
        rows.append(row_str)

    body = "\n".join(rows)

    # Caption text
    caption = (
        f"Optimal detector selection ({title}) depends on both the OOD calibration set "
        f"and the evaluation benchmark. Each cell shows the best-performing detector and its OSA score. "
        f"Abbreviations: CO=COSTARR, DI=DICE, KN=KNN, MD=MDS, MS=MSP, OD=ODIN, PM=PostMax, RE=ReAct, SH=SHE, VI=VIM."
    )

    latex = f"""\\begin{{table*}}[t]
\\centering
\\small
\\caption{{{caption}}}
\\label{{tab:{fname.lower()}}}
\\begin{{tabular}}{{{col_fmt}}}
\\toprule
 & {mc1} & {mc2} \\\\
\\cmidrule(lr){{2-{1+n_cols1}}} \\cmidrule(lr){{{2+n_cols1}-{1+n_cols1+n_cols2}}}
Backbone & {header1} & {header2} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    with open(os.path.join(out_dir, f"{fname}.tex"), "w") as f:
        f.write(latex)
    print(f"  > final_tables/{fname}.tex")

    # Also save as CSV
    csv_rows = []
    for bb in backbones:
        row_data = {"Backbone": bb_map.get(bb, bb)}

        def get_best_for_csv(pivot, cols, bb, prefix):
            try:
                bb_data = pivot.loc[bb]
                if isinstance(bb_data, pd.Series):
                    for col in cols:
                        v = bb_data.get(col, np.nan)
                        col_name = f"{prefix}_{col_short.get(col, col)}"
                        if pd.isna(v):
                            row_data[col_name] = ""
                        else:
                            det_short = det_abbrev.get(bb_data.name.lower(), bb_data.name.upper()[:2])
                            row_data[col_name] = f"{det_short} ({v:.2f})"
                else:
                    for col in cols:
                        col_name = f"{prefix}_{col_short.get(col, col)}"
                        if col in bb_data.columns and bb_data[col].notna().any():
                            best_det = bb_data[col].idxmax()
                            best_val = bb_data[col].max()
                            det_short = det_abbrev.get(best_det.lower(), best_det.upper()[:2])
                            row_data[col_name] = f"{det_short} ({best_val:.2f})"
                        else:
                            row_data[col_name] = ""
            except KeyError:
                for col in cols:
                    col_name = f"{prefix}_{col_short.get(col, col)}"
                    row_data[col_name] = ""

        get_best_for_csv(pivot1, cols1, bb, ood1_short)
        if pivot2 is not None:
            get_best_for_csv(pivot2, cols2, bb, ood2_short)
        csv_rows.append(row_data)

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)
    print(f"  > final_tables/{fname}.csv")