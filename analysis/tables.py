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
    """
    mean_df = get_mean_metrics(agg_df, dataset)
    if mean_df.empty:
        return

    if "backbone" not in mean_df.columns or len(mean_df["backbone"].unique()) < 2:
        # Single backbone - just list detectors
        cols_to_use = ["detector", "CSA", "CCR", "NNR", "OSA_Gap"]
        # Add OSA if available
        if "OSA" in mean_df.columns and not mean_df["OSA"].isna().all():
            cols_to_use.insert(2, "OSA")

        result = mean_df[[c for c in cols_to_use if c in mean_df.columns]].copy()
        result = result.rename(columns={"detector": "Detector", "OSA_Gap": "Gap"})

        for col in result.columns:
            if col != "Detector":
                result[col] = result[col].apply(lambda x: _fmt(x, 2))
        result = result.sort_values("NNR")

        safe = dataset.replace("-", "_")
        result.to_csv(os.path.join(out_dir, f"summary_{safe}.csv"), index=False)
        _save_latex(result, os.path.join(out_dir, f"summary_{safe}.tex"),
                    caption=f"Results on {dataset}.", label=f"tab:{safe.lower()}")
        return

    # Multiple backbones - pivot table
    pivot = mean_df.pivot(index="detector", columns="backbone", values="NNR")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"detector": "Detector"})

    for col in pivot.columns:
        if col != "Detector":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

    safe = dataset.replace("-", "_")
    pivot.to_csv(os.path.join(out_dir, f"NNR_matrix_{safe}.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, f"NNR_matrix_{safe}.tex"),
                caption=f"NNR (Detector × Backbone) on {dataset}.",
                label=f"tab:nnr_matrix_{safe.lower()}")


def generate_full_results_table(agg_df, dataset, out_dir):
    """
    Full table: every (backbone, detector) row with all metrics.
    Now includes OSA and CNR.
    """
    mean_df = get_mean_metrics(agg_df, dataset)
    if mean_df.empty:
        return

    # Add ADR
    adr_df = compute_adr(agg_df[agg_df["dataset"] == dataset] if "dataset" in agg_df.columns else agg_df)
    if not adr_df.empty:
        merge_cols = [c for c in ["backbone", "detector"] if c in mean_df.columns]
        mean_df = mean_df.merge(adr_df[merge_cols + ["ADR"]], on=merge_cols, how="left")

    # Build column list with OSA and CNR
    cols = ["backbone", "detector", "CSA", "CCR", "NNR", "CNR", "OSA_Gap"]
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


def _save_grouped_latex(pivot, path, caption="", label="", fmt=".2f", bold_max=False):
    """
    Save table with rows=(backbone, detector) grouped by backbone.
    Columns = datasets. Bold max (or min if bold_max=False) value per row.
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
                    best_idx = np.nanargmax(vals)
                else:
                    best_idx = np.nanargmax(vals)  # For NNR, higher = worse = bold

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
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    mean_df["Config"] = mean_df["backbone"] + " + " + mean_df["detector"]

    pivot = mean_df.pivot(index="Config", columns="dataset", values="NNR")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot.reset_index()

    for col in pivot.columns:
        if col != "Config":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

    pivot.to_csv(os.path.join(out_dir, "benchmark_NNR.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, "benchmark_NNR.tex"),
                caption="NNR per configuration across benchmarks.",
                label="tab:benchmark_nnr")


def generate_benchmark_osa_table(agg_df, out_dir):
    """
    Table: (backbone+detector) × benchmark with OSA.
    NEW: Shows every configuration's OSA on every benchmark.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if "OSA" not in mean_df.columns or mean_df["OSA"].isna().all():
        print("  > benchmark_OSA.csv skipped (no OSA data)")
        return

    mean_df["Config"] = mean_df["backbone"] + " + " + mean_df["detector"]

    pivot = mean_df.pivot(index="Config", columns="dataset", values="OSA")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]  # Higher OSA at top
    pivot = pivot.reset_index()

    for col in pivot.columns:
        if col != "Config":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

    pivot.to_csv(os.path.join(out_dir, "benchmark_OSA.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, "benchmark_OSA.tex"),
                caption="OSA per configuration across benchmarks.",
                label="tab:benchmark_osa")


def generate_benchmark_by_severity_table(agg_df, out_dir, metric="NNR"):
    """
    Table: (backbone+detector) × severity for each benchmark.
    Generates SEPARATE TABLE FOR EACH DATASET.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]

    if metric not in per_sev.columns or per_sev[metric].isna().all():
        print(f"  > {metric}_by_severity tables skipped (no data)")
        return

    datasets = sorted(per_sev["dataset"].unique())

    for dataset in datasets:
        ds_data = per_sev[per_sev["dataset"] == dataset].copy()
        ds_data["Config"] = ds_data["backbone"] + " + " + ds_data["detector"]

        pivot = ds_data.pivot(index="Config", columns="level", values=metric)
        pivot.columns = [f"L{int(c)}" for c in pivot.columns]
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        pivot = pivot.reset_index()

        for col in pivot.columns:
            if col != "Config":
                pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

        safe = dataset.replace("-", "_")
        pivot.to_csv(os.path.join(out_dir, f"{metric}_by_severity_{safe}.csv"), index=False)
        _save_latex(pivot, os.path.join(out_dir, f"{metric}_by_severity_{safe}.tex"),
                    caption=f"{metric} by severity on {dataset}.",
                    label=f"tab:{metric.lower()}_severity_{safe.lower()}")


# =============================================================================
# CROSS-DATASET TABLES
# =============================================================================

def generate_cross_dataset_summary(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with NNR.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    mean_df["Config"] = mean_df["backbone"] + " + " + mean_df["detector"]

    pivot = mean_df.pivot(index="Config", columns="dataset", values="NNR")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot.reset_index()

    for col in pivot.columns:
        if col != "Config":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

    pivot.to_csv(os.path.join(out_dir, "cross_dataset_NNR.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, "cross_dataset_NNR.tex"),
                caption="NNR per configuration across datasets.",
                label="tab:cross_nnr")


def generate_cross_dataset_osa(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with OSA.
    NEW: Cross-dataset OSA comparison.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if "OSA" not in mean_df.columns or mean_df["OSA"].isna().all():
        print("  > cross_dataset_OSA.csv skipped (no OSA data)")
        return

    mean_df["Config"] = mean_df["backbone"] + " + " + mean_df["detector"]

    pivot = mean_df.pivot(index="Config", columns="dataset", values="OSA")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot.reset_index()

    for col in pivot.columns:
        if col != "Config":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 2))

    pivot.to_csv(os.path.join(out_dir, "cross_dataset_OSA.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, "cross_dataset_OSA.tex"),
                caption="OSA per configuration across datasets.",
                label="tab:cross_osa")


def generate_cross_dataset_adr(agg_df, out_dir):
    """
    Table: (backbone+detector) × dataset with ADR.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR table skipped (no data)")
        return

    adr_df["Config"] = adr_df["backbone"] + " + " + adr_df["detector"]

    pivot = adr_df.pivot(index="Config", columns="dataset", values="ADR")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot.reset_index()

    for col in pivot.columns:
        if col != "Config":
            pivot[col] = pivot[col].apply(lambda x: _fmt(x, 3))

    pivot.to_csv(os.path.join(out_dir, "cross_dataset_ADR.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, "cross_dataset_ADR.tex"),
                caption="ADR per configuration across datasets.",
                label="tab:cross_adr")


def generate_cross_dataset_full(agg_df, out_dir):
    """
    Full table: every (backbone, detector, dataset) row.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    adr_df = compute_adr(agg_df)
    if not adr_df.empty:
        merge_cols = [c for c in ["backbone", "detector", "dataset"] if c in mean_df.columns]
        mean_df = mean_df.merge(adr_df[merge_cols + ["ADR"]], on=merge_cols, how="left")

    cols = ["dataset", "backbone", "detector", "CSA", "CCR", "NNR", "CNR", "OSA_Gap"]
    if "OSA" in mean_df.columns and not mean_df["OSA"].isna().all():
        cols.insert(5, "OSA")
    if "ADR" in mean_df.columns:
        cols.append("ADR")
    cols = [c for c in cols if c in mean_df.columns]

    result = mean_df[cols].copy()
    result = result.rename(columns={"dataset": "Dataset", "backbone": "Backbone",
                                    "detector": "Detector", "OSA_Gap": "Gap"})

    for col in result.columns:
        if col not in ["Dataset", "Backbone", "Detector"]:
            result[col] = result[col].apply(lambda x: _fmt(x, 2))

    result = result.sort_values(["Dataset", "Backbone", "NNR"])

    result.to_csv(os.path.join(out_dir, "full_results.csv"), index=False)
    _save_latex(result, os.path.join(out_dir, "full_results.tex"),
                caption="Full results.", label="tab:full")


def generate_rank_comparison_table(agg_df, out_dir, metric="NNR"):
    """
    Table: configuration ranks per dataset.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    mean_df["Config"] = mean_df["backbone"] + " + " + mean_df["detector"]

    # Compute ranks per dataset
    rankings = []
    ascending = True if metric in ["NNR", "CNR", "OSA_Gap", "ADR"] else False  # Lower is better for NNR

    for ds in mean_df["dataset"].unique():
        ds_data = mean_df[mean_df["dataset"] == ds].copy()
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

    pivot.to_csv(os.path.join(out_dir, f"ranks_{metric}.csv"), index=False)
    _save_latex(pivot, os.path.join(out_dir, f"ranks_{metric}.tex"),
                caption=f"Configuration rankings by {metric}.",
                label=f"tab:ranks_{metric.lower()}")


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