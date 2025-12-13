import pandas as pd
import os
import numpy as np


def generate_detector_leaderboard(agg_df, ood_df, backbone, dataset, out_dir):
    """
    Enhanced Leaderboard:
    - Standard OOD Stats (Mean AUROC, Mean FPR95)
    - Nuisance Stats (Mean CNR) - Calculated as Average of Levels 1-5
    """
    subset_agg = agg_df[
        (agg_df['backbone'] == backbone) &
        (agg_df['dataset'] == dataset)
        ].copy()

    if subset_agg.empty: return

    # 1. Filter out Level 0 for Robustness Metrics
    # We only care about CNR on corrupted data
    robustness_subset = subset_agg[subset_agg['level'] > 0].copy()

    # Calculate Per-Level CNR first
    # (agg_df already has 'CNR' calculated per row in processing.py,
    #  but let's be safe and recalculate if grouping changed)
    grouped_by_level = robustness_subset.groupby(['detector', 'level'])[
        ['Clean_Success', 'Nuisance_Novelty']
    ].sum().reset_index()

    grouped_by_level['Correct_Total'] = grouped_by_level['Clean_Success'] + grouped_by_level['Nuisance_Novelty']

    # Avoid div by zero
    grouped_by_level['CNR'] = grouped_by_level.apply(
        lambda r: r['Nuisance_Novelty'] / r['Correct_Total'] if r['Correct_Total'] > 0 else 0.0, axis=1
    )

    # 2. Calculate Overall Mean CNR (Macro-Average over Levels 1-5)
    mean_cnr = grouped_by_level.groupby('detector')['CNR'].mean().reset_index()
    mean_cnr.rename(columns={'CNR': 'Mean_CNR'}, inplace=True)

    # 3. Pivot for specific levels
    pivot_cnr = grouped_by_level.pivot(
        index='detector', columns='level', values='CNR'
    ).reset_index()

    # Rename columns safely
    for lvl in [1, 3, 5]:
        if lvl in pivot_cnr.columns:
            pivot_cnr.rename(columns={lvl: f'CNR_L{lvl}'}, inplace=True)
        else:
            pivot_cnr[f'CNR_L{lvl}'] = np.nan

    # 4. Process OOD Metrics
    if ood_df is not None and not ood_df.empty:
        subset_ood = ood_df[
            (ood_df['backbone'] == backbone) &
            (ood_df['dataset'] == dataset)
            ].copy()
        if not subset_ood.empty:
            mean_ood = subset_ood.groupby('detector')[['AUROC', 'FPR95']].mean().reset_index()
            mean_ood.rename(columns={'AUROC': 'Mean_AUROC', 'FPR95': 'Mean_FPR95'}, inplace=True)
        else:
            mean_ood = pd.DataFrame(columns=['detector', 'Mean_AUROC', 'Mean_FPR95'])
    else:
        mean_ood = pd.DataFrame(columns=['detector', 'Mean_AUROC', 'Mean_FPR95'])

    # 5. Merge
    final = mean_cnr.merge(mean_ood, on='detector', how='outer')
    final = final.merge(pivot_cnr, on='detector', how='outer')

    # Format
    cols = ['Mean_AUROC', 'Mean_FPR95', 'Mean_CNR', 'CNR_L1', 'CNR_L3', 'CNR_L5']
    for c in cols:
        if c in final.columns:
            final[c] = final[c].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    fname = f"leaderboard_{backbone}_{dataset}.csv"
    final.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"Generated: {fname}")


def generate_master_table(agg_df, ood_df, out_dir):
    """
    The MASTER TABLE.
    Uses MACRO-AVERAGING (Mean of Levels) to avoid Simpson's Paradox.
    """
    # 1. Filter out Level 0
    robustness_df = agg_df[agg_df['level'] > 0].copy()

    if robustness_df.empty:
        robustness_df = agg_df.copy()

    # 2. Group by Dataset/Backbone/Detector/Level first to get per-level metrics
    # Note: agg_df from processing.py already has 'CNR', 'OSA', 'Accuracy' per level.
    # We just need to take the mean of those columns.

    # However, agg_df might be split by 'nuisance' (e.g. gaussian_noise, blur).
    # We should average over nuisances first, then levels? Or all together?
    # Standard practice: Average over all (Level, Nuisance) conditions equaly.

    # Let's take the mean of the metrics column directly.
    # This weights Level 1 (easy) and Level 5 (hard) equally.

    metrics = robustness_df.groupby(['backbone', 'detector', 'dataset'])[
        ['Accuracy', 'OSA', 'CNR', 'Rejection_Rate']
    ].mean().reset_index()

    # Rename to Master Table standard
    metrics = metrics.rename(columns={
        'Accuracy': 'Mean_ID_Acc',
        'OSA': 'Mean_OSA',
        'CNR': 'Mean_CNR',
        'Rejection_Rate': 'Mean_Rej_Rate'
    })

    # 3. Merge OOD Stats
    if ood_df is not None and not ood_df.empty:
        ood_agg = ood_df.groupby(['backbone', 'detector'])[['Mean_AUROC', 'Mean_FPR95']].mean().reset_index()
        final = pd.merge(metrics, ood_agg, on=['backbone', 'detector'], how='left')
    else:
        final = metrics

    # 4. Format
    final = final.sort_values(['dataset', 'backbone', 'detector'])
    cols = ['Mean_ID_Acc', 'Mean_OSA', 'Mean_CNR', 'Mean_Rej_Rate', 'Mean_AUROC', 'Mean_FPR95']
    for c in cols:
        if c in final.columns:
            final[c] = final[c].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    fname = os.path.join(out_dir, "master_table_per_dataset.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")


