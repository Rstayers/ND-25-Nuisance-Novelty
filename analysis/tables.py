import pandas as pd
import os


def generate_detector_leaderboard(agg_df, ood_df, backbone, dataset, out_dir):
    """
    Enhanced Leaderboard:
    - Standard OOD Stats (Mean AUROC, Mean FPR95)
    - Nuisance Stats (Mean CNR)
    - Breakdown (CNR @ L1, L3, L5) aggregated across all nuisances.
    """
    # 1. Filter for Backbone/Dataset
    subset_agg = agg_df[
        (agg_df['backbone'] == backbone) &
        (agg_df['dataset'] == dataset)
        ].copy()

    if subset_agg.empty: return

    # --- FIX: Aggregate across Nuisances first ---
    # We want one CNR value per (detector, level)
    # This averages the CNR of Blur, Noise, etc. into a single score
    grouped_by_level = subset_agg.groupby(['detector', 'level'])['CNR'].mean().reset_index()

    # 2. Calculate Overall Mean CNR (across levels 1-5)
    mean_cnr = grouped_by_level.groupby('detector')['CNR'].mean().reset_index()
    mean_cnr.rename(columns={'CNR': 'Mean_CNR'}, inplace=True)

    # 3. Pivot for specific levels (L1, L3, L5)
    pivot_cnr = grouped_by_level[grouped_by_level['level'].isin([1, 3, 5])].pivot(
        index='detector', columns='level', values='CNR'
    ).reset_index()
    pivot_cnr.rename(columns={1: 'CNR_L1', 3: 'CNR_L3', 5: 'CNR_L5'}, inplace=True)

    # 4. Process OOD Metrics (AUROC / FPR95)
    # These are usually already unique per detector/dataset/level,
    # but let's ensure we average just in case.
    if not ood_df.empty:
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

    # 5. Merge All
    final = mean_cnr.merge(mean_ood, on='detector', how='outer')
    final = final.merge(pivot_cnr, on='detector', how='outer')

    # Format to 3 decimal places
    cols = ['Mean_AUROC', 'Mean_FPR95', 'Mean_CNR', 'CNR_L1', 'CNR_L3', 'CNR_L5']
    for c in cols:
        if c in final.columns:
            final[c] = final[c].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    fname = f"leaderboard_{backbone}_{dataset}.csv"
    final.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"Generated: {fname}")


def generate_transfer_table(agg_df, level, dataset, out_dir):
    """
    Compares Backbones at a specific Level.
    """
    subset = agg_df[
        (agg_df['dataset'] == dataset) &
        (agg_df['level'] == level)
        ].copy()

    if subset.empty: return

    # Sum raw counts across all nuisances
    grouped = subset.groupby(['backbone', 'detector'])[
        ['Clean_Success', 'Nuisance_Novelty', 'Total']].sum().reset_index()

    # Recalculate metrics based on summed counts
    grouped['Accuracy'] = (grouped['Clean_Success'] + grouped['Nuisance_Novelty']) / grouped['Total']

    # Safe division for CNR
    correct_total = grouped['Nuisance_Novelty'] + grouped['Clean_Success']
    grouped['CNR'] = grouped['Nuisance_Novelty'] / correct_total
    grouped['CNR'] = grouped['CNR'].fillna(0.0)

    output = grouped[['backbone', 'detector', 'Accuracy', 'CNR']].sort_values('backbone')

    # Format
    output['Accuracy'] = output['Accuracy'].apply(lambda x: f"{x:.1%}")
    output['CNR'] = output['CNR'].apply(lambda x: f"{x:.1%}")

    fname = f"transfer_lvl{level}_{dataset}.csv"
    output.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"Generated: {fname}")

def generate_acc_accept_table(agg_df, dataset, out_dir):
    """
    For a given dataset, produce a compact table with:
      - Accuracy per level
      - Acceptance rate per level
    Rows:   backbone, detector
    Cols:   Acc_Lk, Accept_Lk for all observed levels k.
    Aggregated across all nuisance types.
    """
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty:
        return

    # 1) Sum counts over nuisances at each level
    grouped = subset.groupby(
        ['backbone', 'detector', 'level']
    )[['Clean_Success',
       'Nuisance_Novelty',
       'Double_Failure',
       'Contained_Misidentification',
       'Total']].sum().reset_index()

    cs = grouped['Clean_Success']
    nn = grouped['Nuisance_Novelty']
    df = grouped['Double_Failure']
    tot = grouped['Total']

    grouped['Accuracy'] = (cs + nn) / tot
    grouped['Rejection_Rate'] = (nn + df) / tot
    grouped['Acceptance_Rate'] = 1.0 - grouped['Rejection_Rate']

    # 2) Pivot Accuracy
    acc_pivot = grouped.pivot(
        index=['backbone', 'detector'],
        columns='level',
        values='Accuracy'
    )
    acc_pivot.columns = [f"Acc_L{int(l)}" for l in acc_pivot.columns]
    acc_pivot = acc_pivot.reset_index()

    # 3) Pivot Acceptance
    accept_pivot = grouped.pivot(
        index=['backbone', 'detector'],
        columns='level',
        values='Acceptance_Rate'
    )
    accept_pivot.columns = [f"Accept_L{int(l)}" for l in accept_pivot.columns]
    accept_pivot = accept_pivot.reset_index()

    # 4) Merge and format
    final = acc_pivot.merge(
        accept_pivot,
        on=['backbone', 'detector'],
        how='outer'
    )

    for c in final.columns:
        if c.startswith("Acc_L") or c.startswith("Accept_L"):
            final[c] = final[c].apply(
                lambda x: f"{x:.1%}" if pd.notnull(x) else "-"
            )

    safe_ds = str(dataset).replace("/", "_")
    fname = os.path.join(out_dir, f"acc_accept_{safe_ds}.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")
def generate_master_table(agg_df, ood_df, out_dir):
    """
    Master table:
    One row per (backbone, detector, dataset) summarizing:
      - Accuracy
      - Acceptance / Rejection
      - CNR (P(reject | correct))
      - SR  (P(reject | wrong))
      - Mean AUROC / FPR95 (if available)
    Aggregated across all levels and nuisances.
    """
    # 1) Sum counts across nuisances and levels
    grouped = agg_df.groupby(
        ['backbone', 'detector', 'dataset']
    )[['Clean_Success',
       'Nuisance_Novelty',
       'Double_Failure',
       'Contained_Misidentification',
       'Total']].sum().reset_index()

    cs = grouped['Clean_Success']
    nn = grouped['Nuisance_Novelty']
    df = grouped['Double_Failure']
    cm = grouped['Contained_Misidentification']
    tot = grouped['Total']

    # Core metrics
    grouped['Accuracy'] = (cs + nn) / tot
    grouped['Rejection_Rate'] = (nn + df) / tot
    grouped['Acceptance_Rate'] = 1.0 - grouped['Rejection_Rate']

    # Conditional metrics
    correct_total = cs + nn
    grouped['CNR'] = nn / correct_total
    grouped['CNR'] = grouped['CNR'].fillna(0.0)

    incorrect_total = df + cm
    grouped['SR'] = df / incorrect_total
    grouped['SR'] = grouped['SR'].fillna(1.0)

    # 2) Aggregate OOD metrics across levels
    if ood_df is not None and not ood_df.empty:
        ood_group = ood_df.groupby(
            ['backbone', 'detector', 'dataset']
        )[['AUROC', 'FPR95']].mean().reset_index()
        ood_group.rename(
            columns={'AUROC': 'Mean_AUROC', 'FPR95': 'Mean_FPR95'},
            inplace=True
        )
        final = grouped.merge(
            ood_group,
            on=['backbone', 'detector', 'dataset'],
            how='left'
        )
    else:
        final = grouped
        final['Mean_AUROC'] = pd.NA
        final['Mean_FPR95'] = pd.NA

    # 3) Nice formatting for paper tables
    pct_cols = ['Accuracy', 'Acceptance_Rate',
                'Rejection_Rate', 'CNR', 'SR']
    for c in pct_cols:
        final[c] = final[c].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "-"
        )

    for c in ['Mean_AUROC', 'Mean_FPR95']:
        if c in final.columns:
            final[c] = final[c].apply(
                lambda x: f"{x:.3f}" if pd.notnull(x) else "-"
            )

    fname = os.path.join(out_dir, "master_table.csv")
    final.to_csv(fname, index=False)
    print(f"Generated: {os.path.basename(fname)}")
