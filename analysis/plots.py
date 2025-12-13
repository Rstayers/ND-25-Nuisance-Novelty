import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

COLORS = {
    "Clean_Success": "#2ecc71",
    "Nuisance_Novelty": "#3498db",
    "Double_Failure": "#95a5a6",
    "Contained_Misidentification": "#e74c3c"
}



def plot_competency_cliff(agg_df, dataset, out_dir):
    """
    Plots CNR vs Severity.
    Explicitly filters Level > 0 to match table logic.
    """
    subset = agg_df[agg_df['dataset'] == dataset].copy()

    # FIX: Filter Level 0
    subset = subset[subset['level'] > 0]

    if subset.empty: return

    # Ensure CNR is calculated (if not already)
    subset['Correct_Total'] = subset['Clean_Success'] + subset['Nuisance_Novelty']
    subset = subset[subset['Correct_Total'] > 0]
    subset['CNR'] = subset['Nuisance_Novelty'] / subset['Correct_Total']

    g = sns.relplot(
        data=subset,
        x="level",
        y="CNR",
        hue="detector",
        style="detector",
        col="backbone",
        kind="line",
        palette="tab10",
        markers=True,
        dashes=False,
        linewidth=2.5,
        height=4,
        aspect=1.2,
        col_wrap=3
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Nuisance Severity", "CNR Score")
    g.set(ylim=(-0.05, 1.05))
    g.set(xticks=[1, 2, 3, 4, 5])
    g.fig.suptitle(f"Detector Robustness across Architectures - {dataset}", fontsize=16, y=1.05)

    fname = os.path.join(out_dir, f"competency_cliff_faceted_{dataset}.png")
    g.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Faceted Plot: {os.path.basename(fname)}")


# ... (Keep other plots: plot_nuisance_fingerprint, plot_accuracy_degradation, etc. unchanged) ...
def plot_nuisance_fingerprint(agg_df, dataset, out_dir):
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty: return
    grouped = subset.groupby(['detector', 'nuisance'])['CNR'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=grouped, x='nuisance', y='CNR', hue='detector', palette="rocket")
    plt.title(f"Nuisance Fingerprint (Avg across Levels) - {dataset}", fontsize=14)
    plt.ylabel("Avg Conditional Nuisance Rate (CNR)", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fingerprint_{dataset}.png"), dpi=300)
    plt.close()





def plot_osa_cliff(agg_df, dataset, out_dir):
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty: return
    subset = subset[subset['level'] > 0]
    grouped = subset.groupby(['backbone', 'detector', 'level'])['OSA'].mean().reset_index()
    backbones = grouped['backbone'].unique()
    for bb in backbones:
        bb_data = grouped[grouped['backbone'] == bb]
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.lineplot(data=bb_data, x='level', y='OSA', hue='detector', style='detector', markers=True, dashes=False,
                     linewidth=2.5, markersize=8)
        plt.title(f"Operational OSA Degradation: {dataset} ({bb})", fontsize=15)
        plt.ylabel("OSA (Correct & Accepted)", fontsize=12)
        plt.xlabel("Severity Level", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Detector")
        plt.tight_layout()
        fname = f"osa_cliff_{dataset}_{bb}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()





def plot_detector_robustness(agg_df, dataset, out_dir):
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty: return
    grouped = subset.groupby('detector')[['Nuisance_Novelty', 'Total']].sum().reset_index()
    grouped['Success_Rate'] = grouped['Nuisance_Novelty'] / grouped['Total']
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    grouped = grouped.sort_values('Success_Rate', ascending=False)
    sns.barplot(data=grouped, x='detector', y='Success_Rate', palette="magma")
    plt.title(f"Detector Robustness: Nuisance Novelty Rate - {dataset}")
    plt.ylabel("Nuisance Novelty Rate (%)")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"detector_robustness_{dataset}.png"), dpi=300)
    plt.close()


# ... existing imports ...

def plot_osa_gap(agg_df, dataset, out_dir):
    """
    Plots Classifier Accuracy vs OSA (Operational Open-Set Accuracy) side-by-side.
    Visualizes the "Collapse" of OSA relative to the Classifier's performance.
    """
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty: return

    # 1. Aggregate across nuisances (Mean over nuisances per level)
    # We include Level 0 to show the starting gap (if any)
    grouped = subset.groupby(['backbone', 'detector', 'level'])[['Accuracy', 'OSA']].mean().reset_index()

    # 2. Melt to Long Format for Seaborn
    melted = grouped.melt(
        id_vars=['backbone', 'detector', 'level'],
        value_vars=['Accuracy', 'OSA'],
        var_name='Metric',
        value_name='Score'
    )

    # 3. Plot per Backbone
    backbones = melted['backbone'].unique()

    for bb in backbones:
        bb_data = melted[melted['backbone'] == bb]

        # Accuracy = Blue (Base), OSA = Red (Collapse)
        palette = {"Accuracy": "#3498db", "OSA": "#e74c3c"}

        g = sns.relplot(
            data=bb_data,
            x="level",
            y="Score",
            hue="Metric",
            style="Metric",
            col="detector",
            col_wrap=4,  # 4 detectors per row for readability
            kind="line",
            markers=True,
            dashes=False,
            height=3.5,
            aspect=1.2,
            palette=palette,
            linewidth=2.5
        )

        g.set_titles("{col_name}")
        g.set_axis_labels("Severity Level", "Score (0-1)")
        g.set(ylim=(-0.05, 1.05))
        g.set(xticks=[0, 1, 2, 3, 4, 5])

        g.fig.suptitle(f"Classifier Accuracy vs OSA Collapse: {dataset} ({bb})", fontsize=16, y=1.02)

        fname = os.path.join(out_dir, f"osa_gap_{dataset}_{bb}.png")
        g.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
