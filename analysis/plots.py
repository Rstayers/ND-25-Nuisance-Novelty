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


def plot_outcomes_stack(agg_df, backbone, detector, dataset, out_dir):
    subset = agg_df[
        (agg_df['backbone'] == backbone) &
        (agg_df['detector'] == detector) &
        (agg_df['dataset'] == dataset)
        ].copy()

    if subset.empty: return

    grouped = subset.groupby('level')[list(COLORS.keys())].sum()
    totals = grouped.sum(axis=1)
    normalized = grouped.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    normalized.plot(kind='bar', stacked=True, color=[COLORS[c] for c in normalized.columns], ax=ax, width=0.8)

    ax.set_title(f"Outcomes: {backbone} + {detector}", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"stack_{backbone}_{detector}_{dataset}.png"), dpi=300)
    plt.close()


def plot_competency_cliff(agg_df, dataset, out_dir):
    subset = agg_df[agg_df['dataset'] == dataset].copy()

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=subset, x='level', y='CNR', hue='backbone', style='detector',
        markers=True, linewidth=2.5, palette="viridis"
    )

    plt.title(f"Competency Cliff: {dataset}", fontsize=14)
    plt.ylabel("Conditional Nuisance Rate (CNR)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cliff_{dataset}.png"), dpi=300)
    plt.close()


def plot_nuisance_fingerprint(agg_df, dataset, out_dir):
    """
    Fingerprint: Average CNR across ALL levels for each Nuisance.
    """
    subset = agg_df[agg_df['dataset'] == dataset].copy()
    if subset.empty: return

    # Average CNR across levels
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


def plot_accuracy_degradation(agg_df, dataset, out_dir):
    detectors = agg_df['detector'].unique()
    for det in detectors:
        subset = agg_df[(agg_df['dataset'] == dataset) & (agg_df['detector'] == det)].copy()

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=subset, x='level', y='Accuracy', hue='backbone', marker='o', linewidth=2.5)
        plt.title(f"Accuracy Degradation: {det} on {dataset}")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"acc_deg_{det}_{dataset}.png"), dpi=300)
        plt.close()


def plot_safety_alignment(agg_df, dataset, out_dir):
    backbones = agg_df['backbone'].unique()
    for bb in backbones:
        subset = agg_df[(agg_df['dataset'] == dataset) & (agg_df['backbone'] == bb)].copy()
        if subset.empty: continue

        plt.figure(figsize=(10, 6))
        grouped = subset.groupby(['detector', 'level'])[['Accuracy', 'Rejection_Rate']].mean().reset_index()

        sns.lineplot(data=grouped, x='level', y='Accuracy', hue='detector', linestyle='-', marker='o')
        sns.lineplot(data=grouped, x='level', y='Rejection_Rate', hue='detector', linestyle='--', marker='X')

        plt.title(f"Safety Alignment: {bb} on {dataset}")
        plt.ylabel("Proportion (0-1)")
        plt.savefig(os.path.join(out_dir, f"alignment_{bb}_{dataset}.png"), dpi=300)
        plt.close()