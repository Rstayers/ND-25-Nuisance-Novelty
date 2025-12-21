import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_CNR(agg_df, dataset, out_dir):
    """
    CNR vs Severity (levels 1-5), faceted by backbone.
    """
    subset = agg_df[agg_df["dataset"] == dataset].copy()
    subset = subset[subset["level"] > 0].copy()
    if subset.empty:
        return

    # Remove undefined CNR (no correct samples)
    subset = subset[subset["Correct_Total"] > 0].copy()

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
        col_wrap=3,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Nuisance Severity", "CNR")
    g.set(ylim=(-0.05, 1.05))
    g.set(xticks=[1, 2, 3, 4, 5])
    g.fig.suptitle(f"CNR vs Severity — {dataset}", fontsize=16, y=1.05)

    fname = os.path.join(out_dir, f"CNR_faceted_{dataset}.png")
    g.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {os.path.basename(fname)}")


def plot_nuisance_fingerprint(agg_df, dataset, out_dir):
    """
    Avg CNR over levels>0, per nuisance type.
    """
    subset = agg_df[agg_df["dataset"] == dataset].copy()
    subset = subset[subset["level"] > 0].copy()
    subset = subset[subset["Correct_Total"] > 0].copy()
    if subset.empty:
        return

    grouped = (
        subset.groupby(["backbone", "detector", "nuisance"])["CNR"]
        .mean()
        .reset_index()
    )

    sns.set_style("whitegrid")
    n_backbones = grouped["backbone"].nunique()
    col_wrap = 3 if n_backbones > 3 else n_backbones

    g = sns.catplot(
        data=grouped,
        kind="bar",
        x="nuisance",
        y="CNR",
        hue="detector",
        col="backbone",
        col_wrap=col_wrap,
        palette="rocket",
        height=3.6,
        aspect=1.25,
        sharey=True,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Nuisance", "CNR (avg over levels>0)")
    g.set(ylim=(0, 1.05))

    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    g.fig.suptitle(f"Nuisance Fingerprint (CNR) — {dataset}", fontsize=16, y=1.02)

    fname = os.path.join(out_dir, f"fingerprint_{dataset}.png")
    g.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {os.path.basename(fname)}")


def plot_detector_leaderboard(agg_df, dataset, out_dir):
    """
    Detector leaderboard plotted as NNR = NN/Total (unconditional nuisance novelty).
    This fixes the previous mismatch where the title said CNR but the value was NN/Total.
    """
    subset = agg_df[agg_df["dataset"] == dataset].copy()
    subset = subset[subset["level"] > 0].copy()
    if subset.empty:
        return

    grouped = (
        subset.groupby(["backbone", "detector"])[["NNR"]]
        .mean()
        .reset_index()
        .rename(columns={"NNR": "Mean_NNR"})
    )

    det_order = (
        grouped.groupby("detector")["Mean_NNR"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    sns.set_style("whitegrid")
    n_backbones = grouped["backbone"].nunique()
    col_wrap = 3 if n_backbones > 3 else n_backbones

    g = sns.catplot(
        data=grouped,
        kind="bar",
        x="detector",
        y="Mean_NNR",
        order=det_order,
        col="backbone",
        col_wrap=col_wrap,
        palette="magma",
        height=3.6,
        aspect=1.25,
        sharey=True,
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Detector", "Mean NNR (NN/Total, levels>0)")
    g.set(ylim=(0, 1.05))

    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    g.fig.suptitle(f"Detector Leaderboard (NNR) — {dataset}", fontsize=16, y=1.02)

    fname = os.path.join(out_dir, f"detector_leaderboard_NNR_{dataset}.png")
    g.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {os.path.basename(fname)}")


def plot_osa_gap(agg_df, dataset, out_dir):
    """
    Accuracy vs OSA_ID (accepted-correct) to visualize 'collapse' across severity.
    """
    subset = agg_df[agg_df["dataset"] == dataset].copy()
    if subset.empty:
        return

    grouped = subset.groupby(["backbone", "detector", "level"])[["Accuracy", "OSA_ID"]].mean().reset_index()

    melted = grouped.melt(
        id_vars=["backbone", "detector", "level"],
        value_vars=["Accuracy", "OSA_ID"],
        var_name="Metric",
        value_name="Score",
    )

    backbones = melted["backbone"].unique()
    for bb in backbones:
        bb_data = melted[melted["backbone"] == bb]

        palette = {"Accuracy": "#3498db", "OSA_ID": "#e74c3c"}

        g = sns.relplot(
            data=bb_data,
            x="level",
            y="Score",
            hue="Metric",
            style="Metric",
            col="detector",
            col_wrap=4,
            kind="line",
            markers=True,
            dashes=False,
            height=3.5,
            aspect=1.2,
            palette=palette,
            linewidth=2.5,
        )

        g.set_titles("{col_name}")
        g.set_axis_labels("Severity Level", "Score (0–1)")
        g.set(ylim=(-0.05, 1.05))
        g.set(xticks=[1, 2, 3, 4, 5])

        g.fig.suptitle(f"Accuracy vs OSA_ID Collapse — {dataset} ({bb})", fontsize=16, y=1.02)

        fname = os.path.join(out_dir, f"osa_gap_{dataset}_{bb}.png")
        g.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated: {os.path.basename(fname)}")
