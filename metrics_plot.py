#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Plot Helpers
# -----------------------------------
def plot_metric(df, metric, ylabel, title, subtitle, out_path):
    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df,
        x="detector",
        y=metric,
        hue="fpr_target"
    )
    # Combined title + subtitle
    plt.title(f"{title}\n({subtitle})", fontsize=12)
    plt.ylabel(ylabel)
    plt.legend(title="FPR Target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")


def plot_head_to_head(nr_df, auc_df, worst_df, task_df, out_dir="results/metrics_plots"):
    os.makedirs(out_dir, exist_ok=True)

    # NR@FPR
    plot_metric(
        nr_df, "nr",
        ylabel="Mean Accept Rate",
        title="NR@FPR — Nuisance Novelty Rate at Target FPR",
        subtitle="How many samples still get accepted when we try to keep ~X% ID false alarms",
        out_path=os.path.join(out_dir, "nr_fpr.png")
    )

    # NR-AUC
    plot_metric(
        auc_df, "nr_auc",
        ylabel="NR-AUC",
        title="NR-AUC — Area under Severity Curve",
        subtitle="Bigger = detector tolerates corruptions better across all severities",
        out_path=os.path.join(out_dir, "nr_auc.png")
    )

    # NR-Worst-20%
    plot_metric(
        worst_df, "nr_worstk",
        ylabel="Worst-Case Accept Rate",
        title="NR-Worst-20% — Robustness on Hardest Corruptions",
        subtitle="Focuses on nastiest 20% corruptions: lower = fragile, higher = robust",
        out_path=os.path.join(out_dir, "nr_worst20.png")
    )

    # Task-aware Usability vs Safety
    plt.figure(figsize=(8,6))
    melted = task_df.melt(
        id_vars=["detector","fpr_target","n"],
        value_vars=["ta_nr","safety_catch"],
        var_name="metric",
        value_name="rate"
    )
    sns.barplot(
        data=melted,
        x="detector",
        y="rate",
        hue="metric"
    )
    plt.title("Task-aware Usability vs Safety\n"
              "(TA-NR = fraction of good accepts; SafetyCatch = fraction of bad rejections)",
              fontsize=12)
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "taskaware.png"), dpi=200)
    plt.close()
    print(f"[PLOT] Saved taskaware.png")


# -----------------------------------
# Example Usage
# -----------------------------------
if __name__ == "__main__":
    # Assuming metrics.py already dumped CSV summaries
    nr_df    = pd.read_csv("results/metrics/nr_at_fpr.csv")
    auc_df   = pd.read_csv("results/metrics/nr_auc.csv")
    worst_df = pd.read_csv("results/metrics/nr_worst20.csv")
    task_df  = pd.read_csv("results/metrics/taskaware.csv")

    plot_head_to_head(nr_df, auc_df, worst_df, task_df, out_dir="results/metrics_plots")
