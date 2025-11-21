# analysis/plots/overview.py
"""
Overview dashboard for Psycho Benchmark
---------------------------------------
Summarizes classifier accuracy, OOD decision accuracy, total nuisance, and outcome composition.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from analysis.compute import aggregate_rates, aggregate_outcomes
from analysis.plots.utils_plot import ensure_dir, joinp, OUTCOME_ORDER, OUTCOME_COLORS


def plot_overview_dashboard(df, out_root: str, fpr_focus: float = 0.05):
    save_dir = joinp(out_root, "overview")
    ensure_dir(save_dir)
    sns.set_theme(style="darkgrid", context="talk")

    # Filter FPR
    if "fpr_target" in df.columns:
        df_fpr = df[np.isclose(df["fpr_target"], fpr_focus)].copy()
        if df_fpr.empty:
            print(f"[WARN] No rows at FPR={fpr_focus}; using all rows for overview.")
            df_fpr = df.copy()
    else:
        df_fpr = df.copy()

    # --- 1. Classifier accuracy vs severity ---
    cls_acc = (
        df_fpr.groupby(["backbone", "severity"], dropna=False)
        .agg(acc=("correct_cls", "mean"), n=("correct_cls", "size"))
        .reset_index()
    )
    if not cls_acc.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for bb, sub in cls_acc.groupby("backbone"):
            sub = sub.sort_values("severity")
            ax.plot(sub["severity"], sub["acc"], marker="o", linewidth=2.0, label=bb)
        ax.set_xlabel("Severity / Scale")
        ax.set_ylabel("Classifier Accuracy")
        ax.set_title(f"Classifier Accuracy vs Severity (FPR={fpr_focus:.2f})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title="Backbone", frameon=False)
        fig.tight_layout()
        fig.savefig(joinp(save_dir, f"overview_classifier_accuracy_FPR{fpr_focus:.2f}.png"))
        plt.close(fig)

    # --- 2. OOD decision accuracy vs severity ---
    if "accept" in df_fpr.columns:
        ood_acc = (
            df_fpr.groupby(["backbone", "severity"], dropna=False)
            .agg(ood_acc=("accept", "mean"))
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        for bb, sub in ood_acc.groupby("backbone"):
            sub = sub.sort_values("severity")
            ax.plot(sub["severity"], sub["ood_acc"], marker="^", linewidth=2.0, label=bb)
        ax.set_xlabel("Severity / Scale")
        ax.set_ylabel("OOD Decision Accuracy")
        ax.set_title(f"OOD Decision Accuracy vs Severity (FPR={fpr_focus:.2f})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(title="Backbone", frameon=False)
        fig.tight_layout()
        fig.savefig(joinp(save_dir, f"overview_ood_accuracy_FPR{fpr_focus:.2f}.png"))
        plt.close(fig)

    # --- 3. Total nuisance ---
    met = aggregate_rates(df_fpr, ["backbone", "detector", "fpr_target", "severity"])
    met["total_nuisance"] = met["full_nuisance"] + met["partial_nuisance"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for bb, sub in met.groupby("backbone"):
        sbb = (
            sub.groupby("severity")
            .agg(total_nuisance=("total_nuisance", "mean"))
            .reset_index()
            .sort_values("severity")
        )
        ax.plot(sbb["severity"], sbb["total_nuisance"], marker="s", linewidth=2.0, label=bb)
    ax.set_xlabel("Severity / Scale")
    ax.set_ylabel("Total Nuisance")
    ax.set_title(f"Nuisance vs Severity (FPR={fpr_focus:.2f})")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Backbone", frameon=False)
    fig.tight_layout()
    fig.savefig(joinp(save_dir, f"overview_total_nuisance_FPR{fpr_focus:.2f}.png"))
    plt.close(fig)

    # --- 4. Outcome composition ---
    comp, outcome_cols = aggregate_outcomes(df_fpr, ["backbone", "fpr_target", "severity"])
    if comp.empty:
        return
    comp = comp[np.isclose(comp["fpr_target"], fpr_focus)]
    order = [c for c in OUTCOME_ORDER if c in outcome_cols]
    prop_cols = [f"{c}_prop" for c in order]

    for bb, sub in comp.groupby("backbone"):
        sub = sub.sort_values("severity")
        sev_vals = sorted(sub["severity"].unique())
        bottoms = np.zeros(len(sub))
        fig, ax = plt.subplots(figsize=(9, 6))
        for col, prop_col in zip(order, prop_cols):
            vals = sub[prop_col].values
            ax.bar(sev_vals, vals, bottom=bottoms, label=col, color=OUTCOME_COLORS[col])
            bottoms += vals
        ax.set_xlabel("Severity / Scale")
        ax.set_ylabel("Proportion of Outcomes")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"{bb} — Outcome Composition (FPR={fpr_focus:.2f})")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(order),
            frameon=False,
            title="Outcome",
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fname = f"overview_composition_{bb}_FPR{fpr_focus:.2f}.png"
        fig.savefig(joinp(save_dir, fname))
        plt.close(fig)

    print(f"[OK] Overview dashboard saved to {save_dir}")
