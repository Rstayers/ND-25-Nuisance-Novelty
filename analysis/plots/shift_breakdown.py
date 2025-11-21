# analysis/plots/shift_breakdown.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.compute import aggregate_rates, aggregate_outcomes, per_backbone
from .utils_plot import (
    ensure_dir,
    joinp,
    set_seaborn_style,
    OUTCOME_ORDER,
    OUTCOME_COLORS,
)


def _pretty(name: str) -> str:
    return str(name).replace("_", " ").replace("jpeg", "JPEG").title()


# ---------------------------------------------------------------------
# 1) Outcome composition for a single backbone on CNS
# ---------------------------------------------------------------------
def plot_cns_outcome_composition_single(
    df,
    out_root: str,
    backbone: str = "resnet50",
    detector: str = "msp",
    fpr_focus: float = 0.05,
):
    set_seaborn_style()
    save_dir = joinp(out_root, "composition_single")
    ensure_dir(save_dir)

    sub = df[df["backbone"] == backbone].copy()
    sub = sub[sub["detector"] == detector]
    if sub.empty:
        print(f"[WARN] No CNS rows for backbone={backbone}, detector={detector}")
        return

    comp, outcome_cols = aggregate_outcomes(
        sub, ["backbone", "detector", "fpr_target", "severity"]
    )
    if comp.empty:
        print("[WARN] No outcome composition data.")
        return

    comp = comp[comp["fpr_target"].sub(fpr_focus).abs() < 1e-8]
    if comp.empty:
        print(f"[WARN] No composition rows at FPR={fpr_focus}")
        return

    comp = comp.sort_values("severity")
    sev_vals = sorted(comp["severity"].unique())
    x = comp["severity"].values
    bottoms = np.zeros(len(comp))

    order = [c for c in OUTCOME_ORDER if c in outcome_cols]
    prop_cols = [f"{c}_prop" for c in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    for col, prop_col in zip(order, prop_cols):
        vals = comp[prop_col].values
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=col,
            color=OUTCOME_COLORS[col],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += vals

    ax.set_xlim(min(sev_vals) - 0.1, max(sev_vals) + 0.1)
    ax.set_xticks(sev_vals)
    ax.set_xlabel("Severity / Scale")
    ax.set_ylabel("Proportion of Outcomes")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"{backbone} / {detector} — Outcome Composition (FPR={fpr_focus:.2f})")
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(order),
        frameon=False,
        title="Outcome",
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    out_path = joinp(save_dir, f"composition_{backbone}_{detector}_FPR{fpr_focus:.2f}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[OK] Saved CNS outcome composition → {out_path}")


# ---------------------------------------------------------------------
# 2) Head-to-head backbones on total nuisance (CNS)
# ---------------------------------------------------------------------
def plot_cns_head2head_total_nuisance(
    df,
    out_root: str,
    detector: str = "msp",
    fpr_focus: float = 0.05,
):
    set_seaborn_style()
    save_dir = joinp(out_root, "head2head_backbones")
    ensure_dir(save_dir)

    met = aggregate_rates(df, ["backbone", "detector", "fpr_target", "severity"])
    met = met[met["fpr_target"].sub(fpr_focus).abs() < 1e-8]
    met = met[met["detector"] == detector]
    if met.empty:
        print(f"[WARN] No backbone head-to-head data at FPR={fpr_focus}, detector={detector}.")
        return

    met["total_nuisance"] = met["full_nuisance"] + met["partial_nuisance"]

    fig, ax = plt.subplots(figsize=(10, 6))
    backbones = sorted(met["backbone"].dropna().unique())
    colors = plt.cm.tab10.colors
    cmap = {bb: colors[i % len(colors)] for i, bb in enumerate(backbones)}

    for bb in backbones:
        sbb = met[met["backbone"] == bb].sort_values("severity")
        if sbb.empty:
            continue
        sev_vals = sorted(sbb["severity"].unique())
        ax.plot(
            sbb["severity"],
            sbb["total_nuisance"],
            marker="o",
            linewidth=2.0,
            color=cmap[bb],
            label=bb,
        )
        ax.set_xticks(sev_vals)

    ax.set_xlabel("Severity / Scale")
    ax.set_ylabel("Total Nuisance\n(Nuisance_Novelty + Double_Failure)")
    ax.set_title(f"{detector} — Backbones vs Severity (FPR={fpr_focus:.2f})")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(
        title="Backbone",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    out_path = joinp(save_dir, f"bb_h2h_total_nuisance_{detector}_FPR{fpr_focus:.2f}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[OK] Saved CNS backbone head-to-head → {out_path}")


# ---------------------------------------------------------------------
# 3) CNS shift heatmaps for a chosen backbone
# ---------------------------------------------------------------------
def plot_cns_shift_heatmap(
    df,
    out_root: str,
    backbone: str,
    detector: str = "msp",
    top_n: int = 15,
    fpr_focus: float = 0.05,
):
    """
    Heatmaps per (corruption, severity) showing:
      - Nuisance_Novelty (full_nuisance)
      - Double_Failure (partial_nuisance)
    for a single backbone+detector on CNS.
    """
    set_seaborn_style()
    save_dir = joinp(out_root, "heatmaps", f"backbone={backbone}")
    ensure_dir(save_dir)

    sub = df[df["backbone"] == backbone].copy()
    sub = sub[sub["detector"] == detector]
    if sub.empty:
        print(f"[WARN] No CNS rows for backbone={backbone}, detector={detector}")
        return

    met = aggregate_rates(
        sub, ["backbone", "detector", "fpr_target", "shift", "severity"]
    )
    met = met[met["fpr_target"].sub(fpr_focus).abs() < 1e-8]
    if met.empty:
        print(f"[WARN] No aggregate data at FPR={fpr_focus}.")
        return

    # Rename shift -> corruption to reuse pretty naming logic
    met = met.rename(columns={"shift": "corruption"})

    # Select top shifts by Double_Failure
    top = (
        met.groupby("corruption")["partial_nuisance"]
           .mean()
           .sort_values(ascending=False)
           .head(top_n)
           .index
    )
    met = met[met["corruption"].isin(top)].copy()
    if met.empty:
        print("[WARN] No rows after selecting top corruptions.")
        return

    met["corruption_pretty"] = met["corruption"].map(_pretty)
    order = [_pretty(c) for c in top]
    met["corruption_pretty"] = met["corruption_pretty"].astype("category")
    met["corruption_pretty"] = met["corruption_pretty"].cat.set_categories(order, ordered=True)

    sev_vals = sorted(met["severity"].unique())

    p_full = met.pivot_table(
        index="corruption_pretty",
        columns="severity",
        values="full_nuisance",
        aggfunc="mean",
        observed=False,
    ).reindex(index=order, columns=sev_vals)

    p_part = met.pivot_table(
        index="corruption_pretty",
        columns="severity",
        values="partial_nuisance",
        aggfunc="mean",
        observed=False,
    ).reindex(index=order, columns=sev_vals)

    vals = np.concatenate([p_full.values.ravel(), p_part.values.ravel()])
    vals = vals[~np.isnan(vals)]
    vmax = max(0.05, float(np.quantile(vals, 0.95))) if len(vals) else 1.0
    vmin = 0.0

    fig = plt.figure(figsize=(12, max(6, 0.45 * len(order))))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.0)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax2.yaxis.set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="6%", pad=0.05)

    sns.heatmap(
        p_full,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax1,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )

    hm2 = sns.heatmap(
        p_part,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax2,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": "Percent of samples"},
        square=True,
        linewidths=0.5,
        linecolor="white",
    )

    ax1.set_title("Nuisance_Novelty (correct → reject)")
    ax1.set_xlabel("Severity / Scale")
    ax1.set_ylabel("Shift / Corruption")
    ax1.set_xticklabels(sev_vals, rotation=0)

    ax2.set_title("Double_Failure (incorrect → reject)")
    ax2.set_xlabel("Severity / Scale")
    ax2.set_xticklabels(sev_vals, rotation=0)

    fig.suptitle(
        f"{backbone} / {detector} — Nuisance heatmaps (FPR={fpr_focus:.2f})",
        fontsize=14,
    )

    cb = hm2.collections[0].colorbar
    cb.formatter = mtick.PercentFormatter(1.0)
    cb.update_ticks()

    fig.subplots_adjust(wspace=0.0)
    fname = f"heatmaps_{backbone}_{detector}_FPR{fpr_focus:.2f}.png".replace("/", "_")
    out_path = joinp(save_dir, fname)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved CNS shift heatmap → {out_path}")
