# analysis/plots/composition.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from analysis.compute import aggregate_outcomes
from analysis.plots.utils_plot import (
    ensure_dir,
    joinp,
    set_mpl_curve_style,
    OUTCOME_ORDER,
    OUTCOME_COLORS,
)




def _filter_by_fpr(df, fpr_focus: float):
    if "fpr_target" in df.columns:
        out = df[df["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if out.empty:
            print(f"[WARN] No rows at FPR={fpr_focus}; using all rows instead.")
            return df.copy()
        return out
    return df.copy()


def _friendly_dataset_name(dataset_label: str) -> str:
    if not dataset_label:
        return ""
    lower = dataset_label.lower()
    if "imagenet_c" in lower:
        return "ImageNet-C"
    if "cns" in lower:
        return "CNS-Bench"
    return dataset_label


# ---------------- 1) Composition by backbone ----------------

def plot_composition_by_backbone(
    df,
    out_root: str,
    dataset_label: str,
    fpr_focus: float = 0.05,
):
    """
    For each backbone and dataset, plot stacked composition of the 4 outcomes
    vs severity, averaged over detectors.

    Each bar at a given severity sums to 1.0.
    """
    df = df.copy()
    if "dataset" in df.columns and dataset_label:
        mask = df["dataset"].astype(str).str.contains(
            dataset_label, case=False, na=False
        )
        if mask.any():
            df = df[mask].copy()

    if "severity" not in df.columns:
        print(f"[WARN] No 'severity' column for {dataset_label}; skipping composition.")
        return

    group_cols = ["backbone", "detector", "fpr_target", "severity"]
    comp, outcome_cols = aggregate_outcomes(df, group_cols)
    if comp.empty:
        print("[WARN] No outcome aggregate data for composition.")
        return

    comp = _filter_by_fpr(comp, fpr_focus)
    if comp.empty:
        print(f"[WARN] No composition data at FPR={fpr_focus}.")
        return

    save_dir = joinp(out_root, "composition_backbones")
    ensure_dir(save_dir)

    friendly_ds = _friendly_dataset_name(dataset_label)
    ds_part = f" — {friendly_ds}" if friendly_ds else ""

    order = [c for c in OUTCOME_ORDER if c in outcome_cols]
    prop_cols = [f"{c}_prop" for c in order]

    # Average proportions over detectors for each backbone × severity
    avg = (
        comp.groupby(["backbone", "severity"])[prop_cols]
            .mean()
            .reset_index()
    )

    backbones = sorted(avg["backbone"].dropna().unique())

    for bb in backbones:
        sub = avg[avg["backbone"] == bb].sort_values("severity")
        if sub.empty:
            continue

        sev_vals = sub["severity"].values
        x = np.arange(len(sev_vals))

        set_mpl_curve_style()
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        bottoms = np.zeros(len(sub))
        for outcome_name, prop_col in zip(order, prop_cols):
            if prop_col not in sub.columns:
                continue
            vals = sub[prop_col].values
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=outcome_name,
                color=OUTCOME_COLORS.get(outcome_name, None),
                edgecolor="white",
                linewidth=0.2,
            )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sev_vals])
        ax.set_xlabel("Severity / Scale")
        ax.set_ylabel("Proportion of Outcomes")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        title = f"{bb} — Outcome Composition vs Severity{ds_part} (FPR={fpr_focus:.2f})"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.25, axis="y")

        ax.legend(
            title="Outcome",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(order),
            frameon=False,
        )

        fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
        fname = f"composition_backbone={bb}_{dataset_label or 'all'}_FPR{fpr_focus:.2f}.png"
        fname = fname.replace("/", "_")
        out_path = joinp(save_dir, fname)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        print(f"[OK] Saved backbone composition → {out_path}")


# ---------------- 2) Composition by detector (averaged over backbones) ----------------

def plot_composition_by_detector(
    df,
    out_root: str,
    dataset_label: str,
    fpr_focus: float = 0.05,
):
    """
    For each detector and dataset, plot stacked composition vs severity,
    averaged over *backbones*.

    This is the detector-compare view in composition form.
    """
    df = df.copy()
    if "dataset" in df.columns and dataset_label:
        mask = df["dataset"].astype(str).str.contains(
            dataset_label, case=False, na=False
        )
        if mask.any():
            df = df[mask].copy()

    if "severity" not in df.columns:
        print(f"[WARN] No 'severity' column for {dataset_label}; skipping detector composition.")
        return

    group_cols = ["backbone", "detector", "fpr_target", "severity"]
    comp, outcome_cols = aggregate_outcomes(df, group_cols)
    if comp.empty:
        print("[WARN] No outcome aggregate data for detector composition.")
        return

    comp = _filter_by_fpr(comp, fpr_focus)
    if comp.empty:
        print(f"[WARN] No detector composition data at FPR={fpr_focus}.")
        return

    save_dir = joinp(out_root, "composition_detectors")
    ensure_dir(save_dir)

    friendly_ds = _friendly_dataset_name(dataset_label)
    ds_part = f" — {friendly_ds}" if friendly_ds else ""

    order = [c for c in OUTCOME_ORDER if c in outcome_cols]
    prop_cols = [f"{c}_prop" for c in order]

    # Average over backbones for each detector × severity
    avg = (
        comp.groupby(["detector", "severity"])[prop_cols]
            .mean()
            .reset_index()
    )

    detectors = sorted(avg["detector"].dropna().unique())

    for det in detectors:
        sub = avg[avg["detector"] == det].sort_values("severity")
        if sub.empty:
            continue

        sev_vals = sub["severity"].values
        x = np.arange(len(sev_vals))

        set_mpl_curve_style()
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        bottoms = np.zeros(len(sub))
        for outcome_name, prop_col in zip(order, prop_cols):
            if prop_col not in sub.columns:
                continue
            vals = sub[prop_col].values
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=outcome_name,
                color=OUTCOME_COLORS.get(outcome_name, None),
                edgecolor="white",
                linewidth=0.2,
            )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in sev_vals])
        ax.set_xlabel("Severity / Scale")
        ax.set_ylabel("Proportion of Outcomes")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        title = (
            f"{det} — Outcome Composition vs Severity"
            f"{ds_part} (avg over backbones, FPR={fpr_focus:.2f})"
        )
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.25, axis="y")

        ax.legend(
            title="Outcome",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(order),
            frameon=False,
        )

        fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
        fname = f"composition_detector={det}_{dataset_label or 'all'}_FPR{fpr_focus:.2f}.png"
        fname = fname.replace("/", "_")
        out_path = joinp(save_dir, fname)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        print(f"[OK] Saved detector composition → {out_path}")