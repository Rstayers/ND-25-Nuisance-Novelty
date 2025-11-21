# analysis/plots/compare_detectors.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from analysis.compute import aggregate_outcomes
from analysis.plots.utils_plot import ensure_dir, joinp, set_mpl_curve_style, OUTCOME_ORDER


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


def _set_relative_ylim(ax, values, pad_frac: float = 0.05):
    """Tight y-range around the data (so 0.90 vs 0.95 is visible)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    if vmin == vmax:
        lo = vmin - 0.02
        hi = vmin + 0.02
    else:
        span = vmax - vmin
        lo = vmin - pad_frac * span
        hi = vmax + pad_frac * span
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    if hi <= lo:
        lo, hi = 0.0, 1.0
    ax.set_ylim(lo, hi)


def plot_outcome_curves_all_detectors(
    df,
    out_root: str,
    dataset_label: str,
    fpr_focus: float = 0.05,
):
    """
    For a given dataset (e.g., 'cns' or 'imagenet_c'), plot outcome curves
    **per backbone** comparing detectors:

        - Directory: detector_compare/backbone=<backbone>/
        - X-axis: severity
        - Y-axis: outcome proportion (relative y-axis)
        - Line: detector

    No averaging over backbones: each backbone gets its own detector comparison.
    """
    df = df.copy()

    # Filter to the dataset
    if "dataset" in df.columns and dataset_label:
        mask = df["dataset"].astype(str).str.contains(
            dataset_label, case=False, na=False
        )
        if mask.any():
            df = df[mask].copy()

    if "severity" not in df.columns:
        print(f"[WARN] No 'severity' column for {dataset_label}; skipping detector comparison.")
        return

    group_cols = ["backbone", "detector", "fpr_target", "severity"]
    comp, outcome_cols = aggregate_outcomes(df, group_cols)
    if comp.empty:
        print("[WARN] No outcome aggregate data for detector comparison.")
        return

    comp = _filter_by_fpr(comp, fpr_focus)
    if comp.empty:
        print(f"[WARN] No outcome data at FPR={fpr_focus} for detector comparison.")
        return

    friendly_ds = _friendly_dataset_name(dataset_label)
    ds_part = f" — {friendly_ds}" if friendly_ds else ""

    # We'll create detector_compare/backbone=<bb>/ for each backbone
    base_dir = joinp(out_root, "detector_compare")
    ensure_dir(base_dir)

    backbones = sorted(comp["backbone"].dropna().unique())
    for bb in backbones:
        bb_comp = comp[comp["backbone"] == bb].copy()
        if bb_comp.empty:
            continue

        sev_vals = sorted(bb_comp["severity"].dropna().unique())
        detectors = sorted(bb_comp["detector"].dropna().unique())
        if not detectors:
            continue

        colors = plt.cm.Dark2.colors
        cmap = {det: colors[i % len(colors)] for i, det in enumerate(detectors)}

        save_dir = joinp(base_dir, f"backbone={bb}")
        ensure_dir(save_dir)

        for outcome_name in OUTCOME_ORDER:
            if outcome_name not in outcome_cols:
                continue

            prop_col = f"{outcome_name}_prop"
            if prop_col not in bb_comp.columns:
                continue

            # For a fixed backbone, each (detector, severity) should be unique,
            # but we group just in case to be robust.
            avg = (
                bb_comp.groupby(["detector", "severity"])[prop_col]
                      .mean()
                      .reset_index()
            )
            if avg.empty:
                continue

            set_mpl_curve_style()
            fig, ax = plt.subplots(figsize=(7.0, 4.2))

            all_y = []

            for det in detectors:
                sub = avg[avg["detector"] == det].sort_values("severity")
                if sub.empty:
                    continue
                yvals = sub[prop_col].values
                all_y.extend(list(yvals))
                ax.plot(
                    sub["severity"],
                    yvals,
                    marker="o",
                    linewidth=2.0,
                    label=det,
                    color=cmap[det],
                )

            ax.set_xticks(sev_vals)
            ax.set_xlabel("Severity / Scale")
            ax.set_ylabel("Proportion of Samples")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            _set_relative_ylim(ax, all_y, pad_frac=0.06)

            title = (
                f"{outcome_name} vs Severity{ds_part}\n"
                f"Backbone = {bb} — Detector comparison (FPR={fpr_focus:.2f})"
            )
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.35)

            ncols = min(len(detectors), 4)
            ax.legend(
                title="Detector",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=ncols,
                frameon=False,
            )

            fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
            fname = f"{outcome_name}_vs_detectors_{dataset_label or 'all'}_{bb}_FPR{fpr_focus:.2f}.png"
            fname = fname.replace(" ", "_")
            out_path = joinp(save_dir, fname)
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            print(f"[OK] Saved detector comparison for {outcome_name}, backbone={bb} → {out_path}")
