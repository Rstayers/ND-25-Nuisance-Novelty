# analysis/plots/scale_curves.py


from analysis.plots.utils_plot import ensure_dir, joinp, set_seaborn_style, set_mpl_curve_style
# analysis/plots/scale_curves.py

import matplotlib.pyplot as plt




def _filter_by_fpr(df, fpr_focus: float):
    if "fpr_target" in df.columns:
        out = df[df["fpr_target"].sub(fpr_focus).abs() < 1e-8].copy()
        if out.empty:
            print(f"[WARN] No rows at FPR={fpr_focus}; using all rows instead.")
            return df.copy()
        return out
    return df.copy()


def _prepare_df(df, fpr_focus: float, dataset_label: str):
    df = df.copy()
    if "dataset" in df.columns and dataset_label:
        mask = df["dataset"].astype(str).str.contains(dataset_label, case=False, na=False)
        if mask.any():
            df = df[mask].copy()

    if "severity" not in df.columns:
        print(f"[WARN] No 'severity' column for {dataset_label}; skipping.")
        return None

    df = _filter_by_fpr(df, fpr_focus)
    return df if not df.empty else None


def _friendly_dataset_name(dataset_label: str) -> str:
    if not dataset_label:
        return ""
    lower = dataset_label.lower()
    if "imagenet_c" in lower:
        return "ImageNet-C"
    if "cns" in lower:
        return "CNS-Bench"
    return dataset_label


def _set_relative_ylim(ax, values, pad_frac: float = 0.05, clamp01: bool = True):
    vals = [v for v in values if v is not None]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    if vmin == vmax:
        # Degenerate case: single value
        lo = vmin - 0.02
        hi = vmin + 0.02
    else:
        span = vmax - vmin
        lo = vmin - pad_frac * span
        hi = vmax + pad_frac * span
    if clamp01:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if hi <= lo:  # safety
            lo, hi = 0.0, 1.0
    ax.set_ylim(lo, hi)


def _plot_curve_core(
    grouped,
    y_col: str,
    y_label: str,
    title_prefix: str,
    out_path: str,
):
    """Shared plotting logic for the two accuracy curves."""
    if grouped.empty:
        print("[WARN] Empty grouped data in _plot_curve_core.")
        return

    set_mpl_curve_style()

    backbones = sorted(grouped["backbone"].dropna().unique())
    colors = plt.cm.tab10.colors
    cmap = {bb: colors[i % len(colors)] for i, bb in enumerate(backbones)}

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    all_y = []

    for bb in backbones:
        sub = grouped[grouped["backbone"] == bb].sort_values("severity")
        if sub.empty:
            continue
        sev_vals = sorted(sub["severity"].dropna().unique())
        ys = sub[y_col].values
        all_y.extend(list(ys))
        ax.plot(
            sub["severity"],
            ys,
            marker="o",
            linewidth=2.0,
            label=bb,
            color=cmap[bb],
        )
        ax.set_xticks(sev_vals)

    ax.set_xlabel("Severity / Scale")
    ax.set_ylabel(y_label)
    ax.set_title(title_prefix)
    _set_relative_ylim(ax, all_y, pad_frac=0.06, clamp01=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Legend below the plot in multiple columns
    ncols = min(len(backbones), 4) if backbones else 1
    ax.legend(
        title="Backbone",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncols,
        frameon=False,
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[OK] Saved curve → {out_path}")


def plot_classifier_accuracy(
    df,
    out_root: str,
    fpr_focus: float = 0.05,
    dataset_label: str = "",
):
    """
    Classifier accuracy vs severity per backbone, averaged over detectors.

    Used for CNS-Bench and ImageNet-C (filter by dataset name if provided).
    """
    df_plot = _prepare_df(df, fpr_focus, dataset_label)
    if df_plot is None:
        return

    save_dir = joinp(out_root, "accuracy_curves")
    ensure_dir(save_dir)

    grouped = (
        df_plot.groupby(["backbone", "severity"])["correct_cls"]
               .mean()
               .reset_index()
    )
    if grouped.empty:
        print("[WARN] No classifier accuracy data after grouping.")
        return

    friendly = _friendly_dataset_name(dataset_label)
    ds_part = f" — {friendly}" if friendly else ""
    title = f"Classifier Accuracy vs Severity{ds_part} (FPR={fpr_focus:.2f})"

    out_path = joinp(
        save_dir,
        f"classifier_accuracy_{dataset_label or 'all'}_FPR{fpr_focus:.2f}.png",
    )
    _plot_curve_core(
        grouped=grouped,
        y_col="correct_cls",
        y_label="Classifier Accuracy",
        title_prefix=title,
        out_path=out_path,
    )


def plot_detector_accuracy(
    df,
    out_root: str,
    fpr_focus: float = 0.05,
    dataset_label: str = "",
):
    """
    Detector/ID decision accuracy vs severity per backbone, averaged over detectors.

    For ID-shift datasets (CNS, ImageNet-C) the "correct" decision is to ACCEPT,
    so we use mean(accept).
    """
    if "accept" not in df.columns:
        print("[WARN] 'accept' column missing; cannot compute detector accuracy.")
        return

    df_plot = _prepare_df(df, fpr_focus, dataset_label)
    if df_plot is None:
        return

    save_dir = joinp(out_root, "detector_curves")
    ensure_dir(save_dir)

    grouped = (
        df_plot.groupby(["backbone", "severity"])["accept"]
               .mean()
               .reset_index()
    )
    if grouped.empty:
        print("[WARN] No detector accuracy data after grouping.")
        return

    friendly = _friendly_dataset_name(dataset_label)
    ds_part = f" — {friendly}" if friendly else ""
    title = f"Detector Accuracy vs Severity{ds_part} (FPR={fpr_focus:.2f})"

    out_path = joinp(
        save_dir,
        f"detector_accuracy_{dataset_label or 'all'}_FPR{fpr_focus:.2f}.png",
    )
    _plot_curve_core(
        grouped=grouped,
        y_col="accept",
        y_label="ID Decision Accuracy (accept = 1)",
        title_prefix=title,
        out_path=out_path,
    )