# analysis/plots.py
# Paper-ready plots with cohesive color scheme (Paul Tol's colorblind-friendly palette)

import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from analysis.processing import (
    get_metrics_by_severity,
    get_mean_metrics,
    compute_adr,
)

# =============================================================================
# COHESIVE COLOR SCHEME - Paul Tol's qualitative palette (colorblind-friendly)
# =============================================================================

# Backbone colors - distinct, professional
BACKBONE_COLORS = {
    'convnext_t': '#4477AA',  # Blue
    'densenet121': '#EE6677',  # Rose/Pink
    'resnet50': '#228833',  # Green
    'swin_t': '#CCBB44',  # Yellow/Gold
    'vit_b_16': '#AA3377',  # Purple/Magenta
}

# Backbone display names (clean for legends)
BACKBONE_NAMES = {
    'convnext_t': 'ConvNeXt-T',
    'densenet121': 'DenseNet-121',
    'resnet50': 'ResNet-50',
    'swin_t': 'Swin-T',
    'vit_b_16': 'ViT-B/16',
}

# Detector colors - maximally distinct, colorblind-friendly
DETECTOR_COLORS = {
    'msp': '#E69F00',  # Orange
    'maxlogit': '#56B4E9',  # Sky blue
    'ebo': '#009E73',  # Bluish green
    'odin': '#F0E442',  # Yellow
    'react': '#0072B2',  # Blue
    'ash': '#D55E00',  # Vermillion
    'dice': '#CC79A7',  # Reddish purple
    'knn': '#000000',  # Black
    'mds': '#882255',  # Wine
    'vim': '#44AA99',  # Teal
    'she': '#332288',  # Indigo
    'postmax': '#7F7F7F',  # Darker gray (was #BBBBBB - too washed out)
    'costarr': '#117733',  # Dark green
}

# Line styles for detectors (for trajectory plots)
DETECTOR_LINESTYLES = {
    'msp': '-',
    'odin': '--',
    'react': '-.',
    'dice': ':',
    'knn': '-',
    'mds': '--',
    'vim': '-.',
    'she': ':',
    'postmax': '-',
    'costarr': '--',
}

# Metric colors
COLORS = {
    'CSA': '#4477AA',  # Blue
    'CCR': '#EE6677',  # Rose
    'NNR': '#CCBB44',  # Gold
    'OSA_Gap': '#AA3377',  # Purple
    'OSA': '#228833',  # Green
}

# Dataset colors
DATASET_COLORS = {
    'ImageNet-LN': '#228833',  # Green
    'ImageNet-C': '#EE6677',  # Rose
    'CNS': '#4477AA',  # Blue
    'CUB-LN': '#AA3377',  # Purple
    'Cars-LN': '#CCBB44',  # Gold
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})


def get_backbone_color(bb):
    """Get color for backbone, with fallback."""
    return BACKBONE_COLORS.get(bb, '#666666')


def get_backbone_name(bb):
    """Get display name for backbone."""
    return BACKBONE_NAMES.get(bb, bb)


def get_detector_color(det):
    """Get color for detector, with fallback."""
    return DETECTOR_COLORS.get(det.lower(), '#666666')


def get_detector_linestyle(det):
    """Get line style for detector, with fallback."""
    return DETECTOR_LINESTYLES.get(det.lower(), '-')


# =============================================================================
# OSA-NNR TRADEOFF LANDSCAPE
# =============================================================================

def plot_osa_nnr_landscape(agg_df, dataset, out_dir):
    """
    OSA vs NNR scatter plot: shows where configs cluster in OSA-NNR space.

    Each point = (backbone, detector, severity_level) tuple
    Point size = severity level (larger = higher severity)
    Color = backbone
    Generates separate plots for each OOD test dataset.
    Saves to osa_nnr/ folder.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0 (e.g., clean validation set)

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print(f"  > osa_nnr/{dataset} skipped (no OSA data)")
        return

    subdir = os.path.join(out_dir, "osa_nnr")
    os.makedirs(subdir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in per_sev.columns:
        ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in per_sev.columns:
            ood_data = per_sev[per_sev["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = per_sev
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        fig, ax = plt.subplots(figsize=(10, 8))

        # Quadrant shading
        ax.axhspan(0.7, 1.0, xmin=0, xmax=0.3, alpha=0.1, color='green', zorder=0)  # Ideal
        ax.axhspan(0, 0.5, xmin=0.5, xmax=1.0, alpha=0.1, color='red', zorder=0)    # Collapse

        # Reference lines
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Plot each (backbone, detector) trajectory
        for backbone in backbones:
            for detector in detectors:
                mask = (ood_data["backbone"] == backbone) & (ood_data["detector"] == detector)
                config_data = ood_data[mask].sort_values("level")

                if len(config_data) < 2:
                    continue

                x = config_data["NNR"].values
                y = config_data["OSA"].values
                levels = config_data["level"].values

                color = get_backbone_color(backbone)

                # Point sizes based on severity level (larger = higher severity)
                sizes = [40 + 25 * lv for lv in levels]
                ax.scatter(x, y, c=[color]*len(x), s=sizes, edgecolors='white',
                           linewidths=0.5, zorder=3, alpha=0.8)

        # Quadrant labels
        ax.text(0.05, 0.95, 'IDEAL', transform=ax.transAxes, fontsize=10,
                fontweight='bold', color='green', alpha=0.7, va='top')
        ax.text(0.85, 0.05, 'COLLAPSE', transform=ax.transAxes, fontsize=10,
                fontweight='bold', color='red', alpha=0.7, va='bottom', ha='right')

        ax.set_xlabel('NNR', fontsize=12)
        ax.set_ylabel('OSA', fontsize=12)
        ax.set_xlim(0, min(1.0, ood_data["NNR"].max() * 1.1))
        ax.set_ylim(max(0, ood_data["OSA"].min() * 0.9), 1.0)
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')

        # Legend - backbone colors
        from matplotlib.lines import Line2D
        bb_handles = [Line2D([0], [0], color=get_backbone_color(bb), linewidth=3,
                             label=get_backbone_name(bb)) for bb in backbones]

        # Legend - severity markers
        sev_handles = [Line2D([0], [0], marker='o', color='gray', markersize=4 + 2*i,
                              linestyle='', label=f'L{i+1}') for i in range(5)]

        leg1 = ax.legend(handles=bb_handles, title='Backbone', loc='upper right',
                         fontsize=9, title_fontsize=10)
        ax.add_artist(leg1)
        ax.legend(handles=sev_handles, title='Severity', loc='lower left',
                  fontsize=8, title_fontsize=9)

        safe = dataset.replace("-", "_")
        fname = os.path.join(subdir, f"landscape_{safe}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > osa_nnr/landscape_{dataset.replace('-', '_')} ({len(ood_datasets)} files)")


def plot_osa_nnr_landscape_faceted(agg_df, out_dir):
    """
    Faceted OSA-NNR scatter plot: 1x3 grid comparing all nuisance datasets.

    Each point = (backbone, detector, severity_level) tuple
    Point size = severity level (larger = higher severity)
    Color = backbone
    Generates separate plots for each OOD test dataset (NINCO, Open-O).
    Saves to osa_nnr/ folder.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print("  > osa_nnr/ skipped (no OSA data)")
        return

    subdir = os.path.join(out_dir, "osa_nnr")
    os.makedirs(subdir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in per_sev.columns:
        ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in per_sev.columns:
            ood_data = per_sev[per_sev["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = per_sev
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
        ds_order = ds_order[:3]

        if len(ds_order) == 0:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        n_ds = len(ds_order)
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), squeeze=False)
        axes = axes.flatten()

        # Severity level markers (different shapes)
        severity_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'P'}  # circle, square, triangle, diamond, plus
        marker_size = 70  # Uniform size for all severity levels

        for ds_idx, dataset in enumerate(ds_order):
            ax = axes[ds_idx]
            ds_data = ood_data[ood_data["dataset"] == dataset]

            for backbone in backbones:
                for detector in detectors:
                    mask = (ds_data["backbone"] == backbone) & (ds_data["detector"] == detector)
                    config_data = ds_data[mask].sort_values("level")

                    if len(config_data) < 2:
                        continue

                    color = get_backbone_color(backbone)

                    # Plot each severity level with different marker shape
                    for _, row in config_data.iterrows():
                        lv = int(row["level"])
                        ax.scatter(row["NNR"], row["OSA"], c=[color], s=marker_size,
                                  marker=severity_markers.get(lv, 'o'),
                                  edgecolors='white', linewidths=0.5, zorder=3, alpha=0.8)

            ax.set_title(dataset, fontsize=20, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='both', labelsize=12)

            if ds_idx == n_ds // 2:
                ax.set_xlabel('NNR', fontsize=18)
            if ds_idx == 0:
                ax.set_ylabel('OSA', fontsize=18)
            else:
                # Remove y-axis tick labels from middle and right plots
                ax.tick_params(axis='y', labelleft=False)

        # Shared legends - positioned on the right side
        from matplotlib.lines import Line2D

        # Backbone legend (upper right)
        bb_handles = [Line2D([0], [0], color=get_backbone_color(bb), linewidth=3,
                             label=get_backbone_name(bb)) for bb in backbones]
        leg1 = fig.legend(handles=bb_handles, title='Backbone', loc='upper right',
                          bbox_to_anchor=(0.995, 0.92), fontsize=12, title_fontsize=13)

        # Severity legend (below backbone legend, left-aligned with it)
        sev_handles = [Line2D([0], [0], marker=severity_markers[i+1], color='gray', markersize=10,
                              linestyle='', markerfacecolor='gray', label=f'L{i+1}') for i in range(5)]
        fig.legend(handles=sev_handles, title='Severity', loc='upper right',
                   bbox_to_anchor=(0.995, 0.58), fontsize=12, title_fontsize=13,
                   labelspacing=1.0)

        plt.tight_layout(rect=[0, 0, 0.88, 1.0])

        fname = os.path.join(subdir, f"landscape_faceted{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > osa_nnr/ landscape_faceted ({len(ood_datasets)} files)")


def plot_osa_nnr_density(agg_df, out_dir, severity=5):
    """
    Density landscape: shows where configs cluster in OSA-NNR space at severity 5.
    Overlay with config points colored by backbone.
    Generates separate plots for each OOD test dataset, backbone, and nuisance dataset.
    Saves to osa_nnr/ folder.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_data = per_sev[per_sev["level"] == severity]

    if sev_data.empty or "OSA" not in sev_data.columns:
        print(f"  > osa_nnr/ density skipped (no data)")
        return

    subdir = os.path.join(out_dir, "osa_nnr")
    os.makedirs(subdir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in sev_data.columns:
        ood_datasets = sorted(sev_data["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    count = 0
    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in sev_data.columns:
            ood_data = sev_data[sev_data["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = sev_data
            ood_suffix = ""

        if ood_data.empty:
            continue

        # 1. Overall density plot (all backbones, all datasets)
        fig, ax = plt.subplots(figsize=(10, 8))
        x = ood_data["NNR"].values
        y = ood_data["OSA"].values

        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xi, yi = np.mgrid[0:1:100j, 0:1:100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=15, cmap='Blues', alpha=0.6)
            ax.contour(xi, yi, zi, levels=5, colors='steelblue', alpha=0.5, linewidths=0.5)
        except Exception:
            pass

        backbones = sorted(ood_data["backbone"].unique())
        for backbone in backbones:
            bb_data = ood_data[ood_data["backbone"] == backbone]
            ax.scatter(bb_data["NNR"], bb_data["OSA"], c=get_backbone_color(backbone),
                       s=80, edgecolors='white', linewidths=0.8, alpha=0.9,
                       label=get_backbone_name(backbone), zorder=3)

        ax.set_xlabel('NNR', fontsize=12)
        ax.set_ylabel('OSA', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'OSA-NNR Density @ L{severity}', fontsize=13, fontweight='bold')
        ax.legend(title='Backbone', loc='lower left', fontsize=9, title_fontsize=10)

        fname = os.path.join(subdir, f"density_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        count += 1

        # 2. Per-backbone density plots
        for backbone in backbones:
            bb_data = ood_data[ood_data["backbone"] == backbone]
            if len(bb_data) < 3:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            x = bb_data["NNR"].values
            y = bb_data["OSA"].values

            try:
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                ax.contourf(xi, yi, zi, levels=15, cmap='Blues', alpha=0.6)
            except Exception:
                pass

            # Color by dataset
            datasets = sorted(bb_data["dataset"].unique())
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds]
                ax.scatter(ds_data["NNR"], ds_data["OSA"], c=DATASET_COLORS.get(ds, '#666666'),
                           s=60, edgecolors='white', linewidths=0.5, alpha=0.9,
                           label=ds, zorder=3)

            ax.set_xlabel('NNR', fontsize=11)
            ax.set_ylabel('OSA', fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'{get_backbone_name(backbone)} @ L{severity}', fontsize=12, fontweight='bold')
            ax.legend(title='Dataset', loc='lower left', fontsize=8)

            bb_safe = backbone.replace("-", "_")
            fname = os.path.join(subdir, f"density_{bb_safe}_L{severity}{ood_suffix}.pdf")
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            count += 1

        # 3. Per-nuisance-dataset density plots
        datasets = sorted(ood_data["dataset"].unique())
        for ds in datasets:
            ds_data = ood_data[ood_data["dataset"] == ds]
            if len(ds_data) < 3:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            x = ds_data["NNR"].values
            y = ds_data["OSA"].values

            try:
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                ax.contourf(xi, yi, zi, levels=15, cmap='Blues', alpha=0.6)
            except Exception:
                pass

            # Color by backbone
            for backbone in backbones:
                bb_data = ds_data[ds_data["backbone"] == backbone]
                ax.scatter(bb_data["NNR"], bb_data["OSA"], c=get_backbone_color(backbone),
                           s=60, edgecolors='white', linewidths=0.5, alpha=0.9,
                           label=get_backbone_name(backbone), zorder=3)

            ax.set_xlabel('NNR', fontsize=11)
            ax.set_ylabel('OSA', fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'{ds} @ L{severity}', fontsize=12, fontweight='bold')
            ax.legend(title='Backbone', loc='lower left', fontsize=8)

            ds_safe = ds.replace("-", "_")
            fname = os.path.join(subdir, f"density_{ds_safe}_L{severity}{ood_suffix}.pdf")
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            count += 1

    print(f"  > osa_nnr/ density ({count} files)")


# =============================================================================
# OOD DETECTION METRIC HEATMAPS (from summary data)
# =============================================================================

def plot_auoscr_heatmaps(summary_df, out_dir):
    """
    AUOSCR heatmaps: detector × backbone, one per test_dataset.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "AUOSCR" not in ood_df.columns:
        print("  > AUOSCR heatmaps skipped (no data)")
        return

    datasets = sorted(ood_df["dataset"].unique())
    backbones = sorted(ood_df["backbone"].unique())
    detectors = sorted(ood_df["detector"].unique())

    subdir = os.path.join(out_dir, "auoscr_heatmaps")
    os.makedirs(subdir, exist_ok=True)

    for dataset in datasets:
        ds_data = ood_df[ood_df["dataset"] == dataset]

        # Average over OOD datasets if multiple
        if "ood_test_dataset" in ds_data.columns or "test_ood_dataset" in ds_data.columns:
            pivot_data = ds_data.groupby(["detector", "backbone"])["AUOSCR"].mean().reset_index()
        else:
            pivot_data = ds_data

        pivot = pivot_data.pivot(index="detector", columns="backbone", values="AUOSCR")
        pivot = pivot.reindex(index=[d for d in detectors if d in pivot.index],
                              columns=[b for b in backbones if b in pivot.columns])
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1.0, linewidths=0.3, ax=ax)
        ax.set_title(f"AUOSCR - {dataset}", fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        safe = dataset.replace("-", "_")
        fname = os.path.join(subdir, f"{safe}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > auoscr_heatmaps/ ({len(datasets)} files)")


def plot_auroc_heatmaps(summary_df, out_dir):
    """
    AUROC heatmaps: detector × backbone, one per test_dataset.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "AUROC" not in ood_df.columns:
        print("  > AUROC heatmaps skipped (no data)")
        return

    datasets = sorted(ood_df["dataset"].unique())
    backbones = sorted(ood_df["backbone"].unique())
    detectors = sorted(ood_df["detector"].unique())

    subdir = os.path.join(out_dir, "auroc_heatmaps")
    os.makedirs(subdir, exist_ok=True)

    for dataset in datasets:
        ds_data = ood_df[ood_df["dataset"] == dataset]

        if "ood_test_dataset" in ds_data.columns or "test_ood_dataset" in ds_data.columns:
            pivot_data = ds_data.groupby(["detector", "backbone"])["AUROC"].mean().reset_index()
        else:
            pivot_data = ds_data

        pivot = pivot_data.pivot(index="detector", columns="backbone", values="AUROC")
        pivot = pivot.reindex(index=[d for d in detectors if d in pivot.index],
                              columns=[b for b in backbones if b in pivot.columns])
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, linewidths=0.3, ax=ax)
        ax.set_title(f"AUROC (OOD Detection) - {dataset}", fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        safe = dataset.replace("-", "_")
        fname = os.path.join(subdir, f"{safe}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > auroc_heatmaps/ ({len(datasets)} files)")


def plot_fpr95_heatmaps(summary_df, out_dir):
    """
    FPR@95% heatmaps: detector × backbone, one per test_dataset.
    Lower is better, so we use reversed colormap.
    """
    from analysis.processing import get_ood_metrics

    ood_df = get_ood_metrics(summary_df)
    if ood_df.empty or "FPR@95TPR" not in ood_df.columns:
        print("  > FPR@95 heatmaps skipped (no data)")
        return

    datasets = sorted(ood_df["dataset"].unique())
    backbones = sorted(ood_df["backbone"].unique())
    detectors = sorted(ood_df["detector"].unique())

    subdir = os.path.join(out_dir, "fpr95_heatmaps")
    os.makedirs(subdir, exist_ok=True)

    for dataset in datasets:
        ds_data = ood_df[ood_df["dataset"] == dataset]

        if "ood_test_dataset" in ds_data.columns or "test_ood_dataset" in ds_data.columns:
            pivot_data = ds_data.groupby(["detector", "backbone"])["FPR@95TPR"].mean().reset_index()
        else:
            pivot_data = ds_data

        pivot = pivot_data.pivot(index="detector", columns="backbone", values="FPR@95TPR")
        pivot = pivot.reindex(index=[d for d in detectors if d in pivot.index],
                              columns=[b for b in backbones if b in pivot.columns])
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',  # Reversed - lower is better
                    vmin=0, vmax=1.0, linewidths=0.3, ax=ax)
        ax.set_title(f"FPR@95% (OOD Detection) - {dataset}", fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        safe = dataset.replace("-", "_")
        fname = os.path.join(subdir, f"{safe}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > fpr95_heatmaps/ ({len(datasets)} files)")


def plot_ood_metrics_combined(summary_df, out_dir):
    """
    Combined 3-panel heatmap: AUOSCR | AUROC | FPR@95 side-by-side.
    One figure per test_dataset (averaged over OOD datasets).
    """
    from analysis.processing import get_ood_metrics
    from matplotlib.gridspec import GridSpec

    ood_df = get_ood_metrics(summary_df)
    required = ["AUOSCR", "AUROC", "FPR@95TPR"]
    if ood_df.empty or not all(m in ood_df.columns for m in required):
        print("  > OOD metrics combined skipped (missing data)")
        return

    datasets = sorted(ood_df["dataset"].unique())
    backbones = sorted(ood_df["backbone"].unique())
    detectors = sorted(ood_df["detector"].unique())

    for dataset in datasets:
        ds_data = ood_df[ood_df["dataset"] == dataset]

        # Average over OOD datasets if multiple
        if "ood_test_dataset" in ds_data.columns or "test_ood_dataset" in ds_data.columns:
            agg = ds_data.groupby(["detector", "backbone"])[required].mean().reset_index()
        else:
            agg = ds_data

        n_det = len([d for d in detectors if d in agg["detector"].values])
        n_bb = len([b for b in backbones if b in agg["backbone"].values])

        fig = plt.figure(figsize=(12, max(5, n_det * 0.5)))
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.08], wspace=0.15)

        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs[0, 3])

        metrics = [("AUOSCR", "AUOSCR", "RdYlGn", 0, 1),
                   ("AUROC", "AUROC", "RdYlGn", 0.5, 1),
                   ("FPR@95TPR", "FPR@95%", "RdYlGn_r", 0, 1)]

        for i, (col, title, cmap, vmin, vmax) in enumerate(metrics):
            ax = axes[i]
            pivot = agg.pivot(index="detector", columns="backbone", values=col)
            pivot = pivot.reindex(index=[d for d in detectors if d in pivot.index],
                                  columns=[b for b in backbones if b in pivot.columns])
            pivot.columns = [get_backbone_name(b) for b in pivot.columns]

            sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidths=0.3,
                        cbar=(i == 2), cbar_ax=cbar_ax if i == 2 else None,
                        ax=ax, yticklabels=(i == 0))
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=45)

        safe = dataset.replace("-", "_")
        fname = os.path.join(out_dir, f"ood_metrics_{safe}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > ood_metrics_*.pdf ({len(datasets)} files)")


def plot_ood_detection_all(summary_df, out_dir):
    """All OOD detection metric plots (from summary data)."""
    print("\n  OOD Detection Metrics:")
    plot_auoscr_heatmaps(summary_df, out_dir)
    plot_auroc_heatmaps(summary_df, out_dir)
    plot_fpr95_heatmaps(summary_df, out_dir)
    plot_ood_metrics_combined(summary_df, out_dir)


# =============================================================================
# 1. CSA vs CCR: All backbones on one figure, 1 file per detector
# =============================================================================

def plot_csa_ccr_by_detector(agg_df, dataset, out_dir):
    """
    CSA and CCR vs severity: all backbones as subplots, ONE FILE PER DETECTOR.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0 (e.g., clean validation set)

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    subdir = os.path.join(out_dir, f"csa_ccr_{dataset.replace('-', '_')}")
    os.makedirs(subdir, exist_ok=True)

    for detector in detectors:
        det_data = per_sev[per_sev["detector"] == detector]

        n_bb = len(backbones)
        n_cols = min(3, n_bb)
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes[idx]
            bb_data = det_data[det_data["backbone"] == backbone].sort_values("level")

            if not bb_data.empty:
                ax.plot(bb_data["level"], bb_data["CSA"], 'o-', color=COLORS['CSA'],
                        label='CSA', markersize=6, linewidth=2)
                ax.plot(bb_data["level"], bb_data["CCR"], 's--', color=COLORS['CCR'],
                        label='CCR', markersize=6, linewidth=2)
                ax.fill_between(bb_data["level"], bb_data["CSA"], bb_data["CCR"],
                                alpha=0.2, color=COLORS['OSA_Gap'])

                final_gap = bb_data["CSA"].iloc[-1] - bb_data["CCR"].iloc[-1]
                ax.text(0.95, 0.05, f'Gap@L5: {final_gap:.3f}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            ax.set_title(get_backbone_name(backbone), fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 1.0)

        for idx in range(len(backbones), len(axes)):
            axes[idx].set_visible(False)

        # Centered x-axis label
        fig.text(0.5, 0.02, 'Severity Level', ha='center', fontsize=11)
        fig.text(0.02, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=11)
        axes[0].legend(loc='lower left', fontsize=9)

        plt.tight_layout(rect=[0.03, 0.05, 1, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > csa_ccr_{dataset.replace('-', '_')}/ ({len(detectors)} files)")


# =============================================================================
# 1b. OSA vs CSA: All backbones on one figure, 1 file per detector
# =============================================================================

def plot_osa_csa_by_detector(agg_df, dataset, out_dir):
    """
    OSA and CSA vs severity: all backbones as subplots, ONE FILE PER DETECTOR.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0 (e.g., clean validation set)

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print(f"  > osa_csa_{dataset.replace('-', '_')}/ skipped (no OSA data)")
        return

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    subdir = os.path.join(out_dir, f"osa_csa_{dataset.replace('-', '_')}")
    os.makedirs(subdir, exist_ok=True)

    for detector in detectors:
        det_data = per_sev[per_sev["detector"] == detector]

        n_bb = len(backbones)
        n_cols = min(3, n_bb)
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes[idx]
            bb_data = det_data[det_data["backbone"] == backbone].sort_values("level")

            if not bb_data.empty and not bb_data["OSA"].isna().all():
                ax.plot(bb_data["level"], bb_data["CSA"], 'o-', color=COLORS['CSA'],
                        label='CSA', markersize=6, linewidth=2)
                ax.plot(bb_data["level"], bb_data["OSA"], 's--', color=COLORS['OSA'],
                        label='OSA', markersize=6, linewidth=2)
                ax.fill_between(bb_data["level"], bb_data["CSA"], bb_data["OSA"],
                                alpha=0.15, color=COLORS['OSA'])

                final_csa = bb_data["CSA"].iloc[-1]
                final_osa = bb_data["OSA"].iloc[-1]
                ax.text(0.95, 0.05, f'CSA: {final_csa:.3f}\nOSA: {final_osa:.3f}',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            ax.set_title(get_backbone_name(backbone), fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 1.0)

        for idx in range(len(backbones), len(axes)):
            axes[idx].set_visible(False)

        fig.text(0.5, 0.02, 'Severity Level', ha='center', fontsize=11)
        fig.text(0.02, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=11)
        axes[0].legend(loc='lower left', fontsize=9)

        plt.tight_layout(rect=[0.03, 0.05, 1, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > osa_csa_{dataset.replace('-', '_')}/ ({len(detectors)} files)")


# =============================================================================
# 2. FINGERPRINT: All backbones on one figure, 1 file per detector
# =============================================================================

def plot_fingerprint_by_detector(agg_df, dataset, out_dir, top_k=12):
    """
    Top nuisances: all backbones as subplots, ONE FILE PER DETECTOR.
    """
    subset = agg_df[(agg_df["dataset"] == dataset) & (agg_df["level"] > 0)].copy()
    if subset.empty or "nuisance" not in subset.columns:
        return

    backbones = sorted(subset["backbone"].unique()) if "backbone" in subset.columns else ["default"]
    detectors = sorted(subset["detector"].unique()) if "detector" in subset.columns else ["default"]

    subdir = os.path.join(out_dir, f"fingerprint_{dataset.replace('-', '_')}")
    os.makedirs(subdir, exist_ok=True)

    for detector in detectors:
        det_data = subset[subset["detector"] == detector]

        n_bb = len(backbones)
        n_cols = min(3, n_bb)
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 sharex=False, sharey=True, squeeze=False)
        axes = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes[idx]
            bb_data = det_data[det_data["backbone"] == backbone]

            if bb_data.empty:
                ax.set_visible(False)
                continue

            nuis_nnr = bb_data.groupby("nuisance")["NNR"].mean().sort_values(ascending=False).head(top_k)

            colors = [COLORS['NNR']] * len(nuis_nnr)
            bars = ax.barh(range(len(nuis_nnr)), nuis_nnr.values, color=colors, edgecolor='white')
            ax.set_yticks(range(len(nuis_nnr)))
            ax.set_yticklabels(nuis_nnr.index, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.0)
            ax.set_title(get_backbone_name(backbone), fontsize=10, fontweight='bold')

        for idx in range(len(backbones), len(axes)):
            axes[idx].set_visible(False)

        fig.text(0.5, 0.02, 'NNR', ha='center', fontsize=11)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > fingerprint_{dataset.replace('-', '_')}/ ({len(detectors)} files)")


# =============================================================================
# 3. GAP COMPARISON: All datasets on same plot, faceted by backbone
# =============================================================================

def plot_gap_comparison_by_detector(agg_df, out_dir):
    """
    Gap vs severity: all datasets on same plot, subplots per backbone, ONE FILE PER DETECTOR.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]
    datasets = sorted(per_sev["dataset"].unique())

    subdir = os.path.join(out_dir, "gap_comparison")
    os.makedirs(subdir, exist_ok=True)

    for detector in detectors:
        det_data = per_sev[per_sev["detector"] == detector]

        n_bb = len(backbones)
        n_cols = min(3, n_bb)
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes[idx]
            bb_data = det_data[det_data["backbone"] == backbone]

            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].sort_values("level")
                if not ds_data.empty:
                    color = DATASET_COLORS.get(ds, '#666666')
                    ax.plot(ds_data["level"], ds_data["OSA_Gap"], 'o-',
                            color=color, label=ds, markersize=6, linewidth=2)

            ax.set_title(get_backbone_name(backbone), fontsize=10, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, None)

        for idx in range(len(backbones), len(axes)):
            axes[idx].set_visible(False)

        fig.text(0.5, 0.02, 'Severity Level', ha='center', fontsize=11)
        fig.text(0.02, 0.5, 'OSA Gap (CSA - CCR)', va='center', rotation='vertical', fontsize=11)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5),
                   fontsize=9, title='Dataset', title_fontsize=10)

        plt.tight_layout(rect=[0.03, 0.05, 0.88, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > gap_comparison/ ({len(detectors)} files)")


# =============================================================================
# 4. NNR vs SEVERITY: 1 file per dataset, subplots STACKED (2 rows), legend under top-right
# =============================================================================

def plot_nnr_severity_by_dataset(agg_df, dataset, out_dir):
    """
    NNR vs severity: ONE FILE PER DATASET.
    Subplots stacked in 2 rows (3 top, 2 bottom), legend under top-right panel.
    Improved readability with larger fonts and better line visibility.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0 (e.g., clean validation set)

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    n_bb = len(backbones)
    n_cols = 3
    n_rows = (n_bb + n_cols - 1) // n_cols  # Typically 2 rows for 5 backbones

    # Larger figure for better readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, backbone in enumerate(backbones):
        ax = axes_flat[idx]
        bb_data = per_sev[per_sev["backbone"] == backbone]

        for detector in detectors:
            det_data = bb_data[bb_data["detector"] == detector].sort_values("level")
            if not det_data.empty:
                color = get_detector_color(detector)
                ax.plot(det_data["level"], det_data["NNR"], 'o-',
                        color=color, label=detector, markersize=7, linewidth=2.2)

        ax.set_title(get_backbone_name(backbone), fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='both', labelsize=16)

        # Only left column gets y-label
        if idx % n_cols == 0:
            ax.set_ylabel('NNR', fontsize=13)

        # X-axis label and tick labels on: bottom row and top-right
        row = idx // n_cols
        col = idx % n_cols
        if (row == n_rows - 1) or (row == 0 and col == n_cols - 1):
            ax.set_xlabel('Severity Level', fontsize=13)
            # Force x-tick labels to be visible on top-right
            if row == 0 and col == n_cols - 1:
                ax.tick_params(axis='x', labelbottom=True)

    # Hide unused axes but use the space for legend
    for idx in range(len(backbones), len(axes_flat)):
        axes_flat[idx].axis('off')

    # Legend in the empty bottom-right cell (if 5 backbones in 2x3 grid, cell [1,2] is empty)
    handles, labels = axes_flat[0].get_legend_handles_labels()

    if len(backbones) < n_rows * n_cols:
        # Place legend in the empty subplot area
        empty_ax = axes_flat[len(backbones)]
        empty_ax.axis('off')
        empty_ax.legend(handles, labels,
                        loc='center',
                        fontsize=12,
                        title='Detector',
                        title_fontsize=14,
                        framealpha=0.95,
                        edgecolor='gray',
                        ncol=2)
    else:
        # Fallback: place outside on right
        fig.legend(handles, labels,
                   loc='center right',
                   bbox_to_anchor=(0.99, 0.5),
                   fontsize=12,
                   title='Detector',
                   title_fontsize=14)

    plt.tight_layout()

    subdir = os.path.join(out_dir, "nnr_severity")
    os.makedirs(subdir, exist_ok=True)
    fname = os.path.join(subdir, f"{dataset.replace('-', '_')}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > nnr_severity/{dataset.replace('-', '_')}.pdf")


# =============================================================================
# 4b. OSA vs SEVERITY - Stacked layout matching NNR
# =============================================================================

def plot_osa_severity_by_dataset(agg_df, dataset, out_dir):
    """
    OSA vs severity: ONE FILE PER DATASET PER OOD TEST DATASET.
    Stacked layout (2 rows x 3 cols), legend in empty cell.
    Generates separate files for each OOD test dataset (e.g., NINCO, OpenImage-O).
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0 (e.g., clean validation set)

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print(f"  > osa_severity/{dataset.replace('-', '_')}.pdf skipped (no OSA data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in per_sev.columns:
        ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in per_sev.columns:
            ood_data = per_sev[per_sev["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
            subdir_name = f"osa_severity_{ood_ds.replace('-', '_')}"
        else:
            ood_data = per_sev
            ood_suffix = ""
            subdir_name = "osa_severity"

        backbones = sorted(ood_data["backbone"].unique()) if "backbone" in ood_data.columns else ["default"]
        detectors = sorted(ood_data["detector"].unique()) if "detector" in ood_data.columns else ["default"]

        n_bb = len(backbones)
        if n_bb == 0:
            continue

        n_cols = 3
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes_flat = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]

            for detector in detectors:
                det_data = bb_data[bb_data["detector"] == detector].sort_values("level")
                if not det_data.empty:
                    color = get_detector_color(detector)
                    ax.plot(det_data["level"], det_data["OSA"], 'o-',
                            color=color, label=detector, markersize=5, linewidth=1.8)

            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 1.0)

            if idx % n_cols == 0:
                ax.set_ylabel('OSA', fontsize=11)

        for idx in range(len(backbones), len(axes_flat)):
            axes_flat[idx].axis('off')

        fig.text(0.5, 0.02, 'Severity Level', ha='center', fontsize=12)

        handles, labels = axes_flat[0].get_legend_handles_labels()

        if len(backbones) < n_rows * n_cols:
            empty_ax = axes_flat[len(backbones)]
            empty_ax.axis('off')
            empty_ax.legend(handles, labels,
                            loc='center',
                            fontsize=9,
                            title='Detector',
                            title_fontsize=10,
                            framealpha=0.95,
                            edgecolor='gray',
                            ncol=2)
        else:
            fig.legend(handles, labels,
                       loc='center right',
                       bbox_to_anchor=(0.99, 0.5),
                       fontsize=9,
                       title='Detector',
                       title_fontsize=10)

        plt.tight_layout(rect=[0, 0.04, 1.0, 1.0])

        subdir = os.path.join(out_dir, subdir_name)
        os.makedirs(subdir, exist_ok=True)
        fname = os.path.join(subdir, f"{dataset.replace('-', '_')}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  > {subdir_name}/{dataset.replace('-', '_')}.pdf")


# =============================================================================
# 5. NNR HEATMAPS: 3 datasets side-by-side, EQUAL SIZE BOXES, range 0-1
# =============================================================================

def plot_nnr_heatmaps_all_datasets(agg_df, out_dir, severity=5):
    """
    NNR@L5 heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    Improved readability with larger fonts.
    Generates separate heatmaps for each OOD test dataset if present.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in sev_df.columns:
        ood_datasets = sorted(sev_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in sev_df.columns:
            ood_data = sev_df[sev_df["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = sev_df
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

        # Limit to 3 datasets for the combined plot
        ds_order = ds_order[:3]

        if len(ds_order) == 0:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        n_ds = len(ds_order)

        # Use GridSpec for uniform heatmap sizing with separate colorbar
        from matplotlib.gridspec import GridSpec

        n_det = len(detectors)
        n_bb = len(backbones)
        cell_size = 0.5  # Compact cells

        # Figure size: 3 equal heatmaps + colorbar column
        heatmap_width = n_bb * cell_size
        fig_width = n_ds * heatmap_width + 3.0  # Extra for y-labels and colorbar
        fig_height = n_det * cell_size + 1.8

        fig = plt.figure(figsize=(fig_width, fig_height))

        # GridSpec: n_ds equal columns for heatmaps, 1 narrow column for colorbar
        width_ratios = [1] * n_ds + [0.08]
        gs = GridSpec(1, n_ds + 1, figure=fig, width_ratios=width_ratios, wspace=0.15)

        axes = [fig.add_subplot(gs[0, i]) for i in range(n_ds)]
        cbar_ax = fig.add_subplot(gs[0, n_ds])

        for ds_idx, dataset in enumerate(ds_order):
            ax = axes[ds_idx]
            ds_data = ood_data[ood_data["dataset"] == dataset]

            pivot = ds_data.pivot(index="detector", columns="backbone", values="NNR")
            pivot = pivot.reindex(index=detectors, columns=backbones)

            # Rename columns to display names
            pivot.columns = [get_backbone_name(b) for b in pivot.columns]

            # Only show y-tick labels (detector names) on leftmost plot
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                        vmin=0, vmax=1.0, linewidths=0.3,
                        cbar=(ds_idx == n_ds - 1),
                        cbar_ax=cbar_ax if ds_idx == n_ds - 1 else None,
                        ax=ax, annot_kws={'fontsize': 9},
                        square=True,
                        yticklabels=(ds_idx == 0))  # Only leftmost gets y-tick labels

            ax.set_title(dataset, fontsize=14, fontweight='bold', pad=4)
            ax.set_xlabel('')  # No axis label
            ax.set_ylabel('')  # No axis label
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            if ds_idx == 0:
                ax.tick_params(axis='y', rotation=0, labelsize=12)

        cbar_ax.set_ylabel(f'NNR@L{severity}', fontsize=11)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"NNR_heatmaps_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  > NNR_heatmaps_L{severity}{ood_suffix}.pdf")


# =============================================================================
# 5b. OSA HEATMAPS
# =============================================================================

def plot_osa_heatmaps_all_datasets(agg_df, out_dir, severity=5):
    """
    OSA@L5 heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    Uses GridSpec for uniform sizing.
    Generates separate heatmaps for each OOD test dataset if present.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty or "OSA" not in sev_df.columns or sev_df["OSA"].isna().all():
        print(f"  > OSA_heatmaps_L{severity}.pdf skipped (no OSA data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in sev_df.columns:
        ood_datasets = sorted(sev_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in sev_df.columns:
            ood_data = sev_df[sev_df["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = sev_df
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
        ds_order = ds_order[:3]

        if len(ds_order) == 0:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        n_ds = len(ds_order)

        # Use GridSpec for uniform heatmap sizing with separate colorbar
        from matplotlib.gridspec import GridSpec

        n_det = len(detectors)
        n_bb = len(backbones)
        cell_size = 0.5

        heatmap_width = n_bb * cell_size
        fig_width = n_ds * heatmap_width + 3.0
        fig_height = n_det * cell_size + 1.8

        fig = plt.figure(figsize=(fig_width, fig_height))
        width_ratios = [1] * n_ds + [0.08]
        gs = GridSpec(1, n_ds + 1, figure=fig, width_ratios=width_ratios, wspace=0.15)

        axes = [fig.add_subplot(gs[0, i]) for i in range(n_ds)]
        cbar_ax = fig.add_subplot(gs[0, n_ds])

        for ds_idx, dataset in enumerate(ds_order):
            ax = axes[ds_idx]
            ds_data = ood_data[ood_data["dataset"] == dataset]

            pivot = ds_data.pivot(index="detector", columns="backbone", values="OSA")
            pivot = pivot.reindex(index=detectors, columns=backbones)
            pivot.columns = [get_backbone_name(b) for b in pivot.columns]

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                        vmin=0.3, vmax=1.0, linewidths=0.3,
                        cbar=(ds_idx == n_ds - 1),
                        cbar_ax=cbar_ax if ds_idx == n_ds - 1 else None,
                        ax=ax, annot_kws={'fontsize': 9},
                        square=True,
                        yticklabels=(ds_idx == 0))

            ax.set_title(dataset, fontsize=14, fontweight='bold', pad=4)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            if ds_idx == 0:
                ax.tick_params(axis='y', rotation=0, labelsize=12)

        ood_label = f" ({ood_ds})" if ood_ds else ""
        cbar_ax.set_ylabel(f'OSA@L{severity}{ood_label}', fontsize=11)

        fname = os.path.join(out_dir, f"OSA_heatmaps_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  > OSA_heatmaps_L{severity}{ood_suffix}.pdf")


# =============================================================================
# 6. ADR HEATMAPS
# =============================================================================

def plot_adr_violin(agg_df, out_dir):
    """
    ADR violin plot (CSA-CCR based): Shows distribution of ADR values per dataset.
    Clearly demonstrates systematic difference between benchmarks.
    Generates separate plots for each OOD test dataset.
    Saves to adr_ccr_violin/ folder.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > adr_ccr_violin/ skipped (no ADR data)")
        return

    subdir = os.path.join(out_dir, "adr_ccr_violin")
    os.makedirs(subdir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in adr_df.columns:
        ood_datasets = sorted(adr_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in adr_df.columns:
            ood_data = adr_df[adr_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = adr_df.copy()
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

        if len(ds_order) == 0:
            continue

        # Reorder dataframe
        ood_data["dataset"] = pd.Categorical(ood_data["dataset"], categories=ds_order, ordered=True)
        ood_data = ood_data.sort_values("dataset")

        fig, ax = plt.subplots(figsize=(8, 5))

        # Color palette for datasets
        palette = [DATASET_COLORS.get(ds, '#666666') for ds in ds_order]

        # Violin plot
        parts = ax.violinplot(
            [ood_data[ood_data["dataset"] == ds]["ADR"].values for ds in ds_order],
            positions=range(len(ds_order)),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(palette[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Style the lines
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)

        # Add individual points (jittered)
        for i, ds in enumerate(ds_order):
            ds_data = ood_data[ood_data["dataset"] == ds]["ADR"].values
            jitter = np.random.uniform(-0.15, 0.15, size=len(ds_data))
            ax.scatter(i + jitter, ds_data, c='black', s=25, alpha=0.6, zorder=3)

        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # Add mean annotations
        for i, ds in enumerate(ds_order):
            mean_val = ood_data[ood_data["dataset"] == ds]["ADR"].mean()
            ax.annotate(f'μ={mean_val:.3f}',
                        xy=(i, mean_val), xytext=(i + 0.35, mean_val),
                        fontsize=10, fontweight='bold',
                        color=palette[i],
                        va='center')

        ax.set_xticks(range(len(ds_order)))
        ax.set_xticklabels(ds_order, fontsize=11)
        ax.set_ylabel('ADR', fontsize=11)
        ax.set_xlabel('Benchmark', fontsize=11)

        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlim(-0.5, len(ds_order) - 0.5)

        plt.tight_layout()
        fname = os.path.join(subdir, f"ADR_ccr{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > adr_ccr_violin/ ({len(ood_datasets)} files)")


def plot_adr_osa_violin(agg_df, out_dir):
    """
    ADR_OSA violin plot (CSA-OSA based): Shows distribution of OSA divergence rates per dataset.
    Measures how quickly OSA diverges from CSA as severity increases.
    Generates separate plots for each OOD test dataset.
    Saves to adr_osa_violin/ folder.
    """
    from analysis.processing import compute_adr_osa

    adr_df = compute_adr_osa(agg_df)
    if adr_df.empty:
        print("  > adr_osa_violin/ skipped (no ADR_OSA data)")
        return

    subdir = os.path.join(out_dir, "adr_osa_violin")
    os.makedirs(subdir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in adr_df.columns:
        ood_datasets = sorted(adr_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in adr_df.columns:
            ood_data = adr_df[adr_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = adr_df.copy()
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

        if len(ds_order) == 0:
            continue

        # Reorder dataframe
        ood_data["dataset"] = pd.Categorical(ood_data["dataset"], categories=ds_order, ordered=True)
        ood_data = ood_data.sort_values("dataset")

        fig, ax = plt.subplots(figsize=(8, 5))

        # Color palette for datasets
        palette = [DATASET_COLORS.get(ds, '#666666') for ds in ds_order]

        # Violin plot
        parts = ax.violinplot(
            [ood_data[ood_data["dataset"] == ds]["ADR_OSA"].values for ds in ds_order],
            positions=range(len(ds_order)),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(palette[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Style the lines
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)

        # Add individual points (jittered)
        for i, ds in enumerate(ds_order):
            ds_data = ood_data[ood_data["dataset"] == ds]["ADR_OSA"].values
            jitter = np.random.uniform(-0.15, 0.15, size=len(ds_data))
            ax.scatter(i + jitter, ds_data, c='black', s=25, alpha=0.6, zorder=3)

        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # Add mean annotations
        for i, ds in enumerate(ds_order):
            mean_val = ood_data[ood_data["dataset"] == ds]["ADR_OSA"].mean()
            ax.annotate(f'μ={mean_val:.3f}',
                        xy=(i, mean_val), xytext=(i + 0.35, mean_val),
                        fontsize=10, fontweight='bold',
                        color=palette[i],
                        va='center')

        ax.set_xticks(range(len(ds_order)))
        ax.set_xticklabels(ds_order, fontsize=11)
        ax.set_ylabel('ADR', fontsize=11)
        ax.set_xlabel('Benchmark', fontsize=11)

        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xlim(-0.5, len(ds_order) - 0.5)

        plt.tight_layout()
        fname = os.path.join(subdir, f"ADR_osa{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > adr_osa_violin/ ({len(ood_datasets)} files)")


def plot_adr_heatmaps_all_datasets(agg_df, out_dir):
    """
    ADR heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    Uses GridSpec for uniform sizing.
    Generates separate heatmaps for each OOD test dataset if present.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR_heatmaps.pdf skipped (no ADR data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in adr_df.columns:
        ood_datasets = sorted(adr_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in adr_df.columns:
            ood_data = adr_df[adr_df["test_ood_dataset"] == ood_ds]
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = adr_df
            ood_suffix = ""

        datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
        ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
        ds_order = ds_order[:3]

        if len(ds_order) == 0:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        n_ds = len(ds_order)

        # Use GridSpec for uniform heatmap sizing with separate colorbar
        from matplotlib.gridspec import GridSpec

        n_det = len(detectors)
        n_bb = len(backbones)
        cell_size = 0.5

        heatmap_width = n_bb * cell_size
        fig_width = n_ds * heatmap_width + 3.0
        fig_height = n_det * cell_size + 1.8

        fig = plt.figure(figsize=(fig_width, fig_height))
        width_ratios = [1] * n_ds + [0.08]  # Dynamic based on number of datasets
        gs = GridSpec(1, n_ds + 1, figure=fig, width_ratios=width_ratios, wspace=0.15)

        axes = [fig.add_subplot(gs[0, i]) for i in range(n_ds)]
        cbar_ax = fig.add_subplot(gs[0, n_ds])

        vmax = ood_data["ADR"].abs().max() * 1.1

        for ds_idx, dataset in enumerate(ds_order):
            ax = axes[ds_idx]
            ds_data = ood_data[ood_data["dataset"] == dataset]

            pivot = ds_data.pivot(index="detector", columns="backbone", values="ADR")
            pivot = pivot.reindex(index=detectors, columns=backbones)
            pivot.columns = [get_backbone_name(b) for b in pivot.columns]

            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                        vmin=-vmax, vmax=vmax, linewidths=0.3, center=0,
                        cbar=(ds_idx == n_ds - 1),
                        cbar_ax=cbar_ax if ds_idx == n_ds - 1 else None,
                        ax=ax, annot_kws={'fontsize': 8},
                        square=True,
                        yticklabels=(ds_idx == 0))

            ax.set_title(dataset, fontsize=14, fontweight='bold', pad=4)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            if ds_idx == 0:
                ax.tick_params(axis='y', rotation=0, labelsize=12)

        cbar_ax.set_ylabel('ADR (slope)', fontsize=11)

        fname = os.path.join(out_dir, f"ADR_heatmaps{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  > ADR_heatmaps{ood_suffix}.pdf")


# =============================================================================
# 7. TRANSFERABILITY: 1 plot per backbone
# =============================================================================

def plot_transferability_by_backbone(agg_df, out_dir, reference="ImageNet-LN", severity=5):
    """
    Transferability scatter: ONE PLOT PER BACKBONE.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty:
        print(f"  > Transferability skipped (no L{severity} data)")
        return

    backbones = sorted(sev_df["backbone"].unique()) if "backbone" in sev_df.columns else ["default"]
    datasets = sorted(sev_df["dataset"].unique())
    other_datasets = [d for d in datasets if d != reference]

    if reference not in datasets or len(other_datasets) == 0:
        print(f"  > Transferability skipped (reference '{reference}' not in {datasets})")
        return

    subdir = os.path.join(out_dir, "transferability")
    os.makedirs(subdir, exist_ok=True)

    detectors = sorted(sev_df["detector"].unique())
    det_color_map = {d: get_detector_color(d) for d in detectors}

    for backbone in backbones:
        bb_data = sev_df[sev_df["backbone"] == backbone]

        if bb_data.empty:
            continue

        n_plots = len(other_datasets)

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
        axes = axes.flatten()

        ref_data = bb_data[bb_data["dataset"] == reference].set_index("detector")["NNR"]

        for ax_idx, other_ds in enumerate(other_datasets):
            ax = axes[ax_idx]
            other_data = bb_data[bb_data["dataset"] == other_ds].set_index("detector")["NNR"]

            common_detectors = ref_data.index.intersection(other_data.index)

            if len(common_detectors) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center')
                ax.set_title(f'{reference} vs {other_ds}')
                continue

            x = ref_data.loc[common_detectors].values
            y = other_data.loc[common_detectors].values

            rho, pval = spearmanr(x, y)

            for det in common_detectors:
                ax.scatter(ref_data.loc[det], other_data.loc[det],
                           s=120, c=[det_color_map[det]], edgecolors='black',
                           linewidths=0.5, alpha=0.8, label=det)

            lims = [0, max(max(x), max(y)) * 1.15]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

            ax.set_xlabel(f'NNR@L{severity} on {reference}', fontsize=10)
            ax.set_ylabel(f'NNR@L{severity} on {other_ds}', fontsize=10)
            ax.set_xlim(0, lims[1])
            ax.set_ylim(0, lims[1])
            ax.set_title(f'{reference} vs {other_ds}', fontweight='bold')

            ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {pval:.4f}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
                   fontsize=9, title='Detector', title_fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 1.0])

        safe_bb = backbone.replace("-", "_").replace(" ", "_")
        fname = os.path.join(subdir, f"{safe_bb}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > transferability/ ({len(backbones)} files)")


# =============================================================================
# 7b. TRANSFERABILITY COMBINED: All backbones on one figure
# =============================================================================

def plot_transferability_combined(agg_df, out_dir, reference="ImageNet-LN", severity=5):
    """
    Combined 5-panel transferability plot: ALL BACKBONES ON ONE FIGURE.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty:
        print(f"  > Transferability combined skipped (no L{severity} data)")
        return

    backbones = sorted(sev_df["backbone"].unique()) if "backbone" in sev_df.columns else ["default"]
    datasets = sorted(sev_df["dataset"].unique())
    other_datasets = [d for d in datasets if d != reference]

    if reference not in datasets or len(other_datasets) == 0:
        print(f"  > Transferability combined skipped (reference '{reference}' not in {datasets})")
        return

    other_ds = other_datasets[0]

    detectors = sorted(sev_df["detector"].unique())
    det_color_map = {d: get_detector_color(d) for d in detectors}

    n_bb = len(backbones)
    n_cols = min(5, n_bb)
    n_rows = (n_bb + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, backbone in enumerate(backbones):
        ax = axes[idx]
        bb_data = sev_df[sev_df["backbone"] == backbone]

        if bb_data.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(get_backbone_name(backbone), fontweight='bold')
            continue

        ref_data = bb_data[bb_data["dataset"] == reference].set_index("detector")["NNR"]
        other_data = bb_data[bb_data["dataset"] == other_ds].set_index("detector")["NNR"]

        common_detectors = ref_data.index.intersection(other_data.index)

        if len(common_detectors) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(get_backbone_name(backbone), fontweight='bold')
            continue

        x = ref_data.loc[common_detectors].values
        y = other_data.loc[common_detectors].values

        rho, pval = spearmanr(x, y)

        for det in common_detectors:
            ax.scatter(ref_data.loc[det], other_data.loc[det],
                       s=120, c=[det_color_map[det]], edgecolors='black',
                       linewidths=0.5, alpha=0.8, label=det if idx == 0 else '')

        lims = [0, max(max(x), max(y)) * 1.15]
        ax.plot(lims, lims, '--', color='gray', alpha=0.5, linewidth=1)

        ax.set_xlim(0, lims[1])
        ax.set_ylim(0, lims[1])
        ax.set_title(get_backbone_name(backbone), fontweight='bold')

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(f'NNR@L{severity} on {reference}', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel(f'NNR@L{severity} on {other_ds}', fontsize=10)

        ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {pval:.4f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    for idx in range(len(backbones), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5),
               fontsize=9, title='Detector', title_fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.88, 1.0])

    fname = os.path.join(out_dir, f"transferability_combined_L{severity}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > transferability_combined_L{severity}.pdf")


# =============================================================================
# 8. CONFIG × BENCHMARK HEATMAP
# =============================================================================

def plot_config_benchmark_heatmap(agg_df, out_dir, metric="NNR"):
    """
    Heatmap: (backbone+detector) rows × dataset columns.
    Generates separate heatmaps for each OOD test dataset if present.
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if metric not in mean_df.columns or mean_df[metric].isna().all():
        print(f"  > config_benchmark_{metric}_heatmap skipped (no data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in mean_df.columns:
        ood_datasets = sorted(mean_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in mean_df.columns:
            ood_data = mean_df[mean_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = mean_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        ood_data["Config"] = ood_data["backbone"].apply(get_backbone_name) + " + " + ood_data["detector"]

        pivot = ood_data.pivot(index="Config", columns="dataset", values=metric)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.35)))

        if metric == "NNR":
            cmap = 'YlOrRd'
            vmin, vmax = 0, 1.0
        elif metric == "CSA":
            cmap = 'RdYlGn'
            vmin, vmax = 0, 1.0
        elif metric == "OSA":
            cmap = 'RdYlGn'
            vmin, vmax = 0.3, 1.0
        else:
            cmap = 'YlOrRd'
            vmin, vmax = None, None

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.5, cbar_kws={'label': f'Mean {metric}'}, ax=ax,
                    square=False)

        ax.set_xlabel('Dataset/Benchmark', fontsize=10)
        ax.set_ylabel('Configuration', fontsize=10)
        ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        fname = os.path.join(out_dir, f"config_benchmark_{metric}_heatmap{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > config_benchmark_{metric}_heatmap ({len(ood_datasets)} files)")


# =============================================================================
# 9. CSA vs CCR BENCHMARK COMPARISON (FIXED LEGEND)
# =============================================================================

def plot_csa_ccr_benchmark_comparison(agg_df, out_dir):
    """
    CSA vs CCR comparison: 3 subplots (one per benchmark).
    Color = backbone, Line style = metric (solid=CSA, dashed=CCR).
    FIXED: Legend not cut off, full backbone names.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0

    datasets = sorted(per_sev["dataset"].unique())
    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

    subdir = os.path.join(out_dir, "csa_ccr_benchmark")
    os.makedirs(subdir, exist_ok=True)

    for detector in detectors:
        det_data = per_sev[per_sev["detector"] == detector]

        n_ds = len(ds_order)
        fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4), sharey=True, squeeze=False)
        axes = axes.flatten()

        for ds_idx, dataset in enumerate(ds_order):
            ax = axes[ds_idx]
            ds_data = det_data[det_data["dataset"] == dataset]

            for backbone in backbones:
                bb_data = ds_data[ds_data["backbone"] == backbone].sort_values("level")
                if bb_data.empty:
                    continue

                color = get_backbone_color(backbone)
                ax.plot(bb_data["level"], bb_data["CSA"], '-o', color=color,
                        markersize=6, linewidth=2)
                ax.plot(bb_data["level"], bb_data["CCR"], '--s', color=color,
                        markersize=6, linewidth=2, alpha=0.85)
                ax.fill_between(bb_data["level"], bb_data["CSA"], bb_data["CCR"],
                                alpha=0.08, color=color)

            ax.set_title(dataset, fontsize=11, fontweight='bold')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 1.0)
            if ds_idx == n_ds // 2:
                ax.set_xlabel('Severity Level', fontsize=11)
            if ds_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=11)

        # Custom legend with FULL BACKBONE NAMES - compact layout
        from matplotlib.lines import Line2D

        bb_handles = [Line2D([0], [0], color=get_backbone_color(bb), linewidth=2.5,
                             label=get_backbone_name(bb))
                      for bb in backbones]

        metric_handles = [
            Line2D([0], [0], color='#333333', linestyle='-', linewidth=2, marker='o', markersize=5, label='CSA'),
            Line2D([0], [0], color='#333333', linestyle='--', linewidth=2, marker='s', markersize=5, label='CCR'),
        ]

        # Compact legends aligned left, close to rightmost subplot
        leg1 = fig.legend(handles=bb_handles, title='Classifier',
                          loc='upper left', bbox_to_anchor=(0.78, 0.98),
                          fontsize=9, title_fontsize=10, framealpha=0.95,
                          edgecolor='gray', borderaxespad=0,
                          labelspacing=0.3, handlelength=1.5, handletextpad=0.4,
                          alignment='left')

        leg2 = fig.legend(handles=metric_handles, title='Metric',
                          loc='upper left', bbox_to_anchor=(0.78, 0.38),
                          fontsize=9, title_fontsize=10, framealpha=0.95,
                          edgecolor='gray', borderaxespad=0,
                          labelspacing=0.3, handlelength=1.5, handletextpad=0.4,
                          alignment='left')

        plt.tight_layout(rect=[0, 0, 0.78, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > csa_ccr_benchmark/ ({len(detectors)} files)")


def plot_osa_csa_benchmark_comparison(agg_df, out_dir):
    """
    OSA vs CSA comparison: 3 subplots (one per benchmark).
    Color = backbone, Line style = metric (solid=OSA, dashed=CSA).
    Same style as CSA-CCR benchmark comparison.
    Generates separate plots for each OOD test dataset (NINCO, OpenImage-O, etc.).
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]
    if per_sev.empty:
        return  # No severity levels > 0

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print("  > osa_csa_benchmark/ skipped (no OSA data)")
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in per_sev.columns:
        ood_datasets = sorted(per_sev["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]  # No OOD dataset column, treat as single

    datasets = sorted(per_sev["dataset"].unique())
    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

    total_files = 0
    for ood_ds in ood_datasets:
        # Create subdir for this OOD dataset
        if ood_ds:
            ood_safe = ood_ds.replace("-", "_")
            subdir = os.path.join(out_dir, f"osa_csa_benchmark_{ood_safe}")
        else:
            subdir = os.path.join(out_dir, "osa_csa_benchmark")
        os.makedirs(subdir, exist_ok=True)

        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in per_sev.columns:
            ood_data = per_sev[per_sev["test_ood_dataset"] == ood_ds]
        else:
            ood_data = per_sev

        for detector in detectors:
            det_data = ood_data[ood_data["detector"] == detector]

            n_ds = len(ds_order)
            fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4), sharey=True, squeeze=False)
            axes = axes.flatten()

            for ds_idx, dataset in enumerate(ds_order):
                ax = axes[ds_idx]
                ds_data = det_data[det_data["dataset"] == dataset]

                for backbone in backbones:
                    bb_data = ds_data[ds_data["backbone"] == backbone].sort_values("level")
                    if bb_data.empty:
                        continue

                    color = get_backbone_color(backbone)
                    ax.plot(bb_data["level"], bb_data["OSA"], '-o', color=color,
                            markersize=6, linewidth=2)
                    ax.plot(bb_data["level"], bb_data["CSA"], '--s', color=color,
                            markersize=6, linewidth=2, alpha=0.85)
                    ax.fill_between(bb_data["level"], bb_data["OSA"], bb_data["CSA"],
                                    alpha=0.08, color=color)

                ax.set_title(dataset, fontsize=11, fontweight='bold')
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_ylim(0, 1.0)
                if ds_idx == n_ds // 2:
                    ax.set_xlabel('Severity Level', fontsize=11)
                if ds_idx == 0:
                    ax.set_ylabel('Accuracy', fontsize=11)

            # Custom legend with FULL BACKBONE NAMES - compact layout
            from matplotlib.lines import Line2D

            bb_handles = [Line2D([0], [0], color=get_backbone_color(bb), linewidth=2.5,
                                 label=get_backbone_name(bb))
                          for bb in backbones]

            metric_handles = [
                Line2D([0], [0], color='#333333', linestyle='-', linewidth=2, marker='o', markersize=5, label='OSA'),
                Line2D([0], [0], color='#333333', linestyle='--', linewidth=2, marker='s', markersize=5, label='CSA'),
            ]

            # Compact legends aligned left, close to rightmost subplot
            fig.legend(handles=bb_handles, title='Classifier',
                       loc='upper left', bbox_to_anchor=(0.78, 0.98),
                       fontsize=9, title_fontsize=10, framealpha=0.95,
                       edgecolor='gray', borderaxespad=0,
                       labelspacing=0.3, handlelength=1.5, handletextpad=0.4,
                       alignment='left')

            fig.legend(handles=metric_handles, title='Metric',
                       loc='upper left', bbox_to_anchor=(0.78, 0.38),
                       fontsize=9, title_fontsize=10, framealpha=0.95,
                       edgecolor='gray', borderaxespad=0,
                       labelspacing=0.3, handlelength=1.5, handletextpad=0.4,
                       alignment='left')

            plt.tight_layout(rect=[0, 0, 0.78, 1.0])

            fname = os.path.join(subdir, f"{detector}.pdf")
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            total_files += 1

        ood_label = ood_ds if ood_ds else "default"
        print(f"  > osa_csa_benchmark_{ood_label}/ ({len(detectors)} files)")


# =============================================================================
# 10. FINE-GRAINED FACETED: CUB vs Cars
# =============================================================================

def plot_finegrained_faceted(agg_df, out_dir, severity=5):
    """
    CUB vs Cars correlation: TWO FIGURES - NNR@L5 and mean NNR.
    Layout: 2 rows x 3 cols, legend in empty cell under top-right subplot.
    """
    per_sev = get_metrics_by_severity(agg_df)
    mean_df = get_mean_metrics(agg_df)

    if per_sev.empty:
        return

    all_datasets = per_sev["dataset"].unique()
    cub_name = None
    cars_name = None

    for ds in all_datasets:
        if "cub" in ds.lower():
            cub_name = ds
        if "car" in ds.lower():
            cars_name = ds

    if cub_name is None or cars_name is None:
        print("  > Fine-grained faceted plot skipped (need CUB and Cars)")
        return

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique())

    det_color_map = {d: get_detector_color(d) for d in detectors}

    variants = []

    sev_data = per_sev[per_sev["level"] == severity]
    if not sev_data.empty:
        variants.append(("NNR_L5", f"NNR@L{severity}", sev_data))

    if not mean_df.empty:
        variants.append(("mean_NNR", "Mean NNR", mean_df))

    for tag, label, data_df in variants:
        n_bb = len(backbones)
        n_cols = 3
        n_rows = (n_bb + n_cols - 1) // n_cols

        # First pass: compute global max across all backbones for shared scale
        all_values = []
        for backbone in backbones:
            bb_data = data_df[data_df["backbone"] == backbone]
            cub_data = bb_data[bb_data["dataset"] == cub_name].set_index("detector")["NNR"]
            cars_data = bb_data[bb_data["dataset"] == cars_name].set_index("detector")["NNR"]
            common = cub_data.index.intersection(cars_data.index)
            if len(common) >= 3:
                all_values.extend(cub_data.loc[common].values.tolist())
                all_values.extend(cars_data.loc[common].values.tolist())
        global_max = max(all_values) * 1.15 if all_values else 1.0  # Add 15% padding

        # Larger figure with more space for readability
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes_flat[idx]
            bb_data = data_df[data_df["backbone"] == backbone]

            cub_data = bb_data[bb_data["dataset"] == cub_name].set_index("detector")["NNR"]
            cars_data = bb_data[bb_data["dataset"] == cars_name].set_index("detector")["NNR"]

            common = cub_data.index.intersection(cars_data.index)

            if len(common) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=16)
                ax.set_title(get_backbone_name(backbone), fontweight='bold', fontsize=20)
                continue

            x = cub_data.loc[common].values
            y = cars_data.loc[common].values

            rho, pval = spearmanr(x, y)

            for det in common:
                ax.scatter(cub_data.loc[det], cars_data.loc[det], s=150,
                           c=[det_color_map[det]], edgecolors='black', linewidths=0.8,
                           alpha=0.85, label=det if idx == 0 else "")

            ax.plot([0, global_max], [0, global_max], '--', color='gray', alpha=0.5, linewidth=1.5)

            ax.set_xlim(0, global_max)
            ax.set_ylim(0, global_max)
            ax.set_title(get_backbone_name(backbone), fontweight='bold', fontsize=20)
            ax.tick_params(axis='both', labelsize=14)

            # Calculate row/col position
            row = idx // n_cols
            col = idx % n_cols

            # Only bottom row gets x-axis label and tick numbers
            if row == n_rows - 1 or idx >= n_bb - n_cols:
                ax.set_xlabel('CUB-LN', fontsize=18)
            else:
                # Hide x-axis tick labels for top row
                ax.tick_params(axis='x', labelbottom=False)

            # Only left column gets y-axis label and tick numbers
            if col == 0:
                ax.set_ylabel('Cars-LN', fontsize=18)
            else:
                # Hide y-axis tick labels for non-left columns
                ax.tick_params(axis='y', labelleft=False)

            # Larger rho/p annotation box
            ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {pval:.4f}',
                    transform=ax.transAxes, fontsize=14, va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5))

        # Hide unused axes but use the space for legend
        for idx in range(len(backbones), len(axes_flat)):
            axes_flat[idx].axis('off')

        # Get legend handles from first subplot
        handles, labels_leg = axes_flat[0].get_legend_handles_labels()

        # Place legend in the empty subplot area (under top-right plot)
        if len(backbones) < n_rows * n_cols:
            empty_ax = axes_flat[len(backbones)]
            empty_ax.axis('off')
            empty_ax.legend(handles, labels_leg,
                            loc='center',
                            fontsize=16,
                            title='Detector',
                            title_fontsize=18,
                            framealpha=0.95,
                            edgecolor='gray',
                            markerscale=1.8,
                            ncol=2)
        else:
            # Fallback: place outside on right
            fig.legend(handles, labels_leg,
                       loc='center right',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=16,
                       title='Detector',
                       title_fontsize=18,
                       markerscale=1.8)

        plt.tight_layout()

        fname = os.path.join(out_dir, f"finegrained_{tag}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > finegrained_NNR_L5.pdf, finegrained_mean_NNR.pdf")


# =============================================================================
# 11. DETECTOR BUMP CHART
# =============================================================================

def plot_detector_bump_chart(agg_df, out_dir, severity=5):
    """
    Bump chart showing detector rank changes across datasets.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty:
        return

    backbones = sorted(sev_df["backbone"].unique())
    datasets = sorted(sev_df["dataset"].unique())
    detectors = sorted(sev_df["detector"].unique())

    if len(datasets) < 2:
        print("  > Bump chart skipped (need >= 2 datasets)")
        return

    subdir = os.path.join(out_dir, "bump_chart")
    os.makedirs(subdir, exist_ok=True)

    det_color_map = {d: get_detector_color(d) for d in detectors}

    for backbone in backbones:
        bb_data = sev_df[sev_df["backbone"] == backbone]

        ranks = {}
        for ds in datasets:
            ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")["NNR"]
            ds_ranks = ds_data.rank(ascending=True)
            ranks[ds] = ds_ranks

        rank_df = pd.DataFrame(ranks)

        fig, ax = plt.subplots(figsize=(8, 6))

        x_positions = np.arange(len(datasets))

        for detector in rank_df.index:
            y_vals = [rank_df.loc[detector, ds] if ds in rank_df.columns and detector in rank_df.index else np.nan
                      for ds in datasets]

            ax.plot(x_positions, y_vals, 'o-', color=det_color_map[detector],
                    label=detector, linewidth=2.5, markersize=10, alpha=0.8)

            if not np.isnan(y_vals[-1]):
                ax.annotate(detector, (x_positions[-1] + 0.1, y_vals[-1]),
                            fontsize=9, va='center', color=det_color_map[detector])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel('Rank (1 = Best)', fontsize=11)
        ax.set_title(f'{get_backbone_name(backbone)}', fontweight='bold')

        ax.invert_yaxis()
        ax.set_ylim(len(detectors) + 0.5, 0.5)
        ax.set_xlim(-0.3, len(datasets) - 0.5)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        safe_bb = backbone.replace("-", "_").replace(" ", "_")
        fname = os.path.join(subdir, f"{safe_bb}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > bump_chart/ ({len(backbones)} files)")


# =============================================================================
# Q2: SYSTEM CONFIGURATION TRAJECTORIES
# =============================================================================

def plot_system_trajectories(agg_df, out_dir, severity=5):
    """
    System Configuration Trajectories: Faceted slope graph showing how each
    (backbone, detector) configuration performs across nuisance benchmarks at L5.

    X-axis: LN, C, CNS (nuisance benchmarks)
    Each line connects the same detector's OSA across these datasets.
    Val baseline shown as a distinct marker at x=-0.5 for each detector.
    Line crossings show ranking instability - the key Q2 insight.

    Layout: 2 rows × 3 columns (5 backbone panels + legend in 6th cell)
    Output: Q2/system_trajectories_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/system_trajectories skipped (no data)")
        return

    # Get baseline data (level == 0, typically ImageNet-Val)
    baseline_df = per_sev[per_sev["level"] == 0].copy()

    # Get severity data (nuisance benchmarks at specified level)
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty:
        print(f"  > Q2/system_trajectories skipped (no level {severity} data)")
        return

    if "OSA" not in sev_df.columns or sev_df["OSA"].isna().all():
        print("  > Q2/system_trajectories skipped (no OSA data)")
        return

    # Create Q2 output directory
    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in sev_df.columns:
        ood_datasets = sorted(sev_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in sev_df.columns:
            ood_data = sev_df[sev_df["test_ood_dataset"] == ood_ds].copy()
            baseline_ood = baseline_df[baseline_df["test_ood_dataset"] == ood_ds].copy() if "test_ood_dataset" in baseline_df.columns else baseline_df.copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = sev_df.copy()
            baseline_ood = baseline_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        # Nuisance datasets only (not Val)
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]
        datasets += [d for d in all_datasets if d not in desired_order and d != "ImageNet-Val"]

        if len(datasets) < 1:
            continue

        # Short labels for x-axis
        ds_labels = []
        for d in datasets:
            if d == "ImageNet-LN":
                ds_labels.append("LN")
            elif d == "ImageNet-C":
                ds_labels.append("C")
            else:
                ds_labels.append(d.replace("ImageNet-", ""))

        # Create figure: 2 rows × 3 columns
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7), squeeze=False)
        axes_flat = axes.flatten()

        # X positions: Val at -0.7, then nuisance datasets at 0, 1, 2
        x_nuisance = np.arange(len(datasets))
        x_val = -0.8

        # Marker styles for better distinction
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

        # Plot each backbone in its own panel
        for idx, backbone in enumerate(backbones):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]
            bb_baseline = baseline_ood[baseline_ood["backbone"] == backbone]

            for det_idx, detector in enumerate(detectors):
                det_data = bb_data[bb_data["detector"] == detector]
                det_baseline = bb_baseline[bb_baseline["detector"] == detector]

                # Get baseline (Val) OSA
                val_osa = None
                if not det_baseline.empty:
                    val_row = det_baseline[det_baseline["dataset"] == "ImageNet-Val"]
                    if not val_row.empty and "OSA" in val_row.columns:
                        val_osa = val_row["OSA"].values[0]

                # Get OSA values for each nuisance dataset
                osa_values = []
                for ds in datasets:
                    ds_val = det_data[det_data["dataset"] == ds]["OSA"]
                    if not ds_val.empty:
                        osa_values.append(ds_val.values[0])
                    else:
                        osa_values.append(np.nan)

                # Skip if all NaN
                if all(np.isnan(v) if isinstance(v, float) else False for v in osa_values):
                    continue

                color = get_detector_color(detector)
                linestyle = get_detector_linestyle(detector)
                marker = markers[det_idx % len(markers)]

                # Plot nuisance trajectory
                ax.plot(x_nuisance, osa_values,
                        marker=marker, linestyle=linestyle, color=color,
                        label=detector.upper(), linewidth=2.2, markersize=7, alpha=0.9,
                        markeredgecolor='white', markeredgewidth=0.5)

                # Plot Val baseline as a distinct diamond marker connected with thin dashed line
                if val_osa is not None and not np.isnan(val_osa):
                    ax.plot(x_val, val_osa, marker='D', color=color, markersize=6,
                            alpha=0.7, markeredgecolor='black', markeredgewidth=0.8)
                    # Connect Val to first nuisance point with thin dashed line
                    if not np.isnan(osa_values[0]):
                        ax.plot([x_val, x_nuisance[0]], [val_osa, osa_values[0]],
                               linestyle=':', color=color, linewidth=1.2, alpha=0.5)

            # Add vertical separator between Val and nuisance datasets
            ax.axvline(x=-0.4, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
            ax.set_xticks([x_val] + list(x_nuisance))
            ax.set_xticklabels(['Val'] + ds_labels, fontsize=10)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(-1.1, len(datasets) - 0.6)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlabel('Benchmark', fontsize=10)

            if idx % n_cols == 0:
                ax.set_ylabel('OSA @ L5', fontsize=11)

        # Use last cell for legend
        if len(backbones) < n_rows * n_cols:
            legend_ax = axes_flat[len(backbones)]
            legend_ax.axis('off')

            # Get handles and labels from first plot
            handles, labels = axes_flat[0].get_legend_handles_labels()

            # Sort by label for consistency
            sorted_pairs = sorted(zip(labels, handles))
            labels, handles = zip(*sorted_pairs) if sorted_pairs else ([], [])

            legend_ax.legend(handles, labels,
                            loc='center',
                            fontsize=10,
                            title='Detector',
                            title_fontsize=11,
                            framealpha=0.95,
                            edgecolor='gray',
                            ncol=2)

        # Hide any unused axes
        for idx in range(len(backbones) + 1, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle(f'System Configuration Trajectories (OSA @ L{severity})',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = os.path.join(q2_dir, f"system_trajectories_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/system_trajectories ({len(ood_datasets)} files)")


def plot_system_spread(agg_df, out_dir, severity=5):
    """
    System Spread: Boxplots showing the variance in OSA across all 50 configurations
    at each dataset. Demonstrates the 24pp gap between best and worst systems.

    Val baseline shown separately from nuisance benchmarks at L5.
    Output: Q2/system_spread_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/system_spread skipped (no data)")
        return

    # Get baseline (Val at level 0) and severity data separately
    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/system_spread skipped (no OSA data at L{severity})")
        return

    # Combine for full picture
    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        # Dataset order: Val first, then nuisance benchmarks
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]
        datasets += [d for d in all_datasets if d not in datasets]

        # Short labels
        ds_labels = []
        for d in datasets:
            if d == "ImageNet-Val":
                ds_labels.append("Val")
            elif d == "ImageNet-LN":
                ds_labels.append("LN")
            elif d == "ImageNet-C":
                ds_labels.append("C")
            else:
                ds_labels.append(d.replace("ImageNet-", ""))

        fig, ax = plt.subplots(figsize=(9, 5))

        # Collect OSA distributions per dataset
        data_to_plot = []
        for ds in datasets:
            ds_data = ood_data[ood_data["dataset"] == ds]["OSA"].dropna()
            data_to_plot.append(ds_data.values)

        # Create boxplot
        bp = ax.boxplot(data_to_plot, positions=range(len(datasets)),
                        patch_artist=True, widths=0.6)

        # Color boxes by dataset
        for i, (patch, ds) in enumerate(zip(bp['boxes'], datasets)):
            color = DATASET_COLORS.get(ds, '#666666')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points (jittered)
        for i, (ds, vals) in enumerate(zip(datasets, data_to_plot)):
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(i + jitter, vals, c='black', s=20, alpha=0.5, zorder=3)

        # Add spread annotation
        for i, vals in enumerate(data_to_plot):
            if len(vals) > 0:
                spread = np.max(vals) - np.min(vals)
                ax.annotate(f'Δ={spread:.0%}', (i, np.max(vals) + 0.02),
                           ha='center', fontsize=9, fontweight='bold')

        # Add vertical line separating Val from nuisance benchmarks
        if "ImageNet-Val" in datasets:
            ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(ds_labels, fontsize=10)
        ax.set_ylabel('OSA', fontsize=11)
        ax.set_xlabel('Benchmark', fontsize=11)
        ax.set_ylim(0.0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

        ax.set_title(f'System Configuration Spread (OSA @ L{severity})',
                     fontsize=12, fontweight='bold')

        plt.tight_layout()

        fname = os.path.join(q2_dir, f"system_spread_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/system_spread ({len(ood_datasets)} files)")


def plot_best_detector_matrix(agg_df, out_dir, severity=5):
    """
    Best Detector Matrix: Heatmap showing which detector is best at each
    (backbone, dataset) cell. Color changes indicate system dependence.

    Val baseline included as first column (level=0), nuisance benchmarks at L5.
    Output: Q2/best_detector_matrix_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/best_detector_matrix skipped (no data)")
        return

    # Get baseline (Val at level 0) and severity data
    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/best_detector_matrix skipped (no OSA data at L{severity})")
        return

    # Combine for full picture
    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())

        # Dataset order: Val first, then nuisance benchmarks
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]
        datasets += [d for d in all_datasets if d not in datasets]

        # Short labels
        ds_labels = []
        for d in datasets:
            if d == "ImageNet-Val":
                ds_labels.append("Val")
            elif d == "ImageNet-LN":
                ds_labels.append("LN")
            elif d == "ImageNet-C":
                ds_labels.append("C")
            else:
                ds_labels.append(d.replace("ImageNet-", ""))

        # Build best detector matrix
        best_matrix = []
        detectors = sorted(ood_data["detector"].unique())
        detector_to_idx = {d: i for i, d in enumerate(detectors)}

        for backbone in backbones:
            row = []
            for ds in datasets:
                cell_data = ood_data[(ood_data["backbone"] == backbone) &
                                      (ood_data["dataset"] == ds)]
                if not cell_data.empty:
                    best_idx = cell_data["OSA"].idxmax()
                    best_det = cell_data.loc[best_idx, "detector"]
                    row.append(detector_to_idx[best_det])
                else:
                    row.append(-1)
            best_matrix.append(row)

        best_matrix = np.array(best_matrix)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Create colormap from detector colors
        cmap_colors = [get_detector_color(d) for d in detectors]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_colors)

        im = ax.imshow(best_matrix, cmap=cmap, vmin=0, vmax=len(detectors)-1, aspect='auto')

        # Add text annotations
        for i, backbone in enumerate(backbones):
            for j, ds in enumerate(datasets):
                if best_matrix[i, j] >= 0:
                    det_name = detectors[best_matrix[i, j]]
                    ax.text(j, i, det_name.upper(), ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white',
                           path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(ds_labels, fontsize=10)
        ax.set_yticks(range(len(backbones)))
        ax.set_yticklabels([get_backbone_name(b) for b in backbones], fontsize=10)

        # Add vertical line separating Val from nuisance benchmarks
        if "ImageNet-Val" in datasets:
            ax.axvline(x=0.5, color='white', linestyle='-', linewidth=2)

        ax.set_xlabel('Benchmark', fontsize=11)
        ax.set_ylabel('Backbone', fontsize=11)
        ax.set_title(f'Best Detector by Configuration (OSA @ L{severity})',
                     fontsize=12, fontweight='bold')

        plt.tight_layout()

        fname = os.path.join(q2_dir, f"best_detector_matrix_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/best_detector_matrix ({len(ood_datasets)} files)")


def plot_q2_bump_chart(agg_df, out_dir, severity=5):
    """
    Q2 Bump Chart: Shows detector rank changes across datasets.
    Ranks computed based on OSA (higher is better, so rank 1 = highest OSA).

    Val shown as baseline, then LN, C, CNS as nuisance benchmarks.
    Output: Q2/bump_chart_L{severity}_{backbone}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/bump_chart skipped (no data)")
        return

    # Get baseline and severity data
    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty:
        print(f"  > Q2/bump_chart skipped (no level {severity} data)")
        return

    # Combine for full picture
    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    # Get unique OOD test datasets
    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        # Dataset order
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]
        datasets += [d for d in all_datasets if d not in datasets]

        if len(datasets) < 2:
            continue

        # Short labels
        ds_labels = []
        for d in datasets:
            if d == "ImageNet-Val":
                ds_labels.append("Val")
            elif d == "ImageNet-LN":
                ds_labels.append("LN")
            elif d == "ImageNet-C":
                ds_labels.append("C")
            else:
                ds_labels.append(d.replace("ImageNet-", ""))

        # Create figure: 2 rows × 3 columns (5 backbones + legend)
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7), squeeze=False)
        axes_flat = axes.flatten()

        x_positions = np.arange(len(datasets))
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

        for idx, backbone in enumerate(backbones):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Compute ranks per dataset (higher OSA = rank 1)
            ranks = {}
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                if "OSA" in ds_data.columns and not ds_data["OSA"].isna().all():
                    # Rank descending (highest OSA = rank 1)
                    ds_ranks = ds_data["OSA"].rank(ascending=False)
                    ranks[ds] = ds_ranks
                else:
                    ranks[ds] = pd.Series(dtype=float)

            if not ranks:
                continue

            rank_df = pd.DataFrame(ranks)

            for det_idx, detector in enumerate(detectors):
                if detector not in rank_df.index:
                    continue

                y_vals = [rank_df.loc[detector, ds] if ds in rank_df.columns and detector in rank_df.index
                         else np.nan for ds in datasets]

                if all(np.isnan(v) for v in y_vals):
                    continue

                color = get_detector_color(detector)
                marker = markers[det_idx % len(markers)]

                ax.plot(x_positions, y_vals, marker=marker, linestyle='-', color=color,
                        label=detector.upper(), linewidth=2.2, markersize=7, alpha=0.85,
                        markeredgecolor='white', markeredgewidth=0.5)

            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(ds_labels, fontsize=10)
            ax.set_ylabel('Rank (1=Best)', fontsize=10)
            ax.set_xlabel('Benchmark', fontsize=10)

            # Invert y-axis so rank 1 is at top
            ax.invert_yaxis()
            ax.set_ylim(len(detectors) + 0.5, 0.5)
            ax.set_xlim(-0.3, len(datasets) - 0.7)
            ax.grid(True, axis='y', alpha=0.3)

        # Legend in last cell
        if len(backbones) < n_rows * n_cols:
            legend_ax = axes_flat[len(backbones)]
            legend_ax.axis('off')

            handles, labels = axes_flat[0].get_legend_handles_labels()
            sorted_pairs = sorted(zip(labels, handles))
            labels, handles = zip(*sorted_pairs) if sorted_pairs else ([], [])

            legend_ax.legend(handles, labels, loc='center', fontsize=10,
                            title='Detector', title_fontsize=11,
                            framealpha=0.95, edgecolor='gray', ncol=2)

        for idx in range(len(backbones) + 1, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle(f'Detector Rank Changes (OSA @ L{severity})',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = os.path.join(q2_dir, f"bump_chart_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/bump_chart ({len(ood_datasets)} files)")


def plot_q2_rank_range(agg_df, out_dir, severity=5):
    """
    Rank Range Error Bars: Shows min-max rank range for each detector per backbone.
    Wide bars = unstable detector, narrow bars = stable.

    Layout: 2 rows × 3 columns (5 backbones + legend)
    Output: Q2/rank_range_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/rank_range skipped (no data)")
        return

    # Get baseline (level 0) and severity data
    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/rank_range skipped (no OSA data at L{severity})")
        return

    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        # Dataset order
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]

        if len(datasets) < 2:
            continue

        # Create figure
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7), squeeze=False)
        axes_flat = axes.flatten()

        for idx, backbone in enumerate(backbones):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Compute ranks per dataset
            ranks_per_det = {det: [] for det in detectors}
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                if "OSA" in ds_data.columns and not ds_data["OSA"].isna().all():
                    ds_ranks = ds_data["OSA"].rank(ascending=False)
                    for det in detectors:
                        if det in ds_ranks.index:
                            ranks_per_det[det].append(ds_ranks[det])

            # Plot error bars
            x_positions = np.arange(len(detectors))
            for i, det in enumerate(detectors):
                ranks = ranks_per_det[det]
                if len(ranks) > 0:
                    min_rank = min(ranks)
                    max_rank = max(ranks)
                    median_rank = np.median(ranks)
                    color = get_detector_color(det)

                    # Error bar from min to max rank
                    ax.errorbar(i, median_rank, yerr=[[median_rank - min_rank], [max_rank - median_rank]],
                               fmt='o', color=color, capsize=4, capthick=2, linewidth=2,
                               markersize=8, markeredgecolor='white', markeredgewidth=0.5)

            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([d.upper()[:4] for d in detectors], fontsize=8, rotation=45, ha='right')
            ax.invert_yaxis()
            ax.set_ylim(len(detectors) + 0.5, 0.5)
            ax.set_ylabel('Rank (1=Best)', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

        # Legend in last cell
        if len(backbones) < n_rows * n_cols:
            legend_ax = axes_flat[len(backbones)]
            legend_ax.axis('off')
            legend_ax.text(0.5, 0.5, "Wide bars = unstable\nNarrow bars = stable",
                          ha='center', va='center', fontsize=11, transform=legend_ax.transAxes)

        for idx in range(len(backbones) + 1, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle(f'Detector Rank Ranges Across Benchmarks (OSA @ L{severity})',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = os.path.join(q2_dir, f"rank_range_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/rank_range ({len(ood_datasets)} files)")


def plot_q2_stability_index(agg_df, out_dir, severity=5):
    """
    Rank Stability Index: Simple bar chart showing mean rank variance per backbone.
    Lower = more stable. ViT should have the lowest bar.

    Output: Q2/stability_index_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/stability_index skipped (no data)")
        return

    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/stability_index skipped (no OSA data at L{severity})")
        return

    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]

        if len(datasets) < 2:
            continue

        # Compute stability index per backbone
        stability_scores = {}
        for backbone in backbones:
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Compute rank variance for each detector, then average
            rank_variances = []
            for det in detectors:
                det_ranks = []
                for ds in datasets:
                    ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                    if "OSA" in ds_data.columns and det in ds_data.index:
                        ds_ranks = ds_data["OSA"].rank(ascending=False)
                        det_ranks.append(ds_ranks[det])

                if len(det_ranks) > 1:
                    rank_variances.append(np.var(det_ranks))

            stability_scores[backbone] = np.mean(rank_variances) if rank_variances else 0

        # Sort by stability (ascending = most stable first)
        sorted_bb = sorted(stability_scores.keys(), key=lambda x: stability_scores[x])

        fig, ax = plt.subplots(figsize=(8, 5))

        y_positions = np.arange(len(sorted_bb))
        colors = [get_backbone_color(bb) for bb in sorted_bb]
        values = [stability_scores[bb] for bb in sorted_bb]

        bars = ax.barh(y_positions, values, color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([get_backbone_name(bb) for bb in sorted_bb], fontsize=11)
        ax.set_xlabel('Mean Rank Variance (lower = more stable)', fontsize=11)
        ax.set_title(f'Ranking Stability Index (OSA @ L{severity})', fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(values) * 1.3)
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        fname = os.path.join(q2_dir, f"stability_index_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/stability_index ({len(ood_datasets)} files)")


def plot_q2_rank_divergence(agg_df, out_dir, severity=5):
    """
    Rank Divergence: Shows mean absolute rank change from Val baseline.
    One line per backbone. ViT should stay low/flat.

    Output: Q2/rank_divergence_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/rank_divergence skipped (no data)")
        return

    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/rank_divergence skipped (no OSA data at L{severity})")
        return

    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        # Nuisance datasets (not Val)
        all_datasets = sorted(ood_data["dataset"].unique())
        nuisance_datasets = ["ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in nuisance_datasets if d in all_datasets]

        if len(datasets) < 1 or "ImageNet-Val" not in all_datasets:
            continue

        # Short labels
        ds_labels = [d.replace("ImageNet-", "") for d in datasets]

        fig, ax = plt.subplots(figsize=(8, 5))

        x_positions = np.arange(len(datasets))

        for backbone in backbones:
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Get Val ranks
            val_data = bb_data[bb_data["dataset"] == "ImageNet-Val"].set_index("detector")
            if "OSA" not in val_data.columns or val_data["OSA"].isna().all():
                continue
            val_ranks = val_data["OSA"].rank(ascending=False)

            # Compute mean absolute rank change for each nuisance dataset
            divergences = []
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                if "OSA" in ds_data.columns and not ds_data["OSA"].isna().all():
                    ds_ranks = ds_data["OSA"].rank(ascending=False)

                    # Mean absolute rank change
                    rank_changes = []
                    for det in detectors:
                        if det in val_ranks.index and det in ds_ranks.index:
                            rank_changes.append(abs(val_ranks[det] - ds_ranks[det]))

                    divergences.append(np.mean(rank_changes) if rank_changes else 0)
                else:
                    divergences.append(np.nan)

            color = get_backbone_color(backbone)
            ax.plot(x_positions, divergences, 'o-', color=color,
                   label=get_backbone_name(backbone), linewidth=2.5, markersize=10)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ds_labels, fontsize=11)
        ax.set_xlabel('Nuisance Benchmark', fontsize=11)
        ax.set_ylabel('Mean Absolute Rank Change from Val', fontsize=11)
        ax.set_title(f'Ranking Divergence from Baseline (OSA @ L{severity})', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

        plt.tight_layout()

        fname = os.path.join(q2_dir, f"rank_divergence_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/rank_divergence ({len(ood_datasets)} files)")


def plot_q2_rank_correlation(agg_df, out_dir, severity=5):
    """
    Rank Correlation Heatmap: Shows Spearman ρ between dataset rankings.
    Faceted by backbone. Low correlation = rankings don't transfer.

    Output: Q2/rank_correlation_L{severity}_{ood_ds}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/rank_correlation skipped (no data)")
        return

    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/rank_correlation skipped (no OSA data at L{severity})")
        return

    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]

        if len(datasets) < 2:
            continue

        # Short labels
        ds_labels = []
        for d in datasets:
            if d == "ImageNet-Val":
                ds_labels.append("Val")
            elif d == "ImageNet-LN":
                ds_labels.append("LN")
            elif d == "ImageNet-C":
                ds_labels.append("C")
            else:
                ds_labels.append(d.replace("ImageNet-", ""))

        # Create faceted figure
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7), squeeze=False)
        axes_flat = axes.flatten()

        for idx, backbone in enumerate(backbones):
            if idx >= len(axes_flat):
                break

            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Get ranks per dataset
            ranks_by_ds = {}
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                if "OSA" in ds_data.columns and not ds_data["OSA"].isna().all():
                    ranks_by_ds[ds] = ds_data["OSA"].rank(ascending=False)

            # Compute correlation matrix
            corr_matrix = np.zeros((len(datasets), len(datasets)))
            for i, ds1 in enumerate(datasets):
                for j, ds2 in enumerate(datasets):
                    if ds1 in ranks_by_ds and ds2 in ranks_by_ds:
                        r1 = ranks_by_ds[ds1]
                        r2 = ranks_by_ds[ds2]
                        common = r1.index.intersection(r2.index)
                        if len(common) > 2:
                            rho, _ = spearmanr(r1[common], r2[common])
                            corr_matrix[i, j] = rho
                        else:
                            corr_matrix[i, j] = np.nan
                    else:
                        corr_matrix[i, j] = np.nan

            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

            # Add text annotations
            for i in range(len(datasets)):
                for j in range(len(datasets)):
                    if not np.isnan(corr_matrix[i, j]):
                        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                               fontsize=9, color=text_color, fontweight='bold')

            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(ds_labels, fontsize=10)
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels(ds_labels, fontsize=10)
            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')

        # Colorbar in last cell
        if len(backbones) < n_rows * n_cols:
            cbar_ax = axes_flat[len(backbones)]
            cbar_ax.axis('off')
            # Add colorbar manually
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-1, 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=cbar_ax, shrink=0.8, aspect=20)
            cbar.set_label("Spearman ρ", fontsize=10)

        for idx in range(len(backbones) + 1, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle(f'Ranking Correlation Between Benchmarks (OSA @ L{severity})',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = os.path.join(q2_dir, f"rank_correlation_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/rank_correlation ({len(ood_datasets)} files)")


def plot_q2_rank_stability_combined(agg_df, out_dir, severity=5):
    """
    Combined Rank Stability Figure: 2×3 grid with rank range error bars (5 panels)
    and stability index bar chart in the 6th cell.

    Design specs:
    - No title (no suptitle)
    - Y-axis "Rank" only on left column (cells 0, 3)
    - X-axis labels only on bottom row (cells 3, 4, 5)
    - Cell 5: Stability index horizontal bar chart with "Mean Rank Variance" x-label

    Output: Q2/rank_stability_combined_L{severity}_{ood}.pdf
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        print("  > Q2/rank_stability_combined skipped (no data)")
        return

    # Get baseline (level 0) and severity data
    baseline_df = per_sev[per_sev["level"] == 0].copy()
    sev_df = per_sev[per_sev["level"] == severity].copy()

    if sev_df.empty or "OSA" not in sev_df.columns:
        print(f"  > Q2/rank_stability_combined skipped (no OSA data at L{severity})")
        return

    combined_df = pd.concat([baseline_df, sev_df], ignore_index=True)

    q2_dir = os.path.join(out_dir, "Q2")
    os.makedirs(q2_dir, exist_ok=True)

    if "test_ood_dataset" in combined_df.columns:
        ood_datasets = sorted(combined_df["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    for ood_ds in ood_datasets:
        if ood_ds and "test_ood_dataset" in combined_df.columns:
            ood_data = combined_df[combined_df["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_').replace(' ', '_')}"
        else:
            ood_data = combined_df.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        backbones = sorted(ood_data["backbone"].unique())
        detectors = sorted(ood_data["detector"].unique())

        # Dataset order
        all_datasets = sorted(ood_data["dataset"].unique())
        desired_order = ["ImageNet-Val", "ImageNet-LN", "ImageNet-C", "CNS"]
        datasets = [d for d in desired_order if d in all_datasets]

        if len(datasets) < 2:
            continue

        # Create 2×3 figure
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7), squeeze=False)
        axes_flat = axes.flatten()

        # --- Compute stability scores for cell 5 ---
        stability_scores = {}
        for backbone in backbones:
            bb_data = ood_data[ood_data["backbone"] == backbone]
            rank_variances = []
            for det in detectors:
                det_ranks = []
                for ds in datasets:
                    ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                    if "OSA" in ds_data.columns and det in ds_data.index:
                        ds_ranks = ds_data["OSA"].rank(ascending=False)
                        det_ranks.append(ds_ranks[det])
                if len(det_ranks) > 1:
                    rank_variances.append(np.var(det_ranks))
            stability_scores[backbone] = np.mean(rank_variances) if rank_variances else 0

        # --- Plot rank range error bars for cells 0-4 ---
        for idx, backbone in enumerate(backbones):
            if idx >= 5:
                break

            ax = axes_flat[idx]
            bb_data = ood_data[ood_data["backbone"] == backbone]

            # Compute ranks per dataset
            ranks_per_det = {det: [] for det in detectors}
            for ds in datasets:
                ds_data = bb_data[bb_data["dataset"] == ds].set_index("detector")
                if "OSA" in ds_data.columns and not ds_data["OSA"].isna().all():
                    ds_ranks = ds_data["OSA"].rank(ascending=False)
                    for det in detectors:
                        if det in ds_ranks.index:
                            ranks_per_det[det].append(ds_ranks[det])

            # Plot error bars
            x_positions = np.arange(len(detectors))
            for i, det in enumerate(detectors):
                ranks = ranks_per_det[det]
                if len(ranks) > 0:
                    min_rank = min(ranks)
                    max_rank = max(ranks)
                    median_rank = np.median(ranks)
                    color = get_detector_color(det)

                    ax.errorbar(i, median_rank, yerr=[[median_rank - min_rank], [max_rank - median_rank]],
                               fmt='o', color=color, capsize=4, capthick=2, linewidth=2,
                               markersize=8, markeredgecolor='white', markeredgewidth=0.5)

            ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.set_ylim(len(detectors) + 0.5, 0.5)
            ax.grid(True, axis='y', alpha=0.3)

            # Y-axis label and ticks only on left column (cells 0, 3)
            if idx in [0, 3]:
                ax.set_ylabel('Rank', fontsize=10)
                ax.set_yticks(range(1, len(detectors) + 1))
            else:
                ax.set_ylabel('')
                ax.set_yticks(range(1, len(detectors) + 1))
                ax.set_yticklabels([])

            # X-axis labels on bottom row (cells 3, 4) and top-right (cell 2)
            if idx in [2, 3, 4]:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([d.upper()[:4] for d in detectors], fontsize=8, rotation=45, ha='right')
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])

        # --- Cell 5: Stability index bar chart ---
        stability_ax = axes_flat[5]

        # Sort backbones by stability (most stable = lowest variance)
        sorted_bb = sorted(stability_scores.keys(), key=lambda x: stability_scores[x])

        y_positions = np.arange(len(sorted_bb))
        colors = [get_backbone_color(bb) for bb in sorted_bb]
        values = [stability_scores[bb] for bb in sorted_bb]
        labels = [get_backbone_name(bb) for bb in sorted_bb]

        bars = stability_ax.barh(y_positions, values, color=colors, edgecolor='black', linewidth=0.5)

        stability_ax.set_yticks(y_positions)
        stability_ax.set_yticklabels(labels, fontsize=9)
        stability_ax.set_xlabel('Mean Rank Variance', fontsize=10)
        stability_ax.invert_yaxis()  # Most stable at top
        stability_ax.grid(True, axis='x', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            stability_ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                            f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()

        fname = os.path.join(q2_dir, f"rank_stability_combined_L{severity}{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > Q2/rank_stability_combined ({len(ood_datasets)} files)")


def plot_q2_all(agg_df, out_dir, severity=5):
    """Generate all Q2 (system-dependent nuisance novelty) plots."""
    print("\n  Q2 - System-Dependent Nuisance Novelty:")
    plot_system_trajectories(agg_df, out_dir, severity=severity)
    plot_system_spread(agg_df, out_dir, severity=severity)
    plot_best_detector_matrix(agg_df, out_dir, severity=severity)
    plot_q2_bump_chart(agg_df, out_dir, severity=severity)
    # Ranking stability visualizations
    plot_q2_rank_range(agg_df, out_dir, severity=severity)
    plot_q2_stability_index(agg_df, out_dir, severity=severity)
    plot_q2_rank_divergence(agg_df, out_dir, severity=severity)
    plot_q2_rank_correlation(agg_df, out_dir, severity=severity)
    # Combined rank stability figure (rank range + stability index)
    plot_q2_rank_stability_combined(agg_df, out_dir, severity=severity)


# =============================================================================
# 12. NUISANCE CATEGORY HEATMAP
# =============================================================================

def plot_nuisance_category_heatmap(agg_df, out_dir, severity=5):
    """
    Heatmap: rows=individual nuisances, cols=(dataset, backbone), values=mean NNR.
    Generates separate heatmaps for each OOD test dataset if present.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty or "nuisance" not in agg_df.columns:
        print("  > Nuisance heatmap skipped (no nuisance data)")
        return

    corrupted = agg_df[(agg_df["level"] == severity)].copy()
    if corrupted.empty:
        return

    # Get unique OOD test datasets
    if "test_ood_dataset" in corrupted.columns:
        ood_datasets = sorted(corrupted["test_ood_dataset"].unique())
    else:
        ood_datasets = [None]

    subdir = os.path.join(out_dir, "nuisance_heatmap")
    os.makedirs(subdir, exist_ok=True)

    for ood_ds in ood_datasets:
        # Filter to this OOD dataset
        if ood_ds and "test_ood_dataset" in corrupted.columns:
            ood_data = corrupted[corrupted["test_ood_dataset"] == ood_ds].copy()
            ood_suffix = f"_{ood_ds.replace('-', '_')}"
        else:
            ood_data = corrupted.copy()
            ood_suffix = ""

        if ood_data.empty:
            continue

        datasets = sorted(ood_data["dataset"].unique())

        ln_datasets = [d for d in datasets if "-ln" in d.lower() or d.lower().endswith("ln")]
        if not ln_datasets:
            ln_datasets = datasets

        # Simplified view: Nuisance × Dataset (averaged over backbones)
        nuis_simple = ood_data.groupby(["nuisance", "dataset"])["NNR"].mean().reset_index()
        nuis_simple = nuis_simple[nuis_simple["dataset"].isin(ln_datasets)]

        if nuis_simple.empty:
            continue

        pivot_simple = nuis_simple.pivot(index="nuisance", columns="dataset", values="NNR")
        pivot_simple = pivot_simple.loc[pivot_simple.mean(axis=1).sort_values(ascending=False).index]

        col_order_simple = [d for d in ln_datasets if d in pivot_simple.columns]
        pivot_simple = pivot_simple[col_order_simple]

        fig, ax = plt.subplots(figsize=(8, max(6, len(pivot_simple) * 0.4)))

        sns.heatmap(pivot_simple, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0, vmax=1.0, linewidths=0.5,
                    cbar_kws={'label': f'Mean NNR@L{severity}'},
                    ax=ax, annot_kws={'fontsize': 9})

        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel('Nuisance Type', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', labelsize=9)

        plt.tight_layout()
        fname = os.path.join(subdir, f"averaged{ood_suffix}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > nuisance_heatmap/ ({len(ood_datasets)} files)")


# =============================================================================
# WRAPPER FUNCTIONS
# =============================================================================

def plot_per_dataset_all(agg_df, dataset, out_dir):
    """All per-dataset plots."""
    print(f"\n  {dataset}:")
    plot_csa_ccr_by_detector(agg_df, dataset, out_dir)
    plot_osa_csa_by_detector(agg_df, dataset, out_dir)
    plot_fingerprint_by_detector(agg_df, dataset, out_dir)
    plot_nnr_severity_by_dataset(agg_df, dataset, out_dir)
    plot_osa_severity_by_dataset(agg_df, dataset, out_dir)


def plot_benchmark_all(agg_df, out_dir):
    """All benchmark comparison plots."""
    print("\n  Benchmark Comparisons:")
    plot_csa_ccr_benchmark_comparison(agg_df, out_dir)
    plot_osa_csa_benchmark_comparison(agg_df, out_dir)  # OSA vs CSA (per OOD dataset)
    plot_gap_comparison_by_detector(agg_df, out_dir)
    plot_nnr_heatmaps_all_datasets(agg_df, out_dir, severity=5)
    plot_osa_heatmaps_all_datasets(agg_df, out_dir, severity=5)
    plot_adr_heatmaps_all_datasets(agg_df, out_dir)
    plot_adr_violin(agg_df, out_dir)        # ADR based on CSA-CCR
    plot_adr_osa_violin(agg_df, out_dir)    # ADR based on CSA-OSA
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="NNR")
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="CSA")
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="OSA")
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="OSA_Gap")


def plot_cross_dataset_all(agg_df, out_dir, reference="ImageNet-LN"):
    """All cross-dataset plots."""
    print("\n  Cross-Dataset:")

    datasets = sorted(agg_df["dataset"].unique()) if "dataset" in agg_df.columns else []
    print(f"    Available datasets: {datasets}")
    print(f"    Reference: {reference}")

    plot_transferability_by_backbone(agg_df, out_dir, reference)
    plot_transferability_combined(agg_df, out_dir, reference)
    plot_finegrained_faceted(agg_df, out_dir)
    plot_detector_bump_chart(agg_df, out_dir)
    plot_nuisance_category_heatmap(agg_df, out_dir)