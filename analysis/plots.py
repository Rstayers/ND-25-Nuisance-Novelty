# analysis/plots.py
# Paper-ready plots with cohesive color scheme and proper formatting
#
# FIXES APPLIED:
# - Legends not cut off, full backbone names
# - Centered x-axis labels for multi-panel plots
# - Cohesive color scheme (Paul Tol's qualitative palette)
# - Equal-sized heatmap boxes with NNR range 0-1
# - No suptitles (removed for paper)

import os
import matplotlib.pyplot as plt
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
    'postmax': '#BBBBBB',  # Gray
}

# Metric colors
COLORS = {
    'CSA': '#4477AA',  # Blue
    'CCR': '#EE6677',  # Rose
    'NNR': '#CCBB44',  # Gold
    'OSA_Gap': '#AA3377',  # Purple
    'OSA': '#228833',  # Green
    'CNR': '#66CCEE',  # Cyan
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
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    n_bb = len(backbones)
    n_cols = 3
    n_rows = (n_bb + n_cols - 1) // n_cols  # Typically 2 rows for 5 backbones

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
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
                        color=color, label=detector, markersize=5, linewidth=1.8)

        ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 1.0)

        # Only left column gets y-label
        if idx % n_cols == 0:
            ax.set_ylabel('NNR', fontsize=11)

    # Hide unused axes but use the space for legend
    for idx in range(len(backbones), len(axes_flat)):
        axes_flat[idx].axis('off')

    # Centered x-axis label at bottom
    fig.text(0.5, 0.02, 'Severity Level', ha='center', fontsize=12)

    # Legend in the empty bottom-right cell (if 5 backbones in 2x3 grid, cell [1,2] is empty)
    handles, labels = axes_flat[0].get_legend_handles_labels()

    if len(backbones) < n_rows * n_cols:
        # Place legend in the empty subplot area
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
        # Fallback: place outside on right
        fig.legend(handles, labels,
                   loc='center right',
                   bbox_to_anchor=(0.99, 0.5),
                   fontsize=9,
                   title='Detector',
                   title_fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1.0, 1.0])

    subdir = os.path.join(out_dir, "nnr_severity")
    os.makedirs(subdir, exist_ok=True)
    fname = os.path.join(subdir, f"{dataset.replace('-', '_')}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > nnr_severity/{dataset.replace('-', '_')}.pdf")


# =============================================================================
# 4b. CNR vs SEVERITY - Stacked layout matching NNR
# =============================================================================

def plot_cnr_severity_by_dataset(agg_df, dataset, out_dir):
    """
    CNR vs severity: ONE FILE PER DATASET.
    Stacked layout (2 rows x 3 cols), legend in empty cell.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]

    if "CNR" not in per_sev.columns or per_sev["CNR"].isna().all():
        print(f"  > cnr_severity/{dataset.replace('-', '_')}.pdf skipped (no CNR data)")
        return

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    n_bb = len(backbones)
    n_cols = 3
    n_rows = (n_bb + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, backbone in enumerate(backbones):
        ax = axes_flat[idx]
        bb_data = per_sev[per_sev["backbone"] == backbone]

        for detector in detectors:
            det_data = bb_data[bb_data["detector"] == detector].sort_values("level")
            if not det_data.empty:
                color = get_detector_color(detector)
                ax.plot(det_data["level"], det_data["CNR"], 'o-',
                        color=color, label=detector, markersize=5, linewidth=1.8)

        ax.set_title(get_backbone_name(backbone), fontsize=11, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 1.0)

        if idx % n_cols == 0:
            ax.set_ylabel('CNR', fontsize=11)

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

    subdir = os.path.join(out_dir, "cnr_severity")
    os.makedirs(subdir, exist_ok=True)
    fname = os.path.join(subdir, f"{dataset.replace('-', '_')}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > cnr_severity/{dataset.replace('-', '_')}.pdf")


# =============================================================================
# 4c. OSA vs SEVERITY - Stacked layout matching NNR
# =============================================================================

def plot_osa_severity_by_dataset(agg_df, dataset, out_dir):
    """
    OSA vs severity: ONE FILE PER DATASET.
    Stacked layout (2 rows x 3 cols), legend in empty cell.
    """
    per_sev = get_metrics_by_severity(agg_df, dataset)
    if per_sev.empty:
        return

    per_sev = per_sev[per_sev["level"] > 0]

    if "OSA" not in per_sev.columns or per_sev["OSA"].isna().all():
        print(f"  > osa_severity/{dataset.replace('-', '_')}.pdf skipped (no OSA data)")
        return

    backbones = sorted(per_sev["backbone"].unique()) if "backbone" in per_sev.columns else ["default"]
    detectors = sorted(per_sev["detector"].unique()) if "detector" in per_sev.columns else ["default"]

    n_bb = len(backbones)
    n_cols = 3
    n_rows = (n_bb + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, backbone in enumerate(backbones):
        ax = axes_flat[idx]
        bb_data = per_sev[per_sev["backbone"] == backbone]

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

    subdir = os.path.join(out_dir, "osa_severity")
    os.makedirs(subdir, exist_ok=True)
    fname = os.path.join(subdir, f"{dataset.replace('-', '_')}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > osa_severity/{dataset.replace('-', '_')}.pdf")


# =============================================================================
# 5. NNR HEATMAPS: 3 datasets side-by-side, EQUAL SIZE BOXES, range 0-1
# =============================================================================

def plot_nnr_heatmaps_all_datasets(agg_df, out_dir, severity=5):
    """
    NNR@L5 heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty:
        return

    datasets = sorted(sev_df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

    # Limit to 3 datasets for the combined plot
    ds_order = ds_order[:3]

    backbones = sorted(sev_df["backbone"].unique())
    detectors = sorted(sev_df["detector"].unique())

    n_ds = len(ds_order)

    # Calculate figure size for equal boxes
    n_det = len(detectors)
    n_bb = len(backbones)
    cell_size = 0.8  # Size per cell in inches
    fig_width = n_ds * (n_bb * cell_size + 1.5) + 2  # Extra for colorbar
    fig_height = n_det * cell_size + 1.5

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ds_idx, dataset in enumerate(ds_order):
        ax = axes[ds_idx]
        ds_data = sev_df[sev_df["dataset"] == dataset]

        pivot = ds_data.pivot(index="detector", columns="backbone", values="NNR")
        pivot = pivot.reindex(index=detectors, columns=backbones)

        # Rename columns to display names
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0, vmax=1.0, linewidths=0.5,
                    cbar=(ds_idx == n_ds - 1),
                    cbar_kws={'label': f'NNR@L{severity}', 'shrink': 0.8} if ds_idx == n_ds - 1 else {},
                    ax=ax, annot_kws={'fontsize': 9},
                    square=True)  # EQUAL SIZE BOXES

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Detector' if ds_idx == 0 else '', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"NNR_heatmaps_L{severity}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > NNR_heatmaps_L{severity}.pdf")


# =============================================================================
# 5b. CNR HEATMAPS
# =============================================================================

def plot_cnr_heatmaps_all_datasets(agg_df, out_dir, severity=5):
    """
    CNR@L5 heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty or "CNR" not in sev_df.columns or sev_df["CNR"].isna().all():
        print(f"  > CNR_heatmaps_L{severity}.pdf skipped (no CNR data)")
        return

    datasets = sorted(sev_df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
    ds_order = ds_order[:3]

    backbones = sorted(sev_df["backbone"].unique())
    detectors = sorted(sev_df["detector"].unique())

    n_ds = len(ds_order)
    n_det = len(detectors)
    n_bb = len(backbones)
    cell_size = 0.8
    fig_width = n_ds * (n_bb * cell_size + 1.5) + 2
    fig_height = n_det * cell_size + 1.5

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ds_idx, dataset in enumerate(ds_order):
        ax = axes[ds_idx]
        ds_data = sev_df[sev_df["dataset"] == dataset]

        pivot = ds_data.pivot(index="detector", columns="backbone", values="CNR")
        pivot = pivot.reindex(index=detectors, columns=backbones)
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0, vmax=1.0, linewidths=0.5,
                    cbar=(ds_idx == n_ds - 1),
                    cbar_kws={'label': f'CNR@L{severity}', 'shrink': 0.8} if ds_idx == n_ds - 1 else {},
                    ax=ax, annot_kws={'fontsize': 9},
                    square=True)

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Detector' if ds_idx == 0 else '', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"CNR_heatmaps_L{severity}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > CNR_heatmaps_L{severity}.pdf")


# =============================================================================
# 5c. OSA HEATMAPS
# =============================================================================

def plot_osa_heatmaps_all_datasets(agg_df, out_dir, severity=5):
    """
    OSA@L5 heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty:
        return

    sev_df = per_sev[per_sev["level"] == severity].copy()
    if sev_df.empty or "OSA" not in sev_df.columns or sev_df["OSA"].isna().all():
        print(f"  > OSA_heatmaps_L{severity}.pdf skipped (no OSA data)")
        return

    datasets = sorted(sev_df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
    ds_order = ds_order[:3]

    backbones = sorted(sev_df["backbone"].unique())
    detectors = sorted(sev_df["detector"].unique())

    n_ds = len(ds_order)
    n_det = len(detectors)
    n_bb = len(backbones)
    cell_size = 0.8
    fig_width = n_ds * (n_bb * cell_size + 1.5) + 2
    fig_height = n_det * cell_size + 1.5

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for ds_idx, dataset in enumerate(ds_order):
        ax = axes[ds_idx]
        ds_data = sev_df[sev_df["dataset"] == dataset]

        pivot = ds_data.pivot(index="detector", columns="backbone", values="OSA")
        pivot = pivot.reindex(index=detectors, columns=backbones)
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.3, vmax=1.0, linewidths=0.5,
                    cbar=(ds_idx == n_ds - 1),
                    cbar_kws={'label': f'OSA@L{severity}', 'shrink': 0.8} if ds_idx == n_ds - 1 else {},
                    ax=ax, annot_kws={'fontsize': 9},
                    square=True)

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Detector' if ds_idx == 0 else '', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"OSA_heatmaps_L{severity}.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > OSA_heatmaps_L{severity}.pdf")


# =============================================================================
# 6. ADR HEATMAPS
# =============================================================================

def plot_adr_violin(agg_df, out_dir):
    """
    ADR violin plot: Shows distribution of ADR values per dataset.
    Clearly demonstrates systematic difference between benchmarks.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR_violin.pdf skipped (no ADR data)")
        return

    datasets = sorted(adr_df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]

    # Reorder dataframe
    adr_df["dataset"] = pd.Categorical(adr_df["dataset"], categories=ds_order, ordered=True)
    adr_df = adr_df.sort_values("dataset")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color palette for datasets
    palette = [DATASET_COLORS.get(ds, '#666666') for ds in ds_order]

    # Violin plot
    parts = ax.violinplot(
        [adr_df[adr_df["dataset"] == ds]["ADR"].values for ds in ds_order],
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
        ds_data = adr_df[adr_df["dataset"] == ds]["ADR"].values
        jitter = np.random.uniform(-0.15, 0.15, size=len(ds_data))
        ax.scatter(i + jitter, ds_data, c='black', s=25, alpha=0.6, zorder=3)

    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Add mean annotations
    for i, ds in enumerate(ds_order):
        mean_val = adr_df[adr_df["dataset"] == ds]["ADR"].mean()
        ax.annotate(f'μ={mean_val:.3f}',
                    xy=(i, mean_val), xytext=(i + 0.35, mean_val),
                    fontsize=10, fontweight='bold',
                    color=palette[i],
                    va='center')

    ax.set_xticks(range(len(ds_order)))
    ax.set_xticklabels(ds_order, fontsize=11)
    ax.set_ylabel('ADR (Accuracy Divergence Rate)', fontsize=11)
    ax.set_xlabel('Benchmark', fontsize=11)

    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.5, len(ds_order) - 0.5)

    plt.tight_layout()
    fname = os.path.join(out_dir, "ADR_violin.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > ADR_violin.pdf")


def plot_adr_heatmaps_all_datasets(agg_df, out_dir):
    """
    ADR heatmaps: 3 datasets side-by-side with EQUAL SIZE BOXES.
    """
    adr_df = compute_adr(agg_df)
    if adr_df.empty:
        print("  > ADR_heatmaps.pdf skipped (no ADR data)")
        return

    datasets = sorted(adr_df["dataset"].unique())
    desired_order = ["ImageNet-LN", "ImageNet-C", "CNS", "CUB-LN", "Cars-LN"]
    ds_order = [d for d in desired_order if d in datasets] + [d for d in datasets if d not in desired_order]
    ds_order = ds_order[:3]

    backbones = sorted(adr_df["backbone"].unique())
    detectors = sorted(adr_df["detector"].unique())

    n_ds = len(ds_order)
    n_det = len(detectors)
    n_bb = len(backbones)
    cell_size = 0.8
    fig_width = n_ds * (n_bb * cell_size + 1.5) + 2
    fig_height = n_det * cell_size + 1.5

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    vmax = adr_df["ADR"].abs().max() * 1.1

    for ds_idx, dataset in enumerate(ds_order):
        ax = axes[ds_idx]
        ds_data = adr_df[adr_df["dataset"] == dataset]

        pivot = ds_data.pivot(index="detector", columns="backbone", values="ADR")
        pivot = pivot.reindex(index=detectors, columns=backbones)
        pivot.columns = [get_backbone_name(b) for b in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    vmin=-vmax, vmax=vmax, linewidths=0.5, center=0,
                    cbar=(ds_idx == n_ds - 1),
                    cbar_kws={'label': 'ADR (slope)', 'shrink': 0.8} if ds_idx == n_ds - 1 else {},
                    ax=ax, annot_kws={'fontsize': 8},
                    square=True)

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Classifier', fontsize=10)
        ax.set_ylabel('Detector' if ds_idx == 0 else '', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fname = os.path.join(out_dir, "ADR_heatmaps.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > ADR_heatmaps.pdf")


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
    """
    mean_df = get_mean_metrics(agg_df)
    if mean_df.empty:
        return

    if metric not in mean_df.columns or mean_df[metric].isna().all():
        print(f"  > config_benchmark_{metric}_heatmap.pdf skipped (no data)")
        return

    mean_df["Config"] = mean_df["backbone"].apply(get_backbone_name) + " + " + mean_df["detector"]

    pivot = mean_df.pivot(index="Config", columns="dataset", values=metric)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.35)))

    if metric in ["NNR", "CNR"]:
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
    fname = os.path.join(out_dir, f"config_benchmark_{metric}_heatmap.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  > config_benchmark_{metric}_heatmap.pdf")


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

        # Custom legend with FULL BACKBONE NAMES
        from matplotlib.lines import Line2D

        bb_handles = [Line2D([0], [0], color=get_backbone_color(bb), linewidth=2.5,
                             label=get_backbone_name(bb))
                      for bb in backbones]

        metric_handles = [
            Line2D([0], [0], color='#333333', linestyle='-', linewidth=2, marker='o', markersize=6, label='CSA'),
            Line2D([0], [0], color='#333333', linestyle='--', linewidth=2, marker='s', markersize=6, label='CCR'),
        ]

        # Place legends OUTSIDE plot area with enough space
        leg1 = fig.legend(handles=bb_handles, title='Classifier',
                          loc='upper right', bbox_to_anchor=(0.99, 0.95),
                          fontsize=10, title_fontsize=11, framealpha=0.95,
                          edgecolor='gray', borderaxespad=0)

        leg2 = fig.legend(handles=metric_handles, title='Metric',
                          loc='lower right', bbox_to_anchor=(0.99, 0.15),
                          fontsize=10, title_fontsize=11, framealpha=0.95,
                          edgecolor='gray', borderaxespad=0)

        plt.tight_layout(rect=[0, 0, 0.82, 1.0])

        fname = os.path.join(subdir, f"{detector}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  > csa_ccr_benchmark/ ({len(detectors)} files)")


# =============================================================================
# 10. FINE-GRAINED FACETED: CUB vs Cars
# =============================================================================

def plot_finegrained_faceted(agg_df, out_dir, severity=5):
    """
    CUB vs Cars correlation: TWO FIGURES - NNR@L5 and mean NNR.
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
        n_cols = min(3, n_bb)
        n_rows = (n_bb + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()

        for idx, backbone in enumerate(backbones):
            ax = axes[idx]
            bb_data = data_df[data_df["backbone"] == backbone]

            cub_data = bb_data[bb_data["dataset"] == cub_name].set_index("detector")["NNR"]
            cars_data = bb_data[bb_data["dataset"] == cars_name].set_index("detector")["NNR"]

            common = cub_data.index.intersection(cars_data.index)

            if len(common) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center')
                ax.set_title(get_backbone_name(backbone), fontweight='bold')
                continue

            x = cub_data.loc[common].values
            y = cars_data.loc[common].values

            rho, pval = spearmanr(x, y)

            for det in common:
                ax.scatter(cub_data.loc[det], cars_data.loc[det], s=120,
                           c=[det_color_map[det]], edgecolors='black', linewidths=0.5,
                           alpha=0.8, label=det if idx == 0 else "")

            max_val = max(max(x), max(y)) * 1.15
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1.5)

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel(f'{label} on {cub_name}', fontsize=10)
            ax.set_ylabel(f'{label} on {cars_name}', fontsize=10)
            ax.set_title(get_backbone_name(backbone), fontweight='bold')

            ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {pval:.4f}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        for idx in range(len(backbones), len(axes)):
            axes[idx].set_visible(False)

        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc='center right', bbox_to_anchor=(1.02, 0.5),
                   fontsize=9, title='Detector', title_fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 1.0])

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
# 12. NUISANCE CATEGORY HEATMAP
# =============================================================================

def plot_nuisance_category_heatmap(agg_df, out_dir, severity=5):
    """
    Heatmap: rows=individual nuisances, cols=(dataset, backbone), values=mean NNR.
    """
    per_sev = get_metrics_by_severity(agg_df)
    if per_sev.empty or "nuisance" not in agg_df.columns:
        print("  > Nuisance heatmap skipped (no nuisance data)")
        return

    corrupted = agg_df[(agg_df["level"] == severity)].copy()
    if corrupted.empty:
        return

    backbones = sorted(corrupted["backbone"].unique()) if "backbone" in corrupted.columns else ["default"]
    datasets = sorted(corrupted["dataset"].unique())

    ln_datasets = [d for d in datasets if "-ln" in d.lower() or d.lower().endswith("ln")]
    if not ln_datasets:
        ln_datasets = datasets

    subdir = os.path.join(out_dir, "nuisance_heatmap")
    os.makedirs(subdir, exist_ok=True)

    # Simplified view: Nuisance × Dataset (averaged over backbones)
    nuis_simple = corrupted.groupby(["nuisance", "dataset"])["NNR"].mean().reset_index()
    nuis_simple = nuis_simple[nuis_simple["dataset"].isin(ln_datasets)]

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
    fname = os.path.join(subdir, "averaged.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  > nuisance_heatmap/ (1 file)")


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
    plot_cnr_severity_by_dataset(agg_df, dataset, out_dir)
    plot_osa_severity_by_dataset(agg_df, dataset, out_dir)


def plot_benchmark_all(agg_df, out_dir):
    """All benchmark comparison plots."""
    print("\n  Benchmark Comparisons:")
    plot_csa_ccr_benchmark_comparison(agg_df, out_dir)
    plot_gap_comparison_by_detector(agg_df, out_dir)
    plot_nnr_heatmaps_all_datasets(agg_df, out_dir, severity=5)
    plot_cnr_heatmaps_all_datasets(agg_df, out_dir, severity=5)
    plot_osa_heatmaps_all_datasets(agg_df, out_dir, severity=5)
    plot_adr_heatmaps_all_datasets(agg_df, out_dir)
    plot_adr_violin(agg_df, out_dir)  # NEW: ADR distribution comparison
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="NNR")
    plot_config_benchmark_heatmap(agg_df, out_dir, metric="CNR")
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