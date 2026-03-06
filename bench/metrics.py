# bench/metrics.py - Streamlined metrics (OSA/CCR focus)

import numpy as np
import torch


def compute_id_threshold(confidences, tpr=0.95):
    """
    Determines the threshold where 95% (TPR) of CLEAN ID data is accepted.
    """
    confidences = np.sort(confidences)
    cutoff_index = int(len(confidences) * (1 - tpr))
    threshold = confidences[cutoff_index]
    return threshold


def classify_outcomes(is_correct_arr, confidences, threshold):
    """
    Vectorized classification of the 4 Outcome Types.
    Returns counts dictionary.
    """
    is_accepted = confidences >= threshold
    is_rejected = ~is_accepted
    is_wrong = ~is_correct_arr

    n_clean_success = np.sum(is_correct_arr & is_accepted)
    n_nuisance_novelty = np.sum(is_correct_arr & is_rejected)
    n_double_failure = np.sum(is_wrong & is_rejected)
    n_contained_misid = np.sum(is_wrong & is_accepted)

    return {
        "Clean_Success": n_clean_success,
        "Nuisance_Novelty": n_nuisance_novelty,
        "Double_Failure": n_double_failure,
        "Contained_Misidentification": n_contained_misid,
        "Total": len(confidences),
    }


# =========================================================
# OSCR / AUOSCR Metrics (COSTARR Protocol)
# =========================================================

def compute_oscr_curve(known_gt, known_pred, known_conf, unknown_conf, num_thresholds=1000):
    """
    Compute OSCR (Open-Set Classification Rate) curve.

    OSCR plots Correct Classification Rate (CCR) vs False Positive Rate (FPR):
    - CCR = correctly classified AND accepted / total knowns
    - FPR = incorrectly accepted unknowns / total unknowns

    Args:
        known_gt: Ground truth labels for known samples (numpy array)
        known_pred: Predictions for known samples (numpy array)
        known_conf: Confidence scores for known samples (numpy array)
        unknown_conf: Confidence scores for unknown samples (numpy array)
        num_thresholds: Number of threshold points to compute

    Returns:
        thresholds: Array of threshold values
        ccr: Correct Classification Rate at each threshold
        fpr: False Positive Rate at each threshold
    """
    known_gt = np.asarray(known_gt)
    known_pred = np.asarray(known_pred)
    known_conf = np.asarray(known_conf)
    unknown_conf = np.asarray(unknown_conf)

    # Get threshold range from combined score distribution
    all_conf = np.concatenate([known_conf, unknown_conf])
    thresholds = np.linspace(all_conf.min(), all_conf.max(), num_thresholds)

    n_known = len(known_gt)
    n_unknown = len(unknown_conf)

    ccr = []
    fpr = []

    # Pre-compute correctness mask
    correct_mask = (known_pred == known_gt)

    for thresh in thresholds:
        # CCR: correctly classified AND accepted (above threshold)
        accepted_known = (known_conf >= thresh)
        ccr_val = (correct_mask & accepted_known).sum() / n_known if n_known > 0 else 0.0
        ccr.append(ccr_val)

        # FPR: unknowns incorrectly accepted (above threshold)
        fpr_val = (unknown_conf >= thresh).sum() / n_unknown if n_unknown > 0 else 0.0
        fpr.append(fpr_val)

    return np.array(thresholds), np.array(ccr), np.array(fpr)


def compute_auoscr(known_gt, known_pred, known_conf, unknown_conf):
    """
    Compute Area Under OSCR Curve using trapezoidal integration.

    AUOSCR measures the trade-off between correctly classifying knowns
    and rejecting unknowns across all thresholds.

    Args:
        known_gt: Ground truth labels for known samples
        known_pred: Predictions for known samples
        known_conf: Confidence scores for known samples
        unknown_conf: Confidence scores for unknown samples

    Returns:
        AUOSCR score in [0, 1] (higher is better)
    """
    _, ccr, fpr = compute_oscr_curve(known_gt, known_pred, known_conf, unknown_conf)

    # Sort by FPR for proper integration (low to high FPR)
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    ccr_sorted = ccr[sorted_idx]

    # Trapezoidal integration
    auoscr = np.trapz(ccr_sorted, fpr_sorted)

    return auoscr


def compute_ood_detection_metrics(known_conf, unknown_conf):
    """
    Compute standard OOD detection metrics: AUROC, FPR@95TPR.

    Convention: Higher confidence = more likely to be known/ID.

    Args:
        known_conf: Confidence scores for known/ID samples
        unknown_conf: Confidence scores for unknown/OOD samples

    Returns:
        Dictionary with AUROC and FPR@95TPR
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    known_conf = np.asarray(known_conf)
    unknown_conf = np.asarray(unknown_conf)

    # Labels: 1 = known (positive), 0 = unknown (negative)
    labels = np.concatenate([np.ones(len(known_conf)), np.zeros(len(unknown_conf))])
    scores = np.concatenate([known_conf, unknown_conf])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR@95TPR
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find FPR where TPR >= 0.95
    idx = np.searchsorted(tpr_arr, 0.95)
    fpr_at_95tpr = fpr_arr[min(idx, len(fpr_arr) - 1)]

    return {
        "AUROC": auroc,
        "FPR@95TPR": fpr_at_95tpr
    }
