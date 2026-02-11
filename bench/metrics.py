# metrics.py
import numpy as np
import torch
import torch
import bench.OSA as osa_mod  # so we can call OSA.py exactly




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
def _average_precision(scores: np.ndarray, labels_pos: np.ndarray) -> float:
    """
    Average precision for binary labels where labels_pos=1 indicates positive class.
    Scores: higher means more positive.
    """
    order = np.argsort(-scores)
    y = labels_pos[order].astype(np.int32)

    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # AP = sum over each positive example of (delta recall) * precision
    pos_idx = np.where(y == 1)[0]
    ap = 0.0
    prev_recall = 0.0
    for i in pos_idx:
        ap += float(recall[i] - prev_recall) * float(precision[i])
        prev_recall = float(recall[i])
    return ap


def _auroc(scores: np.ndarray, labels_pos: np.ndarray) -> float:
    """
    AUROC for binary labels where labels_pos=1 indicates positive class.
    Scores: higher means more positive.
    """
    order = np.argsort(-scores)
    y = labels_pos[order].astype(np.int32)

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # add endpoints
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))


def compute_ood_det_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr: float = 0.95
) -> dict:
    """
    Returns:
      AUROC (ID=positive),
      AUPR_IN (ID=positive),
      AUPR_OUT (OOD=positive, using -score),
      FPR@95TPR (threshold chosen on ID scores to accept 95% ID).
    """
    id_scores = np.asarray(id_scores).astype(np.float64)
    ood_scores = np.asarray(ood_scores).astype(np.float64)

    scores = np.concatenate([id_scores, ood_scores], axis=0)
    labels_in = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)], axis=0)  # ID=1

    auroc = _auroc(scores, labels_in)
    aupr_in = _average_precision(scores, labels_in)

    # AUPR_OUT: OOD positive, invert scores so higher => more OOD
    labels_out = 1 - labels_in
    aupr_out = _average_precision(-scores, labels_out)

    # FPR@95TPR: threshold chosen from ID only
    thr = compute_id_threshold(id_scores, tpr=tpr)  # your existing function
    fpr95 = float(np.mean(ood_scores >= thr))

    return {
        "AUROC": auroc,
        "AUPR_IN": aupr_in,
        "AUPR_OUT": aupr_out,
        "FPR@95TPR": fpr95,
    }