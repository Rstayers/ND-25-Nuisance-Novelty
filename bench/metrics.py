# metrics.py
import numpy as np
import torch
import torch
import bench.OSA as osa_mod  # so we can call OSA.py exactly


def compute_oosa_threshold(id_confs, ood_confs, id_correct_mask=None, steps=2000):
    """
    Finds threshold that maximizes OOSA.
    OOSA = (Correctly Classified ID & Accepted + OOD Rejected) / Total

    Speedup: evaluate only at score change-points using sorting + searchsorted,
    avoiding an (n_thresh x n_samples) broadcast.
    """
    id_confs = np.asarray(id_confs)
    ood_confs = np.asarray(ood_confs)

    if id_correct_mask is None:
        id_correct_mask = np.ones(len(id_confs), dtype=bool)
    else:
        id_correct_mask = np.asarray(id_correct_mask, dtype=bool)

    total = len(id_confs) + len(ood_confs)
    if total == 0:
        return 0.0, 0.0

    id_correct_confs = id_confs[id_correct_mask]

    # If no OOD scores, best threshold is very low (accept everything) for max ID accepted.
    if ood_confs.size == 0:
        if id_correct_confs.size == 0:
            return 0.0, 0.0
        t = float(np.min(id_correct_confs)) - np.finfo(np.float64).eps
        score = float(id_correct_confs.size) / total
        return t, score

    # If no correct-ID scores, best is to reject as much OOD as possible (threshold above max OOD).
    if id_correct_confs.size == 0:
        t = float(np.max(ood_confs)) + np.finfo(np.float64).eps
        score = float(ood_confs.size) / total
        return t, score

    id_sorted = np.sort(id_correct_confs)
    ood_sorted = np.sort(ood_confs)

    # OOSA only changes when threshold crosses a value in {id_correct_confs} ∪ {ood_confs}
    thresholds = np.unique(np.concatenate([id_sorted, ood_sorted]))
    eps = np.finfo(np.float64).eps
    thresholds = np.concatenate([[thresholds[0] - eps], thresholds, [thresholds[-1] + eps]])

    # ID accepted among correct: count(id_correct_confs >= t)
    id_first_ge = np.searchsorted(id_sorted, thresholds, side="left")
    id_ge_count = id_sorted.size - id_first_ge

    # OOD rejected: count(ood_confs < t)
    ood_lt_count = np.searchsorted(ood_sorted, thresholds, side="left")

    oosa_scores = (id_ge_count + ood_lt_count) / total
    best_idx = int(np.argmax(oosa_scores))
    return float(thresholds[best_idx]), float(oosa_scores[best_idx])


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
