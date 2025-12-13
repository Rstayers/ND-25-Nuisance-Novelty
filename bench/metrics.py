import numpy as np
import torch


def compute_oosa_threshold(id_confs, ood_confs, id_correct_mask=None, steps=2000):
    """
    Finds threshold that maximizes OOSA.
    OOSA = (Correctly Classified ID & Accepted + OOD Rejected) / Total
    """
    if id_correct_mask is None:
        id_correct_mask = np.ones(len(id_confs), dtype=bool)

    # Candidate thresholds
    all_scores = np.concatenate([id_confs, ood_confs])

    thresholds = np.unique(all_scores)

    t = thresholds[:, None]

    # ID Component: Must be (Correct Class) AND (Confidence >= Threshold)
    # Shape: [n_thresh, n_id]
    id_accepted = (id_confs[None, :] >= t)
    # Mask out the ones that were wrong class to begin with
    id_success = id_accepted & id_correct_mask[None, :]
    id_count = id_success.sum(axis=1)

    # OOD Component: Must be (Confidence < Threshold)
    ood_rejected = (ood_confs[None, :] < t).sum(axis=1)

    # OOSA
    total = len(id_confs) + len(ood_confs)
    oosa_scores = (id_count + ood_rejected) / total

    best_idx = np.argmax(oosa_scores)
    return thresholds[best_idx], oosa_scores[best_idx]

def compute_id_threshold(confidences, tpr=0.95):
    """
    Determines the threshold where 95% (TPR) of CLEAN ID data is accepted.
    """
    # Sort confidences
    confidences = np.sort(confidences)

    # Index for the cutoff (e.g., bottom 5% are rejected)
    cutoff_index = int(len(confidences) * (1 - tpr))
    threshold = confidences[cutoff_index]

    return threshold


def classify_outcomes(is_correct_arr, confidences, threshold):
    """
    Vectorized classification of the 4 Outcome Types.

    Returns counts dictionary.
    """
    # Boolean masks
    is_accepted = (confidences >= threshold)
    is_rejected = ~is_accepted
    is_wrong = ~is_correct_arr

    # 1. Clean Success (Correct & Accepted)
    # The system works as intended.
    n_clean_success = np.sum(is_correct_arr & is_accepted)

    # 2. Nuisance Novelty (Correct & Rejected) -> THE GOAL
    # The model got it right, but was rightfully uncertain due to the nuisance.
    n_nuisance_novelty = np.sum(is_correct_arr & is_rejected)

    # 3. Double Failure (Wrong & Rejected) -> Safe Fail
    # The model failed, but the detector caught it. System remains safe.
    n_double_failure = np.sum(is_wrong & is_rejected)

    # 4. Contained Misidentification (Wrong & Accepted) -> DANGER
    # The model failed and was confident about it. Safety hazard.
    n_contained_misid = np.sum(is_wrong & is_accepted)

    return {
        "Clean_Success": n_clean_success,
        "Nuisance_Novelty": n_nuisance_novelty,
        "Double_Failure": n_double_failure,
        "Contained_Misidentification": n_contained_misid,
        "Total": len(confidences)
    }