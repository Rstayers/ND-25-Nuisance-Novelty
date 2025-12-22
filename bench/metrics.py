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
