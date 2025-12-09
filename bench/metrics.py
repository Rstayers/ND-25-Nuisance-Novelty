# psycho_bench/metrics_summary.py

from typing import Dict, Any

import numpy as np
from openood.evaluators.metrics import compute_all_metrics


def compute_ood_metrics(
    id_pred: np.ndarray,
    id_conf: np.ndarray,
    id_gt: np.ndarray,
    ood_pred: np.ndarray,
    ood_conf: np.ndarray,
    ood_gt: np.ndarray,
) -> Dict[str, float]:
    """
    Wrapper around OpenOOD's compute_all_metrics that supports both
    5-value and 9-value return formats.
    """
    from openood.evaluators.metrics import compute_all_metrics

    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])

    result = compute_all_metrics(conf, label, pred)

    # Handle both possible output lengths
    if len(result) == 9:
        fpr95, auroc, aupr_in, aupr_out, ccr_4, ccr_3, ccr_2, ccr_1, acc = result
        return {
            "fpr95": float(fpr95),
            "auroc": float(auroc),
            "aupr_in": float(aupr_in),
            "aupr_out": float(aupr_out),
            "ccr_4": float(ccr_4),
            "ccr_3": float(ccr_3),
            "ccr_2": float(ccr_2),
            "ccr_1": float(ccr_1),
            "acc": float(acc),
        }

    elif len(result) == 5:
        fpr95, auroc, aupr_in, aupr_out, acc = result
        return {
            "fpr95": float(fpr95),
            "auroc": float(auroc),
            "aupr_in": float(aupr_in),
            "aupr_out": float(aupr_out),
            "acc": float(acc),
            "ccr_4": np.nan,
            "ccr_3": np.nan,
            "ccr_2": np.nan,
            "ccr_1": np.nan,
        }

    else:
        raise ValueError(f"Unexpected return length from compute_all_metrics: {len(result)}")


def build_metrics_row(
    dataset_name: str,
    backbone: str,
    detector: str,
    metrics: Dict[str, float],
    n_id: int,
    n_ood: int,
    subset: str = "ALL",
) -> Dict[str, Any]:
    """
    Build a summary row similar to OpenOOD's OOD evaluation CSV,
    but augmented with backbone, detector, dataset, subset.
    """
    row: Dict[str, Any] = {
        "backbone": backbone,
        "detector": detector,
        "dataset": dataset_name,
        "subset": subset,
        "n_id": int(n_id),
        "n_ood": int(n_ood),
    }
    row.update(metrics)
    return row
