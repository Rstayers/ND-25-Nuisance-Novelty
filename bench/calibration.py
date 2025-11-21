# psycho_bench/calibration.py

from typing import Dict, List

import numpy as np
from sklearn import metrics


def calibrate_thresholds_openood_style(
    id_conf: np.ndarray,
    ood_conf: np.ndarray,
    fprs: List[float],
) -> Dict[float, float]:
    """
    Calibrate thresholds using the same ROC convention as OpenOOD:

      - Concatenate ID and OOD scores.
      - OOD is treated as "positive" (label -1 internally).
      - Use -conf as the ROC score (ID samples have larger raw conf).

    For each FPR target, pick the smallest threshold on -conf whose FPR
    (over ID samples) is >= target. We then convert back to a threshold
    on the original conf (ID score).
    """
    # Concatenate scores
    conf = np.concatenate([id_conf, ood_conf])
    # ID = 0, OOD = -1 (mirroring OpenOOD's ID vs OOD convention)
    label = np.concatenate([
        np.zeros_like(id_conf, dtype=int),
        -np.ones_like(ood_conf, dtype=int),
    ])

    # OOD = 1, ID = 0
    ood_indicator = (label == -1).astype(int)

    # ROC w.r.t. -conf, with OOD treated as positive
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)

    tau_map: Dict[float, float] = {}
    for f in fprs:
        if (fpr_list >= f).any():
            idx = int(np.argmax(fpr_list >= f))
        else:
            # If we never reach that FPR, take the last point on the ROC curve
            idx = len(fpr_list) - 1

        # thresholds are on -conf; convert back to conf
        tau_conf = -thresholds[idx]
        tau_map[float(f)] = float(tau_conf)

    return tau_map
