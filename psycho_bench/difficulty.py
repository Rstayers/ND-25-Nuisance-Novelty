# psycho_bench/difficulty.py
from __future__ import annotations
from typing import Dict, Tuple

import math


# Simple severity→difficulty mapping; you can later replace this with
# a ResNet50-based psychometric table if desired.
def severity_to_difficulty(severity: float, max_severity: float = 5.0) -> float:
    """
    Map ImageNet-C severity (1..5) to [0,1] difficulty.
    severity=1 → 0.0 (easiest), severity=max_severity → 1.0 (hardest).
    """
    severity = float(severity)
    return max(0.0, min(1.0, (severity - 1.0) / max(1.0, max_severity - 1.0)))


def severity_to_target_exit(
    severity: float,
    n_exits: int,
    max_severity: float = 5.0,
) -> int:
    """
    Map severity to integer exit index in {0, ..., n_exits-1}.
    """
    d = severity_to_difficulty(severity, max_severity=max_severity)
    # 0 ← easiest, n_exits-1 ← hardest
    idx = int(round(d * (n_exits - 1)))
    return max(0, min(n_exits - 1, idx))
