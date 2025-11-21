# analysis/plots/utils_plot.py

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # still used for heatmaps

# ---------------- Basic I/O helpers ----------------

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def joinp(*xs) -> str:
    return os.path.join(*[str(x) for x in xs])


# ---------------- Styling ----------------

def set_seaborn_style():
    """For heatmaps / diagnostics that still use seaborn."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })


def set_mpl_curve_style():
    """
    Clean Matplotlib style for line plots:
    - neutral grid
    - no top/right spines
    - reasonable font sizes
    """
    plt.style.use("default")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })


# ---------------- Outcome taxonomy (for colors / ordering) ----------------

OUTCOME_ORDER = [
    "Clean_Success",
    "Contained_Misidentification",
    "Double_Failure",
    "Nuisance_Novelty",
]

OUTCOME_COLORS = {
    "Clean_Success": "#00b050",                # green
    "Contained_Misidentification": "#9acd32",  # yellow-green
    "Double_Failure": "#ff9d00",               # orange
    "Nuisance_Novelty": "#ff4d4d",             # red
}
