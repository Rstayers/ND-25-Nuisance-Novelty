# analysis/plots/__init__.py
"""
Plot modules for Psycho Benchmark analysis.
"""

from . import overview
from . import backbone_compare
from . import scale_curves
from . import shift_breakdown
from . import ninco
from . import composition
from . import compare_detectors
from . import nuisance_rates
from . import aggregated
from . import ninco_accuracy
from . import utils_plot

__all__ = [
    "overview",
    "backbone_compare",
    "scale_curves",
    "shift_breakdown",
    "ninco",
    "composition",
    "compare_detectors",
    "nuisance_rates",
    "aggregated",
    "ninco_accuracy",
    "utils_plot",
]
