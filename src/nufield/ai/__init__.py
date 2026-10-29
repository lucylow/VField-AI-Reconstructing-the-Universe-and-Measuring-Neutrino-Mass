"""AI research utilities for reproducible simulation-based cosmology inference.

The interfaces in this namespace improve model training and evaluation; they do not certify any
particular model or dataset as a valid cosmological inference pipeline.
"""

from nufield.ai.active_learning import AcquisitionBatch, AcquisitionRecord, select_candidates
from nufield.ai.conformal import ConformalRegressor
from nufield.ai.contracts import AIExperimentContract, ArrayContract, assert_group_disjoint
from nufield.ai.group_conformal import GroupConformalRegressor
from nufield.ai.metrics import gaussian_crps, regression_metrics
from nufield.ai.normalization import BoxTransform, RobustStandardizer, Standardizer
from nufield.ai.uncertainty import EnsembleUncertainty, decompose_ensemble

__all__ = [
    "AcquisitionBatch",
    "AcquisitionRecord",
    "select_candidates",
    "ConformalRegressor",
    "AIExperimentContract",
    "ArrayContract",
    "assert_group_disjoint",
    "GroupConformalRegressor",
    "regression_metrics",
    "gaussian_crps",
    "BoxTransform",
    "RobustStandardizer",
    "Standardizer",
    "EnsembleUncertainty",
    "decompose_ensemble",
]
