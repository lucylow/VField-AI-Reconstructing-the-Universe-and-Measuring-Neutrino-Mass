"""Create immutable train/validation/test partitions before fitting preprocessing or models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.data_registry.preprocessing import Standardizer
from nufield.data_registry.splits import (
    group_train_val_test_split,
    split_fingerprint,
    verify_parameter_support,
)


@dataclass
class PreparedData:
    x_train: np.ndarray
    theta_train: np.ndarray
    x_validation: np.ndarray
    theta_validation: np.ndarray
    x_test: np.ndarray
    theta_test: np.ndarray
    indices: dict[str, np.ndarray]
    group_ids: np.ndarray
    simulation_ids: np.ndarray
    preprocessing_state: dict
    split_report: dict


def prepare_grouped_data(
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    simulation_ids: np.ndarray,
    group_ids: np.ndarray,
    parameter_names: list[str],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    normalize_observations: bool = True,
    mnu_parameter_name: str = "Mnu",
) -> PreparedData:
    """Split by independent parent groups, fit x preprocessing on training rows only, audit support.

    `simulation_ids` may identify individual cubes while `group_ids` identify parent phases, box
    families, mock light cones, or cosmologies that must not cross split boundaries. Select group
    semantics from the data-generation design; unique row IDs are insufficient when multiple crops
    or projections derive from one simulation.
    """
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(theta, dtype=np.float32)
    ids = np.asarray(simulation_ids).astype(str)
    groups = np.asarray(group_ids).astype(str)
    if x.ndim < 2 or y.ndim != 2:
        raise ValueError("observations must be [N,...] and theta [N,P]")
    n = len(y)
    if len(x) != n or len(ids) != n or len(groups) != n:
        raise ValueError("all sample axes must align")
    if y.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("all arrays must be finite")
    split = group_train_val_test_split(groups, fractions=fractions, seed=seed)
    prep = None
    if normalize_observations:
        # Standardizer currently computes a separate mean/scale for each packed-feature index.
        prep = Standardizer().fit(x[split["train"]])
        x = prep.transform(x)
    support = None
    if mnu_parameter_name in parameter_names:
        support = verify_parameter_support(
            y,
            train_indices=split["train"],
            test_indices=split["test"],
            parameter_names=parameter_names,
            name=mnu_parameter_name,
        )
    report = {
        "seed": seed,
        "fractions_requested": list(fractions),
        "n_rows": {k: int(len(v)) for k, v in split.items()},
        "n_groups": {k: int(len(set(groups[v].tolist()))) for k, v in split.items()},
        "split_fingerprints": {k: split_fingerprint(ids, v) for k, v in split.items()},
        "support_audit": support,
        "preprocessing_fit_on": "train only" if prep else "none",
        "warning": "Group split is only as good as group_ids. Validate whether cosmology/phase/mock families must also be held out.",
    }
    return PreparedData(
        x_train=x[split["train"]],
        theta_train=y[split["train"]],
        x_validation=x[split["validation"]],
        theta_validation=y[split["validation"]],
        x_test=x[split["test"]],
        theta_test=y[split["test"]],
        indices=split,
        group_ids=groups,
        simulation_ids=ids,
        preprocessing_state=prep.state_dict() if prep else {},
        split_report=report,
    )
