"""End-to-end grouped-split NPE experiment from an explicitly mapped NPZ dataset.

The workflow intentionally refuses a dataset marked `physics_level: toy` unless the caller opts
in. That flag only authorizes a software smoke test, never a physical mass result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.benchmarking.posterior_metrics import posterior_sample_metrics
from nufield.data_registry.preprocessing import ChannelStandardizer, Standardizer
from nufield.data_registry.splits import stratified_group_train_val_test_split
from nufield.models.hybrid_encoder import pack_observations
from nufield.models.observation_adapters import build_embedding
from nufield.models.sbi_npe import NPEConfig, fit_npe, sample_posterior


def _load_npz(
    path: Path,
    *,
    field_key: str,
    spectrum_key: str,
    theta_key: str,
    simulation_id_key: str,
    group_id_key: str | None,
    lensing_key: str | None,
    allow_synthetic_ids: bool,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        for key in (field_key, spectrum_key, theta_key):
            if key not in keys:
                raise KeyError(f"{key!r} not found; dataset keys={sorted(keys)}")
        fields = np.asarray(z[field_key], dtype=np.float32)
        spectrum = np.asarray(z[spectrum_key], dtype=np.float32)
        theta = np.asarray(z[theta_key], dtype=np.float32)
        if simulation_id_key in keys:
            ids = np.asarray(z[simulation_id_key]).astype(str)
        elif allow_synthetic_ids:
            ids = np.asarray([f"row-{i:08d}" for i in range(len(theta))])
        else:
            raise KeyError(f"required simulation ID key {simulation_id_key!r} missing")
        if group_id_key and group_id_key in keys:
            groups = np.asarray(z[group_id_key]).astype(str)
        elif group_id_key:
            raise KeyError(f"required group ID key {group_id_key!r} missing")
        else:
            groups = ids.copy()
        lensing = (
            np.asarray(z[lensing_key], dtype=np.float32)
            if lensing_key and lensing_key in keys
            else None
        )
        if lensing_key and lensing_key not in keys:
            raise KeyError(f"lensing key {lensing_key!r} not found")
    n = len(theta)
    if fields.shape[0] != n or spectrum.shape[0] != n or len(ids) != n or len(groups) != n:
        raise ValueError("row count mismatch across fields/spectrum/theta/IDs")
    if lensing is not None and len(lensing) != n:
        raise ValueError("lensing row count mismatch")
    return {
        "fields": fields,
        "spectrum": spectrum,
        "theta": theta,
        "ids": ids,
        "groups": groups,
        "lensing": lensing,
    }


def assert_unique_parent_groups_for_sbi_training(
    group_ids: np.ndarray, train_indices: np.ndarray
) -> None:
    """Refuse row-level SBI early-stopping splits when a parent has multiple training views."""
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    idx = np.asarray(train_indices, dtype=int).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= len(groups)):
        raise ValueError("train indices are out of bounds")
    selected = groups[idx]
    unique, counts = np.unique(selected, return_counts=True)
    repeated = unique[counts > 1]
    if len(repeated):
        raise ValueError(
            "SBI's built-in early-stopping split is row-based; training partition contains repeated parent groups "
            f"(examples: {repeated[:8].tolist()}). Aggregate/select one view per parent or implement a custom group-aware SBI trainer."
        )


def train_npe_from_npz(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    field_key: str,
    spectrum_key: str,
    theta_key: str,
    parameter_names: list[str],
    prior_low: list[float],
    prior_high: list[float],
    simulation_id_key: str = "simulation_id",
    group_id_key: str | None = None,
    lensing_key: str | None = None,
    lensing_shape: tuple[int, int] = (16, 16),
    allow_synthetic_ids: bool = False,
    allow_toy: bool = False,
    seed: int = 42,
    train_fraction: float = 0.75,
    config: NPEConfig | None = None,
    n_eval: int = 10,
    posterior_samples: int = 1000,
    architecture: str = "hybrid",
    periodic_box: bool = False,
    fno_modes: tuple[int, int, int] = (6, 6, 6),
    evaluate_test_once: bool = False,
) -> dict:
    """Fit NPE on training groups and score the separate validation partition.

    SBI uses an internal row-level validation split for early stopping. Therefore this function
    requires at most one row per parent group *within the training partition*, so that the internal
    split cannot place two views from one simulation on both sides. The external validation groups
    are used for model selection. The final test partition is not touched unless explicitly opted in
    with ``evaluate_test_once=True`` after the architecture/configuration is frozen.
    """
    dataset_path = Path(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = dataset_path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    physics_level = str(metadata.get("physics_level", "unknown"))
    if physics_level == "toy" and not allow_toy:
        raise ValueError(
            "dataset metadata marks physics_level='toy'; pass --allow-toy only for a software smoke test. Never interpret these results physically."
        )
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    if n_eval < 1 or posterior_samples < 2:
        raise ValueError("n_eval>=1 and posterior_samples>=2 required")
    arrays = _load_npz(
        dataset_path,
        field_key=field_key,
        spectrum_key=spectrum_key,
        theta_key=theta_key,
        simulation_id_key=simulation_id_key,
        group_id_key=group_id_key,
        lensing_key=lensing_key,
        allow_synthetic_ids=allow_synthetic_ids,
    )
    theta = arrays["theta"]
    if theta.ndim != 2 or theta.shape[1] != len(parameter_names):
        raise ValueError("theta dimensions do not match parameter_names")
    if len(set(arrays["ids"].tolist())) != len(arrays["ids"]):
        # Duplicate realization IDs may be legitimate views but then explicit parent groups are mandatory.
        if group_id_key is None:
            raise ValueError(
                "duplicate simulation IDs require explicit group_id_key to prevent split leakage"
            )
    architecture = str(architecture).lower()
    if architecture not in {"hybrid", "spectrum_mlp", "cnn3d", "fno3d"}:
        raise ValueError("architecture must be one of hybrid, spectrum_mlp, cnn3d, fno3d")
    if architecture == "fno3d" and not periodic_box:
        raise ValueError(
            "fno3d requires --periodic-box because this encoder assumes a regular periodic simulation volume"
        )
    fields = np.asarray(arrays["fields"], dtype=np.float32)
    if fields.ndim == 4:
        fields = fields[:, None, ...]
    if fields.ndim != 5:
        raise ValueError("field data must be [N,D,H,W] or [N,C,D,H,W]")
    field_shape = tuple(int(d) for d in fields.shape[1:])
    spectrum = arrays["spectrum"]
    lensing = arrays["lensing"]
    if lensing is None:
        lens_shape = (1, *tuple(lensing_shape))
    else:
        if lensing.ndim == 3:
            lensing = lensing[:, None, ...]
        if lensing.ndim != 4:
            raise ValueError("lensing must be [N,H,W] or [N,C,H,W]")
        lens_shape = tuple(int(d) for d in lensing.shape[1:])
    groups = arrays["groups"]
    # Only independent simulation groups can be assigned to train/test. Keep validation inside the
    # training subset for the SBI API's early stopping unless a custom group-aware trainer is used.
    fraction_test = 1.0 - train_fraction
    mass_idx = (
        parameter_names.index("sum_mnu_ev")
        if "sum_mnu_ev" in parameter_names
        else (parameter_names.index("Mnu") if "Mnu" in parameter_names else 0)
    )
    split = stratified_group_train_val_test_split(
        groups,
        theta[:, mass_idx],
        fractions=(train_fraction - 0.10, 0.10, fraction_test),
        seed=seed,
        n_bins=6,
        require_group_constant=False,
    )
    train_idx = split["train"]
    val_idx = split["validation"]
    test_idx = split["test"]
    # sbi internally partitions rows for early stopping. Refuse multiple views from one parent in
    # the fitting partition because that random split would leak group-level information.
    assert_unique_parent_groups_for_sbi_training(groups, train_idx)
    fit_idx = train_idx
    norm_kind = "featurewise"
    if architecture == "hybrid":
        if lensing is None:
            packed = pack_observations(
                fields, spectrum, lensing=None, lensing_shape=tuple(lensing_shape)
            )
        else:
            packed = pack_observations(fields, spectrum, lensing=lensing)
        raw_observations = packed.values
        scaler = Standardizer().fit(raw_observations[train_idx])
        x_all = scaler.transform(raw_observations)
        observation_metadata = packed.metadata()
    elif architecture == "spectrum_mlp":
        raw_observations = np.asarray(spectrum, dtype=np.float32)
        scaler = Standardizer().fit(raw_observations[train_idx])
        x_all = scaler.transform(raw_observations)
        observation_metadata = {
            "observation_kind": "spectrum",
            "spectrum_dim": int(spectrum.shape[1]),
            "packed_width": int(spectrum.shape[1]),
            "field_shape": list(field_shape),
            "lensing_shape": None,
        }
    else:
        norm_kind = "channel"
        scaler = ChannelStandardizer().fit(fields[train_idx])
        standardized_fields = scaler.transform(fields)
        raw_observations = standardized_fields.reshape(len(fields), -1)
        x_all = raw_observations.astype(np.float32)
        observation_metadata = {
            "observation_kind": "field",
            "field_shape": list(field_shape),
            "spectrum_dim": int(spectrum.shape[1]),
            "packed_width": int(x_all.shape[1]),
            "lensing_shape": None,
            "boundary_condition": "periodic"
            if architecture == "fno3d"
            else "convolutional; edge sensitivity requires testing",
        }
    low = np.asarray(prior_low, dtype=np.float32)
    high = np.asarray(prior_high, dtype=np.float32)
    if low.shape != (theta.shape[1],) or high.shape != (theta.shape[1],):
        raise ValueError("prior bounds must match theta columns")
    if (theta[fit_idx] < low).any() or (theta[fit_idx] > high).any():
        raise ValueError("training parameters outside declared prior bounds")
    model, embedding_spec = build_embedding(
        architecture,
        field_shape=field_shape,
        spectrum_dim=int(spectrum.shape[1]),
        lensing_shape=lens_shape,
        input_width=int(x_all.shape[1]),
        output_dim=128,
        width=12,
        dropout=0.05,
        modes=fno_modes,
    )
    npe_cfg = config or NPEConfig(seed=seed)
    # Use train + validation in the inference API; test groups are never passed to fit_npe.
    result = fit_npe(
        x_all[fit_idx],
        theta[fit_idx],
        prior_low=low,
        prior_high=high,
        parameter_names=parameter_names,
        embedding_net=model,
        config=npe_cfg,
    )
    rng = np.random.default_rng(seed + 9000)
    val_eval_idx = val_idx[: min(n_eval, len(val_idx))]
    val_samples = []
    for i in val_eval_idx:
        val_samples.append(
            sample_posterior(
                result, x_all[i], n_samples=posterior_samples, seed=int(rng.integers(0, 2**31 - 1))
            )
        )
    val_samples_arr = np.asarray(val_samples)
    val_metrics = posterior_sample_metrics(val_samples_arr, theta[val_eval_idx])
    test_eval_idx = np.asarray([], dtype=int)
    test_samples_arr = None
    test_metrics = None
    if evaluate_test_once:
        test_eval_idx = test_idx[: min(n_eval, len(test_idx))]
        test_samples = []
        for i in test_eval_idx:
            test_samples.append(
                sample_posterior(
                    result,
                    x_all[i],
                    n_samples=posterior_samples,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
            )
        test_samples_arr = np.asarray(test_samples)
        test_metrics = posterior_sample_metrics(test_samples_arr, theta[test_eval_idx])
    report = {
        "experiment_id": f"nufield-npe-{seed}",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_metadata_path": str(meta_path.resolve()) if meta_path.exists() else None,
        "physics_level": physics_level,
        "scientific_status": "TOY_SOFTWARE_SMOKE_TEST_ONLY"
        if physics_level == "toy"
        else "UNVALIDATED_NEURAL_INFERENCE",
        "n_total": int(len(theta)),
        "n_fit": int(len(fit_idx)),
        "n_validation_groups": int(len(val_idx)),
        "n_test_groups": int(len(test_idx)),
        "n_validation_evaluated": int(len(val_eval_idx)),
        "n_test_evaluated": int(len(test_eval_idx)),
        "final_test_evaluated": bool(evaluate_test_once),
        "posterior_samples_per_case": int(posterior_samples),
        "parameter_names": parameter_names,
        "prior_low": low.tolist(),
        "prior_high": high.tolist(),
        "architecture": architecture,
        "embedding_spec": embedding_spec.to_dict(),
        "observation_pack": observation_metadata,
        "normalization": f"{norm_kind}, fit only on training groups; validation/test transformed with frozen train statistics",
        "split": {
            key: {
                "rows": int(len(idx)),
                "groups": int(len(set(groups[idx].tolist()))),
                "mnu_min": float(theta[idx, mass_idx].min()),
                "mnu_max": float(theta[idx, mass_idx].max()),
                "mnu_unique": int(len(np.unique(theta[idx, mass_idx]))),
            }
            for key, idx in split.items()
        },
        "fit_config": result.training_summary,
        "validation_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "warnings": [
            "NPE samples are approximate and must be calibrated on independent simulations.",
            "The default hybrid CNN is a candidate architecture, not a proven cosmological posterior estimator.",
            "Physics-level 'toy' results have no scientific interpretation.",
            "The external validation groups are separate from training and should drive model selection.",
            "The final test set is not evaluated unless --evaluate-test-once is set after freezing the model/config.",
            "SBI's internal early-stopping split is row-level; multiple parent views in the training subset are rejected.",
        ],
    }
    (out / "npe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        import importlib.metadata as importlib_metadata

        sbi_version = importlib_metadata.version("sbi")
    except Exception:
        sbi_version = "unknown"
    model_config = {
        "schema_version": "1.0",
        "experiment_id": report["experiment_id"],
        "physics_level": physics_level,
        "scientific_status": report["scientific_status"],
        "field_key": field_key,
        "spectrum_key": spectrum_key,
        "theta_key": theta_key,
        "simulation_id_key": simulation_id_key,
        "group_id_key": group_id_key,
        "lensing_key": lensing_key,
        "parameter_names": parameter_names,
        "prior_low": low.tolist(),
        "prior_high": high.tolist(),
        "architecture": architecture,
        "embedding_spec": embedding_spec.to_dict(),
        "preprocessing_kind": norm_kind,
        "observation_pack": observation_metadata,
        "npe_config": result.training_summary,
        "sbi_version": sbi_version,
        "serialization_note": "posterior.pt and inference.pt use sbi save format; only load trusted files and match sbi/PyTorch versions.",
    }
    (out / "training_config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    # sbi's public save method stores the trained conditional estimator and prior. These are
    # serialized Python objects: do not load artifacts from untrusted sources.
    result.posterior.save(str(out / "posterior.pt"))
    result.inference.save(str(out / "inference.pt"))
    np.savez_compressed(
        out / "validation_posterior_samples.npz",
        samples=val_samples_arr,
        truth=theta[val_eval_idx],
        simulation_id=arrays["ids"][val_eval_idx],
        group_id=groups[val_eval_idx],
    )
    if evaluate_test_once and test_samples_arr is not None:
        np.savez_compressed(
            out / "heldout_posterior_samples.npz",
            samples=test_samples_arr,
            truth=theta[test_eval_idx],
            simulation_id=arrays["ids"][test_eval_idx],
            group_id=groups[test_eval_idx],
        )
    np.savez_compressed(
        out / "preprocessing_state.npz",
        mean=np.asarray(scaler.mean_),
        scale=np.asarray(scaler.scale_),
        kind=np.asarray(norm_kind),
    )
    # Also save the embedding-only checkpoint for architecture diagnostics; posterior.pt is the full inference artifact.
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_spec": embedding_spec.to_dict(),
            "architecture": architecture,
            "field_shape": field_shape,
            "spectrum_dim": int(spectrum.shape[1]),
            "lensing_shape": lens_shape,
            "output_dim": 128,
            "width": 12,
            "modes": fno_modes,
        },
        out / "embedding_state.pt",
    )
    return report
