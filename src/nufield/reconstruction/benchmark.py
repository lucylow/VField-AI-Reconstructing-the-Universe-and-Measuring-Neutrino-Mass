"""Group-safe, auditable field reconstruction benchmark for paired NPZ arrays.

Expected target/galaxy arrays have shape (N,X,Y,Z). Optional lensing maps have shape
(N,X,Y) and are broadcast as a documented line-of-sight-constant channel; optional CMB
features have shape (N,F) and are global conditioning channels. This is a prototype API,
not a validated survey likelihood or scientific neutrino-mass analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from nufield.data_registry.splits import assert_group_disjoint, group_train_val_test_split
from nufield.provenance import build_manifest, sha256_file, write_json
from nufield.reconstruction.config import ReconstructionConfig


def _text_ids(values: np.ndarray, name: str, n: int) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must be a one-dimensional array with {n} rows")
    if a.dtype.kind not in "US":
        raise ValueError(f"{name} must contain non-object string values")
    out = a.astype(str)
    if np.any(np.char.str_len(out) == 0):
        raise ValueError(f"{name} must not contain empty IDs")
    return out


def load_reconstruction_dataset(config: ReconstructionConfig) -> dict[str, Any]:
    """Load arrays without pickle and enforce alignment, geometry, mask, and finiteness."""
    path = Path(config.dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    for key in (config.target_key, config.galaxy_key, config.simulation_id_key):
        if key not in arrays:
            raise ValueError(f"Dataset missing required array {key!r}; available={sorted(arrays)}")
    y = np.asarray(arrays[config.target_key], dtype=np.float32)
    galaxy = np.asarray(arrays[config.galaxy_key], dtype=np.float32)
    if y.ndim != 4 or y.shape[1] != y.shape[2] or y.shape[2] != y.shape[3]:
        raise ValueError("target must have shape (N,X,Y,Z) on a cubic grid")
    if galaxy.shape != y.shape:
        raise ValueError(f"galaxy map {config.galaxy_key!r} must match target shape {y.shape}")
    n = len(y)
    if n < 6:
        raise ValueError("at least six paired realizations are required for a group-disjoint split")
    ids = _text_ids(arrays[config.simulation_id_key], config.simulation_id_key, n)
    if len(np.unique(ids)) != n:
        raise ValueError("simulation IDs must be unique; use the true independent group key")
    if config.group_key in arrays:
        groups = _text_ids(arrays[config.group_key], config.group_key, n)
        group_source = config.group_key
    else:
        groups, group_source = ids.copy(), config.simulation_id_key
    modalities: dict[str, np.ndarray] = {"galaxy": galaxy[:, None]}
    if config.lensing_key in arrays:
        lens = np.asarray(arrays[config.lensing_key], dtype=np.float32)
        if lens.shape != (n, y.shape[1], y.shape[2]):
            raise ValueError(f"{config.lensing_key} must have shape (N,X,Y)")
        modalities["lensing"] = np.broadcast_to(lens[:, None, :, :, None], (n, 1, *y.shape[1:])).copy()
    if config.cmb_key in arrays:
        cmb = np.asarray(arrays[config.cmb_key], dtype=np.float32)
        if cmb.ndim == 1:
            cmb = cmb[:, None]
        if cmb.ndim != 2 or len(cmb) != n or cmb.shape[1] < 1:
            raise ValueError(f"{config.cmb_key} must have shape (N,F), F>=1")
        modalities["cmb"] = np.broadcast_to(
            cmb[:, :, None, None, None], (n, cmb.shape[1], *y.shape[1:])
        ).copy()
    mask = np.ones_like(y, dtype=bool)
    if config.survey_mask_key in arrays:
        raw = np.asarray(arrays[config.survey_mask_key])
        if raw.shape == y.shape[1:]:
            raw = np.broadcast_to(raw, y.shape)
        if raw.shape != y.shape or not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
            raise ValueError("survey mask must be finite, binary, and shaped (X,Y,Z) or (N,X,Y,Z)")
        mask = raw.astype(bool)
        if np.any(mask.reshape(n, -1).sum(axis=1) == 0):
            raise ValueError("each survey mask must include at least one valid voxel")
    for name, a in {"target": y, **modalities}.items():
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains NaN or infinity; clean or explicitly mask missing data")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing provenance sidecar {metadata_path}; declare units, coordinates, data type, and license/terms")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_meta = ("schema_version", "data_type", "is_synthetic", "license", "array_units", "coordinate_system", "coordinate_unit")
    absent = [key for key in required_meta if key not in metadata]
    if absent:
        raise ValueError(f"Provenance sidecar missing required field(s): {absent}")
    if not isinstance(metadata["is_synthetic"], bool) or not isinstance(metadata["array_units"], dict):
        raise ValueError("provenance is_synthetic must be boolean and array_units must be a mapping")
    unit_keys = [config.target_key, config.galaxy_key]
    if config.lensing_key in arrays:
        unit_keys.append(config.lensing_key)
    if config.cmb_key in arrays:
        unit_keys.append(config.cmb_key)
    for key in unit_keys:
        if not isinstance(metadata["array_units"].get(key), str) or not metadata["array_units"][key].strip():
            raise ValueError(f"provenance must declare a non-empty unit for array {key!r}")
    for key in ("coordinate_system", "coordinate_unit", "license", "data_type"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"provenance field {key!r} must be a non-empty string")
    return {
        "target": y,
        "modalities": modalities,
        "simulation_ids": ids,
        "groups": groups,
        "group_source": group_source,
        "mask": mask,
        "metadata": metadata,
        "dataset_sha256": sha256_file(path),
        "dataset_path": str(path.resolve()),
    }


def _standardize_modalities(modalities: dict[str, np.ndarray], train: np.ndarray) -> tuple[dict, dict]:
    """Fit channel statistics on training rows only; constant channels use unit scale."""
    normalized, state = {}, {}
    for name, values in modalities.items():
        mean = values[train].mean(axis=(0, 2, 3, 4), keepdims=True)
        std = values[train].std(axis=(0, 2, 3, 4), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        normalized[name] = (values - mean) / std
        state[name] = {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}
    return normalized, state


def _fit_linear(channels: np.ndarray, target: np.ndarray, mask: np.ndarray, train: np.ndarray, alpha: float):
    x = np.moveaxis(channels[train], 1, -1).reshape(-1, channels.shape[1]).astype(np.float64)
    y = target[train].reshape(-1).astype(np.float64)
    m = mask[train].reshape(-1)
    x, y = x[m], y[m]
    if len(y) < channels.shape[1] + 2:
        raise ValueError("too few unmasked training voxels for the linear baseline")
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(y) + alpha * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, xc.T @ yc / len(y))
    intercept = ym - float(xm @ coef)
    return coef.astype(np.float32), intercept


def _predict_linear(channels: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return (np.einsum("ncxyz,c->nxyz", channels, coef) + intercept).astype(np.float32)


def _power_distance(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    if valid is None:
        valid = np.ones_like(a, dtype=bool)
    if valid.shape != a.shape or not np.any(valid):
        raise ValueError("power comparison requires a non-empty shape-matched validity mask")
    # Zero-fill only after centering on observed voxels. This is a windowed pseudo-spectrum,
    # not a deconvolved estimate; masked-out truth/prediction values cannot affect the score.
    ac = np.where(valid, a - a[valid].mean(), 0.0)
    bc = np.where(valid, b - b[valid].mean(), 0.0)
    fa, fb = np.abs(np.fft.rfftn(ac)) ** 2, np.abs(np.fft.rfftn(bc)) ** 2
    kx, ky, kz = np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), np.fft.rfftfreq(a.shape[2])
    radius = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    bins = np.linspace(0, float(radius.max()) + 1e-9, 9)
    pa, pb = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (radius >= lo) & (radius < hi)
        if sel.any():
            pa.append(float(fa[sel].mean()))
            pb.append(float(fb[sel].mean()))
    if not pa:
        return 0.0
    pa, pb = np.asarray(pa), np.asarray(pb)
    active = (pa + pb) > 1e-12
    return float(np.mean(np.abs(pa[active] - pb[active]) / (pa[active] + pb[active] + 1e-12))) if active.any() else 0.0


def _scores(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    rows = []
    for p, t, m in zip(pred, target, mask, strict=True):
        r = (p - t)[m]
        rows.append((float(np.sqrt(np.mean(r**2))), float(np.mean(r)), _power_distance(p, t, m), float(np.corrcoef(p[m], t[m])[0, 1]) if np.std(p[m]) > 0 and np.std(t[m]) > 0 else 0.0))
    a = np.asarray(rows)
    return {"rmse": float(a[:, 0].mean()), "bias": float(a[:, 1].mean()), "power_spectrum_relative_error": float(a[:, 2].mean()), "cross_correlation": float(a[:, 3].mean())}


def run_reconstruction_benchmark(config: ReconstructionConfig) -> dict[str, Any]:
    """Run train-only-normalized group-disjoint classical benchmarks and optional U-Net."""
    data = load_reconstruction_dataset(config)
    if config.run_neural and any((config.lambda_pk, config.lambda_gradient, config.lambda_mean)):
        if data["metadata"].get("periodic") is not True:
            raise ValueError("periodic physics losses require sidecar periodic: true")
        if config.lambda_pk > 0 and not math.isclose(
            float(data["metadata"].get("box_size_mpc_h", float("nan"))),
            float(config.box_size_mpc_h), rel_tol=1e-8, abs_tol=0.0,
        ):
            raise ValueError("configured box_size_mpc_h must match the documented sidecar box size")
    split = group_train_val_test_split(
        data["groups"],
        fractions=(config.train_fraction, config.validation_fraction, config.test_fraction),
        seed=config.seed,
    )
    assert_group_disjoint(data["groups"], split)
    train, validation, test = (split[k] for k in ("train", "validation", "test"))
    normalized, norm_state = _standardize_modalities(data["modalities"], train)
    target, mask = data["target"], data["mask"]
    names = list(normalized)
    combos = [["galaxy"]]
    for name in names:
        if name != "galaxy":
            combos.append(["galaxy", name])
    if len(names) > 2:
        combos.append(names)
    # Keep each modality subset unique while retaining a galaxy-only control.
    unique_combos = []
    for c in combos:
        if c not in unique_combos:
            unique_combos.append(c)
    results, predictions = {}, {}
    fit_start = time.perf_counter()
    for combo in unique_combos:
        channels = np.concatenate([normalized[k] for k in combo], axis=1)
        t0 = time.perf_counter()
        coef, intercept = _fit_linear(channels, target, mask, train, config.ridge_alpha)
        pred = _predict_linear(channels[test], coef, intercept)
        seconds = time.perf_counter() - t0
        predictions["linear_" + "+".join(combo)] = pred
        results["linear_" + "+".join(combo)] = {
            "model": "global ridge-regression baseline",
            "modalities": combo,
            "n_coefficients": int(len(coef)),
            "fit_and_test_seconds": seconds,
            "metrics": _scores(pred, target[test], mask[test]),
            "noise_stress": {},
        }
        for noise_level in (0.05, 0.20):
            rng = np.random.default_rng(config.seed + int(noise_level * 1000))
            noisy = channels[test] + rng.normal(0, noise_level, channels[test].shape).astype(np.float32)
            noisy_pred = _predict_linear(noisy, coef, intercept)
            results["linear_" + "+".join(combo)]["noise_stress"][str(noise_level)] = _scores(noisy_pred, target[test], mask[test])["rmse"]
        # Small smooth multiplicative systematics probe; magnitude is a stress choice, not survey truth.
        ramp = np.linspace(-0.05, 0.05, target.shape[1], dtype=np.float32)[None, None, None, :]
        stressed = channels[test].copy()
        stressed[:, 0] *= 1 + ramp
        results["linear_" + "+".join(combo)]["gradient_systematic_rmse_5pct"] = _scores(
            _predict_linear(stressed, coef, intercept), target[test], mask[test]
        )["rmse"]
        # Train residual RMS provides a simple explicitly non-posterior predictive interval check.
        train_pred = _predict_linear(channels[train], coef, intercept)
        residual = (train_pred - target[train])[mask[train]]
        sigma = float(np.sqrt(np.mean(residual**2)))
        test_resid = np.abs((pred - target[test])[mask[test]])
        results["linear_" + "+".join(combo)]["residual_scale_diagnostic"] = {
            "training_rms": sigma,
            "test_fraction_within_1_training_rms": float(np.mean(test_resid <= sigma)),
            "test_fraction_within_1_96_training_rms": float(np.mean(test_resid <= 1.96 * sigma)),
            "warning": "Residual-scale diagnostic only; not a calibrated Bayesian posterior or interval coverage claim.",
        }
    neural = {"status": "not_requested"}
    if config.run_neural:
        try:
            import torch
        except ImportError:
            neural = {"status": "skipped", "reason": "PyTorch is not installed; install nufield-starter[ml]."}
        else:
            if not np.all(mask):
                neural = {"status": "skipped", "reason": "Current U-Net trainer has no masked-loss implementation; use classical mask-aware baseline."}
            else:
                from nufield.reconstruction.train import train_unet3d
                torch.use_deterministic_algorithms(True, warn_only=True)
                neural_start = time.perf_counter()
                out = Path(config.output_dir)
                out.mkdir(parents=True, exist_ok=True)
                checkpoint = out / "unet3d_galaxy.pt"
                train_groups = sorted(set(data["groups"][train]))
                provenance = {"dataset_sha256": data["dataset_sha256"], "train_group_ids": train_groups,
                              "data_type": data["metadata"]["data_type"],
                              "is_synthetic": data["metadata"]["is_synthetic"],
                              "license": data["metadata"]["license"]}
                trained = train_unet3d(
                    normalized["galaxy"][:, 0][train], target[train],
                    epochs=config.neural_epochs, batch_size=config.batch_size,
                    learning_rate=config.learning_rate, seed=config.seed, device=config.device,
                    optimizer_name=config.optimizer, scheduler_name=config.scheduler,
                    lambda_field=config.lambda_field, lambda_pk=config.lambda_pk,
                    lambda_gradient=config.lambda_gradient, lambda_mean=config.lambda_mean,
                    box_size_mpc_h=config.box_size_mpc_h,
                    checkpoint_path=checkpoint, resume_from=config.resume_from, provenance=provenance,
                )
                model = trained["model"]
                model.eval()
                with torch.no_grad():
                    tensor = torch.from_numpy(normalized["galaxy"][test]).to(config.device)
                    neural_pred = model(tensor).detach().cpu().numpy()[:, 0]
                neural_seconds = time.perf_counter() - neural_start
                key = "unet3d_galaxy"
                predictions[key] = neural_pred
                results[key] = {"model": "optional UNet3D", "modalities": ["galaxy"], "fit_and_test_seconds": neural_seconds, "training_history": trained["history"], "metrics": _scores(neural_pred, target[test], mask[test]), "noise_stress": {}}
                neural["status"] = "completed"
                neural["checkpoint"] = str(checkpoint)
                neural["resumed_from"] = config.resume_from
    fit_seconds = time.perf_counter() - fit_start
    # Report/CSV are explicit metrics on held-out grouped simulations only.
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    dataset_version = data["metadata"].get("dataset_version", data["metadata"].get("release", data["metadata"]["schema_version"]))
    peak_rss = _peak_rss_bytes()
    artifact_paths = {"metrics_csv": str((out / "metrics.csv").resolve()),
                      "predictions_npz": str((out / "predictions.npz").resolve()),
                      "report_json": str((out / "reconstruction_report.json").resolve()),
                      "config_json": str((out / "run_config.json").resolve()),
                      "manifest_json": str((out / "reconstruction_manifest.json").resolve())}
    rows = []
    for name, result in results.items():
        rows.append({"model": name, **result["metrics"], "runtime_seconds": result["fit_and_test_seconds"],
                     "peak_rss_bytes_process_lifetime": peak_rss, "seed": config.seed,
                     "config_sha256": config_hash, "dataset_sha256": data["dataset_sha256"],
                     "dataset_version": dataset_version, "data_type": data["metadata"]["data_type"],
                     "metrics_csv": artifact_paths["metrics_csv"],
                     "predictions_npz": artifact_paths["predictions_npz"],
                     "report_json": artifact_paths["report_json"]})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "run_config.json", config.to_dict())
    try:
        import subprocess
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    provenance_manifest = build_manifest(
        config=config.to_dict(), seed=config.seed, dataset_path=data["dataset_path"],
        git_revision=revision,
        extra={"dataset_license_or_terms": data["metadata"].get("license", "not supplied"),
               "code_license": "BSD-3-Clause", "group_key": data["group_source"],
               "data_type": data["metadata"].get("data_type", data["metadata"].get("physics_level", "unspecified")),
               "config_sha256": config_hash, "dataset_version": dataset_version},
    )
    provenance_manifest["physics_level"] = data["metadata"].get("physics_level", "unspecified")
    provenance_manifest["scientific_warning"] = "Engineering benchmark only; no scientific validity, calibration, or detection claim is established."
    write_json(out / "reconstruction_manifest.json", provenance_manifest)
    split_ids = {k: data["simulation_ids"][v].tolist() for k, v in split.items()}
    report = {
        "schema_version": "1.0",
        "benchmark_kind": "paired_field_reconstruction",
        "data_type": data["metadata"]["data_type"],
        "dataset_version": dataset_version,
        "is_synthetic": data["metadata"]["is_synthetic"],
        "scientific_approval": False,
        "scientific_warning": "Engineering benchmark only; no scientific validity, calibration, or detection claim is established.",
        "dataset_warning": data["metadata"].get("scientific_warning", "Review source-specific limitations before interpreting results."),
        "interpretation": "This report is engineering validation, not a measurement, detection, or validated posterior.",
        "power_spectrum_metric_note": "Masked maps use zero-filled, valid-voxel-centered windowed pseudo-spectra; no survey-window deconvolution is applied.",
        "stage": "field reconstruction only; emulator prediction and neutrino-mass inference remain separate downstream stages",
        "assumptions": ["paired target is the initial field", "lensing map broadcast along the line of sight", "CMB feature channels are constant over the volume", "group IDs represent independent parent realizations", "noise and 5% gradient stresses are illustrative, not survey-systematics models"],
        "dataset": {"path": data["dataset_path"], "sha256": data["dataset_sha256"], "metadata": data["metadata"]},
        "config": config.to_dict(),
        "config_sha256": config_hash,
        "normalization_fit": "training partition only",
        "normalization_parameters": norm_state,
        "group_key_used": data["group_source"],
        "split_ids": split_ids,
        "group_disjoint": True,
        "n_rows": {k: int(len(v)) for k, v in split.items()},
        "n_groups": {k: int(len(set(data["groups"][v]))) for k, v in split.items()},
        "results": results,
        "neural_model": neural,
        "runtime_seconds_total": fit_seconds,
        "peak_rss_bytes_process_lifetime": peak_rss,
        "artifacts": artifact_paths,
    }
    np.savez_compressed(out / "predictions.npz", truth=target[test], simulation_id=data["simulation_ids"][test], group_id=data["groups"][test], mask=mask[test], **predictions)
    write_json(out / "reconstruction_report.json", report)
    return report


def _peak_rss_bytes() -> int | None:
    try:
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None
