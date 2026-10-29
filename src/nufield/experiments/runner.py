"""End-to-end small-grid benchmark orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nufield.config import InferenceConfig, SimulationConfig, SplitConfig
from nufield.inference.kernel_posterior import KernelPosterior
from nufield.inference.sbc import coverage_table
from nufield.parameters import PARAMETER_NAMES
from nufield.provenance import build_manifest, write_json
from nufield.simulation.dataset import create_dataset, load_dataset, make_split
from nufield.validation.report import summarize_point_predictions, write_benchmark_plots


def run_benchmark(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 123,
    split_config: SplitConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    split_cfg = split_config or SplitConfig()
    infer_cfg = inference_config or InferenceConfig()
    split = make_split(data["simulation_id"], split_cfg, seed)
    tr, va, te = split["train"], split["validation"], split["test"]
    # This starter does not perform model selection on validation rows. They are held out explicitly
    # for future tuning and are excluded from fitting and final test evaluation.
    model = KernelPosterior(infer_cfg.bandwidth, infer_cfg.top_k).fit(
        data["features"][tr], data["theta"][tr], data["simulation_id"][tr]
    )
    rng = np.random.default_rng(seed + 1)
    posterior_list = []
    point_list = []
    ess = []
    nearest = []
    for idx in te:
        summary = model.summarize(
            data["features"][idx], n_samples=infer_cfg.posterior_samples, rng=rng
        )
        posterior_list.append(summary.samples)
        point_list.append(summary.point_estimate)
        ess.append(summary.effective_sample_size)
        nearest.append(summary.nearest_distance)
    truth = data["theta"][te]
    prediction = np.asarray(point_list)
    point_report = summarize_point_predictions(truth, prediction, PARAMETER_NAMES)
    coverage = coverage_table(posterior_list, truth)
    report = {
        **point_report,
        "method": model.metadata(),
        "n_train": int(len(tr)),
        "n_validation_held_out": int(len(va)),
        "n_test": int(len(te)),
        "split_seed": seed,
        "train_ids": data["simulation_id"][tr].tolist(),
        "validation_ids": data["simulation_id"][va].tolist(),
        "test_ids": data["simulation_id"][te].tolist(),
        "split_disjoint": bool(
            not set(data["simulation_id"][tr]) & set(data["simulation_id"][te])
            and not set(data["simulation_id"][tr]) & set(data["simulation_id"][va])
            and not set(data["simulation_id"][va]) & set(data["simulation_id"][te])
        ),
        "coverage_diagnostic": coverage,
        "mean_effective_particle_count": float(np.mean(ess)),
        "median_nearest_distance": float(np.median(nearest)),
        "feature_note": "Radial log power from toy galaxy overdensity; not a survey estimator.",
        "scientific_interpretation": "Software smoke test only; no cosmological inference claim.",
    }
    write_json(out / "benchmark_report.json", report)
    np.savez_compressed(
        out / "benchmark_predictions.npz",
        truth=truth,
        prediction=prediction,
        test_ids=data["simulation_id"][te],
        posterior_samples=np.asarray(posterior_list, dtype=np.float32),
    )
    plot_paths = write_benchmark_plots(out / "figures", truth, prediction, PARAMETER_NAMES)
    report["figures"] = plot_paths
    write_json(out / "benchmark_report.json", report)
    manifest = build_manifest(
        config={
            "split": split_cfg.__dict__
            if hasattr(split_cfg, "__dict__")
            else {k: getattr(split_cfg, k) for k in split_cfg.__dataclass_fields__},
            "inference": {k: getattr(infer_cfg, k) for k in infer_cfg.__dataclass_fields__},
        },
        seed=seed,
        dataset_path=dataset_path,
        extra={"report": str(out / "benchmark_report.json")},
    )
    write_json(out / "manifest.json", manifest)
    return report


def run_demo(
    output_dir: str | Path, *, n_sims: int = 160, grid_size: int = 12, seed: int = 7
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        n_sims=n_sims, grid_size=grid_size, seed=seed, n_k_bins=min(10, max(4, grid_size // 1))
    )
    created = create_dataset(config, out / "dataset.npz")
    report = run_benchmark(out / "dataset.npz", out / "benchmark", seed=seed + 100)
    return {
        "created": created,
        "benchmark_report": str(out / "benchmark" / "benchmark_report.json"),
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "scientific_warning": report["scientific_interpretation"],
    }
