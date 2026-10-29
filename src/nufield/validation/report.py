"""JSON and plot generation for benchmark runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nufield.inference.metrics import median_absolute_error
from nufield.provenance import TOY_WARNING


def summarize_point_predictions(
    truth: np.ndarray, prediction: np.ndarray, parameter_names: list[str]
) -> dict:
    true, pred = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if true.shape != pred.shape or true.ndim != 2 or true.shape[1] != len(parameter_names):
        raise ValueError("incompatible truth/prediction shapes or parameter names")
    bias = np.mean(pred - true, axis=0)
    mae = median_absolute_error(pred, true)
    return {
        "physics_level": "toy",
        "scientific_warning": TOY_WARNING,
        "n_test": int(len(true)),
        "parameter_names": parameter_names,
        "mean_bias": dict(zip(parameter_names, bias.tolist())),
        "median_absolute_error": dict(zip(parameter_names, mae.tolist())),
    }


def write_benchmark_plots(
    outdir: str | Path, truth: np.ndarray, pred: np.ndarray, parameter_names: list[str]
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for j, name in enumerate(parameter_names):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(truth[:, j], pred[:, j], alpha=0.75, s=20)
        lo = min(float(truth[:, j].min()), float(pred[:, j].min()))
        hi = max(float(truth[:, j].max()), float(pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"True {name} (toy)")
        ax.set_ylabel(f"Predicted {name} (toy)")
        ax.set_title("Software smoke test — not a physical measurement")
        fig.tight_layout()
        path = out / f"prediction_{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths
