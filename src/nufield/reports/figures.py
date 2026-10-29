"""Plot functions for scalar diagnostics and toy density fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_field_slices(field: np.ndarray, path: str | Path, title: str = "Toy density field") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(field, float)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError("field must be finite 3-D")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid = arr.shape[-1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, data, subtitle in zip(
        axes, [arr[mid, :, :], arr[:, mid, :], arr[:, :, mid]], ["x slice", "y slice", "z slice"]
    ):
        im = ax.imshow(data, origin="lower", interpolation="nearest")
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title + " (toy physics)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_coverage(expected: np.ndarray, observed: np.ndarray, path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, o = np.asarray(expected, float), np.asarray(observed, float)
    if e.shape != o.shape or e.ndim != 1:
        raise ValueError("expected and observed coverage must be same-shaped vectors")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", label="ideal calibration")
    ax.plot(e, o, "o-", label="empirical")
    ax.set(xlabel="Nominal coverage", ylabel="Observed coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)
