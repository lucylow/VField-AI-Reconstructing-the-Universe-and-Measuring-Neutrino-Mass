"""Stable finite-difference derivatives and response-matrix diagnostics."""

from __future__ import annotations

import numpy as np


def numerical_jacobian(
    function, theta: np.ndarray, step: np.ndarray, *, method: str = "central"
) -> np.ndarray:
    """Numerical Jacobian of a vector-valued function with explicit absolute steps."""
    x, dx = np.asarray(theta, float), np.asarray(step, float)
    if x.ndim != 1 or dx.shape != x.shape or np.any(dx <= 0) or np.any(~np.isfinite(x + dx)):
        raise ValueError("theta and positive step must be matching finite vectors")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")
    y = np.asarray(function(x), float)
    if y.ndim != 1 or np.any(~np.isfinite(y)):
        raise ValueError("function must return a finite vector")
    jac = np.empty((y.size, x.size), float)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += dx[i]
        yp = np.asarray(function(xp), float)
        if yp.shape != y.shape or np.any(~np.isfinite(yp)):
            raise ValueError("function output changed shape or became non-finite")
        if method == "central":
            xm = x.copy()
            xm[i] -= dx[i]
            ym = np.asarray(function(xm), float)
            if ym.shape != y.shape or np.any(~np.isfinite(ym)):
                raise ValueError("function output changed shape or became non-finite")
            jac[:, i] = (yp - ym) / (2 * dx[i])
        else:
            jac[:, i] = (yp - y) / dx[i]
    return jac


def response_singular_values(jacobian: np.ndarray) -> dict[str, object]:
    """Use SVD to identify weakly constrained combinations in a response matrix."""
    j = np.asarray(jacobian, float)
    if j.ndim != 2 or np.any(~np.isfinite(j)):
        raise ValueError("jacobian must be a finite 2-D matrix")
    _, s, vh = np.linalg.svd(j, full_matrices=False)
    condition = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")
    return {
        "singular_values": s.tolist(),
        "condition_number": condition,
        "right_singular_vectors": vh.tolist(),
        "warning": "Small singular values indicate locally degenerate parameter combinations.",
    }


__all__ = ["numerical_jacobian", "response_singular_values"]
