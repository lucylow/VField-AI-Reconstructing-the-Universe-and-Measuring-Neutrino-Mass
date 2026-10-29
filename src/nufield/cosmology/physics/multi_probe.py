"""Joint observable vectors and covariance contracts for multi-probe analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.likelihoods import gaussian_log_likelihood


@dataclass(frozen=True, slots=True)
class ProbeBlock:
    name: str
    data: np.ndarray
    theory: np.ndarray
    covariance: np.ndarray
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        d, t, c = (
            np.asarray(self.data, float),
            np.asarray(self.theory, float),
            np.asarray(self.covariance, float),
        )
        if d.ndim != 1 or t.shape != d.shape or c.shape != (d.size, d.size):
            raise ValueError(
                f"probe block {self.name!r} has inconsistent data/theory/covariance shapes"
            )
        if not self.name or np.any(~np.isfinite(d)) or np.any(~np.isfinite(t + c)):
            raise ValueError("probe name must be nonempty and arrays finite")
        if not np.allclose(c, c.T, atol=1e-10) or np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError("each probe covariance must be symmetric positive definite")
        object.__setattr__(self, "data", d.copy())
        object.__setattr__(self, "theory", t.copy())
        object.__setattr__(self, "covariance", c.copy())


@dataclass(frozen=True, slots=True)
class JointProbeVector:
    names: tuple[str, ...]
    data: np.ndarray
    theory: np.ndarray
    covariance: np.ndarray
    block_slices: dict[str, tuple[int, int]]


def assemble_joint_vector(
    blocks: list[ProbeBlock] | tuple[ProbeBlock, ...],
    cross_covariances: dict[tuple[str, str], np.ndarray] | None = None,
) -> JointProbeVector:
    """Assemble probes into one vector; cross-covariances are mandatory when known/non-negligible.

    Missing cross blocks are set to zero. The resulting covariance is checked positive definite, but
    a successful check does not validate the scientific assumption that omitted correlations vanish.
    """
    if not blocks:
        raise ValueError("at least one probe block is required")
    names = [block.name for block in blocks]
    if len(set(names)) != len(names):
        raise ValueError("probe block names must be unique")
    sizes = [block.data.size for block in blocks]
    offsets = np.cumsum([0] + sizes)
    n = int(offsets[-1])
    data = np.concatenate([b.data for b in blocks])
    theory = np.concatenate([b.theory for b in blocks])
    covariance = np.zeros((n, n), float)
    slices: dict[str, tuple[int, int]] = {}
    for i, block in enumerate(blocks):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        covariance[sl, sl] = block.covariance
        slices[block.name] = (int(offsets[i]), int(offsets[i + 1]))
    cross_covariances = cross_covariances or {}
    for (name_a, name_b), cross in cross_covariances.items():
        if name_a not in slices or name_b not in slices or name_a == name_b:
            raise ValueError(f"invalid cross-covariance key {(name_a, name_b)}")
        sa, sb = slices[name_a], slices[name_b]
        c = np.asarray(cross, float)
        if c.shape != (sa[1] - sa[0], sb[1] - sb[0]) or np.any(~np.isfinite(c)):
            raise ValueError(f"cross covariance {name_a}/{name_b} has invalid shape or values")
        covariance[sa[0] : sa[1], sb[0] : sb[1]] = c
        covariance[sb[0] : sb[1], sa[0] : sa[1]] = c.T
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("joint covariance is not positive definite; inspect cross-probe blocks")
    return JointProbeVector(tuple(names), data, theory, covariance, slices)


def joint_log_likelihood(joint: JointProbeVector, theory_vector: np.ndarray | None = None) -> float:
    """Evaluate the Gaussian joint log likelihood using the full supplied covariance."""
    theory = joint.theory if theory_vector is None else np.asarray(theory_vector, float)
    return gaussian_log_likelihood(joint.data, theory, joint.covariance)


def block_diagonal_covariance(blocks: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """Create a block-diagonal matrix, explicitly suitable only when cross-probe covariance is ignored."""
    mats = [np.asarray(c, float) for c in blocks]
    if not mats or any(c.ndim != 2 or c.shape[0] != c.shape[1] for c in mats):
        raise ValueError("provide at least one square covariance matrix")
    total = sum(c.shape[0] for c in mats)
    out = np.zeros((total, total), float)
    offset = 0
    for c in mats:
        n = c.shape[0]
        out[offset : offset + n, offset : offset + n] = c
        offset += n
    return out


__all__ = [
    "ProbeBlock",
    "JointProbeVector",
    "assemble_joint_vector",
    "joint_log_likelihood",
    "block_diagonal_covariance",
]
