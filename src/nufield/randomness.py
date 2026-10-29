"""Explicit seed utilities. Never rely on process-global NumPy RNG state."""

from __future__ import annotations

import hashlib

import numpy as np


def seed_for_id(master_seed: int, simulation_id: str) -> int:
    """Derive a stable 32-bit seed for one realization ID."""
    digest = hashlib.sha256(f"{master_seed}:{simulation_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def rng_for_id(master_seed: int, simulation_id: str) -> np.random.Generator:
    """Create a deterministic, realization-specific random generator."""
    return np.random.default_rng(seed_for_id(master_seed, simulation_id))


def set_global_seed(seed: int) -> None:
    """Set commonly used global seeds for demos only; scientific code should pass generators."""
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
