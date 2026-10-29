"""Explicit model cards for choosing architectures by scientific task and data regime.

This registry describes assumptions and acceptance tests; it is not an empirical leaderboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    name: str
    purpose: str
    observation_types: tuple[str, ...]
    data_minimums: tuple[str, ...]
    dependencies: tuple[str, ...]
    inductive_bias: str
    strengths: tuple[str, ...]
    failure_modes: tuple[str, ...]
    required_baseline: str
    required_diagnostics: tuple[str, ...]
    avoid_when: tuple[str, ...]
    status: str = "candidate_not_validated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_CARDS: dict[str, ModelCard] = {
    "pk_classical": ModelCard(
        "pk_classical",
        "Power spectrum + classical likelihood",
        "Parameter inference",
        ("P(k)", "correlation function"),
        ("documented binning", "covariance or forward likelihood", "survey window where relevant"),
        ("numpy", "scipy"),
        "Compressed second-order spatial statistics and explicit likelihood assumptions",
        ("Data-efficient", "interpretable degeneracies", "strong sanity baseline"),
        ("Compression can discard non-Gaussian information", "covariance assumptions can fail"),
        "not applicable; primary baseline",
        ("posterior predictive checks", "coverage on independent mocks", "covariance sensitivity"),
        ("making full-field claims from a summary-only analysis",),
    ),
    "mdn_summary_posterior": ModelCard(
        "mdn_summary_posterior",
        "Summary features + conditional mixture-density network",
        "Conditional posterior approximation",
        ("fixed-width summary vector",),
        (
            "independent grouped simulations",
            "explicit finite prior bounds",
            "held-out calibration simulations",
            "train-only feature/target transforms",
        ),
        ("numpy", "scipy", "torch"),
        "Conditional diagonal-Gaussian mixture in logit-transformed parameter coordinates",
        (
            "Runs without the optional sbi dependency",
            "Direct posterior sampling after training",
            "Useful low-dimensional summary baseline",
        ),
        (
            "Mixture component collapse",
            "poor tails near prior bounds",
            "multimodal misspecification",
            "calibration can fail despite low NLL",
        ),
        "pk_classical and summary MLP + NPE",
        (
            "independent SBC",
            "68/95% coverage",
            "normalized CRPS",
            "mixture-component ablation",
            "misspecification stress test",
        ),
        (
            "raw high-dimensional fields without an encoder",
            "data grids that do not support the posterior target",
        ),
    ),
    "mlp_npe": ModelCard(
        "mlp_npe",
        "Summary MLP + neural posterior estimation",
        "Parameter posterior",
        ("P(k)", "correlation summaries", "other fixed-width summaries"),
        (
            "many independent simulations",
            "feature transform fit on training split",
            "declared parameter prior",
        ),
        ("torch", "sbi"),
        "Learns nonlinear compression of scientifically chosen summaries",
        ("Lower compute than 3-D models", "well suited to controlled ablations"),
        ("Poor prior coverage", "posterior overconfidence", "sensitivity to chosen summaries"),
        "pk_classical",
        ("SBC", "68/95% coverage", "cross-cosmology holdout", "simulation budget curve"),
        ("summary statistics are not matched to the science question",),
    ),
    "cnn3d_npe": ModelCard(
        "cnn3d_npe",
        "Residual 3-D CNN + NPE",
        "Parameter posterior or field encoding",
        ("regular 3-D fields",),
        (
            "independent 3-D simulations",
            "consistent mesh and units",
            "documented boundary/selection model",
        ),
        ("torch", "sbi"),
        "Local translation-equivariant convolutional features",
        ("Learns non-Gaussian morphology", "moderate architecture complexity"),
        ("Simulator/grid artifacts", "data hunger", "resolution dependence", "mask shortcut"),
        "pk_classical and mlp_npe",
        ("SBC", "cross-phase holdout", "resolution stress", "shuffled-field ablation"),
        ("few simulations", "unmodeled survey masks", "unmatched simulation families"),
    ),
    "fno3d_npe": ModelCard(
        "fno3d_npe",
        "Fourier Neural Operator embedding + NPE",
        "Parameter posterior on regular periodic boxes",
        ("regular periodic 3-D fields",),
        ("uniform grid", "documented box size", "consistent periodic boundary convention"),
        ("torch", "sbi"),
        "Global learned mixing of retained Fourier modes",
        ("Captures long-range field structure", "distinct inductive bias from CNN"),
        ("Wrong boundary assumptions", "mode truncation bias", "resolution-specific artifacts"),
        "cnn3d_npe and mlp_npe",
        ("matched parameter budget", "mode-count ablation", "cross-resolution holdout", "SBC"),
        ("masked survey footprints", "irregular observational catalogs without a field estimator"),
    ),
    "hybrid_npe": ModelCard(
        "hybrid_npe",
        "3-D field + P(k) + optional lensing hybrid encoder",
        "Multimodal parameter posterior",
        ("3-D fields", "spectra", "optional 2-D maps"),
        (
            "aligned cosmological simulations",
            "matched masks/units",
            "explicit modality-availability bits",
        ),
        ("torch", "sbi"),
        "Separate modality encoders followed by learned fusion",
        ("Ablates modality contribution", "supports partially missing modalities"),
        ("Double counting", "misaligned simulated modalities", "leaking modality identifiers"),
        "single-modality models",
        (
            "each modality alone",
            "joint posterior coverage",
            "shuffled pairings",
            "missing modality stress",
        ),
        ("modalities cannot be derived from the same latent cosmology / data model"),
    ),
    "fno_reconstruction": ModelCard(
        "fno_reconstruction",
        "Fourier Neural Operator field reconstruction",
        "Field-to-field reconstruction",
        ("regular periodic 3-D fields",),
        ("paired input/target fields", "consistent grid", "normalization fitted on training data"),
        ("torch",),
        "Spectral mixing maps one field representation to another",
        ("Efficient long-range interactions", "supports reconstruction objective ablations"),
        ("Unphysical periodicity", "over-smoothing", "P(k) match does not guarantee correct phase"),
        "classical reconstruction and 3-D CNN",
        (
            "field RMSE",
            "cross-correlation",
            "P(k) ratio",
            "phase recovery",
            "uncertainty characterization",
        ),
        ("a reconstruction metric is being presented as parameter posterior validity"),
    ),
}


def get_model_card(model_id: str) -> ModelCard:
    try:
        return MODEL_CARDS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model card {model_id!r}; valid={sorted(MODEL_CARDS)}") from exc


def model_registry() -> list[dict[str, Any]]:
    return [card.to_dict() for card in MODEL_CARDS.values()]
