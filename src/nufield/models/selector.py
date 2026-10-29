"""Explainable AI architecture selection based on data volume, modalities and compute budget.

This is an initial research-planning heuristic. It does not predict model accuracy; run the benchmark
matrix on the target dataset before committing to a model family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Budget = Literal["laptop", "workstation", "hpc"]
Task = Literal[
    "parameter_posterior", "field_reconstruction", "survey_summary", "multimodal_posterior"
]


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    rank: int
    fit_score: int
    resource_estimate: str
    strengths: tuple[str, ...]
    risks: tuple[str, ...]
    suggested_config: dict
    required_ablation: str
    recommended_as_primary: bool
    data_regime_note: str

    def to_dict(self):
        return asdict(self)


def recommend_models(
    *,
    task: Task = "parameter_posterior",
    n_simulations: int,
    field_shape: tuple[int, ...] | None = None,
    n_parameters: int = 5,
    has_power_spectrum: bool = True,
    has_lensing: bool = False,
    has_mock_domain_shift: bool = False,
    budget: Budget = "laptop",
    target_uncertainty: bool = True,
    periodic_box: bool = False,
) -> list[ModelCandidate]:
    if n_simulations < 1 or n_parameters < 1:
        raise ValueError("n_simulations and n_parameters must be positive")
    if task not in {
        "parameter_posterior",
        "field_reconstruction",
        "survey_summary",
        "multimodal_posterior",
    }:
        raise ValueError("unsupported task")
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("unsupported budget")
    shape = tuple(field_shape or ())
    if shape and any(d <= 0 for d in shape):
        raise ValueError("field dimensions must be positive")
    grid_max = max(shape[-3:]) if len(shape) >= 3 else None
    candidates: list[dict] = []
    # Small-data baseline remains first-class: simpler methods often dominate when simulations are few.
    candidates.append(
        dict(
            name="Power-spectrum + classical likelihood/kernel posterior",
            base=95,
            resource="low",
            strengths=(
                "Transparent baseline",
                "Fast and data-efficient",
                "Debuggable parameter degeneracies",
            ),
            risks=(
                "Depends on summary-statistic/covariance assumptions",
                "May discard non-Gaussian spatial information",
            ),
            config={
                "summary": "documented P(k)/xi(r)",
                "scaling": "fit on training split only",
                "inference": "validated likelihood or kernel posterior",
            },
            ablation="Use exactly the same simulations, prior and observation model for all candidate models.",
        )
    )
    if has_power_spectrum:
        candidates.append(
            dict(
                name="Summary features + conditional mixture-density network (MDN)",
                base=86,
                resource="low",
                strengths=(
                    "Runnable conditional density baseline with PyTorch only",
                    "Supports bounded targets through a logit transform",
                    "Produces posterior samples rather than only point predictions",
                ),
                risks=(
                    "Diagonal component covariance may be too restrictive",
                    "Mixture count can be under/over-specified",
                    "Needs independent calibration and coverage tests",
                ),
                config={
                    "density_estimator": "conditional Gaussian mixture density network",
                    "components": 4,
                    "hidden_dim": 64,
                    "bounded_parameter_transform": "logit",
                },
                ablation="Compare normalized CRPS, SBC, coverage and runtime against classical likelihood and summary MLP + SBI NPE.",
            )
        )
        candidates.append(
            dict(
                name="Spectrum MLP + NPE",
                base=90,
                resource="low",
                strengths=(
                    "Strong neural summary baseline",
                    "Lower sample/compute demand than full 3-D CNN",
                ),
                risks=(
                    "No access to field morphology beyond chosen summaries",
                    "Posterior calibration still required",
                ),
                config={
                    "density_estimator": "MAF or NSF comparison",
                    "embedding_dim": 64,
                    "hidden_features": 64,
                },
                ablation="Compare MLP posterior with classical P(k) inference on the locked test simulations.",
            )
        )
    if shape and len(shape) >= 3:
        cnn_base = 76 if n_simulations < 1000 else 88
        if budget == "laptop":
            cnn_base -= 12 if grid_max and grid_max > 64 else 0
        elif budget == "hpc":
            cnn_base += 3
        if n_simulations < 300:
            cnn_base -= 18
        candidates.append(
            dict(
                name="Residual 3-D CNN + NPE",
                base=cnn_base,
                resource="medium" if grid_max and grid_max <= 64 else "high",
                strengths=(
                    "Learns spatial features from density fields",
                    "Embedding can preserve morphology discarded by P(k)",
                ),
                risks=(
                    "High simulation demand",
                    "Can exploit simulator/grid artifacts",
                    "Memory grows with field volume",
                ),
                config={
                    "width": 8 if n_simulations < 1000 else 12,
                    "embedding_dim": 64 if n_simulations < 1000 else 128,
                    "density_estimator": "MAF first, NSF candidate",
                    "dropout": 0.05,
                },
                ablation="Remove/perturb spatial structure and test cross-phase, cross-cosmology and cross-suite generalization.",
            )
        )
        if grid_max and grid_max >= 12 and periodic_box:
            fno_base = cnn_base - 5
            if n_simulations >= 3000:
                fno_base += 5
            candidates.append(
                dict(
                    name="3-D Fourier Neural Operator embedding + NPE",
                    base=fno_base,
                    resource="high" if grid_max > 48 else "medium-high",
                    strengths=(
                        "Global spectral mixing",
                        "Natural candidate for regular periodic simulation boxes",
                        "Provides a model-family comparison distinct from local CNN filters",
                    ),
                    risks=(
                        "Assumes regularly gridded, approximately periodic fields",
                        "Spectral modes and grid resolution can create shortcut learning",
                        "Not directly suited to masked survey cutouts without explicit window treatment",
                    ),
                    config={
                        "width": 12,
                        "modes": [6, 6, 6],
                        "n_blocks": 4,
                        "embedding_dim": 128,
                        "density_estimator": "MAF/NSF sweep",
                        "boundary_condition": "periodic box only",
                    },
                    ablation="Match parameter count and training budget versus 3-D CNN; vary retained modes and evaluate a separate simulation family.",
                )
            )
        if has_power_spectrum:
            hybrid_base = cnn_base - 2
            if n_simulations >= 2000:
                hybrid_base += 4
            candidates.append(
                dict(
                    name="Hybrid 3-D CNN + P(k) + NPE",
                    base=hybrid_base,
                    resource="high" if grid_max and grid_max > 64 else "medium-high",
                    strengths=(
                        "Combines morphology and stable spectral summaries",
                        "Supports explicit field-vs-spectrum ablations",
                    ),
                    risks=(
                        "May duplicate information",
                        "More trainable capacity and preprocessing",
                    ),
                    config={
                        "field_branch": "residual 3D CNN",
                        "spectrum_branch": "MLP",
                        "embedding_dim": 128,
                        "density_estimator": "MAF/NSF validation sweep",
                    },
                    ablation="Compare field-only, P(k)-only, hybrid, shuffled field, and field low-pass variants.",
                )
            )
    if has_lensing and shape and task == "multimodal_posterior":
        candidates.append(
            dict(
                name="Mask-aware multimodal NPE",
                base=72 if n_simulations < 2000 else 86,
                resource="high",
                strengths=(
                    "Tests complementary galaxy/lensing information",
                    "Explicit missing-modality mask",
                ),
                risks=(
                    "Requires physically matched observations",
                    "Joint covariance/systematics may invalidate naive fusion",
                ),
                config={
                    "fusion": "separate CNN encoders + availability mask",
                    "ablation": "single modality and shuffled cross-modal pairing",
                },
                ablation="Compare each modality alone and together under shared simulated latent cosmology.",
            )
        )
    out = []
    simulations_per_parameter = n_simulations / n_parameters
    for row in candidates:
        score = row["base"]
        learned_model = any(
            term in row["name"] for term in ("MDN", "NPE", "CNN", "Neural Operator")
        )
        data_regime_note = "No low-data adjustment applied."
        if learned_model and simulations_per_parameter < 100:
            penalty = min(15, round(0.15 * (100 - simulations_per_parameter)))
            score -= penalty
            data_regime_note = (
                f"Low-data heuristic: {simulations_per_parameter:.1f} independent simulations "
                f"per inferred parameter; score reduced by {penalty}. Validate calibration and "
                "coverage on held-out simulations."
            )
        if target_uncertainty and "NPE" not in row["name"] and "likelihood" not in row["name"]:
            score -= 3
        if budget == "laptop" and row["resource"] in {"high", "medium-high"}:
            score -= 12
        if budget == "hpc" and row["resource"] in {"high", "medium-high"}:
            score += 5
        if task == "survey_summary" and "Power-spectrum" in row["name"]:
            score += 5
        if task == "field_reconstruction" and "CNN" in row["name"]:
            score += 12
        if has_mock_domain_shift and "CNN" in row["name"]:
            score -= 1  # cross-suite tests are required, not impossible
        out.append(
            {
                **row,
                "fit_score": max(0, min(100, int(score))),
                "data_regime_note": data_regime_note,
            }
        )
    out.sort(key=lambda x: x["fit_score"], reverse=True)
    return [
        ModelCandidate(
            name=r["name"],
            rank=i + 1,
            fit_score=r["fit_score"],
            resource_estimate=r["resource"],
            strengths=r["strengths"],
            risks=r["risks"],
            suggested_config=r["config"],
            required_ablation=r["ablation"],
            recommended_as_primary=(i == 0),
            data_regime_note=r["data_regime_note"],
        )
        for i, r in enumerate(out)
    ]
