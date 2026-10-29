"""Named, documented profiles that create specific frontend demo states."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCIENCE_WARNING = (
    "SYNTHETIC DEMO DATA ONLY. Generated values are not observations, not a "
    "calibrated cosmological prediction, not a validated posterior, and not "
    "evidence about neutrino mass. Do not cite as scientific results."
)

PARAMETER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "omega_m": {
        "label": "Ωm",
        "name": "Matter density parameter",
        "unit": "dimensionless",
        "min": 0.18,
        "max": 0.42,
        "center": 0.30,
        "sigma": 0.035,
    },
    "sigma8": {
        "label": "σ₈",
        "name": "Fluctuation normalization",
        "unit": "dimensionless",
        "min": 0.65,
        "max": 0.95,
        "center": 0.81,
        "sigma": 0.035,
    },
    "h": {
        "label": "h",
        "name": "Scaled Hubble parameter",
        "unit": "dimensionless",
        "min": 0.58,
        "max": 0.82,
        "center": 0.68,
        "sigma": 0.025,
    },
    "n_s": {
        "label": "nₛ",
        "name": "Primordial spectral index",
        "unit": "dimensionless",
        "min": 0.88,
        "max": 1.04,
        "center": 0.965,
        "sigma": 0.016,
    },
    "sum_mnu_ev": {
        "label": "Σmν",
        "name": "Illustrative neutrino-mass sum",
        "unit": "eV (toy input)",
        "min": 0.0,
        "max": 0.45,
        "center": 0.12,
        "sigma": 0.055,
    },
    "bias": {
        "label": "b",
        "name": "Illustrative tracer bias",
        "unit": "dimensionless",
        "min": 0.65,
        "max": 1.55,
        "center": 1.0,
        "sigma": 0.11,
    },
}
PARAMETER_KEYS: tuple[str, ...] = tuple(PARAMETER_DEFINITIONS)


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """A deterministic set of knobs for a UI state, not a physical model."""

    scenario_id: str
    label: str
    group: str
    description: str
    center: dict[str, float] = field(default_factory=dict)
    spread: float = 1.0
    observation_noise: float = 0.8
    calibration_offset: float = 0.0
    domain_shift: float = 0.0
    coverage_shift: float = 0.0
    split_leakage: bool = False
    failure_mode: str | None = None
    chart_label: str = "Synthetic reference"
    recommended: bool = False

    def resolved_center(self) -> dict[str, float]:
        """Return a full parameter center, without mutating this profile."""
        result = {key: float(meta["center"]) for key, meta in PARAMETER_DEFINITIONS.items()}
        result.update({key: float(value) for key, value in self.center.items()})
        return result

    def to_dict(self) -> dict[str, Any]:
        """Make a JSON-friendly record for provenance and the scenario picker."""
        result = asdict(self)
        result["center"] = self.resolved_center()
        result["is_synthetic_profile"] = True
        result["scientific_warning"] = SCIENCE_WARNING
        return result


_DEFINITIONS: tuple[DemoScenario, ...] = (
    DemoScenario(
        "baseline",
        "Baseline walkthrough",
        "happy-path",
        "Balanced synthetic samples, a clean split, and modest measurement noise.",
        center={
            "omega_m": 0.30,
            "sigma8": 0.81,
            "h": 0.68,
            "n_s": 0.965,
            "sum_mnu_ev": 0.12,
            "bias": 1.0,
        },
        spread=1.0,
        observation_noise=0.8,
        chart_label="Reference demo",
        recommended=True,
    ),
    DemoScenario(
        "low_neutrino",
        "Low toy Σmν",
        "parameter-shift",
        "Moves the illustrative mass-sum input toward the lower edge of the display range.",
        center={"sum_mnu_ev": 0.035, "sigma8": 0.815},
        spread=0.9,
        observation_noise=0.8,
        chart_label="Low toy mass input",
    ),
    DemoScenario(
        "high_neutrino",
        "High toy Σmν",
        "parameter-shift",
        "Moves the illustrative mass-sum input upward to exercise comparison charts.",
        center={"sum_mnu_ev": 0.36, "sigma8": 0.79},
        spread=0.95,
        observation_noise=0.9,
        chart_label="High toy mass input",
    ),
    DemoScenario(
        "noisy_survey",
        "Noisy survey",
        "observation-quality",
        "Adds heteroscedastic noise and lower quality flags to generated catalog records.",
        spread=1.25,
        observation_noise=1.9,
        coverage_shift=-0.035,
        failure_mode="noisy-observations",
        chart_label="Noise stress test",
    ),
    DemoScenario(
        "calibration_bias",
        "Calibration offset",
        "observation-quality",
        "Adds an artificial estimator offset to demonstrate calibration warning states.",
        observation_noise=0.9,
        calibration_offset=0.075,
        coverage_shift=-0.055,
        failure_mode="calibration-offset",
        chart_label="Calibration stress test",
    ),
    DemoScenario(
        "domain_shift",
        "Held-out domain shift",
        "validation-stress",
        "Shifts held-out rows while leaving most training rows near the reference center.",
        observation_noise=1.25,
        calibration_offset=0.015,
        domain_shift=0.06,
        coverage_shift=-0.045,
        failure_mode="domain-shift",
        chart_label="Domain-shift stress test",
    ),
    DemoScenario(
        "coverage_failure",
        "Coverage failure",
        "validation-stress",
        "Depresses generated coverage values to exercise under-coverage diagnostics.",
        observation_noise=0.9,
        calibration_offset=0.025,
        coverage_shift=-0.17,
        failure_mode="coverage-underflow",
        chart_label="Coverage failure demo",
    ),
    DemoScenario(
        "split_leakage",
        "Split overlap warning",
        "validation-stress",
        "Intentionally reuses a realization identifier across splits to test leakage warnings.",
        observation_noise=0.8,
        split_leakage=True,
        failure_mode="split-overlap",
        chart_label="Leakage warning demo",
    ),
    DemoScenario(
        "sparse_data",
        "Sparse dataset",
        "data-volume",
        "Small data volume and broader toy intervals exercise low-count and empty states.",
        spread=1.7,
        observation_noise=1.35,
        coverage_shift=-0.025,
        failure_mode="small-sample",
        chart_label="Low-count demo",
    ),
    DemoScenario(
        "high_resolution",
        "High-volume dashboard",
        "data-volume",
        "Select larger generation counts to test scrolling, pagination, and chart redraws.",
        spread=0.95,
        observation_noise=0.8,
        chart_label="High-volume demo",
    ),
    DemoScenario(
        "malformed_artifacts",
        "Import error states",
        "ui-test-cases",
        "Creates fixture descriptions for malformed JSON, arrays, and unclassified objects.",
        observation_noise=0.8,
        failure_mode="malformed-imports",
        chart_label="Importer test fixtures",
    ),
)

DEMO_SCENARIOS: dict[str, DemoScenario] = {item.scenario_id: item for item in _DEFINITIONS}


def get_scenario(value: str | DemoScenario | None) -> DemoScenario:
    """Resolve a profile with a safe, documented fallback to baseline."""
    if isinstance(value, DemoScenario):
        return value
    if isinstance(value, str) and value in DEMO_SCENARIOS:
        return DEMO_SCENARIOS[value]
    return DEMO_SCENARIOS["baseline"]


def scenario_catalog() -> list[dict[str, Any]]:
    """Return every profile in stable order for programmatic UI selection."""
    return [scenario.to_dict() for scenario in DEMO_SCENARIOS.values()]
