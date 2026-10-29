"""Curated dataset/software catalogue for νField project planning.

Scores are engineering triage priors (0-5), not scientific quality measurements. Entries include
explicit caveats because some popular cosmology data are unsuitable for direct neutrino-mass
training even when they are valuable for robustness checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Objective = Literal[
    "neutrino_mass_inference",
    "field_reconstruction",
    "survey_validation",
    "baryonic_robustness",
    "multimodal_extension",
    "software_smoke_test",
]


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    name: str
    resource_type: str
    primary_role: str
    objectives: tuple[Objective, ...]
    mass_variation: Literal["explicit", "limited", "not_default", "unknown", "not_applicable"]
    field_level: int
    observational_realism: int
    parameter_coverage: int
    accessibility: int
    compute_friendliness: int
    data_url: str
    docs_url: str
    suggested_products: tuple[str, ...]
    limitations: tuple[str, ...]
    recommended_use: str
    avoid_claiming: tuple[str, ...]
    citations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


DATASETS: dict[str, DatasetSource] = {
    "quijote_mnu": DatasetSource(
        source_id="quijote_mnu",
        name="Quijote massive-neutrino simulation subsets",
        resource_type="N-body simulation suite",
        primary_role="Primary simulation source for controlled neutrino-mass sensitivity experiments",
        objectives=("neutrino_mass_inference", "field_reconstruction"),
        mass_variation="explicit",
        field_level=5,
        observational_realism=2,
        parameter_coverage=4,
        accessibility=3,
        compute_friendliness=2,
        data_url="https://quijote-simulations.readthedocs.io/en/latest/",
        docs_url="https://quijote-simulations.readthedocs.io/en/latest/Mnu.html",
        suggested_products=(
            "nwLH Latin-hypercube product (official docs describe 2,000 simulations and Mnu in [0.01, 1.0] eV; verify the exact local product/columns)",
            "Mnu_p/Mnu_pp/Mnu_ppp derivative-oriented subsets for local response checks",
            "density/halo fields at one documented redshift",
        ),
        limitations=(
            "Large particle fields can be storage- and compute-intensive.",
            "Each subset has its own parameter design and initialization convention; inspect the official documentation for the exact product.",
            "A simulation box is not automatically a survey light cone or a DESI selection-function realization.",
        ),
        recommended_use="Begin with the smallest documented neutrino-varying subset that covers the intended mass parameter. Use consistent redshift, grid, units, and paired/separate realization IDs. Hold out entire seeds/phases, not voxels from a field.",
        avoid_claiming=(
            "Do not infer a physical constraint until priors, parameterization, simulator validity, and coverage are documented.",
            "Do not assume the broad standard Latin-hypercube is the massive-neutrino nwLH product; the exact product and parameter file must be inspected.",
            "The documented nwLH range is not a recommendation for the inference prior; predeclare the prior for the intended physics question.",
        ),
        citations=(
            "Villaescusa-Navarro et al. (2020), ApJS 250, 2, https://doi.org/10.3847/1538-4365/ab9d82",
            "Official Quijote massive-neutrino documentation: https://quijote-simulations.readthedocs.io/en/latest/Mnu.html",
        ),
    ),
    "quijote_sancho": DatasetSource(
        source_id="quijote_sancho",
        name="Quijote Sancho redshift-space galaxy mock catalogs",
        resource_type="Galaxy mock catalog suite",
        primary_role="Tracer/HOD/redshift-space robustness layer",
        objectives=("survey_validation", "neutrino_mass_inference", "field_reconstruction"),
        mass_variation="limited",
        field_level=3,
        observational_realism=4,
        parameter_coverage=3,
        accessibility=3,
        compute_friendliness=3,
        data_url="https://quijote-simulations.readthedocs.io/en/latest/",
        docs_url="https://quijote-simulations.readthedocs.io/en/latest/news.html",
        suggested_products=(
            "Redshift-space mock catalogs",
            "Matched power spectra/bispectra if available for selected mock release",
        ),
        limitations=(
            "Verify which cosmologies and Mnu values appear in the specific release before model training.",
            "Mocks implement a particular HOD/selection recipe; they do not span every real-survey systematic.",
        ),
        recommended_use="After demonstrating recovery on the field-level N-body target, test whether inference survives galaxy-tracer construction, redshift-space distortions, and selected HOD variants.",
        avoid_claiming=("Do not assume one mock recipe fully represents DESI selection effects.",),
        citations=(
            "Official Quijote news and suite docs: https://quijote-simulations.readthedocs.io/en/latest/news.html",
        ),
    ),
    "desi_dr1_mocks": DatasetSource(
        source_id="desi_dr1_mocks",
        name="DESI DR1 large-scale-structure mock catalogs",
        resource_type="Survey mock catalogs",
        primary_role="Survey-window, clustering-estimator, and external-validation layer",
        objectives=("survey_validation", "multimodal_extension"),
        mass_variation="limited",
        field_level=2,
        observational_realism=5,
        parameter_coverage=2,
        accessibility=4,
        compute_friendliness=4,
        data_url="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/mocks",
        docs_url="https://data.desi.lbl.gov/doc/releases/dr1/",
        suggested_products=(
            "1000 EZmock realizations for covariance/analysis stability",
            "25 AbacusSummit high-resolution realizations for selected validation tasks",
        ),
        limitations=(
            "The released mock collections are primarily designed for survey-analysis validation/covariance, not as a dense continuous neutrino-mass training grid.",
            "Do not treat 1000 mocks as 1000 independent cosmologies; many share the same approximate cosmology/design.",
        ),
        recommended_use="Use after simulation-space training: validate catalog handling, masks/windows, weights, and clustering summary code. Record tracer, redshift bin, catalog version, and mock family.",
        avoid_claiming=(
            "Do not train a mass posterior unless the chosen mock labels genuinely span Mnu and include suitable independent cosmologies.",
        ),
        citations=("DESI DR1 documentation: https://data.desi.lbl.gov/doc/releases/dr1/",),
    ),
    "desi_dr1_data": DatasetSource(
        source_id="desi_dr1_data",
        name="DESI DR1 observed LSS clustering catalogs",
        resource_type="Observed galaxy/quasar catalog",
        primary_role="Late-stage observed-data application after simulation-based validation",
        objectives=("survey_validation", "multimodal_extension"),
        mass_variation="not_default",
        field_level=2,
        observational_realism=5,
        parameter_coverage=1,
        accessibility=4,
        compute_friendliness=4,
        data_url="https://data.desi.lbl.gov/doc/releases/dr1/",
        docs_url="https://data.desi.lbl.gov/doc/releases/dr1/",
        suggested_products=(
            "Documented clustering catalogs",
            "Matching random catalogs, weights, masks, window functions, and covariance/likelihood products",
        ),
        limitations=(
            "Observed data have no known true cosmological parameter per realization.",
            "Survey selections, weights, completeness, fiber assignment, and window convolution must be treated consistently.",
            "A point prediction on observed data cannot be calibrated using its unknown true parameter.",
        ),
        recommended_use="Use only after mock-based coverage, posterior predictive checks, and a documented likelihood/summary-statistic path are in place. Compare against established DESI analysis products.",
        avoid_claiming=(
            "Do not use observed DESI data as supervised labels for Mnu.",
            "Do not describe posterior narrowing alone as evidence that the result is correct.",
        ),
        citations=(
            "DESI DR1 documentation and LSS/mock catalog descriptions: https://data.desi.lbl.gov/doc/releases/dr1/",
        ),
    ),
    "abacussummit": DatasetSource(
        source_id="abacussummit",
        name="AbacusSummit simulation suite",
        resource_type="High-resolution N-body simulations and mock products",
        primary_role="High-quality external validation and selected cosmology-response tests",
        objectives=("neutrino_mass_inference", "field_reconstruction", "survey_validation"),
        mass_variation="limited",
        field_level=4,
        observational_realism=3,
        parameter_coverage=3,
        accessibility=2,
        compute_friendliness=1,
        data_url="https://abacussummit.readthedocs.io/en/latest/",
        docs_url="https://abacussummit.readthedocs.io/en/latest/cosmologies.html",
        suggested_products=(
            "Documented c019/c020/c021/c022 neutrino-reference cosmologies where appropriate",
            "Compatible phases/snapshots or released mock catalogs",
        ),
        limitations=(
            "Massive-neutrino coverage in named cosmologies is limited/discrete relative to a full continuous training grid.",
            "Many simulations do not have full time-slice outputs; inspect the product table.",
            "Differences in cosmology, initial conditions, and resolution can confound train/test comparisons.",
        ),
        recommended_use="Use as a cross-code or cross-suite test where products are compatible; don't silently merge fields from different cosmological conventions or resolutions.",
        avoid_claiming=(
            "Do not claim a continuous Mnu emulator is validated by a few discrete cosmologies.",
        ),
        citations=(
            "AbacusSummit cosmology docs: https://abacussummit.readthedocs.io/en/latest/cosmologies.html",
            "AbacusSummit simulation list: https://abacussummit.readthedocs.io/en/latest/simulations.html",
        ),
    ),
    "camels": DatasetSource(
        source_id="camels",
        name="CAMELS simulation suites",
        resource_type="Hydrodynamical and N-body simulation suite",
        primary_role="Baryonic/astrophysical domain-shift and simulation-family robustness",
        objectives=("baryonic_robustness", "field_reconstruction"),
        mass_variation="not_default",
        field_level=4,
        observational_realism=2,
        parameter_coverage=3,
        accessibility=2,
        compute_friendliness=1,
        data_url="https://www.camel-simulations.org/data",
        docs_url="https://camels.readthedocs.io/en/latest/",
        suggested_products=(
            "Matched N-body and hydro pairs",
            "Different hydro codes/subgrid models and documented maps/statistics",
        ),
        limitations=(
            "The standard CAMELS parameters documentation lists Mnu = 0.0 eV for the documented standard parameter sets; it is therefore not a nonzero-Mnu training grid by default.",
            "The full data collection is very large; select only the documented products needed.",
            "Different resolution, subgrid physics, and output products must be normalized carefully.",
        ),
        recommended_use="Use as a separate simulation-family stress test for baryonic/domain shift after establishing the Mnu-sensitive core on a neutrino-aware suite.",
        avoid_claiming=(
            "Do not substitute CAMELS for a massive-neutrino training grid unless the exact chosen suite explicitly varies Mnu.",
        ),
        citations=(
            "CAMELS data portal: https://www.camel-simulations.org/data",
            "CAMELS parameter table: https://camels.readthedocs.io/en/latest/parameters.html",
            "CAMELS documentation: https://camels.readthedocs.io/en/latest/",
        ),
    ),
    "jax_fli": DatasetSource(
        source_id="jax_fli",
        name="JAX Field-Level Inference (JAX-FLI)",
        resource_type="Software and differentiable forward-model toolkit",
        primary_role="Reference implementation for differentiable forward modeling / lensing / field-level methods",
        objectives=("field_reconstruction", "multimodal_extension", "software_smoke_test"),
        mass_variation="not_applicable",
        field_level=5,
        observational_realism=3,
        parameter_coverage=4,
        accessibility=3,
        compute_friendliness=1,
        data_url="https://jax-fli.readthedocs.io/en/stable/",
        docs_url="https://jax-fli.readthedocs.io/en/stable/",
        suggested_products=(
            "Reproducible low-resolution tutorial configs",
            "Reference map/catalog examples and validation experiments",
        ),
        limitations=(
            "This is software, not a turnkey labeled data set.",
            "GPU/HPC tutorials may require adaptation for a laptop.",
            "Reproduced results still need version pins, converged resolutions, and validation.",
        ),
        recommended_use="Study as an implementation reference and possible later backend. Do not add a second forward-model stack before the simple data contract and benchmark are sound.",
        avoid_claiming=(
            "Do not claim that importing the toolkit validates the entire νField forward model.",
        ),
        citations=("JAX-FLI docs: https://jax-fli.readthedocs.io/en/stable/",),
    ),
}


def get_dataset(source_id: str) -> DatasetSource:
    """Return one source record or raise a helpful ValueError."""
    try:
        return DATASETS[source_id]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset source {source_id!r}. Available: {valid}") from exc
