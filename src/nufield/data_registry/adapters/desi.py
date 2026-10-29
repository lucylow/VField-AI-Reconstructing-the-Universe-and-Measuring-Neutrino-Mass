"""DESI large-scale-structure product contract checks.

DESI catalog columns and file names are release/tracer specific. This module deliberately requires
an explicit mapping instead of guessing a universal LSS schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DESICatalogSpec:
    release: str
    catalog_version: str
    tracer: str
    redshift_bin: str
    catalog_role: str
    catalog_path: str
    random_catalog_path: str | None
    completeness_or_weight_columns: tuple[str, ...]
    random_weight_columns: tuple[str, ...]
    mask_definition: str
    window_or_random_handling: str
    covariance_source: str
    official_docs: str = "https://data.desi.lbl.gov/doc/releases/dr1/"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_desi_spec(spec: DESICatalogSpec) -> list[str]:
    issues = []
    if not spec.release.strip() or not spec.catalog_version.strip():
        issues.append("error: release and catalog version are required")
    if not spec.tracer.strip() or not spec.redshift_bin.strip():
        issues.append("error: tracer and redshift bin must be explicit")
    if spec.catalog_role not in {
        "observed_clustering",
        "random_catalog",
        "mock",
        "summary_product",
    }:
        issues.append(
            "error: catalog_role must be observed_clustering/random_catalog/mock/summary_product"
        )
    if not spec.catalog_path.strip():
        issues.append("error: catalog_path is required")
    if spec.catalog_role == "observed_clustering" and not spec.random_catalog_path:
        issues.append(
            "warning: observed clustering analysis normally needs the matching random catalog / selection representation"
        )
    if not spec.completeness_or_weight_columns:
        issues.append(
            "warning: no object-weight/completeness mapping declared; verify LSS catalog construction requirements"
        )
    if not spec.mask_definition.strip():
        issues.append("error: document angular/radial selection and mask treatment")
    if not spec.window_or_random_handling.strip():
        issues.append("error: document survey window / random catalog handling")
    if not spec.covariance_source.strip():
        issues.append("warning: covariance or likelihood source is unspecified")
    issues.append(
        "warning: DESI observed catalogs lack per-object true cosmological parameters; do not use them as supervised theta labels"
    )
    return issues
