# Data Adapter Contract

## Principles

- No adapter guesses a remote data schema from a product name.
- Every column or HDF5 dataset path comes from the exact source data model and is recorded.
- Source files remain immutable; transformations create versioned derivatives.
- Data IDs, parent group IDs, phases, random seeds, and mock-family provenance are first-class fields.
- Units and cosmology conventions travel with the arrays. A bare tensor is not a dataset.
- A successful read means only that the selected bytes and schema were accessible.

## Canonical research bundle

For a simple supervised field-level SBI experiment, the derived bundle should contain:

- `fields`: float32 `[N,C,D,H,W]` or `[N,D,H,W]` representation, with declared field semantics;
- `power_spectrum`: float32 `[N,K]`, bins and units declared in the metadata;
- `theta`: float32 `[N,P]`, parameter columns documented and ordered;
- `simulation_id`: unique row identifier;
- `group_id`: parent simulation/phase/cosmology grouping for splits;
- optional `lensing`: `[N,C,H,W]` map under a matched geometry/selection model;
- optional per-row `modality_available` flags;
- card and manifest, source checksums, transform configuration, package versions, and random seeds.

Don't manufacture `group_id = row number` when several outputs derive from the same parent simulation. If the source lacks IDs, create synthetic identifiers only for documented toy smoke tests; research data must preserve original identifiers or explicitly justify a replacement.

## HDF5 mapping

`inspect_hdf5_tree` prints dataset paths/shapes without assuming the Quijote product uses one universal naming scheme. Use that output to map `observable_key`, `parameter_key`, and IDs into the card. Source-specific files may be particle snapshots, density grids, halo catalogs, or arrays and are not interchangeable.

## DESI mapping

Use the exact release's LSS catalog documentation. Record tracer, target/redshift bin, catalog version, observed/mock status, weights, random catalog, angular/radial selection, mask, window handling and covariance product. The `DESICatalogSpec` performs metadata checks only; it does not run a clustering estimator or validate the survey likelihood.

## Dataset assessment status

The status levels in the card are evidence states, not automatically incremented. `assess_arrays` does not certify physical validity. Human review is required before a source is tagged `simulation_validated` or `research_approved`.
