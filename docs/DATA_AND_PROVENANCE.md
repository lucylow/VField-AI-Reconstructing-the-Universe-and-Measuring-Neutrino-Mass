# Data and provenance contract

## Local artifact policy

Synthetic starter datasets are stored as compressed NumPy archives plus JSON metadata. Real survey
catalogs, mocks, CMB products and lensing simulations are **not bundled**. Each external dataset needs
a source URL, release identifier, access date, license/terms, checksum, coordinate convention, units,
selection documentation, and citation. Keep large data outside Git and avoid committing credentials.

## Dataset metadata fields

- `dataset_name`, `schema_version`, `physics_level` (`toy`, `approximate`, `validated`), `created_utc`
- `source_code_version`, `config_sha256`, `random_seed`, `parameter_names`, `parameter_units`
- `grid_size`, `box_size_mpc_h`, `feature_names`, `feature_units`, `n_simulations`
- `simulation_ids`, `split_assignment`, `software_versions`, `limitations`

## Data leakage controls

- Split by complete simulation realization ID; do not split voxels or random crops from one simulation
  across train/test when the target is generalization to new realizations.
- For interpolation claims, hold out parameter draws; for extrapolation claims, hold out whole parameter
  regions and explicitly label them as extrapolation.
- Fit standardization, feature selection, and calibration transforms on training/validation only.
- Freeze test IDs and file hashes before final evaluation.
- Make repeated seeds explicit; paired-phase experiments require special handling and documentation.

## DESI and other public surveys

Use official release documentation and published analysis conventions. A spectroscopic catalog is not
by itself a cosmological likelihood or a ready-to-use 3-D voxel tensor. Selection, completeness, masks,
redshift distributions, weights, fiber assignment, window functions, and covariance must be handled.
Do not assume a convenient mock suite spans the required neutrino-mass dimension. Verify the simulation
cosmology for each product before training a neutrino-sensitive estimator.
