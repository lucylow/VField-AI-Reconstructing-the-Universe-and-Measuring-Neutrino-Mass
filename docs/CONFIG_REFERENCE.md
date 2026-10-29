# Configuration reference

The demo configuration is a YAML document. The Python dataclasses in `nufield.config` are the source
of truth for supported fields. Unknown keys are rejected rather than silently ignored except for
metadata-only top-level sections retained for human-readable context.

## `simulation`

- `n_sims`: number of independently drawn toy realizations. For smoke tests use 40–100; a real SBI
  project requires a simulation count determined by parameter coverage and posterior calibration.
- `grid_size`: cubic voxel-grid side length. The default demo stores several grids in RAM. Increase
  this only after profiling memory and deciding on chunked data storage.
- `box_size_mpc_h`: periodic box size used for bookkeeping in the toy spectrum. This is not evidence
  that the toy generator has the physics of a box simulated in comoving Mpc/h.
- `mean_counts_per_cell`: average Poisson tracer count before modulation by the latent toy field.
- `seed`: master seed. Per-realization seeds are derived from the simulation ID.
- `n_k_bins`: number of radial bins requested by feature extraction; empty bins are imputed to a
  median log-power for fixed feature dimensionality. That policy should be reviewed for any physical
  estimator.

## `inference`

- `bandwidth`: Gaussian kernel bandwidth in standardized feature space. It is a dimensionless tuning
  parameter, not a physical scale.
- `top_k`: maximum number of nearest training simulation particles used for each approximate posterior.
- `posterior_samples`: number of resampled particles per test observation. More particles reduce Monte
  Carlo noise in reported sample quantiles but do not fix a misspecified kernel posterior.

## `split`

- `train_fraction`, `validation_fraction`, `test_fraction`: each is strictly between zero and one and
  must sum to one. Splits are row-level in v0.1; `simulation_id` uniqueness is enforced.
- The test partition must not be used to choose hyperparameters, feature variants or score weights.
  The current runner excludes validation rows as a future tuning partition, but does not yet implement
  hyperparameter tuning automatically.

## `physics_level`

The starter permits only `toy`. This guard is intentional. A future physics backend must not simply
change the label. Add a dedicated backend implementation, parameter mapping, unit tests, reference
validation, and a review gate before promoting results to `approximate` or `validated`.
