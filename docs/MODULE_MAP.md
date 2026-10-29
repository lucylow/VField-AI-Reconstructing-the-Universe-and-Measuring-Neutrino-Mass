# Expanded module map

This note identifies the intended responsibility of each code area, expected dependencies, and current
limitations. It should be updated when the architecture changes.

## Parameters and configuration

`parameters.py` owns the canonical parameter vector. `config.py` owns runtime settings. Configuration
should be serialized rather than relying on untracked notebook state. Any future change to parameter
order must update the schema version and stored metadata. This is especially important for arrays where
parameters are represented as a positional matrix.

## Randomness

`randomness.py` creates per-realization generators derived from a master seed and ID. This avoids the
common failure where changing simulation order changes all later random draws. A production workflow
should also record the NumPy version and bit generator, because exact streams may change across major
library versions.

## Simulation

`simulation/power.py` is intentionally labeled toy. `linear_field.py` creates a periodic random field;
`galaxies.py` maps it to Poisson counts; `lensing.py` creates a 2-D projection-like observable; and
`pipeline.py` composes those pieces. `observation.py` and `redshift_space.py` provide generic nuisance
stressors rather than physically realistic survey operators. `cosmology/classy_adapter.py` is an optional
seam to a linear solver and is not wired into toy generation.

## Features

`features/power_spectrum.py` defines a radial FFT summary with its own normalization. `field_stats.py`
adds one-point moments, autocorrelation, and cross-statistics; `geometry.py` records periodic-box geometry.
Any scientific comparison must match estimator normalization and windows to the chosen reference method.

## Inference

`KernelPosterior` returns approximate local particles from a finite training simulation bank. The optional
conditional Gaussian MLP is a regression baseline, not a rich posterior. Metrics, weighted particles,
SBC, calibration and scores are kept separate so downstream reports cannot hide the distinction between
point accuracy and uncertainty quality.

## Reconstruction

`UNet3D` is a compact optional architecture. It is not trained by the default demo because the paired
field targets do not represent a validated physical inverse problem. `smoothing.py` provides a classical
low-pass diagnostic, not BAO reconstruction.

## Multimodal

`fixed_width_fuse` carries explicit availability flags. It does not correct cross-covariance or replace a
joint likelihood. `cross_correlation.py` supports descriptive alignment checks for paired test maps.

## Validation and registry

Coverage, bootstrap, leakage, posterior-predictive and acceptance-gate modules keep testing separate
from model code. `ExperimentRegistry` records append-only JSONL entries. For a paper, replace any ad hoc
logs with a controlled experiment registry and archive immutable configuration snapshots.

## Discovery

`discovery/search.py` constrains search to a declared candidate space. An LLM can be added as a producer
of candidate configuration JSON, but should not directly modify evaluation labels, tests, or scientific
assumptions. All candidates must be traceable, repeatable, and subject to the same quality gates.
