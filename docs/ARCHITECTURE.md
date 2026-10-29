# Architecture and module boundaries

## Design principles

The package separates the *simulator*, *observation model*, *feature map*, *inference algorithm*, and
*diagnostic report*. Each boundary has a simple NumPy-facing contract so an eventual CLASS/CAMB and
N-body/PM adapter can be introduced without rewriting every downstream module. The first architecture
is intentionally small enough to audit. The point is not to maximize the number of abstractions; it is
to make every physical assumption visible and every experimental artifact traceable.

## Simulator boundary

A production simulator should accept a parameter record plus a realization seed, and return a typed
bundle of latent fields and observation products. It must record the cosmology, transfer solver version,
initial conditions, redshift, box geometry, mesh convention, neutrino treatment, and any approximation.
The current toy generator returns a Gaussian field shaped with a phenomenological spectrum. It contains
no nonlinear gravitational evolution. Consequently, its `initial_field` and final field are closely
related by construction, unlike a realistic galaxy-survey observation.

## Observation boundary

The current lognormal-Poisson tracer is a pedagogical placeholder. A realistic galaxy likelihood may
need halo occupation, redshift-space mapping, fiber assignment, angular/radial selection, completeness,
shot noise, and a window function. The order of these operations matters. Do not assume that a survey
selection mask is equivalent to multiplying a density mesh by a Boolean mask after voxelization.

## Feature boundary

Features are deterministic functions of the observation and their definition is stored in metadata.
A feature extractor must state its units, normalization, scale cuts, masks, bin edges, and policy for
empty bins. Any scaler or learned transform is fit on training data only. Summary-statistic baselines
should include known estimators from established cosmology packages when the physics is real.

## Inference boundary

The starter kernel posterior is a nearest-feature particle weighting approximation. A production SBI
module should wrap a well-maintained method such as neural posterior estimation, include a proper prior
and proposal scheme, account for sequential simulation proposals when used, and validate posterior
calibration on independent simulations. A `sample()` API is useful but insufficient to prove statistical
validity; training history, sampling diagnostics, model checks and simulation-based calibration are part
of the method.

## Multimodal boundary

Each modality has its own units and observation operator. Joint inference must respect shared cosmological
parameters, correlated sample variance, overlapping sky masks, redshift kernels, beam/noise models, and
cross-covariance. Concatenation is merely a representation choice. It does not automatically combine
likelihoods correctly or guarantee an information gain.

## Algorithm discovery boundary

Search must operate on a reproducible experiment. Never allow a language model to modify evaluation
labels, priors, masks, held-out split IDs, or immutable tests. Constrain candidate edits to schema-checked
hyperparameters or reviewed patches, evaluate in isolated environments, and record code diffs and test
logs. The objective should be multi-dimensional: science accuracy, calibration, robustness, compute, and
maintainability. A lower validation loss alone is not sufficient.
