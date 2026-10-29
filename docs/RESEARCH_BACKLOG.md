# Research backlog (ordered by dependency)

## P0: trustworthy software foundations

- [ ] Freeze parameter and unit conventions in schemas.
- [ ] Add deterministic `SeedSequence` handling for each simulation ID.
- [ ] Verify simulation ID split invariants.
- [ ] Store package versions and config hashes in manifests.
- [ ] Add quick-start and smoke tests to CI.
- [ ] Add benchmark output JSON schema.

## P1: classical scientific baseline

- [ ] Wrap CLASS or CAMB with explicit unit conversion.
- [ ] Compare transfer functions against published reference values.
- [ ] Add a validated linear matter power-spectrum pipeline.
- [ ] Add a Gaussian likelihood / covariance estimator for summary statistics.
- [ ] Reproduce BAO summary statistics on a published benchmark before deep learning.

## P2: physically meaningful simulation

- [ ] Select a neutrino-aware N-body or validated PM/emulator route.
- [ ] Validate the neutrino response over parameter grid and scale range.
- [ ] Add redshift-space distortions and a declared galaxy-bias model.
- [ ] Create mocks and realistic observation operators.
- [ ] Record all simulator limitations by scale, redshift, and cosmology.

## P3: inference

- [ ] Add proper neural posterior estimation (e.g. via a maintained SBI library).
- [ ] Compare neural posterior results with an independent method.
- [ ] Add rank/SBC plots with confidence envelopes.
- [ ] Quantify prior sensitivity and nuisance-parameter degeneracies.
- [ ] Add independent simulations for end-to-end coverage.

## P4: field-level methods

- [ ] Train reconstruction model against known initial conditions.
- [ ] Compare with classical reconstruction using identical inputs.
- [ ] Evaluate posterior-level improvement, not only pixel loss.
- [ ] Report simulation cost and inference speed.

## P5: multimodal and algorithm discovery

- [ ] Add matched lensing maps and a documented lensing noise model.
- [ ] Add modality-specific baselines and missing-modality tests.
- [ ] Add CMB-derived information only with consistent likelihood conventions.
- [ ] Implement LLM candidate proposals through schema-validated config edits.
- [ ] Require all immutable quality gates to pass before a candidate is promoted.
