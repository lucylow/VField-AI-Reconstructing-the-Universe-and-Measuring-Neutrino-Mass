# Implementation plan and technical contract

## 1. Research objective

Create an auditable experimental framework for comparing full-field and summary-statistic inference
under controlled data-generating assumptions. In the first software iteration the scientific
objective is restricted to **testing plumbing**: deterministic simulation, feature extraction,
train/test boundaries, posterior-like particle weighting, and diagnostic reporting. This stage does
not make any empirical claim about cosmological neutrino masses.

## 2. Staged scope

### Stage A: software MVP

- Generate periodic, small 3-D toy Gaussian fields with a named random seed.
- Transform the field into a lognormal-intensity Poisson tracer grid.
- Produce binned radial power-spectrum features.
- Persist simulations, parameters, data IDs, config, and a provenance manifest.
- Fit a nearest-neighbour/kernel posterior using train-set particles only.
- Evaluate point estimates and approximate weighted posterior samples on held-out IDs.
- Produce coverage, interval-width, and rank histogram diagnostics.

Acceptance criteria: test suite passes; rerunning with the same seed reproduces the dataset; test IDs
do not overlap training IDs; output files are machine-readable; all reports state that the physics is
synthetic and unvalidated.

### Stage B: scientific baselines

- Add a trusted Boltzmann solver wrapper and validate `P(k)` units and parameter conventions.
- Use simulations or emulators that explicitly encode massive neutrinos.
- Implement a likelihood / SBI method with documented prior and proposal corrections.
- Compare against a classical power-spectrum likelihood using the same simulated catalog and prior.
- Use independent simulation seeds and cosmology-level holdouts.

Acceptance criteria: independent review of the forward model; comparison plots with documented
scale cuts; calibration tests pass within Monte Carlo uncertainty; no real-data claim before these
criteria are met.

### Stage C: field-level / learned inference

- Add a 3-D U-Net only once an input/target simulation pair is scientifically defensible.
- Compare learned reconstructions with a named classical method.
- Evaluate reconstruction metrics and downstream parameter posterior metrics separately.
- Use deterministic split manifests and preserve the exact evaluation set.

### Stage D: multimodal extension

- Model galaxy density and lensing as separate observation operators with separate noise models.
- Verify coordinate transforms, mask alignment, and common redshift / line-of-sight assumptions.
- Compare galaxy-only, lensing-only, and joint variants.
- Test information gain on matched simulations, not by comparing unrelated posterior widths.

### Stage E: algorithm search

- Permit search only over declarative hyperparameters or sandboxed, reviewed patches.
- Keep tests immutable during candidate scoring.
- Score reconstruction quality, downstream inference bias, calibration, runtime, and memory jointly.
- Retain all candidate outputs and reject changes that violate physical/statistical invariants.

## 3. Parameter handling

The initial parameter vector is `sum_mnu_ev`, `omega_m`, `sigma8`, `h`, `n_s`, and `bias`. Values
are validated against broad ranges by `CosmologyParameters`. These ranges are convenience bounds for a
toy benchmark, not recommended science priors. Configuration files must explicitly record units and
prior distributions. Never silently mix `h/Mpc` and `1/Mpc`, or physical and comoving coordinates.

## 4. Data contracts

Dataset arrays use shape `(n_simulations, n_features)` for `features`, `(n_simulations, n_parameters)`
for `theta`, and unique `simulation_id` strings. Names and orderings are stored in a JSON metadata file
or embedded metadata. `features[i]` and `theta[i]` must always refer to the same realization. A
split is generated from IDs and saved; it is never re-generated during model evaluation.

## 5. Posterior contract

Every inference method must expose at minimum:

- `sample(observation, n_samples, rng) -> ndarray[n_samples, n_parameters]`
- `point_estimate(observation) -> ndarray[n_parameters]`
- metadata: algorithm, training IDs hash, prior description, feature standardization, and seed.

For weighted-particle methods, report effective sample size and the resampling scheme. Do not use a
weighted training particle set as a substitute for a properly validated likelihood-free posterior.

## 6. Validation contract

The validation report must include test-set count, interval coverage at 50%, 68%, and 95%, mean bias,
median absolute error, posterior interval width, particle ESS, and a split hash. For SBC, draw from the
actual prior used by the simulator and rerun the full pipeline. Use many independent simulations and
show rank histograms with uncertainty bands. For misspecified models, report degradation rather than
recalibrating against the same test set.

## 7. Reproducibility

Every experiment records Python/package versions, git revision when known, config checksum, random
seed, array shapes, parameter names and units, source dataset, and creation timestamp. Large data files
stay outside Git; the repository tracks loaders, schemas, hashes, and small fixtures. Exact bitwise
reproducibility across different BLAS/GPU backends is not guaranteed, so record tolerances and runtime
hardware.

## 8. Non-goals for the starter

- No claim of discovering a new neutrino-mass bound.
- No reproduction of DESI scientific likelihoods.
- No real-world joint CMB/lensing analysis.
- No attempt to substitute toy spectrum equations for CLASS/CAMB or validated N-body models.
- No uncontrolled LLM-written code execution.
