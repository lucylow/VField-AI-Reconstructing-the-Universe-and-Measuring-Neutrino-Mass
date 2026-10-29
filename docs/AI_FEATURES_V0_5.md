# νField AI Features v0.5.0 — Technical Guide

## 1. Scope and scientific boundary

This release adds a distinct `nufield.ai` package for probabilistic regression, multimodal encoding,
active simulation design, model auditing, domain-shift diagnostics and reproducible training. It is
an engineering layer around scientific inference; it does not upgrade a synthetic simulator into a
validated cosmological calculation. A model is research-ready only after the data-generating physics,
observation process, group structure, parameter priors, likelihood assumptions and held-out tests have
been reviewed.

The included `make_toy_inverse_problem` is a nonlinear synthetic regression dataset designed to test
software paths. `latent_a`, `latent_b`, and `latent_c` are not cosmological parameters. Do not use the
example's scores, posterior intervals or acquisition points as neutrino constraints.

## 2. AI workflow architecture

The intended data path is:

1. Validate input products and scientific metadata before loading a model.
2. Assign stable parent-simulation identifiers and create group-disjoint training, model-selection,
   variance-calibration, conformal-calibration and test partitions.
3. Fit feature transformations and target transformations on the training groups only.
4. Benchmark a simple summary-statistic/likelihood baseline and a small-data GP or kernel posterior.
5. Train one or more neural encoders using validation data for early stopping.
6. Select a model using declared validation scores and eligibility gates—not the locked test scores.
7. Fit calibration transforms on separate calibration groups. Fit conformal radii on a separate
   calibration subgroup if the scale transform was learned from calibration residuals.
8. Evaluate exactly once on independent test groups and report group-cluster uncertainty on scores.
9. Save the model, configuration, data fingerprints, split manifest and environment record.

The model-selection and calibration subsets are different for a reason. If the same residuals are
used to optimize predictive variance and then to compute a conformal quantile, the ordinary split-
conformal argument no longer directly applies. The CLI demo intentionally separates these roles.

## 3. Choosing model families

### Summary features: `ResidualSummaryEncoder`

Use an MLP for low-dimensional power-spectrum bins, BAO/RSD summaries and other predeclared features.
It is often a more appropriate first neural baseline than a 3-D CNN when the number of independent
simulations is small. Compare it with a classical likelihood on the same summary definition. For
positively skewed power, fit the transform and scale using training simulations only and record the
valid k range, power units, window treatment and shot-noise convention.

### Dense 3-D fields: `CompactFieldEncoder3D`

This compact residual CNN encodes regular-grid volumes. Its inductive bias is local spatial pattern
recognition; it does not inherently know the Fourier-space survey window, neutrino hierarchy, redshift
space, periodic boundary conditions or the physical units of a field. Those assumptions belong in
contracts and forward models, not merely in architecture comments. A field model must be compared with
power-spectrum, reconstruction and Gaussian-likelihood baselines.

### Multimodal inputs: `MultimodalPosteriorNetwork`

Available encoders include 3-D fields, 2-D maps and low-dimensional spectra. Modality fusion uses an
explicit availability mask. Every modality must be generated from a matched latent cosmology or tied
together with a documented likelihood model. A zero-filled missing modality is only valid when the
availability bit is set; a real zero map and an absent map must not be indistinguishable.

The shipped network emits either a diagonal Gaussian approximation or a Gaussian mixture. A mixture
is more expressive for parameter degeneracies but can suffer component collapse and bad tail behavior.
For high-dimensional posterior estimation, also benchmark the optional `sbi` neural posterior
estimation path (including a neural spline flow), a kernel posterior, and classical inference.

### Small simulation budget: `GaussianProcessEmulator`

The exact GP is a useful smooth-function emulator and uncertainty baseline when independent simulation
counts are modest. It uses a Cholesky factorization and costs roughly O(N^3) compute and O(N^2) memory.
The returned variance conditions on a fixed kernel and noise specification and should be calibrated.
The included length-scale grid maximizes a training marginal-likelihood criterion; assess generalization
on grouped validation simulations. It is not a free hyperparameter oracle.

### Variable-size catalog: `MaskedPointSetEncoder`

The set encoder uses masked attention plus masked mean pooling and is permutation invariant. The input
features, object-selection rule, survey mask, coordinate frame and number-density normalization must be
specified. A neural set encoder cannot infer an unrecorded selection function by magic.

### Field compression: `VariationalFieldAutoencoder3D`

The VAE is a latent-compression/reconstruction baseline. Latent dimensions are not automatically
interpretable physical parameters. Track reconstruction error, power-spectrum error, phase-sensitive
field losses and downstream parameter recovery; latent quality alone is insufficient.

## 4. Probabilistic model outputs and losses

`HeteroscedasticRegressor` emits a per-target mean and log variance. The diagonal Gaussian objective is

\[
\mathcal L_{\mathrm{NLL}}=\frac12\sum_p\left[\log(2\pi\sigma_p^2)
+\frac{(\theta_p-\mu_p)^2}{\sigma_p^2}\right].
\]

This captures conditional ambiguity/aleatoric variance under the model. It does not estimate all
uncertainty caused by finite training data. Train independently seeded ensemble members and decompose
variance as

\[
\operatorname{Var}(\theta\mid x) \approx
\operatorname{Var}_{m}(\mu_m(x)) + \mathbb E_m[\sigma_m^2(x)].
\]

The first term is empirical ensemble disagreement and the second is mean predicted aleatoric variance.
This decomposition is useful but not proof of calibration. Check coverage and simulation-based
calibration on independent simulations, including model misspecification tests.

For multimodal posteriors, the mixture-density objective evaluates
`logsumexp(log_softmax(logits) + component_log_prob)`. The component covariance is diagonal. Use the
mixture as a baseline and compare with SBI flows where posterior geometry demands richer dependence.

## 5. Model training

`nufield.ai.training.fit_model` provides:

- AdamW optimization with weight decay;
- best-checkpoint restoration and early stopping on validation loss;
- a ReduceLROnPlateau learning-rate scheduler;
- gradient clipping and nonfinite-loss/gradient guards;
- optional CUDA mixed precision;
- run-seeded Python/NumPy/PyTorch RNGs;
- atomic checkpoint writes and JSON training histories;
- optional direct validation of group disjointness.

For DataLoader use, prefer `ArraySimulationDataset` if arrays fit RAM, or `HDF5SimulationDataset`
for HDF5 products that should be read lazily per item. Set `num_workers` conservatively and retain the
parent-group manifest outside the mini-batches. Never split a 3-D box into overlapping tiles and then
randomly split the tiles across train and test: they are not independent simulations.

The `ModelEMA` utility is standalone. Compare raw and EMA weights using the same validation partition,
then lock the selected version before test evaluation. The training engine doesn't silently apply EMA.

## 6. Calibration and uncertainty

### Variance temperature

`VarianceTemperature` fits a scalar multiplier \(T>0\) to predictive variances by minimizing held-out
Gaussian NLL: \(\sigma^2_{\mathrm{cal}}=T\sigma^2\). It can improve average calibration but does not
force nominal interval coverage at every confidence level. Save the calibration split fingerprint
and fit it on a calibration set that did not select architecture or early stopping.

### Split conformal intervals

`ConformalRegressor` computes a finite-sample order statistic of absolute standardized residuals.
`GroupConformalRegressor` forms a maximum residual score per parent simulation before computing the
quantile, so one simulation with many correlated views does not masquerade as many independent
calibration points. Coverage depends on exchangeability of calibration and future units; it may fail
under shifted cosmology or observation conditions. Use an independent in-domain calibration set and
report shift tests.

### Posterior chains and SBC

`posterior_diagnostics` provides ESS and split-Rhat utilities for chain diagnostics. These apply to
sample sequences and are distinct from simulation-based calibration. SBC ranks use known simulated
truths and independent posterior draws. Flat SBC ranks are a useful diagnostic, not a guarantee against
every misspecification. Also inspect parameter-wise coverage, posterior predictive discrepancies and
physical boundary behavior.

## 7. Active simulation design

`nufield.ai.priors.PriorSpecification` declares bounds, units and uniform/log-uniform coordinate
measures. Its Sobol and Latin-hypercube designs aim to cover parameter space efficiently. The
`active_learning.select_candidates` API supports:

- `maximin`: choose far points in prior-scaled parameter coordinates;
- `ensemble_disagreement`: prioritize predictions that differ across independently trained members;
- `predictive_entropy`: approximate a Gaussian-ensemble information score;
- `hybrid`: combine normalized disagreement and design diversity.

Acquisition scores are proposals, not simulation results. Use the target scale in model-disagreement
scores, enforce simulator feasibility, preserve a design ledger and compare adaptive designs against
random/LHS/Sobol designs at the same simulation budget. Active learning may overfocus on model failure
regions or rare prior edges. Do not discard the independent, predeclared test suite as new simulations
are added.

## 8. Robustness and domain shift

`MahalanobisOOD` and `NearestReferenceOOD` require an independent in-distribution calibration set to
set their thresholds. Their scores are diagnostics rather than calibrated probabilities of being OOD.
`DriftMonitor` compares features with PSI and KS diagnostics. A statistically detectable feature shift
may be irrelevant; a small aggregate shift may hide a scientifically important effect. Interpret alerts
by physical feature, not an arbitrary single score.

The domain-adversarial components use a gradient-reversal head to discourage domain identification.
This can also remove parameter information when domains and target values are confounded. Run the
`domain_balance_report` before training and report both domain classification and target-recovery
metrics. Domain invariance is not the same as physical correctness.

`groupwise_metrics` exposes worst-group errors, while `paired_cluster_bootstrap_difference` resamples
parent simulations rather than individual rows. Use paired held-out rows for candidate comparisons
and report confidence intervals around score differences. Independent network seeds measure training
variability; independent simulations measure data/physics variability. They must not be conflated.

## 9. Explainability

`input_gradient_saliency` and `integrated_gradients` provide local attribution with respect to a chosen
output coordinate. `permutation_importance` estimates predictive reliance on held-out features. These
are model explanations, not causal physical discoveries. For highly correlated k-bins, single-feature
permutation can understate information or create unphysical spectra. Add grouped k-range ablations,
modality ablations, seed stability and physically plausible perturbations.

## 10. Run the demo

```bash
python -m pip install -e '.[ml,dev]'

# Train, calibrate on disjoint groups, then test once on the locked toy split.
nufield-ai demo --output-dir outputs/ai_demo --n-samples 600 --epochs 50 --seed 42

# Compare a GP inverse emulator and neural model on the same toy splits.
python examples/ai/compare_gp_vs_neural.py --n-samples 320 --epochs 20 \
  --output outputs/ai/gp_vs_neural.json

# Propose a prior-bounded maximin design; it does not run the simulator.
nufield-ai acquire --low 0.06 0.6 --high 0.5 1.0 --n-candidates 2048 \
  --n-select 20 --seed 42 --output outputs/ai/acquisition.json
```

The demo's `--domain-shift` flag is intended to test drift-monitoring plumbing on synthetic features.
It is not a realistic DESI systematic model.

## 11. Research checklist before any cosmology claim

- The exact simulation product and schema are confirmed, including parent simulation IDs and true
  parameter values; neutrino mass actually varies over the declared prior.
- Total matter, CDM+baryon, neutrino response, units, cosmological convention and redshift are clear.
- The observation model includes relevant selection functions, mask, shot noise, redshift-space effects,
  lensing geometry and survey window—or the model is explicitly labeled as a simplified prototype.
- Train, validation, variance-calibration, conformal-calibration and test groups are disjoint.
- Normalization, feature selection, HPO, early stopping and variance calibration do not inspect test
  targets.
- Classical likelihood / summary baseline and an appropriate non-neural baseline are included.
- The posterior passes SBC and coverage tests on held-out simulations, including prior edges and
  nuisance/systematic shifts.
- The result is compared across seeds and independent simulation realizations; uncertainty on
  improvements is reported with parent-group resampling.
- A pinned CAMB/CLASS or validated N-body/approximate-simulation reference checks physical quantities.
- Model checkpoint, exact configuration, data hashes, split manifest, environment and failed runs are
  archived; the README does not describe toy output as a neutrino-mass measurement.

## 12. Architecture choice at a glance

| Regime | First comparison | Follow-up candidate | Mandatory check |
|---|---|---|---|
| Few hundred summary sims | Classical likelihood / kernel posterior | GP or residual MLP | Grouped validation and calibrated intervals |
| Many independent summary sims | Classical likelihood | MDN / SBI NPE flow | SBC, prior-edge coverage and misspecification |
| Dense periodic 3-D boxes | Power spectrum + reconstruction | Residual 3-D CNN / FNO | Field and spectrum losses, symmetry tests, resolution transfer |
| Large simulation budget | Summary baseline | 3-D neural posterior embedding | Compute-matched ablation and held-out cosmologies |
| Galaxy + lensing maps | Separate likelihoods / late fusion | Mask-aware multimodal transformer | Each modality alone, missingness ablations, shared cosmology |
| Variable-size catalogs | Clustering summaries | Masked point-set encoder | Survey selection and density-normalization tests |

The final choice should be driven by calibrated accuracy, robustness and compute—not by architecture
novelty or the number of neural layers.
