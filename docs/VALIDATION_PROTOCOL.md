# Validation protocol

## 1. Unit-level invariants

1. Each configuration parameter is finite and inside declared bounds.
2. A fixed seed produces the same toy field on the same numerical stack.
3. Fields and maps contain only finite values.
4. Counts are non-negative integers; overdensity has finite values.
5. Radial power spectrum bin edges are strictly increasing and bins are documented.
6. Train, validation, and test simulation IDs are disjoint.
7. The posterior sampler returns finite arrays with the requested shape.
8. Posterior samples remain within the declared support unless an explicit transformed parameterization
   is used.

## 2. Parameter recovery smoke tests

Use small samples from the toy data generator and verify only that the pipeline runs end-to-end. Do not
set a numerical science accuracy target for tiny test samples. To test whether a feature encodes a
parameter, create a controlled paired experiment where only that parameter changes. Use paired random
phases where appropriate to reduce cosmic variance, then check that the selected feature changes in the
expected direction under the toy equation. Label that result as a toy-model unit test, not physical
validation.

## 3. Coverage and simulation-based calibration

For each replicate:

1. draw parameters from the stated prior;
2. simulate a fresh observation from the exact forward model;
3. run the complete inference procedure without peeking at the true parameter;
4. draw posterior samples;
5. compute rank statistics and credible-interval inclusion indicators;
6. aggregate over independent replicates and compare to finite-sample binomial uncertainty.

SBC can detect certain implementation/calibration errors under the assumed model. It cannot prove that
the assumed simulator is a good description of the real Universe. Use independent misspecification
checks as well.

## 4. Posterior predictive checks

Given an observation, sample parameters from its posterior and simulate replicated observations. Compare
pre-registered summaries such as mean density, count variance, radial power spectrum, anisotropy, and
cross-correlations. Use posterior predictive distributions rather than a single best-fit simulation.
An observed statistic in a tail is a modeling signal to investigate, not proof of one specific failure.

## 5. Stress tests

The future research-grade model should include distinct tests for:

- shot noise and number density;
- galaxy bias, stochasticity, and scale dependence;
- redshift-space distortions;
- survey masks, completeness, and selection function;
- redshift error and coordinate conversion;
- baryonic feedback and nonlinear modeling;
- cosmologies on the edges and outside the training grid;
- transfer/simulation mismatch;
- missing or corrupted modalities;
- feature normalization fit on training data only;
- priors and parameter degeneracies.

Run each stress test under a frozen evaluation manifest. If stress-test parameters were used to tune the
model, create a separate untouched stress-test set for final reporting.

## 6. Reporting minimum

Each benchmark table must include method, code revision, data model, training count, validation count,
test count, parameter prior, feature definition, seed policy, runtime, memory, bias, median absolute
error, interval coverage, interval width, particle ESS, and limitations. Clearly distinguish uncertainty
from test-sample variability and from simulator discrepancy.
