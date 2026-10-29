# Classical mass-grid baseline

## Purpose

The classical mass-grid workflow is a transparent baseline for comparing the observed summary feature vector to simulated summary vectors at a finite set of known cosmological parameter values. It is particularly useful for verifying that the feature pipeline responds as expected before training an expressive neural posterior estimator.

## Likelihood approximation

For each candidate simulation grid point `theta_k`, estimate a mean summary vector `mu_k` and (where the replication design justifies it) a covariance `C_k`. Given observed summary vector `x`, the baseline scores

`log L(x | theta_k) = -0.5 * [(x-mu_k)^T C_k^-1 (x-mu_k) + log det(C_k) + d log(2 pi)]`.

With a discrete prior weight `pi_k`, normalize scores with log-sum-exp:

`p(theta_k | x) proportional to pi_k L(x | theta_k)`.

The implementation uses regularization for covariance stability and returns posterior mass on the **declared finite grid**. It does not interpolate a continuous posterior. Covariance estimation must use independent simulations suitable for the statistic; small samples, correlated boxes or reuse of a sample mean require special care. Record the covariance estimator and regularization value.

## Nuisance policy

The initial implementation refuses to pretend nuisance parameters are handled when they are not. If galaxy bias, shot noise, redshift-space distortions, survey window or baryonic effects vary in the inputs, either model those dimensions, include them as nuisance parameters, or explicitly restrict the baseline experiment to a controlled setting where they are fixed. Omitted varying nuisance structure can bias the posterior.

## Workflow

1. Verify the product-level mass labels and parameter grid.
2. Freeze independent group splits and build all features from training groups only where fitting is needed.
3. Estimate the summary covariance on a justified training/reference sample.
4. Score held-out validation simulations; inspect calibration, confusion between mass levels and parameter degeneracies.
5. Compare against MDN/NPE using the same observations, prior, support and simulation split.
6. Do not open the final test until the feature construction and ranking rule are frozen.

## Interpretation

A narrower posterior is not automatically better. The baseline must recover known parameters with acceptable bias and coverage across independent held-out simulations. For a sparse mass grid, emphasize discrete support and finite-difference response; avoid claims of continuous mass resolution beyond the simulated support.
