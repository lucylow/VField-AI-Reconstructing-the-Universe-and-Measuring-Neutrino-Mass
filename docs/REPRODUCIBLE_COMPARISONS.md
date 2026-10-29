# Reproducible comparisons and posterior archives

## Principle

The comparison unit is a *matched experiment*: the same independent parent simulations, target prior, parameter definitions, observations, allowed nuisance assumptions, and data support. A model comparison across different mass priors or different source products is not a fair architecture comparison unless the difference itself is the stated experiment.

## Compare archived posterior samples

The posterior comparison utility accepts posterior archives with samples, truth values and (where available) simulation/group IDs. Before interpretation, verify the arrays are in the same parameter order and canonical units. Ensure samples refer to the same simulations, and do not compare one model on easy observations to another on hard observations.

```bash
nufield-compare-posteriors --help
```

Follow the CLI's help for archive paths and output options. Preserve the generated JSON report, metric definitions and input checksums beside the plot/report.

## Metrics

- **Proper score:** CRPS or log score for posterior predictions, when its assumptions and numerical implementation are appropriate.
- **Point error:** bias/RMSE is supplementary and must not replace posterior-quality evaluation.
- **Coverage:** compare nominal to empirical coverage for multiple central credible levels; report sample size and uncertainty of the coverage estimate.
- **SBC:** under prior-predictive simulation, inspect rank statistics for calibrated computation; SBC does not protect against simulator misspecification or prior/data mismatch.
- **Posterior predictive check:** draw parameters/posterior-predictive observations and test predeclared summary discrepancies.
- **Domain shift:** repeat evaluation by withheld simulation suite, phase, cosmology, resolution, tracer, noise, and systematic scenario.
- **Compute:** report simulation count, wall-clock time, CPU/GPU and memory, with hardware/software provenance.

## Locked-test procedure

1. Define groups and assign train/inner-validation/outer-validation/test once.
2. Save the group IDs and split fingerprint, not only row indices.
3. Use training groups to fit scalers and model parameters; never standardize using all rows.
4. Use inner groups for early stopping and a separate outer validation set for ranking candidates.
5. Freeze a model, code version, feature configuration, posterior sampler and scoring script.
6. Open the locked test once. Record that the test was evaluated and whether any subsequent decision changed.

If test metrics lead to another architecture, the previous test becomes development data. A new independent test set is then required for another confirmatory claim.

## Required artifact bundle

A model result should include the model/config, input dataset checksum, product contract, split fingerprints, posterior samples, truth arrays, metric report, dependency versions, runtime/resource report, plots, and a concise `limitations.md`. Public results should be reproducible without distributing restricted cosmology data: include an exact retrieval manifest and deterministic scripts where licensing allows.
