# Experiment log template

Copy this file into `outputs/experiments/<experiment-id>/README.md` before each experiment.

## Identity

- Experiment ID:
- Date/time UTC:
- Git revision:
- Parent experiment:
- Owner/reviewer:
- Status: planned / running / complete / failed / superseded

## Scientific question

State one falsifiable question. Avoid phrases such as “improve the model” without a defined metric.

## Model and data

- Physics level: toy / approximate / validated
- Simulator and version:
- Cosmology parameter prior:
- Nuisance parameter prior:
- Data release or simulation suite:
- Observation operator:
- Feature definition and units:
- Train/validation/test manifest hashes:

## Procedure

- Code/config change:
- Random seeds:
- Hyperparameters:
- Hardware/runtime environment:
- Resource limit:
- Metrics and quality gates fixed before looking at the test set:

## Results

- Primary metric and uncertainty:
- Calibration diagnostics:
- Runtime and memory:
- Failed quality gates:
- Figures/tables:
- Difference from baseline:

## Interpretation

Describe what the results support, what they do not support, possible confounds, and follow-up tests.
Keep negative results. Do not convert toy results into cosmological claims.

## Reproduction

Exact command, environment lockfile, hashes of input files, and paths to generated artifacts.
