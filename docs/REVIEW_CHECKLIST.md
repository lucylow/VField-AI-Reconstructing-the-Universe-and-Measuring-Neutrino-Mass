# Independent research review checklist

## Forward model

- [ ] Are cosmological parameters and units fully specified?
- [ ] Is the summed neutrino mass treatment documented and physically appropriate?
- [ ] Are transfer functions and nonlinear evolution matched to the analysis scale?
- [ ] Are the observable and survey window modeled in the correct order?
- [ ] Are stochastic nuisance variables sampled from declared priors?
- [ ] Is there an independent reference comparison?

## Inference and baselines

- [ ] Does the baseline use the same data and prior as the learned method?
- [ ] Are summary statistics and covariance estimates documented?
- [ ] Does the SBI method account for proposal distributions and sequential simulation?
- [ ] Are hyperparameters frozen before test evaluation?
- [ ] Is posterior calibration tested on independent simulated observations?
- [ ] Are parameter degeneracies and prior-boundary effects assessed?

## Multimodal fusion

- [ ] Are fields matched in footprint, pixelization, and redshift interpretation?
- [ ] Are noise models and shared sample variance represented?
- [ ] Are missing-modality and misalignment tests included?
- [ ] Is information gain compared against a matched galaxy-only baseline?
- [ ] Are external likelihood data double-counted?

## Reproducibility

- [ ] Can a clean environment reproduce the experiment from config and scripts?
- [ ] Are software, data and code hashes saved?
- [ ] Is the evaluation split immutable and disjoint?
- [ ] Are all failed runs retained in the experiment log?
- [ ] Do plots name the toy/approximate/validated model level?

## Public claims

- [ ] Are results described as software tests when only toy simulations were used?
- [ ] Are numerical precision claims supported by enough independent simulations?
- [ ] Are uncertainty intervals and their calibration shown?
- [ ] Are limitations stated adjacent to headline results?
- [ ] Has a subject-matter expert reviewed the physical assumptions before claiming a measurement?
