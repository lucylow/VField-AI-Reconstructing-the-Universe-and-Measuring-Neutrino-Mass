# νField AI Model Selection, Architecture, and Training Plan

**Version:** 2.0 · **Primary initial model:** simulation-based neural posterior estimation (NPE) with a hybrid 3-D CNN + power-spectrum embedding.

## Executive recommendation

Do not start with a transformer or a large language model. For structured cosmological fields, the first neural model should be a compact residual 3-D CNN that embeds a density field, an MLP branch for the field's power-spectrum summary, and an optional 2-D CNN branch for a compatible lensing/convergence map. Feed the learned embedding into neural posterior estimation (NPE), initially using a neural spline flow (NSF) if sample size and compute allow. Compare against power-spectrum + likelihood/posterior baseline, classical reconstruction, and a small MLP feature baseline. A neural model earns its complexity only if it improves held-out inference and remains calibrated.

The NPE choice follows the `sbi` library's guidance that high-dimensional observations should use NPE or NRE with an embedding network; NPE is convenient when the posterior must be sampled repeatedly for many observations. NSF can outperform simpler density estimators in some cases but can train more slowly and overfit, so a MAF/MDN comparison is retained.

## 1. Task definition before architecture

The target is a posterior, not only a point label:

`q(θ | x) ≈ p(θ | x)`, where `θ = [Mν, Ωm, σ8, h, ns, ...]` and `x` is a simulated observation under a documented forward model.

A point predictor trained on mean-squared error may appear accurate while hiding multimodality, parameter degeneracy, or uncertainty miscalibration. It is useful as a cheap sanity baseline, but it is not the primary inference model.

The posterior depends on the simulator and the prior. If simulation parameters were sampled from proposal density `r(θ)` rather than the target prior `π(θ)`, standard NPE does not magically correct the experiment design. Either train using the target prior, use a supported sequential/adaptive SBI workflow, or apply a carefully validated proposal/prior correction where method assumptions permit.

## 2. Architecture choices and the reasons for them

### 2.1 Required baselines

1. **Power-spectrum baseline.** Use fixed bins, documented assignment/window corrections, and a conventional likelihood or calibrated SBI estimator. This tests whether the pipeline works and creates a strong transparent comparator.
2. **Summary MLP.** An MLP embeds the same `P(k)` or correlation-function vector. It is a low-cost neural comparator.
3. **Field CNN.** A residual 3-D CNN embeds the spatial density field. This is the main test of spatial information beyond the summary statistic.
4. **Hybrid CNN + P(k).** A CNN field branch and an MLP spectral branch combine before posterior estimation. This may improve optimization by exposing stable summary information alongside a learned representation, but can also duplicate information; perform ablations.
5. **Optional 2-D lensing branch.** Add only if the lensing field is generated under a compatible cosmology/geometry/selection model. A toy projected map is not a lensing dataset.

### 2.2 Why not a giant transformer first?

A transformer has quadratic costs in token count unless patching/sparse mechanisms are used, and data volume is often the bottleneck. A small CNN has a stronger inductive bias for local spatial structure and an easier memory profile. Try attention only after a stable CNN baseline exists and a measured ablation indicates a specific limitation.

### 2.3 Why not an LLM as the cosmology model?

An LLM is not a suitable replacement for a conditional posterior estimator over continuous physical parameters. The LLM feature belongs in algorithm search, configuration assistance, or experiment-report drafting. Candidate algorithm code must be sandboxed and tested, not executed blindly; conclusions still come from numerical benchmarks.

## 3. Spatial field preprocessing

Record and test:

- field type: total matter, cold+baryon, halo count, galaxy count, or overdensity;
- assignment kernel: CIC/TSC/PCS or the exact source mapping;
- grid resolution, box length and units (`Mpc/h` versus `Mpc`);
- redshift and snapshot/light-cone convention;
- smoothing and deconvolution policy;
- masks, selection, noise and random-catalog treatment;
- whether density is `ρ`, counts, or `δ = ρ/<ρ> - 1`.

Never normalize using all simulations before splitting. Fit feature scalers on training rows only. The `Standardizer` utilities in this continuation intentionally expose `state_dict()` so the transformation can be reproduced against the same training data. For fields, assess whether per-voxel normalization is appropriate across independent simulations; in homogeneous periodic boxes, coordinatewise scaling can encode grid artifacts if alignment or boundary conventions are inconsistent. Alternatives include physically motivated global scaling or power normalization. Make it an explicit ablation rather than a silent default.

A `log1p(counts)` transform can stabilize nonnegative count maps, but it is a preprocessing choice, not a Poisson likelihood. Preserve raw counts and metadata so you can compare.

## 4. Hybrid embedding design

The current implementation has three branches:

- `FieldCNN3D`: residual convolutions with group normalization and adaptive pooling for `[C,D,H,W]` fields;
- `SpectrumMLP`: nonlinear embedding of fixed-width `P(k)` bins;
- `MapCNN2D`: convolutional embedding of projected 2-D maps;
- a final fusion head that receives branch embeddings and an availability mask.

The flattened packet is merely a data-transfer representation for an SBI embedding-network API. It is unflattened before encoding. An absent lensing input is represented by zero values **and** an availability bit; otherwise, the model cannot distinguish a missing modality from a physically zero map.

The architecture uses group normalization to reduce dependence on large batch sizes and residual blocks to stabilize optimization. It deliberately avoids spatial positional encodings because the baseline assumes periodic or consistently aligned grids; if the sky mask or light-cone geometry breaks this assumption, the data representation must be redesigned rather than hoping the model learns around it.

## 5. Neural posterior estimation setup

The `nufield-train-npe` tool integrates with the `sbi` library through `posterior_nn(model="nsf", embedding_net=...)` and `NPE`. It requires explicit parameter prior bounds. The training pipeline must:

1. load only arrays mapped explicitly from a dataset card;
2. split at the independent group level before learning preprocessing;
3. fit the standardizer on train+development groups only, never the final test groups;
4. train on the declared prior or a documented sampling proposal;
5. retain all training seeds and hyperparameters;
6. sample posterior on locked test observations;
7. calculate bias, RMSE, coverage and interval width;
8. run SBC with independent parameter draws and simulations;
9. test simulation misspecification and out-of-support behavior.

The code's held-out experiment runner is an engineering baseline. For grouped datasets with multiple crops from one parent, implement group-aware inner validation rather than relying on a row-level internal split. The safer first format is one record per independent simulation group.

## 6. Prior and parameter conventions

`Mν`, `sum_mnu_ev`, and product-specific labels may differ. Map labels from metadata, convert units once with explicit code, and preserve original columns. Establish whether the prior is over total neutrino mass, individual mass eigenstates, degenerate species, or another convention. Record assumed hierarchy and minimum-mass policy where relevant. Do not assume a wide label range automatically corresponds to an allowed physical prior.

The prior ranges should be chosen before opening the final test result. Training points should cover the target prior or use a documented sequential procedure. The posterior cannot be trusted outside the training support simply because the flow returns samples there; check support and report extrapolation warnings.

## 7. Model selection protocol

Compare the following at equal data splits and comparable physical assumptions:

The `nufield.models.selector.recommend_models` planning heuristic also accounts for the ratio of
independent training simulations to inferred parameters. When that ratio is below 100, learned
candidate scores receive a capped penalty and the recommendation includes a `data_regime_note`.
This is a conservative ranking aid, not a sample-complexity theorem or accuracy guarantee; use
grouped held-out calibration, SBC and coverage tests to decide whether a model is acceptable.

| Model | Inputs | Why retain it | Key limitation |
|---|---|---|---|
| Gaussian/likelihood baseline | P(k) or documented summary | transparent reference, quick debugging | depends on approximation and covariance assumptions |
| Kernel posterior | standardized low-dimensional summaries | non-neural SBI baseline | weak scaling, local particle approximation |
| MLP + NPE | summary features | lower compute and sample complexity | may lose spatial information |
| 3-D CNN + NPE | field | main field-level hypothesis | data/compute intensive and susceptible to simulator shortcut learning |
| Hybrid field + P(k) + NPE | both | joins local and spectral structure | duplicate information/overfit risks |
| Hybrid multimodal | field + lensing + masks | tests complementary constraints | requires matched observable models and careful covariance/systematics treatment |

Tune only on training/development groups. Keep the final test groups fixed, and if model choice is repeatedly adjusted after viewing the test results, create a new holdout or report that the test became a development set.

## 8. Metrics

For each posterior-validated simulation, report:

- median/mean bias and RMSE for each parameter;
- empirical coverage of central 50%, 68%, 90%, and 95% intervals;
- average interval width, with scale/units explicit;
- SBC rank histograms across independently drawn `(θ,x)` pairs;
- posterior predictive checks on both fields and summary statistics;
- distance from training support and flag extrapolation;
- compute cost, simulator count, GPU-hours, peak memory and wall time.

Tighter credible intervals are not intrinsically better. A narrow miscalibrated posterior is worse than an appropriately broad posterior. A mass posterior can appear tight because the network exploits a nuisance correlation or an incorrectly fixed parameter; add nuisance variations and cross-suite tests to investigate.

## 9. Training defaults

Initial sweep for a low-resolution field benchmark:

- CNN width: 8, 12, 16;
- embedding dimension: 64, 128;
- posterior estimator: MAF versus NSF, retain MDN as a small-data comparator;
- dropout: 0.0, 0.05, 0.1;
- learning/training settings: begin with library defaults; change one dimension at a time;
- 3–5 training seeds for architecture sensitivity;
- a fixed independent set of simulations for final evaluation.

These are starting values, not optimal hyperparameters. A dataset of only a few hundred simulations is unlikely to justify a large model with many nuisance dimensions. Start with lower-resolution grids and strong baselines, then spend simulation budget where diagnostics show the model needs it.

## 10. Failure analysis

**Training/validation divergence:** inspect target normalization, learning rate, parameter coverage and posterior architecture; lower complexity before increasing epochs.

**Excellent test accuracy but bad SBC:** check whether evaluation simulations follow the same prior and simulator, whether labels were leaked through filenames or metadata, and whether test design is independent.

**Good P(k), weak CNN:** verify field assignment, centering, field channel order and normalization. It may be that spatial information adds little beyond P(k) for this dataset; that is a valid result.

**CNN beats P(k) only on one suite:** run held-out cosmology, phase, resolution, and simulation-family tests. Shortcut features from simulation artifacts are a leading concern.

**Joint maps tighten the posterior dramatically:** check whether modalities share information, covariance was double-counted, masks/units/geometry align, and the joint likelihood is correctly specified.

**Model reports precise Mν on observed DESI data but cannot recover Mν on mocks:** no-go; the observed posterior likely reflects priors/model assumptions or calibration defects rather than validated data information.

## 11. Sources

- `sbi` guidance for inference selection: https://sbi.readthedocs.io/en/stable/how_to_guide/06_choosing_inference_method.html
- `sbi` embedding network guide: https://sbi.readthedocs.io/en/stable/how_to_guide/04_embedding_networks.html
- `sbi` neural network choices: https://sbi.readthedocs.io/en/latest/how_to_guide/03_choose_neural_net.html
- JAX-FLI field-level forward model reference: https://jax-fli.readthedocs.io/en/stable/
- Quijote Mν suite: https://quijote-simulations.readthedocs.io/en/latest/Mnu.html
