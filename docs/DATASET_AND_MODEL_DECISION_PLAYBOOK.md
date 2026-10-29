# νField Dataset and AI Model Decision Playbook

**Status:** engineering decision guide, not dataset certification or scientific approval.  
**Purpose:** choose a dataset for a specific experimental question and choose the least complex inference model that can answer it under controlled validation.

## 1. The central rule

Do not start with a neural architecture. Start with a target estimand, a simulator-to-observation contract, and a data product whose parameter labels are verified. The source-family name is not enough. Before a product can enter a training run, record its exact release/product ID, cosmological conventions, target column and units, snapshot/redshift, coordinate/length units, box geometry, selection effects, independent parent-group definition, license/terms, and SHA-256 hashes.

The repository intentionally separates data *roles*. One source may be excellent for one role and ineligible for another.

| Source family | First role | Use for primary neutrino-mass supervision? | Main condition |
|---|---|---:|---|
| Quijote massive-neutrino products | Controlled mass response and field-level training | Potentially, after exact-product verification | Official docs describe a single `nwLH` Latin-hypercube with 2,000 simulations and Mν in [0.01, 1.0] eV; verify the actual local product, parameter file, units and independent groups before training |
| Quijote finite-difference/derivative products (`Mnu_p`, `Mnu_pp`, `Mnu_ppp`, where provided) | Local response and sensitivity checks | Not automatically a continuous posterior grid | Such products may have sparse, derivative-focused mass support |
| Quijote Latin-hypercube massive-neutrino products (e.g. product families described as `nwLH`, where applicable) | Candidate ML parameter coverage | Potentially, only after inspecting actual schema and labels | Verify every target column and ranges from the exact source product |
| DESI DR1 LSS mocks | Catalog estimator, survey mask/window/weighting and covariance stability | Generally not the first target-training set | Mock count is not cosmology coverage; verify the exact cosmology labels and whether the target varies |
| DESI DR1 observed catalogs | Late-stage observed-data application | No supervised truth per observed sky | No known per-realization true Mν label; use an independently validated likelihood/summary path |
| AbacusSummit | Cross-suite/cross-code robustness | Only for compatible products | Sparse or discrete cosmologies cannot prove a continuous-mass emulator works |
| CAMELS | Baryonic/astrophysical domain-shift study | Not by default | The documented standard parameter table lists Mν = 0.0 eV; verify a specialized product if considering anything else |
| JAX-FLI | Software and forward-model reference | Not a dataset | Pin a version and verify conventions; importing a toolkit is not validation |

Official references to check for each data intake: [Quijote massive-neutrino documentation](https://quijote-simulations.readthedocs.io/en/latest/Mnu.html), [Quijote Latin-hypercube product docs](https://quijote-simulations.readthedocs.io/en/latest/LH.html), [DESI DR1 documentation](https://data.desi.lbl.gov/doc/releases/dr1/), [AbacusSummit cosmology documentation](https://abacussummit.readthedocs.io/en/latest/cosmologies.html), [CAMELS parameter table](https://camels.readthedocs.io/en/latest/parameters.html), [CAMELS data portal](https://www.camel-simulations.org/data), and [JAX-FLI documentation](https://jax-fli.readthedocs.io/en/stable/). These sources describe source families; the actual downloaded product still needs product-level verification.

## 2. Define the estimand before choosing the model

Write the exact question in one sentence. Examples:

1. **Mass-response benchmark:** “Can a summary statistic distinguish the selected simulation products at their supplied discrete Mν values?” This may be a discrete classification or likelihood-ratio problem.
2. **Posterior inference over a continuous prior:** “Given a simulator that covers the full prior, can we estimate p(θ | x) with calibrated intervals over that prior?” Requires a defensible density/simulation design, support coverage, and calibration.
3. **Reconstruction:** “Given a periodic evolved field, can a model recover the known initial field?” Requires paired initial/final fields with matched ICs.
4. **Survey robustness:** “Does an already-trained summary/inference pipeline remain stable under the survey window, masks, weights and noise?” Requires mocks with those effects; not necessarily a mass-supervised target grid.
5. **Observed-data application:** “What does the validated likelihood infer from a specific observed product?” Requires a compatible forward/observation model and systematics analysis, not supervised labels.

These are different tasks. A good architecture for one can be inappropriate for another.

## 3. Dataset intake gates

A local file is not admissible to the training pipeline until the following gates pass:

- **G0 — provenance:** original URL or access route, exact release/product, license/terms, download time, checksum, file inventory and responsible data steward.
- **G1 — schema:** exact arrays/tables, columns, dimensions, dtypes, missing-value policy and ID uniqueness checked against documentation.
- **G2 — units/conventions:** Mν unit and definition, h-scaled quantities, comoving length unit, coordinate frame, Fourier convention, redshift/snapshot, cosmological parameter definitions and neutrino treatment recorded.
- **G3 — target support:** actual target values, count of unique values, independent parent groups per target, range, and prior overlap measured. Do not infer these from a source name.
- **G4 — physical consistency:** initial/final fields pair correctly; fields and parameters belong to the same realization; boxes and units align; neutrino-aware response is present where required.
- **G5 — split contract:** all views of a parent realization, phase pair, box, or related simulation family that would create dependence stay within one split. Decide whether the scientific claim requires a harder cross-cosmology or cross-code holdout.
- **G6 — baseline readiness:** simple summary statistic and estimator run on a small controlled subset; expected trends and limits pass documented tests.
- **G7 — model permission:** only after G0–G6 should the dataset be used for hyperparameter selection.

`nufield-product-review` and the target-contract modules provide partial machine-checkable gates. They do not independently prove that human-entered metadata is correct.

## 4. Primary dataset strategy

### 4.1 First: inspect exact massive-neutrino Quijote products

Use a local data manifest to freeze the exact product. Inspect the official product tables and then the actual file metadata. The official `nwLH` documentation describes 2,000 simulations that vary Mν alongside Ωm, Ωb, h, ns, σ8 and w, with the documented Mν range [0.01, 1.0] eV. This makes it a strong candidate for a controlled mass-sensitive ML benchmark. Still verify the local product’s columns and units, `Cosmo_params.dat` mappings, independent IDs/phases, redshift/snapshot, release conventions, and actual labels before training. Do not assume every Quijote Mν-related suite has identical mass support, independent simulations, box resolution or snapshots. In particular, derivative-focused finite-difference products and Latin-hypercube training designs have different statistical purposes.

For each candidate product, produce:

- a schema report (fields, shapes, IDs, dtypes, compression and file checksums);
- a parameter-grid report (unique masses, other cosmological parameters, correlation, group counts and range);
- an independence report (number of unique boxes, phases, parent runs, snapshots and derived views);
- a target-responsiveness report (difference in P(k) or other predeclared summary across mass levels, with phase/seed variance shown separately);
- a small baseline run with a locked test group set.

The initial objective should be **controlled recovery on a finite, explicit support**. If the target values are sparse, report a discrete posterior or local response; do not dress that grid up as a calibrated continuous mass posterior.

### 4.2 Second: external suite checks

Only after in-suite validation, use compatible external products to probe generalization. A different code, initial-condition method, mass resolution, cosmology or output redshift can all change the domain. Such a test is a *shift experiment*, not merely another random test split. Report the shift axes.

### 4.3 Third: survey realism

Use DESI mocks to validate how your catalog/field pipeline interacts with redshift bins, selection, masks, angular/radial windows, weights, fiber-assignment corrections and covariance estimators. The DESI DR1 release docs describe both 1,000 EZmock realizations and 25 AbacusSummit realizations in relevant released mock collections; these are not equivalent to 1,025 independent cosmologies. Record the mock family and intended purpose rather than just a row count.

Only after the observation model reproduces established catalog/summary checks should observed DESI data enter a late-stage application. No observed sky has a known per-realization Mν label.

### 4.4 CAMELS role

Use CAMELS primarily as an astrophysical/baryonic model-shift test. The documented CAMELS parameter table lists Mν fixed at 0.0 eV for its standard parameter sets, so these standard products should not be used as nonzero-neutrino-mass supervision. Its data portal says access is governed by a data-use policy and may require a request; do not assume every product can be downloaded without authorization. Verify any newer or specialized product explicitly before treating it otherwise. Different hydrodynamic models, feedback settings and resolutions can test whether learned features mistake baryonic changes for a neutrino-mass signal.

## 5. Model ladder: simplest first

### Level A — deterministic feature baseline

- Compute a binned matter/tracer power spectrum with documented Fourier normalization and binning.
- Include mean density, box length, window, shot-noise convention and units in metadata.
- Use a Gaussian summary likelihood only when the covariance is estimated from appropriate independent simulations and the assumptions are explicit.
- Fit nuisance parameters or hold them fixed and say so. A nuisance parameter that varies in the data but is omitted from the inference model can produce deceptively narrow, biased posteriors.

### Level B — compact discriminative/summary model

Use a small MLP to map a fixed-width vector of summaries into learned features or parameters. It is a useful regression baseline, but point estimates alone do not supply a posterior. Calibration requires a separate probabilistic method or explicit interval model.

### Level C — conditional MDN

The implemented `ConditionalMixtureDensityNetwork` models a conditional density as a mixture of diagonal Gaussian components in bounded, transformed parameter space. It is inexpensive, auditable, and useful for a small-to-moderate number of summary features. Its limitations include mixture misspecification and diagonal within-component covariance. Use it as one reference model rather than assuming it is the best model.

### Level D — simulation-based inference with NPE

Use `sbi` NPE with a documented density estimator (e.g. a neural spline flow where supported) when you can provide an observation encoder and enough simulator draws. SBI works best when the simulator can sample from the declared prior and the simulator output actually reflects the target physics. Keep parent groups disjoint. The default SBI workflow can internally split rows for early stopping, so a parent group that contributes repeated rows may leak unless rows are unique parents; this repository's NPE wrapper therefore fails closed on repeated parent groups for its row-based internal split.

For richer designs with multiple views per simulation, use a custom grouped training loop or reduce to one properly selected observation per independent parent for SBI's standard workflow. Do not claim parent-group-safe training while using a row-random internal split.

### Level E — 3D CNN/U-Net

Use a 3D CNN or U-Net only after you have sufficient independent, physically varied boxes and can split by parent/phase/cosmology correctly. A CNN can fit survey or simulation fingerprints and consume much more memory than a summary model. Compare equal simulation budgets and measure posterior calibration rather than reconstruction error alone.

### Level F — neural operators (e.g. FNO)

Consider only when the input/output are regularly sampled fields with consistent geometry, boundary conditions and resolution, and the research goal justifies the added complexity. Periodic-box assumptions do not automatically transfer to masked survey data.

### Level G — multimodal model

Add galaxy clustering + lensing only once their simulations share a physically coherent cosmology, selection assumptions, geometry and compatible nuisance structure. Compare modality-specific posteriors, naive combination, and calibrated joint inference. A model that concatenates unrelated features is not multimodal cosmological inference.

## 6. How to select the model

Choose a primary ranking metric before comparing trials. For posterior prediction, use a proper score (e.g. log score or CRPS) and report calibration as a separate gate. Posterior coverage is not a substitute for a proper score, and a low NLL alone does not guarantee physical correctness.

Suggested candidate selection policy:

1. Pre-register the parameter prior, likelihood/observations, metric, model candidates, compute budget, and split seeds.
2. Fit preprocessing only on training groups.
3. Use an inner train/early-stopping group split inside the training partition.
4. Use outer validation groups for candidate/model/hyperparameter ranking—not for early stopping.
5. Keep the test partition locked throughout model selection.
6. Reject candidates with unstable training, invalid support, major calibration failures, or systematics failures.
7. Freeze code/config/model once. Evaluate the locked test once, then publish results including negative results.
8. Run simulation-based calibration (SBC) and posterior predictive checks before interpreting a mass posterior.

The repository's MDN grid uses a coarse 68% coverage screen and normalized CRPS as a selection heuristic. This is not the final calibration protocol. Larger SBC runs and an untouched test suite remain necessary.

## 7. Data splitting policy

Use three distinct levels of independence:

- **row/view split:** usually insufficient if multiple noisy views or snapshots come from one parent simulation;
- **parent-group split:** minimum expectation; all related views from one realization remain together;
- **domain split:** hold out cosmologies, phases, codes, boxes or survey conditions to test the claim you wish to make.

For early stopping, create a nested group split inside the training partition. For hyperparameter selection, use a separate outer validation group set. The final test set must never be used to choose epochs, learning rates, transforms, features, architecture, training budgets, or priors.

## 8. Reproducibility and reporting

Every run should save a config, source file checksums, dataset checksum(s), source product contract, git commit (if available), dependency versions, random seeds, device, simulation IDs, group IDs, split fingerprints, elapsed time, peak memory (when available), training curves, metric definitions and warnings. Do not place raw restricted datasets or credentials in the repository.

Publish at minimum:

1. parameter support and group-count table;
2. feature/summary measurement uncertainty;
3. classical baseline result;
4. posterior predictive figures and SBC rank histograms;
5. calibration and posterior-coverage curves;
6. performance under systematic perturbations;
7. domain-shift generalization matrix;
8. accuracy/calibration versus GPU-hours/simulation count;
9. list of failed candidates and predeclared gates;
10. limitations and data provenance record.

## 9. Acceptance gates

- **Data gate:** source product, exact columns, units, cosmology and independent group semantics verified.
- **Physics gate:** established summary trends reproduced against a trusted reference; no unvalidated Mν response is used for scientific output.
- **Baseline gate:** conventional estimator and covariance behavior documented.
- **Inference gate:** simulation-based calibration and prior predictive checks run; coverage deviations are explained.
- **Robustness gate:** systematics/domain shifts tested without tuning on the locked test set.
- **Claim gate:** every paper/README claim follows from a named artifact and reproducible evaluation; toy results are never described as a neutrino-mass constraint.

## 10. Recommended decisions for the first research cycle

1. Quijote massive-neutrino product inspection and parameter-grid audit.
2. Discrete response baseline on a verified Mν subset.
3. Power-spectrum Gaussian baseline under declared nuisance assumptions.
4. Conditional MDN on summary features, with a nested grouped split.
5. Independent SBI NPE comparison when `nufield-starter[sbi]` is installed and its split constraints are respected.
6. Cross-suite validation only after same-suite calibration.
7. DESI mock observation pipeline checks; CAMELS baryonic shift checks.
8. 3D CNN only if simulation scale and compute justify it.
9. Joint lensing/CMB only after matching forward models and observations are available.

**Non-claim:** This playbook chooses a defensible experiment sequence. It does not certify that any local dataset is Mν-suitable, establish that neural inference outperforms a baseline, or produce a physical measurement.
