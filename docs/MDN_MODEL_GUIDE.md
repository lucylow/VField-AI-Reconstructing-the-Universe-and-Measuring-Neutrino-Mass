# Conditional MDN: training and inference guide

## What is implemented

The conditional mixture-density network (MDN) learns an approximate conditional probability density for a finite vector of cosmological parameters given a fixed-width observation summary. Each input embedding predicts mixture logits, means and diagonal log-scales; the likelihood is the log-sum-exp of component log densities. The current within-component covariance is diagonal. It cannot represent arbitrary posterior structure perfectly, and model fit is not calibration.

Target parameters are transformed from declared finite physical bounds to unbounded logit coordinates, then standardized using statistics fitted on the training partition only. Posterior samples are transformed back to the physical bounds after sampling. Values exactly at a hard bound are rejected by the forward logit transform; define a prior whose support and simulation labels agree rather than clipping labels silently.

## Expected NPZ schema

The default training archive contains:

- `features`: finite array shaped `[N, F]`;
- `theta`: finite target matrix `[N, P]` in the parameter order explicitly supplied to CLI;
- `simulation_id`: unique ID for each row;
- `group_id`: parent independent simulation ID. Different snapshots/views/derived products from a common parent should share this ID.

An adjacent metadata file with the same stem and `.json` suffix should document `physics_level`, dataset/product ID, provenance, schema version and parameter-unit conventions. The workflow refuses `physics_level: toy` unless `--allow-toy` is passed. That option is for software tests and does not make toy data scientifically valid.

## Leakage-resistant partitioning

There are three different partitions:

1. **Outer training groups**, which are divided again into fitting and early-stopping groups.
2. **Outer validation groups**, which evaluate and rank the model/configuration; they are not used to choose the best epoch.
3. **Locked test groups**, which must not be used for model/feature/hyperparameter selection and are evaluated once after freezing the experiment.

`nufield-mdn-grid` does not open the test partition. The user must freeze the candidate and then explicitly invoke a one-time final evaluation under the same fixed split protocol. Repeated inspection of test metrics is model selection, even if the code does not call it that.

## Minimal example

```bash
nufield-train-mdn \
  --dataset data/processed/verified_summary_dataset.npz \
  --output-dir outputs/mdn-baseline \
  --parameter-names sum_mnu_ev,omega_m,sigma8 \
  --prior-low 0.06,0.20,0.70 \
  --prior-high 0.40,0.40,0.95 \
  --target-name sum_mnu_ev \
  --n-validation-observations 64 \
  --posterior-samples 2048
```

The prior bounds above are illustrative syntax, **not endorsed physical priors**. Select and document priors based on the exact cosmology, the source product and the scientific estimand. Do not copy them into a production analysis without a physics review.

## Hyperparameter/model selection

```bash
nufield-mdn-grid \
  --dataset data/processed/verified_summary_dataset.npz \
  --output-dir outputs/mdn-grid \
  --parameter-names sum_mnu_ev,omega_m,sigma8 \
  --prior-low 0.06,0.20,0.70 \
  --prior-high 0.40,0.40,0.95 \
  --max-trials 4 --budget laptop --epochs 80 --patience 12
```

The grid is intentionally budget-bounded. It ranks eligible candidates using validation mean CRPS normalized by each parameter's configured prior range after applying a coarse 68% coverage screen. The screen should not be treated as a scientific acceptance criterion by itself. Require SBC, posterior predictive checks, independent domain-shift tests and the locked test evaluation before a research claim.

## Model-selection considerations

- Compare a Gaussian summary likelihood, the MDN and (when supported by sufficient simulations) `sbi` NPE under the same priors, simulation products, feature definitions and data splits.
- If the target grid is sparse and discrete, use a discrete likelihood/classification or parameter-response task first. A smooth MDN posterior can interpolate between mass levels even if physics support is absent; smoothness is not evidence.
- If nuisance parameters vary but are omitted, the inferred posterior can be biased or spuriously narrow. Fix them transparently for a controlled method benchmark or include them in `theta` and cover them with simulations.
- If several rows per parent group occur, every view must stay in the same split. The MDN workflow uses nested grouped partitions; the default SBI wrapper is more restrictive because the standard `sbi` trainer may internally split rows.
- Avoid picking the model based only on training NLL, one coverage statistic, or one observed-data posterior width.

## Output contract

`mdn_report.json` records data checksums, split group fingerprints, fitted transforms, training configuration, selected epoch, validation metrics, test status and claim guards. `mdn_checkpoint.pt` stores model state and transforms; only load checkpoints from a trusted local source. `validation_posterior_samples.npz` contains validation samples and truth. `heldout_posterior_samples.npz` is emitted only when the caller explicitly evaluates the test partition once.

A successful software run is not scientific approval. Read `docs/VALIDATION_PROTOCOL.md` and `docs/DATASET_AND_MODEL_DECISION_PLAYBOOK.md` before interpreting a result.
