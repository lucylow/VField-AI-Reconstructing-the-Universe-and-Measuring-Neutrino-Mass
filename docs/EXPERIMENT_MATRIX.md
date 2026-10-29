# Recommended Experimental Matrix

Each row is a separate experiment and should be run on the same locked split wherever comparison is intended.

| ID | Dataset | Inputs | Model | Primary question | Non-negotiable validation |
|---|---|---|---|---|---|
| E00 | synthetic toy | toy P(k) | kernel posterior | does software run? | toy-only warning; expected behavior tests |
| E01 | selected Quijote Mν subset | P(k) bins | kernel / conventional likelihood | are labels/support wired correctly? | train-only preprocessing, known truth, mock coverage |
| E02 | same Quijote simulations | P(k) bins | MLP + NPE | neural summary baseline quality? | SBC, held-out phases/cosmologies |
| E03 | same simulations | 3-D fields | 3-D CNN + NPE | does field representation add info? | matching observations, cost, calibration |
| E04 | same simulations | fields + P(k) | hybrid CNN + NPE | do explicit and learned summaries complement? | ablations; ensure no duplicated/incorrectly weighted likelihood |
| E05 | multiple independent seeds | same as E03/E04 | top models | seed variance and model selection stability? | ≥3 seeds where budget allows; confidence intervals |
| E06 | Sancho mocks or compatible galaxy mocks | catalog-to-grid + P(k) | strongest validated model | what is lost after tracer construction/RSD? | HOD/RSD/mask/selection stress tests |
| E07 | DESI DR1 mocks | catalog estimator / matched summaries | conventional + learned | survey pipeline robustness? | window, weights, randoms, correct covariance products |
| E08 | CAMELS or other simulation family | matched available statistics | frozen model / adaptation study | does model shortcut one simulation family? | explicit field and unit harmonization; domain shift report |
| E09 | DESI DR1 observed catalog | only after E01-E08 pass | locked posterior/model | what does application infer under assumptions? | compare published/standard analyses; no truth-coverage claim on one real sky |
| E10 | lensing simulation + galaxy field | matched joint observables | mask-aware multimodal NPE | does complementary field improve calibrated constraints? | single-modal ablations, cross-covariance, shared priors, masks |

## Recommended plots per model/data pair

1. Posterior marginals and joint contours for every target parameter.
2. SBC rank histograms per parameter and seed.
3. Empirical coverage versus nominal credibility with finite-simulation uncertainty intervals.
4. Posterior interval width versus bias/coverage; never interval width alone.
5. True versus posterior median, colored by simulation family or mass bin.
6. Out-of-support and nearest-training-neighbor diagnostic.
7. P(k) residual and field-level posterior-predictive diagnostics.
8. Accuracy/calibration versus wall time, simulator count, CPU/GPU-hours, memory.
9. A systematic-sensitivity heatmap covering masking, noise, bias, redshift-space and simulator-family perturbations.
10. Failure case gallery with original field, reconstructed field, residual and notes.

## Minimum report table

| run_id | dataset_card_hash | simulator_commit | git_sha | seed | split_hash | model_name | prior_hash | preprocessing_hash | n_simulations | n_trainable_params | wall_s | peak_memory | RMSE_Mnu | coverage68_Mnu | coverage95_Mnu | mean_width68_Mnu | SBC_test_id | notes |
|---|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

Use null for unmeasured metrics rather than guessed values. Keep the full posterior samples for inspection and recomputation.
