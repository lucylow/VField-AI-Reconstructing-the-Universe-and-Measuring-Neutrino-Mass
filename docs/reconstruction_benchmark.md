# Paired-field reconstruction benchmark

This prototype adds a reproducible path for comparing a transparent classical field baseline with the optional 3-D U-Net already in `nufield.reconstruction`. It is deliberately separate from the repository's parameter-emulator and neutrino-mass inference workflows: a reconstructed field is not itself a neutrino-mass constraint.

## Dataset contract

Input is a NumPy `.npz` opened with `allow_pickle=False`. Required arrays are a finite `initial_field` target and shape-matched `galaxy_overdensity`, both `(N,X,Y,Z)` on cubic grids, plus unique string `simulation_id`. Optional `group_id` identifies the independent split unit; when absent, each simulation ID is its own group. Optional weak-lensing maps use `(N,X,Y)` and are explicitly broadcast along Z. Optional `cmb_features` use `(N,F)` and become spatially constant conditioning channels. Optional `survey_mask` is a finite binary `(X,Y,Z)` or `(N,X,Y,Z)` mask. Mismatched units, coordinates or sampling geometry cannot be inferred by this loader: source metadata must document them. Non-finite values are rejected rather than silently imputed.

An adjacent `.json` sidecar is required and must declare `schema_version`, `data_type`, boolean `is_synthetic`, `license`, `array_units` (including each supplied field key), `coordinate_system`, and `coordinate_unit`; see `schemas/reconstruction_dataset_manifest.schema.json`. The loader validates these declarations and includes the sidecar in the report alongside the NPZ SHA-256. Units are recorded but no implicit conversions are attempted. Do not label mocks as observations. Existing toy-data `toy_projection` is not weak lensing and should not be passed as if it were.

## Run

Install the project and test tools (`python -m pip install -e '.[dev]'`); install the optional neural dependency with `python -m pip install -e '.[ml]'`. Make a paired NPZ with the contract above, edit `configs/reconstruction/smoke.yaml` to point to it, then run:

```bash
PYTHONPATH=src python -m nufield.reconstruction.cli --config configs/reconstruction/smoke.yaml
```

For a runnable software smoke test, generate an unmistakably synthetic fixture first:

```bash
PYTHONPATH=src python scripts/generate_reconstruction_mock.py --output outputs/reconstruction_mock.npz --n-sims 48 --grid-size 8 --seed 42
PYTHONPATH=src python -m nufield.reconstruction.cli --config configs/reconstruction/smoke.yaml
```

This artificial signal/noise mock is not a cosmological simulation or observation and must not support scientific conclusions. The `run_neural` flag is off for a baseline-only CPU smoke test. Turning it on fits the existing U-Net on galaxy maps only. If PyTorch is unavailable, the report says the model was skipped. The trainer writes atomic checkpoints containing model, optimizer, optional cosine scheduler, and RNG states; set `resume_from` to continue from that checkpoint with the same data, split, and training settings (the requested total epoch count may be increased). The current U-Net trainer does not support masked losses and is skipped for masked fields. Its default objective is field MSE; optional physics-informed terms are configurable and documented below.

## Method and artifacts

Data are split by parent group using `group_train_val_test_split`; overlap is asserted absent. Input means/scales and the ridge estimator are fitted on training rows only. The classical baseline is a pooled linear ridge mapping from voxel-aligned input channels to the target, not a Wiener reconstruction or physical forward model. Available modality runs include galaxy-only, galaxy+lensing, galaxy+CMB, and all available inputs; comparisons share the held-out rows. The report includes field RMSE, signed bias, shell-averaged relative power error, spatial cross-correlation, runtime, process high-water RSS, deterministic 5%/20% input-noise stress, an illustrative 5% gradient systematic, and a training-residual-scale check on test voxels. With a survey mask, the power metric zero-fills invalid voxels after centering on valid voxels: it is a windowed pseudo-spectrum, not a deconvolved spectrum, and masked-out values do not enter the comparison. Interpret the FFT shell metric physically only for an appropriate periodic, homogeneous cubic field. These stresses are probes, not validated survey error models. The residual diagnostic is not posterior calibration or credible-interval coverage.

The optional U-Net loss weights (`lambda_field`, `lambda_pk`, `lambda_gradient`, `lambda_mean`) use the existing `composite_reconstruction_loss`: `L = λ_field MSE(δ̂,δ) + λ_pk L_logPk + λ_gradient <|D_grid δ̂-D_grid δ|²> + λ_mean <(mean(δ̂)-mean(δ))²>`. `L_logPk` is the existing shell-averaged log-power Huber loss; when enabled it uses `box_size_mpc_h` in Mpc/h, which must match the dataset sidecar and requires `periodic: true`. Finite differences are per grid cell, and field/power/mean terms inherit their inputs' conventions. The code checks finite/nonnegative weights and required box metadata, but does not infer coefficient dimensions or validate physical appropriateness. Leave non-field terms at zero unless field definition, periodicity, units, and loss normalization justify them. `cosmological_parameters` is recorded as an assumption only; it is not passed as model evidence or used to derive a mass constraint.

`reconstruction_report.json`, `metrics.csv`, `predictions.npz`, `run_config.json`, and `reconstruction_manifest.json` are written under `output_dir`; when enabled, U-Net state and training/data metadata are saved as `unet3d_galaxy.pt`. Reports set `scientific_approval: false`. Re-run the same seed and data/config hash to reproduce the split and baseline. Neural determinism depends on the installed PyTorch/backend and hardware.

## Limitations / next steps

This code does not provide a validated hydrodynamic/N-body galaxy survey forward model, physical lensing/CMB projection, coordinate/unit conversion, missing-data imputation, observation likelihood, nuisance-parameter marginalization, emulator, or neutrino-mass posterior. It cannot establish simulation-to-observation domain validity. Next, implement U-Net validation-based early stopping and masked losses, add physical projection operators and validated simulation products, quantify systematics with source-calibrated priors, and connect the reconstructed-field outputs to the repository's separately audited emulator/inference APIs. Do not claim a neutrino-mass detection without those validations and independent scientific review.
