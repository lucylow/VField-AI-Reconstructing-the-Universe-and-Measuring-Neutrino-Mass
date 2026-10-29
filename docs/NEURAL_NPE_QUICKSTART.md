# Neural NPE quickstart

## Install optional stack

```bash
python -m pip install -e '.[sbi,dev]'
```

The `sbi` extra installs the library and PyTorch. GPU/CUDA installation may need a platform-specific PyTorch wheel.

## Run against the existing toy dataset (software smoke test only)

The existing `examples/sample_output/dataset.npz` includes toy fields/features and six toy parameters. It is deliberately blocked unless you opt into a smoke test:

```bash
nufield-train-npe \
  --dataset examples/sample_output/dataset.npz \
  --field-key galaxy_overdensity \
  --spectrum-key features \
  --theta-key theta \
  --parameter-names sum_mnu_ev,omega_m,sigma8,h,n_s,bias \
  --prior-low 0.06,0.27,0.74,0.60,0.92,0.8 \
  --prior-high 0.30,0.36,0.88,0.75,1.01,1.8 \
  --output-dir outputs/npe-toy \
  --epochs 20 \
  --n-eval 3 \
  --posterior-samples 250 \
  --allow-toy
```

This is only a software/shape/training smoke test. Results from the bundled toy simulator have **no physical interpretation** and must not be reported as neutrino-mass constraints.

For a real source product, first build and pass a dataset card, map the exact field and parameter arrays, verify IDs, and confirm that the chosen source subset explicitly spans Mν. Replace the prior bounds with predeclared scientific bounds and use group IDs that prevent phase/crop/parent-simulation leakage. Never use a test set repeatedly for hyperparameter tuning.

## Artifacts

- `npe_report.json`: split, architecture and posterior-metric summary plus caveats;
- `heldout_posterior_samples.npz`: test posterior samples and true simulated parameters;
- `preprocessing_state.npz`: train-fitted normalization arrays;
- `embedding_state.pt`: embedding state dictionary and shape metadata.

The embedding checkpoint is not a fully serialized SBI posterior. It records the feature encoder weights; the exact prior, estimator architecture, SBI version and density-estimator checkpoint must also be retained for faithful posterior reconstruction. The project deliberately avoids pickling an arbitrary inference object as a portability shortcut.
