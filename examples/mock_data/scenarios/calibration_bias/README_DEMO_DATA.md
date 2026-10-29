# νField Synthetic Demo Data

> **Synthetic data warning.** These files are fabricated for frontend demos, UI testing, and software validation. They are not observational data, calibrated cosmology predictions, or scientific inference results.

## Generated profile

- Scenario: `calibration_bias` — Calibration offset
- Seed: `91125`
- Generator: `0.4.0`
- Run ID: `mock-calibration_bias-000163f5`
- Parameter rows: 250
- Catalog rows: 150
- Pseudo-chain rows: 200
- Experiment history rows: 16
- Validation state: `WARN` (fixture checks only)

## Files

- `bundle.json`: all generated artifacts in one object.
- `dataset.json` / `dataset.manifest.json`: importer-compatible metadata and provenance.
- `benchmark_report.json`: synthetic metric values, coverage placeholders, and split IDs.
- `parameter_samples.json`: generated sample rows with train/validation/test labels.
- `posterior_chains.json`: pseudo-traces for line charts; not real MCMC samples.
- `power_spectrum.json`, `cmb_summary.json`, `expansion_history.json`: chart-ready illustrative series.
- `survey_catalog.json`, `lensing_catalog.json`: generated object tables.
- `validation_report.json`, `data_quality_report.json`: controlled UI test states.
- `experiment_history.json`, `ablation_report.json`, `field_summary.json`: timeline, comparison, and field-slice fixtures.
- `checksums.json`: SHA-256 checksums for generated files (integrity, not validity).
- CSV files: flat exports of parameter rows, catalogs, spectra, and each pseudo-chain.

## Scientific boundaries

1. No real survey observations are included.
2. The generator does not call CLASS, CAMB, an N-body solver, or the project inference pipeline.
3. Names like `sigma8`, `omega_m`, `sum_mnu_ev`, and `coverage_diagnostic` are schema/display labels only.
4. A synthetic `PASS` checks fixture consistency only and says nothing about physical validity.
5. The generator is deterministic for a fixed seed and options at the row level; output timestamps differ between runs.

## Rebuild

```bash
python scripts/generate_mock_data.py --profile baseline --seed 90721 --output examples/mock_data/baseline
```
