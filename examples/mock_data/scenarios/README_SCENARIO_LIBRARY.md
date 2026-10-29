# NUFIELD Synthetic Demo Scenario Library

> Synthetic mock data for UI demos, product walkthroughs, and test fixtures. Not observations or scientific results.

Generated scenario count: **11**

| Scenario | Validation fixture state | Parameter rows | Catalog rows | Folder |
|---|---|---:|---:|---|
| Baseline walkthrough | `PASS` | 250 | 150 | `baseline/` |
| Low toy Σmν | `PASS` | 250 | 150 | `low_neutrino/` |
| High toy Σmν | `PASS` | 250 | 150 | `high_neutrino/` |
| Noisy survey | `WARN` | 250 | 150 | `noisy_survey/` |
| Calibration offset | `WARN` | 250 | 150 | `calibration_bias/` |
| Held-out domain shift | `WARN` | 250 | 150 | `domain_shift/` |
| Coverage failure | `WARN` | 250 | 150 | `coverage_failure/` |
| Split overlap warning | `FAIL` | 250 | 150 | `split_leakage/` |
| Sparse dataset | `PASS` | 250 | 150 | `sparse_data/` |
| High-volume dashboard | `PASS` | 250 | 150 | `high_resolution/` |
| Import error states | `PASS` | 250 | 150 | `malformed_artifacts/` |

## What the scenarios exercise

- Baseline rendering and reproducible seed behavior.
- Lower and higher synthetic neutrino-mass input labels for comparison views.
- Increased observation noise and calibration-offset states.
- Held-out domain shift, coverage underflow, and deliberate split overlap.
- Low-count and high-volume table states.
- Importer edge-case descriptions for error-state walkthroughs.

Every scenario folder has per-artifact JSON files, a single `bundle.json`, optional CSVs, checksums, and a ZIP archive.
A synthetic `PASS` only checks expected fixture properties. It does not validate physics, model calibration, or data provenance.
