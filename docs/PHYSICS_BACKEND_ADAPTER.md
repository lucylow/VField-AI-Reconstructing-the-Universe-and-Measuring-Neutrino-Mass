# Physics backend adapter design (future work)

## Required contract

A future `PhysicsBackend` interface should expose:

- `metadata()`: code/version, approximation, units, neutrino treatment, valid parameter domain;
- `linear_power(params, z, k)`: documented units and redshift convention;
- `simulate(params, seed, config)`: field/catalog products with known realization metadata;
- `validate_against(reference)`: numerical comparison suite with stated tolerances and scale range.

## CLASS/CAMB integration checklist

1. Record the exact solver version and build options.
2. Set `N_eff`, `T_cmb`, neutrino hierarchy/degeneracy assumptions, and `sum_mnu` consistently.
3. Document whether returned power is in physical `1/Mpc` or `h/Mpc` coordinates and units.
4. Confirm whether spectra are total matter, CDM+baryon, or another species combination.
5. Compare an independent parameter point against a published/reference result.
6. Add tests for parameter monotonicity only where theory predicts it and under explicit fixed-parameter conditions.
7. Cache by normalized parameter dictionary, solver version, redshift, and k-grid hash.
8. Fail loudly if the requested parameter values are outside backend support.

## Massive-neutrino nonlinear modeling checklist

A linear transfer function alone does not constitute a nonlinear massive-neutrino forward model. Choose a documented simulator/emulator, state its neutrino treatment, and validate nonlinear matter and halo statistics. Check the response to `sum_mnu` in the actual training domain; do not use non-neutrino mocks as if they supplied a neutrino-varying label. Track baryonic effects and scale cuts when comparing with observations.

## Survey observation operator checklist

A DESI-like catalog requires angular/radial completeness, target selection, redshift selection, redshift-space distortions, survey window, weights, and covariance conventions. Document the tracer sample and redshift bin. For lensing, specify source redshift distribution, shear/convergence convention, mask, shape noise, calibration, and angular resolution. For CMB-derived data, use the public likelihood/product convention and avoid double-counting shared information.

## Scientific review gates

Before any public parameter constraint: one independent cosmologist reviews the forward model; an independent implementation or likelihood validates the baseline; the simulation suite supports the quoted parameter domain; calibration passes with uncertainty; and the analysis configuration is frozen before the final test evaluation.
