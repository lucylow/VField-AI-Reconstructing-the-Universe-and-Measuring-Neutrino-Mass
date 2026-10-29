# νField Dataset Selection and Acquisition Plan

**Version:** 2.0 · **Status:** implementation plan / requires source-level verification

## 1. Decision in one sentence

Use the **Quijote massive-neutrino subsets** as the first controlled training/validation source because the selected product must explicitly vary total neutrino mass; use **Quijote/Sancho mock catalogs and DESI DR1 mocks** to add tracer and survey realism; treat **DESI DR1 observed data** as a late-stage application, and use **CAMELS** as an astrophysical/domain-shift check rather than assuming it is an Mν-labelled training set.

This is a source-selection recommendation, not a claim that any raw product is ready to train a model without inspection. Every chosen subset must be recorded in a dataset card, and its exact parameter design, field definition, initial-condition method, snapshot redshift, phase relationships, units, and access terms must be verified from the release documentation.

## 2. What makes a dataset appropriate for neutrino inference?

A dataset for an Mν posterior must satisfy all of these conditions:

1. **Target identifiability:** the selected records explicitly vary Mν/Σmν, and the units and mass convention are documented. A catalog with a single fixed neutrino mass cannot provide supervised mass sensitivity.
2. **Coverage:** the selected mass values and correlated cosmological parameters cover a defined prior. A few discrete cases can test finite-difference sensitivity, but are not automatically enough to train a smooth amortized posterior.
3. **Independent simulations:** train/validation/test splits must respect parent simulation, phase, cosmology family, box, and derived-crop relationships.
4. **Matched data products:** all target fields use compatible redshifts, grid/assignment methods, smoothing, units, volume, and boundary conditions. Any conversions are versioned and reproducible.
5. **Known forward modeling:** the code path used to create the training observations includes the physical effects to which the scientific claim refers. Toy response functions cannot support physical Mν constraints.
6. **Domain relevance:** training simulations are not a substitute for a survey observation model. Galaxy bias/HOD, redshift-space distortions, masks, completeness, shot noise, selection and the survey window need to be addressed for observed-catalog analysis.
7. **Legal/provenance clarity:** access permission, licensing or use terms, acknowledgment text, checksums, source URLs, release versions, and transform history are retained.

## 3. Curated source matrix

| Resource | Recommended role | Why it helps | Important caveat |
|---|---|---|---|
| Quijote Mν suites | Core physical sensitivity/training | Official docs describe massive-neutrino simulations and specific Mnu-oriented subsets such as `Mnu_p`, `Mnu_pp`, `Mnu_ppp`, and `nwLH`. | Exact parameter design varies by subset; inspect the selected product. Large N-body products may need HPC/storage. |
| Quijote Sancho galaxy mocks | Tracer/HOD/RSD robustness | Gives redshift-space mock catalog products and associated clustering summaries in a mock framework. | Check which Mν values, HODs and cosmologies the exact release includes; do not infer broad coverage from the suite name. |
| DESI DR1 EZmocks | Catalog pipeline/covariance and survey-analysis checks | DR1 docs describe 1,000 EZmock realizations for covariance/analysis validation. | Those are not 1,000 independent neutrino-mass cosmologies. |
| DESI DR1 AbacusSummit mock products | Higher-resolution external validation | DR1 docs list 25 AbacusSummit mock realizations; useful for a separate validation layer when the product is compatible. | Small realization count and selected cosmology coverage are not a continuous Mν grid. |
| DESI DR1 observed LSS catalogs | Late-stage observational application | Public clustering catalogs, weights, random catalogs, and released analysis products give real survey context. | Observations have no known true θ per object; cannot be supervised target labels. Must compare to established analysis/likelihood products. |
| AbacusSummit | Selected cross-suite tests | Documented cosmology grid and selected massive-neutrino reference cosmologies are useful for external simulation checks. | Neutrino-mass coverage is limited/discrete in named cases; outputs and resolution differ across simulations. |
| CAMELS | Feedback/baryonic/domain shift | Provides matched simulation families varying cosmological and astrophysical parameters across multiple simulation models. | Standard suites do not automatically provide Mν-varying labels; data volume is large. Use as robustness testing unless exact subset says otherwise. |
| JAX-FLI | Reference software, not a data release | Differentiable cosmological forward model spans initial conditions, LPT/PM evolution, light-cone painting and lensing workflows. | It is software; importing it does not itself make the νField physical model validated. |

## 4. Recommended order of acquisition

### Phase A — minimal usable product

Download **one explicitly documented Quijote Mν product** with manageable summary/map outputs, rather than the largest raw particle dump. Before downloading, write a card containing `suite_name`, product/release, `Mnu` parameter values, redshift, field/grid definition, `L_box`, mass convention, initial-condition method, phase or seed IDs, checksum, citation, use terms, storage estimate, and key mapping.

Do not begin with the broadest dataset. Start with the smallest product that can answer: does the neural method recover a deliberately varied Mν parameter under held-out simulations? The first experiment can use a lower-resolution/density-grid representation, but that is an engineering proxy whose fidelity must be checked against the authoritative product.

### Phase B — physics and resolution checks

Choose a subset with independent initial phases and, where possible, paired phases or matched cosmologies. Use separate splits for:

- **interpolation:** held-out random realizations within the training parameter design;
- **phase generalization:** new phases at trained cosmologies;
- **cosmology generalization:** held-out cosmological parameter points;
- **cross-suite generalization:** compatible AbacusSummit / other validated simulations;
- **observational transfer:** mock catalog data with the relevant survey treatment.

A split by voxel/crop alone is invalid if multiple crops derive from the same parent simulation. The group ID must encode that ancestry.

### Phase C — survey realism

Use DESI DR1 LSS mocks after the simple field pipeline has passed its own ground-truth checks. Record the exact tracer type (for example, LRG/ELG/BGS/QSO), release/catalog version, redshift bin, clustering-catalog file, random catalog, weights, mask/selection construction, and covariance/likelihood product. Read the official data-model notes for that exact version; never assume all catalog versions share the same schema.

The 1,000 EZmock realizations can be useful for covariance-related checks and repeated pipeline tests, but don't count them as a broad independent Mν simulation grid. DR1 docs also list 25 AbacusSummit mock realizations; treat those according to their available cosmologies and products.

### Phase D — external model mismatch

CAMELS is a deliberately different simulation family for testing whether results depend on one feedback prescription. Match output definitions as closely as possible, but explicitly document mismatches in resolution, baryonic physics, box size, and parameter support. Do not train a mass posterior from CAMELS unless the chosen subset explicitly varies Mν with a defensible grid.

## 5. Raw-field versus summary-statistic dataset choice

A project focused on field-level inference ideally requires a 3-D density/matter field and known θ for each independent simulation. Halo catalogs can be painted into meshes, but this introduces a mass-assignment method, smoothing convention, density definition, and possible tracer bias. Galaxy catalogs add realism but also bring nuisance parameters and selection effects.

Maintain two tracks:

- **Track 1 — fast benchmark:** a matched density-field or lower-resolution summary-field dataset plus power-spectrum summaries. This makes it possible to test data loading, inference, and calibration cheaply.
- **Track 2 — research benchmark:** a neutrino-aware, nonlinear, matched simulation suite with a documented mapping from particles/halos to fields and independent realizations. This track is required for scientific claims.

The field model and the power-spectrum baseline must derive from the same underlying mock realization. Comparing different cosmology sets, priors, noise assumptions, or masks confounds architecture and data effects.

## 6. Dataset card and local schemas

Every derived collection should have a machine-readable YAML/JSON card. Use `configs/datasets/quijote_mnu.template.yaml` and fill every `REPLACE`/`VERIFY` field from source documentation. Do not change `status` to `simulation_validated` merely because arrays load. The status ladder is:

- `unverified`: card drafted, source product not yet checked;
- `metadata_checked`: key docs, units, parameters and access verified;
- `arrays_checked`: schema/numeric invariants pass;
- `simulation_validated`: a defined validation protocol passes against independent trusted results;
- `research_approved`: a human research review accepts the model and inference claim.

The software intentionally cannot set `research_approved` automatically.

## 7. Go/no-go thresholds before training

**No-go** if any target parameter is constant; Mν units or parameter definition are unresolved; parent IDs are absent; train/test leakage cannot be ruled out; simulation labels are unknown; or the exact file product does not correspond to the intended source documentation.

**Go for a smoke test** if arrays are finite and well-shaped and the exact model input/target mappings are declared. This only checks software plumbing.

**Go for a research benchmark** only once the simulator family is suitable for the physical question, parameter support is mapped, baselines and held-out groups are fixed, nuisance parameters are included or explicitly scoped out, posterior calibration is measured on independent simulations, and validation against a trusted suite or observable is documented.

## 8. Storage and compute planning

Estimate storage before pulling 3-D grids: `N_sim × N_voxels × bytes_per_voxel`, plus labels, multiple redshifts, channels, initial conditions, intermediate checkpoints, and backups. A 512³ float32 density field is approximately 512 MiB before compression; 1,000 such fields would be about 500 GiB decimal before metadata/multiple channels. Particle snapshots can be vastly larger. Prefer streaming selected fields and summaries over keeping all raw data in RAM.

Keep a raw-data cache read-only. Store derived data separately with a manifest linking each derived artifact to original product checksums, transform/config versions, random seed, and software commit. Never publish an undocumented processed grid detached from source metadata.

## 9. Upstream documentation links

- Quijote main docs: https://quijote-simulations.readthedocs.io/en/latest/
- Quijote massive-neutrino subset documentation: https://quijote-simulations.readthedocs.io/en/latest/Mnu.html
- Quijote suite structure and release news: https://quijote-simulations.readthedocs.io/en/latest/news.html
- DESI DR1 release and LSS/mocks documentation: https://data.desi.lbl.gov/doc/releases/dr1/
- DESI access guidance: https://data.desi.lbl.gov/doc/access/
- AbacusSummit simulation/cosmology tables: https://abacussummit.readthedocs.io/en/latest/cosmologies.html and https://abacussummit.readthedocs.io/en/latest/simulations.html
- CAMELS data/docs: https://www.camel-simulations.org/data and https://camels.readthedocs.io/en/latest/
- JAX-FLI docs: https://jax-fli.readthedocs.io/en/stable/

## 10. Decision log

| Decision | Rationale | Revisit condition |
|---|---|---|
| Quijote Mν product is the primary core | Explicit massive-neutrino suite is essential to target identifiability. | Selected product lacks adequate mass grid, output fields, or verified labels. |
| Power-spectrum baseline is mandatory | Provides a transparent comparator and quick debug signal. | Never remove; add more baselines instead. |
| DESI is a late-stage application | Survey complexity and no per-observation ground truth require pre-validation. | Only after blind mocks and likelihood checks pass. |
| CAMELS is a robustness suite, not primary Mν supervision | Feedback variation is useful but not equivalent to neutrino mass variation. | Exact suite with validated Mν variation becomes available and appropriate. |
| No automatic downloading by the starter | Prevents large, unverified, access/usage-ambiguous downloads. | Add source-specific downloader only with documented license, expected size, resume, checksums and dry-run. |
