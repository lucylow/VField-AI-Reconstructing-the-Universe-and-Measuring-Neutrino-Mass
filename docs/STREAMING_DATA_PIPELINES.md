# Streaming data intake and field-pipeline contracts

## Why this exists

Simulation products may be large enough that accidental full-file materialization is costly. The repository has schema/provenance helpers and generic HDF5/Zarr chunk readers, but it does not claim to understand every upstream product's object naming. Each scientific product needs an explicit adapter that maps its documented arrays into the νField contract.

## Intake sequence

1. **Inventory files:** record product release, file count, file names, sizes and checksums. Keep data outside source control unless redistribution terms permit inclusion.
2. **Profile container structure:** run the HDF5 profiler to list groups/datasets, shape, dtype and chunks. Treat the profile as descriptive only; an array named `PartType1/Coordinates` may represent different units or particles depending on source product and snapshot.
3. **Write explicit mapping:** identify coordinates, box size, cosmological parameters, redshift/snapshot and particle or field semantics. Never assume an undocumented field is Mν or infer units from magnitude.
4. **Select a small deterministic subset:** verify shapes, finite values, coordinate range and matching IDs; compare basic summary statistics against product documentation or a reference implementation.
5. **Transform in chunks:** use `ChunkedArraySource` for generic HDF5/Zarr array slicing or the product-specific particle pipeline for periodic particle coordinates. Persist exact config and source hash.
6. **Inspect outputs:** ensure the field has the intended grid size, mass-assignment scheme, boundary conditions, units and density convention; confirm no non-finite or empty cells beyond what the contract expects.
7. **Generate summaries:** produce features only after the field contract passes. Maintain a link from each feature row to simulation ID, parent group ID, snapshot and source checksum.

## Available commands

```bash
nufield-profile-hdf5 --help
nufield-particle-pipeline --help
nufield-field-pipeline --help
nufield-field-features --help
nufield-product-review --help
nufield-dataset-plan --help
```

CLI flags may vary slightly by module version; use each command's `--help` before running against a real file. Start with one small file and one simulation group. Do not batch a release until output arrays and metadata have been checked manually.

## Particle-field assumptions

The current periodic particle pipeline is a numerical utility for mapped particle coordinates. It uses periodic box geometry and Cloud-In-Cell deposition. It is not an official Quijote parser and does not infer file schema, cosmology metadata or neutrino mass. A product-specific adapter must pass the correct particle species and physical units. Do not merge gas, CDM and neutrino particles without a deliberate physical convention.

## Gridded field input

The generic field-batch pipeline accepts a user-selected already-gridded periodic field array from HDF5 and uses a saved, explicit mapping. It does not itself convert raw DESI catalogs to fields or make masked survey data periodic. Survey geometry and random catalogs require a separate observation/estimator pipeline.

## Provenance contract

Store, at minimum:

- exact source URL/access method and product release;
- file checksum(s), source size, acquisition date, license/terms;
- schema mapping file and mapping version;
- cosmological parameters with units and definitions;
- redshift/snapshot, box length, grid resolution and axis convention;
- particle species, mass assignment and smoothing/deconvolution choices;
- IDs/group IDs, split plan and feature extraction version;
- software version, random seed and output checksums.

No data adapter is considered scientifically verified just because its output has a correct tensor shape.
