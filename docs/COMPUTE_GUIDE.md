# Compute and scaling guide

## Start on a laptop

Begin with `grid_size=8` or `12` and tens to hundreds of realizations. This is enough to verify schemas,
feature extraction, split logic, and report generation. The current dataset stores all voxel grids in one
NPZ archive, so memory and disk use grow as `N * grid_size^3`. Do not jump directly to 512^3 fields.

Approximate array memory before compression, for one float32 scalar grid:

- 8^3: 2 KiB
- 32^3: 128 KiB
- 128^3: 8 MiB
- 512^3: 512 MiB

A full dataset may store multiple fields, particle data, parameter labels, velocity fields, lensing maps,
and temporary FFT arrays. A 512^3 complex128 FFT workspace can use multiple GiB. Profile peak memory
with realistic arrays before allocating cluster jobs. These are raw array sizes, not measured runtime
estimates.

## Scaling plan

1. Add chunked on-disk storage, e.g. Zarr/HDF5, only after the data schema is stable.
2. Use independent per-realization jobs with one manifest per task.
3. Store raw simulator outputs separately from derived features.
4. Compute feature vectors in streaming mode where possible.
5. Avoid materializing all 3-D fields in RAM for training a posterior over low-dimensional parameters.
6. Keep the final evaluation set independent of all exploratory hyperparameter tuning.

## GPU guidance

The optional 3-D U-Net is not automatically the best use of GPU time. A low-dimensional posterior
estimator on simulation-derived summaries may need far less memory. Compare wall-clock, VRAM, inference
latency, and calibration before selecting an architecture. A neural approach should be evaluated at
matched simulator cost and equivalent information, not only at matched training epochs.

## High-performance computing

Before running on an institutional cluster, make a one-job pilot. Record queue time separately from
compute time; measure CPU, GPU, memory, scratch I/O and output size. Use checkpoint/restart only when the
training loop actually persists optimizer/model state. Do not state a fixed CPU-hour estimate until the
selected simulator and mesh resolution have been profiled on the target hardware.
