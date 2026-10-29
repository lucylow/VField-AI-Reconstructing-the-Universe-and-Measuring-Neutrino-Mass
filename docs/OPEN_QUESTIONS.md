# Open technical questions before the first scientific result

These questions are intentionally unresolved; making them explicit is better than hiding assumptions
inside code.

## Target observable and likelihood

1. Is the initial study about BAO reconstruction, broad-field cosmological parameters, or neutrino-mass
   sensitivity? The best observable and scale cuts differ by target.
2. Will inference condition on a galaxy catalog, a gridded density field, or summary statistics?
3. Is the scientific target the total matter field or the cold-dark-matter-plus-baryon field?
4. What exact likelihood baseline will be the comparison point, including window functions and covariance?

## Neutrino model

1. Which hierarchy / mass parameterization is used?
2. Does the simulation represent neutrino perturbations linearly, with particles, or via an emulator?
3. What scales and redshifts are trustworthy for the selected implementation?
4. Which cosmological parameters are varied jointly with the mass, and what prior support is defensible?
5. What reference calculation anchors errors in the nonlinear neutrino response?

## Simulation design

1. Are phases paired across cosmologies for response estimation?
2. Are parameters drawn by a Latin hypercube, a prior sample, or a sequential design?
3. Is the test set an interpolation set, an extrapolation set, or both?
4. What simulation count is needed for calibrated posterior uncertainty at the required precision?
5. How are nuisance variations independent of cosmology to avoid learnable confounding?

## Multimodal analysis

1. How are the galaxy and lensing footprints aligned?
2. What redshift kernel maps the 3-D matter field to the lensing observable?
3. How is shared sample variance represented in joint simulation?
4. Which CMB product or likelihood is used and how is double-counting prevented?
5. How are modality ablations compared at the same prior and simulation truth?

## Scientific governance

1. Who reviews the physics assumptions independently?
2. How are final test sets frozen and archived?
3. How are null results and failed methods reported?
4. What data and code licensing constraints apply to the eventual release?
5. What threshold warrants a claim of scientific improvement rather than an engineering improvement?
