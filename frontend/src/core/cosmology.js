/** Small, explicitly scoped concept helpers. These are not an expansion-history solver. */
export function scaleFactorFromRedshift(redshift) {
  const z = Number(redshift);
  if (!Number.isFinite(z) || z < 0) throw new RangeError("Redshift must be a finite, non-negative number.");
  return 1 / (1 + z);
}

export function redshiftFromScaleFactor(scaleFactor) {
  const a = Number(scaleFactor);
  if (!Number.isFinite(a) || a <= 0 || a > 1) throw new RangeError("Scale factor must be finite and in (0, 1].");
  return (1 / a) - 1;
}

export function formatEpoch(redshift) {
  const z = Number(redshift);
  if (!Number.isFinite(z) || z < 0) throw new RangeError("Redshift must be finite and non-negative.");
  if (z < 0.05) return "Present day";
  if (z < 1) return "Recent cosmic epoch";
  if (z < 6) return "Earlier universe";
  return "High-redshift era";
}
