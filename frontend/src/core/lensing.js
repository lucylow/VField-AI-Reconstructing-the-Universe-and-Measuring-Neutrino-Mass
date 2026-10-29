/**
 * Dimensionless point-mass lens equation utilities for the educational canvas.
 * All coordinates are angular-like display units, not astrophysical distances.
 */
export function pointMassLensImages(beta, thetaE) {
  const sourceOffset = Number(beta);
  const einsteinAngle = Number(thetaE);
  if (!Number.isFinite(sourceOffset) || sourceOffset < 0) {
    throw new RangeError("Source offset beta must be finite and non-negative.");
  }
  if (!Number.isFinite(einsteinAngle) || einsteinAngle < 0) {
    throw new RangeError("Einstein angle must be finite and non-negative.");
  }
  if (einsteinAngle === 0) return { positive: sourceOffset, negative: null };
  const discriminant = Math.sqrt(sourceOffset ** 2 + 4 * einsteinAngle ** 2);
  return {
    positive: 0.5 * (sourceOffset + discriminant),
    negative: 0.5 * (sourceOffset - discriminant),
  };
}

export function pointMassLensResidual(theta, beta, thetaE) {
  const imageOffset = Number(theta);
  const sourceOffset = Number(beta);
  const einsteinAngle = Number(thetaE);
  if (![imageOffset, sourceOffset, einsteinAngle].every(Number.isFinite) || imageOffset === 0 || einsteinAngle < 0) {
    throw new RangeError("Lens residual requires finite inputs, non-zero image offset, and non-negative thetaE.");
  }
  return imageOffset - (einsteinAngle ** 2 / imageOffset) - sourceOffset;
}
