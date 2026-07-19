import type { PointCloud } from '../types';
import { getConfig } from '../config';

/**
 * Give the reconstructed asset real-world scale so it drops into an aquascape
 * scene at the right size.
 *
 * Two modes (see config.scale.mode):
 *  - 'turntable' : the user inputs the turntable disc diameter. We measure the
 *    cloud's current horizontal extent (which spans roughly the disc) and scale
 *    so it matches. Coarse but zero-effort.
 *  - 'reference' : the user taps two points on a frame across a known length.
 *    The caller converts that to a metres-per-unit factor and passes it here.
 */

export interface ScaleResult {
  metersPerUnit: number;
  appliedTo: PointCloud;
}

/** Scale a cloud in place by a known metres-per-current-unit factor. */
export function applyMetricScale(cloud: PointCloud, metersPerUnit: number): PointCloud {
  const p = cloud.positions;
  for (let i = 0; i < p.length; i++) p[i] *= metersPerUnit;
  return cloud;
}

/** Horizontal (XZ) extent of the cloud in current units. */
export function horizontalExtent(cloud: PointCloud): number {
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  const p = cloud.positions;
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i];
    const z = p[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  return Math.max(maxX - minX, maxZ - minZ);
}

/** Auto-calibrate from the configured turntable diameter. */
export function calibrateFromTurntable(cloud: PointCloud): ScaleResult {
  const cfg = getConfig();
  const extent = horizontalExtent(cloud) || 1;
  const metersPerUnit = cfg.scale.turntableDiameterM / extent;
  applyMetricScale(cloud, metersPerUnit);
  return { metersPerUnit, appliedTo: cloud };
}

/**
 * Calibrate from a tapped reference: `pixelLength` is the on-screen distance the
 * user marked, `frameDepthUnits` is the cloud-space depth at that location, and
 * `focalPx` the camera focal. Converts to metres-per-unit via similar triangles.
 */
export function calibrateFromReference(
  cloud: PointCloud,
  knownLengthM: number,
  pixelLength: number,
  frameDepthUnits: number,
  focalPx: number,
): ScaleResult {
  // world length of the tapped span, in current cloud units:
  const unitsLength = (pixelLength / focalPx) * frameDepthUnits;
  const metersPerUnit = unitsLength > 0 ? knownLengthM / unitsLength : 1;
  applyMetricScale(cloud, metersPerUnit);
  return { metersPerUnit, appliedTo: cloud };
}
