import type { CameraPose, Keyframe } from '../types';
import { getConfig } from '../config';

/**
 * Derive camera poses from the TURNTABLE assumption: the object rotates (or the
 * phone circles it) at a roughly constant angular step. This deliberately
 * avoids full Structure-from-Motion — the browser bottleneck — and instead
 * lays cameras on a circle looking at the object centre.
 *
 * Conventions (computer-vision, right-handed):
 *   - World origin = object centre.
 *   - Camera looks down its +Z axis; +X right, +Y down (image convention).
 *   - `matrix` is a column-major 4x4 camera-to-world transform.
 *
 * Camera distance D is chosen so that a pixel at the *midpoint* of the depth
 * band maps to the world origin, keeping the back-projected cloud centred and
 * coherent regardless of the (unknown) true radius. Real-world size is applied
 * later by scale calibration.
 */

const DEG2RAD = Math.PI / 180;

export function deriveTurntablePoses(
  keyframes: Keyframe[],
  opts?: {
    horizontalFovDeg?: number;
    elevationDeg?: number;
    /** Explicit per-frame angles (radians). When omitted, a constant step is
     *  used. `refineTurntableAngles` produces these from feature tracking. */
    thetasRad?: number[];
  },
): CameraPose[] {
  const cfg = getConfig();
  const n = keyframes.length;
  const fovDeg = opts?.horizontalFovDeg ?? 60;
  const elevationDeg = opts?.elevationDeg ?? 12; // slight downward look

  const D = (cfg.depth.nearMeters + cfg.depth.farMeters) / 2;
  const sweep = cfg.capture.totalSweepDeg * DEG2RAD;
  const elev = elevationDeg * DEG2RAD;

  const poses: CameraPose[] = [];
  for (let i = 0; i < n; i++) {
    const kf = keyframes[i];
    const theta = opts?.thetasRad ? opts.thetasRad[i] : n > 1 ? (sweep * i) / (n - 1) : 0;

    // Camera position on a circle of radius D, lifted by elevation.
    const cosE = Math.cos(elev);
    const camPos: Vec3 = [
      D * cosE * Math.sin(theta),
      -D * Math.sin(elev), // -Y is up in our image-Y-down world
      D * cosE * Math.cos(theta),
    ];

    // Viewing direction: from camera toward origin => camera +Z axis.
    const zAxis = normalize(sub([0, 0, 0], camPos)); // forward
    // World up is -Y (since +Y is down). Build orthonormal basis.
    const worldUp: Vec3 = [0, -1, 0];
    let xAxis = normalize(cross(worldUp, zAxis)); // right
    if (!isFinite(xAxis[0])) xAxis = [1, 0, 0];
    const yAxis = cross(zAxis, xAxis); // down

    // Column-major camera-to-world: columns are basis vectors + translation.
    const m = new Float32Array(16);
    m[0] = xAxis[0]; m[1] = xAxis[1]; m[2] = xAxis[2]; m[3] = 0;
    m[4] = yAxis[0]; m[5] = yAxis[1]; m[6] = yAxis[2]; m[7] = 0;
    m[8] = zAxis[0]; m[9] = zAxis[1]; m[10] = zAxis[2]; m[11] = 0;
    m[12] = camPos[0]; m[13] = camPos[1]; m[14] = camPos[2]; m[15] = 1;

    const focalPx = kf.width / 2 / Math.tan((fovDeg * DEG2RAD) / 2);
    poses.push({
      index: i,
      matrix: m,
      focalPx,
      cx: kf.width / 2,
      cy: kf.height / 2,
    });
  }
  return poses;
}

// ---- tiny vec3 helpers -----------------------------------------------------
type Vec3 = [number, number, number];
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function normalize(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
