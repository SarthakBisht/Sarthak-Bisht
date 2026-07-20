import type { CameraPose, DepthMap, Keyframe, Matte } from '../types';
import { getConfig } from '../config';

/** Result of back-projecting a single frame. */
export interface FramePoints {
  positions: Float32Array; // count*3
  colors: Float32Array; // count*3 (0..1)
  confidence: Float32Array; // count
  count: number;
}

/**
 * Back-project one frame's masked depth into world-space points.
 *
 * metric distance = near + depth01 * (far - near)   (depth01: 1 = far)
 * camera-space    = ((u-cx)/f * d, (v-cy)/f * d, d) (camera looks down +Z)
 * world           = cam2world * cameraPoint
 *
 * Confidence falls off near the mask edge (edge pixels are where matte error and
 * depth-edge flyers concentrate); low-confidence points feed Tier-1
 * densification of concavities/crevices.
 */
export function backprojectFrame(
  kf: Keyframe,
  depth: DepthMap,
  matte: Matte,
  pose: CameraPose,
): FramePoints {
  const cfg = getConfig();
  const w = kf.width;
  const h = kf.height;
  const near = cfg.depth.nearMeters;
  const far = cfg.depth.farMeters;
  const f = pose.focalPx;
  const { cx, cy } = pose;
  const m = pose.matrix;

  const dist = chamferEdgeDistance(matte.alpha, w, h);
  const falloff = Math.max(1, cfg.fusion.edgeConfidenceFalloffPx);
  const keepProb = Math.min(1, Math.max(0, cfg.fusion.pointDensity));

  // Count foreground pixels first to size arrays.
  let fg = 0;
  for (let i = 0; i < matte.alpha.length; i++) if (matte.alpha[i] === 255) fg++;
  const cap = Math.ceil(fg * keepProb);
  const positions = new Float32Array(cap * 3);
  const colors = new Float32Array(cap * 3);
  const confidence = new Float32Array(cap);

  let n = 0;
  let seed = (pose.index + 1) * 2654435761; // deterministic subsample RNG
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (matte.alpha[idx] !== 255) continue;
      if (keepProb < 1 && rand() > keepProb) continue;
      if (n >= cap) break;

      const d01 = depth.depth[idx];
      // Drop far points — the background / turntable surface sits near the far
      // plane and single-view depth places it there.
      if (d01 > cfg.fusion.farCull01) continue;
      // Drop low-confidence edge flyers (near the silhouette boundary).
      const conf = Math.min(1, dist[idx] / falloff);
      if (conf < cfg.fusion.minConfidence) continue;
      const d = near + d01 * (far - near);

      // Camera-space (image +Y down matches our world basis).
      const xc = ((x - cx) / f) * d;
      const yc = ((y - cy) / f) * d;
      const zc = d;

      // world = M (column-major) * [xc,yc,zc,1]
      const wx = m[0] * xc + m[4] * yc + m[8] * zc + m[12];
      const wy = m[1] * xc + m[5] * yc + m[9] * zc + m[13];
      const wz = m[2] * xc + m[6] * yc + m[10] * zc + m[14];

      const p3 = n * 3;
      positions[p3] = wx;
      positions[p3 + 1] = wy;
      positions[p3 + 2] = wz;

      const c = idx * 4;
      colors[p3] = kf.rgba[c] / 255;
      colors[p3 + 1] = kf.rgba[c + 1] / 255;
      colors[p3 + 2] = kf.rgba[c + 2] / 255;

      confidence[n] = conf;
      n++;
    }
  }

  return {
    positions: positions.subarray(0, n * 3),
    colors: colors.subarray(0, n * 3),
    confidence: confidence.subarray(0, n),
    count: n,
  };
}

/**
 * Two-pass chamfer distance transform: for each foreground pixel, approximate
 * distance (in px) to the nearest background pixel.
 */
export function chamferEdgeDistance(alpha: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e6;
  const dist = new Float32Array(w * h);
  for (let i = 0; i < dist.length; i++) dist[i] = alpha[i] === 255 ? INF : 0;

  const d1 = 1;
  const d2 = Math.SQRT2;
  // Forward pass.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = dist[i];
      if (x > 0) v = Math.min(v, dist[i - 1] + d1);
      if (y > 0) v = Math.min(v, dist[i - w] + d1);
      if (x > 0 && y > 0) v = Math.min(v, dist[i - w - 1] + d2);
      if (x < w - 1 && y > 0) v = Math.min(v, dist[i - w + 1] + d2);
      dist[i] = v;
    }
  }
  // Backward pass.
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = dist[i];
      if (x < w - 1) v = Math.min(v, dist[i + 1] + d1);
      if (y < h - 1) v = Math.min(v, dist[i + w] + d1);
      if (x < w - 1 && y < h - 1) v = Math.min(v, dist[i + w + 1] + d2);
      if (x > 0 && y < h - 1) v = Math.min(v, dist[i + w - 1] + d2);
      dist[i] = v;
    }
  }
  return dist;
}
