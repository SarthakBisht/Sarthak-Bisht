import type { Keyframe, ProgressFn } from '../types';
import { getConfig } from '../config';

/**
 * Refine the ideal constant-angular-step turntable poses using feature
 * tracking. High-frequency surface texture (wood grain, mineral speckle) is
 * exactly what makes this reliable: we measure the real per-gap image motion
 * between adjacent frames and redistribute the camera angles so that angular
 * velocity matches the *observed* motion instead of assuming it is constant.
 *
 * We do NOT attempt full SfM. Turntable motion is dominantly horizontal in the
 * image, so a robust median horizontal displacement per gap is a good proxy for
 * that gap's angular size. Returns per-frame angles in radians spanning the
 * configured total sweep.
 */
export function refineTurntableAngles(
  keyframes: Keyframe[],
  onProgress?: ProgressFn,
): number[] {
  const cfg = getConfig();
  const n = keyframes.length;
  const sweep = (cfg.capture.totalSweepDeg * Math.PI) / 180;
  if (n < 3 || !cfg.poses.refine) {
    // Constant step fallback.
    return Array.from({ length: n }, (_, i) => (n > 1 ? (sweep * i) / (n - 1) : 0));
  }

  // Per-gap motion magnitude (pixels). gap[i] is motion from frame i -> i+1.
  const gaps: number[] = new Array(n - 1).fill(0);
  for (let i = 0; i < n - 1; i++) {
    gaps[i] = medianHorizontalMotion(keyframes[i], keyframes[i + 1], cfg.poses.maxFeatures);
    onProgress?.('poses', (i + 1) / (n - 1), `gap ${i + 1}/${n - 1}`);
  }

  // Smooth a couple of iterations to suppress tracking noise, keep positivity.
  let weights = gaps.map((g) => Math.max(1e-3, g));
  for (let it = 0; it < cfg.poses.iterations; it++) {
    weights = smooth(weights);
  }

  // Cumulative angle proportional to cumulative motion, normalized to sweep.
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  const thetas = new Array(n).fill(0);
  let acc = 0;
  for (let i = 1; i < n; i++) {
    acc += weights[i - 1];
    thetas[i] = (sweep * acc) / total;
  }
  return thetas;
}

/**
 * Estimate the robust (median) horizontal displacement between two frames by
 * matching high-gradient feature patches along a horizontal search band.
 */
function medianHorizontalMotion(a: Keyframe, b: Keyframe, maxFeatures: number): number {
  const w = a.width;
  const h = a.height;
  const grayA = toGray(a.rgba, w, h);
  const grayB = toGray(b.rgba, w, h);

  const feats = detectFeatures(grayA, w, h, maxFeatures);
  const patch = 3; // => 7x7 patch
  const searchX = Math.min(48, Math.floor(w * 0.12));
  const dxs: number[] = [];

  for (const [fx, fy] of feats) {
    if (fx < patch + searchX || fx >= w - patch - searchX || fy < patch || fy >= h - patch) {
      continue;
    }
    let bestDx = 0;
    let bestScore = Infinity;
    for (let dx = -searchX; dx <= searchX; dx++) {
      let sad = 0;
      for (let py = -patch; py <= patch; py++) {
        const rowA = (fy + py) * w;
        const rowB = (fy + py) * w;
        for (let px = -patch; px <= patch; px++) {
          const va = grayA[rowA + fx + px];
          const vb = grayB[rowB + fx + px + dx];
          const d = va - vb;
          sad += d < 0 ? -d : d;
        }
      }
      if (sad < bestScore) {
        bestScore = sad;
        bestDx = dx;
      }
    }
    dxs.push(bestDx);
  }
  if (dxs.length === 0) return 1;
  dxs.sort((x, y) => x - y);
  // Use magnitude of median signed displacement.
  const med = dxs[Math.floor(dxs.length / 2)];
  return Math.max(1e-3, Math.abs(med));
}

function toGray(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const g = new Float32Array(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) {
    g[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return g;
}

/** Cheap Harris-ish corner sampling on a coarse grid; returns [x,y] pairs. */
function detectFeatures(gray: Float32Array, w: number, h: number, max: number): [number, number][] {
  const cellsX = 24;
  const cellsY = 16;
  const cw = Math.floor(w / cellsX);
  const ch = Math.floor(h / cellsY);
  const feats: { x: number; y: number; score: number }[] = [];
  for (let cy = 0; cy < cellsY; cy++) {
    for (let cx = 0; cx < cellsX; cx++) {
      let best = { x: 0, y: 0, score: 0 };
      const x0 = cx * cw + 2;
      const y0 = cy * ch + 2;
      const x1 = Math.min(w - 3, x0 + cw);
      const y1 = Math.min(h - 3, y0 + ch);
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const gx = gray[y * w + x + 1] - gray[y * w + x - 1];
          const gy = gray[(y + 1) * w + x] - gray[(y - 1) * w + x];
          const score = gx * gx + gy * gy;
          if (score > best.score) best = { x, y, score };
        }
      }
      if (best.score > 200) feats.push(best);
    }
  }
  feats.sort((p, q) => q.score - p.score);
  return feats.slice(0, max).map((f) => [f.x, f.y] as [number, number]);
}

function smooth(v: number[]): number[] {
  const out = v.slice();
  for (let i = 0; i < v.length; i++) {
    const l = v[Math.max(0, i - 1)];
    const r = v[Math.min(v.length - 1, i + 1)];
    out[i] = 0.25 * l + 0.5 * v[i] + 0.25 * r;
  }
  return out;
}
