import type { DepthMap, ProgressFn } from '../types';
import { getConfig } from '../config';
import { tick } from '../util/tick';

/**
 * Depth denoising. High-frequency surface texture (wood grain, mineral speckle)
 * is great for pose refinement but injects per-frame depth noise. We remove that
 * with two passes:
 *
 *  1. Bilateral spatial filter — edge-preserving smoothing so silhouettes and
 *     crevice boundaries stay sharp while flat facets get denoised.
 *  2. Cross-frame temporal blend — nudges each pixel toward the median of the
 *     same pixel in temporally-adjacent keyframes. Adjacent turntable frames are
 *     only a small angle apart, so this is an approximate but effective flicker
 *     killer (kept at a low weight by default).
 */
export async function smoothDepthMaps(
  maps: DepthMap[],
  onProgress?: ProgressFn,
): Promise<DepthMap[]> {
  const cfg = getConfig();
  let work = maps;

  if (cfg.depth.bilateral.enabled) {
    const out: DepthMap[] = new Array(maps.length);
    for (let i = 0; i < work.length; i++) {
      out[i] = bilateral(
        work[i],
        cfg.depth.bilateral.diameter,
        cfg.depth.bilateral.sigmaSpace,
        cfg.depth.bilateral.sigmaDepth,
      );
      onProgress?.('smooth', (i + 1) / work.length, `frame ${i + 1}/${work.length}`);
      await tick(); // yield so the UI repaints between frames
    }
    work = out;
  }

  if (cfg.depth.temporalWeight > 0 && work.length >= 3) {
    work = temporalMedianBlend(work, cfg.depth.temporalWeight);
    await tick();
  }
  return work;
}

/** Edge-preserving bilateral filter over a single depth map. */
export function bilateral(
  map: DepthMap,
  diameter: number,
  sigmaSpace: number,
  sigmaDepth: number,
): DepthMap {
  const { width: w, height: h, depth } = map;
  const r = Math.max(1, Math.floor(diameter / 2));
  const out = new Float32Array(w * h);

  // Precompute spatial gaussian weights.
  const spatial: number[] = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      spatial.push(Math.exp(-(dx * dx + dy * dy) / (2 * sigmaSpace * sigmaSpace)));
    }
  }
  const invDepth2 = 1 / (2 * sigmaDepth * sigmaDepth);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const center = depth[y * w + x];
      let sum = 0;
      let wsum = 0;
      let k = 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) {
          k += 2 * r + 1;
          continue;
        }
        for (let dx = -r; dx <= r; dx++, k++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = depth[yy * w + xx];
          const dd = v - center;
          const wgt = spatial[k] * Math.exp(-dd * dd * invDepth2);
          sum += v * wgt;
          wsum += wgt;
        }
      }
      out[y * w + x] = wsum > 0 ? sum / wsum : center;
    }
  }
  return { width: w, height: h, depth: out };
}

/**
 * Blend each pixel toward the temporal median of its neighbours (prev, self,
 * next). Assumes small inter-frame motion (adjacent turntable frames).
 */
export function temporalMedianBlend(maps: DepthMap[], weight: number): DepthMap[] {
  const n = maps.length;
  const w = maps[0].width;
  const h = maps[0].height;
  const out: DepthMap[] = [];
  for (let i = 0; i < n; i++) {
    const cur = maps[i].depth;
    const prev = maps[Math.max(0, i - 1)].depth;
    const next = maps[Math.min(n - 1, i + 1)].depth;
    const d = new Float32Array(w * h);
    for (let p = 0; p < d.length; p++) {
      const a = prev[p], b = cur[p], c = next[p];
      const med = a < b ? (b < c ? b : a < c ? c : a) : b < c ? (a < c ? a : c) : b;
      d[p] = (1 - weight) * b + weight * med;
    }
    out.push({ width: w, height: h, depth: d });
  }
  return out;
}
