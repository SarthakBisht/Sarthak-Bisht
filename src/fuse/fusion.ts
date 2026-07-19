import type {
  CameraPose,
  DepthMap,
  Keyframe,
  Matte,
  PointCloud,
  ProgressFn,
} from '../types';
import { getConfig } from '../config';
import { backprojectFrame, type FramePoints } from './backproject';
import { tick } from '../util/tick';

/**
 * Fuse all back-projected frames into a single coloured point cloud. Multi-view
 * fusion is what recovers geometry a single depth view misses — concavities,
 * crevices, and the undersides of driftwood arches — because each frame sees
 * around a different part of the object.
 *
 * Pipeline: back-project every frame -> concatenate -> optional voxel dedup ->
 * statistical outlier removal -> cap to maxPoints. Point density is kept high by
 * default so thin protrusions (branches) are not decimated away.
 */
export async function fuseFrames(
  keyframes: Keyframe[],
  depths: DepthMap[],
  mattes: Matte[],
  poses: CameraPose[],
  onProgress?: ProgressFn,
): Promise<PointCloud> {
  const cfg = getConfig();
  const frames: FramePoints[] = [];
  let total = 0;
  for (let i = 0; i < keyframes.length; i++) {
    const fp = backprojectFrame(keyframes[i], depths[i], mattes[i], poses[i]);
    frames.push(fp);
    total += fp.count;
    onProgress?.('fuse', (i + 1) / keyframes.length, `frame ${i + 1}/${keyframes.length}`);
    await tick(); // yield between frames
  }

  // Concatenate.
  const positions = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const confidence = new Float32Array(total);
  let off = 0;
  for (const fp of frames) {
    positions.set(fp.positions, off * 3);
    colors.set(fp.colors, off * 3);
    confidence.set(fp.confidence, off);
    off += fp.count;
  }
  let cloud: PointCloud = { count: total, positions, colors, confidence };

  // Voxel downsample / dedup.
  if (cfg.fusion.voxelSizeM > 0) {
    onProgress?.('fuse', 0.985, 'merging points');
    await tick();
    cloud = voxelDownsample(cloud, cfg.fusion.voxelSizeM);
  }

  // Statistical outlier removal.
  if (cfg.fusion.outlier.enabled && cloud.count > cfg.fusion.outlier.k * 2) {
    onProgress?.('fuse', 0.99, 'removing outliers');
    await tick();
    cloud = removeOutliers(cloud, cfg.fusion.outlier.k, cfg.fusion.outlier.stdRatio);
  }

  // Cap total points (keep highest-confidence).
  if (cloud.count > cfg.fusion.maxPoints) {
    cloud = capByConfidence(cloud, cfg.fusion.maxPoints);
  }

  return cloud;
}

/** Average points falling in the same voxel; keeps memory bounded and dedups. */
export function voxelDownsample(cloud: PointCloud, voxel: number): PointCloud {
  const inv = 1 / voxel;
  const map = new Map<string, { px: number; py: number; pz: number; cr: number; cg: number; cb: number; conf: number; n: number }>();
  const p = cloud.positions;
  const c = cloud.colors;
  const cf = cloud.confidence;
  for (let i = 0; i < cloud.count; i++) {
    const i3 = i * 3;
    const kx = Math.floor(p[i3] * inv);
    const ky = Math.floor(p[i3 + 1] * inv);
    const kz = Math.floor(p[i3 + 2] * inv);
    const key = `${kx},${ky},${kz}`;
    let e = map.get(key);
    if (!e) {
      e = { px: 0, py: 0, pz: 0, cr: 0, cg: 0, cb: 0, conf: 0, n: 0 };
      map.set(key, e);
    }
    e.px += p[i3]; e.py += p[i3 + 1]; e.pz += p[i3 + 2];
    e.cr += c[i3]; e.cg += c[i3 + 1]; e.cb += c[i3 + 2];
    e.conf += cf[i];
    e.n++;
  }
  const count = map.size;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const confidence = new Float32Array(count);
  let i = 0;
  for (const e of map.values()) {
    const i3 = i * 3;
    positions[i3] = e.px / e.n; positions[i3 + 1] = e.py / e.n; positions[i3 + 2] = e.pz / e.n;
    colors[i3] = e.cr / e.n; colors[i3 + 1] = e.cg / e.n; colors[i3 + 2] = e.cb / e.n;
    confidence[i] = e.conf / e.n;
    i++;
  }
  return { count, positions, colors, confidence };
}

/**
 * Statistical outlier removal using a uniform spatial grid for kNN lookup.
 * Drops depth-edge flyers whose mean neighbour distance is an outlier.
 */
export function removeOutliers(cloud: PointCloud, k: number, stdRatio: number): PointCloud {
  const { count } = cloud;
  const grid = buildGrid(cloud);
  const meanDist = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    meanDist[i] = grid.meanKnnDistance(i, k);
  }
  let mean = 0;
  for (let i = 0; i < count; i++) mean += meanDist[i];
  mean /= count || 1;
  let variance = 0;
  for (let i = 0; i < count; i++) variance += (meanDist[i] - mean) ** 2;
  const std = Math.sqrt(variance / (count || 1));
  const threshold = mean + stdRatio * std;

  const keep: number[] = [];
  for (let i = 0; i < count; i++) if (meanDist[i] <= threshold) keep.push(i);
  return gather(cloud, keep);
}

function capByConfidence(cloud: PointCloud, max: number): PointCloud {
  const order = Array.from({ length: cloud.count }, (_, i) => i);
  order.sort((a, b) => cloud.confidence[b] - cloud.confidence[a]);
  return gather(cloud, order.slice(0, max));
}

function gather(cloud: PointCloud, idx: number[]): PointCloud {
  const count = idx.length;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const confidence = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const s = idx[i] * 3;
    const d = i * 3;
    positions[d] = cloud.positions[s];
    positions[d + 1] = cloud.positions[s + 1];
    positions[d + 2] = cloud.positions[s + 2];
    colors[d] = cloud.colors[s];
    colors[d + 1] = cloud.colors[s + 1];
    colors[d + 2] = cloud.colors[s + 2];
    confidence[i] = cloud.confidence[idx[i]];
  }
  return { count, positions, colors, confidence };
}

// ---- uniform-grid kNN ------------------------------------------------------
interface Grid {
  meanKnnDistance(i: number, k: number): number;
}

function buildGrid(cloud: PointCloud): Grid {
  const { count, positions } = cloud;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    minX = Math.min(minX, positions[i3]); maxX = Math.max(maxX, positions[i3]);
    minY = Math.min(minY, positions[i3 + 1]); maxY = Math.max(maxY, positions[i3 + 1]);
    minZ = Math.min(minZ, positions[i3 + 2]); maxZ = Math.max(maxZ, positions[i3 + 2]);
  }
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const cell = Math.max(diag / 64, 1e-4);
  const inv = 1 / cell;
  const nx = Math.max(1, Math.ceil((maxX - minX) * inv) + 1);
  const ny = Math.max(1, Math.ceil((maxY - minY) * inv) + 1);
  const buckets = new Map<number, number[]>();
  const cellOf = (i: number) => {
    const i3 = i * 3;
    const cx = Math.floor((positions[i3] - minX) * inv);
    const cy = Math.floor((positions[i3 + 1] - minY) * inv);
    const cz = Math.floor((positions[i3 + 2] - minZ) * inv);
    return (cz * ny + cy) * nx + cx;
  };
  for (let i = 0; i < count; i++) {
    const key = cellOf(i);
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = []));
    b.push(i);
  }

  return {
    meanKnnDistance(i: number, k: number): number {
      const i3 = i * 3;
      const px = positions[i3], py = positions[i3 + 1], pz = positions[i3 + 2];
      const cx = Math.floor((px - minX) * inv);
      const cy = Math.floor((py - minY) * inv);
      const cz = Math.floor((pz - minZ) * inv);
      const dists: number[] = [];
      for (let dz = -1; dz <= 1; dz++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const key = ((cz + dz) * ny + (cy + dy)) * nx + (cx + dx);
            const b = buckets.get(key);
            if (!b) continue;
            for (const j of b) {
              if (j === i) continue;
              const j3 = j * 3;
              const d = Math.hypot(px - positions[j3], py - positions[j3 + 1], pz - positions[j3 + 2]);
              dists.push(d);
            }
          }
        }
      }
      if (dists.length === 0) return 0;
      dists.sort((a, b) => a - b);
      const m = Math.min(k, dists.length);
      let sum = 0;
      for (let n = 0; n < m; n++) sum += dists[n];
      return sum / m;
    },
  };
}
