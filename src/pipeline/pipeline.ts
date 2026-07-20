import type {
  CameraPose,
  DepthMap,
  Keyframe,
  Matte,
  PointCloud,
  ProgressFn,
} from '../types';
import { getConfig, patchConfig } from '../config';
import { detectGpu } from '../webgpu/detect';
import { tick } from '../util/tick';
import { extractKeyframes } from '../capture/keyframes';
import { matteKeyframes, disposeSegmenter } from '../matte/segment';
import { estimateDepth, disposeEstimator } from '../depth/estimate';
import { smoothDepthMaps } from '../depth/smooth';
import { refineTurntableAngles } from '../poses/refine';
import { deriveTurntablePoses } from '../poses/turntable';
import { fuseFrames } from '../fuse/fusion';
import { calibrateFromTurntable, horizontalExtent } from '../poses/scale';
import { carveVisualHull } from '../fuse/visualHull';
import type { TriMesh } from '../mesh/poisson';

export interface Tier0Result {
  keyframes: Keyframe[];
  mattes: Matte[];
  depths: DepthMap[];
  poses: CameraPose[];
  cloud: PointCloud;
  metersPerUnit: number;
}

/**
 * Tier 0 — the guaranteed MVP pipeline, end to end:
 *   keyframes -> matte -> depth -> smooth -> refined turntable poses ->
 *   back-project + fuse -> metric scale.
 *
 * `onProgress(stage, fraction)` reports per-stage progress; the overall bar in
 * the UI maps stages to weighted spans.
 */
export async function runTier0(
  source: Blob | string,
  onProgress?: ProgressFn,
): Promise<Tier0Result> {
  await autoTuneForDevice();
  const cfg = getConfig();

  const keyframes = await extractKeyframes(source, onProgress);
  if (keyframes.length < 3) {
    throw new Error('Not enough usable frames extracted — capture a longer, steadier clip.');
  }
  await tick();

  const rawMattes = await matteKeyframes(keyframes, onProgress);
  await tick();

  // Drop frames where the phone was tilted (object drifts vertically) — the
  // fixed-elevation turntable model can't place them and they corrupt the cloud.
  const { keyframes: frames, mattes, dropped } = rejectTiltedFrames(keyframes, rawMattes);
  if (dropped > 0) onProgress?.('matte', 1, `dropped ${dropped} tilted frame(s)`);

  const rawDepths = await estimateDepth(frames, onProgress);
  await tick();
  const depths = await smoothDepthMaps(rawDepths, onProgress);

  onProgress?.('poses', 0, 'refining turntable poses');
  const thetas = await refineTurntableAngles(frames, onProgress);
  const poses = deriveTurntablePoses(frames, { thetasRad: thetas });
  await tick();

  let cloud = await fuseFrames(frames, depths, mattes, poses, onProgress);

  // Metric scale (turntable diameter by default; UI can recalibrate later).
  let metersPerUnit = 1;
  if (cfg.scale.mode === 'turntable') {
    const res = calibrateFromTurntable(cloud);
    cloud = res.appliedTo;
    metersPerUnit = res.metersPerUnit;
  }

  if (cfg.runtime.aggressiveDispose) {
    // Depth/matte models are large; free them once the cloud exists. They are
    // re-created lazily if the user re-runs.
    disposeEstimator();
    disposeSegmenter();
  }

  onProgress?.('done', 1, `${cloud.count.toLocaleString()} points`);
  return { keyframes: frames, mattes, depths, poses, cloud, metersPerUnit };
}

export interface GuidedResult {
  keyframes: Keyframe[];
  poses: CameraPose[];
  mesh: TriMesh;
  cloud: PointCloud;
  metersPerUnit: number;
}

/**
 * Guided-capture reconstruction: controlled stills at KNOWN angles → clean
 * silhouettes → visual hull → coloured mesh. Accurate poses (no derivation) and
 * background-free by construction.
 */
export async function runGuided(
  keyframes: Keyframe[],
  azimuthsRad: number[],
  onProgress?: ProgressFn,
): Promise<GuidedResult> {
  const cfg = getConfig();
  if (keyframes.length < 3) {
    throw new Error('Need at least 3 captured views for a hull.');
  }

  const mattes = await matteKeyframes(keyframes, onProgress);
  await tick();

  const poses = deriveTurntablePoses(keyframes, {
    thetasRad: azimuthsRad,
    elevationDeg: cfg.guided.elevationDeg,
  });
  const { mesh, cloud } = await carveVisualHull(keyframes, mattes, poses, onProgress);

  // Metric scale from the turntable diameter (horizontal extent → diameter).
  let metersPerUnit = 1;
  const extent = horizontalExtent(cloud) || 1;
  metersPerUnit = cfg.scale.turntableDiameterM / extent;
  for (let i = 0; i < cloud.positions.length; i++) cloud.positions[i] *= metersPerUnit;
  for (let i = 0; i < mesh.positions.length; i++) mesh.positions[i] *= metersPerUnit;

  if (cfg.runtime.aggressiveDispose) disposeSegmenter();

  onProgress?.('done', 1, `${mesh.triangleCount.toLocaleString()} triangles`);
  return { keyframes, poses, mesh, cloud, metersPerUnit };
}

/**
 * Reject frames where the phone tilted (the object's vertical centroid drifts
 * from the median) or that lost the object — the fixed-elevation turntable model
 * can only handle a single camera height. Conservative: if it would drop too
 * many frames, keep them all (avoids nuking a valid scan on a noisy estimate).
 */
function rejectTiltedFrames(
  keyframes: Keyframe[],
  mattes: Matte[],
): { keyframes: Keyframe[]; mattes: Matte[]; dropped: number } {
  const cfg = getConfig();
  const thr = cfg.poses.maxVerticalDriftFrac;
  const n = keyframes.length;
  if (thr >= 1 || n < 5) return { keyframes, mattes, dropped: 0 };

  const info = mattes.map((m) => {
    let sum = 0;
    let count = 0;
    for (let y = 0; y < m.height; y++) {
      let rc = 0;
      for (let x = 0; x < m.width; x++) if (m.alpha[y * m.width + x] === 255) rc++;
      sum += y * rc;
      count += rc;
    }
    return { cy: count ? sum / count : m.height / 2, area: count };
  });

  const h = keyframes[0].height;
  const medCy = [...info.map((i) => i.cy)].sort((a, b) => a - b)[n >> 1];
  const medArea = [...info.map((i) => i.area)].sort((a, b) => a - b)[n >> 1] || 1;

  const keep: number[] = [];
  for (let i = 0; i < n; i++) {
    const okArea = info[i].area > medArea * 0.25;
    const okDrift = Math.abs(info[i].cy - medCy) <= thr * h;
    if (okArea && okDrift) keep.push(i);
  }
  if (keep.length < Math.max(3, Math.floor(n * 0.5))) {
    return { keyframes, mattes, dropped: 0 }; // too aggressive → keep all
  }
  const kf = keep.map((idx, newI) => ({ ...keyframes[idx], index: newI }));
  const km = keep.map((idx) => mattes[idx]);
  return { keyframes: kf, mattes: km, dropped: n - keep.length };
}

/**
 * On mobile or when WebGPU is unavailable (depth on WASM), reduce the work so
 * the pipeline completes in a reasonable time and does not exhaust memory.
 * Desktop WebGPU keeps full quality. Only ever lowers values (never raises),
 * so an explicit user/preset choice that's already lighter is respected.
 */
async function autoTuneForDevice(): Promise<void> {
  const caps = await detectGpu();
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
  if (caps.deviceOk && !mobile) return; // desktop WebGPU → full quality

  const c = getConfig();
  patchConfig({
    keyframes: {
      count: Math.min(c.keyframes.count, 20),
      maxEdgePx: Math.min(c.keyframes.maxEdgePx, 480),
    },
    depth: {
      batchSize: Math.min(c.depth.batchSize, 2),
      bilateral: { diameter: Math.min(c.depth.bilateral.diameter, 5) },
    },
    fusion: {
      maxPoints: Math.min(c.fusion.maxPoints, 150_000),
      outlier: { k: Math.min(c.fusion.outlier.k, 8) },
    },
  });
}
