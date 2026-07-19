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
import { calibrateFromTurntable } from '../poses/scale';

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

  const mattes = await matteKeyframes(keyframes, onProgress);
  await tick();

  const rawDepths = await estimateDepth(keyframes, onProgress);
  await tick();
  const depths = await smoothDepthMaps(rawDepths, onProgress);

  onProgress?.('poses', 0, 'refining turntable poses');
  const thetas = await refineTurntableAngles(keyframes, onProgress);
  const poses = deriveTurntablePoses(keyframes, { thetasRad: thetas });
  await tick();

  let cloud = await fuseFrames(keyframes, depths, mattes, poses, onProgress);

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
  return { keyframes, mattes, depths, poses, cloud, metersPerUnit };
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
