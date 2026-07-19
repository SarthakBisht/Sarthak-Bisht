import type {
  CameraPose,
  DepthMap,
  Keyframe,
  Matte,
  PointCloud,
  ProgressFn,
} from '../types';
import { getConfig } from '../config';
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
  const cfg = getConfig();

  const keyframes = await extractKeyframes(source, onProgress);
  if (keyframes.length < 3) {
    throw new Error('Not enough usable frames extracted — capture a longer, steadier clip.');
  }

  const mattes = await matteKeyframes(keyframes, onProgress);

  const rawDepths = await estimateDepth(keyframes, onProgress);
  const depths = smoothDepthMaps(rawDepths);

  onProgress?.('poses', 0, 'refining turntable poses');
  const thetas = refineTurntableAngles(keyframes, onProgress);
  const poses = deriveTurntablePoses(keyframes, { thetasRad: thetas });

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
