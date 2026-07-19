import type { Keyframe, PointCloud, ProgressFn } from '../types';
import { getConfig } from '../config';

/**
 * Tier 2 — pose-free reconstruction (DUSt3R / MASt3R / InstantSplat family) via
 * onnxruntime-web.
 *
 * This is the most speculative tier: at the time of writing there is no
 * lightweight, browser-exportable ONNX build of these models that fits a mobile
 * compute/memory budget. So this module is a GUARDED STUB — it advertises the
 * capability, checks whether a model URL is configured and whether onnxruntime
 * is importable, and otherwise SKIPS CLEANLY, leaving the turntable pipeline as
 * the result. No hard dependency is taken.
 */
export interface PosefreeResult {
  supported: boolean;
  cloud?: PointCloud;
  reason?: string;
}

export async function tryPosefree(
  _keyframes: Keyframe[],
  onProgress?: ProgressFn,
): Promise<PosefreeResult> {
  const cfg = getConfig();
  if (!cfg.posefree.attempt) {
    return { supported: false, reason: 'Pose-free tier disabled in config.' };
  }
  if (!cfg.posefree.modelUrl) {
    return {
      supported: false,
      reason: 'No ONNX model configured (config.posefree.modelUrl is empty).',
    };
  }

  onProgress?.('posefree', 0.1, 'probing onnxruntime-web');
  let ort: unknown;
  try {
    // Optional dependency — only imported if present. Vite marks it external.
    ort = await import(/* @vite-ignore */ 'onnxruntime-web');
  } catch {
    return {
      supported: false,
      reason: 'onnxruntime-web is not installed; skipping Tier 2.',
    };
  }

  // Rough memory guard before attempting to fetch a heavy model.
  const budget = cfg.posefree.memoryBudgetMB;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (mem && mem * 1024 < budget) {
    return {
      supported: false,
      reason: `Device memory (~${mem}GB) below Tier 2 budget (${budget}MB); skipping.`,
    };
  }

  // A real DUSt3R/MASt3R integration would create an InferenceSession here,
  // run pairwise inference, align point maps, and fuse. Until a suitable
  // browser-exportable model exists we skip rather than ship something broken.
  void ort;
  return {
    supported: false,
    reason:
      'Tier 2 model integration not available in this build — skipped gracefully; ' +
      'using turntable reconstruction instead.',
  };
}
