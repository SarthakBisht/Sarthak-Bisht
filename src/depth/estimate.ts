import { pipeline } from '@huggingface/transformers';
import type { DepthEstimationPipeline } from '@huggingface/transformers';
import { RawImage } from '@huggingface/transformers';
import type { DepthMap, Keyframe, ProgressFn } from '../types';
import { getConfig } from '../config';
import { configureEnv, pickDevice } from '../pipeline/env';

/**
 * Monocular depth via Depth Anything V2 (small) through transformers.js.
 * WebGPU-first with an automatic WASM fallback (depth only). Runs in batches to
 * bound peak memory and disposes tensors between batches.
 *
 * Depth Anything outputs a *relative* disparity-like map where larger values are
 * CLOSER. We normalize per-frame and store depth as 0..1 with 1 = FAR (see
 * DepthMap), so downstream back-projection can map it into a metric band.
 */

let estimator: DepthEstimationPipeline | null = null;
let estimatorDevice: 'webgpu' | 'wasm' | null = null;

async function getEstimator(): Promise<{ pipe: DepthEstimationPipeline; device: 'webgpu' | 'wasm' }> {
  if (estimator && estimatorDevice) return { pipe: estimator, device: estimatorDevice };
  configureEnv();
  const cfg = getConfig();
  const device = await pickDevice();
  const dtype = device === 'webgpu' ? cfg.depth.webgpuDtype : cfg.depth.wasmDtype;
  estimator = (await pipeline('depth-estimation', cfg.depth.modelId, {
    device,
    dtype,
  })) as DepthEstimationPipeline;
  estimatorDevice = device;
  return { pipe: estimator, device };
}

export async function estimateDepth(
  keyframes: Keyframe[],
  onProgress?: ProgressFn,
): Promise<DepthMap[]> {
  const cfg = getConfig();
  const { pipe } = await getEstimator();
  const out: DepthMap[] = new Array(keyframes.length);
  const batch = Math.max(1, cfg.depth.batchSize);

  let done = 0;
  for (let start = 0; start < keyframes.length; start += batch) {
    const slice = keyframes.slice(start, start + batch);
    for (const kf of slice) {
      const input = new RawImage(kf.rgba, kf.width, kf.height, 4).rgb();
      const result = await pipe(input);
      const single = Array.isArray(result) ? result[0] : result;
      out[kf.index] = tensorToDepthMap(single.predicted_depth, kf.width, kf.height);
      // Free the ONNX tensor promptly.
      single.predicted_depth.dispose?.();
      done++;
      onProgress?.('depth', done / keyframes.length, `frame ${done}/${keyframes.length}`);
    }
  }
  return out;
}

interface DepthTensor {
  data: Float32Array | number[];
  dims: number[];
  dispose?: () => void;
}

/** Convert a predicted_depth tensor to a normalized DepthMap (1 = far). */
function tensorToDepthMap(tensor: DepthTensor, targetW: number, targetH: number): DepthMap {
  const dims = tensor.dims;
  // dims is [H, W] or [1, H, W].
  const h = dims.length === 3 ? dims[1] : dims[0];
  const w = dims.length === 3 ? dims[2] : dims[1];
  const data = tensor.data as Float32Array;

  // Per-frame min/max for normalization.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  const depth = new Float32Array(targetW * targetH);
  for (let y = 0; y < targetH; y++) {
    const sy = Math.min(h - 1, (y * h / targetH) | 0);
    for (let x = 0; x < targetW; x++) {
      const sx = Math.min(w - 1, (x * w / targetW) | 0);
      const disparity = (data[sy * w + sx] - min) / range; // 1 = closer
      depth[y * targetW + x] = 1 - disparity; // store as 1 = far
    }
  }
  return { width: targetW, height: targetH, depth };
}

export function disposeEstimator(): void {
  estimator?.dispose?.();
  estimator = null;
  estimatorDevice = null;
}
