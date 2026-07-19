import { AutoModel, AutoProcessor, RawImage } from '@huggingface/transformers';
import type { Keyframe, Matte, ProgressFn } from '../types';
import { getConfig } from '../config';
import { configureEnv, pickDevice } from '../pipeline/env';

/**
 * Client-side matte for irregular organic silhouettes (driftwood, rock).
 *
 * RMBG-1.4 is a custom architecture that the high-level `background-removal`
 * pipeline misidentifies (it reports it as "SegformerForSemanticSegmentation").
 * The documented, working path is the low-level `AutoModel` + `AutoProcessor`
 * with `model_type: 'custom'` and an explicit preprocessing config. We take the
 * model's single-channel matte, threshold it, then ERODE by a few pixels to drop
 * halo/fringe pixels that would smear back-projected geometry into the
 * background.
 */

// The model/processor I/O is dynamically shaped (ONNX names), so these are
// intentionally loosely typed.
type CallableModel = ((inputs: Record<string, unknown>) => Promise<Record<string, unknown>>) & {
  dispose?: () => void;
};
interface Processor {
  (image: RawImage): Promise<{ pixel_values: unknown }>;
}

let model: CallableModel | null = null;
let processor: Processor | null = null;

async function ensureLoaded(): Promise<{ model: CallableModel; processor: Processor }> {
  if (model && processor) return { model, processor };
  configureEnv();
  const cfg = getConfig();
  const device = await pickDevice();

  // RMBG-1.4 needs the "custom" model type; fp32 is the reliable dtype.
  const modelOpts = { config: { model_type: 'custom' }, device, dtype: 'fp32' };
  model = (await AutoModel.from_pretrained(
    cfg.matte.modelId,
    modelOpts as unknown as Parameters<typeof AutoModel.from_pretrained>[1],
  )) as unknown as CallableModel;

  // RMBG-1.4 ships no preprocessor_config, so supply it explicitly.
  const procOpts = {
    config: {
      do_normalize: true,
      do_pad: false,
      do_rescale: true,
      do_resize: true,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [1.0, 1.0, 1.0],
      resample: 2,
      rescale_factor: 1 / 255,
      size: { width: 1024, height: 1024 },
    },
  };
  processor = (await AutoProcessor.from_pretrained(
    cfg.matte.modelId,
    procOpts as unknown as Parameters<typeof AutoProcessor.from_pretrained>[1],
  )) as unknown as Processor;

  return { model, processor };
}

export async function matteKeyframes(
  keyframes: Keyframe[],
  onProgress?: ProgressFn,
): Promise<Matte[]> {
  const cfg = getConfig();
  const { model: net, processor: proc } = await ensureLoaded();
  const out: Matte[] = [];

  for (let i = 0; i < keyframes.length; i++) {
    const kf = keyframes[i];
    const image = new RawImage(kf.rgba, kf.width, kf.height, 4).rgb();
    const { pixel_values } = await proc(image);
    const result = await net({ input: pixel_values });

    // RMBG returns a single [1,1,H,W] matte tensor in [0,1]; grab it robustly.
    const tensor = (result.output ?? Object.values(result)[0]) as MatteTensor;
    const maskTensor = tensor[0].mul(255).to('uint8') as Parameters<typeof RawImage.fromTensor>[0];
    const maskImg = await RawImage.fromTensor(maskTensor).resize(kf.width, kf.height);

    let alpha = thresholdAlpha(maskImg, kf.width, kf.height, cfg.matte.threshold);
    // Restrict to the centered scan-zone ellipse so off-center background and the
    // turntable surface never enter the geometry.
    if (cfg.matte.roi.enabled) {
      alpha = applyRoi(alpha, kf.width, kf.height, cfg.matte.roi.radiusXFrac, cfg.matte.roi.radiusYFrac);
    }
    // Keep only the biggest blob (drops stray background regions RMBG let through).
    if (cfg.matte.largestComponentOnly) {
      alpha = largestComponent(alpha, kf.width, kf.height);
    }
    alpha = erode(alpha, kf.width, kf.height, cfg.matte.erosionPx);
    out.push({ width: kf.width, height: kf.height, alpha });
    onProgress?.('matte', (i + 1) / keyframes.length, `frame ${i + 1}/${keyframes.length}`);
  }
  return out;
}

interface MatteTensor {
  [index: number]: { mul(v: number): { to(t: string): unknown } };
}

/** Threshold a single-channel matte RawImage into a binary mask. */
function thresholdAlpha(img: RawImage, w: number, h: number, threshold: number): Uint8Array {
  const data = img.data;
  const ch = img.channels;
  const alpha = new Uint8Array(w * h);
  const cut = Math.round(threshold * 255);
  // The resized matte matches w*h; read the (first) channel.
  for (let i = 0; i < w * h; i++) {
    alpha[i] = data[i * ch] >= cut ? 255 : 0;
  }
  return alpha;
}

/**
 * Binary erosion by `px` pixels (4-neighbour min filter, `px` iterations).
 * Removes the halo ring around organic silhouettes.
 */
export function erode(mask: Uint8Array, w: number, h: number, px: number): Uint8Array {
  if (px <= 0) return mask;
  let src = mask;
  for (let iter = 0; iter < px; iter++) {
    const dst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let keep = src[i] === 255;
        if (keep && x > 0) keep = src[i - 1] === 255;
        if (keep && x < w - 1) keep = src[i + 1] === 255;
        if (keep && y > 0) keep = src[i - w] === 255;
        if (keep && y < h - 1) keep = src[i + w] === 255;
        dst[i] = keep ? 255 : 0;
      }
    }
    src = dst;
  }
  return src;
}

/** Zero out mask pixels outside a centered ellipse (the scan zone). */
export function applyRoi(
  mask: Uint8Array,
  w: number,
  h: number,
  rxFrac: number,
  ryFrac: number,
): Uint8Array {
  const cx = w / 2;
  const cy = h / 2;
  const rx = w * rxFrac;
  const ry = h * ryFrac;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const dy = (y - cy) / ry;
    for (let x = 0; x < w; x++) {
      const dx = (x - cx) / rx;
      out[y * w + x] = dx * dx + dy * dy <= 1 ? mask[y * w + x] : 0;
    }
  }
  return out;
}

/**
 * Keep only the largest 4-connected foreground component. Iterative flood fill
 * with a typed-array stack (safe for large masks).
 */
export function largestComponent(mask: Uint8Array, w: number, h: number): Uint8Array {
  const n = w * h;
  const label = new Int32Array(n).fill(-1);
  const stack = new Int32Array(n);
  let bestLabel = -1;
  let bestSize = 0;
  let cur = 0;
  const sizes: number[] = [];

  for (let start = 0; start < n; start++) {
    if (mask[start] !== 255 || label[start] !== -1) continue;
    let sp = 0;
    stack[sp++] = start;
    label[start] = cur;
    let size = 0;
    while (sp > 0) {
      const p = stack[--sp];
      size++;
      const x = p % w;
      const y = (p / w) | 0;
      if (x > 0 && mask[p - 1] === 255 && label[p - 1] === -1) { label[p - 1] = cur; stack[sp++] = p - 1; }
      if (x < w - 1 && mask[p + 1] === 255 && label[p + 1] === -1) { label[p + 1] = cur; stack[sp++] = p + 1; }
      if (y > 0 && mask[p - w] === 255 && label[p - w] === -1) { label[p - w] = cur; stack[sp++] = p - w; }
      if (y < h - 1 && mask[p + w] === 255 && label[p + w] === -1) { label[p + w] = cur; stack[sp++] = p + w; }
    }
    sizes[cur] = size;
    if (size > bestSize) { bestSize = size; bestLabel = cur; }
    cur++;
  }
  if (bestLabel < 0) return mask;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = label[i] === bestLabel ? 255 : 0;
  return out;
}

export function disposeSegmenter(): void {
  model?.dispose?.();
  model = null;
  processor = null;
}
