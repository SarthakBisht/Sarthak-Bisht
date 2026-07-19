import { pipeline, RawImage } from '@huggingface/transformers';
import type { BackgroundRemovalPipeline } from '@huggingface/transformers';
import type { Keyframe, Matte, ProgressFn } from '../types';
import { getConfig } from '../config';
import { configureEnv, pickDevice } from '../pipeline/env';

/**
 * Client-side matte for irregular organic silhouettes (driftwood, rock) using
 * the RMBG background-removal model. We take the model's alpha channel as a soft
 * matte, threshold it, then ERODE by a few pixels to drop halo/fringe pixels
 * that would otherwise smear back-projected geometry into the background.
 */

let segmenter: BackgroundRemovalPipeline | null = null;

async function getSegmenter(): Promise<BackgroundRemovalPipeline> {
  if (segmenter) return segmenter;
  configureEnv();
  const cfg = getConfig();
  const device = await pickDevice();
  segmenter = (await pipeline('background-removal', cfg.matte.modelId, {
    device,
  })) as BackgroundRemovalPipeline;
  return segmenter;
}

export async function matteKeyframes(
  keyframes: Keyframe[],
  onProgress?: ProgressFn,
): Promise<Matte[]> {
  const cfg = getConfig();
  const seg = await getSegmenter();
  const out: Matte[] = [];

  for (let i = 0; i < keyframes.length; i++) {
    const kf = keyframes[i];
    const input = new RawImage(kf.rgba, kf.width, kf.height, 4);
    const result = await seg(input);
    const cutout = Array.isArray(result) ? result[0] : result;
    const alpha = extractAlpha(cutout, kf.width, kf.height, cfg.matte.threshold);
    const eroded = erode(alpha, kf.width, kf.height, cfg.matte.erosionPx);
    out.push({ width: kf.width, height: kf.height, alpha: eroded });
    onProgress?.('matte', (i + 1) / keyframes.length, `frame ${i + 1}/${keyframes.length}`);
  }
  return out;
}

/** Pull the alpha channel from an RGBA cutout, thresholded to a binary mask. */
function extractAlpha(img: RawImage, w: number, h: number, threshold: number): Uint8Array {
  const data = img.data;
  const ch = img.channels;
  const alpha = new Uint8Array(w * h);
  const cut = Math.round(threshold * 255);
  // Cutout may be at a different resolution than the source; resample nearest.
  const sw = img.width;
  const sh = img.height;
  for (let y = 0; y < h; y++) {
    const sy = Math.min(sh - 1, (y * sh / h) | 0);
    for (let x = 0; x < w; x++) {
      const sx = Math.min(sw - 1, (x * sw / w) | 0);
      const a = ch === 4 ? data[(sy * sw + sx) * ch + 3] : data[(sy * sw + sx) * ch];
      alpha[y * w + x] = a >= cut ? 255 : 0;
    }
  }
  return alpha;
}

/**
 * Binary erosion by `px` pixels (approximated with a separable box min filter
 * over `px` iterations). Removes the halo ring around organic silhouettes.
 */
export function erode(mask: Uint8Array, w: number, h: number, px: number): Uint8Array {
  if (px <= 0) return mask;
  let src = mask;
  for (let iter = 0; iter < px; iter++) {
    const dst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        // 4-neighbour min: foreground only if all neighbours are foreground.
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

export function disposeSegmenter(): void {
  segmenter?.dispose?.();
  segmenter = null;
}
