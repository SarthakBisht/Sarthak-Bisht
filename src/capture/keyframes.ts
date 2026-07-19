import type { Keyframe, ProgressFn } from '../types';
import { getConfig } from '../config';

/**
 * Extract evenly-spaced keyframes from a turntable clip (recorded Blob or a
 * bundled sample URL), reject motion-blurred frames, and cap resolution to
 * bound downstream memory.
 *
 * Uses an <video> element + seeking. This is the most compatible way to decode
 * arbitrary user-recorded video across mobile Chrome without WebCodecs quirks.
 */
export async function extractKeyframes(
  source: Blob | string,
  onProgress?: ProgressFn,
): Promise<Keyframe[]> {
  const cfg = getConfig();
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const url = typeof source === 'string' ? source : URL.createObjectURL(source);
  video.src = url;

  try {
    await once(video, 'loadedmetadata');
    // Some browsers need a play()/pause() nudge before seeking is reliable.
    try {
      await video.play();
      video.pause();
    } catch {
      /* autoplay may be blocked; seeking still works */
    }

    const duration = isFinite(video.duration) ? video.duration : await probeDuration(video);
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) throw new Error('Could not read video dimensions.');

    const scale = Math.min(1, cfg.keyframes.maxEdgePx / Math.max(vw, vh));
    const w = Math.max(2, Math.round(vw * scale));
    const h = Math.max(2, Math.round(vh * scale));

    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D canvas unavailable.');

    // Oversample so we can drop blurry frames and still hit the target count.
    const oversample = Math.ceil(cfg.keyframes.count * 1.6);
    // Avoid the very first/last frames (often black / motion-heavy).
    const t0 = duration * 0.02;
    const t1 = duration * 0.98;

    const candidates: Keyframe[] = [];
    for (let i = 0; i < oversample; i++) {
      const time = t0 + ((t1 - t0) * i) / (oversample - 1);
      await seek(video, time);
      ctx.drawImage(video, 0, 0, w, h);
      const img = ctx.getImageData(0, 0, w, h);
      const sharpness = varianceOfLaplacian(img.data, w, h);
      candidates.push({
        index: i,
        time,
        width: w,
        height: h,
        rgba: img.data,
        sharpness,
      });
      onProgress?.('keyframes', (i + 1) / oversample, `frame ${i + 1}/${oversample}`);
    }

    const selected = selectSharpEvenly(candidates, cfg.keyframes.count, cfg.keyframes.blurRejectVar);
    // Re-index sequentially for the turntable pose model.
    selected.forEach((k, idx) => (k.index = idx));
    return selected;
  } finally {
    video.removeAttribute('src');
    video.load();
    if (typeof source !== 'string') URL.revokeObjectURL(url);
  }
}

/**
 * Choose `count` frames spread evenly across the clip, but within each slot
 * prefer the sharpest candidate above the blur threshold. Guarantees even
 * angular coverage while dropping motion blur.
 */
function selectSharpEvenly(cands: Keyframe[], count: number, blurVar: number): Keyframe[] {
  const n = cands.length;
  const out: Keyframe[] = [];
  const used = new Set<number>();
  for (let s = 0; s < count; s++) {
    const center = ((s + 0.5) / count) * n;
    const lo = Math.max(0, Math.floor(center - n / (2 * count)));
    const hi = Math.min(n - 1, Math.ceil(center + n / (2 * count)));
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = lo; i <= hi; i++) {
      if (used.has(i)) continue;
      const sharp = cands[i].sharpness;
      // Reward sharpness; heavily penalise sub-threshold blur.
      const score = sharp < blurVar ? sharp - blurVar * 2 : sharp;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    if (bestIdx < 0) bestIdx = Math.min(n - 1, Math.round(center));
    used.add(bestIdx);
    out.push(cands[bestIdx]);
  }
  return out.sort((a, b) => a.time - b.time);
}

/**
 * Variance of the Laplacian — a standard focus/blur metric. Higher = sharper.
 * Operates on luma of a downsampled grid for speed.
 */
export function varianceOfLaplacian(rgba: Uint8ClampedArray, w: number, h: number): number {
  const step = Math.max(1, Math.floor(Math.min(w, h) / 240));
  const luma = (x: number, y: number): number => {
    const i = (y * w + x) * 4;
    return 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  };
  let mean = 0;
  let m2 = 0;
  let count = 0;
  for (let y = step; y < h - step; y += step) {
    for (let x = step; x < w - step; x += step) {
      const lap =
        luma(x - step, y) +
        luma(x + step, y) +
        luma(x, y - step) +
        luma(x, y + step) -
        4 * luma(x, y);
      count++;
      const delta = lap - mean;
      mean += delta / count;
      m2 += delta * (lap - mean);
    }
  }
  return count > 1 ? m2 / (count - 1) : 0;
}

// ---- video helpers ---------------------------------------------------------

function once(el: HTMLMediaElement, ev: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ok = () => {
      cleanup();
      resolve();
    };
    const err = () => {
      cleanup();
      reject(new Error(`Video failed to ${ev}.`));
    };
    const cleanup = () => {
      el.removeEventListener(ev, ok);
      el.removeEventListener('error', err);
    };
    el.addEventListener(ev, ok, { once: true });
    el.addEventListener('error', err, { once: true });
  });
}

function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onErr);
      resolve();
    };
    const onErr = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onErr);
      reject(new Error('Seek failed.'));
    };
    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onErr);
    video.currentTime = Math.max(0, time);
  });
}

/** Fallback duration probe for streams that report Infinity initially. */
async function probeDuration(video: HTMLVideoElement): Promise<number> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      const d = isFinite(video.duration) ? video.duration : video.currentTime;
      video.currentTime = 0;
      resolve(d || 1);
    };
    video.addEventListener('seeked', onSeeked);
    // Seek far ahead to force the browser to resolve real duration.
    video.currentTime = 1e6;
  });
}
