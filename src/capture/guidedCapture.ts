import type { Keyframe } from '../types';
import { getConfig } from '../config';
import { varianceOfLaplacian } from './keyframes';
import { drawCoverageSphere } from './coverageSphere';

/**
 * Guided discrete capture — STEP based, no flaky heading tracking.
 *
 * The user takes N shots around the object. Poses are evenly spaced by
 * construction (N shots → 360/N), so the UI just shows "Shot k / N" and which
 * sides are done. Each shot fires either by tapping the shutter, or
 * automatically: we ARM on a rotation (motion spike), then capture once the
 * object settles to still with good framing — one shot per step.
 */
export interface GuidedOpts {
  onCoach?: (text: string, ok: boolean) => void;
  onProgress?: (captured: number, total: number) => void;
  onComplete?: (keyframes: Keyframe[], azimuthsRad: number[]) => void;
}

export class GuidedCapture {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sample: OffscreenCanvas;
  private sctx: OffscreenCanvasRenderingContext2D;
  private cap: OffscreenCanvas;
  private capctx: OffscreenCanvasRenderingContext2D;
  private video: HTMLVideoElement;
  private raf = 0;
  private running = false;
  private lastAnalysis = 0;
  private opts: GuidedOpts;

  private readonly N: number;
  private captured: Keyframe[] = [];

  // metrics
  private prevLuma: Float32Array | null = null;
  private motion = 0;
  private brightness = 128;
  private sharp = 0;
  private baseCy = -1;
  private bbox = { cx: 0.5, cy: 0.5, w: 0, h: 0, found: false };

  // auto-capture state
  private armed = true;
  private stillSince = 0;
  private cooldownUntil = 0;

  private readonly SW = 96;
  private readonly SH = 128;
  private readonly HOLD_MS = 350;

  constructor(parent: HTMLElement, video: HTMLVideoElement, opts: GuidedOpts = {}) {
    const cfg = getConfig();
    this.video = video;
    this.opts = opts;
    this.N = Math.max(4, cfg.guided.viewpoints);

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'ring-guide';
    parent.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;

    this.sample = new OffscreenCanvas(this.SW, this.SH);
    this.sctx = this.sample.getContext('2d', { willReadFrequently: true })!;

    const maxEdge = cfg.keyframes.maxEdgePx;
    const vw = video.videoWidth || 720;
    const vh = video.videoHeight || 960;
    const scale = Math.min(1, maxEdge / Math.max(vw, vh));
    this.cap = new OffscreenCanvas(Math.max(2, Math.round(vw * scale)), Math.max(2, Math.round(vh * scale)));
    this.capctx = this.cap.getContext('2d', { willReadFrequently: true })!;

    this.resize();
    window.addEventListener('resize', this.resize);
    this.loop();
  }

  start() {
    this.running = true;
    this.armed = true;
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.resize);
    this.canvas.remove();
  }

  /** Manual shutter — capture the current view immediately. */
  captureNow() {
    if (!this.running || this.captured.length >= this.N) return;
    this.capture();
  }

  private resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const now = performance.now();
    if (now - this.lastAnalysis > 90 && this.video.readyState >= 2) {
      this.analyse();
      this.lastAnalysis = now;
    }
    if (this.running) this.autoCapture(now);
    this.draw(now);
  };

  private framingOk(): boolean {
    return (
      this.bbox.found &&
      Math.max(this.bbox.w, this.bbox.h) >= 0.28 &&
      Math.hypot(this.bbox.cx - 0.5, this.bbox.cy - 0.5) <= 0.18 &&
      this.brightness >= 45 &&
      (this.baseCy < 0 || Math.abs(this.bbox.cy - this.baseCy) <= 0.16)
    );
  }

  private autoCapture(now: number) {
    if (this.captured.length >= this.N || now < this.cooldownUntil) return;
    const cfg = getConfig();
    const moveThresh = 0.02;
    const stillThresh = 0.01;

    // Re-arm when the user rotates (motion spike).
    if (this.motion > moveThresh) {
      this.armed = true;
      this.stillSince = 0;
      return;
    }
    if (!this.armed) return;
    // Settled? track how long we've been still.
    const still = this.motion < stillThresh;
    if (!still) { this.stillSince = 0; return; }
    if (this.stillSince === 0) this.stillSince = now;

    const sharpOk = this.sharp >= cfg.keyframes.blurRejectVar;
    if (now - this.stillSince >= this.HOLD_MS && this.framingOk() && sharpOk) {
      this.capture();
    }
  }

  private capture() {
    const k = this.captured.length;
    if (k >= this.N) return;
    if (this.baseCy < 0 && this.bbox.found) this.baseCy = this.bbox.cy;
    this.capctx.drawImage(this.video, 0, 0, this.cap.width, this.cap.height);
    const img = this.capctx.getImageData(0, 0, this.cap.width, this.cap.height);
    this.captured.push({
      index: k,
      time: 0,
      width: this.cap.width,
      height: this.cap.height,
      rgba: img.data,
      sharpness: this.sharp,
    });
    this.armed = false;
    this.stillSince = 0;
    this.cooldownUntil = performance.now() + 500;
    this.opts.onProgress?.(this.captured.length, this.N);

    if (this.captured.length >= this.N) {
      const az = this.captured.map((_, i) => (i / this.N) * Math.PI * 2);
      this.running = false;
      this.opts.onComplete?.(this.captured, az);
    }
  }

  private analyse() {
    try {
      this.sctx.drawImage(this.video, 0, 0, this.SW, this.SH);
    } catch {
      return;
    }
    const data = this.sctx.getImageData(0, 0, this.SW, this.SH).data;
    const w = this.SW, h = this.SH;
    const luma = new Float32Array(w * h);
    let brightSum = 0;
    for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
      const l = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      luma[i] = l;
      brightSum += l;
    }
    this.brightness = 0.6 * this.brightness + 0.4 * (brightSum / luma.length);
    this.sharp = varianceOfLaplacian(data, w, h);

    if (this.prevLuma) {
      const prev = this.prevLuma;
      let diff = 0;
      for (let i = 0; i < luma.length; i++) diff += Math.abs(luma[i] - prev[i]);
      this.motion = 0.5 * this.motion + 0.5 * (diff / (luma.length * 255));
    }
    this.prevLuma = luma;

    let minX = w, minY = h, maxX = 0, maxY = 0, sumX = 0, sumY = 0, count = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = Math.abs(luma[i + 1] - luma[i - 1]);
        const gy = Math.abs(luma[i + w] - luma[i - w]);
        if (gx + gy > 26) {
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          sumX += x; sumY += y; count++;
        }
      }
    }
    this.bbox = count > 40
      ? { cx: sumX / count / w, cy: sumY / count / h, w: (maxX - minX) / w, h: (maxY - minY) / h, found: true }
      : { cx: 0.5, cy: 0.5, w: 0, h: 0, found: false };
  }

  private coach(now: number): { text: string; ok: boolean } {
    const k = this.captured.length;
    if (k >= this.N) return { text: '✅ All shots captured!', ok: true };
    if (this.brightness < 45) return { text: '💡 Too dark — add even light', ok: false };
    if (!this.bbox.found || Math.max(this.bbox.w, this.bbox.h) < 0.28)
      return { text: '🔍 Move closer — fill the circle', ok: false };
    if (Math.hypot(this.bbox.cx - 0.5, this.bbox.cy - 0.5) > 0.18)
      return { text: '🎯 Center the object', ok: false };
    if (this.baseCy >= 0 && Math.abs(this.bbox.cy - this.baseCy) > 0.16)
      return { text: '📏 Keep the phone level', ok: false };
    if (this.armed && this.motion < 0.01 && this.stillSince > 0 && now - this.stillSince < this.HOLD_MS)
      return { text: '✋ Hold still — capturing…', ok: true };
    return { text: `↻ Rotate a step & hold — or tap Capture  (shot ${k + 1}/${this.N})`, ok: true };
  }

  private draw(now: number) {
    const ctx = this.ctx;
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    const cw = rect.width, ch = rect.height;
    ctx.clearRect(0, 0, cw, ch);
    const cx = cw / 2, cy = ch * 0.44;
    const rx = Math.min(cw, ch) * 0.34, ry = rx * 1.15;

    // Dim background outside the scan-zone ellipse.
    ctx.save();
    ctx.fillStyle = 'rgba(6,9,12,0.55)';
    ctx.fillRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    const coach = this.coach(now);
    const k = this.captured.length;

    // Scan-zone outline.
    ctx.lineWidth = 3;
    ctx.strokeStyle = coach.ok ? 'rgba(79,209,197,0.9)' : 'rgba(255,209,102,0.95)';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Hold-still countdown ring while settling.
    if (this.armed && this.stillSince > 0 && k < this.N) {
      const frac = Math.min(1, (now - this.stillSince) / this.HOLD_MS);
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx + 8, ry + 8, 0, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
      ctx.strokeStyle = '#4fd1c5';
      ctx.lineWidth = 5;
      ctx.stroke();
    }

    // Center reticle.
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
    ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
    ctx.stroke();

    // Coverage sphere (fills one wedge per captured shot).
    const sphR = Math.min(cw, ch) * 0.11;
    const covered = Array.from({ length: this.N }, (_, i) => i < k);
    drawCoverageSphere(ctx, cw - sphR - 14, sphR + 18, sphR, (k / this.N) * Math.PI * 2, covered, this.N);

    // Big shot counter.
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    ctx.font = '800 22px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Shot ${Math.min(k + 1, this.N)} / ${this.N}`, cx, cy + ry + 44);

    // Progress dots row.
    const dotY = cy + ry + 66;
    const gap = Math.min(18, (cw - 40) / this.N);
    const startX = cx - (gap * (this.N - 1)) / 2;
    for (let i = 0; i < this.N; i++) {
      ctx.beginPath();
      ctx.arc(startX + i * gap, dotY, 4, 0, Math.PI * 2);
      ctx.fillStyle = i < k ? '#4fd1c5' : i === k ? '#ffd166' : 'rgba(255,255,255,0.28)';
      ctx.fill();
    }

    this.opts.onCoach?.(coach.text, coach.ok);
  }
}
