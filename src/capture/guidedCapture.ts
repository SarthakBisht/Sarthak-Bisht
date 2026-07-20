import type { Keyframe } from '../types';
import { getConfig } from '../config';
import { varianceOfLaplacian } from './keyframes';
import { drawCoverageSphere } from './coverageSphere';

/**
 * Guided discrete capture. Instead of shaky video with guessed poses, the user
 * aligns to N target viewpoints; when the heading matches a target and the
 * framing is good and steady, we auto-snap a sharp still tagged with that
 * target's EXACT azimuth. Accurate, known-pose inputs → accurate output.
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
  private readonly targets: number[]; // azimuths [0..2π)
  private captured: (Keyframe | null)[];
  private count = 0;

  // metrics
  private prevLuma: Float32Array | null = null;
  private azimuth = 0;
  private flowDx = 0;
  private lastDir = 1;
  private lastNow = 0;
  private motion = 0;
  private brightness = 128;
  private sharp = 0;
  private baseCy = -1;
  private bbox = { cx: 0.5, cy: 0.5, w: 0, h: 0, found: false };
  private cooldownUntil = 0;

  private readonly SW = 96;
  private readonly SH = 128;

  constructor(parent: HTMLElement, video: HTMLVideoElement, opts: GuidedOpts = {}) {
    const cfg = getConfig();
    this.video = video;
    this.opts = opts;
    this.N = Math.max(4, cfg.guided.viewpoints);
    this.targets = Array.from({ length: this.N }, (_, i) => (i / this.N) * Math.PI * 2);
    this.captured = new Array(this.N).fill(null);

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'ring-guide';
    parent.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable.');
    this.ctx = ctx;

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
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.resize);
    this.canvas.remove();
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
    const dt = this.lastNow ? now - this.lastNow : 16;
    this.lastNow = now;
    if (this.running) {
      if (this.baseCy < 0 && this.bbox.found) this.baseCy = this.bbox.cy;
      // Advance heading estimate from measured rotation.
      const motionFactor = Math.min(1.6, this.motion / 0.02);
      if (Math.abs(this.flowDx) > 0.4) this.lastDir = this.flowDx > 0 ? 1 : -1;
      this.azimuth += this.lastDir * ((Math.PI * 2) / 12000) * motionFactor * dt;
      this.tryCapture(now);
    }
    this.draw();
  };

  private wrap(a: number): number {
    const t = a % (Math.PI * 2);
    return t < 0 ? t + Math.PI * 2 : t;
  }

  /** Nearest uncaptured target to the current heading; returns {index, dist}. */
  private nearestTarget(): { index: number; dist: number } {
    const az = this.wrap(this.azimuth);
    let best = Infinity;
    let bi = -1;
    for (let i = 0; i < this.N; i++) {
      if (this.captured[i]) continue;
      let d = Math.abs(az - this.targets[i]);
      d = Math.min(d, Math.PI * 2 - d);
      if (d < best) { best = d; bi = i; }
    }
    return { index: bi, dist: best };
  }

  private tryCapture(now: number) {
    if (this.count >= this.N || now < this.cooldownUntil) return;
    const cfg = getConfig();
    const tol = (cfg.guided.azimuthToleranceDeg * Math.PI) / 180;
    const { index, dist } = this.nearestTarget();
    if (index < 0) return;

    const framingOk =
      this.bbox.found &&
      Math.max(this.bbox.w, this.bbox.h) >= 0.28 &&
      Math.hypot(this.bbox.cx - 0.5, this.bbox.cy - 0.5) <= 0.18 &&
      this.brightness >= 45 &&
      (this.baseCy < 0 || Math.abs(this.bbox.cy - this.baseCy) <= 0.16);
    const still = !cfg.guided.requireStill || this.motion < 0.012;
    const sharpOk = this.sharp >= cfg.keyframes.blurRejectVar;

    if (dist <= tol && framingOk && still && sharpOk) {
      this.snap(index);
      this.cooldownUntil = now + 500;
    }
  }

  private snap(i: number) {
    this.capctx.drawImage(this.video, 0, 0, this.cap.width, this.cap.height);
    const img = this.capctx.getImageData(0, 0, this.cap.width, this.cap.height);
    this.captured[i] = {
      index: i,
      time: 0,
      width: this.cap.width,
      height: this.cap.height,
      rgba: img.data,
      sharpness: this.sharp,
    };
    this.count++;
    this.opts.onProgress?.(this.count, this.N);
    if (this.count >= this.N) {
      const kfs: Keyframe[] = [];
      const az: number[] = [];
      for (let k = 0; k < this.N; k++) {
        const c = this.captured[k];
        if (!c) continue;
        c.index = kfs.length;
        kfs.push(c);
        az.push(this.targets[k]);
      }
      this.running = false;
      this.opts.onComplete?.(kfs, az);
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
      let bestDx = 0, bestSad = Infinity;
      for (let dx = -8; dx <= 8; dx++) {
        let sad = 0, cnt = 0;
        for (let y = 6; y < h - 6; y += 3) {
          for (let x = 10; x < w - 10; x += 3) { sad += Math.abs(luma[y * w + x] - prev[y * w + x + dx]); cnt++; }
        }
        sad /= cnt || 1;
        if (sad < bestSad) { bestSad = sad; bestDx = dx; }
      }
      this.flowDx = 0.5 * this.flowDx + 0.5 * bestDx;
    }
    this.prevLuma = luma;

    // Object framing bbox from edges.
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

  private coach(): { text: string; ok: boolean } {
    if (this.count >= this.N) return { text: '✅ All viewpoints captured!', ok: true };
    if (this.brightness < 45) return { text: '💡 Too dark — add even light', ok: false };
    if (!this.bbox.found || Math.max(this.bbox.w, this.bbox.h) < 0.28)
      return { text: '🔍 Move closer — fill the circle', ok: false };
    if (Math.hypot(this.bbox.cx - 0.5, this.bbox.cy - 0.5) > 0.18)
      return { text: '🎯 Center the object', ok: false };
    if (this.baseCy >= 0 && Math.abs(this.bbox.cy - this.baseCy) > 0.16)
      return { text: '📏 Keep the phone level', ok: false };
    const { dist } = this.nearestTarget();
    const tol = (getConfig().guided.azimuthToleranceDeg * Math.PI) / 180;
    if (dist <= tol) {
      if (this.motion >= 0.012) return { text: '✋ Hold still — capturing…', ok: true };
      return { text: '📸 Capturing…', ok: true };
    }
    return { text: `↻ Turn to the next lit target (${this.count}/${this.N})`, ok: true };
  }

  private draw() {
    const ctx = this.ctx;
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    const cw = rect.width, ch = rect.height;
    ctx.clearRect(0, 0, cw, ch);
    const cx = cw / 2, cy = ch * 0.46;
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

    const coach = this.coach();
    ctx.lineWidth = 3;
    ctx.strokeStyle = coach.ok ? 'rgba(79,209,197,0.9)' : 'rgba(255,209,102,0.95)';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Target ticks around the ring.
    const ringR = Math.max(rx, ry) + 16;
    const next = this.nearestTarget().index;
    for (let i = 0; i < this.N; i++) {
      const a = this.targets[i] - Math.PI / 2;
      const tx = cx + Math.cos(a) * ringR;
      const ty = cy + Math.sin(a) * ringR;
      ctx.beginPath();
      ctx.arc(tx, ty, ringR * 0.05, 0, Math.PI * 2);
      if (this.captured[i]) ctx.fillStyle = '#4fd1c5';
      else if (i === next) ctx.fillStyle = '#ffd166';
      else ctx.fillStyle = 'rgba(255,255,255,0.28)';
      ctx.fill();
    }

    // Current heading marker.
    const ha = this.azimuth - Math.PI / 2;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(ha) * ringR, cy + Math.sin(ha) * ringR, ringR * 0.03, 0, Math.PI * 2);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Coverage sphere (captured longitudes lit).
    const sphR = Math.min(cw, ch) * 0.11;
    const covered = this.captured.map((c) => !!c);
    drawCoverageSphere(ctx, cw - sphR - 14, sphR + 18, sphR, this.azimuth, covered, this.N);

    // Center reticle.
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
    ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
    ctx.stroke();

    // Count.
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '700 15px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${this.count} / ${this.N} captured`, cx, cy + ry + 40);

    this.opts.onCoach?.(coach.text, coach.ok);
  }
}
