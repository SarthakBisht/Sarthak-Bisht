import { drawCoverageSphere } from './coverageSphere';

/**
 * Live capture HUD drawn over the camera preview. It coaches the user in real
 * time so the resulting clip actually reconstructs well:
 *
 *   - SCAN ZONE      : a bright central ellipse marks what will be captured;
 *                      everything outside is dimmed and labelled "background
 *                      (ignored)" so the user sees what is / isn't scanned.
 *   - FRAMING        : estimates the object's edge bounding box and nudges
 *                      "center the object" / "move closer" / "move back".
 *   - STEADINESS     : frame-to-frame motion meter → "slow down (blur)" vs
 *                      "keep rotating"; this is the "how long to hold" cue.
 *   - COVERAGE       : a ring fills with elapsed angle, with a leading marker,
 *                      a turn-direction arrow, remaining seconds, and per-
 *                      quarter milestones.
 *
 * All analysis is cheap 2D canvas work on a downscaled copy of the preview — no
 * ML runs during capture, so recording stays smooth.
 */
export interface CoachState {
  primary: string; // main instruction
  ok: boolean; // whether current framing/motion is good enough
  coverage: number; // 0..1
}

export class ScanGuide {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private sample: OffscreenCanvas;
  private sctx: OffscreenCanvasRenderingContext2D;
  private video: HTMLVideoElement;
  private raf = 0;
  private durationMs: number;
  private segments: number;
  private filled: boolean[];
  private running = false;
  private prevLuma: Float32Array | null = null;
  private lastAnalysis = 0;
  private onCoach?: (s: CoachState) => void;

  // Smoothed metrics.
  private motion = 0;
  private brightness = 128;
  private bbox = { cx: 0.5, cy: 0.5, w: 0, h: 0, found: false };

  // Rotation estimate (from optical flow) → azimuth-based coverage.
  private azimuth = 0;
  private flowDx = 0;
  private lastDir = 1;
  private lastNow = 0;
  private baseCy = -1; // object vertical centroid at record start (tilt reference)

  private readonly SW = 96;
  private readonly SH = 128;

  constructor(
    parent: HTMLElement,
    video: HTMLVideoElement,
    durationSeconds: number,
    onCoach?: (s: CoachState) => void,
    segments = 24,
  ) {
    this.video = video;
    this.onCoach = onCoach;
    this.durationMs = durationSeconds * 1000;
    this.segments = segments;
    this.filled = new Array(segments).fill(false);

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'ring-guide';
    parent.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable for scan guide.');
    this.ctx = ctx;

    this.sample = new OffscreenCanvas(this.SW, this.SH);
    const sctx = this.sample.getContext('2d', { willReadFrequently: true });
    if (!sctx) throw new Error('2D sample canvas unavailable.');
    this.sctx = sctx;

    this.resize();
    window.addEventListener('resize', this.resize);
    this.loop(); // draw idle immediately
  }

  coverage(): number {
    return this.filled.filter(Boolean).length / this.segments;
  }

  start() {
    this.filled.fill(false);
    this.azimuth = 0;
    this.lastDir = 1;
    this.baseCy = -1;
    this.running = true;
  }

  stop() {
    this.running = false;
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
    // Analyse the preview a few times per second.
    if (now - this.lastAnalysis > 90 && this.video.readyState >= 2) {
      this.analyse();
      this.lastAnalysis = now;
    }
    const dtNow = this.lastNow ? now - this.lastNow : 16;
    this.lastNow = now;
    if (this.running) {
      if (this.baseCy < 0 && this.bbox.found) this.baseCy = this.bbox.cy;
      // Advance azimuth by actual rotation: rate scales with measured motion,
      // direction follows optical flow. Pausing stops progress; a steady full
      // pass fills the ring over roughly the target duration.
      const baseRate = (Math.PI * 2) / this.durationMs;
      const motionFactor = Math.min(1.6, this.motion / 0.02);
      if (Math.abs(this.flowDx) > 0.4) this.lastDir = this.flowDx > 0 ? 1 : -1;
      this.azimuth += this.lastDir * baseRate * motionFactor * dtNow;
      const twoPi = Math.PI * 2;
      const sec =
        ((Math.floor((this.azimuth / twoPi) * this.segments) % this.segments) + this.segments) %
        this.segments;
      this.filled[sec] = true;
    }
    this.draw();
  };

  /** Cheap per-frame analysis: brightness, motion, object edge bbox. */
  private analyse() {
    try {
      this.sctx.drawImage(this.video, 0, 0, this.SW, this.SH);
    } catch {
      return;
    }
    const img = this.sctx.getImageData(0, 0, this.SW, this.SH).data;
    const w = this.SW;
    const h = this.SH;
    const luma = new Float32Array(w * h);
    let brightSum = 0;
    for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
      const l = 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2];
      luma[i] = l;
      brightSum += l;
    }
    this.brightness = 0.6 * this.brightness + 0.4 * (brightSum / luma.length);

    // Motion vs previous sample.
    if (this.prevLuma) {
      const prev = this.prevLuma;
      let diff = 0;
      for (let i = 0; i < luma.length; i++) diff += Math.abs(luma[i] - prev[i]);
      const norm = diff / (luma.length * 255);
      this.motion = 0.5 * this.motion + 0.5 * norm;

      // Signed horizontal shift (block match) → rotation direction & speed.
      let bestDx = 0;
      let bestSad = Infinity;
      for (let dx = -8; dx <= 8; dx++) {
        let sad = 0;
        let cnt = 0;
        for (let y = 6; y < h - 6; y += 3) {
          for (let x = 10; x < w - 10; x += 3) {
            sad += Math.abs(luma[y * w + x] - prev[y * w + x + dx]);
            cnt++;
          }
        }
        sad /= cnt || 1;
        if (sad < bestSad) {
          bestSad = sad;
          bestDx = dx;
        }
      }
      this.flowDx = 0.5 * this.flowDx + 0.5 * bestDx;
    }
    this.prevLuma = luma;

    // Object framing: bbox of high-gradient pixels (edges) within the frame.
    let minX = w, minY = h, maxX = 0, maxY = 0, sumX = 0, sumY = 0, count = 0;
    const thr = 26;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = Math.abs(luma[i + 1] - luma[i - 1]);
        const gy = Math.abs(luma[i + w] - luma[i - w]);
        if (gx + gy > thr) {
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          sumX += x; sumY += y; count++;
        }
      }
    }
    if (count > 40) {
      this.bbox = {
        cx: sumX / count / w,
        cy: sumY / count / h,
        w: (maxX - minX) / w,
        h: (maxY - minY) / h,
        found: true,
      };
    } else {
      this.bbox = { cx: 0.5, cy: 0.5, w: 0, h: 0, found: false };
    }
  }

  /** Decide the single most important instruction to show. */
  private coach(): CoachState {
    const cov = this.coverage();
    let primary = '';
    let ok = true;

    if (this.brightness < 45) {
      primary = '💡 Too dark — add even light';
      ok = false;
    } else if (!this.bbox.found || Math.max(this.bbox.w, this.bbox.h) < 0.28) {
      primary = '🔍 Move closer — fill the circle with the object';
      ok = false;
    } else if (this.bbox.w > 0.92 || this.bbox.h > 0.92) {
      primary = '↔️ Move back — keep the whole object in frame';
      ok = false;
    } else if (Math.hypot(this.bbox.cx - 0.5, this.bbox.cy - 0.5) > 0.16) {
      primary = `🎯 Center the object ${this.arrow()}`;
      ok = false;
    } else if (
      this.running &&
      this.baseCy >= 0 &&
      this.bbox.found &&
      Math.abs(this.bbox.cy - this.baseCy) > 0.14
    ) {
      primary = '📏 Keep the phone level — don’t tilt up/down';
      ok = false;
    } else if (this.running && this.motion > 0.055) {
      primary = '🐢 Slower — hold steady to avoid blur';
      ok = false;
    } else if (this.running && this.motion < 0.006) {
      primary = '🔄 Keep rotating — fill the grey sides on the sphere';
    } else if (this.running) {
      if (cov > 0.92) primary = '✅ Full coverage — you can stop';
      else primary = `✅ Keep turning ▶ — fill grey sides (${Math.round(cov * 100)}% covered)`;
    } else {
      primary = '✅ Framing looks good — press Start';
    }
    return { primary, ok, coverage: cov };
  }

  private arrow(): string {
    const dx = 0.5 - this.bbox.cx;
    const dy = 0.5 - this.bbox.cy;
    if (Math.abs(dx) > Math.abs(dy)) return dx > 0 ? '➡️' : '⬅️';
    return dy > 0 ? '⬇️' : '⬆️';
  }

  private draw() {
    const ctx = this.ctx;
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;
    ctx.clearRect(0, 0, cw, ch);

    const cx = cw / 2;
    const cy = ch * 0.46;
    const rx = Math.min(cw, ch) * 0.34;
    const ry = rx * 1.15;

    // Dim the background (everything outside the scan-zone ellipse).
    ctx.save();
    ctx.fillStyle = 'rgba(6,9,12,0.58)';
    ctx.fillRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Scan-zone outline.
    const coach = this.coach();

    ctx.lineWidth = 3;
    ctx.strokeStyle = coach.ok ? 'rgba(79,209,197,0.9)' : 'rgba(255,209,102,0.95)';
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Coverage ring (just outside the ellipse).
    const ringR = Math.max(rx, ry) + 16;
    ctx.lineWidth = Math.max(6, ringR * 0.05);
    for (let i = 0; i < this.segments; i++) {
      const a0 = (i / this.segments) * Math.PI * 2 - Math.PI / 2;
      const a1 = ((i + 1) / this.segments) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, ringR, a0 + 0.03, a1 - 0.03);
      ctx.strokeStyle = this.filled[i] ? '#4fd1c5' : 'rgba(255,255,255,0.22)';
      ctx.stroke();
    }

    // Leading marker at the current (flow-estimated) heading.
    if (this.running) {
      const a = this.azimuth - Math.PI / 2;
      const mx = cx + Math.cos(a) * ringR;
      const my = cy + Math.sin(a) * ringR;
      ctx.beginPath();
      ctx.arc(mx, my, ringR * 0.08, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd166';
      ctx.fill();
    }

    // Live coverage sphere in the top-right corner — grey sides still need the
    // camera; teal sides are captured.
    const sphR = Math.min(cw, ch) * 0.11;
    drawCoverageSphere(ctx, cw - sphR - 14, sphR + 18, sphR, this.azimuth, this.filled, this.segments);

    // Detected-object box (what is being scanned) inside the zone.
    if (this.bbox.found) {
      const bw = this.bbox.w * rx * 2;
      const bh = this.bbox.h * ry * 2;
      const bx = cx + (this.bbox.cx - 0.5) * cw;
      const by = cy + (this.bbox.cy - 0.5) * ch * 0.9;
      ctx.strokeStyle = 'rgba(79,209,197,0.5)';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      ctx.strokeRect(bx - bw / 2, by - bh / 2, bw, bh);
      ctx.setLineDash([]);
    }

    // Centre reticle.
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
    ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
    ctx.stroke();

    // Labels.
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillText('background — ignored', cx, ch - 14);
    ctx.fillStyle = coach.ok ? 'rgba(79,209,197,0.95)' : 'rgba(255,209,102,0.95)';
    ctx.font = '700 12px system-ui, sans-serif';
    ctx.fillText('SCAN ZONE', cx, cy - ry - 26);

    // Coverage % + hold hint.
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '700 15px system-ui, sans-serif';
    ctx.fillText(`${Math.round(this.coverage() * 100)}% covered`, cx, cy + ry + 40);
    if (this.running) {
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '600 12px system-ui, sans-serif';
      ctx.fillText('hold ~1s at each side, turn evenly', cx, cy + ry + 60);
    }

    this.onCoach?.(coach);
  }
}
