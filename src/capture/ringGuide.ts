/**
 * On-screen ring guide overlay drawn on a transparent canvas above the camera
 * preview. It shows a circular path and lights up wedges as the user sweeps the
 * phone around the object, enforcing even 360° coverage.
 *
 * Coverage is estimated from elapsed capture time (constant angular velocity
 * assumption — the same assumption the turntable pose model makes), so the
 * guide and the reconstruction agree on where "even" is.
 */
export class RingGuide {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private raf = 0;
  private startTime = 0;
  private durationMs: number;
  private segments: number;
  private filled: boolean[];
  private running = false;

  constructor(parent: HTMLElement, durationSeconds: number, segments = 24) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'ring-guide';
    parent.appendChild(this.canvas);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable for ring guide.');
    this.ctx = ctx;
    this.durationMs = durationSeconds * 1000;
    this.segments = segments;
    this.filled = new Array(segments).fill(false);
    this.resize();
    window.addEventListener('resize', this.resize);
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

  /** Coverage fraction 0..1 based on lit wedges. */
  coverage(): number {
    return this.filled.filter(Boolean).length / this.segments;
  }

  start() {
    this.startTime = performance.now();
    this.filled.fill(false);
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      this.draw();
      this.raf = requestAnimationFrame(loop);
    };
    loop();
  }

  /** Draw idle (pre-record) state without advancing coverage. */
  showIdle() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.draw(true);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  dispose() {
    this.stop();
    window.removeEventListener('resize', this.resize);
    this.canvas.remove();
  }

  private draw(idle = false) {
    const ctx = this.ctx;
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;
    ctx.clearRect(0, 0, cw, ch);

    const cx = cw / 2;
    const cy = ch / 2;
    const radius = Math.min(cw, ch) * 0.36;

    // Advance coverage from elapsed time (constant angular velocity).
    let progress = 0;
    if (!idle && this.running) {
      progress = Math.min(1, (performance.now() - this.startTime) / this.durationMs);
      const litCount = Math.floor(progress * this.segments);
      for (let i = 0; i < litCount; i++) this.filled[i] = true;
    }

    // Base ring.
    ctx.lineWidth = Math.max(6, radius * 0.06);
    for (let i = 0; i < this.segments; i++) {
      const a0 = (i / this.segments) * Math.PI * 2 - Math.PI / 2;
      const a1 = ((i + 1) / this.segments) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, a0 + 0.02, a1 - 0.02);
      ctx.strokeStyle = this.filled[i] ? '#4fd1c5' : 'rgba(255,255,255,0.22)';
      ctx.stroke();
    }

    // Leading marker.
    if (!idle && this.running) {
      const a = progress * Math.PI * 2 - Math.PI / 2;
      const mx = cx + Math.cos(a) * radius;
      const my = cy + Math.sin(a) * radius;
      ctx.beginPath();
      ctx.arc(mx, my, radius * 0.09, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd166';
      ctx.fill();
    }

    // Centre reticle + hint.
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.16, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = `600 ${Math.round(radius * 0.13)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    const pct = Math.round(this.coverage() * 100);
    ctx.fillText(idle ? 'Keep object centred' : `${pct}% coverage`, cx, cy + radius + 28);
  }
}
