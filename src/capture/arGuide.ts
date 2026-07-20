/**
 * Lightweight on-screen "AR" placement guide using the phone's orientation
 * sensors (DeviceOrientation). It shows a bubble-level / horizon so the user
 * holds the phone LEVEL (no side roll) and at a consistent slight downward
 * angle — keeping the real camera angle steady reduces visual-hull distortion.
 *
 * No WebXR, no install. On Android/Chrome the sensor works over HTTPS without a
 * prompt; iOS needs a tap-triggered permission request (see enable()). Degrades
 * gracefully (draws nothing) if unavailable.
 */
export class ARGuide {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private raf = 0;
  private have = false;
  private roll = 0; // gamma: left-right tilt (deg)
  private listening = false;

  constructor(parent: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'ar-guide';
    parent.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    this.resize();
    window.addEventListener('resize', this.resize);
    // Android/desktop: attach immediately (no permission needed).
    if (!needsPermission()) this.attach();
    this.loop();
  }

  /** iOS permission is gesture-gated — call from a button tap. */
  async enable(): Promise<void> {
    if (this.listening) return;
    const DOE = window.DeviceOrientationEvent as unknown as {
      requestPermission?: () => Promise<'granted' | 'denied'>;
    };
    if (DOE && typeof DOE.requestPermission === 'function') {
      try {
        const res = await DOE.requestPermission();
        if (res !== 'granted') return;
      } catch {
        return;
      }
    }
    this.attach();
  }

  private attach() {
    if (this.listening) return;
    this.listening = true;
    window.addEventListener('deviceorientation', this.onOrient);
  }

  private onOrient = (e: DeviceOrientationEvent) => {
    if (e.gamma == null && e.beta == null) return;
    this.have = true;
    this.roll = e.gamma ?? 0;
  };

  /** Is the phone held level (no side roll)? */
  isLevel(): boolean {
    return this.have && Math.abs(this.roll) < 6;
  }

  private resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = 84;
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const ctx = this.ctx;
    const s = 84;
    ctx.clearRect(0, 0, s, s);
    if (!this.have) return;

    const cx = s / 2, cy = s / 2, r = s * 0.42;
    const level = this.isLevel();
    const col = level ? '#4fd1c5' : '#ffd166';

    // Dial background.
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(6,9,12,0.55)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Horizon line tilted by roll (gamma).
    const a = (this.roll * Math.PI) / 180;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(a);
    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-r * 0.8, 0);
    ctx.lineTo(r * 0.8, 0);
    ctx.stroke();
    // Bubble offset by roll.
    ctx.beginPath();
    ctx.arc(Math.max(-r * 0.7, Math.min(r * 0.7, -this.roll * 1.5)), 0, 4, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.restore();

    // Label.
    ctx.fillStyle = col;
    ctx.font = '700 10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(level ? 'LEVEL' : 'tilt', cx, cy + r + 12);
  };

  dispose() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.resize);
    window.removeEventListener('deviceorientation', this.onOrient);
    this.canvas.remove();
  }
}

function needsPermission(): boolean {
  const DOE = window.DeviceOrientationEvent as unknown as { requestPermission?: unknown } | undefined;
  return !!DOE && typeof DOE.requestPermission === 'function';
}
