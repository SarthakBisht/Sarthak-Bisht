/**
 * Draws a small low-poly coverage sphere (2D-projected, no GL context) that
 * shows which sides of the object have been captured. Longitudes that have been
 * covered light up teal; uncovered ones stay grey, so the user can see exactly
 * which side to point the camera at next. The sphere spins with the estimated
 * azimuth so its front face matches what the camera is currently seeing.
 *
 * Turntable capture only sweeps a latitude band, so the top/bottom caps are
 * intentionally left grey (with a "tilt for top/bottom" hint elsewhere).
 */
const LON = 20;
const LAT = 10;
const TILT = 0.42; // view slightly from above (radians)
const COVER_LAT = 0.95; // |sin(lat)| below this is considered coverable band

export function drawCoverageSphere(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  azimuth: number,
  covered: boolean[],
  sectors: number,
) {
  const cosT = Math.cos(TILT);
  const sinT = Math.sin(TILT);

  interface Face {
    pts: [number, number][];
    depth: number;
    covered: boolean;
    coverable: boolean;
    front: boolean;
  }
  const faces: Face[] = [];

  const project = (lonWorld: number, lat: number): { x: number; y: number; z: number } => {
    const lonView = lonWorld - azimuth;
    const cl = Math.cos(lat);
    const x = cl * Math.sin(lonView);
    const y0 = Math.sin(lat);
    const z0 = cl * Math.cos(lonView);
    // tilt around X so we look slightly down
    const y = y0 * cosT - z0 * sinT;
    const z = y0 * sinT + z0 * cosT;
    return { x, y, z };
  };

  for (let j = 0; j < LAT; j++) {
    const lat0 = -Math.PI / 2 + (j / LAT) * Math.PI;
    const lat1 = -Math.PI / 2 + ((j + 1) / LAT) * Math.PI;
    for (let i = 0; i < LON; i++) {
      const lonW0 = (i / LON) * Math.PI * 2;
      const lonW1 = ((i + 1) / LON) * Math.PI * 2;
      const c = [
        project(lonW0, lat0),
        project(lonW1, lat0),
        project(lonW1, lat1),
        project(lonW0, lat1),
      ];
      const depth = (c[0].z + c[1].z + c[2].z + c[3].z) / 4;
      const midLat = (lat0 + lat1) / 2;
      const coverable = Math.abs(Math.sin(midLat)) < COVER_LAT;
      const sector = Math.floor((i / LON) * sectors) % sectors;
      const pts = c.map((p) => [cx + p.x * radius, cy - p.y * radius] as [number, number]);
      faces.push({
        pts,
        depth,
        covered: coverable && !!covered[sector],
        coverable,
        front: depth > 0,
      });
    }
  }

  // Painter's algorithm: back to front.
  faces.sort((a, b) => a.depth - b.depth);

  ctx.save();
  for (const f of faces) {
    ctx.beginPath();
    ctx.moveTo(f.pts[0][0], f.pts[0][1]);
    for (let k = 1; k < f.pts.length; k++) ctx.lineTo(f.pts[k][0], f.pts[k][1]);
    ctx.closePath();

    const shade = f.front ? 1 : 0.5; // dim the back hemisphere
    if (f.covered) {
      ctx.fillStyle = `rgba(79,209,197,${(f.front ? 0.85 : 0.4).toFixed(2)})`;
    } else if (f.coverable) {
      ctx.fillStyle = `rgba(120,140,155,${(0.5 * shade).toFixed(2)})`;
    } else {
      ctx.fillStyle = `rgba(70,80,90,${(0.5 * shade).toFixed(2)})`;
    }
    ctx.fill();
    ctx.lineWidth = 0.6;
    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
    ctx.stroke();
  }

  // Front heading marker (where the camera is pointing now).
  const h = project(azimuth, 0); // world lon == azimuth faces the viewer
  if (h.z > 0) {
    ctx.beginPath();
    ctx.arc(cx + h.x * radius, cy - h.y * radius, radius * 0.1, 0, Math.PI * 2);
    ctx.fillStyle = '#ffd166';
    ctx.fill();
  }
  ctx.restore();

  // Caption.
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.font = '600 10px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('coverage', cx, cy + radius + 12);
}
