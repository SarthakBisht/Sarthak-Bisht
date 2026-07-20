import type { CameraPose, Keyframe, Matte, PointCloud, ProgressFn } from '../types';
import { getConfig } from '../config';
import { tick } from '../util/tick';
import { meshFromField, type TriMesh } from '../mesh/poisson';
import { detectGpu } from '../webgpu/detect';

/**
 * Silhouette VISUAL HULL (space carving).
 *
 * Given clean silhouettes (mattes) captured at KNOWN angles (guided capture),
 * carve a voxel volume: a voxel is INSIDE the object if it projects within the
 * silhouette in (nearly) every view. The result is a genuinely 3D, view-
 * consistent shape with NO background — background voxels project outside some
 * silhouette and get carved away, and there are no monocular-depth flyers.
 *
 * Limitation: the hull is the intersection of silhouette cones, so it cannot
 * carve concavities/undercuts hidden from every silhouette, and the surface is
 * smoother than a photometric reconstruction. It is an excellent clean base.
 */
export interface HullResult {
  mesh: TriMesh;
  cloud: PointCloud; // mesh vertices as a coloured point cloud (for .ply)
}

export async function carveVisualHull(
  keyframes: Keyframe[],
  mattes: Matte[],
  poses: CameraPose[],
  onProgress?: ProgressFn,
): Promise<HullResult> {
  const cfg = getConfig();
  const caps = await detectGpu();
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(
    typeof navigator !== 'undefined' ? navigator.userAgent : '',
  );
  const res = !caps.deviceOk || mobile ? cfg.hull.mobileGridRes : cfg.hull.gridRes;

  const N = poses.length;
  const allowedMisses = Math.floor(N * cfg.hull.allowedMissFrac);
  // Auto-fit the carving cube tightly around the object so the grid resolution
  // is spent on the object (not empty space) → much sharper, less blobby.
  const half = cfg.hull.autoFit
    ? estimateHalfExtent(mattes, poses, (cfg.depth.nearMeters + cfg.depth.farMeters) / 2)
    : cfg.hull.halfExtentM;
  const min = -half;
  const cell = (2 * half) / (res - 1);

  // Precompute world-to-camera per view.
  const w2c = poses.map((p) => invRigid(p.matrix));

  const occ = new Float32Array(res * res * res);
  const gIdx = (x: number, y: number, z: number) => (z * res + y) * res + x;

  for (let gz = 0; gz < res; gz++) {
    const wz = min + gz * cell;
    for (let gy = 0; gy < res; gy++) {
      const wy = min + gy * cell;
      for (let gx = 0; gx < res; gx++) {
        const wx = min + gx * cell;
        let valid = 0;
        let inside = 0;
        for (let v = 0; v < N; v++) {
          const m = w2c[v];
          const cxg = m[0] * wx + m[4] * wy + m[8] * wz + m[12];
          const cyg = m[1] * wx + m[5] * wy + m[9] * wz + m[13];
          const czg = m[2] * wx + m[6] * wy + m[10] * wz + m[14];
          if (czg <= 1e-4) continue;
          const pose = poses[v];
          const u = Math.round(pose.cx + (pose.focalPx * cxg) / czg);
          const vv = Math.round(pose.cy + (pose.focalPx * cyg) / czg);
          const mt = mattes[v];
          if (u < 0 || vv < 0 || u >= mt.width || vv >= mt.height) continue;
          valid++;
          if (mt.alpha[vv * mt.width + u] === 255) inside++;
        }
        if (valid >= Math.ceil(N * 0.5) && valid - inside <= allowedMisses) {
          occ[gIdx(gx, gy, gz)] = 1;
        }
      }
    }
    if ((gz & 7) === 0) {
      onProgress?.('carve', gz / res, `carving ${Math.round((gz / res) * 100)}%`);
      await tick();
    }
  }

  // Optional light smoothing to reduce voxel staircasing, then mesh. Fewer
  // iterations keep more surface detail (less "blurry").
  let field: Float32Array = occ;
  for (let s = 0; s < cfg.hull.smoothIters; s++) field = smooth3(field, res);
  onProgress?.('carve', 1, 'meshing');
  await tick();
  const geo = meshFromField(field, res, res, res, min, min, min, cell, 0.5);

  // Colour each vertex by projecting the captured photos onto it — blending all
  // views where the vertex is front-facing AND not occluded (visibility via the
  // occupancy volume), with bilinear sampling. Looks like the real object.
  onProgress?.('carve', 1, 'texturing');
  await tick();
  const colors = colorVertices(geo.positions, geo.normals, keyframes, poses, w2c, occ, res, min, cell);

  const mesh: TriMesh = {
    positions: geo.positions,
    indices: geo.indices,
    normals: geo.normals,
    colors,
    vertexCount: geo.positions.length / 3,
    triangleCount: geo.indices.length / 3,
  };
  const cloud: PointCloud = {
    count: mesh.vertexCount,
    positions: geo.positions.slice(),
    colors: colors.slice(),
    confidence: new Float32Array(mesh.vertexCount).fill(1),
  };
  return { mesh, cloud };
}

function colorVertices(
  positions: Float32Array,
  normals: Float32Array,
  keyframes: Keyframe[],
  poses: CameraPose[],
  w2c: Float32Array[],
  occ: Float32Array,
  res: number,
  min: number,
  cell: number,
): Float32Array {
  const nv = positions.length / 3;
  const colors = new Float32Array(nv * 3);
  const camPos = poses.map((p) => [p.matrix[12], p.matrix[13], p.matrix[14]] as const);
  const N = poses.length;

  for (let i = 0; i < nv; i++) {
    const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
    const nx = normals[i * 3], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];

    let wr = 0, wg = 0, wb = 0, wsum = 0;
    for (let v = 0; v < N; v++) {
      // Front-facing toward this camera?
      let dx = camPos[v][0] - px, dy = camPos[v][1] - py, dz = camPos[v][2] - pz;
      const l = Math.hypot(dx, dy, dz) || 1;
      dx /= l; dy /= l; dz /= l;
      const facing = dx * nx + dy * ny + dz * nz;
      if (facing <= 0.08) continue;

      // Project into the view.
      const m = w2c[v];
      const czg = m[2] * px + m[6] * py + m[10] * pz + m[14];
      if (czg <= 1e-4) continue;
      const cxg = m[0] * px + m[4] * py + m[8] * pz + m[12];
      const cyg = m[1] * px + m[5] * py + m[9] * pz + m[13];
      const pose = poses[v];
      const u = pose.cx + (pose.focalPx * cxg) / czg;
      const vv = pose.cy + (pose.focalPx * cyg) / czg;
      const kf = keyframes[v];
      if (u < 0 || vv < 0 || u >= kf.width - 1 || vv >= kf.height - 1) continue;

      // Occlusion: is anything between this vertex and the camera?
      if (occludedInView(px, py, pz, camPos[v][0], camPos[v][1], camPos[v][2], occ, res, min, cell)) {
        continue;
      }

      const [r, g, b] = sampleBilinear(kf, u, vv);
      const w = facing * facing;
      wr += r * w; wg += g * w; wb += b * w; wsum += w;
    }

    if (wsum > 0) {
      colors[i * 3] = wr / wsum; colors[i * 3 + 1] = wg / wsum; colors[i * 3 + 2] = wb / wsum;
    } else {
      colors[i * 3] = 0.6; colors[i * 3 + 1] = 0.52; colors[i * 3 + 2] = 0.42;
    }
  }
  return colors;
}

/** True if the segment from P toward camera C passes through occupied voxels. */
function occludedInView(
  px: number, py: number, pz: number,
  cxw: number, cyw: number, czw: number,
  occ: Float32Array, res: number, min: number, cell: number,
): boolean {
  let dx = cxw - px, dy = cyw - py, dz = czw - pz;
  const dist = Math.hypot(dx, dy, dz) || 1;
  dx /= dist; dy /= dist; dz /= dist;
  // Start a couple of cells off the surface to avoid self-occlusion.
  for (let t = cell * 2.5; t < dist; t += cell) {
    const gx = Math.round((px + dx * t - min) / cell);
    const gy = Math.round((py + dy * t - min) / cell);
    const gz = Math.round((pz + dz * t - min) / cell);
    if (gx < 0 || gy < 0 || gz < 0 || gx >= res || gy >= res || gz >= res) return false;
    if (occ[(gz * res + gy) * res + gx] > 0.5) return true;
  }
  return false;
}

/** Bilinear RGB sample (0..1) of a keyframe. */
function sampleBilinear(kf: Keyframe, u: number, v: number): [number, number, number] {
  const w = kf.width;
  const x0 = Math.floor(u), y0 = Math.floor(v);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(kf.height - 1, y0 + 1);
  const fx = u - x0, fy = v - y0;
  const d = kf.rgba;
  const at = (x: number, y: number, c: number) => d[(y * w + x) * 4 + c] / 255;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const r = lerp(lerp(at(x0, y0, 0), at(x1, y0, 0), fx), lerp(at(x0, y1, 0), at(x1, y1, 0), fx), fy);
  const g = lerp(lerp(at(x0, y0, 1), at(x1, y0, 1), fx), lerp(at(x0, y1, 1), at(x1, y1, 1), fx), fy);
  const b = lerp(lerp(at(x0, y0, 2), at(x1, y0, 2), fx), lerp(at(x0, y1, 2), at(x1, y1, 2), fx), fy);
  return [r, g, b];
}

/**
 * Estimate the object's world half-extent from the silhouettes so the carving
 * cube fits tightly. For each view, the farthest silhouette pixel from the
 * principal point maps (at camera distance D) to a world half-extent; take the
 * max across views and add a small margin.
 */
function estimateHalfExtent(mattes: Matte[], poses: CameraPose[], D: number): number {
  let maxHalf = 0;
  for (let v = 0; v < mattes.length; v++) {
    const m = mattes[v];
    const pose = poses[v];
    let maxPx = 0;
    for (let y = 0; y < m.height; y++) {
      for (let x = 0; x < m.width; x++) {
        if (m.alpha[y * m.width + x] !== 255) continue;
        const dx = Math.abs(x - pose.cx);
        const dy = Math.abs(y - pose.cy);
        const d = Math.max(dx, dy);
        if (d > maxPx) maxPx = d;
      }
    }
    const worldHalf = (maxPx / pose.focalPx) * D;
    if (worldHalf > maxHalf) maxHalf = worldHalf;
  }
  return maxHalf > 1e-3 ? maxHalf * 1.15 : 0.25;
}

/** 3x3x3 box smoothing over a cubic field. */
function smooth3(src: Float32Array, res: number): Float32Array {
  const out = new Float32Array(src.length);
  const idx = (x: number, y: number, z: number) => (z * res + y) * res + x;
  for (let z = 0; z < res; z++) {
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        let s = 0, n = 0;
        for (let dz = -1; dz <= 1; dz++) {
          const zz = z + dz; if (zz < 0 || zz >= res) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy; if (yy < 0 || yy >= res) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx; if (xx < 0 || xx >= res) continue;
              s += src[idx(xx, yy, zz)]; n++;
            }
          }
        }
        out[idx(x, y, z)] = s / n;
      }
    }
  }
  return out;
}

/** Invert a column-major 4x4 rigid (R|t) transform → world-to-camera. */
function invRigid(m: Float32Array): Float32Array {
  const t = [m[12], m[13], m[14]];
  const rt = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const nt = [
    -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
    -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
    -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
  ];
  const out = new Float32Array(16);
  out[0] = rt[0]; out[1] = rt[3]; out[2] = rt[6];
  out[4] = rt[1]; out[5] = rt[4]; out[6] = rt[7];
  out[8] = rt[2]; out[9] = rt[5]; out[10] = rt[8];
  out[12] = nt[0]; out[13] = nt[1]; out[14] = nt[2]; out[15] = 1;
  return out;
}
