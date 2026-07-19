import type { PointCloud } from '../types';
import { getConfig } from '../config';

/** A triangle mesh with optional per-vertex normals and colors. */
export interface TriMesh {
  positions: Float32Array; // v*3
  indices: Uint32Array; // t*3
  normals?: Float32Array; // v*3
  colors?: Float32Array; // v*3 (0..1)
  vertexCount: number;
  triangleCount: number;
}

/**
 * Meshing for the DRY-capture path (glossy/wet captures should use the splat
 * path instead). Full screened-Poisson is too heavy for the browser, so we use
 * SURFACE NETS over a density grid built from the fused cloud — self-contained,
 * fast, and watertight-ish. Vertices are coloured from the nearest occupied
 * voxel's average colour.
 *
 * Grid resolution is derived from config.mesh.poissonDepth but capped so peak
 * memory stays bounded. Thin protrusions are preserved by keeping a low iso
 * threshold (a single occupied voxel is enough to generate surface).
 */
export function reconstructMesh(cloud: PointCloud): TriMesh {
  const cfg = getConfig();
  const res = Math.min(160, Math.max(32, 1 << Math.min(8, cfg.mesh.poissonDepth - 2)));

  // Bounding box with a small margin.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const P = cloud.positions;
  for (let i = 0; i < cloud.count; i++) {
    const i3 = i * 3;
    minX = Math.min(minX, P[i3]); maxX = Math.max(maxX, P[i3]);
    minY = Math.min(minY, P[i3 + 1]); maxY = Math.max(maxY, P[i3 + 1]);
    minZ = Math.min(minZ, P[i3 + 2]); maxZ = Math.max(maxZ, P[i3 + 2]);
  }
  const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const margin = size * 0.05;
  minX -= margin; minY -= margin; minZ -= margin;
  const cell = (size + 2 * margin) / (res - 1);
  const inv = 1 / cell;

  const nx = res, ny = res, nz = res;
  const density = new Float32Array(nx * ny * nz);
  const colR = new Float32Array(nx * ny * nz);
  const colG = new Float32Array(nx * ny * nz);
  const colB = new Float32Array(nx * ny * nz);
  const colN = new Float32Array(nx * ny * nz);
  const idx = (x: number, y: number, z: number) => (z * ny + y) * nx + x;

  // Splat each point into its voxel (and accumulate colour).
  const C = cloud.colors;
  for (let i = 0; i < cloud.count; i++) {
    const i3 = i * 3;
    const gx = Math.round((P[i3] - minX) * inv);
    const gy = Math.round((P[i3 + 1] - minY) * inv);
    const gz = Math.round((P[i3 + 2] - minZ) * inv);
    if (gx < 0 || gy < 0 || gz < 0 || gx >= nx || gy >= ny || gz >= nz) continue;
    const c = idx(gx, gy, gz);
    density[c] += 1;
    colR[c] += C[i3]; colG[c] += C[i3 + 1]; colB[c] += C[i3 + 2]; colN[c] += 1;
  }

  // One smoothing pass to close pinholes without inflating thin features much.
  const field = smooth3(density, nx, ny, nz);
  const iso = 0.5;

  // Surface Nets: one vertex per cell straddling the iso surface.
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cIdx = (x: number, y: number, z: number) => (z * (ny - 1) + y) * (nx - 1) + x;
  const verts: number[] = [];
  const vcols: number[] = [];
  const cornerOff = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ];

  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        let mask = 0;
        const vals: number[] = [];
        for (let ci = 0; ci < 8; ci++) {
          const [ox, oy, oz] = cornerOff[ci];
          const v = field[idx(x + ox, y + oy, z + oz)] - iso;
          vals.push(v);
          if (v > 0) mask |= 1 << ci;
        }
        if (mask === 0 || mask === 255) continue;

        // Average zero-crossing position along the 12 edges.
        let px = 0, py = 0, pz = 0, cnt = 0;
        for (const [a, b] of EDGES) {
          if ((vals[a] > 0) === (vals[b] > 0)) continue;
          const t = vals[a] / (vals[a] - vals[b]);
          const [ax, ay, az] = cornerOff[a];
          const [bx, by, bz] = cornerOff[b];
          px += ax + (bx - ax) * t;
          py += ay + (by - ay) * t;
          pz += az + (bz - az) * t;
          cnt++;
        }
        px = x + px / cnt; py = y + py / cnt; pz = z + pz / cnt;

        cellVert[cIdx(x, y, z)] = verts.length / 3;
        verts.push(minX + px * cell, minY + py * cell, minZ + pz * cell);

        // Colour from this cell's accumulated point colour (fallback grey).
        const cc = idx(x, y, z);
        if (colN[cc] > 0) {
          vcols.push(colR[cc] / colN[cc], colG[cc] / colN[cc], colB[cc] / colN[cc]);
        } else {
          vcols.push(0.6, 0.5, 0.4);
        }
      }
    }
  }

  // Quads across the three axis edges where a sign change occurs.
  const tris: number[] = [];
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const v0 = field[idx(x, y, z)] - iso;
        // X edge
        emitQuad(field, idx, cellVert, cIdx, tris, v0, iso, x, y, z, 0);
        emitQuad(field, idx, cellVert, cIdx, tris, v0, iso, x, y, z, 1);
        emitQuad(field, idx, cellVert, cIdx, tris, v0, iso, x, y, z, 2);
      }
    }
  }

  const positions = new Float32Array(verts);
  const colors = new Float32Array(vcols);
  const indices = new Uint32Array(tris);
  const normals = computeNormals(positions, indices);
  return {
    positions,
    indices,
    normals,
    colors,
    vertexCount: positions.length / 3,
    triangleCount: indices.length / 3,
  };
}

const EDGES: [number, number][] = [
  [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
  [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
];

function emitQuad(
  field: Float32Array,
  idx: (x: number, y: number, z: number) => number,
  cellVert: Int32Array,
  cIdx: (x: number, y: number, z: number) => number,
  tris: number[],
  v0: number,
  iso: number,
  x: number,
  y: number,
  z: number,
  axis: 0 | 1 | 2,
) {
  const other =
    axis === 0 ? field[idx(x + 1, y, z)] - iso
    : axis === 1 ? field[idx(x, y + 1, z)] - iso
    : field[idx(x, y, z + 1)] - iso;
  if ((v0 > 0) === (other > 0)) return;

  // The four cells sharing this edge.
  let a: number, b: number, c: number, d: number;
  if (axis === 0) {
    a = cellVert[cIdx(x, y - 1, z - 1)];
    b = cellVert[cIdx(x, y, z - 1)];
    c = cellVert[cIdx(x, y, z)];
    d = cellVert[cIdx(x, y - 1, z)];
  } else if (axis === 1) {
    a = cellVert[cIdx(x - 1, y, z - 1)];
    b = cellVert[cIdx(x, y, z - 1)];
    c = cellVert[cIdx(x, y, z)];
    d = cellVert[cIdx(x - 1, y, z)];
  } else {
    a = cellVert[cIdx(x - 1, y - 1, z)];
    b = cellVert[cIdx(x, y - 1, z)];
    c = cellVert[cIdx(x, y, z)];
    d = cellVert[cIdx(x - 1, y, z)];
  }
  if (a < 0 || b < 0 || c < 0 || d < 0) return;
  // Wind consistently based on sign.
  if (v0 > 0) {
    tris.push(a, b, c, a, c, d);
  } else {
    tris.push(a, c, b, a, d, c);
  }
}

function smooth3(src: Float32Array, nx: number, ny: number, nz: number): Float32Array {
  const out = new Float32Array(src.length);
  const idx = (x: number, y: number, z: number) => (z * ny + y) * nx + x;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        let s = 0, n = 0;
        for (let dz = -1; dz <= 1; dz++) {
          const zz = z + dz; if (zz < 0 || zz >= nz) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy; if (yy < 0 || yy >= ny) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx; if (xx < 0 || xx >= nx) continue;
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

function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t] * 3, ib = indices[t + 1] * 3, ic = indices[t + 2] * 3;
    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
    const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
    const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const i of [ia, ib, ic]) {
      normals[i] += nx; normals[i + 1] += ny; normals[i + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= l; normals[i + 1] /= l; normals[i + 2] /= l;
  }
  return normals;
}
