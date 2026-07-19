import type { PointCloud, SplatScene } from '../types';

/**
 * PLY exporters.
 *  - `pointCloudToPly`  : binary little-endian point cloud (xyz + rgb) — the
 *    Tier-0 deliverable.
 *  - `splatToGaussianPly`: the 3DGS "gaussian ply" layout (x,y,z, nx,ny,nz,
 *    f_dc_0..2, opacity, scale_0..2, rot_0..3) understood by splat viewers.
 */

export function pointCloudToPly(cloud: PointCloud): Blob {
  const n = cloud.count;
  const header =
    `ply\n` +
    `format binary_little_endian 1.0\n` +
    `comment Driftwood 3D Scanner point cloud\n` +
    `element vertex ${n}\n` +
    `property float x\nproperty float y\nproperty float z\n` +
    `property uchar red\nproperty uchar green\nproperty uchar blue\n` +
    `end_header\n`;
  const headerBytes = new TextEncoder().encode(header);
  const stride = 12 + 3; // 3 floats + 3 uchar
  const body = new ArrayBuffer(n * stride);
  const dv = new DataView(body);
  for (let i = 0; i < n; i++) {
    const o = i * stride;
    const p = i * 3;
    dv.setFloat32(o, cloud.positions[p], true);
    dv.setFloat32(o + 4, cloud.positions[p + 1], true);
    dv.setFloat32(o + 8, cloud.positions[p + 2], true);
    dv.setUint8(o + 12, clamp8(cloud.colors[p] * 255));
    dv.setUint8(o + 13, clamp8(cloud.colors[p + 1] * 255));
    dv.setUint8(o + 14, clamp8(cloud.colors[p + 2] * 255));
  }
  return new Blob([headerBytes, body], { type: 'application/octet-stream' });
}

const SH_C0 = 0.28209479177387814; // Y_0^0

export function splatToGaussianPly(scene: SplatScene): Blob {
  const n = scene.count;
  const props = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  const header =
    `ply\n` +
    `format binary_little_endian 1.0\n` +
    `comment Driftwood 3D Scanner gaussian splat\n` +
    `element vertex ${n}\n` +
    props.map((p) => `property float ${p}`).join('\n') +
    `\nend_header\n`;
  const headerBytes = new TextEncoder().encode(header);
  const stride = props.length * 4;
  const body = new ArrayBuffer(n * stride);
  const dv = new DataView(body);
  for (let i = 0; i < n; i++) {
    const o = i * stride;
    const p = i * 3;
    const q = i * 4;
    // position
    dv.setFloat32(o, scene.positions[p], true);
    dv.setFloat32(o + 4, scene.positions[p + 1], true);
    dv.setFloat32(o + 8, scene.positions[p + 2], true);
    // normals (unused, 0)
    dv.setFloat32(o + 12, 0, true);
    dv.setFloat32(o + 16, 0, true);
    dv.setFloat32(o + 20, 0, true);
    // SH DC term from linear colour: f_dc = (c - 0.5) / SH_C0
    dv.setFloat32(o + 24, (scene.colors[p] - 0.5) / SH_C0, true);
    dv.setFloat32(o + 28, (scene.colors[p + 1] - 0.5) / SH_C0, true);
    dv.setFloat32(o + 32, (scene.colors[p + 2] - 0.5) / SH_C0, true);
    // opacity stored as inverse-sigmoid (logit)
    dv.setFloat32(o + 36, logit(scene.opacities[i]), true);
    // scale stored as log
    dv.setFloat32(o + 40, Math.log(Math.max(1e-8, scene.scales[p])), true);
    dv.setFloat32(o + 44, Math.log(Math.max(1e-8, scene.scales[p + 1])), true);
    dv.setFloat32(o + 48, Math.log(Math.max(1e-8, scene.scales[p + 2])), true);
    // rotation quaternion (w,x,y,z order for gaussian ply)
    dv.setFloat32(o + 52, scene.rotations[q + 3], true);
    dv.setFloat32(o + 56, scene.rotations[q], true);
    dv.setFloat32(o + 60, scene.rotations[q + 1], true);
    dv.setFloat32(o + 64, scene.rotations[q + 2], true);
  }
  return new Blob([headerBytes, body], { type: 'application/octet-stream' });
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}
function logit(x: number): number {
  const c = Math.min(1 - 1e-6, Math.max(1e-6, x));
  return Math.log(c / (1 - c));
}
