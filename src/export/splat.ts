import type { SplatScene } from '../types';

/**
 * `.splat` exporter (the antimatter15 / gsplat 32-bytes-per-splat layout used by
 * most web splat viewers):
 *   position : 3 x float32   (12 bytes)
 *   scale    : 3 x float32   (12 bytes)
 *   color    : 4 x uint8 rgba ( 4 bytes)
 *   rotation : 4 x uint8 quat ( 4 bytes, each = q*128+128)
 */
export function splatToSplatFile(scene: SplatScene): Blob {
  const n = scene.count;
  const stride = 32;
  const buf = new ArrayBuffer(n * stride);
  const dv = new DataView(buf);
  for (let i = 0; i < n; i++) {
    const o = i * stride;
    const p = i * 3;
    const q = i * 4;
    dv.setFloat32(o, scene.positions[p], true);
    dv.setFloat32(o + 4, scene.positions[p + 1], true);
    dv.setFloat32(o + 8, scene.positions[p + 2], true);
    dv.setFloat32(o + 12, scene.scales[p], true);
    dv.setFloat32(o + 16, scene.scales[p + 1], true);
    dv.setFloat32(o + 20, scene.scales[p + 2], true);
    dv.setUint8(o + 24, clamp8(scene.colors[p] * 255));
    dv.setUint8(o + 25, clamp8(scene.colors[p + 1] * 255));
    dv.setUint8(o + 26, clamp8(scene.colors[p + 2] * 255));
    dv.setUint8(o + 27, clamp8(scene.opacities[i] * 255));
    // Normalize quaternion, pack to uint8.
    const qw = scene.rotations[q + 3];
    const qx = scene.rotations[q];
    const qy = scene.rotations[q + 1];
    const qz = scene.rotations[q + 2];
    const len = Math.hypot(qx, qy, qz, qw) || 1;
    dv.setUint8(o + 28, clamp8((qw / len) * 128 + 128));
    dv.setUint8(o + 29, clamp8((qx / len) * 128 + 128));
    dv.setUint8(o + 30, clamp8((qy / len) * 128 + 128));
    dv.setUint8(o + 31, clamp8((qz / len) * 128 + 128));
  }
  return new Blob([buf], { type: 'application/octet-stream' });
}

function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}
