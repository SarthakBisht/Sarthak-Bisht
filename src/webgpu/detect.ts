/**
 * WebGPU capability detection.
 *
 * The app is WebGPU-first. When WebGPU is missing we still allow the depth
 * model to run on the WASM backend (Tier 0 works, just slower), but Tier 1
 * (Gaussian-splat training) and the WebGPU viewer require a real device.
 */

export interface GpuCaps {
  hasWebGPU: boolean;
  adapterName: string | null;
  /** Whether we could actually request a device (some browsers expose the
   *  API object but fail to hand out a device). */
  deviceOk: boolean;
  maxBufferSizeMB: number | null;
  maxStorageBufferMB: number | null;
  reason?: string;
}

let cached: GpuCaps | null = null;

export async function detectGpu(force = false): Promise<GpuCaps> {
  if (cached && !force) return cached;

  const nav = navigator as Navigator & { gpu?: GPU };
  if (!nav.gpu) {
    cached = {
      hasWebGPU: false,
      adapterName: null,
      deviceOk: false,
      maxBufferSizeMB: null,
      maxStorageBufferMB: null,
      reason: 'navigator.gpu is undefined (WebGPU unsupported or disabled).',
    };
    return cached;
  }

  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      cached = {
        hasWebGPU: true,
        adapterName: null,
        deviceOk: false,
        maxBufferSizeMB: null,
        maxStorageBufferMB: null,
        reason: 'No GPU adapter available.',
      };
      return cached;
    }

    const device = await adapter.requestDevice();
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const limits = device.limits;
    cached = {
      hasWebGPU: true,
      adapterName: info?.description || info?.vendor || 'unknown',
      deviceOk: true,
      maxBufferSizeMB: Math.round(Number(limits.maxBufferSize) / (1024 * 1024)),
      maxStorageBufferMB: Math.round(
        Number(limits.maxStorageBufferBindingSize) / (1024 * 1024),
      ),
    };
    // We don't hold the device here; consumers (viewer, splat trainer) request
    // their own. Release this probe device.
    device.destroy();
    return cached;
  } catch (err) {
    cached = {
      hasWebGPU: true,
      adapterName: null,
      deviceOk: false,
      maxBufferSizeMB: null,
      maxStorageBufferMB: null,
      reason: `WebGPU device request failed: ${(err as Error).message}`,
    };
    return cached;
  }
}

/** Request a fresh device for a consumer that needs one (viewer/trainer). */
export async function requestDevice(): Promise<GPUDevice | null> {
  const nav = navigator as Navigator & { gpu?: GPU };
  if (!nav.gpu) return null;
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    return await adapter.requestDevice();
  } catch {
    return null;
  }
}
