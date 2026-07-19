import { env } from '@huggingface/transformers';
import { detectGpu } from '../webgpu/detect';
import { getConfig } from '../config';

/**
 * One-time transformers.js environment configuration + device selection.
 * Models are fetched from the Hugging Face hub on first use and cached by the
 * browser (and our service worker) for offline use — no backend, no telemetry.
 */
let configured = false;

export function configureEnv(): void {
  if (configured) return;
  configured = true;
  // We only run remote (hub) models; no local model server.
  env.allowLocalModels = false;
  // Use the browser cache for model weights.
  env.useBrowserCache = true;
}

export type Device = 'webgpu' | 'wasm';

/** Decide which backend the depth/matte models should use. */
export async function pickDevice(): Promise<Device> {
  const caps = await detectGpu();
  if (caps.hasWebGPU && caps.deviceOk) return 'webgpu';
  const cfg = getConfig();
  if (!cfg.runtime.allowWasmDepthFallback) {
    throw new Error('WebGPU unavailable and WASM fallback disabled in config.');
  }
  return 'wasm';
}
