import { getConfig, patchConfig, type DeepPartial, type ScannerConfig } from '../config';

/**
 * Live tuning panel. Every control maps to a field in the single config object
 * (config.ts). Editing here calls patchConfig so a re-run uses the new values —
 * the same numbers you'd otherwise edit in the config file, surfaced for
 * per-material retuning.
 */

interface SliderSpec {
  group: string;
  label: string;
  min: number;
  max: number;
  step: number;
  get: (c: ScannerConfig) => number;
  set: (v: number) => DeepPartial<ScannerConfig>;
  fmt?: (v: number) => string;
}

const SLIDERS: SliderSpec[] = [
  // Capture / frames
  { group: 'Frames', label: 'Keyframe count', min: 12, max: 60, step: 1,
    get: (c) => c.keyframes.count, set: (v) => ({ keyframes: { count: v } }) },
  { group: 'Frames', label: 'Max edge (px)', min: 384, max: 1024, step: 32,
    get: (c) => c.keyframes.maxEdgePx, set: (v) => ({ keyframes: { maxEdgePx: v } }) },
  { group: 'Frames', label: 'Blur reject var', min: 0, max: 200, step: 5,
    get: (c) => c.keyframes.blurRejectVar, set: (v) => ({ keyframes: { blurRejectVar: v } }) },
  // Matte
  { group: 'Matte', label: 'Mask erosion (px)', min: 0, max: 10, step: 1,
    get: (c) => c.matte.erosionPx, set: (v) => ({ matte: { erosionPx: v } }) },
  { group: 'Matte', label: 'Alpha threshold', min: 0.1, max: 0.9, step: 0.05,
    get: (c) => c.matte.threshold, set: (v) => ({ matte: { threshold: v } }),
    fmt: (v) => v.toFixed(2) },
  // Depth smoothing
  { group: 'Depth', label: 'Bilateral σ-depth', min: 0.01, max: 0.25, step: 0.01,
    get: (c) => c.depth.bilateral.sigmaDepth, set: (v) => ({ depth: { bilateral: { sigmaDepth: v } } }),
    fmt: (v) => v.toFixed(2) },
  { group: 'Depth', label: 'Temporal weight', min: 0, max: 0.8, step: 0.05,
    get: (c) => c.depth.temporalWeight, set: (v) => ({ depth: { temporalWeight: v } }),
    fmt: (v) => v.toFixed(2) },
  { group: 'Depth', label: 'Depth batch size', min: 1, max: 8, step: 1,
    get: (c) => c.depth.batchSize, set: (v) => ({ depth: { batchSize: v } }) },
  // Fusion
  { group: 'Fusion', label: 'Point density', min: 0.1, max: 1, step: 0.05,
    get: (c) => c.fusion.pointDensity, set: (v) => ({ fusion: { pointDensity: v } }),
    fmt: (v) => v.toFixed(2) },
  { group: 'Fusion', label: 'Voxel size (mm)', min: 0, max: 6, step: 0.5,
    get: (c) => c.fusion.voxelSizeM * 1000, set: (v) => ({ fusion: { voxelSizeM: v / 1000 } }),
    fmt: (v) => v.toFixed(1) },
  { group: 'Fusion', label: 'Max points (k)', min: 50, max: 1000, step: 50,
    get: (c) => c.fusion.maxPoints / 1000, set: (v) => ({ fusion: { maxPoints: v * 1000 } }) },
  // Splat (Tier 1)
  { group: 'Splat (Tier 1)', label: 'Densify threshold (×1e-4)', min: 0.5, max: 10, step: 0.5,
    get: (c) => c.splat.densifyThreshold * 1e4, set: (v) => ({ splat: { densifyThreshold: v / 1e4 } }),
    fmt: (v) => v.toFixed(1) },
  { group: 'Splat (Tier 1)', label: 'Prune threshold (×1e-3)', min: 1, max: 20, step: 1,
    get: (c) => c.splat.pruneThreshold * 1e3, set: (v) => ({ splat: { pruneThreshold: v / 1e3 } }),
    fmt: (v) => v.toFixed(0) },
  { group: 'Splat (Tier 1)', label: 'Max gaussians (k)', min: 30, max: 400, step: 10,
    get: (c) => c.splat.maxGaussians / 1000, set: (v) => ({ splat: { maxGaussians: v * 1000 } }) },
  { group: 'Splat (Tier 1)', label: 'Iterations', min: 300, max: 4000, step: 100,
    get: (c) => c.splat.iterations, set: (v) => ({ splat: { iterations: v } }) },
  // Mesh
  { group: 'Mesh (dry)', label: 'Keep triangle ratio', min: 0.3, max: 1, step: 0.05,
    get: (c) => c.mesh.keepTriangleRatio, set: (v) => ({ mesh: { keepTriangleRatio: v } }),
    fmt: (v) => v.toFixed(2) },
];

export function buildParamPanel(container: HTMLElement, onChange?: () => void): void {
  container.innerHTML = '';
  const params = document.createElement('div');
  params.className = 'params';

  let lastGroup = '';
  for (const spec of SLIDERS) {
    if (spec.group !== lastGroup) {
      const title = document.createElement('div');
      title.className = 'param-group-title';
      title.textContent = spec.group;
      params.appendChild(title);
      lastGroup = spec.group;
    }
    const wrap = document.createElement('div');
    wrap.className = 'param';
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = spec.label;
    const val = document.createElement('span');
    val.className = 'val';
    const fmt = spec.fmt ?? ((v: number) => `${v}`);
    const cur = spec.get(getConfig());
    val.textContent = fmt(cur);
    label.append(name, val);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(cur);
    input.addEventListener('input', () => {
      const v = Number(input.value);
      val.textContent = fmt(v);
      patchConfig(spec.set(v));
      onChange?.();
    });

    wrap.append(label, input);
    params.appendChild(wrap);
  }
  container.appendChild(params);
}
