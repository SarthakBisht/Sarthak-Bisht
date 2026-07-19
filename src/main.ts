import './style.css';
import { detectGpu } from './webgpu/detect';
import {
  getConfig,
  getActivePreset,
  setPreset,
  patchConfig,
  PRESETS,
  type MaterialPreset,
} from './config';
import { createRecorder, type RecorderHandle } from './capture/recorder';
import { RingGuide } from './capture/ringGuide';
import { runTier0, type Tier0Result } from './pipeline/pipeline';
import { trainSplatScene } from './splat/trainer';
import { tryPosefree } from './posefree/dust3r';
import { reconstructMesh } from './mesh/poisson';
import { createViewer, type OrbitViewer } from './view/viewer';
import { buildParamPanel } from './ui/panel';
import { downloadBlob } from './ui/download';
import { pointCloudToPly, splatToGaussianPly } from './export/ply';
import { splatToSplatFile } from './export/splat';
import { meshToGlb, pointCloudToGlb } from './export/gltf';
import { calibrateFromTurntable } from './poses/scale';
import type { SplatScene } from './types';

// ---------------------------------------------------------------------------
// Small DOM helper.
// ---------------------------------------------------------------------------
type Child = Node | string;
function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

// ---------------------------------------------------------------------------
// App state.
// ---------------------------------------------------------------------------
interface AppState {
  gpuOk: boolean;
  gpuBackend: string;
  result: Tier0Result | null;
  splat: SplatScene | null;
  viewer: OrbitViewer | null;
  recorder: RecorderHandle | null;
  ring: RingGuide | null;
}
const state: AppState = {
  gpuOk: false,
  gpuBackend: 'unknown',
  result: null,
  splat: null,
  viewer: null,
  recorder: null,
  ring: null,
};

const root = document.getElementById('app')!;

function topbar(): HTMLElement {
  const bar = el('div', { class: 'topbar' });
  bar.append(el('h1', {}, 'Driftwood 3D Scanner'));
  const badge = el('span', { class: 'badge' }, state.gpuOk ? `WebGPU: ${state.gpuBackend}` : 'WebGPU: fallback');
  bar.append(badge);
  return bar;
}

function clearContent() {
  if (state.viewer) {
    state.viewer.dispose();
    state.viewer = null;
  }
  if (state.recorder) {
    state.recorder.dispose();
    state.recorder = null;
  }
  if (state.ring) {
    state.ring.dispose();
    state.ring = null;
  }
  root.innerHTML = '';
}

// ---------------------------------------------------------------------------
// Home / capture-source screen.
// ---------------------------------------------------------------------------
function showHome() {
  clearContent();
  root.append(topbar());
  const screen = el('div', { class: 'screen' });

  if (!state.gpuOk) {
    screen.append(
      el(
        'div',
        { class: 'banner' },
        'WebGPU is unavailable — depth will run on the slower WASM backend and the ' +
          'Gaussian-splat stage (Tier 1) is disabled. Use a recent Chrome/Android or a ' +
          'WebGPU-capable desktop browser for full quality.',
      ),
    );
  }

  // Material preset selection.
  const presetCard = el('div', { class: 'card stack' });
  presetCard.append(el('div', { class: 'param-group-title' }, 'Material preset'));
  const grid = el('div', { class: 'preset-grid' });
  const presets: MaterialPreset[] = ['driftwood-dry', 'driftwood-wet', 'rock-dry', 'rock-wet'];
  for (const p of presets) {
    const chip = el('div', { class: `chip${getActivePreset() === p ? ' active' : ''}` }, prettyPreset(p));
    chip.addEventListener('click', () => {
      setPreset(p);
      grid.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
    });
    grid.append(chip);
  }
  presetCard.append(grid);
  presetCard.append(
    el('div', { class: 'muted' }, 'Wet/glossy presets favour the splat path; dry presets enable clean mesh export.'),
  );
  screen.append(presetCard);

  // Capture source card.
  const srcCard = el('div', { class: 'card stack' });
  srcCard.append(el('div', { class: 'param-group-title' }, 'Capture source'));
  srcCard.append(
    el(
      'div',
      { class: 'muted' },
      'Place the object on a turntable (or circle it with the phone). Diffuse even lighting, ' +
        'matte background, ~30s of smooth 360° coverage. Keep the object centred in the ring.',
    ),
  );

  const row = el('div', { class: 'row' });
  const recordBtn = el('button', { class: 'primary grow' }, '● Record turntable');
  recordBtn.addEventListener('click', () => showCapture());
  const uploadBtn = el('button', { class: 'grow' }, '⤒ Upload video');
  const fileInput = el('input', { type: 'file', accept: 'video/*', class: 'hidden' }) as HTMLInputElement;
  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (f) showProcessing(f);
  });
  row.append(recordBtn, uploadBtn);
  srcCard.append(row, fileInput);

  const sampleBtn = el('button', { class: '' }, '▶ Use bundled sample clip');
  sampleBtn.addEventListener('click', () => showProcessing('./samples/sample-turntable.webm'));
  srcCard.append(sampleBtn);

  screen.append(srcCard);
  root.append(screen);
}

// ---------------------------------------------------------------------------
// Live capture screen with ring guide.
// ---------------------------------------------------------------------------
async function showCapture() {
  clearContent();
  root.append(topbar());
  const screen = el('div', { class: 'screen' });
  const wrap = el('div', { class: 'capture-wrap' });
  screen.append(wrap);

  const controls = el('div', { class: 'row' });
  const backBtn = el('button', {}, '‹ Back');
  backBtn.addEventListener('click', () => showHome());
  const recBtn = el('button', { class: 'primary grow' }, '● Start');
  controls.append(backBtn, recBtn);
  screen.append(controls);
  const hint = el('div', { class: 'muted' }, 'Fill the ring evenly. Recording auto-stops at the max duration.');
  screen.append(hint);
  root.append(screen);

  let recorder: RecorderHandle;
  try {
    recorder = await createRecorder(wrap);
  } catch (err) {
    hint.textContent = `Camera unavailable: ${(err as Error).message}. Try uploading a video instead.`;
    return;
  }
  state.recorder = recorder;
  const ring = new RingGuide(wrap, getConfig().capture.targetSeconds);
  state.ring = ring;
  ring.showIdle();

  let recording = false;
  recBtn.addEventListener('click', async () => {
    if (!recording) {
      recorder.start();
      ring.start();
      recording = true;
      recBtn.textContent = '■ Stop';
      recBtn.classList.add('warn');
      recBtn.classList.remove('primary');
    } else {
      recBtn.disabled = true;
      const blob = await recorder.stop();
      ring.stop();
      showProcessing(blob);
    }
  });
}

// ---------------------------------------------------------------------------
// Processing screen with progress bar.
// ---------------------------------------------------------------------------
const STAGE_WEIGHTS: Record<string, [number, number]> = {
  keyframes: [0.0, 0.12],
  matte: [0.12, 0.4],
  depth: [0.4, 0.72],
  poses: [0.72, 0.8],
  fuse: [0.8, 0.97],
  done: [0.97, 1.0],
};

async function showProcessing(source: Blob | string) {
  clearContent();
  root.append(topbar());
  const screen = el('div', { class: 'screen' });
  const card = el('div', { class: 'card stack' });
  card.append(el('div', { class: 'param-group-title' }, 'Reconstructing (Tier 0)'));
  const bar = el('div', { class: 'progress' });
  const fill = el('span', {});
  bar.append(fill);
  const label = el('div', { class: 'progress-label' }, 'Starting…');
  card.append(bar, label);
  card.append(
    el(
      'div',
      { class: 'muted' },
      'First run downloads the depth & matte models (~60 MB) and caches them for offline use. ' +
        'Everything runs on your device.',
    ),
  );
  screen.append(card);
  root.append(screen);

  try {
    const result = await runTier0(source, (stage, frac, detail) => {
      const span = STAGE_WEIGHTS[stage] ?? [0, 1];
      const overall = span[0] + (span[1] - span[0]) * frac;
      fill.style.width = `${Math.round(overall * 100)}%`;
      label.textContent = `${prettyStage(stage)} — ${detail ?? ''}`;
    });
    state.result = result;
    state.splat = null;
    showResult();
  } catch (err) {
    label.textContent = '';
    card.append(el('div', { class: 'banner' }, `Reconstruction failed: ${(err as Error).message}`));
    const back = el('button', {}, '‹ Back');
    back.addEventListener('click', () => showHome());
    card.append(back);
  }
}

// ---------------------------------------------------------------------------
// Result screen: viewer + exports + tuning.
// ---------------------------------------------------------------------------
async function showResult() {
  clearContent();
  root.append(topbar());
  const result = state.result!;
  const screen = el('div', { class: 'screen' });

  const viewerEl = el('div', { class: 'viewer' });
  screen.append(viewerEl);

  const stats = el('div', { class: 'stat-row' });
  const updateStats = () => {
    stats.innerHTML = '';
    stats.append(
      el('span', {}, ...frag(`Points: `, b(result.cloud.count.toLocaleString()))),
      el('span', {}, ...frag(`Frames: `, b(String(result.keyframes.length)))),
      el('span', {}, ...frag(`Width: `, b(`${(metricWidth(result) * 100).toFixed(1)} cm`))),
      el('span', {}, ...frag(`Backend: `, b(state.viewer?.backendName() ?? state.gpuBackend))),
    );
  };

  screen.append(stats);

  // Action rows.
  const actions = el('div', { class: 'row' });
  const backBtn = el('button', {}, '‹ New scan');
  backBtn.addEventListener('click', () => showHome());

  const plyBtn = el('button', { class: 'primary' }, 'Export .ply');
  plyBtn.addEventListener('click', () => downloadBlob(pointCloudToPly(result.cloud), 'driftwood.ply'));

  const glbBtn = el('button', {}, 'Export points .glb');
  glbBtn.addEventListener('click', async () => {
    glbBtn.disabled = true;
    downloadBlob(await pointCloudToGlb(result.cloud), 'driftwood-points.glb');
    glbBtn.disabled = false;
  });

  actions.append(backBtn, plyBtn, glbBtn);

  // Mesh export (dry presets only).
  if (getConfig().mesh.enabled) {
    const meshBtn = el('button', {}, 'Export mesh .glb');
    meshBtn.addEventListener('click', async () => {
      meshBtn.disabled = true;
      meshBtn.textContent = 'Meshing…';
      try {
        const mesh = reconstructMesh(result.cloud);
        state.viewer?.showMesh(mesh);
        downloadBlob(await meshToGlb(mesh), 'driftwood-mesh.glb');
      } finally {
        meshBtn.disabled = false;
        meshBtn.textContent = 'Export mesh .glb';
      }
    });
    actions.append(meshBtn);
  }
  screen.append(actions);

  // Tier 1 splat row.
  const splatRow = el('div', { class: 'row' });
  const splatBtn = el('button', { class: 'primary grow' }, '✦ Build Gaussian splat (Tier 1)');
  const splatMsg = el('div', { class: 'progress-label' }, '');
  if (!state.gpuOk) {
    splatBtn.disabled = true;
    splatMsg.textContent = 'Tier 1 needs WebGPU — unavailable on this device.';
  }
  splatBtn.addEventListener('click', async () => {
    splatBtn.disabled = true;
    try {
      const scene = await trainSplatScene(result.cloud, (_s, frac, detail) => {
        splatMsg.textContent = `Splatting ${Math.round(frac * 100)}% — ${detail ?? ''}`;
      });
      state.splat = scene;
      state.viewer?.showSplatScene(scene);
      splatMsg.textContent = `Splat ready: ${scene.count.toLocaleString()} gaussians (preview as points).`;
      addSplatExports(actions, scene);
    } catch (err) {
      splatMsg.textContent = `Tier 1 skipped: ${(err as Error).message}`;
    } finally {
      splatBtn.disabled = false;
    }
  });
  splatRow.append(splatBtn);
  screen.append(splatRow, splatMsg);

  // Tier 2 pose-free (experimental).
  const pfRow = el('div', { class: 'row' });
  const pfBtn = el('button', {}, 'Try pose-free (Tier 2, experimental)');
  const pfMsg = el('div', { class: 'progress-label' }, '');
  pfBtn.addEventListener('click', async () => {
    pfBtn.disabled = true;
    const res = await tryPosefree(result.keyframes, (_s, frac, detail) => {
      pfMsg.textContent = `${Math.round(frac * 100)}% — ${detail ?? ''}`;
    });
    pfMsg.textContent = res.reason ?? (res.supported ? 'Pose-free reconstruction complete.' : 'Skipped.');
    pfBtn.disabled = false;
  });
  pfRow.append(pfBtn);
  screen.append(pfRow, pfMsg);

  // Scale calibration (turntable diameter).
  const scaleCard = el('div', { class: 'card stack' });
  scaleCard.append(el('div', { class: 'param-group-title' }, 'Real-world scale'));
  const scaleRow = el('div', { class: 'row' });
  scaleRow.append(el('span', { class: 'muted' }, 'Turntable diameter (cm):'));
  const diaInput = el('input', {
    type: 'number',
    min: '1',
    max: '200',
    step: '0.5',
    value: String(getConfig().scale.turntableDiameterM * 100),
  }) as HTMLInputElement;
  const applyScale = el('button', {}, 'Apply');
  applyScale.addEventListener('click', () => {
    const cm = Number(diaInput.value);
    if (cm > 0) {
      patchConfig({ scale: { turntableDiameterM: cm / 100 } });
      // Reset then recalibrate from current extent.
      const res = calibrateFromTurntable(result.cloud);
      result.metersPerUnit = res.metersPerUnit;
      state.viewer?.showPointCloud(result.cloud);
      updateStats();
    }
  });
  scaleRow.append(diaInput, applyScale);
  scaleCard.append(scaleRow);
  screen.append(scaleCard);

  // Tuning panel (collapsible).
  const tuneCard = el('div', { class: 'card stack' });
  const tuneHead = el('div', { class: 'row' });
  tuneHead.append(el('div', { class: 'param-group-title grow' }, 'Tuning parameters (config.ts)'));
  const toggle = el('button', { class: 'panel-toggle' }, 'Show');
  tuneHead.append(toggle);
  tuneCard.append(tuneHead);
  const panelHost = el('div', { class: 'hidden' });
  buildParamPanel(panelHost);
  toggle.addEventListener('click', () => {
    const hidden = panelHost.classList.toggle('hidden');
    toggle.textContent = hidden ? 'Show' : 'Hide';
  });
  tuneCard.append(panelHost);
  tuneCard.append(
    el('div', { class: 'muted' }, 'Changes apply to the next scan. These map 1:1 to the fields in src/config.ts.'),
  );
  screen.append(tuneCard);

  root.append(screen);

  // Build the viewer and show the cloud.
  state.viewer = await createViewer(viewerEl);
  state.viewer.showPointCloud(result.cloud);
  updateStats();
}

function addSplatExports(actions: HTMLElement, scene: SplatScene) {
  if (actions.querySelector('[data-splat-export]')) return;
  const splatBtn = el('button', { 'data-splat-export': '1' }, 'Export .splat');
  splatBtn.addEventListener('click', () => downloadBlob(splatToSplatFile(scene), 'driftwood.splat'));
  const gplyBtn = el('button', { 'data-splat-export': '1' }, 'Export gaussian .ply');
  gplyBtn.addEventListener('click', () => downloadBlob(splatToGaussianPly(scene), 'driftwood-gaussian.ply'));
  actions.append(splatBtn, gplyBtn);
}

// ---------------------------------------------------------------------------
// Utilities.
// ---------------------------------------------------------------------------
function prettyPreset(p: MaterialPreset): string {
  return { 'driftwood-dry': 'Driftwood (dry)', 'driftwood-wet': 'Driftwood (wet)', 'rock-dry': 'Rock (dry)', 'rock-wet': 'Rock (wet)' }[p];
}
function prettyStage(s: string): string {
  return (
    {
      keyframes: 'Extracting keyframes',
      matte: 'Matting frames',
      depth: 'Estimating depth',
      poses: 'Refining poses',
      fuse: 'Fusing point cloud',
      done: 'Finalizing',
    } as Record<string, string>
  )[s] ?? s;
}
function b(text: string): HTMLElement {
  return el('b', {}, text);
}
function frag(...items: Child[]): Child[] {
  return items;
}
function metricWidth(r: Tier0Result): number {
  // Positions are already in metres after scale calibration.
  const p = r.cloud.positions;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    minX = Math.min(minX, p[i]); maxX = Math.max(maxX, p[i]);
    minZ = Math.min(minZ, p[i + 2]); maxZ = Math.max(maxZ, p[i + 2]);
  }
  return Math.max(maxX - minX, maxZ - minZ);
}

// ---------------------------------------------------------------------------
// Boot.
// ---------------------------------------------------------------------------
async function boot() {
  // Ensure presets object is referenced (tree-shake guard) and config ready.
  void PRESETS;
  const caps = await detectGpu();
  state.gpuOk = caps.hasWebGPU && caps.deviceOk;
  state.gpuBackend = caps.adapterName ?? (state.gpuOk ? 'ready' : 'WebGL2/WASM');
  if (!state.gpuOk && caps.reason) console.info('[webgpu]', caps.reason);
  showHome();
}

boot();
