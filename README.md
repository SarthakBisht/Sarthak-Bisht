# Driftwood 3D Scanner

Turn a **turntable video** of a driftwood piece or aquascaping rock into an
**orbitable 3D asset** (point cloud → Gaussian splat) that you can export and
drop into a three.js aquascaping scene.

Everything runs **fully in your browser** — **no backend, no paid APIs, no
telemetry**. It's a PWA: after the first load the ML models are cached for
**offline** use. WebGPU-first, with a WASM fallback for depth.

> **Target device:** recent **Chrome on Android** (or any WebGPU-capable
> desktop browser). WebGPU is required for the Gaussian-splat stage; depth
> estimation falls back to WASM when WebGPU is unavailable.

---

## The capture model (why it's fast)

Full Structure-from-Motion is the browser bottleneck, so this app **doesn't do
it**. Instead it assumes a **turntable capture**: the object sits on a rotating
surface (or you circle it with the phone) at a roughly constant angular step.
Camera poses are **derived** from that constant step and then **refined** with
feature tracking on the object's own high-frequency texture (wood grain,
mineral speckle). An on-screen **ring guide** during capture enforces even 360°
coverage.

## How to capture (read this — it determines quality)

- **Turntable:** put the object on a lazy-Susan / plate you can spin slowly and
  evenly. Or walk the phone around it in a smooth circle.
- **Even coverage:** ~**30 seconds**, one smooth **360°** pass. Fill the ring
  guide evenly — don't linger or rush.
- **Lighting:** diffuse, even light. Avoid hard shadows and hotspots.
- **Background:** matte, solid, contrasting (a cloth or paper backdrop). This
  makes the silhouette matte clean.
- **Dry if possible:** dry surfaces mesh cleanly. Wet/glossy surfaces should use
  the **splat** path (it handles view-dependent specular highlights).
- **Avoid motion blur:** move slowly and keep the object centred and in focus.
- **Keep the object centred** in the ring the whole time.

## Scale

The export is given real-world scale so it places correctly in an aquascape:
- **Turntable diameter** (default): type the disc diameter and the app scales
  the asset's footprint to match.
- **Reference length** (config `scale.mode: 'reference'`): tap a known length on
  a frame.

---

## Pipeline (built in tiers)

**Tier 0 — guaranteed MVP** (always available):
extract keyframes → background-matte each frame (RMBG, eroded to drop halos) →
**Depth Anything V2** per keyframe (WebGPU, fp16; WASM fallback) → temporal +
bilateral depth smoothing → **turntable poses** (derived + feature-refined) →
back-project & fuse into a coloured point cloud → orbit viewer → **export
`.ply`**.

**Tier 1 — Gaussian Splatting** (requires WebGPU):
initialize Gaussians from the fused multi-view cloud (scale from local point
spacing, opacity from per-point confidence), then run the config-driven
**densify/prune** schedule — densifying more in low-confidence regions
(concavities, crevices, arch undersides) and pruning gently so thin protrusions
survive. A WebGPU compute pass refines per-Gaussian scale. Export **`.splat`**
and **gaussian `.ply`**. Better than a mesh for wet/glossy surfaces.

> **Honest scope note:** a full *photometric* 3DGS optimizer that
> back-propagates through a differentiable rasterizer against every keyframe is
> still at the edge of what browsers can do. Tier 1 here ships the working
> **initialize + geometric densify/prune** stage (the intended attach point for
> the photometric loop lives in `src/splat/schedule.ts` and
> `src/splat/trainer.ts`). It produces valid, orbitable, exportable splats and
> degrades gracefully to Tier 0 if the compute budget is exceeded.

**Tier 2 — pose-free** (experimental, graceful skip):
attempts a DUSt3R/MASt3R/InstantSplat-style model via `onnxruntime-web`. There is
no lightweight browser-exportable build of these yet, so it **skips cleanly** and
falls back to the turntable pipeline (`src/posefree/dust3r.ts`).

## Exports

| Format | Path | Use |
| --- | --- | --- |
| `.ply` (points) | Tier 0 | universal point cloud |
| points `.glb` | Tier 0 | glTF POINTS for quick three.js import |
| mesh `.glb` | dry presets | **recommended** for aquascaping import |
| `.splat` | Tier 1 | web splat viewers |
| gaussian `.ply` | Tier 1 | 3DGS tools |

## Material tuning — one config file

Every quality knob lives in **`src/config.ts`** — a single editable object with
per-material presets (`driftwood-dry`, `driftwood-wet`, `rock-dry`, `rock-wet`).
The in-app **Tuning** panel is bound 1:1 to these fields. Key knobs:

- `matte.erosionPx` — erode organic silhouettes to drop halo pixels.
- `depth.bilateral` / `depth.temporalWeight` — kill per-frame depth noise from
  wood grain / mineral speckle.
- `splat.densifyThreshold` / `splat.pruneThreshold` /
  `splat.lowConfidenceDensifyBoost` — recover concavities & crevices.
- `fusion.pointDensity` / `mesh.keepTriangleRatio` — keep thin driftwood
  branches from snapping off (keep high).
- `scale.mode` / `scale.turntableDiameterM` / `scale.referenceLengthM`.

Retune per material by editing the preset in `config.ts`, or live via the panel.

## Engineering notes

- **WebGPU detection** with a visible fallback banner; depth-only WASM fallback.
- **Bounded memory:** frames processed in batches, resolution capped
  (`keyframes.maxEdgePx`), tensors/GPU buffers disposed between stages, models
  freed after use.
- **Offline PWA:** the app shell is precached; the multi-MB ONNX wasm and the HF
  model shards are runtime-cached (CacheFirst) on first use — a second visit
  works offline.
- **Verified API versions** (checked at build time): `@huggingface/transformers`
  v4, `three` r185 `WebGPURenderer` (`three/webgpu`), `vite-plugin-pwa` v1.

## Sample clip

`public/samples/sample-turntable.webm` is a **synthetic** rotating textured blob
(rendered procedurally) so you can exercise the full pipeline offline. It is not
real driftwood — record your own clip for real results.

## Develop

```bash
npm install
npm run dev        # local dev server
npm run build      # production PWA build -> dist/
npm run preview    # serve the built PWA
npm run typecheck  # tsc --noEmit
```

First real run downloads the depth (~50 MB) and matte models from the Hugging
Face CDN and caches them; everything after runs on-device and offline.

## Project layout

```
src/
  config.ts          ★ single tuning config + material presets
  webgpu/detect.ts   WebGPU capability probe
  capture/           recorder, ring guide, keyframe extraction (blur reject)
  poses/             turntable poses, feature-tracking refine, scale
  matte/segment.ts   RMBG matte + mask erosion
  depth/             Depth Anything V2 + bilateral/temporal smoothing
  fuse/              back-projection + multi-view fusion (voxel/outlier/cap)
  splat/             Tier 1 gaussian splat build + densify/prune schedule
  posefree/          Tier 2 pose-free stub (graceful skip)
  mesh/poisson.ts    dry-path surface mesh (Surface Nets)
  view/viewer.ts     three/webgpu orbit viewer
  export/            .ply, .splat, glTF/GLB writers
  ui/                param panel, download helper
  pipeline/          env setup + Tier 0 orchestrator
```

## License

MIT
