/**
 * ============================================================================
 *  DRIFTWOOD 3D SCANNER — SINGLE SOURCE OF TRUTH FOR ALL TUNING PARAMETERS
 * ============================================================================
 *
 * Every knob that affects reconstruction quality lives here. Retune per
 * material by editing a preset (or the UI param panel, which is bound live to
 * this object). Nothing else in the app hard-codes these numbers.
 *
 * The material presets encode the "baked-in defaults" for driftwood & rock:
 *  - wet/glossy surfaces  -> prefer the splat path (view-dependent specular)
 *  - dry surfaces         -> allow the clean mesh (glTF) path
 *  - thin protrusions     -> keep point density high, never decimate hard
 *  - concavities/crevices -> densify more / prune less in low-confidence areas
 */

export type MaterialPreset =
  | 'driftwood-dry'
  | 'driftwood-wet'
  | 'rock-dry'
  | 'rock-wet';

export interface ScannerConfig {
  // ---- Capture ------------------------------------------------------------
  capture: {
    /** Target seconds of turntable footage. */
    targetSeconds: number;
    /** Longest recommended capture, hard-stops the recorder. */
    maxSeconds: number;
    /** Assumed full sweep of the object relative to the camera, degrees. */
    totalSweepDeg: number;
  };

  // ---- Keyframe extraction ------------------------------------------------
  keyframes: {
    /** How many evenly-spaced frames to pull from the clip. */
    count: number;
    /** Cap the long edge of every processed frame (px). Bounds memory. */
    maxEdgePx: number;
    /**
     * Motion-blur reject threshold (variance of Laplacian). Frames below this
     * are dropped and replaced by their nearest sharp neighbour. Lower = keep
     * more frames.
     */
    blurRejectVar: number;
  };

  // ---- Segmentation / matte ----------------------------------------------
  matte: {
    /** Alpha cutoff [0..1] for turning the RMBG soft matte into a mask. */
    threshold: number;
    /**
     * Erode the mask by this many pixels to drop halo/fringe pixels that would
     * otherwise corrupt back-projected geometry. Irregular organic silhouettes
     * fringe badly, so a few px helps a lot.
     */
    erosionPx: number;
    /** Model id (transformers.js `background-removal` task). */
    modelId: string;
    /**
     * Restrict the matte to a centered ellipse (the capture scan zone) so
     * off-center background / the turntable surface never enters the geometry.
     */
    roi: {
      enabled: boolean;
      radiusXFrac: number; // half-width as fraction of image width
      radiusYFrac: number; // half-height as fraction of image height
      featherPx: number; // soft edge
    };
    /** Keep only the largest connected foreground blob (drops stray regions). */
    largestComponentOnly: boolean;
  };

  // ---- Depth --------------------------------------------------------------
  depth: {
    /** transformers.js depth-estimation model id. */
    modelId: string;
    /** Preferred dtype when running on WebGPU. */
    webgpuDtype: 'fp16' | 'fp32';
    /** dtype when falling back to WASM (fp32 is the safe default). */
    wasmDtype: 'fp32' | 'q8';
    /** Frames processed per batch (bounds peak GPU/host memory). */
    batchSize: number;
    /**
     * Bilateral spatial smoothing on each depth map. Kills per-frame speckle
     * from wood grain / mineral texture without blurring silhouettes.
     */
    bilateral: {
      enabled: boolean;
      diameter: number; // px window
      sigmaSpace: number;
      sigmaDepth: number; // in normalized depth units
    };
    /**
     * Cross-frame temporal weight [0..1]. Blends a pixel's depth toward the
     * median of temporally-adjacent keyframes. 0 disables.
     */
    temporalWeight: number;
    /**
     * Depth-Anything outputs *relative* inverse-ish depth. These map the
     * normalized [0..1] output into a metric-ish near/far band (metres) before
     * scale calibration is applied.
     */
    nearMeters: number;
    farMeters: number;
  };

  // ---- Fusion / point cloud ----------------------------------------------
  fusion: {
    /** Keep at most this many points (post-fusion) to bound memory. */
    maxPoints: number;
    /**
     * Point-density retention [0..1]. 1 = keep everything (protects thin
     * driftwood branches). Lower only if memory-bound.
     */
    pointDensity: number;
    /** Voxel size (metres) for dedup/downsample after fusion. 0 disables. */
    voxelSizeM: number;
    /**
     * Statistical outlier removal: drop points whose mean distance to their
     * kNN neighbours exceeds mean + stdRatio*std. Cleans depth-edge flyers.
     */
    outlier: { enabled: boolean; k: number; stdRatio: number };
    /**
     * Confidence weighting. Points near mask edges / with large per-frame depth
     * disagreement get lower confidence and feed Tier-1 densification.
     */
    edgeConfidenceFalloffPx: number;
    /**
     * Drop back-projected points whose normalized depth (0=near,1=far) exceeds
     * this — the background / turntable surface sits near the far plane. 1 = off.
     */
    farCull01: number;
    /** Drop points with confidence below this (edge flyers). 0 = off. */
    minConfidence: number;
  };

  // ---- Guided capture (discrete controlled stills) -----------------------
  guided: {
    /** Number of target viewpoints the user captures around the object. */
    viewpoints: number;
    /** How close (deg) the heading must be to a target to auto-snap. */
    azimuthToleranceDeg: number;
    /** Require the object to be briefly still (low motion) before snapping. */
    requireStill: boolean;
  };

  // ---- Visual hull (silhouette space-carving) ----------------------------
  hull: {
    /** Voxel grid resolution per axis (desktop). */
    gridRes: number;
    /** Voxel grid resolution on mobile / WASM. */
    mobileGridRes: number;
    /** Fraction of views a voxel may fall outside the silhouette and still
     *  count as inside (robustness to matte error). */
    allowedMissFrac: number;
    /** Half-extent (metres, pre-scale) of the carving cube around the object. */
    halfExtentM: number;
  };

  // ---- Pose refinement ----------------------------------------------------
  poses: {
    /** Enable feature-tracking refinement of the ideal turntable poses. */
    refine: boolean;
    /** Max features tracked between adjacent frames. */
    maxFeatures: number;
    /** Refinement iterations (angular-step + axis correction). */
    iterations: number;
    /**
     * Reject frames whose object vertical centroid drifts more than this
     * fraction of image height from the median — i.e. the phone was tilted,
     * which the fixed-elevation turntable model can't handle. 1 = keep all.
     */
    maxVerticalDriftFrac: number;
  };

  // ---- Tier 1: Gaussian Splatting ----------------------------------------
  splat: {
    enabled: boolean;
    /** Hard cap on Gaussian count (mobile-safe budget). */
    maxGaussians: number;
    /** Optimizer iterations. */
    iterations: number;
    learningRate: {
      position: number;
      scale: number;
      rotation: number;
      color: number;
      opacity: number;
    };
    /**
     * Densify Gaussians whose view-space positional gradient exceeds this.
     * LOWER => densify more aggressively (recovers concavities / crevices that
     * single-view depth misses). Raised automatically in low-confidence regions.
     */
    densifyThreshold: number;
    /**
     * Prune Gaussians whose opacity falls below this. LOWER => keep more (so
     * thin protrusions and dim crevice detail survive).
     */
    pruneThreshold: number;
    /** Iteration interval at which densify/prune runs. */
    densifyInterval: number;
    /** Stop densifying after this fraction of total iterations. */
    densifyUntilFrac: number;
    /**
     * Multiplier applied to densifyThreshold in low-confidence regions
     * (<1 => densify even more there).
     */
    lowConfidenceDensifyBoost: number;
  };

  // ---- Mesh path (dry captures only) -------------------------------------
  mesh: {
    /** Only offered when the active preset marks the surface as dry. */
    enabled: boolean;
    /** Poisson reconstruction depth (octree). Higher = more detail + slower. */
    poissonDepth: number;
    /**
     * Target decimation ratio [0..1] of output triangles. Keep HIGH (near 1)
     * for driftwood so thin branches don't snap off. 1 = no decimation.
     */
    keepTriangleRatio: number;
  };

  // ---- Scale calibration --------------------------------------------------
  scale: {
    /** 'reference' = user taps a known length; 'turntable' = disc diameter. */
    mode: 'reference' | 'turntable';
    /** Default reference length in metres (e.g. a 0.10 m marker). */
    referenceLengthM: number;
    /** Default turntable diameter in metres. */
    turntableDiameterM: number;
  };

  // ---- Tier 2: pose-free (experimental, may skip) ------------------------
  posefree: {
    /** Attempt DUSt3R/MASt3R-style onnxruntime-web reconstruction. */
    attempt: boolean;
    /** ONNX model url; empty => feature disabled (graceful skip). */
    modelUrl: string;
    /** Skip if estimated model memory exceeds this (MB). */
    memoryBudgetMB: number;
  };

  // ---- Runtime / engineering ---------------------------------------------
  runtime: {
    /** Allow WASM fallback for depth when WebGPU is unavailable. */
    allowWasmDepthFallback: boolean;
    /** Free models/tensors when leaving a processing screen. */
    aggressiveDispose: boolean;
  };
}

/** Base defaults; presets below override the material-sensitive fields. */
export const BASE_CONFIG: ScannerConfig = {
  capture: { targetSeconds: 30, maxSeconds: 45, totalSweepDeg: 360 },
  keyframes: { count: 30, maxEdgePx: 720, blurRejectVar: 60 },
  matte: {
    threshold: 0.5,
    erosionPx: 3,
    modelId: 'briaai/RMBG-1.4',
    roi: { enabled: true, radiusXFrac: 0.42, radiusYFrac: 0.46, featherPx: 12 },
    largestComponentOnly: true,
  },
  depth: {
    modelId: 'onnx-community/depth-anything-v2-small',
    webgpuDtype: 'fp16',
    wasmDtype: 'fp32',
    batchSize: 4,
    bilateral: { enabled: true, diameter: 7, sigmaSpace: 4, sigmaDepth: 0.08 },
    temporalWeight: 0.35,
    nearMeters: 0.15,
    farMeters: 0.6,
  },
  fusion: {
    maxPoints: 600_000,
    pointDensity: 1.0,
    voxelSizeM: 0.0015,
    outlier: { enabled: true, k: 12, stdRatio: 2.0 },
    edgeConfidenceFalloffPx: 6,
    farCull01: 0.85,
    minConfidence: 0.12,
  },
  guided: { viewpoints: 16, azimuthToleranceDeg: 8, requireStill: true },
  hull: { gridRes: 128, mobileGridRes: 96, allowedMissFrac: 0.12, halfExtentM: 0.25 },
  poses: { refine: true, maxFeatures: 400, iterations: 3, maxVerticalDriftFrac: 0.12 },
  splat: {
    enabled: true,
    maxGaussians: 250_000,
    iterations: 1500,
    learningRate: {
      position: 0.00016,
      scale: 0.005,
      rotation: 0.001,
      color: 0.0025,
      opacity: 0.05,
    },
    densifyThreshold: 0.0002,
    pruneThreshold: 0.005,
    densifyInterval: 100,
    densifyUntilFrac: 0.6,
    lowConfidenceDensifyBoost: 0.5,
  },
  mesh: { enabled: false, poissonDepth: 9, keepTriangleRatio: 0.9 },
  scale: {
    mode: 'turntable',
    referenceLengthM: 0.1,
    turntableDiameterM: 0.25,
  },
  posefree: { attempt: false, modelUrl: '', memoryBudgetMB: 1500 },
  runtime: { allowWasmDepthFallback: true, aggressiveDispose: true },
};

/** Deep-ish clone adequate for our plain-data config. */
function clone(c: ScannerConfig): ScannerConfig {
  return structuredClone(c);
}

/**
 * Material presets. These bake in the driftwood/rock fine-tuning defaults
 * described in the project brief.
 */
export const PRESETS: Record<MaterialPreset, ScannerConfig> = {
  'driftwood-dry': (() => {
    const c = clone(BASE_CONFIG);
    c.mesh.enabled = true; // dry => clean mesh export allowed
    c.mesh.keepTriangleRatio = 0.92; // protect thin branches
    c.splat.pruneThreshold = 0.004; // keep dim protrusion detail
    c.fusion.pointDensity = 1.0;
    return c;
  })(),
  'driftwood-wet': (() => {
    const c = clone(BASE_CONFIG);
    c.mesh.enabled = false; // wet/glossy => splat path
    c.splat.densifyThreshold = 0.00016; // recover crevices under arches
    c.splat.lowConfidenceDensifyBoost = 0.4;
    c.matte.erosionPx = 3;
    return c;
  })(),
  'rock-dry': (() => {
    const c = clone(BASE_CONFIG);
    c.mesh.enabled = true;
    c.mesh.keepTriangleRatio = 0.85;
    c.depth.bilateral.sigmaDepth = 0.06; // sharper mineral facets
    c.fusion.voxelSizeM = 0.002;
    return c;
  })(),
  'rock-wet': (() => {
    const c = clone(BASE_CONFIG);
    c.mesh.enabled = false; // wet rock is glossy => splat
    c.splat.densifyThreshold = 0.00015;
    c.splat.pruneThreshold = 0.004;
    c.matte.erosionPx = 4; // wet rock fringes more against background
    return c;
  })(),
};

let active: ScannerConfig = clone(PRESETS['driftwood-dry']);
let activePreset: MaterialPreset = 'driftwood-dry';

export function getConfig(): ScannerConfig {
  return active;
}

export function getActivePreset(): MaterialPreset {
  return activePreset;
}

export function setPreset(preset: MaterialPreset): ScannerConfig {
  activePreset = preset;
  active = clone(PRESETS[preset]);
  return active;
}

/** Shallow-merge a partial override into the active config (used by the UI). */
export function patchConfig(patch: DeepPartial<ScannerConfig>): ScannerConfig {
  active = deepMerge(active, patch);
  return active;
}

// ---- helpers ---------------------------------------------------------------
export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function deepMerge<T>(base: T, patch: DeepPartial<T>): T {
  const out: Record<string, unknown> = isObject(base) ? { ...base } : ({} as Record<string, unknown>);
  for (const key of Object.keys(patch as object)) {
    const pv = (patch as Record<string, unknown>)[key];
    const bv = out[key];
    if (isObject(pv) && isObject(bv)) {
      out[key] = deepMerge(bv, pv as DeepPartial<unknown>);
    } else if (pv !== undefined) {
      out[key] = pv;
    }
  }
  return out as T;
}
