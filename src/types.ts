/** Shared data types passed between pipeline stages. */

/** A single processed keyframe: RGB image data plus provenance. */
export interface Keyframe {
  index: number;
  /** Time in the source clip (seconds). */
  time: number;
  width: number;
  height: number;
  /** RGBA pixels, length = width*height*4. */
  rgba: Uint8ClampedArray;
  /** Sharpness score (variance of Laplacian) used for blur rejection. */
  sharpness: number;
}

/** Foreground matte for a keyframe (single-channel alpha, 0..255). */
export interface Matte {
  width: number;
  height: number;
  /** length = width*height, 255 = foreground. */
  alpha: Uint8Array;
}

/** Per-keyframe depth map (normalized 0..1, 1 = far). */
export interface DepthMap {
  width: number;
  height: number;
  /** length = width*height, normalized 0..1. */
  depth: Float32Array;
}

/** 4x4 column-major camera-to-world matrix + pinhole intrinsics. */
export interface CameraPose {
  index: number;
  /** column-major 4x4 camera-to-world */
  matrix: Float32Array;
  /** focal length in pixels (fx == fy assumed) */
  focalPx: number;
  cx: number;
  cy: number;
}

/** Fused coloured point cloud with per-point confidence. */
export interface PointCloud {
  count: number;
  /** xyz, length = count*3 (metres, after scale calibration). */
  positions: Float32Array;
  /** rgb 0..1, length = count*3. */
  colors: Float32Array;
  /** optional per-point normals, length = count*3. */
  normals?: Float32Array;
  /** per-point confidence 0..1, length = count. */
  confidence: Float32Array;
}

/** A trained Gaussian splat scene (Tier 1). */
export interface SplatScene {
  count: number;
  positions: Float32Array; // count*3
  scales: Float32Array; // count*3 (log-space during training, linear here)
  rotations: Float32Array; // count*4 (quaternion xyzw)
  colors: Float32Array; // count*3 (SH DC / linear rgb 0..1)
  opacities: Float32Array; // count
}

export type ProgressFn = (stage: string, fraction: number, detail?: string) => void;
