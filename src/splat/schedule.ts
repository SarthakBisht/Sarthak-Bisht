import { getConfig } from '../config';

/**
 * Densification / prune schedule for the Gaussian-splat stage. Centralises the
 * "when do we densify vs prune" logic so both the current geometric refiner and
 * a future photometric optimizer read the same knobs from config.ts.
 */
export interface SplatSchedule {
  totalIterations: number;
  densifyInterval: number;
  densifyUntilIter: number;
  densifyThreshold: number;
  pruneThreshold: number;
  lowConfidenceDensifyBoost: number;
  maxGaussians: number;
}

export function makeSchedule(): SplatSchedule {
  const s = getConfig().splat;
  return {
    totalIterations: s.iterations,
    densifyInterval: s.densifyInterval,
    densifyUntilIter: Math.round(s.iterations * s.densifyUntilFrac),
    densifyThreshold: s.densifyThreshold,
    pruneThreshold: s.pruneThreshold,
    lowConfidenceDensifyBoost: s.lowConfidenceDensifyBoost,
    maxGaussians: s.maxGaussians,
  };
}

/**
 * Effective densify threshold for a region of given confidence. Low-confidence
 * regions (concavities, crevices, arch undersides — where single-view depth is
 * unreliable) densify MORE, so multi-view fusion detail is preserved.
 */
export function effectiveDensifyThreshold(base: number, boost: number, confidence: number): number {
  // confidence 0 => threshold * boost (<1 => densify more); confidence 1 => base.
  const factor = boost + (1 - boost) * clamp01(confidence);
  return base * factor;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
