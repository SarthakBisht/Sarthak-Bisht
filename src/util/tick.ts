/**
 * Yield to the event loop so the browser can paint and stay responsive during
 * long CPU-bound stages. `setTimeout(0)` (a macrotask) reliably lets a paint
 * happen, unlike microtask-only awaits.
 */
export const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));
