import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

// Driftwood 3D Scanner — fully client-side PWA.
// Everything runs in-browser; models are fetched from the Hugging Face CDN on
// first load and then cached by the service worker for offline use.
export default defineConfig({
  base: './',
  build: {
    target: 'esnext', // top-level await + WebGPU code paths
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // transformers.js ships its own wasm/ort backend; let Vite pre-bundle it.
    exclude: ['onnxruntime-web'],
  },
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      manifest: false, // we ship our own public/manifest.webmanifest
      workbox: {
        // Precache the app shell but NOT the multi-MB ONNX wasm — those are
        // runtime-cached on first use instead (precaching them would mean a
        // ~50 MB service-worker install).
        globPatterns: ['**/*.{js,css,html,svg,png,ico,webmanifest}'],
        globIgnores: ['**/ort-*.wasm', '**/*.asyncify.wasm'],
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        // Runtime-cache the heavy model shards + wasm so the app works fully
        // offline after the first successful load.
        runtimeCaching: [
          {
            // Same-origin ONNX Runtime wasm emitted into /assets by Vite.
            urlPattern: ({ url, sameOrigin }) => sameOrigin && url.pathname.endsWith('.wasm'),
            handler: 'CacheFirst',
            options: {
              cacheName: 'ort-wasm',
              expiration: { maxEntries: 12, maxAgeSeconds: 60 * 60 * 24 * 60 },
              cacheableResponse: { statuses: [0, 200] },
              rangeRequests: true,
            },
          },
          {
            urlPattern: ({ url }) =>
              url.hostname === 'huggingface.co' ||
              url.hostname === 'cdn-lfs.huggingface.co' ||
              url.hostname === 'cdn-lfs-us-1.hf.co' ||
              url.hostname.endsWith('.hf.co'),
            handler: 'CacheFirst',
            options: {
              cacheName: 'hf-models',
              expiration: {
                maxEntries: 128,
                maxAgeSeconds: 60 * 60 * 24 * 60, // 60 days
              },
              cacheableResponse: { statuses: [0, 200] },
              rangeRequests: true,
            },
          },
          {
            // ONNX Runtime / transformers.js wasm shards served from jsDelivr.
            urlPattern: ({ url }) => url.hostname === 'cdn.jsdelivr.net',
            handler: 'CacheFirst',
            options: {
              cacheName: 'wasm-cdn',
              expiration: { maxEntries: 64, maxAgeSeconds: 60 * 60 * 24 * 60 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
      devOptions: { enabled: false },
    }),
  ],
});
