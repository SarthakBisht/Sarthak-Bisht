import { getConfig } from '../config';

/**
 * Thin wrapper around getUserMedia + MediaRecorder for capturing a turntable
 * clip from the rear camera. Fully local — the Blob never leaves the device.
 */
export interface RecorderHandle {
  stream: MediaStream;
  video: HTMLVideoElement;
  start(): void;
  stop(): Promise<Blob>;
  dispose(): void;
  isRecording(): boolean;
}

const MIME_CANDIDATES = [
  'video/mp4;codecs=avc1',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

function pickMime(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
}

export async function createRecorder(previewParent: HTMLElement): Promise<RecorderHandle> {
  const cfg = getConfig();
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });

  const video = document.createElement('video');
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  video.className = 'capture-preview';
  previewParent.appendChild(video);
  await video.play().catch(() => undefined);

  const mimeType = pickMime();
  let recorder: MediaRecorder | null = null;
  let chunks: Blob[] = [];
  let recording = false;
  let autoStop: number | undefined;

  return {
    stream,
    video,
    isRecording: () => recording,
    start() {
      if (recording) return;
      chunks = [];
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.start(200);
      recording = true;
      // Hard safety stop.
      autoStop = window.setTimeout(() => {
        if (recording) recorder?.stop();
      }, cfg.capture.maxSeconds * 1000);
    },
    stop() {
      return new Promise<Blob>((resolve, reject) => {
        if (!recorder || !recording) {
          reject(new Error('Not recording.'));
          return;
        }
        if (autoStop) clearTimeout(autoStop);
        recorder.onstop = () => {
          recording = false;
          resolve(new Blob(chunks, { type: mimeType || 'video/webm' }));
        };
        recorder.stop();
      });
    },
    dispose() {
      if (autoStop) clearTimeout(autoStop);
      try {
        recorder?.state !== 'inactive' && recorder?.stop();
      } catch {
        /* ignore */
      }
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      video.remove();
    },
  };
}
