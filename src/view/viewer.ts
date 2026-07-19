import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { PointCloud, SplatScene, ProgressFn } from '../types';
import type { TriMesh } from '../mesh/poisson';

/**
 * three.js WebGPU orbit viewer (rotate / zoom / pan). Renders three kinds of
 * asset:
 *   - point cloud  (Tier 0)
 *   - gaussian splat scene (Tier 1) — previewed here as size/opacity-scaled
 *     point sprites; export to .splat/.ply for a full splat renderer.
 *   - triangle mesh (dry path)
 *
 * Uses WebGPURenderer, which falls back to WebGL2 automatically when WebGPU is
 * unavailable, so the viewer works even in the depth-only WASM configuration.
 */
export class OrbitViewer {
  private renderer!: WebGPURenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private content: THREE.Object3D | null = null;
  private raf = 0;
  private container: HTMLElement;
  private ready: Promise<void>;

  constructor(container: HTMLElement) {
    this.container = container;
    this.scene.background = new THREE.Color(0x0b0f14);
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 100);
    this.camera.position.set(0.3, 0.2, 0.4);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(1, 2, 1);
    this.scene.add(dir);
    const grid = new THREE.GridHelper(1, 20, 0x2f6d8c, 0x1a2a33);
    (grid.material as THREE.Material).opacity = 0.25;
    (grid.material as THREE.Material).transparent = true;
    this.scene.add(grid);

    this.ready = this.initRenderer(w, h);
  }

  private async initRenderer(w: number, h: number): Promise<void> {
    this.renderer = new WebGPURenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    await this.renderer.init();
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    window.addEventListener('resize', this.onResize);
    this.animate();
  }

  whenReady(): Promise<void> {
    return this.ready;
  }

  backendName(): string {
    // WebGPURenderer exposes `backend` after init.
    const backend = (this.renderer as unknown as { backend?: { isWebGPUBackend?: boolean } }).backend;
    return backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  }

  private onResize = () => {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  private animate = () => {
    this.raf = requestAnimationFrame(this.animate);
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  };

  private clearContent() {
    if (this.content) {
      this.scene.remove(this.content);
      disposeObject(this.content);
      this.content = null;
    }
  }

  showPointCloud(cloud: PointCloud, pointSize = 0.0025) {
    this.clearContent();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3));
    const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true });
    const pts = new THREE.Points(geom, mat);
    this.content = pts;
    this.scene.add(pts);
    this.frameContent(cloud.positions);
  }

  showSplatScene(scene: SplatScene) {
    // Preview: draw as colour+opacity point sprites sized by mean scale.
    this.clearContent();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(scene.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(scene.colors, 3));
    const meanScale = averageScale(scene.scales);
    const mat = new THREE.PointsMaterial({
      size: Math.max(0.002, meanScale * 2),
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.85,
    });
    const pts = new THREE.Points(geom, mat);
    this.content = pts;
    this.scene.add(pts);
    this.frameContent(scene.positions);
  }

  showMesh(mesh: TriMesh) {
    this.clearContent();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    if (mesh.normals) geom.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
    if (mesh.colors) geom.setAttribute('color', new THREE.BufferAttribute(mesh.colors, 3));
    geom.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: !!mesh.colors,
      roughness: 0.85,
      metalness: 0.0,
      side: THREE.DoubleSide,
    });
    const obj = new THREE.Mesh(geom, mat);
    this.content = obj;
    this.scene.add(obj);
    this.frameContent(mesh.positions);
  }

  /** Center the camera/orbit target on the content's bounding box. */
  private frameContent(positions: Float32Array) {
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < positions.length; i += 3) {
      box.expandByPoint(v.set(positions[i], positions[i + 1], positions[i + 2]));
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.4, size * 0.8));
    this.camera.near = size / 100;
    this.camera.far = size * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    this.clearContent();
    this.controls?.dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }
}

function disposeObject(obj: THREE.Object3D) {
  obj.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const mat = mesh.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else mat?.dispose?.();
  });
}

function averageScale(scales: Float32Array): number {
  let s = 0;
  const n = scales.length / 3;
  for (let i = 0; i < scales.length; i += 3) {
    s += (scales[i] + scales[i + 1] + scales[i + 2]) / 3;
  }
  return n > 0 ? s / n : 0.005;
}

/** Async factory that resolves once the WebGPU/WebGL backend is ready. */
export async function createViewer(container: HTMLElement, onProgress?: ProgressFn): Promise<OrbitViewer> {
  onProgress?.('viewer', 0.5, 'initializing renderer');
  const viewer = new OrbitViewer(container);
  await viewer.whenReady();
  onProgress?.('viewer', 1, viewer.backendName());
  return viewer;
}
