import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import type { PointCloud } from '../types';
import type { TriMesh } from '../mesh/poisson';

/**
 * glTF / GLB exporters. GLB matters because the target is a three.js aquascaping
 * app — a GLB drops straight in via GLTFLoader.
 *
 *  - `meshToGlb`       : the dry-capture mesh path (recommended for import).
 *  - `pointCloudToGlb` : a POINTS-primitive glTF, handy when only the Tier-0
 *    cloud exists.
 */

export async function meshToGlb(mesh: TriMesh): Promise<Blob> {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
  if (mesh.normals) geom.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
  if (mesh.colors) geom.setAttribute('color', new THREE.BufferAttribute(mesh.colors, 3));
  geom.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  if (!mesh.normals) geom.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: !!mesh.colors,
    roughness: 0.85,
    metalness: 0.0,
  });
  const object = new THREE.Mesh(geom, material);
  return exportGlb(object);
}

export async function pointCloudToGlb(cloud: PointCloud): Promise<Blob> {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3));
  const material = new THREE.PointsMaterial({ size: 0.003, vertexColors: true });
  const object = new THREE.Points(geom, material);
  return exportGlb(object);
}

function exportGlb(object: THREE.Object3D): Promise<Blob> {
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(
      object,
      (result) => {
        const glb = result as ArrayBuffer;
        resolve(new Blob([glb], { type: 'model/gltf-binary' }));
      },
      (err) => reject(err),
      { binary: true },
    );
  });
}
