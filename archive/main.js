import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';


const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 1.6, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(5, 5, 5);
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));

const loader = new GLTFLoader();
let currentModel = null;
let currentAction = "";

function loadModel(action) {
  if (action === currentAction) return;
  currentAction = action;
  document.getElementById('loading').innerText = `Loading: ${action}.glb`;

  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse(child => {
      if (child.isMesh) {
        child.geometry.dispose();
        child.material.dispose();
      }
    });
  }

  loader.load(`./3d_models/${action.toLowerCase()}.glb`, (gltf) => {
    currentModel = gltf.scene;
    scene.add(currentModel);
    document.getElementById('loading').innerText = `Showing: ${action}`;
  }, undefined, (error) => {
    console.error(`Failed to load model for ${action}`, error);
    document.getElementById('loading').innerText = `Model for "${action}" not found.`;
  });
}

async function pollAction() {
  try {
    const res = await fetch('/current_action');
    const action = (await res.text()).trim();
    if (action) loadModel(action);
  } catch (err) {
    console.error("Polling error:", err);
  }
}

setInterval(pollAction, 2000);

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});