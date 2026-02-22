// Main entry point — Three.js scene setup
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadAll, getWkr, getAllWkr } from './data.js';
import { buildMap, updateMap } from './map.js';
import { initUI, getState, showTooltip, showInfoPanel,
  updateLegend, initChips } from './ui.js';

let scene, camera, renderer, controls;
let mapGroup, meshes;
let raycaster, mouse;
let hovered = null;

function initScene() {
  const canvas = document.getElementById('canvas');
  renderer = new THREE.WebGLRenderer({
    canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x1a1a2e);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    55, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 55, 35);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.maxPolarAngle = Math.PI / 2.1;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 20, 10);
  scene.add(dir);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function refresh() {
  const { party, metric, extrude } = getState();
  const r = updateMap(meshes, window._geoData,
    getAllWkr(), party, metric, extrude);
  updateLegend(r.min, r.max);
}

function onMouseMove(e) {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshList = Object.values(meshes);
  const hits = raycaster.intersectObjects(meshList);
  if (hovered) {
    hovered.material.emissive.setHex(0x000000);
    hovered = null;
  }
  if (hits.length > 0) {
    hovered = hits[0].object;
    hovered.material.emissive.setHex(0x333333);
    const wkr = hovered.userData.wkr;
    const d = getWkr(wkr);
    const { party, metric } = getState();
    showTooltip(e.clientX, e.clientY, d, metric, party);
  } else {
    showTooltip(0, 0, null);
  }
}

function onClick(e) {
  if (!hovered) {
    showInfoPanel(null); return;
  }
  const d = getWkr(hovered.userData.wkr);
  showInfoPanel(d);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function init() {
  initScene();
  scene.add(new THREE.AxesHelper(10));
  animate();
  try {
    const { wkrData, geoData, summary } = await loadAll();
    window._geoData = geoData;
    const result = buildMap(geoData);
    mapGroup = result.group;
    meshes = result.meshes;
    scene.add(mapGroup);
    initUI(refresh);
    initChips(summary);
    refresh();
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('click', onClick);
  } catch (e) {
    const d = document.createElement('div');
    d.style.cssText =
      'position:fixed;top:60px;left:230px;' +
      'color:red;font-size:16px;z-index:99';
    d.textContent = 'Load error: ' + e.message;
    document.body.appendChild(d);
    console.error(e);
  }
}

init();
