// viz/brain_heatmap/js/brain.js
import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { coolwarmRgb } from './colors.js';

const canvas = document.getElementById('three-canvas');
const container = document.getElementById('canvas-container');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0d);

const camera = new THREE.PerspectiveCamera(
    45, container.clientWidth / container.clientHeight, 0.1, 1000
);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 2, 3);
scene.add(dir);

let brainMesh = null;
let vizData = null;
let currentType = null;
let currentLayer = 'mid';

export function init(data) {
    vizData = data;
    const types = Object.keys(data.activation_maps).sort();
    currentType = types[0];

    const sel = document.getElementById('type-select');
    types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        sel.appendChild(opt);
    });
    sel.value = currentType;

    loadMesh();
    animate();
    window.addEventListener('resize', onResize);
    document.getElementById('status').textContent = 'Ready';
}

export function setType(t) {
    currentType = t;
    applyColors();
}

export function setLayer(l) {
    currentLayer = l;
    applyColors();
}

function loadMesh() {
    new OBJLoader().load(
        'assets/brain.obj',
        obj => {
            obj.traverse(child => {
                if (child.isMesh) buildMesh(child.geometry.toNonIndexed());
            });
        },
        undefined,
        () => buildMesh(new THREE.IcosahedronGeometry(1, 6).toNonIndexed())
    );
}

function buildMesh(geometry) {
    const mat = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 30 });
    brainMesh = new THREE.Mesh(geometry, mat);
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    brainMesh.position.sub(center);
    const size = geometry.boundingBox.getSize(new THREE.Vector3()).length();
    brainMesh.scale.setScalar(2.5 / size);
    scene.add(brainMesh);
    applyColors();
}

function applyColors() {
    if (!brainMesh || !vizData || !currentType) return;
    const entry = vizData.activation_maps[currentType];
    if (!entry) return;
    const layerData = entry[currentLayer];
    if (!layerData || layerData.length === 0) return;
    const nNeurons = layerData.length;
    const positions = brainMesh.geometry.attributes.position;
    const nVerts = positions.count;
    const groupSize = Math.ceil(nVerts / nNeurons);
    const colors = new Float32Array(nVerts * 3);
    for (let v = 0; v < nVerts; v++) {
        const idx = Math.min(Math.floor(v / groupSize), nNeurons - 1);
        const [r, g, b] = coolwarmRgb(layerData[idx]);
        colors[v * 3]     = r;
        colors[v * 3 + 1] = g;
        colors[v * 3 + 2] = b;
    }
    brainMesh.geometry.setAttribute(
        'color', new THREE.BufferAttribute(colors, 3)
    );
    brainMesh.geometry.attributes.color.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
