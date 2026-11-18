// Makrion: a hero page holding one live subject. G-buffer, network, steadying, screen.
import { GBuffer, loadAsset } from './gbuffer.js';
import { loadNetwork } from './nn.js';
import { Display } from './display.js';

const $ = (s) => document.querySelector(s);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const IMG_W = 1280, IMG_H = 800;  // the frame the network trained on

// pointer target and its smoothed position, -1..1 across the screen
const P = { x: 0, y: 0, tx: 0, ty: 0, last: -10 };
const POSE = { hy: 0, hp: 0, hr: 0, by: 0, bp: 0, gx: 0, gy: 0 };
let headX = 0, headY = 0;

function updatePose(t, dt) {
  if (t - P.last > 3.5 && !reduced) { // idle or touch: it looks around by itself
    P.tx = 0.55 * Math.sin(t * 0.31) * Math.cos(t * 0.13);
    P.ty = 0.3 * Math.sin(t * 0.23 + 1.2);
  }
  const k = 1 - Math.exp(-dt * 5.5);
  P.x += (P.tx - P.x) * k; P.y += (P.ty - P.y) * k;
  // eyes lead, the head catches up
  const h = 1 - Math.exp(-dt * 2.6);
  headX += (P.x - headX) * h; headY += (P.y - headY) * h;
  const breath = reduced ? 0 : Math.sin(t * 0.95);
  POSE.hy = headX * 0.52; POSE.hp = headY * 0.20 + breath * 0.005; POSE.hr = -headX * 0.05;
  POSE.by = headX * 0.12; POSE.bp = breath * 0.006;
  POSE.gx = (P.x - headX) * 0.7 + P.x * 0.2; POSE.gy = -(P.y - headY) * 0.45 - P.y * 0.14;
}

async function start() {
  const canvas = $('#gpu');
  const fail = (e) => { console.warn(e); $('#loader').style.display = 'none'; $('#nogpu').hidden = false; };
  if (!navigator.gpu) return fail('no webgpu');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('no adapter');
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.addEventListener('uncapturederror', (e) => console.error(e.error.message));
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });

  const progress = (p) => { $('#load-pct').textContent = Math.round(p * 100); $('#load-line').style.transform = `scaleX(${p})`; };
  const asset = await loadAsset(device, './assets/', (p) => progress(p * 0.6));
  const gb = new GBuffer(device, asset);
  // a phone gets half the pixels; the network trained at full size, so expect it softer there
  const phone = /Android|iPhone|iPad/i.test(navigator.userAgent);
  let RW = phone ? IMG_W / 2 : IMG_W, RH = phone ? IMG_H / 2 : IMG_H;
  gb.resize(RW, RH);
  $('#load-label').textContent = 'Waking the network';
  progress(0.7);
  const net = await loadNetwork(device, './assets/');
  await net.build(RW, RH, gb);
  const disp = new Display(device, ctx, format, gb, net);
  disp.bind();
  progress(1);

  let W = 1, H = 1;
  const resize = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = Math.max(1, Math.round(innerWidth * dpr)); H = Math.max(1, Math.round(innerHeight * dpr));
    canvas.width = W; canvas.height = H;
  };
  addEventListener('resize', resize); resize();

  // fill the screen like a poster. On wide windows only the bottom is cropped, so the head stays in view.
  const cover = () => {
    const s = Math.max(W / IMG_W, H / IMG_H), w = IMG_W * s, h = IMG_H * s;
    return { x: (W - w) / 2, y: h > H && W / H > IMG_W / IMG_H ? 0 : (H - h) / 2, w, h };
  };

  const hx = [0, 1 / 2, 1 / 4, 3 / 4, 1 / 8, 5 / 8, 3 / 8, 7 / 8], hy = [0, 1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9, 2 / 9, 5 / 9];
  const born = performance.now();
  let last = born, n = 0, gpuMs = 8, inFlight = 0, statT = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    if (inFlight > 1 || innerWidth < 2 || window.__pause) return; // never queue up on a slow card, nor draw to a hidden pane
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    updatePose(now / 1000, dt);
    const age = (now - born) / 1000;
    const enc = device.createCommandEncoder();
    const j = n++ % 8;
    gb.encode(enc, POSE, [(hx[j] - 0.5) * 2 / RW, (hy[j] - 0.5) * 2 / RH]); // one pixel is 2 / size in clip units
    net.encode(enc);
    disp.encode(enc, cover(), W, H, reduced ? 1 : smooth(0.7, 2.8, age), smooth(0, 0.8, age));
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    inFlight++;
    device.queue.onSubmittedWorkDone().then(() => { inFlight--; gpuMs += (performance.now() - t0 - gpuMs) * 0.08; });
    statT += dt;
    if (statT > 0.5) { statT = 0; $('#stat').textContent = `Live · ${RW}×${RH} · ${gpuMs.toFixed(1)} ms a frame`; }
  }
  await document.fonts.ready;
  $('#loader').style.opacity = 0;
  setTimeout(() => $('#loader').remove(), 800);
  setTimeout(() => document.documentElement.classList.add('ready'), reduced ? 0 : 900);
  requestAnimationFrame(frame);
  window.__makrion = { P, POSE, net, gb, disp, get gpuMs() { return gpuMs; } };

  addEventListener('pointermove', (e) => {
    P.tx = (e.clientX / innerWidth) * 2 - 1;
    P.ty = (e.clientY / innerHeight) * 2 - 1;
    P.last = performance.now() / 1000;
  }, { passive: true });
}

start().catch((e) => { console.error(e); $('#load-label').textContent = 'Could not start: ' + e.message; });
