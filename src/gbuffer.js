// Rasterizes the model into what the network reads. deform() must match pipeline/rig.py.

const SHADER = /* wgsl */ `
struct Pose { head: vec4f, body: vec4f };   // hp hy hr _ | bp by gx gy
struct U { now: Pose, prev: Pose, jitter: vec4f, eyeC: vec4f };
@group(0) @binding(0) var<uniform> u: U;
const HEAD_PIVOT = vec3f(0.0, 1.105, 0.0);
const BODY_PIVOT = vec3f(0.0, 0.60, 0.0);
const CAM = vec3f(0.0, 1.10, 1.10);
const K = 77.3 / 18.0;
const ASPECT = 1280.0 / 800.0;
const L_KEY = vec3f(-0.75, 0.75, 0.95) / length(vec3f(-0.75, 0.75, 0.95));

fn rod(p: vec3f, rv: vec3f) -> vec3f {
  let a = length(rv);
  if (a < 1e-6) { return p; }
  let k = rv / a; let c = cos(a); let s = sin(a);
  return p * c + cross(k, p) * s + k * dot(k, p) * (1.0 - c);
}
fn iris(p: vec3f, g: f32, pose: Pose, dir: bool) -> vec3f {
  if (g != 1.0 && g != 2.0) { return p; }
  let rv = vec3f(-pose.body.w, pose.body.z, 0.0);
  if (dir) { return rod(p, rv); }
  let c = u.eyeC.xyz * vec3f(select(-1.0, 1.0, g == 1.0), 1.0, 1.0);
  return rod(p - c, rv) + c;
}
fn deform(p: vec3f, w: f32, pose: Pose) -> vec3f {
  let q = rod(p - HEAD_PIVOT, pose.head.xyz * w) + HEAD_PIVOT;
  return rod(q - BODY_PIVOT, vec3f(pose.body.xy, 0.0)) + BODY_PIVOT;
}
fn deformDir(n: vec3f, w: f32, pose: Pose) -> vec3f {
  return rod(rod(n, pose.head.xyz * w), vec3f(pose.body.xy, 0.0));
}
fn project(p: vec3f, jit: vec2f) -> vec4f {
  let D = CAM.z - p.z;
  return vec4f(K * (p.x - CAM.x) + jit.x * D, K * ASPECT * (p.y - CAM.y) + jit.y * D, (D - 0.3) / 1.6 * D, D);
}
struct VO { @builtin(position) pos: vec4f, @location(0) n: vec3f, @location(1) uv: vec2f, @location(2) d: f32,
  @location(3) cur: vec4f, @location(4) pre: vec4f };
@vertex fn vs(@location(0) p16: vec4u, @location(1) uv: vec2f, @location(2) n8: vec4f) -> VO {
  var o: VO;
  let p = vec3f(p16.xyz) / 65535.0 * 2.0 - vec3f(0.5);
  let wh = f32(p16.w) / 65535.0;
  let g = n8.w * 127.0;
  let pc = deform(iris(p, g, u.now, false), wh, u.now);
  let pp = deform(iris(p, g, u.prev, false), wh, u.prev);
  o.n = deformDir(iris(n8.xyz, g, u.now, true), wh, u.now);
  o.cur = project(pc, vec2f(0.0));
  o.pre = project(pp, vec2f(0.0));
  o.pos = project(pc, u.jitter.xy);
  o.uv = uv;
  o.d = (CAM.z - pc.z - 1.0) / 0.4;
  return o;
}
@group(0) @binding(1) var alb: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
struct FO { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec4f, @location(3) m: vec4f };
@fragment fn fs(i: VO) -> FO {
  var o: FO;
  let n = normalize(i.n);
  o.a = vec4f(textureSample(alb, smp, vec2f(i.uv.x, 1.0 - i.uv.y)).rgb, 1.0);
  o.b = vec4f(n, i.d);
  o.c = vec4f(i.uv, max(dot(n, L_KEY), 0.0), select(0.0, 1.0, i.uv.x >= 0.75));
  o.m = vec4f(i.cur.xy / i.cur.w - i.pre.xy / i.pre.w, 0.0, 1.0);
  return o;
}`;

export const FORMATS = { gA: 'rgba8unorm', gB: 'rgba16float', gC: 'rgba16float', motion: 'rg16float' };

export class GBuffer {
  constructor(device, asset) {
    this.device = device; this.asset = asset;
    const d = device;
    this.ubuf = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const m = d.createShaderModule({ code: SHADER });
    this.pipe = d.createRenderPipeline({ layout: 'auto',
      vertex: { module: m, entryPoint: 'vs', buffers: [{ arrayStride: 16, attributes: [
        { shaderLocation: 0, offset: 0, format: 'uint16x4' },
        { shaderLocation: 1, offset: 8, format: 'unorm16x2' },
        { shaderLocation: 2, offset: 12, format: 'snorm8x4' }] }] },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format: FORMATS.gA }, { format: FORMATS.gB }, { format: FORMATS.gC }, { format: FORMATS.motion }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' } });
    const sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8, addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.bg = d.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.ubuf } }, { binding: 1, resource: asset.albedo.createView() }, { binding: 2, resource: sampler }] });
    this.eye = new Float32Array([...asset.meta.eyeC, 0]);
    this.prev = null;
  }

  resize(W, H) {
    if (this.W === W && this.H === H) return;
    this.W = W; this.H = H;
    const d = this.device;
    this.tex = {};
    for (const [k, format] of Object.entries(FORMATS)) this.tex[k] = d.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.view = Object.fromEntries(Object.entries(this.tex).map(([k, t]) => [k, t.createView()]));
    this.depthView = d.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT }).createView();
  }

  encode(enc, pose, jitter = [0, 0]) {
    const pack = (p) => [p.hp || 0, p.hy || 0, p.hr || 0, 0, p.bp || 0, p.by || 0, p.gx || 0, p.gy || 0];
    const a = new Float32Array(24);
    a.set(pack(pose), 0); a.set(pack(this.prev || pose), 8);
    a[16] = jitter[0]; a[17] = jitter[1];
    a.set(this.eye, 20);
    this.device.queue.writeBuffer(this.ubuf, 0, a);
    this.prev = { ...pose };
    const c = (view) => ({ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' });
    const pass = enc.beginRenderPass({ colorAttachments: [c(this.view.gA), c(this.view.gB), c(this.view.gC), c(this.view.motion)],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg);
    pass.setVertexBuffer(0, this.asset.verts);
    pass.setIndexBuffer(this.asset.index, 'uint32');
    pass.drawIndexed(this.asset.meta.indexCount);
    pass.end();
  }
}

export async function loadAsset(device, base = './assets/', onProgress = () => {}) {
  const meta = await (await fetch(base + 'makrion.json')).json();
  const res = await fetch(base + 'makrion.bin');
  const total = +res.headers.get('content-length') || 0;
  const chunks = []; let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) onProgress(got / total);
  }
  const bin = new Uint8Array(got);
  let o = 0; for (const c of chunks) { bin.set(c, o); o += c.length; }
  const buffer = (data, usage) => {
    const b = device.createBuffer({ size: (data.byteLength + 3) & ~3, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };
  const blob = await (await fetch(base + 'albedo.webp')).blob();
  const levels = 5;
  const albedo = device.createTexture({ size: [8192, 2048], format: 'rgba8unorm', mipLevelCount: levels, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  for (let l = 0; l < levels; l++) {
    const w = 8192 >> l, h = 2048 >> l;
    const bmp = await createImageBitmap(blob, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high', colorSpaceConversion: 'none' });
    device.queue.copyExternalImageToTexture({ source: bmp }, { texture: albedo, mipLevel: l }, [w, h]);
    bmp.close();
  }
  return { meta, albedo, verts: buffer(bin.subarray(0, meta.vertexCount * 16), GPUBufferUsage.VERTEX), index: buffer(bin.subarray(meta.indexOffset), GPUBufferUsage.INDEX) };
}
