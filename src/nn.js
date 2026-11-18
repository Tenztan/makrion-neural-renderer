// The network in compute shaders, built from the JSON train.py exports.
// A feature map is one storage buffer of vec4f, laid out [y][x][ch/4]. Weights sit in a uniform buffer per layer.

const WG = 8, TILE = 16;

function convCode(l, in4, out4) {
  const K = l.k, T = K * K * in4, r = K >> 1;
  return /* wgsl */ `
struct D { inW: u32, inH: u32, outW: u32, outH: u32 };
struct Weights { m: array<mat4x4f, ${out4 * T}>, b: array<vec4f, ${out4}> };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4f>;
@group(0) @binding(3) var<uniform> w: Weights;
@group(0) @binding(4) var<storage, read> tileOn: array<u32>;
@group(0) @binding(5) var<uniform> t: vec4u;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.outW || id.y >= d.outH) { return; }
  let at = (id.y * d.outW + id.x) * ${out4}u;
  // empty background costs nothing. t = pixels per tile at this layer's scale, tiles across
  if (tileOn[(id.y / t.x) * t.y + id.x / t.x] == 0u) {
    for (var o = 0u; o < ${out4}u; o++) { y[at + o] = vec4f(0.0); }
    return;
  }
  let cx = i32(id.x * ${l.s}u); let cy = i32(id.y * ${l.s}u);
  let iw = i32(d.inW); let ih = i32(d.inH);
  var v: array<vec4f, ${T}>;
  var n = 0u;
  for (var ky = -${r}; ky <= ${r}; ky++) {
    let py = clamp(cy + ky, 0, ih - 1) * iw;
    for (var kx = -${r}; kx <= ${r}; kx++) {
      let base = u32(py + clamp(cx + kx, 0, iw - 1)) * ${in4}u;
      for (var c = 0u; c < ${in4}u; c++) { v[n] = x[base + c]; n++; }
    }
  }
  for (var o = 0u; o < ${out4}u; o++) {
    var a = w.b[o];
    let m0 = o * ${T}u;
    for (var i = 0u; i < ${T}u; i++) { a += w.m[m0 + i] * v[i]; }
    y[at + o] = ${l.last ? 'a' : 'max(a, a * 0.1)'};  // leaky activation, LEAK in train.py
  }
}`;
}

// 2x bilinear of a (half-pixel centres, as F.interpolate align_corners=False), then skip b appended.
const UPCAT = /* wgsl */ `
override A4: u32; override B4: u32;
struct D { aW: u32, aH: u32, W: u32, H: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> a: array<vec4f>;
@group(0) @binding(2) var<storage, read> b: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> y: array<vec4f>;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.W || id.y >= d.H) { return; }
  let fx = (f32(id.x) + 0.5) * 0.5 - 0.5; let fy = (f32(id.y) + 0.5) * 0.5 - 0.5;
  let x0 = floor(fx); let y0 = floor(fy);
  let tx = fx - x0; let ty = fy - y0;
  let ax = i32(d.aW) - 1; let ay = i32(d.aH) - 1;
  let xa = clamp(i32(x0), 0, ax); let xb = clamp(i32(x0) + 1, 0, ax);
  let ya = clamp(i32(y0), 0, ay); let yb = clamp(i32(y0) + 1, 0, ay);
  let at = (id.y * d.W + id.x) * (A4 + B4);
  let aw = i32(d.aW);
  for (var c = 0u; c < A4; c++) {
    let p00 = a[u32(ya * aw + xa) * A4 + c]; let p01 = a[u32(ya * aw + xb) * A4 + c];
    let p10 = a[u32(yb * aw + xa) * A4 + c]; let p11 = a[u32(yb * aw + xb) * A4 + c];
    y[at + c] = mix(mix(p00, p01, tx), mix(p10, p11, tx), ty);
  }
  let base = (id.y * d.W + id.x) * B4;
  for (var c = 0u; c < B4; c++) { y[at + A4 + c] = b[base + c]; }
}`;

// Packs the G-buffer and neural texture into the first feature map.
const INPUT = /* wgsl */ `
struct D { W: u32, H: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var gA: texture_2d<f32>;
@group(0) @binding(2) var gB: texture_2d<f32>;
@group(0) @binding(3) var gC: texture_2d<f32>;
@group(0) @binding(4) var nA: texture_2d<f32>;
@group(0) @binding(5) var nB: texture_2d<f32>;
@group(0) @binding(6) var smp: sampler;
@group(0) @binding(7) var<storage, read_write> y: array<vec4f>;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.W || id.y >= d.H) { return; }
  let p = vec2i(id.xy);
  let a = textureLoad(gA, p, 0); let c = textureLoad(gC, p, 0);
  let uv = vec2f(c.x, 1.0 - c.y);
  let at = (id.y * d.W + id.x) * 5u;
  y[at] = a;
  y[at + 1u] = textureLoad(gB, p, 0);
  y[at + 2u] = c;
  y[at + 3u] = textureSampleLevel(nA, smp, uv, 0.0) * a.a;
  y[at + 4u] = textureSampleLevel(nB, smp, uv, 0.0) * a.a;
}`;

// Which tiles hold any of the model, grown by one so the network's reach is covered.
const TILES = /* wgsl */ `
struct D { W: u32, H: u32, tw: u32, th: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var gA: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> hit: array<u32>;
@compute @workgroup_size(4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.tw || id.y >= d.th) { return; }
  var found = 0u;
  let o = vec2i(id.xy) * ${TILE};
  for (var y = 0; y < ${TILE} && found == 0u; y += 2) {
    for (var x = 0; x < ${TILE}; x += 2) {
      let p = o + vec2i(x, y);
      if (p.x < i32(d.W) && p.y < i32(d.H) && textureLoad(gA, p, 0).a > 0.0) { found = 1u; break; }
    }
  }
  hit[id.y * d.tw + id.x] = found;
}`;
const GROW = /* wgsl */ `
struct D { W: u32, H: u32, tw: u32, th: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> hit: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileOn: array<u32>;
@compute @workgroup_size(4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.tw || id.y >= d.th) { return; }
  var found = 0u;
  for (var y = -2; y <= 2; y++) { for (var x = -2; x <= 2; x++) {
    let q = vec2i(id.xy) + vec2i(x, y);
    if (all(q >= vec2i(0)) && q.x < i32(d.tw) && q.y < i32(d.th)) { found |= hit[u32(q.y) * d.tw + u32(q.x)]; }
  } }
  tileOn[id.y * d.tw + id.x] = found;
}`;

const OUTPUT = /* wgsl */ `
struct D { W: u32, H: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var o: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var mask: texture_2d<f32>;
@compute @workgroup_size(${WG}, ${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= d.W || id.y >= d.H) { return; }
  let c = clamp(x[id.y * d.W + id.x], vec4f(0.0), vec4f(1.0));
  // the body is solid: only its soft outer edge may be see-through
  let a = max(c.a, textureLoad(mask, vec2i(id.xy), 0).a);
  textureStore(o, vec2i(id.xy), vec4f(min(c.rgb, vec3f(a)), a));
}`;

export class Network {
  constructor(device, spec, weights, tex) {
    this.device = device; this.spec = spec; this.weights = weights; this.tex = tex;
    const d = device, mod = (code) => d.createShaderModule({ code });
    const pipe = (code) => d.createComputePipeline({ layout: 'auto', compute: { module: mod(code), entryPoint: 'main' } });
    this.upMod = mod(UPCAT);
    this.inputPipe = pipe(INPUT); this.outputPipe = pipe(OUTPUT); this.tilePipe = pipe(TILES); this.growPipe = pipe(GROW);
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.pipes = new Map();
  }

  // Pipelines compile off the main path so the first frame never stalls on them.
  async pipeline(key, make) {
    if (!this.pipes.has(key)) this.pipes.set(key, await make());
    return this.pipes.get(key);
  }

  async build(W, H, gb) {
    const d = this.device;
    const buf = (bytes, extra = 0) => d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | extra });
    const uni = (arr) => {
      const b = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, new Uint32Array(arr));
      return b;
    };
    const weightsOf = (l) => {
      const b = d.createBuffer({ size: l.bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, this.weights.buffer, l.offset, l.bytes);
      return b;
    };
    const tw = Math.ceil(W / TILE), th = Math.ceil(H / TILE);
    const hit = buf(tw * th * 4), tileOn = buf(tw * th * 4), tileD = uni([W, H, tw, th]);
    this.tw = tw; this.th = th;
    this.tileBG = d.createBindGroup({ layout: this.tilePipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: tileD } }, { binding: 1, resource: gb.view.gA }, { binding: 2, resource: { buffer: hit } }] });
    this.growBG = d.createBindGroup({ layout: this.growPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: tileD } }, { binding: 1, resource: { buffer: hit } }, { binding: 2, resource: { buffer: tileOn } }] });
    const steps = [];
    const T = { input: { w: W, h: H, c4: 5, buf: buf(W * H * 5 * 16) } };
    for (const l of this.spec.layers) {
      let src;
      if (l.src.length === 1) {
        src = T[l.src[0]];
      } else {
        const a = T[l.src[0].slice(1)], b = T[l.src[1]];
        const cat = { w: b.w, h: b.h, c4: a.c4 + b.c4, buf: buf(b.w * b.h * (a.c4 + b.c4) * 16) };
        const pipe = await this.pipeline(`up${a.c4}_${b.c4}`, () => d.createComputePipelineAsync({ layout: 'auto', compute: { module: this.upMod, entryPoint: 'main', constants: { A4: a.c4, B4: b.c4 } } }));
        steps.push({ pipe, x: cat.w, y: cat.h, bg: d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: uni([a.w, a.h, b.w, b.h]) } },
          { binding: 1, resource: { buffer: a.buf } }, { binding: 2, resource: { buffer: b.buf } }, { binding: 3, resource: { buffer: cat.buf } }] }) });
        src = cat;
      }
      const w = Math.ceil(src.w / l.s), h = Math.ceil(src.h / l.s), c4 = l.outCh / 4;
      const dst = { w, h, c4, buf: buf(w * h * c4 * 16) };
      T[l.name] = dst;
      const pipe = await this.pipeline(l.name, () => d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code: convCode(l, src.c4, c4) }), entryPoint: 'main' } }));
      steps.push({ pipe, x: w, y: h, bg: d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni([src.w, src.h, w, h]) } },
        { binding: 1, resource: { buffer: src.buf } }, { binding: 2, resource: { buffer: dst.buf } },
        { binding: 3, resource: { buffer: weightsOf(l) } }, { binding: 4, resource: { buffer: tileOn } },
        { binding: 5, resource: { buffer: uni([Math.max(1, Math.round(TILE * w / W)), tw, 0, 0]) } }] }) });
    }
    this.steps = steps; this.W = W; this.H = H;
    this.output = d.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.outputView = this.output.createView();
    const dims = uni([W, H, 0, 0]);
    this.inputBG = d.createBindGroup({ layout: this.inputPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: dims } }, { binding: 1, resource: gb.view.gA }, { binding: 2, resource: gb.view.gB }, { binding: 3, resource: gb.view.gC },
      { binding: 4, resource: this.tex.a.createView() }, { binding: 5, resource: this.tex.b.createView() },
      { binding: 6, resource: this.sampler }, { binding: 7, resource: { buffer: T.input.buf } }] });
    this.outputBG = d.createBindGroup({ layout: this.outputPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: dims } }, { binding: 1, resource: { buffer: T.out.buf } },
      { binding: 2, resource: this.outputView }, { binding: 3, resource: gb.view.gA }] });
  }

  encode(enc) {
    const { W, H } = this, gw = (n) => Math.ceil(n / WG);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.tilePipe); pass.setBindGroup(0, this.tileBG);
    pass.dispatchWorkgroups(Math.ceil(this.tw / 4), Math.ceil(this.th / 4));
    pass.setPipeline(this.growPipe); pass.setBindGroup(0, this.growBG);
    pass.dispatchWorkgroups(Math.ceil(this.tw / 4), Math.ceil(this.th / 4));
    pass.setPipeline(this.inputPipe); pass.setBindGroup(0, this.inputBG);
    pass.dispatchWorkgroups(gw(W), gw(H));
    for (const s of this.steps) {
      pass.setPipeline(s.pipe); pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(gw(s.x), gw(s.y));
    }
    pass.setPipeline(this.outputPipe); pass.setBindGroup(0, this.outputBG);
    pass.dispatchWorkgroups(gw(W), gw(H));
    pass.end();
  }
}

export async function loadNetwork(device, base = './assets/') {
  const spec = await (await fetch(base + 'net.json')).json();
  const half = new Uint16Array(await (await fetch(base + 'net.bin')).arrayBuffer());
  const weights = new Float32Array(half.length);  // half precision on disk, 32-bit in the shaders
  for (let i = 0; i < half.length; i++) {
    const h = half[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    weights[i] = s * (e === 0 ? f * 2 ** -24 : (1 + f / 1024) * 2 ** (e - 15));
  }
  const load = async (n) => {
    const bmp = await createImageBitmap(await (await fetch(base + n)).blob(), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    const t = device.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    device.queue.copyExternalImageToTexture({ source: bmp }, { texture: t, premultipliedAlpha: false }, [bmp.width, bmp.height]);
    return t;
  };
  return new Network(device, spec, weights, { a: await load('neural_a.png'), b: await load('neural_b.png') });
}
