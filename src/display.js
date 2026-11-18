// Steadies the network's output over frames, then draws it. During the intro a scan line
// hands over from a plain clay shading of the mesh to the painted picture.

const TEMPORAL = /* wgsl */ `
@group(0) @binding(0) var cur: texture_2d<f32>;
@group(0) @binding(1) var his: texture_2d<f32>;
@group(0) @binding(2) var mot: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> blend: vec4f;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(cur));
  if (i32(id.x) >= size.x || i32(id.y) >= size.y) { return; }
  let p = vec2i(id.xy);
  let c = textureLoad(cur, p, 0);
  var lo = c; var hi = c; var longest = vec2f(0.0);
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let q = clamp(p + vec2i(x, y), vec2i(0), size - 1);
    let s = textureLoad(cur, q, 0);
    lo = min(lo, s); hi = max(hi, s);
    let m = textureLoad(mot, q, 0).xy;
    if (dot(m, m) > dot(longest, longest)) { longest = m; }
  } }
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let prev = uv - longest * vec2f(0.5, -0.5);
  var h = textureSampleLevel(his, smp, prev, 0.0);
  let widen = (hi - lo) * 0.1;
  h = clamp(h, lo - widen, hi + widen);
  var a = blend.x;
  if (any(prev < vec2f(0.0)) || any(prev > vec2f(1.0))) { a = 1.0; }
  textureStore(dst, p, mix(h, c, a));
}`;

const SHOW = /* wgsl */ `
struct S { rect: vec4f, size: vec2f, scan: f32, fade: f32 };
@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var neural: texture_2d<f32>;
@group(0) @binding(2) var gA: texture_2d<f32>;
@group(0) @binding(3) var gB: texture_2d<f32>;
@group(0) @binding(4) var smp: sampler;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) v: u32) -> VO {
  let c = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
  var o: VO;
  o.pos = vec4f((s.rect.xy + c[v] * s.rect.zw) / s.size * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.uv = c[v];
  return o;
}
// Catmull-Rom, so upscaling to a large screen stays crisp instead of going soft
fn sharp(uv: vec2f) -> vec4f {
  let size = vec2f(textureDimensions(neural));
  let p = uv * size - 0.5;
  let f = fract(p);
  let b = (floor(p) + 0.5) / size;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = b - 1.0 / size; let t3 = b + 2.0 / size; let t12 = b + (w2 / w12) / size;
  var r = textureSampleLevel(neural, smp, vec2f(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureSampleLevel(neural, smp, vec2f(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureSampleLevel(neural, smp, vec2f(t12.x, t12.y), 0.0) * w12.x * w12.y;
  r += textureSampleLevel(neural, smp, vec2f(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureSampleLevel(neural, smp, vec2f(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r /= w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  let a = clamp(r.a, 0.0, 1.0);
  return vec4f(clamp(r.rgb, vec3f(0.0), vec3f(a)), a);
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let n = sharp(i.uv);
  if (s.scan >= 1.0) { return n; }
  let m = textureSampleLevel(gA, smp, i.uv, 0.0).a;
  let nor = textureSampleLevel(gB, smp, i.uv, 0.0).xyz;
  let l = max(dot(nor, normalize(vec3f(-0.75, 0.75, 0.95))), 0.0) * 0.7 + 0.1;
  let clay = vec4f(vec3f(0.62, 0.70, 0.74) * l * m, m) * s.fade;
  // the scan rises from the chest: painted below the line, clay above it
  let edge = 1.0 - s.scan;
  let k = smoothstep(edge - 0.004, edge + 0.004, i.uv.y);
  let seam = (1.0 - smoothstep(0.0, 0.0018, abs(i.uv.y - edge))) * m * s.fade;
  let col = mix(clay, n, k);
  return vec4f(col.rgb + vec3f(0.84, 0.72, 0.42) * seam, max(col.a, seam));
}`;

export class Display {
  constructor(device, ctx, format, gb, net) {
    this.device = device; this.ctx = ctx; this.gb = gb; this.net = net;
    const d = device;
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.tPipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: TEMPORAL }), entryPoint: 'main' } });
    const m = d.createShaderModule({ code: SHOW });
    this.pipe = d.createRenderPipeline({ layout: 'auto', vertex: { module: m, entryPoint: 'vs' },
      fragment: { module: m, entryPoint: 'fs', targets: [{ format,
        blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
      primitive: { topology: 'triangle-list' } });
    this.sBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.frame = 0;
  }

  // Called after Network.build, whenever the resolution changes
  bind() {
    const d = this.device, { W, H } = this.net, gb = this.gb;
    const mk = () => d.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.hisV = [mk().createView(), mk().createView()];
    this.tBG = [0, 1].map((i) => d.createBindGroup({ layout: this.tPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.net.outputView }, { binding: 1, resource: this.hisV[1 - i] },
      { binding: 2, resource: gb.view.motion }, { binding: 3, resource: this.sampler },
      { binding: 4, resource: this.hisV[i] }, { binding: 5, resource: { buffer: this.tBuf } }] }));
    this.bg = [0, 1].map((i) => d.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.sBuf } }, { binding: 1, resource: this.hisV[i] },
      { binding: 2, resource: gb.view.gA }, { binding: 3, resource: gb.view.gB }, { binding: 4, resource: this.sampler }] }));
    this.frame = 0;
  }

  encode(enc, rect, W, H, scan, fade) {
    const d = this.device, net = this.net;
    const i = this.frame++ % 2;
    d.queue.writeBuffer(this.tBuf, 0, new Float32Array([this.frame < 2 ? 1 : 0.2, 0, 0, 0]));
    let pass = enc.beginComputePass();
    pass.setPipeline(this.tPipe); pass.setBindGroup(0, this.tBG[i]);
    pass.dispatchWorkgroups(Math.ceil(net.W / 8), Math.ceil(net.H / 8));
    pass.end();
    d.queue.writeBuffer(this.sBuf, 0, new Float32Array([rect.x, rect.y, rect.w, rect.h, W, H, scan, fade]));
    pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg[i]);
    pass.draw(6);
    pass.end();
  }
}
