"""Teaches the network and its neural texture to paint the Cycles frames from G-buffers.
python pipeline/train.py --iters 10000     writes assets/net.bin, net.json, neural_a.png, neural_b.png"""
import argparse, glob, json, os, queue, random, threading, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
ASSETS = os.path.join(ROOT, "assets")
W, H = 1280, 800
TEX_W, TEX_H, TEX_CH = 2048, 512, 8
HOLDOUT = 12  # the last frames are never trained on
LEAK = 0.1    # src/nn.js uses the same activation
L_KEY = np.array([-0.75, 0.75, 0.95]) / np.linalg.norm([-0.75, 0.75, 0.95])
INPUT_CH = 12 + TEX_CH

ARCH = [
    dict(name="c1", src=["input"], ch=16, k=3, s=1),
    dict(name="d1", src=["c1"], ch=24, k=3, s=2),
    dict(name="c2", src=["d1"], ch=24, k=3, s=1),
    dict(name="d2", src=["c2"], ch=32, k=3, s=2),
    dict(name="c3", src=["d2"], ch=32, k=3, s=1),
    dict(name="c4", src=["c3"], ch=32, k=3, s=1),
    dict(name="c5", src=["^c4", "c2"], ch=24, k=3, s=1),
    dict(name="c6", src=["c5"], ch=24, k=3, s=1),
    dict(name="c7", src=["^c6", "c1"], ch=16, k=3, s=1),
    dict(name="c8", src=["c7"], ch=16, k=1, s=1),
    dict(name="out", src=["c8"], ch=4, k=1, s=1, last=True),
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # flat start: texels the camera never sees stay flat, so the PNG stays small
        self.tex = nn.Parameter(torch.full((1, TEX_CH, TEX_H, TEX_W), 0.5))
        chans = {"input": INPUT_CH}
        self.layers = nn.ModuleDict()
        for l in ARCH:
            cin = sum(chans[s.lstrip("^")] for s in l["src"])
            self.layers[l["name"]] = nn.Conv2d(cin, l["ch"], l["k"], l["s"], l["k"] // 2, padding_mode="replicate")
            chans[l["name"]] = l["ch"]
        with torch.no_grad():
            for m in self.layers.values():
                nn.init.kaiming_normal_(m.weight, a=LEAK, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
            self.layers["out"].weight *= 0.1

    def forward(self, g):
        grid = torch.stack([g[:, 8] * 2 - 1, (1 - g[:, 9]) * 2 - 1], -1)
        tex = F.grid_sample(self.tex.clamp(0, 1).expand(g.shape[0], -1, -1, -1), grid, mode="bilinear", padding_mode="border", align_corners=False) * g[:, 3:4]
        v = {"input": torch.cat([g, tex], 1)}
        for l in ARCH:
            xs = [F.interpolate(v[s[1:]], scale_factor=2, mode="bilinear", align_corners=False) if s.startswith("^") else v[s] for s in l["src"]]
            x = self.layers[l["name"]](torch.cat(xs, 1))
            v[l["name"]] = x if l.get("last") else F.leaky_relu(x, LEAK)
        return v["out"]


def read_gbuf(path, y0=0, x0=0, h=H, w=W):
    """Returns 12,h,w: albedo+mask, normal+depth, u, v, N.L, is-iris. Same as src/gbuffer.js writes."""
    mm = np.memmap(path, np.uint8, "r")
    def part(off, bpp, dtype, ch):
        rows = [mm[off + (y * W + x0) * bpp: off + (y * W + x0) * bpp + w * bpp] for y in range(y0, y0 + h)]
        return np.frombuffer(np.concatenate(rows).tobytes(), dtype).reshape(h, w, ch).astype(np.float32)
    a = part(0, 4, np.uint8, 4) / 255.0
    b = part(4 * W * H, 8, np.float16, 4)
    c = part(12 * W * H, 4, np.float16, 2)
    ndl = np.maximum((b[..., :3] * L_KEY.astype(np.float32)).sum(-1), 0) * a[..., 3]
    iris = (c[..., 0] >= 0.75).astype(np.float32) * a[..., 3]
    out = np.concatenate([a, b, c, ndl[..., None], iris[..., None]], -1)
    return out.transpose(2, 0, 1)


def load_target(i):
    a = np.asarray(Image.open(os.path.join(OUT, "frames", f"{i:04d}.png")).convert("RGBA"), np.float32) / 255.0
    a = np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1)  # premultiplied in display space
    return (a * 255 + 0.5).astype(np.uint8)


class Data:
    def __init__(self, ids, crop, batch, eyes=False):
        self.ids, self.crop, self.batch, self.eyes = ids, crop, batch, eyes
        t0 = time.time()
        self.targets = {i: load_target(i) for i in ids}
        print("targets loaded", len(ids), round(time.time() - t0), "s", flush=True)
        self.q = queue.Queue(10)
        for _ in range(4):
            threading.Thread(target=self.worker, daemon=True).start()

    def sample(self):
        c = self.crop
        for _ in range(12):
            i = random.choice(self.ids)
            if self.eyes and random.random() < 0.4:  # enough face crops for the eyes, enough body crops to keep it right
                x = int(np.clip(random.gauss(W / 2, 70) - c / 2, 0, W - c)); y = random.randint(0, 120)
            elif random.random() < 0.5:  # half of all crops go to the head: people look at the face
                x = int(np.clip(random.gauss(W / 2, 90) - c / 2, 0, W - c)); y = random.randint(0, 200)
            else:
                x, y = random.randint(0, W - c), random.randint(0, H - c)
            t = self.targets[i][y:y + c, x:x + c]
            if t[..., 3].mean() > 50 or random.random() < 0.05:
                break
        g = read_gbuf(os.path.join(OUT, "gbuf", f"{i:04d}.bin"), y, x, c, c)
        return g, t.astype(np.float32).transpose(2, 0, 1) / 255.0

    def worker(self):
        while True:
            items = [self.sample() for _ in range(self.batch)]
            self.q.put((np.stack([a for a, _ in items]), np.stack([b for _, b in items])))

    def next(self, dev):
        g, t = self.q.get()
        return torch.from_numpy(g).to(dev), torch.from_numpy(t).to(dev)


class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in self.f.parameters():
            p.requires_grad_(False)
        self.register_buffer("m", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("s", torch.tensor([.229, .224, .225])[None, :, None, None])

    def feats(self, x):
        x = (x - self.m) / self.s
        out = []
        for i, l in enumerate(self.f):
            x = l(x)
            if i in (3, 8, 15):
                out.append(x)
        return out

    def forward(self, a, b):
        return sum(F.l1_loss(x, y) for x, y in zip(self.feats(a), self.feats(b)))


def export(net, frames):
    """Weights as mat4 blocks [out/4][tap][in/4], then biases. Offsets are for floats; the file is half precision."""
    blobs, layers, off = [], [], 0
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    for l in ARCH:
        Wt, B = sd[f"layers.{l['name']}.weight"], sd[f"layers.{l['name']}.bias"]
        co, ci, kh, kw = Wt.shape
        mats = np.zeros((co // 4, kh, kw, ci // 4, 16), np.float32)
        for o4 in range(co // 4):
            for i4 in range(ci // 4):
                for j in range(4):
                    for i in range(4):
                        mats[o4, :, :, i4, j * 4 + i] = Wt[o4 * 4 + i, i4 * 4 + j]
        b = mats.tobytes() + B.astype(np.float32).tobytes()
        layers.append(dict(l, inCh=ci, outCh=co, offset=off, bytes=len(b), biasOffset=off + mats.size * 4))
        blobs.append(b); off += len(b)
    with open(os.path.join(ASSETS, "net.bin"), "wb") as f:
        f.write(np.frombuffer(b"".join(blobs), np.float32).astype(np.float16).tobytes())
    tex = (net.tex.detach().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(tex[..., :4], "RGBA").save(os.path.join(ASSETS, "neural_a.png"), optimize=True)
    Image.fromarray(tex[..., 4:8], "RGBA").save(os.path.join(ASSETS, "neural_b.png"), optimize=True)
    json.dump(dict(inputCh=INPUT_CH, texW=TEX_W, texH=TEX_H, frames=frames, layers=layers), open(os.path.join(ASSETS, "net.json"), "w"), indent=1)


def evaluate(net, ids, dev, path):
    net.eval(); ps, sheet = [], []
    with torch.no_grad():
        for n, i in enumerate(ids):
            g = torch.from_numpy(read_gbuf(os.path.join(OUT, "gbuf", f"{i:04d}.bin")))[None].to(dev)
            t = load_target(i).astype(np.float32).transpose(2, 0, 1)[None] / 255
            with torch.autocast("cuda", dtype=torch.float16):
                p = net(g).float().clamp(0, 1).cpu().numpy()
            ps.append(-10 * np.log10(np.mean((p[:, :3] - t[:, :3]) ** 2) + 1e-9))
            if n < 2:  # network | Cycles, on black
                sheet.append(np.concatenate([p[0, :3], t[0, :3]], 2))
    Image.fromarray((np.concatenate(sheet, 1).transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)
    net.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--export-every", type=int, default=2500)
    ap.add_argument("--resume", default="", help="continue from a saved net.pt with a gentler learning rate")
    ap.add_argument("--eye", type=float, default=0.0, help="extra loss weight on the eyes: a few pixels nothing else would notice")
    a = ap.parse_args()
    dev = "cuda"
    have = sorted(int(os.path.basename(p)[:4]) for p in glob.glob(os.path.join(OUT, "gbuf", "*.bin")) if os.path.exists(os.path.join(OUT, "frames", os.path.basename(p)[:4] + ".png")))
    hold, ids = have[-HOLDOUT:], have[:-HOLDOUT]
    print("train", len(ids), "holdout", len(hold), flush=True)
    data = Data(ids, a.crop, a.batch, eyes=a.eye > 0)
    net = Net().to(dev)
    print("weights", sum(p.numel() for k, p in net.named_parameters() if k != "tex"), flush=True)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev))
    vgg = Perceptual().to(dev)
    opt = torch.optim.AdamW([dict(params=[p for k, p in net.named_parameters() if k != "tex"], lr=1e-3, weight_decay=1e-5), dict(params=[net.tex], lr=4e-3, weight_decay=0.0)])
    peak = 0.2 if a.resume else 1.0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1.5e-3 * peak, 6e-3 * peak], total_steps=a.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    for it in range(1, a.iters + 1):
        g, t = data.next(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(g); pc = p.clamp(0, 1)
            l1 = F.l1_loss(p, t)
            lg = F.l1_loss(pc[..., 1:, :] - pc[..., :-1, :], t[..., 1:, :] - t[..., :-1, :]) + F.l1_loss(pc[..., 1:] - pc[..., :-1], t[..., 1:] - t[..., :-1])
            lp = vgg(pc[:, :3], t[:, :3])
            lc = F.l1_loss(pc[:, 0] - pc[:, 1], t[:, 0] - t[:, 1]) + F.l1_loss(pc[:, 2] - pc[:, 1], t[:, 2] - t[:, 1])
            loss = l1 + 0.5 * lg + 0.06 * lp + 2.0 * lc  # colour differences need their own term or they come out grey
            if a.eye > 0:
                iris = F.max_pool2d(g[:, 11:12], 5, 1, 2)  # the is-iris channel, grown a little over the cornea
                loss = loss + a.eye * ((p[:, :3] - t[:, :3]).abs() * iris).sum() / (3 * iris.sum() + 1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} l1 {l1.item():.4f} {(time.time() - t0) / it * 1000:.0f} ms/it", flush=True)
        if it % a.export_every == 0 or it == a.iters:
            print(f"== it {it} holdout PSNR {evaluate(net, hold, dev, os.path.join(OUT, f'eval_{it:06d}.png')):.2f} dB", flush=True)
            torch.save(net.state_dict(), os.path.join(OUT, "net.pt"))
            export(net, len(ids))
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
