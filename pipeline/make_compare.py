"""Copies the never-trained Cycles frames for compare.html, and makes the fallback poster."""
import glob, json, os, re
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
DEST = os.path.join(ROOT, "assets", "compare")
os.makedirs(DEST, exist_ok=True)
HOLDOUT = 12
poses = json.load(open(os.path.join(OUT, "poses.json")))
have = sorted(int(os.path.basename(p)[:4]) for p in glob.glob(os.path.join(OUT, "frames", "*.png")))
frames = []
for i in have[-HOLDOUT:]:
    im = Image.open(os.path.join(OUT, "frames", f"{i:04d}.png")).convert("RGBA")
    name = f"{i:04d}.webp"
    im.save(os.path.join(DEST, name), quality=95, method=4)
    frames.append(dict(file=name, pose=poses[i]))
times = [float(x) for x in re.findall(r"@@ frame \d+ ([\d.]+)s", open(os.path.join(OUT, "render.log"), errors="ignore").read())]
json.dump(dict(frames=frames, cyclesSeconds=round(sorted(times)[len(times) // 2]) if times else 5), open(os.path.join(DEST, "index.json"), "w"), indent=1)
bg = Image.new("RGBA", im.size, (5, 9, 12, 255))
bg.alpha_composite(Image.open(os.path.join(OUT, "frames", f"{have[0]:04d}.png")).convert("RGBA"))
bg.convert("RGB").save(os.path.join(ROOT, "assets", "poster.webp"), quality=86, method=4)
print("compare frames", len(frames), "median Cycles s", times and sorted(times)[len(times) // 2])
