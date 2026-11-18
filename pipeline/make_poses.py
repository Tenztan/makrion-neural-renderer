"""Random poses shared by Cycles and the WebGPU G-buffer dump. Slightly wider than what the page uses."""
import json, os
import numpy as np

N = 200
r = np.random.default_rng(7)
tri = lambda a: float(a * (r.uniform(-1, 1) + r.uniform(-1, 1)) * 0.75)
poses = [dict(hy=tri(0.62), hp=tri(0.28), hr=tri(0.10), by=tri(0.20), bp=tri(0.05), gx=tri(0.35), gy=tri(0.25)) for _ in range(N)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
json.dump(poses, open(os.path.join(out, "poses.json"), "w"))
json.dump([dict(), dict(hy=0.5, hp=0.05, by=0.12, gx=0.2), dict(hy=-0.45, hp=-0.15, hr=0.08, by=-0.1, gx=-0.2, gy=0.15), dict(hy=0.15, hp=0.2, gy=-0.2)], open(os.path.join(out, "poses_test.json"), "w"))
print(len(poses))
