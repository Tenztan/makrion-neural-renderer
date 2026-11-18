"""Does the WebGPU mask land exactly on the Cycles silhouette? Prints overlap for a few frames."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train import read_gbuf, load_target, OUT
for f in [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 10]:
    g = read_gbuf(os.path.join(OUT, "gbuf", f"{f:04d}.bin"))
    a = load_target(f)[..., 3] > 127
    m = g[3] > 0.5
    print(f, "IoU", round((a & m).sum() / (a | m).sum(), 4), "mesh outside alpha", int((m & ~a).sum()), "alpha outside mesh", int((a & ~m).sum()))
