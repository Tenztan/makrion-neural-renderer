"""Packs out/rig.npz and the textures for the browser: assets/makrion.bin + .json + albedo.webp.
Run with the project venv. The cornea stays out: only Cycles needs it, the network learns its highlights."""
import json, os, sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)
sys.path.insert(0, HERE)
R = np.load(os.path.join(HERE, "out", "rig.npz"))
V, N, G, W, tv, tuv, tm = R["V"], R["N"], R["G"], R["wHead"], R["tv"], R["tuv"], R["tm"]
keep = tm != int(R["cornea_mat"])
tv, tuv, tm = tv[keep], tuv[keep], tm[keep]
iris = tm == int(R["iris_mat"])
# atlas: three body tiles side by side, then the eye. u runs 0..1 across all four.
auv = tuv.copy()
auv[iris, 0] += 3.0
auv[..., 0] /= 4.0
key = np.c_[tv.ravel(), np.round(auv.reshape(-1, 2) * 65535).astype(np.int64)]
uniq, inv = np.unique(key, axis=0, return_inverse=True)
vid = uniq[:, 0]
print("runtime verts", len(uniq), "tris", len(tv))
POS_LO, POS_RANGE = -0.5, 2.0
dt = np.dtype([("p", "<u2", 3), ("wh", "<u2"), ("uv", "<u2", 2), ("n", "i1", 3), ("g", "u1")])
out = np.zeros(len(uniq), dt)
out["p"] = np.round((V[vid] - POS_LO) / POS_RANGE * 65535)
out["wh"] = np.round(W[vid] * 65535)
out["uv"] = uniq[:, 1:]
out["n"] = np.clip(np.round(N[vid] / np.linalg.norm(N[vid], axis=1, keepdims=True) * 127), -127, 127)
out["g"] = G[vid]
index = inv.ravel().astype(np.uint32)
vb, ib = out.tobytes(), index.tobytes()
with open(os.path.join(ASSETS, "makrion.bin"), "wb") as f:
    f.write(vb + b"\0" * ((-len(vb)) % 16) + ib)
json.dump(dict(vertexCount=len(uniq), indexCount=len(index), triCount=len(tv), indexOffset=len(vb) + (-len(vb)) % 16, posLo=POS_LO, posRange=POS_RANGE,
               eyeC=R["eye_c"].tolist()), open(os.path.join(ASSETS, "makrion.json"), "w"), indent=1)
tiles = ["T_Body_mtl_Diffuse.1001.png", "T_Body_mtl_Diffuse.1002.png", "T_Body_mtl_Diffuse.1003.png", "T_Eyes_mtl_Diffuse.1001.png"]
atlas = Image.new("RGB", (8192, 2048))
for i, t in enumerate(tiles):
    atlas.paste(Image.open(os.path.join(ROOT, "shark", t)).convert("RGB").resize((2048, 2048), Image.LANCZOS), (i * 2048, 0))
atlas.save(os.path.join(ASSETS, "albedo.webp"), quality=90, method=4)
print("bin MB", round(os.path.getsize(os.path.join(ASSETS, "makrion.bin")) / 1e6, 2), "albedo MB", round(os.path.getsize(os.path.join(ASSETS, "albedo.webp")) / 1e6, 2))
