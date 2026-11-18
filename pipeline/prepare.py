"""Reads the FBX, turns the irises to face forward, writes out/rig.npz. The model files are not changed.
blender -b --factory-startup --python-exit-code 1 -P pipeline/prepare.py"""
import bpy, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rig import head_weight, rodrigues

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.import_scene.fbx(filepath=os.path.join(ROOT, "shark", "Makrion_80k.fbx"))
ob = next(x for x in bpy.data.objects if x.type == "MESH")
me = ob.data
M = np.array(ob.matrix_world)
n = len(me.vertices)
co = np.zeros(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
nor = np.zeros(n * 3); me.vertex_normals.foreach_get("vector", nor)
Rn = M[:3, :3] / np.linalg.norm(M[:3, 0])
nor = nor.reshape(-1, 3) @ Rn.T
to_mine = lambda p: np.c_[p[:, 0], p[:, 2], -p[:, 1]]
V, N = to_mine(co), to_mine(nor)

me.calc_loop_triangles()
T = len(me.loop_triangles)
tv = np.zeros(T * 3, int); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
tl = np.zeros(T * 3, int); me.loop_triangles.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
tm = np.zeros(T, int); me.loop_triangles.foreach_get("material_index", tm)
uv = np.zeros(len(me.loops) * 2); me.uv_layers[0].uv.foreach_get("vector", uv); uv = uv.reshape(-1, 2)
names = [m.name.lower() for m in me.materials]
iris_mat = next(i for i, x in enumerate(names) if x.startswith("eyes"))
cornea_mat = next(i for i, x in enumerate(names) if x.startswith("cornea"))

# group per vertex: 0 body, 1 right iris, 2 left iris, 3 cornea (Cycles only)
G = np.zeros(n, np.int32)
iris_v = np.unique(tv[tm == iris_mat])
corn_v = np.unique(tv[tm == cornea_mat])
G[iris_v[V[iris_v, 0] > 0]] = 1
G[iris_v[V[iris_v, 0] < 0]] = 2
G[corn_v] = 3

r_iris = iris_v[V[iris_v, 0] > 0]
eye_c = (V[r_iris].min(0) + V[r_iris].max(0)) / 2
print("eye centre", eye_c.round(4))
# which way does the texture's iris centre face?  uv (.5,.5) is the pupil
vert_uv = np.zeros((n, 2)); vert_uv[tv.ravel()] = uv[tl.ravel()]
for side, g in ((1, 1), (-1, 2)):
    idx = np.nonzero(G == g)[0]
    c = eye_c * [side, 1, 1]
    near = idx[np.linalg.norm(vert_uv[idx] - [0.5, 0.5], axis=1) < 0.06]
    axis = (V[near] - c).mean(0)
    axis /= np.linalg.norm(axis)
    print("iris axis side", side, axis.round(3), "from", len(near), "verts")
    want = np.array([side * 0.10, 0.0, 1.0]); want /= np.linalg.norm(want)
    k = np.cross(axis, want)
    s, co_ = np.linalg.norm(k), float(axis @ want)
    rv = k / max(s, 1e-9) * np.arctan2(s, co_)
    V[idx] = rodrigues(V[idx] - c, rv) + c
    N[idx] = rodrigues(N[idx], rv)

np.savez_compressed(os.path.join(HERE, "out", "rig.npz"), V=V.astype(np.float32), N=N.astype(np.float32), tv=tv.astype(np.int32),
                    tuv=uv[tl].astype(np.float32), G=G, wHead=head_weight(V).astype(np.float32), eye_c=eye_c.astype(np.float32),
                    tm=tm.astype(np.int32), iris_mat=iris_mat, cornea_mat=cornea_mat)
print("rig verts", n, "tris", T, "bbox", V.min(0).round(3), V.max(0).round(3))
