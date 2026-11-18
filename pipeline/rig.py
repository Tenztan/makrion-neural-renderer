"""Pose deformation and camera. deform() in src/gbuffer.js is the same maths: keep them in step.
Coordinates: X viewer's right, Y up, Z towards the camera. One unit is one metre."""
import numpy as np

CAM = np.array([0.0, 1.10, 1.10])  # fixed camera looking down -Z
LENS = 77.3                         # 36 mm sensor across: half width about 0.256 m at the chest
WIDTH, HEIGHT = 1280, 800
HEAD_PIVOT = np.array([0.0, 1.105, 0.0])
BODY_PIVOT = np.array([0.0, 0.60, 0.0])


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def head_weight(P):
    return sstep(1.085, 1.15, P[:, 1] + 0.25 * P[:, 2])


def rodrigues(p, rv):
    rv = np.broadcast_to(rv, p.shape)
    a = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.maximum(a, 1e-9)
    c, s = np.cos(a), np.sin(a)
    return p * c + np.cross(k, p) * s + k * np.sum(k * p, 1, keepdims=True) * (1 - c)


def apply_pose(R, pose):
    """R: rig npz. Irises turn about their own centres first, then head and body carry everything."""
    P = R["V"].copy()
    G = R["G"]
    gaze = np.array([-pose.get("gy", 0.0), pose.get("gx", 0.0), 0.0])
    for side, g in ((1, 1), (-1, 2)):
        m = G == g
        c = R["eye_c"] * [side, 1, 1]
        P[m] = rodrigues(P[m] - c, gaze) + c
    rh = np.array([pose.get("hp", 0.0), pose.get("hy", 0.0), pose.get("hr", 0.0)])
    P = rodrigues(P - HEAD_PIVOT, rh[None] * R["wHead"][:, None]) + HEAD_PIVOT
    rb = np.array([pose.get("bp", 0.0), pose.get("by", 0.0), 0.0])
    return rodrigues(P - BODY_PIVOT, rb) + BODY_PIVOT
