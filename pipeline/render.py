"""Cycles ground truth. blender -b --factory-startup --python-exit-code 1 -P pipeline/render.py -- --start 0 --end 200 --samples 96 --res 960"""
import bpy, json, math, os, sys, time
import numpy as np
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from rig import apply_pose, CAM, LENS, WIDTH, HEIGHT

args = sys.argv[sys.argv.index("--") + 1:]
arg = lambda k, d=None: args[args.index(k) + 1] if k in args else d
START, END = int(arg("--start", 0)), int(arg("--end", 1))
SAMPLES, RES = int(arg("--samples", 96)), int(arg("--res", 0))
DEST = arg("--dest", os.path.join(OUT, "frames"))
POSES = arg("--poses", os.path.join(OUT, "poses.json"))
os.makedirs(DEST, exist_ok=True)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.import_scene.fbx(filepath=os.path.join(ROOT, "shark", "Makrion_80k.fbx"))
ob = next(x for x in bpy.data.objects if x.type == "MESH")
me = ob.data
ob.matrix_world.identity()
ob.scale = (1, 1, 1); ob.rotation_euler = (0, 0, 0); ob.location = (0, 0, 0)
s = bpy.context.scene
R = np.load(os.path.join(OUT, "rig.npz"))
to_b = lambda p: np.c_[p[:, 0], -p[:, 2], p[:, 1]]

# ---- materials ----
tex = os.path.join(ROOT, "shark")
body = bpy.data.images.load(os.path.join(tex, "T_Body_mtl_Diffuse.1001.png"))
body.source = "TILED"
for t in (1002, 1003):
    body.tiles.new(tile_number=t)
body.reload()
eye = bpy.data.images.load(os.path.join(tex, "T_Eyes_mtl_Diffuse.1001.png"))


def principled(m):
    return next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


for m in bpy.data.materials:
    if not m.node_tree or not any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes):
        continue
    b, nt, low = principled(m), m.node_tree, m.name.lower()
    for n in list(nt.nodes):  # drop the FBX's empty normal-map node
        if n.type == "NORMAL_MAP":
            nt.nodes.remove(n)
    b.inputs["Metallic"].default_value = 0.0
    if low.startswith("cornea"):
        b.inputs["Base Color"].default_value = (1, 1, 1, 1)
        b.inputs["Roughness"].default_value = 0.02
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["IOR"].default_value = 1.34
        continue
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = eye if low.startswith("eyes") else body
    t.interpolation = "Cubic"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if low.startswith("eyes"):
        b.inputs["Roughness"].default_value = 0.25
        # the iris carries a faint glow of its own: gold eyes must read at a few pixels wide
        b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = 1.1
        continue
    # wet shark skin. Bump comes from the albedo so the G-buffer knows every wrinkle.
    b.inputs["Roughness"].default_value = 0.36
    b.inputs["Coat Weight"].default_value = 0.35
    b.inputs["Coat Roughness"].default_value = 0.22
    b.inputs["Subsurface Weight"].default_value = 0.18
    b.inputs["Subsurface Radius"].default_value = (0.012, 0.006, 0.004)
    b.inputs["Subsurface Scale"].default_value = 1.0
    bu = nt.nodes.new("ShaderNodeBump")
    bu.inputs["Distance"].default_value = 0.0012
    bu.inputs["Strength"].default_value = 0.55
    nt.links.new(t.outputs["Color"], bu.inputs["Height"])
    nt.links.new(bu.outputs[0], b.inputs["Normal"])


def area(name, loc, size, power, color):
    l = bpy.data.lights.new(name, "AREA")
    l.shape = "RECTANGLE"; l.size, l.size_y = size
    l.energy = power; l.color = color
    o = bpy.data.objects.new(name, l)
    s.collection.objects.link(o)
    tgt = Vector((0, 0.0, 1.07))
    o.location = Vector((loc[0], -loc[2], loc[1])) + tgt
    o.rotation_euler = (tgt - o.location).to_track_quat("-Z", "Y").to_euler()


# moody key from upper left, two cold rims to cut the dark back from the page, a whisper of fill
area("key", (-0.75, 0.75, 0.95), (0.9, 0.9), 17.0, (1.0, 0.95, 0.88))
area("rimR", (1.0, 0.25, -0.75), (0.3, 1.1), 42.0, (0.70, 0.86, 1.0))
area("rimL", (-1.05, 0.15, -0.70), (0.3, 1.1), 30.0, (0.70, 0.86, 1.0))
area("top", (0.1, 1.1, -0.2), (0.8, 0.5), 9.0, (0.85, 0.92, 1.0))
area("fill", (0.5, -0.65, 0.9), (1.2, 0.6), 2.2, (0.65, 0.8, 1.0))
s.world = bpy.data.worlds.new("w")
s.world.use_nodes = True
s.world.node_tree.nodes["Background"].inputs[0].default_value = (0.010, 0.016, 0.022, 1)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
s.collection.objects.link(cam); s.camera = cam
cam.location = (CAM[0], -CAM[2], CAM[1])
cam.rotation_euler = (math.pi / 2, 0, 0)
cam.data.sensor_width = 36; cam.data.sensor_fit = "HORIZONTAL"; cam.data.lens = LENS

s.render.engine = "CYCLES"
pr = bpy.context.preferences.addons["cycles"].preferences
pr.compute_device_type = "OPTIX"; pr.get_devices()
for d in pr.devices:
    d.use = d.type == "OPTIX"
s.cycles.device = "GPU"
s.cycles.samples = SAMPLES
s.cycles.use_denoising = True
s.cycles.denoiser = "OPENIMAGEDENOISE"
s.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
s.cycles.caustics_reflective = False; s.cycles.caustics_refractive = False
s.render.film_transparent = True
s.render.resolution_x, s.render.resolution_y = (WIDTH, HEIGHT) if RES == 0 else (RES, RES * HEIGHT // WIDTH)
s.render.image_settings.file_format = "PNG"
s.render.image_settings.color_mode = "RGBA"
s.view_settings.view_transform = "AgX"
s.view_settings.look = "AgX - Medium High Contrast"

poses = json.load(open(POSES))
for i in range(START, min(END, len(poses))):
    f = os.path.join(DEST, f"{i:04d}.png")
    if os.path.exists(f):
        continue
    t0 = time.time()
    me.vertices.foreach_set("co", to_b(apply_pose(R, poses[i])).ravel().astype(np.float32))
    me.update()
    s.render.filepath = f
    bpy.ops.render.render(write_still=True)
    print(f"@@ frame {i} {time.time() - t0:.1f}s", flush=True)
