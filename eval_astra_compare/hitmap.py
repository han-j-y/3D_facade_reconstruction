"""Blender: object-hit map through the front camera. blender -b scene.blend -P hitmap.py -- out.json W H GW mode
mode 'astra': camera named *front* (else the render_front.py construction); 'ours': the scene camera.
Writes {"names": [...], "colls": [[...]], "grid": GW x GH ints (-1 = no hit), "size": [W, H]}."""
import bpy, json, math, sys
from mathutils import Vector
out, W, H, GW, mode = sys.argv[sys.argv.index('--') + 1:][:5]
W, H, GW = int(W), int(H), int(GW); GH = max(1, round(GW * H / W))
sc = bpy.context.scene
if mode == 'ours':
    cam = sc.camera
else:
    cams = [o for o in sc.objects if o.type == 'CAMERA']
    cam = next((c for c in cams if 'front' in c.name.lower()), None)
    if cam is None:
        big = {o.name for o in sc.objects if o.type == 'MESH' and max(o.dimensions) > 60}
        pts = [o.matrix_world @ Vector(c) for o in sc.objects if o.type == 'MESH' and o.name not in big for c in o.bound_box]
        xs = [p.x for p in pts]; zs = [p.z for p in pts]
        x0, x1, z0, z1 = min(xs), max(xs), max(min(zs), 0), max(zs)
        cd = bpy.data.cameras.new('Front elevation'); cd.type = 'ORTHO'
        cam = bpy.data.objects.new('Front elevation', cd); sc.collection.objects.link(cam)
        cam.location = ((x0 + x1) / 2, -60, (z0 + z1) / 2); cam.rotation_euler = (math.radians(90), 0, 0)
        cd.ortho_scale = max((x1 - x0) * 1.08, (z1 - z0) * 1.08)
        bpy.context.view_layer.update()
sc.render.resolution_x, sc.render.resolution_y = W, H
mw = cam.matrix_world
tr, br, bl, tl = [mw @ v for v in cam.data.view_frame(scene=sc)]
d = (mw.to_quaternion() @ Vector((0, 0, -1))).normalized()
persp = cam.data.type != 'ORTHO'
dg = bpy.context.evaluated_depsgraph_get()
idx = {}; names = []; colls = []; grid = []
for j in range(GH):
    row = []
    v = (j + 0.5) / GH
    for i in range(GW):
        u = (i + 0.5) / GW
        p = tl.lerp(tr, u).lerp(bl.lerp(br, u), v)
        if persp:
            o = mw.translation; dd = (p - o).normalized()
        else:
            o = p - d * 200.0; dd = d
        hit, loc, nrm, fi, ob, m = sc.ray_cast(dg, o, dd)
        if not hit or ob is None:
            row.append(-1); continue
        if ob.name not in idx:
            idx[ob.name] = len(names); names.append(ob.name); colls.append([c.name for c in ob.users_collection])
        row.append(idx[ob.name])
    grid.append(row)
json.dump({'names': names, 'colls': colls, 'grid': grid, 'size': [W, H]}, open(out, 'w'))
print('hitmap', GW, GH, len(names))
