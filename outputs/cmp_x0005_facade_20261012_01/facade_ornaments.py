"""Editable geometric ornament inferred only from the supplied facade photograph.

All public helpers return a list of newly linked Blender objects. Z is up;
negative Y faces the street. Importing this module does not change the scene.
"""

import math
import bpy


def _curve_bundle(name, lines, radius, material):
    if not lines:
        return None
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 12
    data.bevel_depth = radius
    data.bevel_resolution = 2
    data.use_fill_caps = True
    for coords in lines:
        if len(coords) < 2:
            continue
        spline = data.splines.new('POLY')
        spline.points.add(len(coords) - 1)
        for p, co in zip(spline.points, coords):
            p.co = (co[0], co[1], co[2], 1.0)
    if material is not None:
        data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def _mesh(name, verts, faces, material, smooth=False):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    if material is not None:
        data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    if smooth:
        for poly in data.polygons:
            poly.use_smooth = True
    return obj


def _path_sampler(path):
    pts = [(float(p[0]), float(p[1])) for p in path]
    clean = []
    for p in pts:
        if not clean or math.dist(clean[-1], p) > 1.0e-6:
            clean.append(p)
    if len(clean) < 2:
        raise ValueError('A railing path needs at least two distinct XY points.')
    distances = [0.0]
    for a, b in zip(clean[:-1], clean[1:]):
        distances.append(distances[-1] + math.dist(a, b))
    total = distances[-1]

    def sample(s):
        s = min(max(float(s), 0.0), total)
        i = 0
        while i < len(clean) - 2 and distances[i + 1] < s:
            i += 1
        a, b = clean[i], clean[i + 1]
        t = (s - distances[i]) / (distances[i + 1] - distances[i])
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    return sample, total


def wrought_railing(name, path, z, height, material, ornate=True):
    """Create a metal guard on an open XY path, between z and z + height.

    Bars are spaced at approximately 0.16 units (intended meters). Curves are
    bundled into a few editable objects, even for the ornate balcony variant.
    Returns [rail_object, bar_object, optional_scroll_object].
    """
    if height <= 0:
        raise ValueError('Railing height must be positive.')
    sample, length = _path_sampler(path)

    def xyz(s, h, depth=0.0):
        x, y = sample(s)
        return (x, y - depth, z + h)

    divisions = max(12, int(length / 0.045))
    samples = [length * i / divisions for i in range(divisions + 1)]
    rail_lines = [[xyz(s, h) for s in samples]
                  for h in (0.045 * height, 0.15 * height, 0.94 * height, height)]
    rails = _curve_bundle(name + '_Handrails', rail_lines, 0.024, material)
    bar_count = max(2, round(length / 0.16))
    bars = [[xyz(length * i / bar_count, 0.05 * height),
             xyz(length * i / bar_count, 0.98 * height)]
            for i in range(bar_count + 1)]
    bar_obj = _curve_bundle(name + '_Uprights', bars, 0.012, material)
    result = [rails, bar_obj]
    if not ornate:
        return result

    curls = []
    panels = max(1, round(length / 0.55))
    panel_w = length / panels
    # Paired waves tie the ornamental upper and lower belts together.
    for base, amplitude in ((0.24, 0.035), (0.29, 0.035),
                            (0.76, 0.055), (0.82, 0.045)):
        curls.append([xyz(s, height * (base + amplitude *
                       math.sin(s / length * panels * math.tau)), 0.01)
                      for s in samples])

    for panel in range(panels):
        c = (panel + 0.5) * panel_w
        half = panel_w * 0.44
        # Two mirrored rising S stems suggest wrought botanical stalks.
        for sign in (-1.0, 1.0):
            curls.append([xyz(c + sign * half * math.sin(math.pi * t) *
                                  (0.75 - 0.4 * t),
                              height * (0.17 + 0.67 * t), 0.018)
                          for t in [i / 35 for i in range(36)]])
            # Inward rolled C terminals at two heights.
            for level, direction in ((0.38, 1.0), (0.70, -1.0)):
                cc = c + sign * panel_w * 0.22
                coords = []
                for i in range(43):
                    t = i / 42
                    angle = -math.pi * 0.6 + direction * t * math.pi * 2.35
                    radius = 1.0 - 0.84 * t
                    coords.append(xyz(cc + sign * panel_w * 0.19 * radius *
                                      math.cos(angle),
                                      height * (level + 0.14 * radius *
                                                math.sin(angle)), 0.025))
                curls.append(coords)
        # Narrow pointed oval, akin to the photograph's ironwork leaves.
        curls.append([xyz(c + panel_w * 0.085 * math.sin(a),
                          height * (0.54 + 0.18 * math.cos(a)), 0.03)
                      for a in [math.tau * i / 32 for i in range(33)]])
    result.append(_curve_bundle(name + '_BotanicalScrolls', curls, 0.011, material))
    return result


def floral_relief(name, x, y, z, width, height, material):
    """Create limestone swags and acanthus-like curls centered on (x, y, z).

    Width/height bound the design in X/Z. The ornament projects toward -Y.
    Returns editable curve objects and a single aggregated leaf mesh.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Relief dimensions must be positive.')
    curves = []
    fine = []
    leaf_verts, leaf_faces = [], []
    scale = min(width, height * 1.8)

    def p(u, v, depth=0.04):
        return (x + width * u, y - scale * depth, z + height * v)

    # Three draped cords with a central dip and gently lifted ends.
    for j in range(3):
        curves.append([p(-0.45 + 0.90 * t,
                         0.28 - 0.45 * math.sin(math.pi * t) - j * 0.09,
                         0.045 + 0.018 * math.sin(math.pi * t))
                       for t in [i / 40 for i in range(41)]])

    # Rolled acanthus tendrils in the outer shoulders of each panel.
    for side in (-1.0, 1.0):
        for off in (0.0, 0.045):
            coords = []
            for i in range(55):
                t = i / 54
                a = math.pi * 0.1 + t * math.pi * 2.55
                r = (1.0 - 0.87 * t)
                coords.append(p(side * (0.31 + (0.14 - off) * r * math.cos(a)),
                                0.11 + (0.28 - off) * r * math.sin(a),
                                0.06 + 0.017 * (1 - t)))
            (curves if off == 0 else fine).append(coords)

    def leaf(u0, v0, u1, v1, breadth, depth=0.065):
        # A pointed leaf with a raised median ridge; all leaves share one mesh.
        du, dv = (u1 - u0) * width, (v1 - v0) * height
        norm = max(math.hypot(du, dv), 1.0e-6)
        nx, nz = -dv / norm, du / norm
        base = len(leaf_verts)
        rows = 9
        for i in range(rows):
            t = i / (rows - 1)
            bulge = math.sin(math.pi * t) ** 0.85
            for cross in (-1, 0, 1):
                side = breadth * width * bulge * cross
                px = x + width * (u0 + (u1 - u0) * t) + nx * side
                pz = z + height * (v0 + (v1 - v0) * t) + nz * side
                py = y - scale * (depth + (0.052 if cross == 0 else 0.014) * bulge)
                leaf_verts.append((px, py, pz))
        for i in range(rows - 1):
            for j in range(2):
                a = base + i * 3 + j
                leaf_faces.append((a, a + 3, a + 4, a + 1))
        fine.append([p(u0 + (u1 - u0) * t, v0 + (v1 - v0) * t,
                       depth + 0.055 * math.sin(math.pi * t))
                     for t in [i / 12 for i in range(13)]])

    # Fan of carved leaves, with asymmetrical outer leaves like hand carving.
    leaf(0, -0.16, 0, -0.49, 0.054, 0.078)
    for side in (-1.0, 1.0):
        leaf(side * 0.018, -0.12, side * 0.17, -0.40, 0.047)
        leaf(side * 0.12, -0.15, side * 0.30, -0.30, 0.041)
        leaf(side * 0.34, 0.16, side * 0.47, 0.37, 0.035)
        leaf(side * 0.27, 0.07, side * 0.20, 0.38, 0.032)
    # Small central oval bud crowns the fan.
    curves.append([p(0.045 * math.cos(a), -0.105 + 0.065 * math.sin(a), 0.09)
                   for a in [math.tau * i / 28 for i in range(29)]])
    result = [_curve_bundle(name + '_CarvedSwags', curves, scale * 0.026, material),
              _curve_bundle(name + '_FineRibs', fine, scale * 0.012, material),
              _mesh(name + '_AcanthusLeaves', leaf_verts, leaf_faces, material, True)]
    return result


def _rectangular_path_rail(name, path, bottom, height, width, material):
    sample, length = _path_sampler(path)
    # Sampling keeps broad, curved terrace copings continuous.
    count = max(len(path), int(length / 0.10) + 1)
    verts, faces = [], []
    for i in range(count):
        s = length * i / (count - 1)
        px, py = sample(s)
        a = sample(max(0, s - 0.015))
        b = sample(min(length, s + 0.015))
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = max(math.hypot(dx, dy), 1.0e-6)
        nx, ny = -dy / norm, dx / norm
        for side, zz in ((-1, bottom), (1, bottom),
                         (1, bottom + height), (-1, bottom + height)):
            verts.append((px + side * nx * width / 2,
                          py + side * ny * width / 2, zz))
    for i in range(count - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    faces.extend([(3, 2, 1, 0), tuple((count - 1) * 4 + j for j in range(4))])
    obj = _mesh(name, verts, faces, material)
    bevel = obj.modifiers.new('Soft stone edges', 'BEVEL')
    bevel.width = 0.012
    bevel.segments = 2
    bevel.limit_method = 'ANGLE'
    return obj


def stone_balustrade(name, path, z, height, material):
    """Create continuous stone rails and turned balusters along an open XY path.

    z is the bottom of the lower rail; z + height is the coping top. Balusters
    occur at approximately 0.23-unit spacing. Returns three editable objects.
    """
    if height <= 0.20:
        raise ValueError('Stone balustrade needs a height greater than 0.20.')
    sample, length = _path_sampler(path)
    base_h = min(0.14, height * 0.17)
    cap_h = min(0.13, height * 0.16)
    bottom = _rectangular_path_rail(name + '_LowerRail', path, z, base_h, 0.26, material)
    top = _rectangular_path_rail(name + '_Coping', path, z + height - cap_h,
                                cap_h, 0.31, material)
    body_h = height - base_h - cap_h
    # Deliberately swollen turned profiles, seen in the lower terrace guards.
    profile = [(0.00, .075), (.07, .075), (.10, .055), (.17, .044),
               (.24, .044), (.31, .067), (.40, .082), (.49, .079),
               (.57, .065), (.67, .039), (.77, .034), (.83, .045),
               (.91, .045), (.94, .071), (1.00, .071)]
    verts, faces = [], []
    count = max(1, round(length / 0.23))
    sides = 12
    for j in range(count + 1):
        px, py = sample(length * j / count)
        base = len(verts)
        for h, radius in profile:
            for k in range(sides):
                a = math.tau * k / sides
                verts.append((px + radius * math.cos(a), py + radius * math.sin(a),
                              z + base_h + body_h * h))
        for ring in range(len(profile) - 1):
            for k in range(sides):
                a = base + ring * sides + k
                b = base + ring * sides + (k + 1) % sides
                faces.append((a, b, b + sides, a + sides))
        faces.append(tuple(base + k for k in reversed(range(sides))))
        final = base + (len(profile) - 1) * sides
        faces.append(tuple(final + k for k in range(sides)))
    balusters = _mesh(name + '_TurnedBalusters', verts, faces, material, True)
    return [bottom, top, balusters]
