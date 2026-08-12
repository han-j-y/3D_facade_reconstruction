"""Execute window grammar program operations."""

from __future__ import annotations

import copy
import math
from typing import Any

from context import Anchor, CompileContext
from geom2d import Region, Segment, Vec2, normalized_point
from values import resolve_normalized_coord, resolve_scalar


def execute_program(ctx: CompileContext, program: list[dict[str, Any]]) -> None:
    for step in program:
        op = step.get("op")
        if op is None:
            raise ValueError(f"program step missing op: {step}")
        handler = _OP_HANDLERS.get(op)
        if handler is None:
            raise ValueError(f"unsupported op: {op!r}")
        handler(ctx, step)


def _region_for_step(ctx: CompileContext, step: dict[str, Any]) -> Region:
    region = ctx.region(step["region"])
    if "bounds" not in step and "crop" not in step:
        return region
    u0 = float((step.get("bounds") or step.get("crop") or {}).get("u0", 0.0))
    v0 = float((step.get("bounds") or step.get("crop") or {}).get("v0", 0.0))
    u1 = float((step.get("bounds") or step.get("crop") or {}).get("u1", 1.0))
    v1 = float((step.get("bounds") or step.get("crop") or {}).get("v1", 1.0))
    zone_id = step.get("zone") or f"_{step.get('id', 'zone')}_bounds"
    if zone_id not in ctx.regions:
        ctx.regions[zone_id] = region.crop_normalized(u0, v0, u1, v1, zone_id)
    return ctx.regions[zone_id]


def _op_define_region(ctx: CompileContext, step: dict[str, Any]) -> None:
    rid = step["id"]
    source = ctx.region(step["from"])
    op = step.get("operation", "inset")

    if op == "crop" or "bounds" in step or "crop" in step:
        bounds = step.get("bounds") or step.get("crop") or {}
        u0 = float(bounds.get("u0", 0.0))
        v0 = float(bounds.get("v0", 0.0))
        u1 = float(bounds.get("u1", 1.0))
        v1 = float(bounds.get("v1", 1.0))
        ctx.regions[rid] = source.crop_normalized(u0, v0, u1, v1, rid)
        return

    amount = resolve_scalar(step.get("amount", 0.0), source)
    ctx.regions[rid] = source.inset(amount, rid)


def _op_inset(ctx: CompileContext, step: dict[str, Any]) -> None:
    _op_define_region(ctx, step)


def _op_split_vertical(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = _region_for_step(ctx, step)
    x = resolve_normalized_coord(step["x"], axis="x", region=region)
    seg = Segment(
        id=step.get("id", "v_split"),
        a=Vec2(x, region.min_y),
        b=Vec2(x, region.max_y),
    )
    ctx.add_segment(seg, region_id=region.id)


def _op_split_horizontal(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = _region_for_step(ctx, step)
    y = resolve_normalized_coord(step["y"], axis="y", region=region)
    seg = Segment(
        id=step.get("id", "h_split"),
        a=Vec2(region.min_x, y),
        b=Vec2(region.max_x, y),
    )
    ctx.add_segment(seg, region_id=region.id)


def _op_add_grid(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = _region_for_step(ctx, step)
    n_v = int(step.get("vertical", step.get("columns", 1)))
    n_h = int(step.get("horizontal", step.get("rows", 1)))

    for i in range(1, n_v + 1):
        u = i / (n_v + 1)
        x = region.min_x + u * region.width
        ctx.add_segment(
            Segment(id=f"{step.get('id', 'grid')}_v{i}", a=Vec2(x, region.min_y), b=Vec2(x, region.max_y)),
            region_id=region.id,
        )
    for j in range(1, n_h + 1):
        v = j / (n_h + 1)
        y = region.min_y + v * region.height
        ctx.add_segment(
            Segment(id=f"{step.get('id', 'grid')}_h{j}", a=Vec2(region.min_x, y), b=Vec2(region.max_x, y)),
            region_id=region.id,
        )


def _op_split_radial(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = ctx.region(step["region"])
    origin = normalized_point(region, float(step.get("ox", 0.5)), float(step.get("oy", 0.5)))
    start_r = _radial_start_radius(ctx, step, region)
    angles = _radial_angles(step)
    span = max(region.width, region.height) * 2.0
    rid = step.get("id", "rad")
    for i, ang in enumerate(angles):
        dx, dy = math.cos(ang), math.sin(ang)
        start = Vec2(origin.x + dx * start_r, origin.y + dy * start_r)
        far = Vec2(origin.x + dx * span, origin.y + dy * span)
        ctx.add_segment(Segment(id=f"{rid}_{i}", a=start, b=far), region_id=region.id)


def _radial_start_radius(ctx: CompileContext, step: dict[str, Any], region: Region) -> float:
    if "from_arc" in step:
        spec = ctx.arc_muntins.get(str(step["from_arc"]))
        if spec is None:
            raise ValueError(f"unknown arc muntin: {step['from_arc']!r}")
        if spec["region_id"] != region.id:
            raise ValueError(f"arc {step['from_arc']!r} is not in region {region.id!r}")
        step.setdefault("ox", spec["ox"])
        step.setdefault("oy", spec["oy"])
        return float(spec["radius"])
    if "start_radius" in step:
        return resolve_scalar(step["start_radius"], region)
    if "offset" in step:
        return resolve_scalar(step["offset"], region)
    return 0.0


def _radial_angles(step: dict[str, Any]) -> list[float]:
    if "angles" in step:
        raw = step["angles"]
        if not isinstance(raw, list):
            raise ValueError("split_radial angles must be a list")
        return [math.radians(float(a)) for a in raw]
    if "arc_start" in step and "arc_end" in step:
        count = int(step.get("count", 1))
        a0 = math.radians(float(step["arc_start"]))
        a1 = math.radians(float(step["arc_end"]))
        if count <= 1:
            return [(a0 + a1) / 2.0]
        return [a0 + (a1 - a0) * i / (count - 1) for i in range(count)]
    count = int(step.get("count", 4))
    offset = float(step.get("angle_offset", 0.0))
    return [2.0 * math.pi * i / count + offset for i in range(count)]


def _op_define_anchor(ctx: CompileContext, step: dict[str, Any]) -> None:
    aid = step["id"]
    kind = step.get("kind", "point")
    region = ctx.region(step["region"]) if "region" in step else None

    if kind == "center" and region is not None:
        pos = region.center
    elif kind == "point" and region is not None:
        pos = normalized_point(region, float(step.get("u", 0.5)), float(step.get("v", 0.5)))
    elif "at" in step:
        at = step["at"]
        pos = Vec2(float(at[0]), float(at[1]))
    else:
        raise ValueError(f"cannot resolve anchor: {step}")

    ctx.add_anchor(Anchor(id=aid, position=pos, kind=kind))


def _resolve_anchor(ctx: CompileContext, ref: str | dict[str, Any]) -> Vec2:
    if isinstance(ref, str):
        return ctx.anchors[ref].position
    if isinstance(ref, dict):
        if "anchor" in ref:
            return ctx.anchors[ref["anchor"]].position
        if "at" in ref:
            at = ref["at"]
            return Vec2(float(at[0]), float(at[1]))
    raise ValueError(f"bad anchor ref: {ref!r}")


def _op_connect(ctx: CompileContext, step: dict[str, Any]) -> None:
    a = _resolve_anchor(ctx, step["from"])
    b = _resolve_anchor(ctx, step["to"])
    seg = Segment(id=step.get("id", "connect"), a=a, b=b)
    region_id = step.get("region")
    ctx.add_segment(seg, region_id=region_id)


def _op_insert_shape(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = ctx.region(step["region"])
    shape = step.get("shape", "rectangle")
    if shape == "line":
        a = normalized_point(region, float(step.get("u0", 0.2)), float(step.get("v0", 0.5)))
        b = normalized_point(region, float(step.get("u1", 0.8)), float(step.get("v1", 0.5)))
        ctx.add_segment(Segment(id=step.get("id", "line"), a=a, b=b), region_id=region.id)
        return
    if shape == "arc":
        _insert_arc(ctx, step, region)
        return
    raise ValueError(f"insert_shape not implemented for {shape!r}")


def _insert_arc(ctx: CompileContext, step: dict[str, Any], region: Region) -> None:
    from shapes import _segmental_arch_radius

    cx_n = float(step.get("cx", 0.5))
    cy_n = float(step.get("cy", 0.0))
    cx = region.min_x + cx_n * region.width
    cy = region.min_y + cy_n * region.height
    base_id = str(step.get("id", "arc"))
    n = int(step.get("segments", 16))

    if "rise" in step:
        rise = resolve_scalar(step["rise"], region)
        chord_u = float(step.get("chord", 1.0))
        chord_w = max(region.width * chord_u, 1e-4)
        _, cy_c, r = _segmental_arch_radius(chord_w, rise)
        cy_c += cy
        x_left = cx - chord_w / 2.0
        x_right = cx + chord_w / 2.0
        a0 = math.atan2(cy - cy_c, x_left - cx)
        a1 = math.atan2(cy - cy_c, x_right - cx)
        ctx.arc_muntins[base_id] = {
            "radius": r,
            "ox": cx_n,
            "oy": cy_n,
            "region_id": region.id,
            "rise": rise,
        }
    else:
        r = resolve_scalar(step.get("radius", 0.2), region)
        a0 = float(step.get("angle_start", 0.0))
        a1 = float(step.get("angle_end", math.pi))
        ctx.arc_muntins[base_id] = {
            "radius": r,
            "ox": cx_n,
            "oy": cy_n,
            "region_id": region.id,
        }

    prev: Vec2 | None = None
    for i in range(n + 1):
        t = a0 + (a1 - a0) * (i / n)
        if "rise" in step:
            p = Vec2(cx + r * math.cos(t), cy_c + r * math.sin(t))
        else:
            p = Vec2(cx + r * math.cos(t), cy + r * math.sin(t))
        if prev is not None:
            ctx.add_segment(
                Segment(id=f"{base_id}_{i}", a=prev, b=p),
                region_id=region.id,
            )
        prev = p


def _op_clip(ctx: CompileContext, step: dict[str, Any]) -> None:
    region = ctx.region(step["boundary"])
    kept: list[Segment] = []
    for seg in ctx.segments:
        clipped = region.clip_segment(seg)
        if clipped is not None:
            kept.append(clipped)
    ctx.segments = kept


def _op_mirror(ctx: CompileContext, step: dict[str, Any]) -> None:
    axis = step.get("axis", "y")
    region = ctx.region(step["region"]) if "region" in step else None
    origin = region.center if region else Vec2(0.5, 0.5)
    mirrored: list[Segment] = []
    for seg in list(ctx.segments):
        if step.get("target") and not seg.id.startswith(str(step["target"])):
            continue
        ma, mb = _mirror_point(seg.a, origin, axis), _mirror_point(seg.b, origin, axis)
        mirrored.append(Segment(id=f"{seg.id}_mir", a=ma, b=mb, layer=seg.layer))
    ctx.segments.extend(mirrored)


def _mirror_point(p: Vec2, origin: Vec2, axis: str) -> Vec2:
    if axis == "x":
        return Vec2(p.x, origin.y - (p.y - origin.y))
    if axis == "y":
        return Vec2(origin.x - (p.x - origin.x), p.y)
    return p


def _op_repeat(ctx: CompileContext, step: dict[str, Any]) -> None:
    sub = step.get("program") or step.get("body") or []
    count = int(step.get("count", 1))
    dx = float(step.get("dx", 0.0))
    dy = float(step.get("dy", 0.0))
    for i in range(count):
        local = copy.deepcopy(sub)
        for op in local:
            op["id"] = f"{op.get('id', 'rep')}_{i}"
            if "region" not in op and "region" in step:
                op["region"] = step["region"]
        execute_program(ctx, local)
        if dx or dy:
            shift = Vec2(dx * i, dy * i)
            for seg in ctx.segments:
                seg.a = Vec2(seg.a.x + shift.x, seg.a.y + shift.y)
                seg.b = Vec2(seg.b.x + shift.x, seg.b.y + shift.y)


def _op_delete(ctx: CompileContext, step: dict[str, Any]) -> None:
    target = step.get("target")
    if target is None:
        return
    ctx.segments = [s for s in ctx.segments if s.id != target and not s.id.startswith(f"{target}_")]


def _op_trim(ctx: CompileContext, step: dict[str, Any]) -> None:
    # v1: trim = clip segments to mask region
    mask = ctx.region(step["mask"])
    target = step.get("target")
    out: list[Segment] = []
    for seg in ctx.segments:
        if target and not seg.id.startswith(str(target)):
            out.append(seg)
            continue
        clipped = mask.clip_segment(seg)
        if clipped is not None:
            out.append(clipped)
    ctx.segments = out


def _op_merge(ctx: CompileContext, step: dict[str, Any]) -> None:
    # no-op in v1 — segments remain separate curve objects
    _ = step


_OP_HANDLERS = {
    "define_region": _op_define_region,
    "inset": _op_inset,
    "split_vertical": _op_split_vertical,
    "split_horizontal": _op_split_horizontal,
    "add_grid": _op_add_grid,
    "split_radial": _op_split_radial,
    "define_anchor": _op_define_anchor,
    "connect": _op_connect,
    "insert_shape": _op_insert_shape,
    "clip": _op_clip,
    "mirror": _op_mirror,
    "repeat": _op_repeat,
    "delete": _op_delete,
    "trim": _op_trim,
    "merge": _op_merge,
}
