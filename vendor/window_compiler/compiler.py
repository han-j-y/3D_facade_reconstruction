"""Compile window grammar JSON into CompileContext."""

from __future__ import annotations

from typing import Any

from context import CompileContext
from geom2d import Region, build_boundary
from muntin_resolve import resolve_muntins
from operations import execute_program
from pane_compile import compile_panes
from values import resolve_scalar


def compile_spec(spec: dict[str, Any]) -> CompileContext:
    if spec.get("type") != "window":
        raise ValueError('spec.type must be "window"')

    ctx = CompileContext(spec=spec, debug=bool(spec.get("debug", False)))

    root = build_boundary(spec["boundary"])
    ctx.regions[root.id] = root

    for region_def in spec.get("regions", []):
        _compile_region_def(ctx, region_def)

    execute_program(ctx, spec.get("program", []))

    panes_spec = spec.get("panes")
    if panes_spec:
        # Allow one pane tree or a list of trees (e.g. body + head).
        if isinstance(panes_spec, list):
            for entry in panes_spec:
                if entry:
                    compile_panes(ctx, entry)
        else:
            compile_panes(ctx, panes_spec)

    glass_id = _glass_region_id(ctx)
    if glass_id in ctx.regions and ctx.segments:
        inner = ctx.regions[glass_id]
        outer = ctx.regions.get("root")
        ctx.segments = resolve_muntins(ctx.segments, inner, outer, ctx.regions)

    return ctx


def _glass_region_id(ctx: CompileContext) -> str:
    if "glass" in ctx.regions:
        return "glass"
    for rid in ctx.regions:
        if rid != "root":
            return rid
    return "root"


def _compile_region_def(ctx: CompileContext, region_def: dict[str, Any]) -> None:
    op = region_def.get("operation", "inset")
    rid = region_def["id"]
    source_id = region_def.get("from", "root")
    source = ctx.region(source_id)

    if op == "inset":
        amount = resolve_scalar(region_def.get("amount", 0.0), source)
        ctx.regions[rid] = source.inset(amount, rid)
        return

    if op == "crop":
        bounds = region_def.get("bounds") or region_def.get("crop") or {}
        u0 = float(bounds.get("u0", 0.0))
        v0 = float(bounds.get("v0", 0.0))
        u1 = float(bounds.get("u1", 1.0))
        v1 = float(bounds.get("v1", 1.0))
        ctx.regions[rid] = source.crop_normalized(u0, v0, u1, v1, rid)
        return

    if op == "body":
        root = ctx.regions.get("root")
        if root is None or root.shape not in {"head_body", "arch_head", "rect_eyebrow", "springline_arch"}:
            raise ValueError('region operation "body" requires a head_body or arch_head boundary')
        body_h = float(root.params.get("body_height", root.height * 0.65))
        v1 = (body_h - source.min_y) / max(source.height, 1e-9)
        v1 = max(0.0, min(1.0, v1))
        ctx.regions[rid] = source.crop_normalized(0.0, 0.0, 1.0, v1, rid)
        return

    if op == "head":
        root = ctx.regions.get("root")
        if root is None or root.shape not in {"head_body", "arch_head", "rect_eyebrow", "springline_arch"}:
            raise ValueError('region operation "head" requires a head_body or arch_head boundary')
        body_h = float(root.params.get("body_height", root.height * 0.65))
        v0 = (body_h - source.min_y) / max(source.height, 1e-9)
        v0 = max(0.0, min(1.0, v0))
        ctx.regions[rid] = source.crop_normalized(0.0, v0, 1.0, 1.0, rid)
        return

    raise ValueError(f"unsupported region operation: {op!r}")


def output_params(spec: dict[str, Any], root: Region) -> dict[str, float]:
    out = dict(spec.get("output") or {})
    return {
        "muntin_width": resolve_scalar(out.get("muntin_width", 0.02), root),
        "muntin_depth": resolve_scalar(out.get("muntin_depth", 0.04), root),
        "glass_thickness": resolve_scalar(out.get("glass_thickness", 0.01), root),
        "frame_depth": resolve_scalar((spec.get("frame") or {}).get("depth", 0.08), root),
        "frame_thickness": resolve_scalar(
            (spec.get("frame") or {}).get("thickness", 0.06), root
        ),
    }
