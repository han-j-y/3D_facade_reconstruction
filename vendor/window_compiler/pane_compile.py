"""Compile recursive pane grammar into muntin segments."""

from __future__ import annotations

import re
from typing import Any

from context import CompileContext
from geom2d import Region, Segment, Vec2

_PANE_WRAP = re.compile(r"^pane\((.+)\)$", re.I)


def compile_panes(ctx: CompileContext, pane_spec: dict[str, Any]) -> None:
    region_id = pane_spec.get("region", "glass")
    root = pane_spec.get("root", "root")
    rules: dict[str, Any] = pane_spec["rules"]
    if root not in rules:
        raise ValueError(f"pane root {root!r} not defined")
    zone = ctx.region(region_id)
    _visit_pane(ctx, root, zone, rules, path=root)


def _visit_pane(
    ctx: CompileContext,
    name: str,
    zone: Region,
    rules: dict[str, Any],
    *,
    path: str,
) -> None:
    if name not in rules:
        raise ValueError(f"undefined pane rule: {name!r}")
    rule = rules[name]
    op = rule["op"]

    if op == "eps":
        return

    if op == "grid":
        n_v = int(rule.get("vertical", 1))
        n_h = int(rule.get("horizontal", 1))
        _emit_grid(ctx, zone, n_v, n_h, prefix=path)
        children = rule.get("children") or []
        if not children:
            return
        cells = _grid_cells(n_v, n_h)
        if len(children) != len(cells):
            raise ValueError(
                f"pane {name!r}: grid {n_v}x{n_h} needs {len(cells)} children, got {len(children)}"
            )
        for child, (u0, v0, u1, v1) in zip(children, cells):
            sub = _subzone(zone, u0, v0, u1, v1, f"{path}_{child}")
            ctx.regions[sub.id] = sub
            _visit_pane(ctx, child, sub, rules, path=f"{path}_{child}")
        return

    if op != "split":
        raise ValueError(f"unknown pane op: {op!r}")

    axis = rule["axis"]
    at = float(rule["at"])
    children = rule["children"]
    if len(children) != 2:
        raise ValueError(f"split pane {name!r} requires exactly 2 children")
    for child in children:
        if child not in rules:
            defined = ", ".join(sorted(rules))
            raise ValueError(
                f"undefined pane rule {child!r} (referenced from {name!r}); defined: {defined}"
            )

    if axis == "v":
        x = zone.min_x + at * zone.width
        ctx.add_segment(
            Segment(f"{path}_v", Vec2(x, zone.min_y), Vec2(x, zone.max_y)),
            region_id=zone.id,
        )
        left = _subzone(zone, 0.0, 0.0, at, 1.0, f"{path}_{children[0]}")
        right = _subzone(zone, at, 0.0, 1.0, 1.0, f"{path}_{children[1]}")
    else:
        y = zone.min_y + at * zone.height
        ctx.add_segment(
            Segment(f"{path}_h", Vec2(zone.min_x, y), Vec2(zone.max_x, y)),
            region_id=zone.id,
        )
        left = _subzone(zone, 0.0, 0.0, 1.0, at, f"{path}_{children[0]}")
        right = _subzone(zone, 0.0, at, 1.0, 1.0, f"{path}_{children[1]}")

    ctx.regions[left.id] = left
    ctx.regions[right.id] = right
    _visit_pane(ctx, children[0], left, rules, path=f"{path}_{children[0]}")
    _visit_pane(ctx, children[1], right, rules, path=f"{path}_{children[1]}")


def _subzone(parent: Region, u0: float, v0: float, u1: float, v1: float, rid: str) -> Region:
    return parent.crop_normalized(u0, v0, u1, v1, rid)


def _emit_grid(ctx: CompileContext, zone: Region, n_v: int, n_h: int, *, prefix: str) -> None:
    for i in range(1, n_v + 1):
        u = i / (n_v + 1)
        x = zone.min_x + u * zone.width
        ctx.add_segment(
            Segment(f"{prefix}_v{i}", Vec2(x, zone.min_y), Vec2(x, zone.max_y)),
            region_id=zone.id,
        )
    for j in range(1, n_h + 1):
        v = j / (n_h + 1)
        y = zone.min_y + v * zone.height
        ctx.add_segment(
            Segment(f"{prefix}_h{j}", Vec2(zone.min_x, y), Vec2(zone.max_x, y)),
            region_id=zone.id,
        )


def _grid_cells(n_v: int, n_h: int) -> list[tuple[float, float, float, float]]:
    cols, rows = n_v + 1, n_h + 1
    out: list[tuple[float, float, float, float]] = []
    for row in range(rows):
        for col in range(cols):
            u0 = col / cols
            u1 = (col + 1) / cols
            v0 = row / rows
            v1 = (row + 1) / rows
            out.append((u0, v0, u1, v1))
    return out


def normalize_pane_name(token: str) -> str:
    token = token.strip()
    m = _PANE_WRAP.match(token)
    if m:
        return m.group(1)
    return token
