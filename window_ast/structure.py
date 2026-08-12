"""Structure-only IR view and tokenization (shape / ops / discrete counts).

Continuous floats (sizes, split positions, thicknesses) are dropped so that
exact token match ≡ exact structure match.
"""

from __future__ import annotations

from typing import Any

_DISCRETE_BOUNDARY_KEYS = ("sides", "orient", "head", "flat", "corner")

# Defaults used when decoding structure tokens back into a renderable IR
_DEFAULT_FRAME = {
    "thickness": {"relative_to": "min_side", "value": 0.06},
    "depth": 0.08,
    "material": "painted_wood",
}
_DEFAULT_OUTPUT = {
    "muntin_width": {"relative_to": "min_side", "value": 0.025},
    "muntin_depth": 0.035,
    "glass_thickness": 0.01,
}


def _parse_kv(token: str) -> tuple[str, str]:
    if "=" not in token:
        raise ValueError(f"expected key=value token, got {token!r}")
    return token.split("=", 1)


def _parse_str(token: str, key: str) -> str:
    k, v = _parse_kv(token)
    if k != key:
        raise ValueError(f"expected {key}=..., got {token!r}")
    return v


def _parse_int(token: str, key: str) -> int:
    return int(_parse_str(token, key))


def _is_seam_op(op: dict[str, Any]) -> bool:
    """True if this program step is a head/body junction seam (not a mid-pane split_h)."""
    if op.get("op") == "seam":
        return True
    return op.get("op") == "split_horizontal" and op.get("id") == "seam"


def _normalize_program_entry(op: dict[str, Any]) -> dict[str, Any]:
    """Canonical program entry for structure fingerprints / tokens."""
    if _is_seam_op(op):
        entry: dict[str, Any] = {
            "op": "seam",
            "id": "seam",
            "region": op.get("region", "body"),
        }
        return entry
    entry = {"op": op.get("op")}
    for k in ("id", "region", "shape", "from_arc"):
        if k in op:
            entry[k] = op[k]
    for k in ("vertical", "horizontal", "count", "segments"):
        if k in op:
            entry[k] = int(op[k])
    return entry


def structure_view(ir: dict[str, Any]) -> dict[str, Any]:
    """Canonical structure fingerprint (no continuous floats).

    Pane trees are BSP-order-canonicalized (through-crosses → H-then-V) and
    compared as nested shapes (node ids ignored).
    """
    params = ir.get("boundary", {}).get("params") or {}
    shape_attrs = {k: params[k] for k in _DISCRETE_BOUNDARY_KEYS if k in params}

    regions = []
    for r in ir.get("regions") or []:
        regions.append(
            {
                "id": r.get("id"),
                "from": r.get("from"),
                "operation": r.get("operation"),
                # crop bounds presence matters structurally; ignore numeric values
                "has_bounds": "bounds" in r,
            }
        )

    program = [_normalize_program_entry(op) for op in (ir.get("program") or [])]

    panes_fp = None
    raw_panes = ir.get("panes")
    if isinstance(raw_panes, list):
        fps = []
        for entry in raw_panes:
            if entry and (entry.get("rules") or {}):
                fps.append(_pane_tree_fingerprint(canonicalize_pane_bsp(entry)))
        panes_fp = fps or None
    elif raw_panes and (raw_panes.get("rules") or {}):
        panes_fp = _pane_tree_fingerprint(canonicalize_pane_bsp(raw_panes))

    return {
        "shape": ir.get("boundary", {}).get("shape"),
        "shape_attrs": shape_attrs,
        "regions": regions,
        "program": program,
        "panes": panes_fp,
    }


def structure_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return structure_view(a) == structure_view(b)


def _copy_panes(panes: dict[str, Any]) -> dict[str, Any]:
    rules = {}
    for rid, rule in (panes.get("rules") or {}).items():
        rr = dict(rule)
        if "children" in rr:
            rr["children"] = list(rr["children"])
        rules[rid] = rr
    return {
        "region": panes.get("region", "glass"),
        "root": panes.get("root", "root"),
        "rules": rules,
    }


def _at_of(rule: dict[str, Any], default: float = 0.5) -> float:
    try:
        return float(rule.get("at", default))
    except (TypeError, ValueError):
        return default


def _is_through_cross(
    rules: dict[str, Any],
    nid: str,
    *,
    primary_axis: str,
    secondary_axis: str,
    atol: float = 1e-4,
) -> bool:
    """True if nid is primary_axis split whose both kids are secondary_axis with matching at."""
    rule = rules.get(nid) or {}
    if rule.get("op") != "split" or rule.get("axis") != primary_axis:
        return False
    kids = list(rule.get("children") or [])
    if len(kids) != 2:
        return False
    r0 = rules.get(kids[0]) or {}
    r1 = rules.get(kids[1]) or {}
    if r0.get("op") != "split" or r0.get("axis") != secondary_axis:
        return False
    if r1.get("op") != "split" or r1.get("axis") != secondary_axis:
        return False
    if abs(_at_of(r0) - _at_of(r1)) > atol:
        return False
    if len(list(r0.get("children") or [])) != 2 or len(list(r1.get("children") or [])) != 2:
        return False
    return True


def canonicalize_pane_bsp(
    panes: dict[str, Any],
    *,
    prefer_root_axis: str = "h",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Rewrite through-crosses to a preferred BSP order (default H-then-V).

    Only rewrites when a split's two children are opposite-axis splits with the
    same ``at`` (full rectangular cross in that cell). T-junctions and
    asymmetric trees are left unchanged. Recurses until a fixed point.
    """
    out = _copy_panes(panes)
    rules = out["rules"]
    if prefer_root_axis not in ("h", "v"):
        raise ValueError(f"prefer_root_axis must be 'h' or 'v', got {prefer_root_axis!r}")
    other_axis = "v" if prefer_root_axis == "h" else "h"

    def rewrite_node(nid: str, *, depth: int = 0, stack: set[str] | None = None) -> bool:
        """If nid is other→prefer cross, rewrite to prefer→other. Recurse first."""
        if depth > 64:
            return False
        if stack is None:
            stack = set()
        if nid in stack:
            return False  # cyclic / malformed pane tree
        rule = rules.get(nid)
        if not rule or rule.get("op") != "split":
            return False
        stack.add(nid)
        changed = False
        try:
            for ch in list(rule.get("children") or []):
                if rewrite_node(ch, depth=depth + 1, stack=stack):
                    changed = True
            # After children are canonical, try rewriting this node.
            if _is_through_cross(
                rules, nid, primary_axis=other_axis, secondary_axis=prefer_root_axis, atol=atol
            ):
                # e.g. V→H×2 → H→V×2 when prefer_root_axis='h'
                a0, a1 = rules[nid]["children"]  # left/right if V, or bot/top if H
                r0, r1 = rules[a0], rules[a1]
                b00, b01 = list(r0["children"])  # if V→H: L_bot, L_top
                b10, b11 = list(r1["children"])  # R_bot, R_top
                at_primary = _at_of(rules[nid])  # old primary (other) at
                at_secondary = _at_of(r0)  # shared secondary (prefer) at
                # Reuse a0/a1 as the new prefer-axis children.
                if prefer_root_axis == "h":
                    # old was V→H: a0=L, a1=R; b00=Lb,b01=Lt,b10=Rb,b11=Rt
                    # new H→V: bot=[Lb,Rb], top=[Lt,Rt]
                    rules[nid] = {
                        "op": "split",
                        "axis": "h",
                        "at": at_secondary,
                        "children": [a0, a1],
                    }
                    rules[a0] = {
                        "op": "split",
                        "axis": "v",
                        "at": at_primary,
                        "children": [b00, b10],
                    }
                    rules[a1] = {
                        "op": "split",
                        "axis": "v",
                        "at": at_primary,
                        "children": [b01, b11],
                    }
                else:
                    # prefer V: rewrite H→V×2 → V→H×2
                    # old H→V: a0=BOT, a1=TOP; b00=BL,b01=BR,b10=TL,b11=TR
                    # new V→H: L=[BL,TL], R=[BR,TR]
                    rules[nid] = {
                        "op": "split",
                        "axis": "v",
                        "at": at_secondary,
                        "children": [a0, a1],
                    }
                    rules[a0] = {
                        "op": "split",
                        "axis": "h",
                        "at": at_primary,
                        "children": [b00, b10],
                    }
                    rules[a1] = {
                        "op": "split",
                        "axis": "h",
                        "at": at_primary,
                        "children": [b01, b11],
                    }
                changed = True
                # New children may themselves contain crosses further down — recurse.
                rewrite_node(a0, depth=depth + 1, stack=stack)
                rewrite_node(a1, depth=depth + 1, stack=stack)
        finally:
            stack.discard(nid)
        return changed

    root = out.get("root", "root")
    if root in rules:
        # Iterate to fixed point (nested crosses).
        for _ in range(32):
            if not rewrite_node(root):
                break
    return out


def _pane_tree_fingerprint(panes: dict[str, Any]) -> Any:
    """Nested pane shape with node ids stripped (after BSP canonicalize).

    Regular through-grids collapse to ``('grid', nv, nh)`` so a BSP spelling and
    a compact grid rule compare equal.
    """
    panes = canonicalize_pane_bsp(panes)
    compressed = try_compress_through_grid(panes)
    if compressed is not None:
        root = compressed["rules"]["root"]
        return {
            "region": compressed.get("region"),
            "tree": ("grid", int(root.get("vertical") or 0), int(root.get("horizontal") or 0)),
        }

    rules = panes.get("rules") or {}
    root = panes.get("root", "root")

    def walk(nid: str) -> Any:
        rule = rules.get(nid) or {}
        op = rule.get("op")
        if op == "split":
            kids = list(rule.get("children") or [])
            return (
                "split",
                rule.get("axis"),
                tuple(walk(ch) for ch in kids),
            )
        if op == "grid":
            return (
                "grid",
                int(rule.get("vertical") or 0),
                int(rule.get("horizontal") or 0),
            )
        return ("eps",)

    return {
        "region": panes.get("region"),
        "tree": walk(root) if root in rules else None,
    }


def try_compress_through_grid(
    panes: dict[str, Any],
    *,
    use_at: bool = False,
    max_v: int = 8,
    max_h: int = 8,
) -> dict[str, Any] | None:
    """If panes form a full rectangular through-grid, return compact ``op=grid`` panes.

    Requires every V muntin to span full height and every H muntin full width, with
    ``n_leaves == (nv+1)*(nh+1)``. Empty (no muntins) returns None (keep ``eps``).
    """
    if not panes or not (panes.get("rules") or {}):
        return None
    # Already a single grid rule.
    root_id = panes.get("root", "root")
    root_rule = (panes.get("rules") or {}).get(root_id) or {}
    if root_rule.get("op") == "grid" and len(panes.get("rules") or {}) == 1:
        nv = int(root_rule.get("vertical") or 0)
        nh = int(root_rule.get("horizontal") or 0)
        if nv == 0 and nh == 0:
            return None
        if nv > max_v or nh > max_h:
            return None
        return {
            "region": panes.get("region", "glass"),
            "root": "root",
            "rules": {"root": {"op": "grid", "vertical": nv, "horizontal": nh}},
        }

    v_lines, h_lines, n_leaves = _layout_pane_segments(panes, use_at=use_at)
    nv = len(v_lines)
    nh = len(h_lines)
    if nv == 0 and nh == 0:
        return None
    if nv > max_v or nh > max_h:
        return None
    for spans in v_lines.values():
        if spans != [(0.0, 1.0)]:
            return None
    for spans in h_lines.values():
        if spans != [(0.0, 1.0)]:
            return None
    if int(n_leaves) != (nv + 1) * (nh + 1):
        return None
    return {
        "region": panes.get("region", "glass"),
        "root": "root",
        "rules": {"root": {"op": "grid", "vertical": nv, "horizontal": nh}},
    }


def muntin_counts(ir: dict[str, Any], *, use_at: bool = False) -> tuple[int, int]:
    """Return (#vertical muntin lines, #horizontal muntin lines)."""
    tv = topology_view(ir, use_at=use_at)
    if tv.get("through_grid") is not None:
        return int(tv["through_grid"][0]), int(tv["through_grid"][1])
    return len(tv["v_segments"]), len(tv["h_segments"])


def _quantize(x: float, ndigits: int = 4) -> float:
    return round(float(x), ndigits)


def _merge_intervals(intervals: list[tuple[float, float]], *, eps: float = 1e-6) -> list[tuple[float, float]]:
    if not intervals:
        return []
    items = sorted((min(a, b), max(a, b)) for a, b in intervals)
    out: list[tuple[float, float]] = [items[0]]
    for a, b in items[1:]:
        la, lb = out[-1]
        if a <= lb + eps:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _layout_pane_segments(
    panes: dict[str, Any],
    *,
    ndigits: int = 4,
    use_at: bool = True,
) -> tuple[dict[float, list[tuple[float, float]]], dict[float, list[tuple[float, float]]], int]:
    """Return (v_lines, h_lines, n_leaves).

    v_lines[x] = list of (y0,y1) spans in IR UV (v up).
    h_lines[y] = list of (x0,x1) spans.
    If ``use_at`` is True, uses rule.at when present else 0.5.
    If False, always splits at 0.5 (structure-token / combinatorial topology).
    """
    rules = panes.get("rules") or {}
    root = panes.get("root", "root")
    v_raw: dict[float, list[tuple[float, float]]] = {}
    h_raw: dict[float, list[tuple[float, float]]] = {}
    n_leaves = 0

    def add_v(x: float, y0: float, y1: float) -> None:
        xq = _quantize(x, ndigits)
        v_raw.setdefault(xq, []).append((_quantize(y0, ndigits), _quantize(y1, ndigits)))

    def add_h(y: float, x0: float, x1: float) -> None:
        yq = _quantize(y, ndigits)
        h_raw.setdefault(yq, []).append((_quantize(x0, ndigits), _quantize(x1, ndigits)))

    def walk(pid: str, u0: float, v0: float, u1: float, v1: float) -> None:
        nonlocal n_leaves
        rule = rules.get(pid)
        if not rule:
            return
        op = rule.get("op")
        if op in (None, "eps"):
            n_leaves += 1
            return
        if op == "grid":
            # Equal muntins inside this leaf (structure-level grid).
            nv = int(rule.get("vertical") or 0)
            nh = int(rule.get("horizontal") or 0)
            for i in range(1, nv + 1):
                add_v(u0 + (u1 - u0) * i / (nv + 1), v0, v1)
            for j in range(1, nh + 1):
                add_h(v0 + (v1 - v0) * j / (nh + 1), u0, u1)
            n_leaves += (nv + 1) * (nh + 1)
            return
        if op != "split":
            n_leaves += 1
            return
        children = list(rule.get("children") or [])
        if len(children) != 2:
            n_leaves += 1
            return
        if use_at:
            at = float(rule.get("at", 0.5))
            at = min(0.999, max(0.001, at))
        else:
            at = 0.5
        axis = rule.get("axis")
        c0, c1 = children
        if axis == "v":
            u = u0 + at * (u1 - u0)
            add_v(u, v0, v1)
            walk(c0, u0, v0, u, v1)
            walk(c1, u, v0, u1, v1)
        elif axis == "h":
            v = v0 + at * (v1 - v0)
            add_h(v, u0, u1)
            walk(c0, u0, v0, u1, v)  # bottom
            walk(c1, u0, v, u1, v1)  # top
        else:
            n_leaves += 1

    if root in rules:
        walk(root, 0.0, 0.0, 1.0, 1.0)
    elif rules:
        # fallback: no walkable root
        n_leaves = sum(1 for r in rules.values() if r.get("op") in (None, "eps"))

    v_lines = {x: _merge_intervals(spans) for x, spans in sorted(v_raw.items())}
    h_lines = {y: _merge_intervals(spans) for y, spans in sorted(h_raw.items())}
    return v_lines, h_lines, n_leaves


def topology_view(
    ir: dict[str, Any],
    *,
    ndigits: int = 4,
    use_at: bool = True,
) -> dict[str, Any]:
    """Dissection fingerprint: shape + maximal muntin segments + leaf count.

    Invariant to BSP construction order (e.g. V-then-H vs H-then-V full cross).
    With ``use_at=True``, uses continuous rule.at when present (else 0.5).
    With ``use_at=False``, always splits at 0.5 — correct for structure-token models
    whose labels ignore continuous ratios.

    Regular through-grids (including compact ``op=grid``) collapse to
    ``through_grid=(nv, nh)`` so equal-count grids match regardless of spacing.
    """
    params = ir.get("boundary", {}).get("params") or {}
    shape_attrs = {k: params[k] for k in _DISCRETE_BOUNDARY_KEYS if k in params}
    program = [_normalize_program_entry(op) for op in (ir.get("program") or [])]

    panes = ir.get("panes")
    # Multi-region labels store panes as a list of {region, root, rules} entries.
    # Prefer the glass tree when present; otherwise the first structured entry.
    if isinstance(panes, list):
        chosen = None
        for entry in panes:
            if not isinstance(entry, dict) or not (entry.get("rules") or {}):
                continue
            if entry.get("region") == "glass":
                chosen = entry
                break
            if chosen is None:
                chosen = entry
        panes = chosen
    if panes and (panes.get("rules") or {}):
        compressed = try_compress_through_grid(panes, use_at=use_at)
        if compressed is not None:
            root = compressed["rules"]["root"]
            nv = int(root.get("vertical") or 0)
            nh = int(root.get("horizontal") or 0)
            return {
                "shape": ir.get("boundary", {}).get("shape"),
                "shape_attrs": shape_attrs,
                "program": program,
                "through_grid": (nv, nh),
                "n_leaves": (nv + 1) * (nh + 1),
                "v_segments": (),
                "h_segments": (),
            }
        v_lines, h_lines, n_leaves = _layout_pane_segments(
            panes, ndigits=ndigits, use_at=use_at
        )
        v_segs = tuple((x, tuple(spans)) for x, spans in v_lines.items())
        h_segs = tuple((y, tuple(spans)) for y, spans in h_lines.items())
    else:
        v_segs, h_segs, n_leaves = (), (), 1

    return {
        "shape": ir.get("boundary", {}).get("shape"),
        "shape_attrs": shape_attrs,
        "program": program,
        "through_grid": None,
        "n_leaves": n_leaves,
        "v_segments": v_segs,
        "h_segments": h_segs,
    }


def topology_equal(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    ndigits: int = 4,
    use_at: bool = True,
) -> bool:
    """True if A and B induce the same shape + muntin dissection (order-invariant)."""
    return topology_view(a, ndigits=ndigits, use_at=use_at) == topology_view(
        b, ndigits=ndigits, use_at=use_at
    )


def _default_boundary_params(shape: str, attrs: dict[str, Any]) -> dict[str, Any]:
    """Fill reasonable continuous params so decoded IR can render."""
    params = dict(attrs)
    if shape == "circle":
        params.setdefault("diameter", 1.0)
    elif shape == "ellipse":
        params.setdefault("width", 1.4)
        params.setdefault("height", 0.9)
    elif shape == "semicircle":
        params.setdefault("width", 1.2)
        params.setdefault("flat", params.get("flat", "bottom"))
    elif shape == "eyebrow":
        params.setdefault("width", 1.2)
        params.setdefault("rise", 0.3)
    elif shape == "trapezoid":
        params.setdefault("width", 1.0)
        params.setdefault("height", 1.2)
        params.setdefault("top_width", 0.5)
    elif shape == "polygon":
        params.setdefault("sides", int(params.get("sides", 6)))
        params.setdefault("width", 1.0)
        params.setdefault("height", 1.0)
        params.setdefault("orient", params.get("orient", "flat"))
    elif shape == "quadrant":
        params.setdefault("size", 1.0)
        params.setdefault("corner", params.get("corner", "bl"))
    elif shape == "arch_head":
        params.setdefault("width", 1.0)
        params.setdefault("height", 1.4)
        params.setdefault("rise", 0.35)
    elif shape == "head_body":
        params.setdefault("width", 1.0)
        params.setdefault("body_height", 1.3)
        params.setdefault("head_height", 0.4)
        params.setdefault("head", params.get("head", "semicircle"))
    else:  # rectangle and fallbacks
        params.setdefault("width", 1.0)
        params.setdefault("height", 1.4)
    return params


def _pane_rule_order(rules: dict[str, Any], root: str) -> list[str]:
    """Root-first DFS over the pane tree; append any unreachable rules alphabetically."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen or nid not in rules:
            return
        seen.add(nid)
        order.append(nid)
        rule = rules[nid]
        if rule.get("op") == "split":
            for ch in rule.get("children") or []:
                visit(ch)

    if root in rules:
        visit(root)
    for nid in sorted(rules.keys()):
        if nid not in seen:
            visit(nid)
    return order


def ir_to_structure_tokens(ir: dict[str, Any]) -> list[str]:
    """Serialize discrete structure only (no continuous floats)."""
    tokens: list[str] = []
    boundary = ir["boundary"]
    tokens.append("BOUNDARY")
    tokens.append(f"shape={boundary['shape']}")
    params = boundary.get("params") or {}
    for key in _DISCRETE_BOUNDARY_KEYS:
        if key not in params:
            continue
        val = params[key]
        if key == "sides":
            tokens.append(f"sides={int(val)}")
        else:
            tokens.append(f"{key}={val}")

    tokens.append("REGION")
    for region in ir.get("regions") or []:
        tokens.append(f"id={region['id']}")
        tokens.append(f"from={region['from']}")
        tokens.append(f"operation={region['operation']}")
        if "bounds" in region:
            tokens.append("has_bounds=1")

    program = ir.get("program") or []
    tokens.append("PROGRAM")
    tokens.append(f"n={len(program)}")
    for op in program:
        tokens.append("OP")
        if _is_seam_op(op):
            # Dedicated seam token: head/body junction (never mid-pane split_h)
            tokens.append("op=seam")
            tokens.append(f"R={op.get('region', 'body')}")
            continue
        tokens.append(f"op={op['op']}")
        if "id" in op:
            tokens.append(f"id={op['id']}")
        if "region" in op:
            tokens.append(f"R={op['region']}")
        if "shape" in op:
            tokens.append(f"shape={op['shape']}")
        if "vertical" in op:
            tokens.append(f"V={int(op['vertical'])}")
        if "horizontal" in op:
            tokens.append(f"H={int(op['horizontal'])}")
        if "count" in op:
            tokens.append(f"count={int(op['count'])}")
        if "segments" in op:
            tokens.append(f"segments={int(op['segments'])}")
        if "from_arc" in op:
            tokens.append(f"from_arc={op['from_arc']}")

    panes = ir.get("panes")
    if panes:
        pane_entries = panes if isinstance(panes, list) else [panes]
        for entry in pane_entries:
            if not entry or not (entry.get("rules") or {}):
                continue
            entry = canonicalize_pane_bsp(entry)
            compressed = try_compress_through_grid(entry)
            if compressed is not None:
                entry = compressed
            tokens.append("PANES")
            tokens.append(f"region={entry.get('region', 'glass')}")
            tokens.append(f"root={entry.get('root', 'root')}")
            rules = entry.get("rules") or {}
            for rule_id in _pane_rule_order(rules, entry.get("root", "root")):
                rule = rules[rule_id]
                tokens.append("RULE")
                tokens.append(f"id={rule_id}")
                tokens.append(f"op={rule['op']}")
                if rule["op"] == "grid":
                    tokens.append(f"V={int(rule.get('vertical') or 0)}")
                    tokens.append(f"H={int(rule.get('horizontal') or 0)}")
                elif rule["op"] == "split":
                    tokens.append(f"axis={rule['axis']}")
                    children = rule.get("children") or []
                    tokens.append(f"nc={len(children)}")
                    for ci, ch in enumerate(children):
                        tokens.append(f"c{ci}={ch}")

    tokens.append("END")
    return tokens

def structure_tokens_to_ir(tokens: list[str]) -> dict[str, Any]:
    """Decode structure tokens into a renderable IR with default floats."""
    i = 0
    n = len(tokens)

    def peek() -> str | None:
        return tokens[i] if i < n else None

    def take() -> str:
        nonlocal i
        if i >= n:
            raise ValueError("unexpected end of tokens")
        t = tokens[i]
        i += 1
        return t

    def expect(word: str) -> None:
        t = take()
        if t != word:
            raise ValueError(f"expected {word!r}, got {t!r}")

    expect("BOUNDARY")
    shape = _parse_str(take(), "shape")
    attrs: dict[str, Any] = {}
    while peek() not in (None, "REGION"):
        k, v = _parse_kv(take())
        if k == "sides":
            attrs[k] = int(v)
        else:
            attrs[k] = v
    boundary = {
        "id": "root",
        "shape": shape,
        "params": _default_boundary_params(shape, attrs),
    }

    expect("REGION")
    regions: list[dict[str, Any]] = []
    while peek() not in (None, "PROGRAM"):
        rid = _parse_str(take(), "id")
        frm = _parse_str(take(), "from")
        operation = _parse_str(take(), "operation")
        region: dict[str, Any] = {"id": rid, "from": frm, "operation": operation}
        if operation == "inset":
            region["amount"] = {"relative_to": "min_side", "value": 0.06}
        if peek() == "has_bounds=1":
            take()
            # generic crop; compile may refine
            region["bounds"] = {"u0": 0.0, "v0": 0.0, "u1": 1.0, "v1": 0.7}
        regions.append(region)

    expect("PROGRAM")
    n_ops = _parse_int(take(), "n")
    program: list[dict[str, Any]] = []
    for _ in range(n_ops):
        expect("OP")
        op: dict[str, Any] = {"op": _parse_str(take(), "op")}
        while peek() not in (None, "OP", "PANES", "END"):
            t = peek()
            assert t is not None
            if t.startswith("id="):
                op["id"] = _parse_str(take(), "id")
            elif t.startswith("R="):
                op["region"] = _parse_str(take(), "R")
            elif t.startswith("shape="):
                op["shape"] = _parse_str(take(), "shape")
            elif t.startswith("V="):
                op["vertical"] = _parse_int(take(), "V")
            elif t.startswith("H="):
                op["horizontal"] = _parse_int(take(), "H")
            elif t.startswith("count="):
                op["count"] = _parse_int(take(), "count")
            elif t.startswith("segments="):
                op["segments"] = _parse_int(take(), "segments")
            elif t.startswith("from_arc="):
                op["from_arc"] = _parse_str(take(), "from_arc")
            else:
                raise ValueError(f"unexpected structure program token {t!r}")
        # default continuous fields for known ops
        if op["op"] == "seam":
            # Head/body junction: top of body (or bottom of head)
            region = op.get("region", "body")
            y = 0.0 if region == "head" else 1.0
            op = {
                "op": "split_horizontal",
                "id": "seam",
                "region": region,
                "y": y,
            }
        elif op["op"] == "split_vertical":
            op.setdefault("x", 0.5)
            op.setdefault("region", "glass")
            op.setdefault("id", "v")
        elif op["op"] == "split_horizontal":
            # Mid-pane split — never a seam
            op.setdefault("y", 0.5)
            op.setdefault("region", "glass")
            op.setdefault("id", "h")
        elif op["op"] == "add_grid":
            op.setdefault("region", "glass")
            op.setdefault("id", "grid")
            op.setdefault("vertical", 1)
            op.setdefault("horizontal", 1)
        elif op["op"] == "insert_shape":
            op.setdefault("shape", "arc")
            op.setdefault("id", "hub")
            op.setdefault("region", "glass")
            op.setdefault("cx", 0.5)
            op.setdefault("cy", 0.0)
            op.setdefault("radius", {"relative_to": "min_side", "value": 0.25})
            op.setdefault("angle_start", 0.0)
            import math

            op.setdefault("angle_end", math.pi)
            op.setdefault("segments", 16)
        elif op["op"] == "split_radial":
            op.setdefault("id", "radial")
            op.setdefault("region", "glass")
            op.setdefault("ox", 0.5)
            op.setdefault("oy", 0.0)
            op.setdefault("count", 3)
            op.setdefault("arc_start", 20.0)
            op.setdefault("arc_end", 160.0)
        program.append(op)

    ir: dict[str, Any] = {
        "type": "window",
        "debug": False,
        "boundary": boundary,
        "frame": dict(_DEFAULT_FRAME),
        "regions": regions,
        "program": program,
        "output": dict(_DEFAULT_OUTPUT),
    }

    pane_list: list[dict[str, Any]] = []
    while peek() == "PANES":
        take()
        panes_region = _parse_str(take(), "region")
        panes_root = _parse_str(take(), "root")
        rules: dict[str, Any] = {}
        while peek() == "RULE":
            take()
            rule_id = _parse_str(take(), "id")
            rule_op = _parse_str(take(), "op")
            rule: dict[str, Any] = {"op": rule_op}
            if rule_op == "split":
                rule["axis"] = _parse_str(take(), "axis")
                nc = _parse_int(take(), "nc")
                children = [_parse_str(take(), f"c{ci}") for ci in range(nc)]
                rule["children"] = children
                rule["at"] = 0.5
            elif rule_op == "grid":
                rule["vertical"] = _parse_int(take(), "V")
                rule["horizontal"] = _parse_int(take(), "H")
            rules[rule_id] = rule
        pane_list.append({"region": panes_region, "root": panes_root, "rules": rules})
    if len(pane_list) == 1:
        ir["panes"] = pane_list[0]
    elif len(pane_list) > 1:
        ir["panes"] = pane_list

    if peek() == "END":
        take()
    if i != n:
        raise ValueError(f"trailing tokens: {tokens[i:]}")
    return ir


def structure_round_trip_ok(ir: dict[str, Any]) -> bool:
    t1 = ir_to_structure_tokens(ir)
    ir2 = structure_tokens_to_ir(t1)
    return structure_equal(ir, ir2)
