#!/usr/bin/env python3
"""
Parse window_compiler WDSL (window DSL) into JSON spec dict.

  python3 parse_wdsl.py examples/example_grid.wdsl
  python3 parse_wdsl.py examples/example_grid.wdsl --out /tmp/spec.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


_RELPAT = re.compile(
    r"^\s*(?P<val>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\*\s*(?P<rel>min_side|max_side|width|height)\s*$",
    re.I,
)
from pane_compile import normalize_pane_name

_SPLIT_RULE = re.compile(
    r"^(\w+)\s+split\s*\(\s*([vh])\s*=\s*([\d.]+)\s*\)\s+(.+)$",
    re.I,
)
_GRID_RULE = re.compile(r"^(\w+)\s+grid\s*\(\s*(\d+)\s*x\s*(\d+)\s*\)\s*(.*)$", re.I)
_EPS_RULE = re.compile(r"^(\w+)\s+eps\s*$", re.I)


def parse_wdsl(text: str) -> dict[str, Any]:
    lines = _strip_comments(text)
    blocks = _split_blocks(lines)
    spec: dict[str, Any] = {"type": "window", "debug": False}

    if "window" in blocks:
        spec["boundary"] = _parse_window(blocks["window"])
    else:
        raise ValueError("WDSL requires a window line")

    if "frame" in blocks:
        spec["frame"] = _parse_frame(blocks["frame"])
    if "regions" in blocks:
        spec["regions"] = _parse_regions(blocks["regions"])
    if "program" in blocks:
        spec["program"] = _parse_program(blocks["program"])
    if "panes" in blocks and isinstance(blocks["panes"], dict):
        spec["panes"] = blocks["panes"]
    if "output" in blocks:
        spec["output"] = _parse_output(blocks["output"])
    if blocks.get("debug"):
        spec["debug"] = True

    return spec


def parse_wdsl_file(path: str | Path) -> dict[str, Any]:
    return parse_wdsl(Path(path).read_text(encoding="utf-8-sig"))


def _strip_comments(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.rstrip())
    return out


def _split_blocks(lines: list[str]) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current
        if current is None:
            return
        if current in {"regions", "program", "output"}:
            blocks[current] = buf
        elif current == "panes":
            pass
        elif current == "debug":
            blocks["debug"] = True
        else:
            blocks[current] = " ".join(buf)
        buf = []

    for line in lines:
        stripped = line.strip()
        if stripped == "debug":
            flush()
            current = "debug"
            flush()
            current = None
            continue
        if stripped.startswith("panes"):
            flush()
            parts = stripped.rstrip(":").split()
            region = parts[1] if len(parts) > 1 else "glass"
            blocks["panes"] = {"region": region, "rules": {}, "root": "root"}
            current = "panes"
            buf = []
            continue
        if stripped.endswith(":") and stripped[:-1] in {"regions", "program", "output"}:
            flush()
            current = stripped[:-1]
            continue
        head = stripped.split(None, 1)[0].lower() if stripped else ""
        if head in {"window", "frame"} and not line.startswith((" ", "\t")):
            flush()
            current = head
            rest = stripped.split(None, 1)[1] if " " in stripped else ""
            buf = [rest] if rest else []
            flush()
            current = None
            continue
        if current in {"regions", "program", "output", "panes"}:
            if current == "panes":
                pane_spec = blocks.setdefault("panes", {"region": "glass", "rules": {}, "root": "root"})
                _parse_pane_line(stripped, pane_spec)
            else:
                buf.append(line)
        elif current in {"window", "frame"}:
            buf.append(stripped)
    flush()
    return blocks


def _parse_scalar(token: str) -> Any:
    token = token.strip().rstrip(",")
    if not token:
        raise ValueError("empty scalar")
    m = _RELPAT.match(token)
    if m:
        return {"relative_to": m.group("rel"), "value": float(m.group("val"))}
    if token.endswith("*min_side"):
        return {"relative_to": "min_side", "value": float(token[:-9])}
    if token.endswith("*width"):
        return {"relative_to": "width", "value": float(token[:-6])}
    if token.endswith("*height"):
        return {"relative_to": "height", "value": float(token[:-7])}
    if token.endswith("*max_side"):
        return {"relative_to": "max_side", "value": float(token[:-9])}
    try:
        return float(token)
    except ValueError:
        return token


def _parse_call(text: str) -> tuple[str, dict[str, Any], list[Any]]:
    text = text.strip()
    if "(" not in text:
        return text, {}, []
    name, rest = text.split("(", 1)
    if not rest.endswith(")"):
        raise ValueError(f"unclosed call: {text}")
    inner = rest[:-1].strip()
    if not inner:
        return name.strip(), {}, []
    kwargs: dict[str, Any] = {}
    args: list[Any] = []
    for part in _split_args(inner):
        if "=" in part:
            k, v = part.split("=", 1)
            kwargs[k.strip()] = _parse_value(v.strip())
        else:
            args.append(_parse_value(part.strip()))
    return name.strip(), kwargs, args


def _split_args(inner: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in inner:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_value(token: str) -> Any:
    token = token.strip()
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return token[1:-1]
    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        return _parse_scalar(token)


def _parse_window(text: str) -> dict[str, Any]:
    name, kw, args = _parse_call(text)
    shape = name
    params: dict[str, Any] = dict(kw)

    if shape == "rectangle":
        if args:
            params.setdefault("width", args[0])
            if len(args) > 1:
                params.setdefault("height", args[1])
        params.setdefault("width", params.pop("w", params.get("width", 1.0)))
        params.setdefault("height", params.pop("h", params.get("height", 1.0)))
    elif shape == "polygon":
        if args:
            if len(args) == 1:
                params.setdefault("sides", int(args[0]))
            else:
                params.setdefault("sides", int(args[0]))
                params.setdefault("width", args[1])
                if len(args) > 2:
                    params.setdefault("height", args[2])
        sides = int(params.get("sides", 6))
        size = params.pop("size", None)
        if size is not None:
            params.setdefault("width", size)
            params.setdefault("height", size)
        params.setdefault("width", params.get("width", 1.0))
        params.setdefault("height", params.get("height", params["width"]))
        params["sides"] = sides
    elif shape == "head_body":
        if "body" in params:
            params["body_height"] = params.pop("body")
        params.setdefault("width", 1.0)
        params.setdefault("body_height", params.get("body_height", 1.0))
        params.setdefault("head", "semicircle")
        params.setdefault("head_height", params.get("head_height", float(params["width"]) * 0.5))
        head = str(params.get("head", "semicircle"))
        if head in {"triangle", "pointed", "pointed_arch"}:
            params["head"] = "trapezoid"
            params["top_width"] = 0.0
        elif head == "trapezoid":
            w = float(params["width"])
            params.setdefault("top_width", params.get("head_top_width", w * 0.65))
    elif shape in {"arch_head", "rect_eyebrow", "springline_arch"}:
        if "body" in params:
            params["body_height"] = params.pop("body")
        params.setdefault("width", 1.0)
        params.setdefault("body_height", params.get("body_height", 1.0))
        params.setdefault("rise", params.pop("arch_height", params.get("rise", float(params["width"]) * 0.18)))
    elif shape == "circle":
        size = params.pop("size", params.pop("diameter", args[0] if args else 1.0))
        params["diameter"] = size
    elif shape == "ellipse":
        if args:
            params.setdefault("width", args[0])
            if len(args) > 1:
                params.setdefault("height", args[1])
    elif shape == "semicircle":
        params.setdefault("diameter", params.pop("size", params.get("diameter", 1.0)))
        params.setdefault("flat", "bottom")
    elif shape in {"eyebrow", "segmental_arch"}:
        params.setdefault("width", params.pop("size", params.get("width", 1.0)))
        params.setdefault(
            "rise",
            params.pop("arch_height", params.get("rise", float(params["width"]) * 0.2)),
        )
    elif shape in {"trapezoid", "trapezoid_head"}:
        params.setdefault("width", params.pop("size", params.get("width", 1.0)))
        w = float(params["width"])
        params.setdefault("height", params.pop("rise", params.get("height", w * 0.35)))
        params.setdefault("top_width", params.get("head_top_width", w * 0.65))
    elif shape == "triangle":
        w = float(params.get("width", 1.0))
        h = float(params.get("height", 1.0))
        shape = "trapezoid"
        params = {"width": w, "height": h, "top_width": 0.0}
    elif shape == "quadrant":
        params.setdefault("radius", params.pop("size", params.get("radius", 1.0)))
        params.setdefault("corner", "bl")

    return {"id": "root", "shape": shape, "params": params}


def _parse_frame(text: str) -> dict[str, Any]:
    tokens = text.split()
    frame: dict[str, Any] = {"material": "painted_wood"}
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key == "thickness" and i + 1 < len(tokens):
            frame["thickness"] = _parse_scalar(tokens[i + 1])
            i += 2
            continue
        if key == "depth" and i + 1 < len(tokens):
            frame["depth"] = _parse_scalar(tokens[i + 1])
            i += 2
            continue
        if key == "material" and i + 1 < len(tokens):
            frame["material"] = tokens[i + 1]
            i += 2
            continue
        i += 1
    if "thickness" not in frame:
        frame["thickness"] = {"relative_to": "min_side", "value": 0.06}
    if "depth" not in frame:
        frame["depth"] = 0.08
    return frame


def _parse_regions(lines: list[str]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(f"bad region line: {line!r}")
        rid, rhs = stripped.split("=", 1)
        rid = rid.strip()
        rhs = rhs.strip()
        name, kw, args = _parse_call(rhs)
        op = name
        entry: dict[str, Any] = {"id": rid, "operation": op}

        if op == "inset":
            entry["from"] = args[0] if args else kw.get("from", "root")
            if len(args) > 1:
                amount = args[1]
            elif "amount" in kw:
                amount = kw["amount"]
            else:
                amount = 0.06
            if isinstance(amount, str):
                amount = _parse_scalar(amount)
            entry["amount"] = amount
        elif op == "crop":
            entry["from"] = args[0] if args else kw.get("from", "glass")
            if len(args) >= 5:
                u0, v0, u1, v1 = (float(args[i]) for i in range(1, 5))
            else:
                u0 = float(kw.get("u0", 0.0))
                v0 = float(kw.get("v0", 0.0))
                u1 = float(kw.get("u1", 1.0))
                v1 = float(kw.get("v1", kw.get("v", 1.0)))
            entry["bounds"] = {"u0": u0, "v0": v0, "u1": u1, "v1": v1}
        elif op == "body":
            entry["from"] = args[0] if args else kw.get("from", "glass")
        elif op == "head":
            entry["from"] = args[0] if args else kw.get("from", "glass")
        else:
            raise ValueError(f"unknown region operation: {op!r}")
        regions.append(entry)
    return regions


def _parse_pane_line(line: str, pane_spec: dict[str, Any]) -> None:
    stripped = line.strip()
    if not stripped:
        return
    if stripped.lower().startswith("@root"):
        pane_spec["root"] = stripped.split(None, 1)[1].strip() if " " in stripped else "root"
        return

    m = _EPS_RULE.match(stripped)
    if m:
        name = m.group(1)
        if name in pane_spec["rules"] and pane_spec["rules"][name].get("op") != "eps":
            raise ValueError(
                f"pane rule {name!r} already defined as {pane_spec['rules'][name]!r}; "
                f"cannot overwrite with eps"
            )
        pane_spec["rules"][name] = {"op": "eps"}
        return

    m = _SPLIT_RULE.match(stripped)
    if m:
        name, axis, at, rest = m.group(1), m.group(2).lower(), float(m.group(3)), m.group(4)
        if name in pane_spec["rules"]:
            raise ValueError(f"duplicate pane rule: {name!r}")
        children = [normalize_pane_name(t) for t in rest.split()]
        pane_spec["rules"][name] = {"op": "split", "axis": axis, "at": at, "children": children}
        return

    m = _GRID_RULE.match(stripped)
    if m:
        name, n_v, n_h, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).strip()
        children = [normalize_pane_name(t) for t in rest.split()] if rest else []
        pane_spec["rules"][name] = {
            "op": "grid",
            "vertical": n_v,
            "horizontal": n_h,
            "children": children,
        }
        return

    # Indented tree: split v=0.5  (handled via _parse_pane_tree if line starts with split)
    if stripped.lower().startswith("split "):
        raise ValueError(
            "indented pane trees are not on this line — use rule form: "
            f"NAME split(v=0.5) childA childB  ({stripped!r})"
        )

    raise ValueError(f"bad pane rule: {line!r}")


def _parse_split_radial(rest: list[str], op: str) -> dict[str, Any]:
    if not rest:
        raise ValueError(f"{op} needs a region")
    region = rest[0]
    step: dict[str, Any] = {"op": "split_radial", "id": "radial", "region": region}
    if op == "fan":
        step.update({"ox": 0.5, "oy": 0.0, "arc_start": 45, "arc_end": 135, "count": 3})
    i = 1
    if op == "fan" and i < len(rest) and rest[i].isdigit():
        step["count"] = int(rest[i])
        i += 1
    while i < len(rest):
        key = rest[i].lower()
        if key == "from_arc" and i + 1 < len(rest):
            step["from_arc"] = rest[i + 1]
            i += 2
            continue
        if key == "from" and i + 1 < len(rest) and not _looks_numeric(rest[i + 1]):
            step["from_arc"] = rest[i + 1]
            i += 2
            continue
        if key in {"origin", "from"} and i + 2 < len(rest):
            step["ox"] = float(rest[i + 1])
            step["oy"] = float(rest[i + 2])
            i += 3
            continue
        if key == "count" and i + 1 < len(rest):
            step["count"] = int(rest[i + 1])
            i += 2
            continue
        if key == "arc" and i + 2 < len(rest):
            step["arc_start"] = float(rest[i + 1])
            step["arc_end"] = float(rest[i + 2])
            i += 3
            continue
        if key == "angles":
            angles: list[float] = []
            i += 1
            while i < len(rest) and _looks_numeric(rest[i]):
                angles.append(float(rest[i]))
                i += 1
            step["angles"] = angles
            continue
        if key == "offset" and i + 1 < len(rest):
            step["start_radius"] = _parse_scalar(rest[i + 1])
            i += 2
            continue
        if key == "rotate" and i + 1 < len(rest):
            step["angle_offset"] = math.radians(float(rest[i + 1]))
            i += 2
            continue
        raise ValueError(f"bad {op} token: {rest[i]!r}")
    return step


_ARC_MUNTIN_KEYS = {
    "origin",
    "center",
    "at",
    "radius",
    "rise",
    "from",
    "to",
    "semicircle",
    "half",
    "eyebrow",
    "segments",
}


def _parse_arc_muntin(rest: list[str]) -> list[dict[str, Any]]:
    if not rest:
        raise ValueError("arc needs a region")
    region = rest[0]
    step: dict[str, Any] = {
        "op": "insert_shape",
        "shape": "arc",
        "id": "arc",
        "region": region,
        "cx": 0.5,
        "cy": 0.0,
        "radius": {"relative_to": "min_side", "value": 0.2},
        "angle_start": 0.0,
        "angle_end": math.pi,
        "segments": 16,
    }
    i = 1
    if i < len(rest) and rest[i].lower() not in _ARC_MUNTIN_KEYS:
        step["id"] = rest[i]
        i += 1
    while i < len(rest):
        key = rest[i].lower()
        if key in {"origin", "center", "at"} and i + 2 < len(rest):
            step["cx"] = float(rest[i + 1])
            step["cy"] = float(rest[i + 2])
            i += 3
            continue
        if key == "radius" and i + 1 < len(rest):
            step["radius"] = _parse_scalar(rest[i + 1])
            i += 2
            continue
        if key == "rise" and i + 1 < len(rest):
            step["rise"] = _parse_scalar(rest[i + 1])
            step.pop("radius", None)
            i += 2
            continue
        if key == "eyebrow" and i + 1 < len(rest):
            step["rise"] = _parse_scalar(rest[i + 1])
            step.pop("radius", None)
            i += 2
            continue
        if key == "from" and i + 1 < len(rest):
            step["angle_start"] = math.radians(float(rest[i + 1]))
            i += 2
            continue
        if key == "to" and i + 1 < len(rest):
            step["angle_end"] = math.radians(float(rest[i + 1]))
            i += 2
            continue
        if key in {"semicircle", "half"}:
            step["angle_start"] = 0.0
            step["angle_end"] = math.pi
            i += 1
            continue
        if key == "segments" and i + 1 < len(rest):
            step["segments"] = int(rest[i + 1])
            i += 2
            continue
        raise ValueError(f"bad arc token: {rest[i]!r}")
    return [step]


def _parse_lunette_program(rest: list[str]) -> list[dict[str, Any]]:
    """Expand lunette to hub arc + fan (compositional alias)."""
    if not rest:
        raise ValueError("lunette needs a region")
    region = rest[0]
    hub_radius: Any = {"relative_to": "min_side", "value": 0.16}
    fan: dict[str, Any] = {"arc_start": 45, "arc_end": 135, "count": 3}
    i = 1
    while i < len(rest):
        key = rest[i].lower()
        if key == "hub" and i + 1 < len(rest):
            hub_radius = _parse_scalar(rest[i + 1])
            i += 2
            continue
        if key == "count" and i + 1 < len(rest):
            fan["count"] = int(rest[i + 1])
            i += 2
            continue
        if key == "arc" and i + 2 < len(rest):
            fan["arc_start"] = float(rest[i + 1])
            fan["arc_end"] = float(rest[i + 2])
            i += 3
            continue
        raise ValueError(f"bad lunette token: {rest[i]!r}")
    arc_step: dict[str, Any] = {
        "op": "insert_shape",
        "shape": "arc",
        "id": "hub",
        "region": region,
        "cx": 0.5,
        "cy": 0.0,
        "radius": hub_radius,
        "angle_start": 0.0,
        "angle_end": math.pi,
        "segments": 16,
    }
    fan_step = _parse_split_radial([region], "fan")
    fan_step.update(fan)
    fan_step["from_arc"] = "hub"
    return [arc_step, fan_step]


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_program(lines: list[str]) -> list[dict[str, Any]]:
    program: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        op = tokens[0].lower()
        rest = tokens[1:]

        if op in {"grid", "add_grid"}:
            region = rest[0]
            n_v, n_h = 1, 1
            if len(rest) > 1:
                m = re.match(r"(\d+)\s*x\s*(\d+)", rest[1], re.I)
                if not m:
                    raise ValueError(f"grid size must be NxM, got {rest[1]!r}")
                n_v, n_h = int(m.group(1)), int(m.group(2))
            program.append(
                {
                    "op": "add_grid",
                    "id": "grid",
                    "region": region,
                    "vertical": n_v,
                    "horizontal": n_h,
                }
            )
            continue

        if op in {"split_v", "split_vertical"}:
            region = rest[0]
            x = _parse_coord(rest[1:] if len(rest) > 1 else [], "x", 0.5)
            program.append({"op": "split_vertical", "id": "v_split", "region": region, "x": x})
            continue

        if op in {"split_h", "split_horizontal"}:
            region = rest[0]
            y = _parse_coord(rest[1:] if len(rest) > 1 else [], "y", 0.5)
            program.append({"op": "split_horizontal", "id": "h_split", "region": region, "y": y})
            continue

        if op == "seam":
            program.append(_parse_seam(rest))
            continue

        if op in {"split_radial", "radial", "fan"}:
            program.append(_parse_split_radial(rest, op))
            continue

        if op == "arc":
            program.extend(_parse_arc_muntin(rest))
            continue

        if op == "lunette":
            program.extend(_parse_lunette_program(rest))
            continue

        if op == "connect":
            if len(rest) < 3:
                raise ValueError(f"connect needs id from to [region]: {stripped!r}")
            cid, a0, a1 = rest[0], rest[1], rest[2]
            region = rest[3] if len(rest) > 3 else "glass"
            program.append({"op": "connect", "id": cid, "from": a0, "to": a1, "region": region})
            continue

        if op == "anchor":
            if len(rest) < 4:
                raise ValueError(f"anchor needs id region u v: {stripped!r}")
            program.append(
                {
                    "op": "define_anchor",
                    "id": rest[0],
                    "region": rest[1],
                    "kind": "point",
                    "u": float(rest[2]),
                    "v": float(rest[3]),
                }
            )
            continue

        if op == "mirror":
            axis = rest[0] if rest else "y"
            region = rest[1] if len(rest) > 1 else "glass"
            program.append({"op": "mirror", "axis": axis, "region": region})
            continue

        raise ValueError(f"unknown program op: {op!r}")

    return program


def _parse_seam(rest: list[str]) -> dict[str, Any]:
    """Horizontal muntin on a zone edge. `seam body` = top of body (= head/body junction)."""
    if not rest:
        raise ValueError("seam needs a region")
    region = rest[0]
    edge = rest[1].lower() if len(rest) > 1 else None
    if edge is None:
        y = 0.0 if region == "head" else 1.0
    elif edge in {"top", "upper"}:
        y = 1.0
    elif edge in {"bottom", "lower", "base"}:
        y = 0.0
    else:
        raise ValueError(f"seam edge must be top or bottom, got {edge!r}")
    return {"op": "split_horizontal", "id": "seam", "region": region, "y": y}


def _parse_coord(tokens: list[str], axis: str, default: float) -> Any:
    for tok in tokens:
        low = tok.lower()
        if axis == "y" and low == "top":
            return 1.0
        if axis == "y" and low == "bottom":
            return 0.0
        if axis == "x" and low in {"left", "l"}:
            return 0.0
        if axis == "x" and low in {"right", "r"}:
            return 1.0
        if tok.startswith(f"{axis}="):
            return _parse_scalar(tok.split("=", 1)[1])
    if tokens:
        low = tokens[0].lower()
        if axis == "y" and low in {"top", "bottom"}:
            return 1.0 if low == "top" else 0.0
        if axis == "x" and low in {"left", "right"}:
            return 0.0 if low == "left" else 1.0
        return _parse_scalar(tokens[0])
    return default


def _parse_output(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        key = tokens[0]
        out[key] = _parse_scalar(tokens[1])
    if "muntin_width" not in out:
        out["muntin_width"] = {"relative_to": "min_side", "value": 0.025}
    if "muntin_depth" not in out:
        out["muntin_depth"] = 0.04
    if "glass_thickness" not in out:
        out["glass_thickness"] = 0.01
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse WDSL to JSON window spec")
    ap.add_argument("wdsl_file", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    spec = parse_wdsl_file(args.wdsl_file)
    text = json.dumps(spec, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
