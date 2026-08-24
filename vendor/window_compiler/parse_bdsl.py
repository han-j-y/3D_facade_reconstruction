#!/usr/bin/env python3
"""Parse BDSL (balcony DSL) into JSON IR, optionally wrapped as a 1-cell façade.

  python vendor/window_compiler/parse_bdsl.py vendor/window_compiler/examples/example_balcony_projecting.bdsl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_RELPAT = re.compile(
    r"^\s*(?P<val>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*\*\s*(?P<rel>min_side|max_side|width|height)\s*$",
    re.I,
)
_SPLIT_RULE = re.compile(
    r"^(\w+)\s+split\s*\(\s*([vh])\s*=\s*([\d.]+)\s*\)\s+(.+)$",
    re.I,
)
_GRID_RULE = re.compile(r"^(\w+)\s+grid\s*\(\s*(\d+)\s*x\s*(\d+)\s*\)\s*(.*)$", re.I)
_EPS_RULE = re.compile(r"^(\w+)\s+eps\s*$", re.I)
_PANE_WRAP = re.compile(r"^pane\((.+)\)$", re.I)

FLOOR_SHAPES = frozenset({"rectangle", "circle", "triangle"})
STRUCTURES = frozenset({"projecting", "inset", "composite", "free_standing"})
ENCLOSURES = frozenset({"open", "enclosed"})
RAIL_KINDS = frozenset({"baluster", "solid", "glass"})
RAIL_ALIASES = {"metal": "baluster"}
OPENINGS = frozenset({"door", "window", "none"})
BLOCK_NAMES = frozenset({"floor", "railing", "supports", "glazing", "output"})


def parse_bdsl(text: str) -> dict[str, Any]:
    lines = _keep_indented_code_lines(text)
    ir: dict[str, Any] = {
        "type": "balcony",
        "debug": False,
        "structure": "projecting",
        "enclosure": "open",
        "floor": {"shape": "rectangle", "params": {"width": 1.0, "depth": 0.8}},
        "railing": {"kind": "baluster", "height": 1.1},
        "supports": {"count": 0},
        "opening": "door",
        "glazing": None,
        "output": {"slab_thickness": 0.12, "railing_thickness": 0.04},
    }
    saw_balcony = False
    i = 0
    while i < len(lines):
        raw, stripped = lines[i]
        if stripped == "debug":
            ir["debug"] = True
            i += 1
            continue
        if stripped.endswith(":") and stripped[:-1] in BLOCK_NAMES:
            name = stripped[:-1]
            body, i = _take_block(lines, i + 1)
            _apply_block(ir, name, body)
            continue
        head = stripped.split(None, 1)[0]
        rest = stripped.split(None, 1)[1] if " " in stripped else ""
        if head == "balcony":
            ir["floor"] = _parse_balcony_line(rest)
            saw_balcony = True
        elif head == "structure":
            name = rest.strip()
            if name not in STRUCTURES:
                raise ValueError(f"unknown structure {name!r}")
            ir["structure"] = name
        elif head == "enclosure":
            name = rest.strip()
            if name not in ENCLOSURES:
                raise ValueError(f"unknown enclosure {name!r}")
            ir["enclosure"] = name
        elif head == "railing":
            ir["railing"] = _parse_railing_line(rest, ir["railing"])
        elif head == "opening":
            name = rest.strip()
            if name not in OPENINGS:
                raise ValueError(f"unknown opening {name!r}")
            ir["opening"] = name
        else:
            raise ValueError(f"unexpected BDSL line: {stripped!r}")
        i += 1
    if not saw_balcony:
        raise ValueError("BDSL requires a balcony line")
    if ir["enclosure"] != "enclosed":
        ir["glazing"] = None
    return ir


def parse_bdsl_file(path: str | Path) -> dict[str, Any]:
    return parse_bdsl(Path(path).read_text(encoding="utf-8-sig"))


def to_facade_spec(ir: dict[str, Any], *, name: str = "balcony") -> dict[str, Any]:
    """Wrap one balcony IR as a 1-cell façade the compiler can mesh."""
    params = (ir.get("floor") or {}).get("params") or {}
    width = float(params.get("width") or params.get("diameter") or 1.8)
    depth = float(params.get("depth") or 0.9)
    structure = str(ir.get("structure") or "projecting")
    height = float(params.get("height") or 0.0)
    enclosure = str(ir.get("enclosure") or "open")
    if structure in ("inset", "composite"):
        void_h = height if height > 0.05 else 2.5
        col_w = max(4.0, width + 2.2)
        row_h = max(4.0, void_h + 1.5)
        bottom = 0.12
    elif enclosure == "enclosed":
        clear = height if height > 0.05 else 2.5
        slab_t = float((ir.get("output") or {}).get("slab_thickness") or 0.20)
        col_w = max(2.4, width + 0.6)
        row_h = max(4.5, clear + 2.0 * slab_t + 1.6)
        bottom = 0.12
    elif structure == "free_standing":
        col_w = max(2.4, width + 0.4)
        deck = height if height > 0.05 else 1.2
        row_h = max(3.4, deck + 1.8)
        bottom = 0.08
    else:
        col_w = max(2.4, width + 0.4)
        row_h = 3.4
        bottom = 0.14
    return {
        "type": "facade",
        "schema": "window_compiler_facade_v1",
        "wall": {"depth": 0.42, "base_front_y": 0.0, "material": "wall"},
        "grid": {
            "rows": [{"name": "F0", "h": row_h}],
            "cols": [{"name": "B0", "w": col_w}],
        },
        "windows": {},
        "placement": [[None]],
        "balconies": {name: ir},
        "balcony_placement": [{"row": 0, "col0": 0, "col1": 0, "type": name}],
        "placement_params": {
            "width_ratio": 0.55,
            "height_ratio": 0.60,
            "bottom_margin_ratio": bottom,
            "recess": 0.01,
            "mirror_x": True,
        },
    }


def parse_bdsl_as_facade(path: str | Path, *, name: str | None = None) -> dict[str, Any]:
    p = Path(path)
    ir = parse_bdsl_file(p)
    return to_facade_spec(ir, name=name or p.stem)


def _keep_indented_code_lines(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        code = raw.split("#", 1)[0].rstrip()
        if code.strip():
            out.append((code, code.strip()))
    return out


def _indent(raw: str) -> int:
    return len(raw) - len(raw.lstrip(" "))


def _take_block(lines: list[tuple[str, str]], start: int) -> tuple[list[tuple[str, str]], int]:
    body: list[tuple[str, str]] = []
    i = start
    while i < len(lines):
        raw, stripped = lines[i]
        if _indent(raw) == 0:
            break
        body.append((raw, stripped))
        i += 1
    return body, i


def _apply_block(ir: dict[str, Any], name: str, body: list[tuple[str, str]]) -> None:
    if name == "floor":
        ir["floor"] = _parse_floor_block(body, ir["floor"])
    elif name == "railing":
        ir["railing"] = _parse_railing_block(body, ir["railing"])
    elif name == "supports":
        ir["supports"] = _parse_supports_block(body, ir["supports"])
    elif name == "glazing":
        text = "\n".join(_dedent([raw for raw, _ in body]))
        ir["glazing"] = _parse_nested_wdsl(text)
    elif name == "output":
        ir["output"] = _parse_output_block(body, ir["output"])


def _dedent(raws: list[str]) -> list[str]:
    if not raws:
        return []
    pad = min(_indent(r) for r in raws if r.strip())
    return [r[pad:] if len(r) >= pad else r for r in raws]


def _parse_scalar(token: str) -> Any:
    token = token.strip().rstrip(",")
    if not token:
        raise ValueError("empty scalar")
    m = _RELPAT.match(token)
    if m:
        return {"relative_to": m.group("rel"), "value": float(m.group("val"))}
    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token) if token.lstrip("-").isdigit() else float(token)
    except ValueError:
        return token


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
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    return _parse_scalar(token)


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


def _parse_balcony_line(text: str) -> dict[str, Any]:
    shape, kw, args = _parse_call(text)
    if shape not in FLOOR_SHAPES:
        raise ValueError(f"unknown floor shape {shape!r}")
    params = dict(kw)
    if shape == "circle":
        if args:
            params.setdefault("diameter", args[0])
            if len(args) > 1:
                params.setdefault("depth", args[1])
        params.setdefault("diameter", params.pop("size", params.get("diameter", 1.0)))
        params.setdefault("depth", params.pop("d", params.get("depth", 0.8)))
        params["width"] = params["diameter"]
    else:
        if args:
            params.setdefault("width", args[0])
            if len(args) > 1:
                params.setdefault("depth", args[1])
            if len(args) > 2:
                params.setdefault("height", args[2])
        params.setdefault("width", params.pop("w", params.get("width", 1.0)))
        params.setdefault("depth", params.pop("d", params.get("depth", 0.8)))
        if "h" in params and "height" not in params:
            params["height"] = params.pop("h")
    return {"shape": shape, "params": params}


def _normalize_rail_kind(kind: str) -> str:
    k = RAIL_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if k not in RAIL_KINDS:
        raise ValueError(f"unknown railing kind {kind!r}")
    return k


def _parse_railing_line(text: str, current: dict[str, Any]) -> dict[str, Any]:
    tokens = text.split()
    out = dict(current)
    if tokens and (tokens[0] in RAIL_KINDS or tokens[0] in RAIL_ALIASES):
        out["kind"] = _normalize_rail_kind(tokens[0])
        tokens = tokens[1:]
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key == "height" and i + 1 < len(tokens):
            out["height"] = _parse_scalar(tokens[i + 1])
            i += 2
            continue
        if key == "material" and i + 1 < len(tokens):
            out["material"] = tokens[i + 1]
            i += 2
            continue
        i += 1
    return out


def _parse_floor_block(body: list[tuple[str, str]], current: dict[str, Any]) -> dict[str, Any]:
    out = {"shape": current.get("shape", "rectangle"), "params": dict(current.get("params") or {})}
    for _, stripped in body:
        key, _, rest = stripped.partition(" ")
        key = key.lower()
        if key == "shape":
            shape = rest.strip()
            if shape not in FLOOR_SHAPES:
                raise ValueError(f"unknown floor shape {shape!r}")
            out["shape"] = shape
        elif key in {"width", "depth", "height", "diameter", "kind"}:
            if key == "kind":
                out["params"]["kind"] = rest.strip()
            else:
                out["params"][key] = _parse_scalar(rest)
        else:
            raise ValueError(f"unknown floor field {stripped!r}")
    return out


def _parse_railing_block(body: list[tuple[str, str]], current: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    for _, stripped in body:
        key, _, rest = stripped.partition(" ")
        key = key.lower()
        if key == "kind":
            out["kind"] = _normalize_rail_kind(rest.strip())
        elif key == "height":
            out["height"] = _parse_scalar(rest)
        elif key == "material":
            out["material"] = rest.strip()
        else:
            raise ValueError(f"unknown railing field {stripped!r}")
    return out


def _parse_supports_block(body: list[tuple[str, str]], current: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    for _, stripped in body:
        key, _, rest = stripped.partition(" ")
        key = key.lower()
        if key == "count":
            out["count"] = int(_parse_scalar(rest))
        elif key == "style":
            out["style"] = rest.strip()
        elif key == "spacing":
            out["spacing"] = _parse_scalar(rest)
        else:
            raise ValueError(f"unknown supports field {stripped!r}")
    return out


def _parse_output_block(body: list[tuple[str, str]], current: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    for _, stripped in body:
        key, _, rest = stripped.partition(" ")
        out[key] = _parse_scalar(rest)
    return out


def _normalize_pane_name(token: str) -> str:
    token = token.strip()
    m = _PANE_WRAP.match(token)
    return m.group(1) if m else token


def _parse_nested_wdsl(text: str) -> dict[str, Any]:
    """Subset of WDSL used inside glazing: (window / frame / regions / panes / output)."""
    spec: dict[str, Any] = {"type": "window", "debug": False}
    raw_lines = [ln for ln in text.splitlines() if ln.split("#", 1)[0].strip()]
    i = 0
    while i < len(raw_lines):
        stripped = raw_lines[i].split("#", 1)[0].strip()
        if stripped.endswith(":") and stripped[:-1] in {"regions", "output"}:
            name = stripped[:-1]
            body: list[str] = []
            i += 1
            while i < len(raw_lines) and (
                raw_lines[i].startswith(" ") or raw_lines[i].startswith("\t")
            ):
                body.append(raw_lines[i])
                i += 1
            if name == "regions":
                spec["regions"] = _parse_wdsl_regions(body)
            else:
                spec["output"] = _parse_wdsl_output(body)
            continue
        if stripped.startswith("panes"):
            parts = stripped.rstrip(":").split()
            region = parts[1] if len(parts) > 1 else "glass"
            pane_spec: dict[str, Any] = {"region": region, "rules": {}, "root": "root"}
            i += 1
            while i < len(raw_lines) and (
                raw_lines[i].startswith(" ") or raw_lines[i].startswith("\t")
            ):
                _parse_pane_line(raw_lines[i].strip(), pane_spec)
                i += 1
            spec["panes"] = pane_spec
            continue
        head = stripped.split(None, 1)[0]
        rest = stripped.split(None, 1)[1] if " " in stripped else ""
        if head == "window":
            spec["boundary"] = _parse_wdsl_window(rest)
        elif head == "frame":
            spec["frame"] = _parse_wdsl_frame(rest)
        else:
            raise ValueError(f"unexpected WDSL in glazing: {stripped!r}")
        i += 1
    if "boundary" not in spec:
        raise ValueError("glazing: WDSL requires a window line")
    return spec


def _parse_wdsl_window(text: str) -> dict[str, Any]:
    name, kw, args = _parse_call(text)
    params = dict(kw)
    if name == "rectangle":
        if args:
            params.setdefault("width", args[0])
            if len(args) > 1:
                params.setdefault("height", args[1])
        params.setdefault("width", params.pop("w", params.get("width", 1.0)))
        params.setdefault("height", params.pop("h", params.get("height", 1.0)))
    return {"id": "root", "shape": name, "params": params}


def _parse_wdsl_frame(text: str) -> dict[str, Any]:
    tokens = text.split()
    frame: dict[str, Any] = {"material": "painted_wood"}
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key in {"thickness", "depth"} and i + 1 < len(tokens):
            frame[key] = _parse_scalar(tokens[i + 1])
            i += 2
            continue
        if key == "material" and i + 1 < len(tokens):
            frame["material"] = tokens[i + 1]
            i += 2
            continue
        i += 1
    return frame


def _parse_wdsl_regions(lines: list[str]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if "=" not in stripped:
            raise ValueError(f"bad region line: {line!r}")
        rid, rhs = stripped.split("=", 1)
        name, kw, args = _parse_call(rhs.strip())
        entry: dict[str, Any] = {"id": rid.strip(), "operation": name}
        if name == "inset":
            entry["from"] = args[0] if args else kw.get("from", "root")
            amount = args[1] if len(args) > 1 else kw.get("amount", 0.06)
            if isinstance(amount, str):
                amount = _parse_scalar(amount)
            entry["amount"] = amount
        regions.append(entry)
    return regions


def _parse_wdsl_output(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in lines:
        stripped = line.strip()
        key, _, rest = stripped.partition(" ")
        out[key] = _parse_scalar(rest)
    return out


def _parse_pane_line(line: str, pane_spec: dict[str, Any]) -> None:
    stripped = line.strip()
    if not stripped:
        return
    if stripped.lower().startswith("@root"):
        pane_spec["root"] = stripped.split(None, 1)[1].strip() if " " in stripped else "root"
        return
    m = _EPS_RULE.match(stripped)
    if m:
        pane_spec["rules"][m.group(1)] = {"op": "eps"}
        return
    m = _SPLIT_RULE.match(stripped)
    if m:
        children = [_normalize_pane_name(t) for t in m.group(4).split()]
        pane_spec["rules"][m.group(1)] = {
            "op": "split",
            "axis": m.group(2).lower(),
            "at": float(m.group(3)),
            "children": children,
        }
        return
    m = _GRID_RULE.match(stripped)
    if m:
        rest = m.group(4).strip()
        children = [_normalize_pane_name(t) for t in rest.split()] if rest else []
        pane_spec["rules"][m.group(1)] = {
            "op": "grid",
            "vertical": int(m.group(2)),
            "horizontal": int(m.group(3)),
            "children": children,
        }
        return
    raise ValueError(f"bad pane rule: {line!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bdsl_file", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--facade", action="store_true", help="wrap as 1-cell façade JSON")
    args = ap.parse_args()
    ir = parse_bdsl_file(args.bdsl_file)
    data = to_facade_spec(ir, name=args.bdsl_file.stem) if args.facade else ir
    text = json.dumps(data, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
