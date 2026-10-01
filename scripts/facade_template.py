#!/usr/bin/env python3
"""Floor / bay templates for recovery façade DSLs.

extract      repaired DSL  -> editable template JSON
             (bay symbols + floor archetypes + per-cell window type and in-cell fit)
instantiate  template JSON -> synthetic recovery DSL at any floor × bay count

Bay widths and floor heights are stored in meters and written back as ``w_m`` /
``h_m`` so adding bays or floors grows the building instead of squeezing it.

  python scripts/facade_template.py extract --dsl runs/X/cmp_b0226/facade_dsl_repaired.json \
      --out runs/facade_templates/cmp_b0226.template.json

  python scripts/facade_template.py instantiate --template runs/facade_templates/cmp_b0226.template.json \
      --floors "top typical*3 ground" --bays "E M*3 D M*3 E" \
      --out runs/facade_templates/cmp_b0226_5x9/facade_dsl_repaired.json

  python scripts/facade_template.py instantiate --template ... --add-floors 2 --add-bays 2 --out ...

Template cell values: window/door type name, ``null`` (empty), ``{"type": T, "span": k}``
(window covering k bays starting here) or ``"^"`` (covered by a span to the left).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import string
from pathlib import Path
from statistics import median
from typing import Any

EMPTY = (None, "", "empty")
COVERED = "^"
PX_PER_M = 100.0
STOREY_HEIGHT = 3.0  # must match render_facade.py default (--storey-height)


# ----------------------------------------------------------------------------- extract


def _world_sizes(dsl: dict) -> tuple[list[float], list[float]]:
    """Floor heights / bay widths in meters, mirroring normalize_facade_spec."""
    layout, meta = dsl["layout"], dsl.get("meta") or {}
    floors, bays = layout["floors"], layout["bays"]
    if all("h_m" in f for f in floors) and all("w_m" in b for b in bays):
        return [float(f["h_m"]) for f in floors], [float(b["w_m"]) for b in bays]
    h_n = [max(0.05, float(f.get("h_norm", 1.0))) for f in floors]
    w_n = [max(0.05, float(b.get("w_norm", 1.0))) for b in bays]
    total_h = STOREY_HEIGHT * len(floors)
    iw, ih = meta.get("image_size") or [1, 1]
    aspect = max(0.25, float(iw) / max(1.0, float(ih)))
    cols, rows = meta.get("columns_xy"), meta.get("floors_y")
    if cols and rows:
        cw = float(cols[-1][1]) - float(cols[0][0])
        ch = float(rows[-1][1]) - float(rows[0][0])
        if cw > 1.0 and ch > 1.0:
            aspect = max(0.25, cw / ch)
    total_w = total_h * aspect
    return (
        [total_h * h / sum(h_n) for h in h_n],
        [total_w * w / sum(w_n) for w in w_n],
    )


def _cell_fit(box: list[float], left: float, top: float, right: float, bottom: float) -> dict:
    x0, y0, x1, y1 = (float(v) for v in box)
    cw, ch = max(1.0, right - left), max(1.0, bottom - top)
    clamp = lambda v, lo, hi: min(hi, max(lo, v))  # noqa: E731
    return {
        "width_ratio": clamp((x1 - x0) / cw, 0.05, 0.98),
        "height_ratio": clamp((y1 - y0) / ch, 0.05, 0.98),
        "cx_ratio": clamp((0.5 * (x0 + x1) - left) / cw, 0.02, 0.98),
        "cy_ratio": clamp((0.5 * (y0 + y1) - top) / ch, 0.02, 0.98),
    }


def _cell_value(tok: Any, span: int) -> Any:
    if tok in EMPTY:
        return COVERED if span == 0 else None
    if span > 1:
        return {"type": str(tok), "span": int(span)}
    return str(tok)


def _cell_type(val: Any) -> str | None:
    if isinstance(val, dict):
        return val.get("type")
    if val in EMPTY or val == COVERED:
        return None
    return str(val)


def _symbol_names(n: int) -> list[str]:
    pool = [c for c in string.ascii_uppercase if c not in "ED"]
    return [pool[i] if i < len(pool) else f"X{i}" for i in range(n)]


def extract_template(dsl: dict, src_path: str = "") -> dict:
    layout, meta = dsl["layout"], dsl.get("meta") or {}
    floors, bays = layout["floors"], layout["bays"]
    placement = layout.get("placement") or []
    spans = layout.get("placement_spans") or []
    n_f, n_b = len(floors), len(bays)
    h_m, w_m = _world_sizes(dsl)

    grid = [
        [
            _cell_value(
                placement[r][c] if r < len(placement) and c < len(placement[r]) else None,
                int(spans[r][c]) if r < len(spans) and c < len(spans[r]) and spans[r][c] is not None
                else (0 if (placement[r][c] if r < len(placement) and c < len(placement[r]) else None) in EMPTY else 1),
            )
            for c in range(n_b)
        ]
        for r in range(n_f)
    ]

    # --- bay symbols: columns with the same top-to-bottom cell signature share a symbol
    col_sig = [json.dumps([grid[r][c] for r in range(n_f)]) for c in range(n_b)]
    sig_to_sym: dict[str, str] = {}
    order: list[str] = []
    for s in col_sig:
        if s not in sig_to_sym:
            order.append(s)
            sig_to_sym[s] = ""
    names = _symbol_names(len(order))
    for s, nm in zip(order, names):
        sig_to_sym[s] = nm
    # friendly names: outermost columns -> E, door column(s) -> D
    edge_sig = col_sig[0] if n_b >= 3 and col_sig[0] == col_sig[-1] else None
    door_names = {str(d.get("name")) for d in dsl.get("door_types") or []}
    door_sigs = [s for s in order if any(_cell_type(v) in door_names for v in json.loads(s))]
    if edge_sig is not None:
        sig_to_sym[edge_sig] = "E"
    if len(door_sigs) == 1 and door_sigs[0] != edge_sig:
        sig_to_sym[door_sigs[0]] = "D"
    bay_pattern = [sig_to_sym[s] for s in col_sig]

    bay_defs: dict[str, dict] = {}
    for sym in dict.fromkeys(bay_pattern):
        ws = [w_m[c] for c in range(n_b) if bay_pattern[c] == sym]
        bay_defs[sym] = {"w_m": round(sum(ws) / len(ws), 4)}

    # --- floor archetypes: rows with identical cells (per symbol) merge
    def row_cells(r: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in range(n_b):
            out.setdefault(bay_pattern[c], grid[r][c])
        return out

    row_sig = [json.dumps(row_cells(r), sort_keys=True) for r in range(n_f)]
    arch_of_sig: dict[str, str] = {}
    floor_pattern: list[str] = []
    mid_count = 0
    for r in range(n_f):
        s = row_sig[r]
        if s in arch_of_sig:
            floor_pattern.append(arch_of_sig[s])
            continue
        if r == n_f - 1:
            nm = "ground"
        elif r == 0:
            nm = "top"
        else:
            mid_count += 1
            nm = "typical" if mid_count == 1 else f"mid{mid_count}"
        arch_of_sig[s] = nm
        floor_pattern.append(nm)

    # --- per (archetype, symbol) fit + door shape from segmentation instances
    cols_px, rows_px = meta.get("columns_xy"), meta.get("floors_y")
    fid_to_row = {int(f["id"]): i for i, f in enumerate(floors)}
    acc: dict[tuple[str, str], list[dict]] = {}
    door_shape: dict[tuple[str, str], dict] = {}
    for inst in dsl.get("instances") or []:
        r = fid_to_row.get(int(inst["floor"]))
        c = int(inst["bay"])
        if r is None or not (0 <= c < n_b) or not cols_px or not rows_px:
            continue
        span = max(1, int(inst.get("colspan", 1)))
        end = min(n_b, c + span) - 1
        fit = _cell_fit(
            inst["box_xyxy"],
            float(cols_px[c][0]), float(rows_px[r][0]),
            float(cols_px[end][1]), float(rows_px[r][1]),
        )
        key = (floor_pattern[r], bay_pattern[c])
        acc.setdefault(key, []).append(fit)
        if inst.get("kind") == "door" and inst.get("shape"):
            door_shape.setdefault(key, {
                "shape": inst["shape"],
                "arch_rise_ratio": float(inst.get("arch_rise_ratio") or 0.0),
            })

    archetypes: dict[str, dict] = {}
    for r, nm in enumerate(floor_pattern):
        if nm in archetypes:
            continue
        rows_of = [i for i, a in enumerate(floor_pattern) if a == nm]
        cells = row_cells(r)
        fits: dict[str, dict] = {}
        for sym in cells:
            pts = acc.get((nm, sym))
            if not pts:
                continue
            f = {k: round(median(p[k] for p in pts), 4) for k in pts[0]}
            f.update(door_shape.get((nm, sym), {}))
            fits[sym] = f
        archetypes[nm] = {
            "h_m": round(sum(h_m[i] for i in rows_of) / len(rows_of), 4),
            "cells": cells,
            "fits": fits,
        }

    return {
        "schema": "facade_template_v1",
        "source": {
            "dsl": src_path,
            "facade_id": meta.get("facade_id"),
            "image": meta.get("image"),
            "floors": n_f,
            "bays": n_b,
        },
        "bay_pattern": " ".join(bay_pattern),
        "floor_pattern": " ".join(floor_pattern),
        "repeat_floor": _default_repeat_floor(floor_pattern),
        "repeat_bay": _default_repeat_bay(bay_pattern, archetypes),
        "bays": bay_defs,
        "floors": archetypes,
        "window_types": copy.deepcopy(dsl.get("window_types") or []),
        "door_types": copy.deepcopy(dsl.get("door_types") or []),
    }


def _default_repeat_floor(pattern: list[str]) -> str:
    if "typical" in pattern:
        return "typical"
    non_ground = [p for p in pattern if p != "ground"]
    return (non_ground or pattern)[-1]


def _default_repeat_bay(pattern: list[str], archetypes: dict[str, dict]) -> str:
    """Most common interior symbol, widened to cover any multi-bay window starting there."""
    interior = pattern[1:-1] if len(pattern) > 2 else pattern
    counts: dict[str, int] = {}
    for s in interior:
        if s != "D":
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        counts = {s: interior.count(s) for s in interior}
    sym = max(counts, key=lambda s: (counts[s], -interior.index(s)))
    span = max(
        (int(a["cells"][sym].get("span", 1)) for a in archetypes.values()
         if isinstance(a["cells"].get(sym), dict)),
        default=1,
    )
    start = pattern.index(sym, 1 if len(pattern) > 2 else 0)
    return " ".join(pattern[start : start + span])


# ----------------------------------------------------------------------------- instantiate


def parse_pattern(text: str) -> list[str]:
    """``"E M*3 (F G)*2 E"`` -> ``[E, M, M, M, F, G, F, G, E]``."""
    out: list[str] = []
    text = text.replace(",", " ")
    pos = 0
    for m in re.finditer(r"\s*(?:\(([^()]*)\)|([A-Za-z_][\w-]*))(?:\*(\d+))?\s*", text):
        if m.start() != pos:
            break
        pos = m.end()
        unit = m.group(1).split() if m.group(1) is not None else [m.group(2)]
        out.extend(unit * int(m.group(3) or 1))
    if pos != len(text):
        raise SystemExit(f"bad pattern near {text[pos:]!r} (use NAME, NAME*k, (A B)*k)")
    return out


def add_floors(pattern: list[str], n: int, repeat: str) -> list[str]:
    if n <= 0:
        return pattern
    if repeat not in pattern:
        raise SystemExit(f"repeat floor {repeat!r} not in floor pattern {pattern}")
    i = len(pattern) - 1 - pattern[::-1].index(repeat)
    return pattern[: i + 1] + [repeat] * n + pattern[i + 1 :]


def _unit_start(p: list[str], unit: list[str], near: int) -> int:
    """Index of the unit occurrence closest to ``near`` (fallback: ``near``)."""
    k = len(unit)
    hits = [i for i in range(len(p) - k + 1) if p[i : i + k] == unit]
    return min(hits, key=lambda i: abs(i + k / 2 - near)) if hits else near


def add_bays(pattern: list[str], n: int, repeat: list[str]) -> list[str]:
    """Insert ``n`` copies of the ``repeat`` unit next to the copy nearest the center.

    Inserting whole units right before an existing copy keeps multi-bay windows
    (``{"span": k}`` cells) intact, and even ``n`` keeps a symmetric façade symmetric
    when the center copies are part of a periodic run.
    """
    if n <= 0:
        return list(pattern)
    p, k = list(pattern), len(repeat)
    hits = [i for i in range(len(p) - k + 1) if p[i : i + k] == repeat]
    left = [i for i in hits if i + k <= len(p) / 2]
    if p == p[::-1] and left and n >= 2:
        i = max(left)
        j = len(p) - i  # mirror of "before the unit at i"
        half = n // 2
        p = p[:j] + repeat * half + p[j:]
        return p[:i] + repeat * (n - half) + p[i:]
    at = _unit_start(p, repeat, len(p) // 2)
    return p[:at] + repeat * n + p[at:]


def instantiate(tpl: dict, floor_pattern: list[str], bay_pattern: list[str], facade_id: str) -> dict:
    floors_def, bays_def = tpl["floors"], tpl["bays"]
    for a in floor_pattern:
        if a not in floors_def:
            raise SystemExit(f"unknown floor archetype {a!r}; known: {list(floors_def)}")
    for s in bay_pattern:
        if s not in bays_def:
            raise SystemExit(f"unknown bay symbol {s!r}; known: {list(bays_def)}")

    win_ids = {str(w.get("name") or f"win_type_{int(w['type_id']):02d}"): int(w["type_id"])
               for w in tpl.get("window_types") or []}
    door_ids = {str(d.get("name") or f"d{int(d['type_id']):02d}"): int(d["type_id"])
                for d in tpl.get("door_types") or []}

    n_f, n_b = len(floor_pattern), len(bay_pattern)
    h_m = [float(floors_def[a]["h_m"]) for a in floor_pattern]
    w_m = [float(bays_def[s]["w_m"]) for s in bay_pattern]
    total_h, total_w = sum(h_m), sum(w_m)

    rows_px, y = [], 0.0
    for h in h_m:
        rows_px.append([round(y, 2), round(y + h * PX_PER_M, 2)])
        y += h * PX_PER_M
    cols_px, x = [], 0.0
    for w in w_m:
        cols_px.append([round(x, 2), round(x + w * PX_PER_M, 2)])
        x += w * PX_PER_M

    placement: list[list[str | None]] = []
    spans: list[list[int]] = []
    instances: list[dict] = []
    n_counts: dict[str, int] = {}
    warned: set[str] = set()
    for r, arch in enumerate(floor_pattern):
        cells, fits = floors_def[arch]["cells"], floors_def[arch].get("fits") or {}
        prow: list[str | None] = [None] * n_b
        srow: list[int] = [0] * n_b
        c = 0
        while c < n_b:
            sym = bay_pattern[c]
            val = cells.get(sym)
            tok = _cell_type(val)
            span = int(val.get("span", 1)) if isinstance(val, dict) else 1
            span = max(1, min(span, n_b - c))
            if tok is None:
                srow[c] = 1
                c += 1
                continue
            if tok not in win_ids and tok not in door_ids:
                if tok not in warned:
                    print(f"warn: unknown type {tok!r} in archetype {arch!r}; left empty")
                    warned.add(tok)
                srow[c] = 1
                c += 1
                continue
            prow[c], srow[c] = tok, span
            fit = fits.get(sym) or {}
            left, right = cols_px[c][0], cols_px[c + span - 1][1]
            top, bottom = rows_px[r]
            cw, ch = right - left, bottom - top
            bw = float(fit.get("width_ratio", 0.55)) * cw
            bh = float(fit.get("height_ratio", 0.6)) * ch
            cx = left + float(fit.get("cx_ratio", 0.5)) * cw
            cy = top + float(fit.get("cy_ratio", 0.5)) * ch
            box = [round(cx - bw / 2, 2), round(cy - bh / 2, 2), round(cx + bw / 2, 2), round(cy + bh / 2, 2)]
            inst: dict[str, Any] = {
                "unit_id": f"r{r}_c{c}",
                "box_xyxy": box,
                "floor": r,
                "bay": c,
                "colspan": span,
            }
            if tok in door_ids:
                inst.update({
                    "kind": "door",
                    "type_id": door_ids[tok],
                    "name": tok,
                    "shape": fit.get("shape", "rectangle"),
                    "arch_rise_ratio": float(fit.get("arch_rise_ratio", 0.0)),
                })
            else:
                inst.update({"kind": "window", "type_id": win_ids[tok]})
            instances.append(inst)
            n_counts[tok] = n_counts.get(tok, 0) + 1
            c += span
        placement.append(prow)
        spans.append(srow)

    window_types = copy.deepcopy(tpl.get("window_types") or [])
    for w in window_types:
        w["n_instances"] = n_counts.get(str(w.get("name")), 0)
    door_types = copy.deepcopy(tpl.get("door_types") or [])
    for d in door_types:
        d["n_instances"] = n_counts.get(str(d.get("name")), 0)

    src = tpl.get("source") or {}
    return {
        "schema": "facade_recovery_dsl_v1",
        "meta": {
            "facade_id": facade_id,
            "synthetic": True,
            "source_facade_id": src.get("facade_id"),
            "source_image": src.get("image"),
            "floor_pattern": " ".join(floor_pattern),
            "bay_pattern": " ".join(bay_pattern),
            "image_size": [int(round(total_w * PX_PER_M)), int(round(total_h * PX_PER_M))],
            "px_per_m": PX_PER_M,
            "n_units": sum(1 for i in instances if i["kind"] == "window"),
            "n_types": len(window_types),
            "n_doors": sum(1 for i in instances if i["kind"] == "door"),
            "n_door_types": len(door_types),
            "columns_xy": cols_px,
            "floors_y": rows_px,
            "notes": "Synthetic DSL instantiated from a floor/bay template (facade_template.py).",
        },
        "layout": {
            "floors": [
                {"name": f"F{r}", "id": r, "h_norm": round(h / total_h, 6), "h_m": round(h, 4),
                 "archetype": floor_pattern[r]}
                for r, h in enumerate(h_m)
            ],
            "bays": [
                {"name": f"B{c}", "id": c, "w_norm": round(w / total_w, 6), "w_m": round(w, 4),
                 "symbol": bay_pattern[c]}
                for c, w in enumerate(w_m)
            ],
            "placement": placement,
            "placement_spans": spans,
        },
        "window_types": window_types,
        "door_types": door_types,
        "instances": instances,
    }


# ----------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="repaired DSL -> template JSON")
    ex.add_argument("--dsl", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)

    ins = sub.add_parser("instantiate", help="template JSON -> synthetic DSL")
    ins.add_argument("--template", type=Path, required=True)
    ins.add_argument("--floors", default=None, help='e.g. "top typical*3 ground" (top to bottom)')
    ins.add_argument("--bays", default=None, help='e.g. "E M*3 D M*3 E" (left to right)')
    ins.add_argument("--add-floors", type=int, default=0)
    ins.add_argument("--add-bays", type=int, default=0,
                     help="number of repeat_bay units to insert (a unit is >1 bay when a window spans it)")
    ins.add_argument("--repeat-floor", default=None, help="archetype used by --add-floors")
    ins.add_argument("--repeat-bay", default=None, help='bay unit used by --add-bays, e.g. "M" or "F G"')
    ins.add_argument("--facade-id", default=None)
    ins.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "extract":
        dsl = json.loads(args.dsl.read_text(encoding="utf-8"))
        tpl = extract_template(dsl, str(args.dsl))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(tpl, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        print(f"  bays   : {tpl['bay_pattern']}   (repeat_bay={tpl['repeat_bay']})")
        print(f"  floors : {tpl['floor_pattern']}   (repeat_floor={tpl['repeat_floor']})")
        for nm, a in tpl["floors"].items():
            print(f"  {nm:>8} h={a['h_m']:.2f}m  " + "  ".join(f"{s}:{_cell_type(v) or ('^' if v == COVERED else '-')}" for s, v in a["cells"].items()))
        return

    tpl = json.loads(args.template.read_text(encoding="utf-8"))
    fp = parse_pattern(args.floors or tpl["floor_pattern"])
    bp = parse_pattern(args.bays or tpl["bay_pattern"])
    fp = add_floors(fp, args.add_floors, args.repeat_floor or tpl.get("repeat_floor") or fp[-1])
    bp = add_bays(bp, args.add_bays, parse_pattern(args.repeat_bay or tpl.get("repeat_bay") or bp[len(bp) // 2]))
    src_id = (tpl.get("source") or {}).get("facade_id") or "facade"
    fid = args.facade_id or f"{src_id}_{len(fp)}x{len(bp)}"
    dsl = instantiate(tpl, fp, bp, fid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dsl, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  floors ({len(fp)}): {' '.join(fp)}")
    print(f"  bays   ({len(bp)}): {' '.join(bp)}")
    print(f"  size   : {dsl['meta']['image_size'][0] / PX_PER_M:.2f} × {dsl['meta']['image_size'][1] / PX_PER_M:.2f} m, "
          f"{dsl['meta']['n_units']} windows, {dsl['meta']['n_doors']} doors")


if __name__ == "__main__":
    main()
