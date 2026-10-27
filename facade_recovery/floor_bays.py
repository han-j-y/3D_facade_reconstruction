"""Per-floor bay grid and side-facade filter.

Every floor keeps its own bays (one per window / door, split at gap midpoints). Split lines that match across
floors become shared lines; a floor's empty stretches (missing windows) inherit shared lines from the nearest
floor with similar window height. The Blender compiler reads one floors x columns grid, so
:func:`apply_floor_bays` makes the columns the union of all floors' lines and every floor bay a span over them.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from facade_recovery.column_layout import assign_floors


def assign_door_floors(W: np.ndarray, D: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Doors join the window floor they overlap most vertically (>= 0.3 of the door height); the rest form
    ground floors of their own, below the window floors."""
    nf = int(floor.max()) + 1 if len(floor) else 0
    spans = [(W[floor == f, 1].min(), W[floor == f, 3].max()) for f in range(nf)]
    out: list[int | None] = []
    for d in D:
        ov = [max(0.0, min(d[3], s1) - max(d[1], s0)) / max(d[3] - d[1], 1.0) for s0, s1 in spans]
        out.append(int(np.argmax(ov)) if ov and max(ov) >= 0.3 else None)
    # doors left over: doors overlapping each other vertically (>= 0.3 of the shorter) share a new floor
    groups: list[list[int]] = []
    for k in sorted((k for k in range(len(D)) if out[k] is None), key=lambda k: D[k][1]):
        for g in groups:
            s0, s1 = min(D[j][1] for j in g), max(D[j][3] for j in g)
            if min(D[k][3], s1) - max(D[k][1], s0) >= 0.3 * min(D[k][3] - D[k][1], s1 - s0):
                g.append(k)
                break
        else:
            groups.append([k])
    for gi, g in enumerate(groups):
        for k in g:
            out[k] = nf + gi
    return np.asarray(out, dtype=np.int32)


def build_floor_bays(
    boxes: list,
    iw: int,
    ih: int,
    *,
    doors: list = (),
    row_tol: float = 0.055,
    tol_frac: float = 0.4,
    min_piece: float = 0.6,
) -> dict[str, Any]:
    """Per-floor bay lines for window ``boxes`` and ``doors`` (xyxy px).

    Returns ``floor`` (per item, windows then doors, floors ranked top -> bottom), ``nf``, ``lines`` (per floor
    ``(x, kind)``, kind own | shared | inherited), ``bays`` (per floor ``{x0, x1, win}``, ``win`` = item indices),
    ``shared``, ``B`` (all items) and ``is_door``."""
    W = np.asarray(boxes, float).reshape(-1, 4)
    D = np.asarray(doors, float).reshape(-1, 4)
    wfloor = assign_floors(0.5 * (W[:, 1] + W[:, 3]) / ih, row_tol)
    dfloor = assign_door_floors(W, D, wfloor) if len(D) else np.zeros(0, dtype=np.int32)
    B = np.concatenate([W, D])
    is_door = np.array([False] * len(W) + [True] * len(D))
    floor = np.concatenate([wfloor, dfloor]).astype(np.int32)
    cy = 0.5 * (B[:, 1] + B[:, 3])
    rank = {f: k for k, f in enumerate(sorted(set(floor.tolist()), key=lambda f: cy[floor == f].mean()))}
    floor = np.array([rank[f] for f in floor], dtype=np.int32)
    nf = int(floor.max()) + 1
    w = B[:, 2] - B[:, 0]
    tol = max(6.0, tol_frac * float(np.median(w[~is_door])))
    rows = [sorted(np.where(floor == f)[0], key=lambda i: B[i, 0] + B[i, 2]) for f in range(nf)]

    h = B[:, 3] - B[:, 1]
    # door-only floors have no window height to compare: they accept lines from any floor
    med_h = [float(np.median(h[[i for i in r if not is_door[i]]])) if any(not is_door[i] for i in r) else None
             for r in rows]

    def cuts(x: float, f: int) -> bool:
        return any(B[i, 0] + 0.2 * w[i] < x < B[i, 2] - 0.2 * w[i] for i in rows[f])

    # 1) each floor: one bay per window, lines at gap midpoints
    cands = sorted((0.5 * (B[a, 2] + B[b, 0]), f) for f in range(nf) for a, b in zip(rows[f], rows[f][1:]))
    # 2) lines from different floors that match become one shared line
    clusters: list[list[tuple[float, int]]] = []
    for x, f in cands:
        if clusters and x - np.mean([c[0] for c in clusters[-1]]) <= tol:
            clusters[-1].append((x, f))
        else:
            clusters.append([(x, f)])
    shared = []
    lines: list[list[tuple[float, str]]] = [[] for _ in range(nf)]
    for cl in clusters:
        pos = float(np.median([c[0] for c in cl]))
        fl = {c[1] for c in cl}
        if len(fl) >= 2:
            shared.append(dict(x=pos, floors=fl))
        for x, f in cl:
            if len(fl) >= 2 and not cuts(pos, f):
                lines[f].append((pos, "shared"))
            else:
                lines[f].append((x, "own"))

    # 3) empty stretches of a floor (missing windows) take shared lines from the nearest floor with similar
    #    window height; spacing is judged by the source floor's pitch (this floor's own pitch is inflated
    #    exactly when windows are missing, cmp_b0198 mezzanine), and a farther floor only fills the part
    #    of the stretch a nearer floor left uncovered
    def row_pitch(r: list[int]) -> float | None:
        p = [0.5 * (B[b, 0] + B[b, 2] - B[a, 0] - B[a, 2]) for a, b in zip(r, r[1:])]
        return float(np.median(p)) if p else None

    all_p = [p for p in map(row_pitch, rows) if p]
    pitch_glob = float(np.median(all_p)) if all_p else float(iw)
    src = [list(l) for l in lines]
    for f in range(nf):
        edges = [0.0] + [v for i in rows[f] for v in (B[i, 0], B[i, 2])] + [float(iw)]
        for g0, g1 in zip(edges[::2], edges[1::2]):
            own, added, covered = None, 0, []
            for g in sorted((g for g in range(nf) if g != f), key=lambda g: (abs(g - f), g)):
                if med_h[g] and med_h[f] and not 0.5 <= med_h[g] / med_h[f] <= 2.0:
                    continue
                pg = row_pitch(rows[g]) or pitch_glob
                S = [x for x, kind in src[g] if kind != "own" and g0 < x < g1
                     and not any(c0 <= x <= c1 for c0, c1 in covered)]
                if not S:
                    continue
                mine = [l for l in lines[f] if l[1] == "shared" and g0 < l[0] < g1]
                if all(any(abs(x - l[0]) <= tol for l in mine) for x in S):
                    covered.append((min(S) - 0.5 * pg, max(S) + 0.5 * pg))
                    continue
                if own is None:
                    # this floor's own gap-midpoint line gives way to the nearest floor's lines
                    own = [l for l in lines[f] if l[1] != "inherited" and g0 < l[0] < g1]
                    for l in own:
                        lines[f].remove(l)
                    keep = [l for l in own if l[1] == "shared" and any(abs(x - l[0]) <= tol for x in S)]
                    lines[f].extend(keep)
                    own = [l for l in own if l not in keep]
                    S = [x for x in S if not any(abs(x - l[0]) <= tol for l in keep)]
                acc = []
                for x in S:
                    near_win = (g0 > 0 and x - g0 < 0.1 * pg) or (g1 < iw and g1 - x < 0.1 * pg)
                    pieces_ok = all(abs(x - x2) >= min_piece * pg for x2, _ in lines[f] if g0 < x2 < g1)
                    if pieces_ok and not near_win:
                        lines[f].append((x, "inherited"))
                        acc.append(x)
                if acc:
                    covered.append((min(acc) - 0.5 * pg, max(acc) + 0.5 * pg))
                    added += len(acc)
            if own and not added:
                lines[f].extend(own)
    bays = []
    for f in range(nf):
        lines[f].sort()
        xs = [0.0] + [x for x, _ in lines[f]] + [float(iw)]
        bays.append([dict(x0=b0, x1=b1, win=[i for i in rows[f] if b0 <= 0.5 * (B[i, 0] + B[i, 2]) < b1])
                     for b0, b1 in zip(xs[:-1], xs[1:])])
    return dict(floor=floor, nf=nf, lines=lines, bays=bays, shared=shared, B=B, is_door=is_door)


def drop_side_facades(
    boxes: list,
    iw: int,
    ih: int,
    *,
    doors: list = (),
    core: tuple[float, float] = (0.25, 0.75),
    min_aligned: float = 0.5,
) -> list[int]:
    """Indices of ``boxes`` on side facades (neighbouring buildings, side walls) at the image edges.

    Main floors: floors of the windows and doors whose centre lies in the central ``core`` fraction of the image
    width, each with its median top / bottom. A window sits on a main floor if it is 0.5-2x as tall as that floor's
    typical window and either its centre is within 0.3 x that height of the floor's centre or it overlaps the
    floor vertically by >= 0.7 of the shorter height (balcony doors reach lower than the windows). Columns:
    windows chained by x-overlap. From each edge inward, a column is a side facade unless more than
    ``min_aligned`` of its windows sit on a main floor or it is attached (one window per main floor)."""
    if len(boxes) < 4:
        return []
    B = np.asarray(boxes, float)
    A = np.concatenate([B, np.asarray(doors, float).reshape(-1, 4)])
    cx = 0.5 * (B[:, 0] + B[:, 2])
    acx, acy = 0.5 * (A[:, 0] + A[:, 2]), 0.5 * (A[:, 1] + A[:, 3])
    inner = np.where((acx >= core[0] * iw) & (acx <= core[1] * iw))[0]
    if len(inner) < 3:
        return []
    fl = assign_floors(acy[inner] / ih, 0.055)
    bands = [(float(np.median(A[inner[fl == f], 1])), float(np.median(A[inner[fl == f], 3])))
             for f in set(fl.tolist())]

    def on_floor(b: np.ndarray) -> bool:
        h, c = b[3] - b[1], 0.5 * (b[1] + b[3])
        for y0, y1 in bands:
            hb = y1 - y0
            level = abs(c - 0.5 * (y0 + y1)) <= 0.3 * hb or min(b[3], y1) - max(b[1], y0) >= 0.7 * min(h, hb)
            if level and 0.5 <= h / hb <= 2.0:
                return True
        return False

    aligned = np.array([on_floor(b) for b in B])
    med_w = float(np.median(B[:, 2] - B[:, 0]))

    def attached(seq: list[list[int]], k: int) -> bool:
        """Part of the same building though its floors are offset (cmp_b0215 corner tower): a single outermost
        column next to a main-facade column, >= 3 windows each on a different main floor (overlap >= 0.25,
        median >= 0.5), not foreshortened like a side wall seen at an angle."""
        c = seq[k]
        if k != 0 or len(seq) < 2 or aligned[seq[1]].mean() <= min_aligned:
            return False
        if len(c) < 3 or np.median(B[c, 2] - B[c, 0]) < 0.85 * med_w:
            return False
        hit, best = [], []
        for b in B[c]:
            ov = [(min(b[3], y1) - max(b[1], y0)) / min(b[3] - b[1], y1 - y0) for y0, y1 in bands]
            hit.append(int(np.argmax(ov)))
            best.append(max(ov))
        return min(best) >= 0.25 and np.median(best) >= 0.5 and len(set(hit)) == len(hit)

    cols: list[list[int]] = []
    for i in np.argsort(cx):
        if cols:
            x0, x1 = B[cols[-1], 0].min(), B[cols[-1], 2].max()
            if min(x1, B[i, 2]) - max(x0, B[i, 0]) >= 0.3 * min(x1 - x0, B[i, 2] - B[i, 0]):
                cols[-1].append(int(i))
                continue
        cols.append([int(i)])
    side: set[int] = set()
    for seq in (cols, cols[::-1]):
        for k, c in enumerate(seq):
            if aligned[c].mean() > min_aligned or core[0] * iw <= cx[c].mean() <= core[1] * iw or attached(seq, k):
                break
            side.update(c)
    return sorted(side)


def _floor_rows(B: np.ndarray, floor: np.ndarray, is_door: np.ndarray, nf: int, ih: int) -> list[tuple[float, float]]:
    """Row bounds (image y) at the midpoints between floors' median window extents."""
    top, bot = [], []
    for f in range(nf):
        m = floor == f
        src = (m & ~is_door) if (m & ~is_door).any() else m
        top.append(float(np.median(B[src, 1])))
        bot.append(float(np.median(B[src, 3])))
    gaps = [max(0.0, top[f + 1] - bot[f]) for f in range(nf - 1)]
    half = 0.5 * float(np.median(gaps)) if gaps else 0.1 * (bot[0] - top[0])
    splits = [0.5 * (bot[f] + top[f + 1]) for f in range(nf - 1)]
    y0 = max(0.0, min(top[0] - half, B[floor == 0, 1].min() - 2))
    y1 = min(float(ih), max(bot[-1] + half, B[floor == nf - 1, 3].max() + 2))
    edges = [y0] + splits + [y1]
    return [(edges[f], edges[f + 1]) for f in range(nf)]


def apply_floor_bays(dsl: dict[str, Any], *, storey_m: float = 3.0, row_tol: float = 0.055) -> dict[str, Any]:
    """Replace a recovery DSL's layout with the per-floor bay grid of its window and door instances.

    Columns are the union of all floors' bay lines inside the facade extent (item envelope + half the typical
    gap); each floor bay spans the columns between its lines. Floors / bays carry ``h_m`` / ``w_m`` so the
    compiler keeps thin union columns as they are (it widens any column under 5% of the facade otherwise).
    Instance ``floor`` / ``bay`` / ``colspan`` are rewritten; the per-floor bays go to ``meta.floor_bays``."""
    dsl = copy.deepcopy(dsl)
    iw, ih = dsl["meta"]["image_size"]
    wins = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
    doors = [u for u in dsl["instances"] if u.get("kind") == "door"]
    G = build_floor_bays([u["box_xyxy"] for u in wins], iw, ih, doors=[u["box_xyxy"] for u in doors],
                         row_tol=row_tol)
    B, floor, is_door, nf = G["B"], G["floor"], G["is_door"], G["nf"]
    items = wins + doors

    item_gaps = []
    for f in range(nf):
        r = sorted(np.where(floor == f)[0], key=lambda i: B[i, 0])
        item_gaps += [B[b, 0] - B[a, 2] for a, b in zip(r, r[1:]) if B[b, 0] > B[a, 2]]
    half = 0.5 * float(np.median(item_gaps)) if item_gaps else 0.1 * float(np.median(B[:, 2] - B[:, 0]))
    left = max(0.0, float(B[:, 0].min()) - half)
    right = min(float(iw), float(B[:, 2].max()) + half)

    floor_lines = [[x for x, _ in G["lines"][f] if left + 2 < x < right - 2] for f in range(nf)]
    union: list[float] = []
    for x in sorted(x for L in floor_lines for x in L):
        if not union or x - union[-1] > 2.0:
            union.append(x)
    bounds = [left] + union + [right]
    cols = list(zip(bounds[:-1], bounds[1:]))

    def col_of(x: float) -> int:
        return int(np.argmin([abs(x - b) for b in bounds]))

    placement: list[list[str | None]] = [[None] * len(cols) for _ in range(nf)]
    spans = [[1] * len(cols) for _ in range(nf)]
    placed, floor_bays = [], []
    for f in range(nf):
        fl = [left] + floor_lines[f] + [right]
        row = []
        for b0, b1 in zip(fl[:-1], fl[1:]):
            c0, c1 = col_of(b0), col_of(b1)
            here = [int(i) for i in np.where(floor == f)[0] if b0 <= 0.5 * (B[i, 0] + B[i, 2]) < b1]
            row.append(dict(x0=round(b0, 1), x1=round(b1, 1), c0=c0, c1=c1,
                            item=("door" if is_door[here[0]] else "window") if here else "empty"))
            if not here:
                continue
            u = items[here[0]]
            tok = str(u.get("name") or "d00") if is_door[here[0]] else f"win_type_{int(u['type_id']):02d}"
            u.update(floor=f, bay=c0, colspan=c1 - c0)
            placed.append(u)
            placement[f][c0] = tok
            spans[f][c0] = c1 - c0
            for c in range(c0 + 1, c1):
                spans[f][c] = 0
        floor_bays.append(row)

    rows = _floor_rows(B, floor, is_door, nf, ih)
    s = storey_m * nf / (rows[-1][1] - rows[0][0])
    dsl["instances"] = placed
    dsl["layout"] = {
        "floors": [{"name": f"F{f}", "id": f, "h_norm": (y1 - y0) / ih, "h_m": round((y1 - y0) * s, 4)}
                   for f, (y0, y1) in enumerate(rows)],
        "bays": [{"name": f"B{c}", "id": c, "w_norm": (x1 - x0) / iw, "w_m": round((x1 - x0) * s, 4)}
                 for c, (x0, x1) in enumerate(cols)],
        "placement": placement,
        "placement_spans": spans,
    }
    m = dsl["meta"]
    for k in ("inter_bay_gaps_px", "inter_bay_gap_norm", "inter_floor_gaps_px", "inter_floor_gap_norm"):
        m.pop(k, None)
    m.update(n_doors=sum(u.get("kind") == "door" for u in placed), layout="floor_bays", columns_xy=[[round(a, 1), round(b, 1)] for a, b in cols],
             floors_y=[[round(a, 1), round(b, 1)] for a, b in rows], floor_bays=floor_bays,
             notes="Per-floor bays (facade_recovery/floor_bays.py): columns = union of all floors' bay lines, "
                   "each floor bay spans its columns; h_m / w_m in metres (storey_m per floor on average).")
    return dsl
