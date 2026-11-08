"""Floor / bay order lists: add, delete and reorder floors and bays with 1-2 words of DSL.

A recovery DSL may carry ``layout.floor_order`` (floor ids, top -> bottom) and ``layout.bay_order`` (bay ids,
left -> right). An id may appear more than once (a copy) or not at all (deleted). ``apply_grid_order`` lays the
floors and bays out again in that order, in photo pixels, before the DSL is compiled:

* each floor keeps its pixel height (band + the gap below it), each bay its width (band + the gap right of it);
* every element (window / door instance, open balcony, enclosed cell) moves by the offset of its floor row and of
  its first bay column, so its place inside its cell is unchanged; photo norms (``*_norm``, fractions of image
  width) move with it;
* the first occurrence of an id keeps the id, later occurrences are copies with fresh floor / bay / unit ids
  (a copied balcony keeps the window it hangs under, as that window's copy);
* without order lists, or with the identity order, the DSL is returned unchanged.

The compiler is not changed: it receives an ordinary recovery DSL.
"""

from __future__ import annotations

import copy
from typing import Any

NORM_X = ("cx_norm", "window_cx_norm", "bay_cx_norm")


def _bands(meta: dict[str, Any], key: str, n: int, size: float, norms: list[float]) -> list[tuple[float, float]]:
    raw = meta.get(key)
    if isinstance(raw, list) and len(raw) == n:
        return [(float(a), float(b)) for a, b in raw]
    edges, acc = [], 0.0
    tot = sum(norms) or 1.0
    for v in norms:
        edges.append((acc / tot * size, (acc + v) / tot * size))
        acc += v
    return edges


def _pitch(bands: list[tuple[float, float]]) -> list[float]:
    """Band size plus the gap after it (the last band keeps its own size)."""
    out = []
    for i, (a, b) in enumerate(bands):
        out.append((bands[i + 1][0] - a) if i + 1 < len(bands) else (b - a))
    return out


def _layout(order: list[int], pos_of: dict[int, int], bands, pitch, start: float):
    """New band per order slot, laid out from ``start`` with each source's pitch."""
    new, at = [], start
    for i in order:
        p = pos_of[i]
        a, b = bands[p]
        new.append((at, at + (b - a)))
        at += pitch[p]
    return new, at


def apply_grid_order(dsl: dict[str, Any]) -> dict[str, Any]:
    lay = dsl.get("layout") or {}
    f_order, b_order = lay.get("floor_order"), lay.get("bay_order")
    floors, bays = lay.get("floors") or [], lay.get("bays") or []
    f_ids, b_ids = [int(f["id"]) for f in floors], [int(b["id"]) for b in bays]
    f_order = [int(i) for i in f_order] if f_order is not None else f_ids
    b_order = [int(i) for i in b_order] if b_order is not None else b_ids
    if f_order == f_ids and b_order == b_ids:
        return dsl  # the recovered order: nothing to lay out (the lists are ignored downstream)
    unknown = (set(f_order) - set(f_ids)) | (set(b_order) - set(b_ids))
    if unknown:
        raise ValueError(f"order lists name unknown ids {sorted(unknown)}")

    meta = dsl.get("meta") or {}
    iw, ih = (float(v) for v in (meta.get("image_size") or [1000, 1000])[:2])
    fpos = {i: k for k, i in enumerate(f_ids)}
    bpos = {i: k for k, i in enumerate(b_ids)}
    fb = _bands(meta, "floors_y", len(floors), ih, [float(f.get("h_norm", 1.0)) for f in floors])
    bb = _bands(meta, "columns_xy", len(bays), iw, [float(b.get("w_norm", 1.0)) for b in bays])
    new_fb, f_end = _layout(f_order, fpos, fb, _pitch(fb), fb[0][0] if fb else 0.0)
    new_bb, b_end = _layout(b_order, bpos, bb, _pitch(bb), bb[0][0] if bb else 0.0)
    old_f_end = fb[-1][1] if fb else 0.0
    old_b_end = bb[-1][1] if bb else 0.0
    d_h = (new_fb[-1][1] if new_fb else 0.0) - old_f_end
    d_w = (new_bb[-1][1] if new_bb else 0.0) - old_b_end
    niw, nih = iw + d_w, ih + d_h

    # slots: (source id, new id); the first occurrence keeps the id
    def slots(order: list[int], ids: list[int]) -> list[tuple[int, int]]:
        nxt, seen, out = max(ids) + 1 if ids else 0, set(), []
        for i in order:
            if i in seen:
                out.append((i, nxt))
                nxt += 1
            else:
                seen.add(i)
                out.append((i, i))
        return out

    f_slots, b_slots = slots(f_order, f_ids), slots(b_order, b_ids)

    out = copy.deepcopy(dsl)
    olay = out["layout"]
    olay.pop("floor_order", None)
    olay.pop("bay_order", None)
    fdef = {int(f["id"]): f for f in floors}
    bdef = {int(b["id"]): b for b in bays}
    olay["floors"] = []
    for (src, nid), (a, b) in zip(f_slots, new_fb):
        f = copy.deepcopy(fdef[src])
        f["id"], f["h_norm"] = nid, round((b - a) / max(1e-6, nih), 6)
        if nid != src:
            f["name"] = f"F{nid}"
        olay["floors"].append(f)
    olay["bays"] = []
    for (src, nid), (a, b) in zip(b_slots, new_bb):
        bd = copy.deepcopy(bdef[src])
        bd["id"], bd["w_norm"] = nid, round((b - a) / max(1e-6, niw), 6)
        if nid != src:
            bd["name"] = f"B{nid}"
        olay["bays"].append(bd)

    def remap_matrix(m: list | None) -> list | None:
        if not isinstance(m, list) or not m:
            return m
        return [[copy.deepcopy(m[fpos[fs]][bpos[bs]]) if fpos[fs] < len(m) and bpos[bs] < len(m[fpos[fs]]) else None
                 for bs, _ in b_slots] for fs, _ in f_slots]

    olay["placement"] = remap_matrix(lay.get("placement"))
    spans = remap_matrix(lay.get("placement_spans"))
    if spans is not None:
        # a span stays only where the bays it covered still follow it in the new order
        for r in range(len(spans)):
            for c in range(len(b_slots)):
                k = int(spans[r][c] or 1)
                src = bpos[b_slots[c][0]]
                ok = all(c + j < len(b_slots) and bpos[b_slots[c + j][0]] == src + j for j in range(k))
                if not ok:
                    spans[r][c] = 1
        olay["placement_spans"] = spans

    om = out.setdefault("meta", {})
    if isinstance(meta.get("floors_y"), list):
        om["floors_y"] = [[round(a, 1), round(b, 1)] for a, b in new_fb]
    if isinstance(meta.get("columns_xy"), list):
        om["columns_xy"] = [[round(a, 1), round(b, 1)] for a, b in new_bb]
    om["image_size"] = [int(round(niw)), int(round(nih))]
    if meta.get("facade_region_xyxy"):
        x0, y0, x1, y1 = meta["facade_region_xyxy"]
        om["facade_region_xyxy"] = [x0, y0, x1 + d_w, y1 + d_h]
    om.pop("floor_bays", None)  # per-floor bay spans index columns by position; the compiler does not read them

    f_targets = {}  # source floor id -> [(new id, dy)]
    for (src, nid), (a, _b) in zip(f_slots, new_fb):
        f_targets.setdefault(src, []).append((nid, a - fb[fpos[src]][0]))
    b_targets = {}
    for (src, nid), (a, _b) in zip(b_slots, new_bb):
        b_targets.setdefault(src, []).append((nid, a - bb[bpos[src]][0]))

    unit_next = [1 + max([int(u["unit_id"]) for u in dsl.get("instances") or [] if isinstance(u.get("unit_id"), int)] or [0])]
    unit_copies: dict[tuple, list[tuple[int, Any]]] = {}  # (old unit id, new floor id) -> [(new bay slot, new id)]
    slot_of_bay = {nid: k for k, (_s, nid) in enumerate(b_slots)}

    def move_box(box, dx, dy):
        return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

    insts = []
    for u in dsl.get("instances") or []:
        fs, bs = int(u["floor"]), int(u["bay"])
        for nf, dy in f_targets.get(fs, []):
            for nb, dx in b_targets.get(bs, []):
                v = copy.deepcopy(u)
                v["floor"], v["bay"] = nf, nb
                v["box_xyxy"] = move_box(u["box_xyxy"], dx, dy)
                uid = u.get("unit_id")
                if (nf, nb) != (fs, bs) and uid is not None:
                    if isinstance(uid, int):
                        v["unit_id"] = unit_next[0]
                        unit_next[0] += 1
                    else:
                        v["unit_id"] = f"{uid}_{nf}_{nb}"
                unit_copies.setdefault((uid, nf), []).append((slot_of_bay[nb], v.get("unit_id")))
                insts.append(v)
    out["instances"] = insts

    balcs = []
    for b in lay.get("balconies") or []:
        fs, bs, be = int(b["floor"]), int(b["bay_start"]), int(b["bay_end"])
        width = bpos.get(be, bpos[bs]) - bpos[bs]
        for nf, dy in f_targets.get(fs, []):
            for nb, dx in b_targets.get(bs, []):
                v = copy.deepcopy(b)
                # the balcony keeps its bay run where the same bays still follow in the new order
                c = [s[1] for s in b_slots].index(nb)
                run = [b_slots[c + j] for j in range(width + 1) if c + j < len(b_slots)
                       and bpos[b_slots[c + j][0]] == bpos[bs] + j]
                v["floor"], v["bay_start"], v["bay_end"], v["bay"] = nf, nb, run[-1][1], nb
                ids = {src: new for src, new in run}
                for key in ("bays", "bays_center"):
                    if isinstance(b.get(key), list):
                        v[key] = [ids[x] for x in b[key] if x in ids] or [nb]
                if b.get("box_xyxy"):
                    v["box_xyxy"] = [int(round(x)) for x in move_box(b["box_xyxy"], dx, dy)]
                if b.get("width_norm") is not None:
                    v["width_norm"] = round(float(b["width_norm"]) * iw / niw, 6)
                for key in NORM_X:
                    if b.get(key) is not None:
                        v[key] = round((float(b[key]) * iw + dx) / niw, 6)
                for key in ("partner_window_unit_ids", "far_window_unit_ids"):
                    if b.get(key):
                        # the copy of that window on this floor, nearest this balcony's bay
                        v[key] = [min(unit_copies.get((w, nf), [(c, w)]), key=lambda t: abs(t[0] - c))[1]
                                  for w in b[key]]
                balcs.append(v)
    if lay.get("balconies") is not None:
        olay["balconies"] = balcs
    return out
