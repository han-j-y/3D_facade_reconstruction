"""Type fit check after the clustering vote: a window that does not fit its type in the rendered façade leaves it.

Every window is rendered at its type's median size (``unify_type_sizes``), on its type's median sill per floor
(``align_type_bottoms``) and, with ``meta.align_column_center``, on its bay's median centre x
(``align_column_centers``). A window is a *misfit* when that rendered box

* overlaps another window or door by more than ``overlap`` of the smaller of the two, or
* sticks out of its free region by more than ``excess`` of its own width / height on any side. The free region is
  its cell (the bay span of its floor); items stacked in one bay (``slot``) each get a sub-cell, the vertical band
  between the midpoints to their slot neighbours' photo boxes.

both measured beyond what the window already has at its own photo size and position: only conflict the type adds
counts.

Misfits are resolved one at a time, worst first: the window moves to another window type of the façade under which
it fits without making any other window a misfit, trying only types whose voted structure matches its own
prediction (closest median size first); if none fits, it becomes a type of its own (it never takes another
structure). Then every window the loop moved (or left unresolved) takes its local-symmetry mirror partner's type if
it fits there (:func:`_symmetry_pass`). Afterwards every type whose
membership changed is voted again over its current members' own predictions (``pred_ir``), so each type's IR is
the vote of the windows that render with it. Geometry is the renderer's
(``normalize_facade_spec`` + ``planned_windows``), so "fits" means fits in the render.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "vendor" / "window_compiler", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from facade_recovery.structure_vote import structure_fingerprint, vote_cluster_ir  # noqa: E402
from facade_spec import get_cell_span, normalize_facade_spec  # noqa: E402


def pred_of(inst: dict[str, Any]) -> tuple[dict | None, list | None]:
    """A window's own structure prediction (ir, tokens); older DSLs stored it as structure_ir_member /
    structure_tokens."""
    ir = inst.get("pred_ir", inst.get("structure_ir_member"))
    toks = inst.get("pred_tokens", inst.get("structure_tokens") if "structure_ir_member" in inst else None)
    return ir, toks


def _fingerprint(ir: dict | None) -> str | None:
    try:
        return structure_fingerprint(ir) if isinstance(ir, dict) else None
    except Exception:
        return None


def _planned(dsl: dict[str, Any], *, photo: bool = False):
    """Compiled façade and its planned element boxes; ``photo`` = each element at its own photo size and position
    (no type-size unify, no sill snap)."""
    from render_facade import planned_windows  # imported lazily: pulls in the compiler helpers

    facade = normalize_facade_spec(dsl)
    if photo:
        facade["placement_params"] = {**(facade.get("placement_params") or {}), "unify_type_size": False,
                                      "align_type_bottom": False, "align_column_center": False}
    return facade, planned_windows(facade)


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _free_region(facade: dict, dsl: dict, inst: dict, r: int) -> tuple[float, float, float, float]:
    """World (x0, z0, x1, z1) of the instance's cell, or of its sub-cell when it is stacked with others."""
    c0 = int(inst["bay"])
    cell = get_cell_span(facade, r, c0, c0 + max(1, int(inst.get("colspan", 1))),
                         mirror_x=bool((facade.get("placement_params") or {}).get("mirror_x", True)))
    x0, x1, z0, z1 = cell["x0"], cell["x1"], cell["z0"], cell["z1"]
    if "slot" not in inst:
        return x0, z0, x1, z1
    stack = sorted((u for u in dsl["instances"] if u["floor"] == inst["floor"] and u["bay"] == inst["bay"]),
                   key=lambda u: int(u.get("slot", 0)))
    k = stack.index(inst)
    fy0, fy1 = dsl["meta"]["floors_y"][int(inst["floor"])]
    top = fy0 if k == 0 else 0.5 * (stack[k - 1]["box_xyxy"][3] + inst["box_xyxy"][1])
    bot = fy1 if k == len(stack) - 1 else 0.5 * (inst["box_xyxy"][3] + stack[k + 1]["box_xyxy"][1])
    to_z = lambda y: z1 - (y - fy0) / max(1.0, fy1 - fy0) * (z1 - z0)  # noqa: E731  photo y → world z
    return x0, to_z(bot), x1, to_z(top)


def _boxes(dsl: dict[str, Any], *, photo: bool = False):
    facade, planned = _planned(dsl, photo=photo)
    floor_ids = facade.get("floor_ids") or []
    key = {(int(u["floor"]), int(u["bay"]), int(u.get("slot", 0))): i for i, u in enumerate(dsl["instances"])}
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for p in planned:
        i = key.get((int(floor_ids[p["row"]]) if p["row"] < len(floor_ids) else p["row"], int(p["col"]),
                     int(p.get("slot", 0))))
        if i is not None:
            boxes[i] = (p["x0"], p["z0"], p["x1"], p["z1"])
    return facade, boxes


def _conflict(i: int, b, boxes, region) -> tuple[float, float]:
    """(worst overlap with another element as a share of the smaller of the two, worst excess beyond region as a
    share of b's width / height)."""
    a = _area(b)
    w, h = max(1e-9, b[2] - b[0]), max(1e-9, b[3] - b[1])
    # overlap as a share of the smaller box, so the larger of two colliding windows is flagged too
    ov = max((_inter(b, o) / max(1e-9, min(a, _area(o))) for j, o in boxes.items() if j != i), default=0.0)
    fx0, fz0, fx1, fz1 = region
    ex = max((fx0 - b[0]) / w, (b[2] - fx1) / w, (fz0 - b[1]) / h, (b[3] - fz1) / h, 0.0)
    return ov, ex


def misfits(dsl: dict[str, Any], *, overlap: float = 0.01, excess: float = 0.10) -> dict[int, float]:
    """``{instance index: badness}`` of the windows their type does not fit. Only conflict the type adds counts:
    overlap / excess of the rendered box beyond that of the window at its own photo size and position (a photo box
    already sticking out of its floor band is a grid matter, not a type one). Badness = worst ratio to threshold."""
    facade, boxes = _boxes(dsl)
    _, photo = _boxes(dsl, photo=True)
    row_of = {int(f): r for r, f in enumerate(facade.get("floor_ids") or [])}
    out: dict[int, float] = {}
    for i, b in boxes.items():
        inst = dsl["instances"][i]
        if inst.get("kind") == "door" or i not in photo:
            continue
        region = _free_region(facade, dsl, inst, row_of[int(inst["floor"])])
        ov, ex = _conflict(i, b, boxes, region)
        ov0, ex0 = _conflict(i, photo[i], photo, region)
        bad = 0.0
        if ov - ov0 > overlap:
            bad = (ov - ov0) / overlap
        if ex - ex0 > excess:
            bad = max(bad, (ex - ex0) / excess)
        if bad > 0:
            out[i] = bad
    return out


def _set_type(dsl: dict[str, Any], i: int, type_id: int) -> None:
    """Point instance ``i`` (and its placement token) at window type ``type_id``."""
    inst = dsl["instances"][i]
    inst["type_id"] = type_id
    tok = f"win_type_{type_id:02d}"
    cell = dsl["layout"]["placement"][int(inst["floor"])][int(inst["bay"])]
    if isinstance(cell, list):
        cell[int(inst.get("slot", 0))] = tok
    else:
        dsl["layout"]["placement"][int(inst["floor"])][int(inst["bay"])] = tok


def _own_type(dsl: dict[str, Any], i: int) -> int:
    """New window type for instance ``i`` (its structure is voted by :func:`revote` once membership settles)."""
    inst = dsl["instances"][i]
    ir, toks = pred_of(inst)
    tid = 1 + max(int(t["type_id"]) for t in dsl["window_types"])
    dsl["window_types"].append({
        "type_id": tid, "name": f"win_type_{tid:02d}", "n_instances": 1, "exemplar_unit": inst.get("unit_id"),
        "exemplar_asset": inst.get("asset"), "structure_ir": copy.deepcopy(ir),
        "structure_tokens": list(toks or []), "split_by_type_fit": True,
    })
    return tid


def revote(dsl: dict[str, Any], type_ids: set[int]) -> None:
    """Re-run the structure vote of ``type_ids`` over their current members' own predictions (majority, ties to the
    exemplar), so every type's IR is the vote of the windows that actually render with it. The exemplar stays if it
    is still a member, else the member whose box is closest to the type's median size."""
    wins = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
    for t in dsl["window_types"]:
        tid = int(t["type_id"])
        if tid not in type_ids:
            continue
        members = [u for u in wins if int(u["type_id"]) == tid]
        if not members:
            continue
        ids = [u.get("unit_id") for u in members]
        if t.get("exemplar_unit") not in ids:
            mw = sorted(u["box_xyxy"][2] - u["box_xyxy"][0] for u in members)[len(members) // 2]
            mh = sorted(u["box_xyxy"][3] - u["box_xyxy"][1] for u in members)[len(members) // 2]
            ex = min(members, key=lambda u: abs(u["box_xyxy"][2] - u["box_xyxy"][0] - mw)
                     + abs(u["box_xyxy"][3] - u["box_xyxy"][1] - mh))
            t["exemplar_unit"], t["exemplar_asset"] = ex.get("unit_id"), ex.get("asset")
        preds = []
        for u in members:
            ir, toks = pred_of(u)
            preds.append({"unit_id": u.get("unit_id"), "ir": ir, "tokens": toks})
        voted = vote_cluster_ir(preds, prefer_unit_id=t.get("exemplar_unit"))
        if voted.get("structure_ir") is not None:
            t["structure_ir"], t["structure_tokens"] = voted["structure_ir"], voted["structure_tokens"]
        t["structure_vote"] = {**voted["vote"], "revoted_after_type_fit": True}
        t["structure_members"] = voted["members"]


def _symmetry_pass(dsl: dict[str, Any], moves: list[dict[str, Any]], settled: set[int], *, overlap: float,
                   excess: float, min_strength: float) -> list[dict[str, Any]]:
    """After the misfit loop: every window type fit moved (or left unresolved) takes the type of its local-symmetry
    mirror partner (``meta.local_sym_pairs``, strongest axis first, axes below ``min_strength`` ignored) if it fits
    there without making any other window a misfit. Without local symmetry (no pairs) this does nothing."""
    pairs = dsl["meta"].get("local_sym_pairs") or []
    if not pairs:
        return []
    idx_of = {u.get("unit_id"): k for k, u in enumerate(dsl["instances"]) if u.get("kind", "window") == "window"}
    partners: dict[int, list[tuple[float, int]]] = {}
    for a, b, s in pairs:
        if float(s) < min_strength or a not in idx_of or b not in idx_of:
            continue
        partners.setdefault(a, []).append((float(s), b))
        partners.setdefault(b, []).append((float(s), a))
    todo = list(dict.fromkeys([m["unit_id"] for m in moves]
                              + [dsl["instances"][i].get("unit_id") for i in sorted(settled)]))
    out: list[dict[str, Any]] = []
    for uid in todo:
        i = idx_of.get(uid)
        if i is None or uid not in partners:
            continue
        cur = int(dsl["instances"][i]["type_id"])
        before = set(misfits(dsl, overlap=overlap, excess=excess))
        for s, puid in sorted(partners[uid], reverse=True):
            tid = int(dsl["instances"][idx_of[puid]]["type_id"])
            if tid == cur:
                break  # already with its partner
            _set_type(dsl, i, tid)
            after = set(misfits(dsl, overlap=overlap, excess=excess))
            if i not in after and after - {i} <= before:
                out.append({"unit_id": uid, "partner": puid, "axis_strength": s, "from": cur, "to": tid})
                break
            _set_type(dsl, i, cur)
    return out


def fit_types(dsl: dict[str, Any], *, overlap: float = 0.01, excess: float = 0.10,
              max_moves: int = 200, sym_min_strength: float = 0.5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve misfits (module doc). Returns the updated DSL and the list of moves made."""
    dsl = copy.deepcopy(dsl)
    moves: list[dict[str, Any]] = []
    settled: set[int] = set()  # own-type windows that still do not fit (their photo boxes overlap): left alone
    for _ in range(max_moves):
        bad = {i: v for i, v in misfits(dsl, overlap=overlap, excess=excess).items() if i not in settled}
        if not bad:
            break
        i = max(bad, key=bad.get)
        inst = dsl["instances"][i]
        old = int(inst["type_id"])
        others_before = set(bad) - {i}
        x0, y0, x1, y1 = inst["box_xyxy"]
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        wins = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]

        def size_gap(tid: int) -> float:
            m = [u["box_xyxy"] for u in wins if int(u["type_id"]) == tid and u is not inst]
            mw = sorted(b[2] - b[0] for b in m)[len(m) // 2]
            mh = sorted(b[3] - b[1] for b in m)[len(m) // 2]
            return abs(math.log(mw / bw)) + abs(math.log(mh / bh))

        # targets: only types whose voted structure matches the window's own prediction (closest size first);
        # a window that fits none of them gets a type of its own rather than another structure
        own_fp = _fingerprint(pred_of(inst)[0])
        type_fp = {int(t["type_id"]): _fingerprint(t.get("structure_ir")) for t in dsl["window_types"]}
        cands = sorted((tid for tid in {int(u["type_id"]) for u in wins if u is not inst} - {old}
                        if own_fp is not None and type_fp.get(tid) == own_fp), key=size_gap)
        new = None
        for tid in cands:
            _set_type(dsl, i, tid)
            after = misfits(dsl, overlap=overlap, excess=excess)
            if i not in after and set(after) - settled <= others_before:
                new = tid
                break
        if new is None:
            if sum(int(u["type_id"]) == old for u in wins) == 1:  # already alone in its type: nothing to split
                _set_type(dsl, i, old)
                settled.add(i)
                continue
            new = _own_type(dsl, i)
            _set_type(dsl, i, new)
            if i in misfits(dsl, overlap=overlap, excess=excess):
                settled.add(i)
        moves.append({"unit_id": inst.get("unit_id"), "box_xyxy": inst["box_xyxy"], "from": old, "to": new,
                      "own_type": any(t.get("split_by_type_fit") and int(t["type_id"]) == new
                                      for t in dsl["window_types"]),
                      "same_structure": own_fp is not None and own_fp == type_fp.get(new),
                      "badness": round(bad[i], 2)})
    sym_moves = _symmetry_pass(dsl, moves, settled, overlap=overlap, excess=excess, min_strength=sym_min_strength)
    # every type whose membership changed is voted again over its current members
    revote(dsl, {m["from"] for m in moves + sym_moves} | {m["to"] for m in moves + sym_moves})
    # drop emptied types, recount members
    used = {int(u["type_id"]) for u in dsl["instances"] if u.get("kind", "window") == "window"}
    dsl["window_types"] = [t for t in dsl["window_types"] if int(t["type_id"]) in used]
    for t in dsl["window_types"]:
        t["n_instances"] = sum(int(u["type_id"]) == int(t["type_id"]) for u in dsl["instances"]
                               if u.get("kind", "window") == "window")
    dsl["meta"]["n_types"] = len(dsl["window_types"])
    dsl["meta"]["type_fit"] = {"overlap": overlap, "excess": excess, "moves": moves, "sym_moves": sym_moves,
                               "unresolved": sorted(dsl["instances"][i].get("unit_id") for i in settled)}
    return dsl, moves


def misfit_count(dsl: dict[str, Any], **kw) -> int:
    return len(misfits(dsl, **kw))

