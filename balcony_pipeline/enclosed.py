"""Place SAM3+DA3 enclosed volumes onto the window floor×bay grid.

Open-railing recovery stays in ``run_balcony``. This module only turns a
projecting volume box into one enclosed placement per overlapping storey,
snapped to the columns it covers. Compiler stacking / front windows are unchanged.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

from snap import _bands_from_windows_dsl, _overlap_1d

ROOT = Path(__file__).resolve().parent.parent

MIN_FLOOR_COVER = 0.50
# A box that continues into the storey below must cover this much of the
# upper storey. A lower edge still uses MIN_FLOOR_COVER.
MIN_UPPER_FLOOR_COVER = 0.90
MIN_COLUMN_COVER = 0.40
# Drop an open railing if this fraction of its box sits on an enclosed volume.
OPEN_ON_ENCLOSED_FRAC = 0.25

ENCLOSED_IR: dict[str, Any] = {
    "type": "balcony",
    "debug": False,
    "structure": "projecting",
    "enclosure": "enclosed",
    "floor": {"shape": "rectangle", "params": {"depth": 1.5}},
    "supports": {"count": 0},
    "glazing": None,
    "output": {"slab_thickness": 0.20},
}


def enclosed_ir() -> dict[str, Any]:
    return copy.deepcopy(ENCLOSED_IR)


def ensure_da3_depth(
    image_path: Path,
    depth_dir: Path,
    *,
    device: str = "cuda",
    python_exe: str | None = None,
) -> Path | None:
    """Return ``<stem>_depth.npy``, inferring DA3 in a subprocess if missing.

    A separate process keeps DA3METRIC-LARGE off the SAM3 / railing GPU heap.
    """
    image_path = Path(image_path)
    depth_dir = Path(depth_dir)
    depth_path = depth_dir / f"{image_path.stem}_depth.npy"
    if depth_path.is_file():
        return depth_path
    if not image_path.is_file():
        print(f"warn: image not found {image_path}; skip enclosed volumes")
        return None
    depth_dir.mkdir(parents=True, exist_ok=True)
    exe = python_exe or sys.executable
    script = ROOT / "balcony_train" / "try_da3_enclosure.py"
    cmd = [
        str(exe),
        str(script),
        "--out-dir",
        str(depth_dir),
        "--device",
        str(device),
        "--image",
        str(image_path),
    ]
    print(f"DA3 depth missing; infer {image_path.name} -> {depth_path}", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warn: DA3 infer failed ({exc}); skip enclosed volumes")
        return None
    if not depth_path.is_file():
        print(f"warn: DA3 wrote no {depth_path.name}; skip enclosed volumes")
        return None
    return depth_path


def median_floor_height(windows_dsl: dict[str, Any]) -> float | None:
    """Median storey height in pixels from the window grid, or None if empty."""
    floor_y, _ = _bands_from_windows_dsl(windows_dsl)
    heights = [float(fy1 - fy0) for fy0, fy1 in floor_y.values() if fy1 > fy0]
    if not heights:
        return None
    heights.sort()
    n = len(heights)
    mid = n // 2
    if n % 2:
        return heights[mid]
    return 0.5 * (heights[mid - 1] + heights[mid])


def _box_xyxy(rec: dict[str, Any] | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(rec, dict):
        box = rec.get("box_xyxy") or rec.get("box") or [0, 0, 1, 1]
    else:
        box = rec
    return [int(round(float(v))) for v in box[:4]]


def overlapping_floors(
    box: list[int] | list[float],
    floor_y: dict[int, tuple[int, int]],
    *,
    min_cover: float = MIN_FLOOR_COVER,
    min_upper_cover: float = MIN_UPPER_FLOOR_COVER,
) -> list[int]:
    """Floor ids the box covers.

    The highest floor counts only when the box also continues below it and
    covers at least ``min_upper_cover`` of that storey. Every other floor
    needs ``min_cover``. A floor under its fraction is not an enclosure.
    """
    _x0, y0, _x1, y1 = (float(v) for v in box[:4])
    touched: list[tuple[int, float, float, float]] = []
    for fid, (fy0, fy1) in floor_y.items():
        top, bot = float(fy0), float(fy1)
        overlap = _overlap_1d(y0, y1, top, bot)
        if overlap <= 0:
            continue
        cover = overlap / max(1.0, bot - top)
        touched.append((int(fid), top, bot, cover))
    if not touched:
        return []
    top_fid, _top, top_bot, _cover = min(touched, key=lambda item: item[1])
    upper_is_spill = y1 > top_bot
    hits: list[tuple[float, int]] = []
    for fid, _fy0, _fy1, cover in touched:
        need = min_upper_cover if fid == top_fid and upper_is_spill else min_cover
        if cover >= need:
            hits.append((cover, fid))
    return [fid for _c, fid in sorted(hits, key=lambda item: item[1])]


def snap_columns(
    box: list[int] | list[float],
    bay_x: dict[int, tuple[int, int]],
    *,
    min_cover: float = MIN_COLUMN_COVER,
) -> dict[str, Any]:
    """Span every column the box covers; slivers below ``min_cover`` are ignored.

    Width is the full ``bay_start..bay_end`` range (not the photo box).
    """
    x0, x1 = float(box[0]), float(box[2])
    hits: list[tuple[int, float]] = []
    for bid, (bx0, bx1) in bay_x.items():
        cover = _overlap_1d(x0, x1, float(bx0), float(bx1)) / max(1.0, float(bx1) - float(bx0))
        if cover > 0.0:
            hits.append((int(bid), cover))
    covered = sorted(bid for bid, cover in hits if cover >= min_cover)
    if not covered and hits:
        covered = [max(hits, key=lambda t: t[1])[0]]
    if not covered:
        covered = [0]
    lo, hi = min(covered), max(covered)
    primary = max(hits, key=lambda t: t[1])[0] if hits else lo
    return {
        "bay_start": lo,
        "bay_end": hi,
        "bays": list(range(lo, hi + 1)),
        "bays_center": covered,
        "bay": int(primary),
    }


def slice_volume(
    box: list[int] | list[float],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    *,
    min_floor_cover: float = MIN_FLOOR_COVER,
    min_column_cover: float = MIN_COLUMN_COVER,
) -> list[dict[str, Any]]:
    """One record per overlapping storey, snapped to covered columns."""
    cols = snap_columns(box, bay_x, min_cover=min_column_cover)
    floors = overlapping_floors(box, floor_y, min_cover=min_floor_cover)
    x0, x1 = int(box[0]), int(box[2])
    out: list[dict[str, Any]] = []
    for fid in floors:
        fy0, fy1 = floor_y.get(fid, (int(box[1]), int(box[3])))
        out.append(
            {
                "floor": int(fid),
                "box_xyxy": [x0, int(fy0), x1, int(fy1)],
                "source": "enclosed_volume",
                **cols,
            }
        )
    return out


def _intersect_over_a(
    a: list[int] | list[float] | tuple[int, ...],
    b: list[int] | list[float] | tuple[int, ...],
) -> float:
    """Intersection area / area(a). 0 when ``a`` is empty."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    iw = _overlap_1d(ax0, ax1, bx0, bx1)
    ih = _overlap_1d(ay0, ay1, by0, by1)
    inter = iw * ih
    area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if area <= 1e-6:
        return 0.0
    return inter / area


def open_conflicts_enclosed(
    open_u: dict[str, Any],
    enc_u: dict[str, Any],
    *,
    min_frac: float = OPEN_ON_ENCLOSED_FRAC,
) -> bool:
    """True when enclosed should win on the *same* floor.

    The open terrace on the floor above the stack is not a conflict: its slab
    is the shared roof and must keep its railing. Volume boxes often grow into
    that terrace, so box overlap is ignored across floors.
    """
    if int(open_u.get("floor", -1)) != int(enc_u.get("floor", -2)):
        return False
    if _bay_set(open_u) & _bay_set(enc_u):
        return True
    ob = open_u.get("box_xyxy")
    if not ob:
        return False
    for key in ("box_xyxy", "volume_box_xyxy"):
        eb = enc_u.get(key)
        if eb and _intersect_over_a(ob, eb) >= min_frac:
            return True
    return False


def _bay_set(unit: dict[str, Any]) -> set[int]:
    raw = unit.get("bays")
    if isinstance(raw, list) and raw:
        return {int(b) for b in raw}
    lo = int(unit.get("bay_start", unit.get("bay", 0)))
    hi = int(unit.get("bay_end", lo))
    return set(range(min(lo, hi), max(lo, hi) + 1))


def drop_open_under_enclosed(
    open_units: list[dict[str, Any]],
    enclosed_units: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enclosed owns the cell and any open box that sits on the volume.

    An open railing on the floor above the stack is kept so ``link_stacks``
    can share the slab and still compile the handrail.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for u in open_units:
        hit = any(open_conflicts_enclosed(u, e) for e in enclosed_units)
        if hit:
            dropped.append(u)
        else:
            kept.append(u)
    return kept, dropped


def vote_enclosed_floor_shape(shapes: list[str | None]) -> tuple[str, dict[str, int]]:
    """One plan: circle only wins a strict majority.

    ``triangle`` and any other label count as ``rectangle``. A tie is
    ``rectangle``. Open-balcony clusters use the same rule, separately from
    the railing vote.
    """
    counts = {"rectangle": 0, "circle": 0}
    for raw in shapes:
        kind = "circle" if str(raw or "").strip().lower() == "circle" else "rectangle"
        counts[kind] += 1
    winner = "circle" if counts["circle"] > counts["rectangle"] else "rectangle"
    return winner, counts


def apply_floor_majority(
    ir: dict[str, Any] | None,
    shapes: list[str | None],
) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    """Copy ``ir`` and set its floor to the majority plan. The input is unchanged."""
    shape, counts = vote_enclosed_floor_shape(shapes)
    if not isinstance(ir, dict):
        return ir, shape, counts
    out = copy.deepcopy(ir)
    floor = dict(out.get("floor") or {})
    floor["shape"] = shape
    out["floor"] = floor
    return out, shape, counts


def vote_enclosed_skin(skins: list[str | None]) -> tuple[str, dict[str, int]]:
    """One shell for a photo: curtain wall only wins a strict majority.

    ``None`` and any other label count as ``punched_window``. A tie is
    ``punched_window``.
    """
    counts = {"punched_window": 0, "curtain_wall": 0}
    for raw in skins:
        kind = "curtain_wall" if str(raw or "").strip().lower() == "curtain_wall" else "punched_window"
        counts[kind] += 1
    winner = "curtain_wall" if counts["curtain_wall"] > counts["punched_window"] else "punched_window"
    return winner, counts


def assign_enclosed_plan(
    units: list[dict[str, Any]],
    shape: str,
    skin: str = "punched_window",
) -> dict[str, Any]:
    """Write one floor plan and one shell onto every enclosed unit."""
    plan = "circle" if str(shape).strip().lower() == "circle" else "rectangle"
    curtain = str(skin).strip().lower() == "curtain_wall"
    ir = enclosed_ir()
    ir["floor"]["shape"] = plan
    if curtain:
        ir["walls"] = {"infill": "glass", "curtain_wall": True}
    else:
        ir["walls"] = {"infill": "wall"}
    for unit in units:
        unit["structure_ir"] = copy.deepcopy(ir)
    return ir


def assign_enclosed_floor(units: list[dict[str, Any]], shape: str) -> dict[str, Any]:
    """Write the same floor plan onto every enclosed unit. Shell stays punched."""
    return assign_enclosed_plan(units, shape, "punched_window")


def place_enclosed_cells(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    windows_dsl: dict[str, Any],
    *,
    unit_id0: int = 0,
) -> list[dict[str, Any]]:
    """One enclosed unit per floor×bay cell, on that cell's grid box."""
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    if not floor_y or not bay_x:
        return []
    units: list[dict[str, Any]] = []
    uid = int(unit_id0)
    for vi, (floor, bay) in enumerate(sorted((int(f), int(b)) for f, b in cells)):
        if floor not in floor_y or bay not in bay_x:
            continue
        y0, y1 = floor_y[floor]
        x0, x1 = bay_x[bay]
        box = [int(x0), int(y0), int(x1), int(y1)]
        units.append(
            {
                "floor": floor,
                "box_xyxy": box,
                "source": "enclosed_volume",
                "bay_start": bay,
                "bay_end": bay,
                "bays": [bay],
                "bays_center": [bay],
                "bay": bay,
                "unit_id": uid,
                "volume_id": vi,
                "volume_box_xyxy": list(box),
                "structure_ir": enclosed_ir(),
            }
        )
        uid += 1
    return units


def place_enclosed_volumes(
    volumes: list[dict[str, Any] | list[int]],
    windows_dsl: dict[str, Any],
    *,
    unit_id0: int = 0,
) -> list[dict[str, Any]]:
    """Slice each volume onto the window grid and attach enclosed IR."""
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    if not floor_y or not bay_x:
        return []
    units: list[dict[str, Any]] = []
    uid = int(unit_id0)
    for vi, rec in enumerate(volumes):
        box = _box_xyxy(rec)
        for sl in slice_volume(box, floor_y, bay_x):
            item = dict(sl)
            item["unit_id"] = uid
            item["volume_id"] = vi
            item["volume_box_xyxy"] = list(box)
            item["structure_ir"] = enclosed_ir()
            item["source"] = "enclosed_volume"
            units.append(item)
            uid += 1
    return units
