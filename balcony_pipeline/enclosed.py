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

MIN_FLOOR_COVER = 0.35
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
) -> list[int]:
    """Floor ids whose band is covered by at least ``min_cover`` of the storey height."""
    _x0, y0, _x1, y1 = (float(v) for v in box[:4])
    hits: list[tuple[float, int]] = []
    for fid, (fy0, fy1) in floor_y.items():
        cover = _overlap_1d(y0, y1, float(fy0), float(fy1)) / max(1.0, float(fy1) - float(fy0))
        if cover >= min_cover:
            hits.append((cover, int(fid)))
    if hits:
        return [fid for _c, fid in sorted(hits, key=lambda t: t[1])]
    if not floor_y:
        return []
    best = max(
        floor_y,
        key=lambda f: _overlap_1d(y0, y1, float(floor_y[f][0]), float(floor_y[f][1])),
    )
    return [int(best)]


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
