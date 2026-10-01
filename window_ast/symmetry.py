"""Integral symmetry / SymScore from Zhang et al., SIGGRAPH 2013.

Paper: *Layered Analysis of Irregular Facades via Symmetry Maximization*
(ACM TOG 32(4), 10.1145/2461912.2461923).

Core ideas implemented here (Sec. 4):
  - Intra- / inter-box reflection **symmetry profiles** over a box set
  - **Integral symmetry** ``I(S)`` with centricity Gaussian weight
  - Normalized IS sum ``Ns`` / ``Nl`` and decomposition **SymScore**

Boxes are axis-aligned ``(x0, y0, x1, y1)`` in a shared facade plane.
Optional ``types`` mark repetition groups: inter-box profiles are only
nonzero between boxes of the same type (paper: only repeated elements
can be symmetric). If ``types`` is omitted, every box is treated as the
same type (typical when scoring one cluster).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


def _as_boxes(boxes: Sequence[Sequence[float]]) -> list[Box]:
    out: list[Box] = []
    for b in boxes:
        x0, y0, x1, y1 = map(float, b[:4])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        out.append((x0, y0, x1, y1))
    return out


def box_area(b: Sequence[float]) -> float:
    x0, y0, x1, y1 = map(float, b[:4])
    return abs(x1 - x0) * abs(y1 - y0)


def total_box_area(boxes: Sequence[Sequence[float]]) -> float:
    return float(sum(box_area(b) for b in _as_boxes(boxes)))


def bounding_box(boxes: Sequence[Sequence[float]]) -> Box | None:
    bs = _as_boxes(boxes)
    if not bs:
        return None
    return (
        min(b[0] for b in bs),
        min(b[1] for b in bs),
        max(b[2] for b in bs),
        max(b[3] for b in bs),
    )


def _area_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def reflect_overlap_vertical_axis(b1: Box, b2: Box, x: float) -> float:
    """Area of ``B1 ∩ reflect_vertical(B2, x)`` (paper inter-/intra-box)."""
    x0, y0, x1, y1 = b1
    u0, v0, u1, v1 = b2
    # reflect B2 about vertical line x: u' = 2x - u
    ru0, ru1 = 2.0 * x - u1, 2.0 * x - u0
    if ru0 > ru1:
        ru0, ru1 = ru1, ru0
    ox = _area_overlap_1d(x0, x1, ru0, ru1)
    oy = _area_overlap_1d(y0, y1, v0, v1)
    return ox * oy


def reflect_overlap_horizontal_axis(b1: Box, b2: Box, y: float) -> float:
    """Area of ``B1 ∩ reflect_horizontal(B2, y)``."""
    x0, y0, x1, y1 = b1
    u0, v0, u1, v1 = b2
    rv0, rv1 = 2.0 * y - v1, 2.0 * y - v0
    if rv0 > rv1:
        rv0, rv1 = rv1, rv0
    ox = _area_overlap_1d(x0, x1, u0, u1)
    oy = _area_overlap_1d(y0, y1, rv0, rv1)
    return ox * oy


def _pair_profile(
    b1: Box,
    b2: Box,
    axis: float,
    *,
    horizontal: bool,
) -> float:
    """Order-independent inter-box profile (paper: order does not matter)."""
    if horizontal:
        a = reflect_overlap_vertical_axis(b1, b2, axis)
        b = reflect_overlap_vertical_axis(b2, b1, axis)
    else:
        a = reflect_overlap_horizontal_axis(b1, b2, axis)
        b = reflect_overlap_horizontal_axis(b2, b1, axis)
    return 0.5 * (a + b)


def _sanitize_boxes_types(
    boxes: Sequence[Sequence[float]],
    types: Sequence[int] | None,
) -> tuple[list[Box], list[int]]:
    if types is not None and len(types) != len(boxes):
        raise ValueError("types length must match boxes")
    bs: list[Box] = []
    types_l: list[int] = []
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = map(float, b[:4])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        bs.append((x0, y0, x1, y1))
        types_l.append(0 if types is None else int(types[i]))
    return bs, types_l


def symmetry_profile_parts(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    horizontal: bool = True,
    n_samples: int = 512,
    extent: Box | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample intra-box and inter-box profiles separately (Zhang Sec. 4).

    Returns ``(coords, p_intra, p_inter)``. Paper total profile is
    ``p_S = p_intra + p_inter``. Inter-box terms are zero between different
    ``types`` (only repeated / same-cluster elements pair).
    """
    bs, types_l = _sanitize_boxes_types(boxes, types)
    if not bs:
        z = np.zeros(0)
        return z, z, z

    bb = extent if extent is not None else bounding_box(bs)
    assert bb is not None
    if horizontal:
        lo, hi = bb[0], bb[2]
    else:
        lo, hi = bb[1], bb[3]
    if hi <= lo:
        z = np.zeros(0)
        return z, z, z

    xs = np.linspace(lo, hi, int(n_samples), dtype=np.float64)
    p_intra = np.zeros_like(xs)
    p_inter = np.zeros_like(xs)

    for b in bs:
        if horizontal:
            p_intra += np.array(
                [reflect_overlap_vertical_axis(b, b, float(x)) for x in xs]
            )
        else:
            p_intra += np.array(
                [reflect_overlap_horizontal_axis(b, b, float(x)) for x in xs]
            )

    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            if types_l[i] != types_l[j]:
                continue
            p_inter += np.array(
                [
                    _pair_profile(bs[i], bs[j], float(x), horizontal=horizontal)
                    for x in xs
                ]
            )
    return xs, p_intra, p_inter


def symmetry_profile(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    horizontal: bool = True,
    n_samples: int = 512,
    extent: Box | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the integrated symmetry profile ``p_S`` along one axis.

    Returns ``(coords, p)`` where ``coords`` lie in the tight extent of
    ``extent`` (default: bbox of ``boxes``), and ``p[i]`` is
    ``Σ_intra + Σ_inter`` at that reflection axis (Zhang et al. Eq. before (2)).
    """
    xs, p_intra, p_inter = symmetry_profile_parts(
        boxes,
        types=types,
        horizontal=horizontal,
        n_samples=n_samples,
        extent=extent,
    )
    return xs, p_intra + p_inter


def _centricity_integral(xs: np.ndarray, p: np.ndarray) -> float:
    """``∫ p(x) g(x) dx`` with ``g(x)=exp(-(x-c)^2 / (w/3)^2)`` (Eq. 2)."""
    if xs.size < 2:
        return 0.0
    lo, hi = float(xs[0]), float(xs[-1])
    w = hi - lo
    if w <= 0:
        return 0.0
    c = 0.5 * (lo + hi)
    sigma = w / 3.0
    g = np.exp(-((xs - c) / sigma) ** 2)
    return float(np.trapz(p * g, xs))


def integral_symmetry_1d(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    horizontal: bool = True,
    n_samples: int = 512,
    extent: Box | None = None,
) -> float:
    """Horizontal or vertical integral symmetry ``I_h(S)`` / ``I_v(S)``."""
    xs, p = symmetry_profile(
        boxes,
        types=types,
        horizontal=horizontal,
        n_samples=n_samples,
        extent=extent,
    )
    return _centricity_integral(xs, p)


def integral_symmetry_1d_parts(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    horizontal: bool = True,
    n_samples: int = 512,
    extent: Box | None = None,
) -> tuple[float, float]:
    """``(I_intra, I_inter)`` for one axis — paper in-box vs between-box."""
    xs, p_intra, p_inter = symmetry_profile_parts(
        boxes,
        types=types,
        horizontal=horizontal,
        n_samples=n_samples,
        extent=extent,
    )
    return _centricity_integral(xs, p_intra), _centricity_integral(xs, p_inter)


def integral_symmetry(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
    extent: Box | None = None,
) -> float:
    """Combined integral symmetry ``I(S)`` (weighted ``I_h + I_v``).

    Paper: weighted sum of horizontal and vertical IS; default equal weights.
    """
    wh = float(horizontal_weight)
    wv = float(vertical_weight)
    if wh == 0.0 and wv == 0.0:
        return 0.0
    ih = (
        integral_symmetry_1d(
            boxes,
            types=types,
            horizontal=True,
            n_samples=n_samples,
            extent=extent,
        )
        if wh
        else 0.0
    )
    iv = (
        integral_symmetry_1d(
            boxes,
            types=types,
            horizontal=False,
            n_samples=n_samples,
            extent=extent,
        )
        if wv
        else 0.0
    )
    return wh * ih + wv * iv


def integral_symmetry_parts(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
    extent: Box | None = None,
) -> tuple[float, float]:
    """Weighted ``(I_in_box, I_global_pairs)`` = intra vs inter integral symmetry.

    - **In-box** (intra): each box mirrored onto itself — local rectangular
      self-symmetry, independent of other windows.
    - **Global pairs** (inter): same-``type`` boxes mirrored onto each other
      over the search extent (facade-wide when ``extent`` is the facade bbox).
    """
    wh = float(horizontal_weight)
    wv = float(vertical_weight)
    intra = inter = 0.0
    if wh:
        a, b = integral_symmetry_1d_parts(
            boxes,
            types=types,
            horizontal=True,
            n_samples=n_samples,
            extent=extent,
        )
        intra += wh * a
        inter += wh * b
    if wv:
        a, b = integral_symmetry_1d_parts(
            boxes,
            types=types,
            horizontal=False,
            n_samples=n_samples,
            extent=extent,
        )
        intra += wv * a
        inter += wv * b
    return float(intra), float(inter)


@dataclass(frozen=True)
class FacadeSymmetryScore:
    """Whole-facade Zhang scores: in-box + global (inter) + normalized total.

    ``extent`` is the facade bbox (all windows). Profiles are integrated with
    centricity toward the facade center — not a single cluster's AABB.
    ``types`` are cluster ids so inter-box terms only pair same-cluster windows.
    """

    in_box: float  # I_intra
    global_pairs: float  # I_inter (same-type pairs over facade)
    total: float  # I_intra + I_inter
    blank: float  # I(β(facade))
    normalized: float  # total / blank ∈ [0, 1]
    normalized_in_box: float
    normalized_global: float
    extent: Box


def facade_symmetry(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    extent: Box | None = None,
    image_size: tuple[float, float] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> FacadeSymmetryScore:
    """Paper-style whole-facade symmetry (in-box + global inter-box).

    Parameters
    ----------
    boxes
        All window boxes on the facade.
    types
        Per-box cluster / repetition id. Inter-box symmetry only between
        equal types (paper). ``None`` → all same type.
    extent
        Axis search domain. Default: tight bbox of ``boxes``. Pass the full
        image ``(0,0,W,H)`` to force centricity toward the image center.
    image_size
        If given and ``extent`` is None, use ``(0, 0, W, H)`` as extent.
    """
    bs, _ = _sanitize_boxes_types(boxes, types)
    if not bs:
        empty = (0.0, 0.0, 0.0, 0.0)
        return FacadeSymmetryScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, empty)

    if extent is None:
        if image_size is not None:
            extent = (0.0, 0.0, float(image_size[0]), float(image_size[1]))
        else:
            extent = bounding_box(bs)
    assert extent is not None

    kw = dict(
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
        extent=extent,
    )
    intra, inter = integral_symmetry_parts(bs, types=types, **kw)
    total = intra + inter
    blank = integral_symmetry([extent], types=[0], **kw)
    if blank <= 1e-12:
        return FacadeSymmetryScore(
            intra, inter, total, blank, 0.0, 0.0, 0.0, extent
        )
    return FacadeSymmetryScore(
        in_box=intra,
        global_pairs=inter,
        total=total,
        blank=blank,
        normalized=float(np.clip(total / blank, 0.0, 1.0)),
        normalized_in_box=float(np.clip(intra / blank, 0.0, 1.0)),
        normalized_global=float(np.clip(inter / blank, 0.0, 1.0)),
        extent=extent,
    )


def normalized_integral_symmetry(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
    extent: Box | None = None,
) -> float:
    """``I(S) / I(β) ∈ [0, 1]`` — natural symmetry score.

    Default ``β = bbox(S)`` (per-cluster / local). Pass a facade-wide
    ``extent`` (or use :func:`facade_symmetry`) for whole-facade scoring.
    """
    bs = _as_boxes(boxes)
    if not bs:
        return 0.0
    bb = extent if extent is not None else bounding_box(bs)
    assert bb is not None
    i_s = integral_symmetry(
        bs,
        types=types,
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
        extent=bb,
    )
    i_b = integral_symmetry(
        [bb],
        types=[0],
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
        extent=bb,
    )
    if i_b <= 1e-12:
        return 0.0
    return float(np.clip(i_s / i_b, 0.0, 1.0))


def area_normalized_integral_symmetry(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> float:
    """``I(S) / Σ area(B)`` — no empty-bbox occupancy penalty.

    Unlike :func:`normalized_integral_symmetry`, the denominator is only the
    boxes themselves, so a sparse but mirrored set (e.g. four corner openings
    around a staircase) is not crushed by the unused gap in ``β(S)``.

    Units are length (``I`` is ∫ area-overlap dx). For a dimensionless
    alternative see :func:`self_normalized_integral_symmetry`.
    """
    bs = _as_boxes(boxes)
    if not bs:
        return 0.0
    area = total_box_area(bs)
    if area <= 1e-12:
        return 0.0
    i_s = integral_symmetry(
        bs,
        types=types,
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
    return float(i_s / area)


def self_normalized_integral_symmetry(
    boxes: Sequence[Sequence[float]],
    *,
    types: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> float:
    """``I(S) / Σ_i I({B_i})`` — 1 = only per-box self-symmetry.

    Inter-box pairings push the score above 1. Independent of how much
    empty space sits inside the cluster bounding box.
    """
    bs = _as_boxes(boxes)
    if not bs:
        return 0.0
    if types is None:
        types_l = [0] * len(bs)
    else:
        types_l = [int(t) for t in types]
        if len(types_l) != len(bs):
            raise ValueError("types length must match boxes")
    kw = dict(
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
    i_s = integral_symmetry(bs, types=types_l, **kw)
    i_self = 0.0
    for b, t in zip(bs, types_l):
        i_self += integral_symmetry([b], types=[t], **kw)
    if i_self <= 1e-12:
        return 0.0
    return float(i_s / i_self)


def nis_split(
    s1: Sequence[Sequence[float]],
    s2: Sequence[Sequence[float]],
    *,
    types1: Sequence[int] | None = None,
    types2: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> float:
    """Normalized IS sum ``Ns`` for a **split** (Eq. 3), in ``[0, 1]``."""
    kw = dict(
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
    n1 = normalized_integral_symmetry(s1, types=types1, **kw)
    n2 = normalized_integral_symmetry(s2, types=types2, **kw)
    return 0.5 * (n1 + n2)


def nis_layer(
    bottom: Sequence[Sequence[float]],
    top: Sequence[Sequence[float]],
    *,
    bottom_completed: Sequence[Sequence[float]] | None = None,
    types_bottom: Sequence[int] | None = None,
    types_top: Sequence[int] | None = None,
    types_bottom_completed: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> float:
    """Normalized IS sum ``Nl`` for a **layering** (Eq. 4), in ``[0, 1]``.

    ``bottom_completed`` is ``S1*`` after structure completion; if omitted,
    uses ``bottom`` as-is (lazy / no occlusion fill).
    """
    s1 = list(bottom_completed) if bottom_completed is not None else list(bottom)
    t1 = types_bottom_completed if bottom_completed is not None else types_bottom
    bb1 = bounding_box(bottom)
    bb2 = bounding_box(top)
    if bb1 is None or bb2 is None:
        return 0.0
    kw = dict(
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
    i1 = integral_symmetry(s1, types=t1, **kw)
    i2 = integral_symmetry(top, types=types_top, **kw)
    ib1 = integral_symmetry([bb1], types=[0], **kw)
    ib2 = integral_symmetry([bb2], types=[0], **kw)
    den = ib1 + ib2
    if den <= 1e-12:
        return 0.0
    return float(np.clip((i1 + i2) / den, 0.0, 1.0))


@dataclass(frozen=True)
class SymScoreResult:
    """Decomposition SymScore (Eq. 5): ``N(S1,S2) · I(β(S))``."""

    sym_score: float
    nis: float
    i_bbox: float
    mode: Literal["split", "layer"]


def sym_score(
    parent: Sequence[Sequence[float]],
    s1: Sequence[Sequence[float]],
    s2: Sequence[Sequence[float]],
    *,
    mode: Literal["split", "layer"] = "split",
    types_parent: Sequence[int] | None = None,
    types1: Sequence[int] | None = None,
    types2: Sequence[int] | None = None,
    s1_completed: Sequence[Sequence[float]] | None = None,
    types1_completed: Sequence[int] | None = None,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> SymScoreResult:
    """Symmetry score of a binary decomposition of ``parent`` into ``s1,s2``.

    For clustering you usually want :func:`normalized_integral_symmetry` on
    one group's boxes; this function is the paper's node objective used when
    searching split/layer hierarchies.
    """
    kw = dict(
        n_samples=n_samples,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
    )
    bb = bounding_box(parent)
    if bb is None:
        return SymScoreResult(0.0, 0.0, 0.0, mode)
    i_b = integral_symmetry([bb], types=[0], **kw)
    if mode == "split":
        n = nis_split(s1, s2, types1=types1, types2=types2, **kw)
    elif mode == "layer":
        n = nis_layer(
            s1,
            s2,
            bottom_completed=s1_completed,
            types_bottom=types1,
            types_top=types2,
            types_bottom_completed=types1_completed,
            **kw,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return SymScoreResult(sym_score=n * i_b, nis=n, i_bbox=i_b, mode=mode)


def cluster_symmetry_scores(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_samples: int = 512,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
    normalize: Literal["bbox", "area", "self"] = "bbox",
) -> dict[int, float]:
    """Per-cluster symmetry for facade-plane box groups.

    ``normalize``:
      - ``bbox`` — paper ``I(S)/I(β(S))`` (penalizes empty gaps in the bbox)
      - ``area`` — ``I(S) / Σ area(B)`` (no occupancy penalty)
      - ``self`` — ``I(S) / Σ I({B_i})`` (1 = self-symmetry only)
    """
    if len(boxes) != len(labels):
        raise ValueError("boxes/labels length mismatch")
    fn = {
        "bbox": normalized_integral_symmetry,
        "area": area_normalized_integral_symmetry,
        "self": self_normalized_integral_symmetry,
    }.get(normalize)
    if fn is None:
        raise ValueError(f"unknown normalize={normalize!r}")
    by: dict[int, list[Sequence[float]]] = {}
    for b, lab in zip(boxes, labels):
        by.setdefault(int(lab), []).append(b)
    return {
        lab: fn(
            group,
            n_samples=n_samples,
            horizontal_weight=horizontal_weight,
            vertical_weight=vertical_weight,
        )
        for lab, group in sorted(by.items())
    }


def reflect_box_vertical(b: Sequence[float], x_center: float) -> Box:
    """Mirror box about vertical line ``x = x_center``."""
    x0, y0, x1, y1 = map(float, b[:4])
    return (2.0 * x_center - x1, y0, 2.0 * x_center - x0, y1)


def reflect_boxes_vertical(
    boxes: Sequence[Sequence[float]], x_center: float
) -> list[Box]:
    return [reflect_box_vertical(b, x_center) for b in _as_boxes(boxes)]


def facade_center_x(
    boxes: Sequence[Sequence[float]] | None = None,
    *,
    image_width: float | None = None,
) -> float:
    """Vertical midline for bilateral facade symmetry.

    Prefers the center of the tight bbox of all boxes; falls back to half the
    image width when no boxes are given.
    """
    if boxes:
        bb = bounding_box(boxes)
        if bb is not None:
            return 0.5 * (bb[0] + bb[2])
    if image_width is not None:
        return 0.5 * float(image_width)
    raise ValueError("need boxes or image_width")


def box_iou(a: Box, b: Box) -> float:
    ox = _area_overlap_1d(a[0], a[2], b[0], b[2])
    oy = _area_overlap_1d(a[1], a[3], b[1], b[3])
    inter = ox * oy
    if inter <= 0:
        return 0.0
    aa = box_area(a)
    ab = box_area(b)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def mirror_match_recall(
    source: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    x_center: float,
    *,
    iou_thresh: float = 0.25,
) -> float:
    """Fraction of ``source`` area that finds a mirror partner in ``target``.

    Reflect each box in ``source`` about ``x_center`` and greedily match the
    best IoU partner in ``target`` (each target used at most once).
    """
    src = _as_boxes(source)
    tgt = _as_boxes(target)
    if not src:
        return 0.0
    reflected = reflect_boxes_vertical(src, x_center)
    used = [False] * len(tgt)
    matched_area = 0.0
    total_area = total_box_area(src)
    for rb, sb in zip(reflected, src):
        best_j = -1
        best_iou = 0.0
        for j, tb in enumerate(tgt):
            if used[j]:
                continue
            iou = box_iou(rb, tb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            used[best_j] = True
            matched_area += box_area(sb)
    return float(matched_area / total_area) if total_area > 0 else 0.0


def cluster_mirror_score(
    boxes_a: Sequence[Sequence[float]],
    boxes_b: Sequence[Sequence[float]],
    x_center: float,
    *,
    iou_thresh: float = 0.25,
) -> float:
    """Symmetric bilateral match between two clusters about ``x_center``.

    Returns the minimum of mirror recall(A→B) and mirror recall(B→A), in
    ``[0, 1]``. High values indicate the two groups are mirror copies and
    strong merge candidates (e.g. left/right columns).
    """
    ab = mirror_match_recall(boxes_a, boxes_b, x_center, iou_thresh=iou_thresh)
    ba = mirror_match_recall(boxes_b, boxes_a, x_center, iou_thresh=iou_thresh)
    return float(min(ab, ba))


def facade_bilateral_symmetry(
    boxes: Sequence[Sequence[float]],
    x_center: float,
    *,
    iou_thresh: float = 0.25,
) -> float:
    """Self-symmetry of one box set about the facade vertical midline.

    Each box must mirror to another box in the same set (partner matching).
    """
    return mirror_match_recall(boxes, boxes, x_center, iou_thresh=iou_thresh)


def suggest_mirror_merges(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    x_center: float,
    *,
    iou_thresh: float = 0.25,
    min_score: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Rank unordered cluster pairs by facade-center mirror score.

    Returns ``[(label_a, label_b, score), ...]`` sorted descending.
    """
    by: dict[int, list[Sequence[float]]] = {}
    for b, lab in zip(boxes, labels):
        by.setdefault(int(lab), []).append(b)
    labs = sorted(by.keys())
    out: list[tuple[int, int, float]] = []
    for i, la in enumerate(labs):
        for lb in labs[i + 1 :]:
            s = cluster_mirror_score(by[la], by[lb], x_center, iou_thresh=iou_thresh)
            if s >= min_score:
                out.append((la, lb, s))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


@dataclass(frozen=True)
class MirrorPartner:
    """Facade-center mirror partner for one box."""

    index: int
    partner_index: int | None
    mirror_iou: float
    partner_label: int | None
    same_cluster: bool
    outlier: bool  # partner missing or in another cluster
    similarity: float | None = None  # optional appearance sim to partner


def _row_match(b1: Box, b2: Box, *, row_frac: float = 0.35) -> bool:
    """Same approximate floor band (vertical overlap or close centers)."""
    cy1 = 0.5 * (b1[1] + b1[3])
    cy2 = 0.5 * (b2[1] + b2[3])
    h = max(b1[3] - b1[1], b2[3] - b2[1], 1.0)
    return abs(cy1 - cy2) <= row_frac * h


def find_mirror_partner_index(
    boxes: Sequence[Sequence[float]],
    index: int,
    x_center: float,
    *,
    iou_thresh: float = 0.15,
    same_row: bool = True,
    row_frac: float = 0.35,
    exclude_self: bool = True,
) -> tuple[int | None, float]:
    """Best mirror partner for ``boxes[index]`` about ``x_center``.

    Returns ``(partner_index, mirror_iou)`` or ``(None, 0)``.
    """
    bs = _as_boxes(boxes)
    if index < 0 or index >= len(bs):
        return None, 0.0
    rb = reflect_box_vertical(bs[index], x_center)
    best_j: int | None = None
    best_iou = 0.0
    for j, cand in enumerate(bs):
        if exclude_self and j == index:
            continue
        if same_row and not _row_match(bs[index], cand, row_frac=row_frac):
            continue
        iou = box_iou(rb, cand)
        if iou > best_iou:
            best_iou = iou
            best_j = j
    if best_j is None or best_iou < iou_thresh:
        return None, 0.0
    return best_j, float(best_iou)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def box_geometry_similarity(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Cheap fallback when DINO features are unavailable."""
    w1 = abs(float(b1[2]) - float(b1[0]))
    h1 = abs(float(b1[3]) - float(b1[1]))
    w2 = abs(float(b2[2]) - float(b2[0]))
    h2 = abs(float(b2[3]) - float(b2[1]))
    sw = 1.0 - abs(w1 - w2) / max(w1, w2, 1.0)
    sh = 1.0 - abs(h1 - h2) / max(h1, h2, 1.0)
    a1, a2 = w1 * h1, w2 * h2
    sa = 1.0 - abs(a1 - a2) / max(a1, a2, 1.0)
    return float(max(0.0, 0.4 * sw + 0.4 * sh + 0.2 * sa))


def analyze_mirror_partners(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    x_center: float,
    *,
    features: np.ndarray | None = None,
    iou_thresh: float = 0.15,
    sim_fn=None,
) -> list[MirrorPartner]:
    """Per-box facade-center mirror analysis.

    A box is an **outlier** when its mirror partner is missing or belongs to
    another cluster — the cue to merge/reassign after checking similarity.
    """
    if len(boxes) != len(labels):
        raise ValueError("boxes/labels length mismatch")
    n = len(boxes)
    if sim_fn is None:
        sim_fn = (
            lambda i, j: cosine_similarity(features[i], features[j])
            if features is not None
            else box_geometry_similarity(boxes[i], boxes[j])
        )
    out: list[MirrorPartner] = []
    for i in range(n):
        lab = int(labels[i])
        pj, miou = find_mirror_partner_index(
            boxes, i, x_center, iou_thresh=iou_thresh
        )
        if pj is None:
            out.append(
                MirrorPartner(
                    index=i,
                    partner_index=None,
                    mirror_iou=0.0,
                    partner_label=None,
                    same_cluster=False,
                    outlier=True,
                    similarity=None,
                )
            )
            continue
        plab = int(labels[pj])
        same = plab == lab
        sim = float(sim_fn(i, pj))
        out.append(
            MirrorPartner(
                index=i,
                partner_index=pj,
                mirror_iou=miou,
                partner_label=plab,
                same_cluster=same,
                outlier=not same,
                similarity=sim,
            )
        )
    return out


def suggest_mirror_reassignments(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    x_center: float,
    *,
    features: np.ndarray | None = None,
    iou_thresh: float = 0.15,
    sim_thresh: float = 0.7,
    sim_fn=None,
) -> list[dict]:
    """Outlier boxes whose mirror partner is similar → merge/reassign edges.

    Each dict: ``{from_index, from_label, to_index, to_label, mirror_iou, similarity}``.
    """
    partners = analyze_mirror_partners(
        boxes,
        labels,
        x_center,
        features=features,
        iou_thresh=iou_thresh,
        sim_fn=sim_fn,
    )
    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for p in partners:
        if not p.outlier or p.partner_index is None or p.similarity is None:
            continue
        if p.similarity < sim_thresh:
            continue
        i, j = p.index, p.partner_index
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "from_index": i,
                "from_label": int(labels[i]),
                "to_index": j,
                "to_label": p.partner_label,
                "mirror_iou": p.mirror_iou,
                "similarity": p.similarity,
            }
        )
    edges.sort(key=lambda e: (-e["similarity"], -e["mirror_iou"]))
    return edges


def suggest_cluster_merges_from_mirrors(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    x_center: float,
    *,
    features: np.ndarray | None = None,
    iou_thresh: float = 0.15,
    sim_thresh: float = 0.7,
    min_votes: int = 1,
) -> list[tuple[int, int, float, int]]:
    """Aggregate per-box mirror edges into cluster merge pairs.

    Returns ``[(label_a, label_b, mean_similarity, n_votes), ...]`` sorted by
    votes then similarity. Includes symmetric column pairs (t02↔t03) and
    outlier↔partner pairs (t01↔t00).
    """
    edges = suggest_mirror_reassignments(
        boxes,
        labels,
        x_center,
        features=features,
        iou_thresh=iou_thresh,
        sim_thresh=sim_thresh,
    )
    # Also add cluster-level mirror pairs
    for la, lb, ms in suggest_mirror_merges(
        boxes, labels, x_center, iou_thresh=iou_thresh, min_score=0.5
    ):
        edges.append(
            {
                "from_label": la,
                "to_label": lb,
                "similarity": ms,
                "mirror_iou": ms,
                "from_index": -1,
                "to_index": -1,
            }
        )
    tally: dict[tuple[int, int], list[float]] = {}
    for e in edges:
        la, lb = int(e["from_label"]), int(e["to_label"])
        if la == lb:
            continue
        key = (min(la, lb), max(la, lb))
        tally.setdefault(key, []).append(float(e["similarity"]))
    out: list[tuple[int, int, float, int]] = []
    for (la, lb), sims in tally.items():
        if len(sims) < min_votes:
            continue
        out.append((la, lb, float(np.mean(sims)), len(sims)))
    out.sort(key=lambda t: (-t[3], -t[2]))
    return out
