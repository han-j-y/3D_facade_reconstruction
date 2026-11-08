"""Open-balcony appearance clustering: k = 1 allowed, width rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cluster import (  # noqa: E402
    label_distance,
    link_atoms,
    select_k_allow_one,
    split_by_labels,
    split_by_width,
    window_relations,
)


def _p(kind=(1, 0, 0), material=(1, 0), structure=(1, 0), floor=(1, 0, 0)):
    return {"kind": list(kind), "material": list(material), "structure": list(structure), "floor": list(floor)}


class LabelSplitTests(unittest.TestCase):
    def test_distance_is_largest_head(self) -> None:
        self.assertAlmostEqual(label_distance(_p(), _p(structure=(0.2, 0.8))), 0.8)
        self.assertAlmostEqual(label_distance(_p(), _p()), 0.0)

    def test_material_counts_only_for_open_work(self) -> None:
        a, b = _p(kind=(0, 1, 0), material=(1, 0)), _p(kind=(0, 1, 0), material=(0, 1))
        self.assertAlmostEqual(label_distance(a, b), 0.0)

    def test_triangle_folds_into_rectangle(self) -> None:
        self.assertAlmostEqual(label_distance(_p(floor=(1, 0, 0)), _p(floor=(0, 1, 0))), 0.0)

    def test_split_confident_structure(self) -> None:
        probs = [_p(structure=(0.9, 0.1)), _p(structure=(0.95, 0.05)), _p(structure=(0.1, 0.9)), _p(structure=(0.05, 0.95))]
        out, log = split_by_labels(np.zeros(4), probs, thr=0.5)
        self.assertEqual(out.tolist(), [0, 0, 1, 1])
        self.assertEqual(log[0]["parts"], 2)

    def test_uncertain_kept_together(self) -> None:
        probs = [_p(structure=(0.6, 0.4)), _p(structure=(0.45, 0.55)), _p(structure=(0.55, 0.45))]
        out, _ = split_by_labels(np.zeros(3), probs, thr=0.5)
        self.assertEqual(out.tolist(), [0, 0, 0])

    def test_atoms_not_split(self) -> None:
        probs = [_p(structure=(0.9, 0.1)), _p(structure=(0.1, 0.9))]
        out, _ = split_by_labels(np.zeros(2), probs, thr=0.5, atoms=np.array([0, 0]))
        self.assertEqual(out.tolist(), [0, 0])


class _Backend:
    """Two clusters by the sign of the first feature."""

    def select_k(self, feats, method, k_max, seed):
        return 2

    def cluster_features(self, feats, method, k, seed):
        return (feats[:, 0] > 0).astype(np.int64)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class SplitByWidthTests(unittest.TestCase):
    def test_keeps_type_under_ratio(self) -> None:
        boxes = [[0, 0, 100, 30], [0, 0, 190, 30], [0, 0, 120, 60]]
        self.assertEqual(split_by_width(np.zeros(3), boxes, max_ratio=2.0).tolist(), [0, 0, 0])

    def test_splits_at_largest_gap(self) -> None:
        # 50, 55 | 200: ratio 4, largest log gap between 55 and 200
        boxes = [[0, 0, 50, 20], [0, 0, 200, 20], [0, 0, 55, 20]]
        self.assertEqual(split_by_width(np.zeros(3), boxes, max_ratio=2.0).tolist(), [0, 1, 0])

    def test_exactly_double_is_split(self) -> None:
        boxes = [[0, 0, 50, 20], [0, 0, 100, 20]]
        self.assertEqual(len(set(split_by_width(np.zeros(2), boxes, max_ratio=2.0).tolist())), 2)

    def test_recursive_and_other_types_kept(self) -> None:
        boxes = [[0, 0, 10, 5], [0, 0, 25, 5], [0, 0, 60, 5], [0, 0, 30, 5]]
        out = split_by_width(np.array([0, 0, 0, 1]), boxes, max_ratio=2.0).tolist()
        self.assertEqual(len({out[0], out[1], out[2]}), 3)
        self.assertNotIn(out[3], {out[0], out[1], out[2]})

    def test_off(self) -> None:
        boxes = [[0, 0, 10, 5], [0, 0, 100, 5]]
        self.assertEqual(split_by_width(np.zeros(2), boxes, max_ratio=0).tolist(), [0, 0])


class SelectKAllowOneTests(unittest.TestCase):
    def test_pair_similar_is_one_type(self) -> None:
        f = _unit([[1, 0.1], [1, 0.12]])
        k, labels, _ = select_k_allow_one(f, f, base=_Backend(), k_max=8, seed=0, sil_min=0.25, pair_sim_min=0.8)
        self.assertEqual((k, labels.tolist()), (1, [0, 0]))

    def test_pair_different_is_two_types(self) -> None:
        f = _unit([[1, 0], [0, 1]])
        k, labels, _ = select_k_allow_one(f, f, base=_Backend(), k_max=8, seed=0, sil_min=0.25, pair_sim_min=0.8)
        self.assertEqual((k, labels.tolist()), (2, [0, 1]))

    def test_clear_split_kept(self) -> None:
        f = _unit([[1, 0.05], [1, -0.05], [-1, 0.05], [-1, -0.05]])
        k, labels, sil = select_k_allow_one(f, f, base=_Backend(), k_max=3, seed=0, sil_min=0.25, pair_sim_min=0.8)
        self.assertEqual(k, 2)
        self.assertGreater(sil, 0.25)

    def test_weak_split_is_one_type(self) -> None:
        f = _unit([[0.05, 1], [-0.05, 1], [0.1, 1], [-0.1, 1]])
        k, labels, sil = select_k_allow_one(f, f, base=_Backend(), k_max=3, seed=0, sil_min=0.9, pair_sim_min=0.8)
        self.assertEqual((k, labels.tolist()), (1, [0, 0, 0, 0]))


class LinkAtomTests(unittest.TestCase):
    # val30 v08: balcony below its window (floor 2) above two window grilles (floor 3)
    BOXES = [[627, 654, 683, 681], [489, 826, 546, 872], [630, 827, 688, 871]]
    FLOORS = [2, 3, 3]
    WINDOWS = [
        {"floor": 2, "kind": "window", "box_xyxy": [633, 555, 681, 652]},
        {"floor": 3, "kind": "window", "box_xyxy": [489, 779, 547, 872]},
        {"floor": 3, "kind": "window", "box_xyxy": [629, 778, 688, 872]},
    ]

    def test_column_needs_same_window_relation(self) -> None:
        rel = window_relations(self.BOXES, self.FLOORS, self.WINDOWS)
        atoms, links = link_atoms(self.BOXES, self.FLOORS, rel)
        self.assertEqual(atoms.tolist(), [0, 1, 1])
        self.assertEqual([k for *_, k in links], ["row"])

    def test_column_without_windows_links(self) -> None:
        atoms, _ = link_atoms(self.BOXES, self.FLOORS, [None, None, None])
        self.assertEqual(len(set(atoms.tolist())), 1)

    def test_row_chain_stops_at_width_ratio(self) -> None:
        # val30 v25: tops aligned, widths 95 / 111 / 125 / 286 (two railings in one box)
        boxes = [[80, 669, 175, 768], [231, 669, 342, 745], [384, 672, 509, 770], [510, 673, 796, 772]]
        atoms, _ = link_atoms(boxes, [2, 2, 2, 2], [None] * 4)
        self.assertEqual(atoms.tolist(), [0, 0, 0, 1])

    def test_split_keeps_atoms_whole(self) -> None:
        boxes = [[0, 0, 60, 5], [0, 0, 100, 5], [0, 0, 130, 5]]
        out = split_by_width(np.zeros(3), boxes, max_ratio=2.0, atoms=np.array([0, 0, 1])).tolist()
        self.assertEqual(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
