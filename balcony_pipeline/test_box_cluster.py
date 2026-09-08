"""Window type clustering: box geometry then optional appearance splits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.box_cluster import (  # noqa: E402
    box_geometry_feats,
    cluster_box_geometry_pipeline,
    cluster_within_groups,
)


class _Backend:
    def select_k(self, feats: np.ndarray, method: str, k_max: int, seed: int) -> int:
        return min(2, max(1, k_max))

    def cluster_features(
        self, feats: np.ndarray, method: str, k: int, seed: int
    ) -> np.ndarray:
        if k <= 1:
            return np.zeros(len(feats), dtype=np.int32)
        col = feats[:, 0] if feats.shape[1] >= 1 else feats[:, 0]
        thr = float(np.median(col))
        return (col > thr).astype(np.int32)


class BoxClusterTests(unittest.TestCase):
    def test_wide_vs_tall_boxes_split(self) -> None:
        boxes = [
            [0, 0, 80, 40],
            [90, 0, 170, 40],
            [0, 50, 30, 130],
            [40, 50, 70, 130],
        ]
        labels, _ = cluster_box_geometry_pipeline(
            boxes,
            (200, 200),
            _Backend(),
            method="gmm_diag",
            k_max=4,
            seed=0,
            conservative_k=False,
            merge_near=False,
            snap_wh=False,
        )
        self.assertEqual(len(set(int(x) for x in labels.tolist())), 2)

    def test_within_group_keeps_cohesive_feats(self) -> None:
        box_labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        feats = np.array(
            [
                [1.0, 0.0],
                [1.01, 0.01],
                [0.99, -0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        labels, info = cluster_within_groups(
            feats, box_labels, _Backend(), k_max_within=3, seed=0
        )
        self.assertEqual(len(info), 2)
        self.assertGreaterEqual(len(set(int(x) for x in labels.tolist())), 1)

    def test_geometry_feats_shape(self) -> None:
        x = box_geometry_feats([[0, 0, 10, 20], [0, 0, 30, 10]], (100, 100))
        self.assertEqual(x.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
