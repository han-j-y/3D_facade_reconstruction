"""Tests for domain AUC helpers (no GPU / DINOv2)."""

from __future__ import annotations

import unittest

import numpy as np

from balcony_train.domain_auc_real_vs_synth import (
    domain_auc_from_features,
    labels_from_sources,
)


class DomainAucHelperTests(unittest.TestCase):
    def test_labels_from_sources(self) -> None:
        y = labels_from_sources(["real", "synth", "real"])
        np.testing.assert_array_equal(y, [0, 1, 0])

    def test_domain_auc_separable(self) -> None:
        # Perfectly separable along dim 0.
        rng = np.random.default_rng(0)
        real = rng.normal(loc=0.0, scale=0.1, size=(40, 8))
        synth = rng.normal(loc=5.0, scale=0.1, size=(40, 8))
        x = np.vstack([real, synth])
        y = np.array([0] * 40 + [1] * 40)
        metrics = domain_auc_from_features(x, y, test_frac=0.25, seed=0)
        self.assertGreaterEqual(metrics["auc"], 0.95)
        self.assertEqual(metrics["n_total"], 80)

    def test_domain_auc_chance(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 8))
        y = np.array([0] * 40 + [1] * 40)
        metrics = domain_auc_from_features(x, y, test_frac=0.25, seed=1)
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
