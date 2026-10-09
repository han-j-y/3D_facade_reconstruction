"""Floor-slice and column-snap for enclosed volumes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from enclosed import (  # noqa: E402
    drop_open_under_enclosed,
    enclosed_ir,
    ensure_da3_depth,
    overlapping_floors,
    place_enclosed_volumes,
    slice_volume,
    snap_columns,
)
from merge_dsl import merge_balcony_into_windows_dsl  # noqa: E402


def _grid() -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    floor_y = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
    bay_x = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
    return floor_y, bay_x


def _dsl() -> dict:
    return {
        "schema": "facade_recovery_dsl_v1",
        "meta": {
            "image_size": [300, 300],
            "columns_xy": [[0, 100], [100, 200], [200, 300]],
            "floors_y": [[0, 100], [100, 200], [200, 300]],
        },
        "layout": {
            "floors": [{"id": 0}, {"id": 1}, {"id": 2}],
            "bays": [{"id": 0}, {"id": 1}, {"id": 2}],
            "placement": [],
        },
        "window_types": [],
        "instances": [
            {"floor": 0, "bay": 0, "box_xyxy": [10, 10, 90, 90]},
            {"floor": 1, "bay": 1, "box_xyxy": [110, 110, 190, 190]},
            {"floor": 2, "bay": 1, "box_xyxy": [110, 210, 190, 290]},
        ],
    }


class EnclosedPlaceTests(unittest.TestCase):
    def test_three_storey_volume_slices_three_floors(self) -> None:
        floor_y, bay_x = _grid()
        rows = slice_volume([110, 20, 190, 280], floor_y, bay_x)
        self.assertEqual([r["floor"] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]["bay_start"], 1)
        self.assertEqual(rows[0]["bay_end"], 1)

    def test_cornice_sliver_does_not_add_a_floor(self) -> None:
        floor_y, bay_x = _grid()
        rows = slice_volume([110, 90, 190, 195], floor_y, bay_x)
        self.assertEqual([r["floor"] for r in rows], [1])

    def test_one_column_when_neighbor_is_a_sliver(self) -> None:
        _floor_y, bay_x = _grid()
        loc = snap_columns([20, 0, 105, 50], bay_x)
        self.assertEqual(loc["bay_start"], 0)
        self.assertEqual(loc["bay_end"], 0)
        self.assertEqual(loc["bays"], [0])

    def test_two_columns_extend_to_full_span(self) -> None:
        _floor_y, bay_x = _grid()
        loc = snap_columns([20, 0, 190, 50], bay_x)
        self.assertEqual(loc["bay_start"], 0)
        self.assertEqual(loc["bay_end"], 1)
        self.assertEqual(loc["bays"], [0, 1])

    def test_open_on_same_cell_dropped_open_above_kept(self) -> None:
        enclosed = [{"floor": 1, "bay_start": 1, "bay_end": 1, "bays": [1]}]
        open_units = [
            {"floor": 1, "bay_start": 1, "bay_end": 1, "bays": [1], "unit_id": 0},
            {"floor": 0, "bay_start": 1, "bay_end": 1, "bays": [1], "unit_id": 1},
        ]
        kept, dropped = drop_open_under_enclosed(open_units, enclosed)
        self.assertEqual([u["unit_id"] for u in dropped], [0])
        self.assertEqual([u["unit_id"] for u in kept], [1])

    def test_open_box_on_volume_dropped_even_if_other_bay(self) -> None:
        enclosed = [
            {
                "floor": 1,
                "bay_start": 1,
                "bay_end": 1,
                "bays": [1],
                "box_xyxy": [100, 100, 200, 200],
                "volume_box_xyxy": [100, 50, 200, 250],
            }
        ]
        on_volume = {
            "floor": 1,
            "bay_start": 0,
            "bay_end": 0,
            "bays": [0],
            "box_xyxy": [110, 120, 190, 180],
            "unit_id": 0,
        }
        beside = {
            "floor": 1,
            "bay_start": 2,
            "bay_end": 2,
            "bays": [2],
            "box_xyxy": [220, 120, 280, 180],
            "unit_id": 1,
        }
        below = {
            "floor": 2,
            "bay_start": 1,
            "bay_end": 1,
            "bays": [1],
            "box_xyxy": [110, 260, 190, 300],
            "unit_id": 2,
        }
        terrace_above = {
            "floor": 0,
            "bay_start": 1,
            "bay_end": 1,
            "bays": [1],
            "box_xyxy": [110, 55, 190, 95],
            "unit_id": 3,
        }
        kept, dropped = drop_open_under_enclosed(
            [on_volume, beside, below, terrace_above], enclosed
        )
        self.assertEqual([u["unit_id"] for u in dropped], [0])
        self.assertEqual([u["unit_id"] for u in kept], [1, 2, 3])

    def test_place_attaches_enclosed_ir(self) -> None:
        units = place_enclosed_volumes([[110, 20, 190, 280]], _dsl())
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0]["source"], "enclosed_volume")
        self.assertEqual(units[0]["structure_ir"]["enclosure"], "enclosed")
        self.assertNotIn("railing", units[0]["structure_ir"])

    def test_merge_open_uses_bay_center_when_enclosed_present(self) -> None:
        dsl = _dsl()
        dsl["instances"] = [
            {"unit_id": 14, "floor": 1, "bay": 1, "box_xyxy": [110, 110, 190, 190]},
        ]
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 0,
                "bays": [0],
                "bays_center": [0],
                "bay": 0,
                "box_xyxy": [10, 110, 90, 180],
                "structure_ir": {"railing": {"kind": "open_work", "material": "metal"}},
            },
            {
                "unit_id": 1,
                "source": "enclosed_volume",
                "floor": 1,
                "bay_start": 1,
                "bay_end": 1,
                "bays": [1],
                "bays_center": [1],
                "bay": 1,
                "box_xyxy": [110, 100, 190, 200],
                "structure_ir": enclosed_ir(),
            },
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[
                {"name": "balc_open_work_metal", "structure_ir": units[0]["structure_ir"]},
                {"name": "balc_enclosed", "structure_ir": enclosed_ir()},
            ],
            units=units,
            center_mode="window",
        )
        recs = out["layout"]["balconies"]
        open_rec = next(r for r in recs if r["type"] != "balc_enclosed")
        self.assertNotIn("window_cx_norm", open_rec)
        self.assertIn("bay_cx_norm", open_rec)
        self.assertAlmostEqual(open_rec["bay_cx_norm"], 50 / 300, places=5)
        self.assertEqual(out["meta"]["balcony_center_mode"], "bay")

    def test_merge_drops_open_sitting_on_enclosed_box(self) -> None:
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 0,
                "bays": [0],
                "bay": 0,
                "box_xyxy": [120, 120, 180, 180],
                "structure_ir": {"railing": {"kind": "open_work"}},
            },
            {
                "unit_id": 1,
                "source": "enclosed_volume",
                "floor": 1,
                "bay_start": 1,
                "bay_end": 1,
                "bays": [1],
                "bay": 1,
                "box_xyxy": [100, 100, 200, 200],
                "volume_box_xyxy": [100, 100, 200, 200],
                "structure_ir": enclosed_ir(),
            },
        ]
        out = merge_balcony_into_windows_dsl(
            _dsl(),
            balcony_types=[{"name": "balc_enclosed", "structure_ir": enclosed_ir()}],
            units=units,
        )
        types = [r["type"] for r in out["layout"]["balconies"]]
        self.assertEqual(types, ["balc_enclosed"])

    def test_merge_omits_photo_width_for_enclosed(self) -> None:
        units = place_enclosed_volumes([[110, 20, 190, 280]], _dsl())
        out = merge_balcony_into_windows_dsl(
            _dsl(),
            balcony_types=[{"name": "balc_enclosed", "structure_ir": enclosed_ir()}],
            units=units,
        )
        recs = out["layout"]["balconies"]
        self.assertEqual(len(recs), 3)
        for rec in recs:
            self.assertEqual(rec["type"], "balc_enclosed")
            self.assertNotIn("width_norm", rec)
            self.assertEqual(rec["bay_start"], 1)
            self.assertEqual(rec["bay_end"], 1)

    def test_overlapping_floors_fallback_to_best(self) -> None:
        floor_y, _bay_x = _grid()
        self.assertEqual(overlapping_floors([10, 10, 20, 20], floor_y), [0])

    def test_ensure_da3_depth_reuses_existing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "cmp_x.jpg"
            photo.write_bytes(b"x")
            depth_dir = root / "depth"
            depth_dir.mkdir()
            npy = depth_dir / "cmp_x_depth.npy"
            npy.write_bytes(b"n")
            with patch("enclosed.subprocess.run") as run:
                out = ensure_da3_depth(photo, depth_dir, device="cuda")
            self.assertEqual(out, npy)
            run.assert_not_called()

    def test_ensure_da3_depth_infers_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "cmp_x.jpg"
            photo.write_bytes(b"x")
            depth_dir = root / "depth"

            def _write(_cmd, **_kw):
                (depth_dir / "cmp_x_depth.npy").write_bytes(b"n")
                return None

            with patch("enclosed.subprocess.run", side_effect=_write) as run:
                out = ensure_da3_depth(photo, depth_dir, device="cpu")
            self.assertEqual(out, depth_dir / "cmp_x_depth.npy")
            run.assert_called_once()
            cmd = run.call_args[0][0]
            self.assertIn("--image", cmd)
            self.assertIn(str(photo), cmd)

    def test_ensure_da3_depth_none_when_infer_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "cmp_x.jpg"
            photo.write_bytes(b"x")
            with patch("enclosed.subprocess.run", side_effect=OSError("no gpu")):
                self.assertIsNone(ensure_da3_depth(photo, root / "depth"))


if __name__ == "__main__":
    unittest.main()
