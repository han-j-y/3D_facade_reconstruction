"""BDSL railing kinds: open_work / surface_panel / solid."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from parse_bdsl import parse_bdsl  # noqa: E402


class ParseRailKindTests(unittest.TestCase):
    def _parse(self, kind_line: str) -> str:
        text = (
            "balcony rectangle(width=2.0, depth=1.5)\n"
            "structure projecting\n"
            "enclosure open\n"
            f"{kind_line}\n"
        )
        return str(parse_bdsl(text)["railing"]["kind"])

    def test_catalog_kinds(self) -> None:
        self.assertEqual(self._parse("railing open_work height 1.1"), "open_work")
        self.assertEqual(self._parse("railing surface_panel height 1.1"), "surface_panel")
        self.assertEqual(self._parse("railing solid height 1.1"), "solid")

    def test_aliases(self) -> None:
        self.assertEqual(self._parse("railing baluster"), "open_work")
        self.assertEqual(self._parse("railing metal"), "open_work")
        self.assertEqual(self._parse("railing glass"), "surface_panel")

    def test_trapezoid_floor(self) -> None:
        ir = parse_bdsl(
            "balcony trapezoid(width=2.0, depth=1.5)\n"
            "structure projecting\n"
            "enclosure open\n"
            "railing open_work height 1.1\n"
        )
        self.assertEqual(ir["floor"]["shape"], "trapezoid")

    def test_hexagon_floor(self) -> None:
        ir = parse_bdsl(
            "balcony hexagon(width=3.0, depth=1.5)\n"
            "structure projecting\n"
            "enclosure open\n"
            "railing open_work height 1.1\n"
        )
        self.assertEqual(ir["floor"]["shape"], "hexagon")

    def test_default_kind(self) -> None:
        ir = parse_bdsl("balcony rectangle(width=1, depth=0.8)\n")
        self.assertEqual(ir["railing"]["kind"], "open_work")

    def test_examples_parse(self) -> None:
        examples = HERE / "examples"
        self.assertEqual(
            parse_bdsl((examples / "example_balcony_projecting.bdsl").read_text(encoding="utf-8"))[
                "railing"
            ]["kind"],
            "open_work",
        )
        self.assertEqual(
            parse_bdsl((examples / "example_balcony_solid.bdsl").read_text(encoding="utf-8"))[
                "railing"
            ]["kind"],
            "solid",
        )
        self.assertEqual(
            parse_bdsl(
                (examples / "example_balcony_surface_panel.bdsl").read_text(encoding="utf-8")
            )["railing"]["kind"],
            "surface_panel",
        )


class ParseEnclosureTests(unittest.TestCase):
    def test_open_has_no_enclosure_blocks(self) -> None:
        ir = parse_bdsl("balcony rectangle(width=2, depth=1)\nenclosure open\n")
        self.assertEqual(ir["enclosure"], "open")
        self.assertNotIn("columns", ir)
        self.assertNotIn("walls", ir)

    def test_half_enclosed_columns(self) -> None:
        ir = parse_bdsl(
            "balcony rectangle(width=2.4, depth=1.0)\n"
            "enclosure half_enclosed\n"
            "columns:\n"
            "  count 3\n"
            "  width 0.3\n"
            "  style round\n"
            "railing open_work material masonry\n"
        )
        self.assertEqual(ir["enclosure"], "half_enclosed")
        self.assertEqual(ir["columns"], {"count": 3, "width": 0.3, "style": "round"})
        self.assertEqual(ir["railing"]["kind"], "open_work")

    def test_half_enclosed_column_defaults(self) -> None:
        ir = parse_bdsl("balcony hexagon(width=3, depth=1)\nenclosure half_enclosed\n")
        self.assertEqual(ir["columns"], {"count": 0, "width": 0.25, "style": "square"})

    def test_enclosed_walls(self) -> None:
        ir = parse_bdsl(
            "balcony hexagon(width=2.6, depth=1.0)\n"
            "enclosure enclosed\n"
            "walls:\n"
            "  infill wall\n"
        )
        self.assertEqual(ir["walls"], {"infill": "wall"})
        self.assertNotIn("columns", ir)
        default = parse_bdsl("balcony rectangle(width=2, depth=1)\nenclosure enclosed\n")
        self.assertEqual(default["walls"], {"infill": "glass"})

    def test_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_bdsl("balcony rectangle(width=2, depth=1)\nenclosure loggia\n")
        with self.assertRaises(ValueError):
            parse_bdsl(
                "balcony rectangle(width=2, depth=1)\nenclosure half_enclosed\n"
                "columns:\n  style fluted\n"
            )
        with self.assertRaises(ValueError):
            parse_bdsl(
                "balcony rectangle(width=2, depth=1)\nenclosure enclosed\n"
                "walls:\n  infill brick\n"
            )

    def test_enclosure_examples_parse(self) -> None:
        examples = HERE / "examples"
        half = parse_bdsl(
            (examples / "example_balcony_half_enclosed.bdsl").read_text(encoding="utf-8")
        )
        self.assertEqual(half["enclosure"], "half_enclosed")
        oriel = parse_bdsl(
            (examples / "example_balcony_enclosed_hexagon.bdsl").read_text(encoding="utf-8")
        )
        self.assertEqual(oriel["enclosure"], "enclosed")
        self.assertEqual(oriel["floor"]["shape"], "hexagon")


if __name__ == "__main__":
    unittest.main()
