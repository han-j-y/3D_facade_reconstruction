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


if __name__ == "__main__":
    unittest.main()
