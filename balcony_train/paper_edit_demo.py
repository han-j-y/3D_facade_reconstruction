"""Editing demo for the paper: change the recovered DSL, recompile, count what moved.

Each edit touches a few DSL fields. Because repeated elements share one type
definition, every instance of that type changes with it. For each edit we
report the DSL fields changed and the 3D instances whose definition changed,
and render the result next to the original.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/paper_edit_demo.py --stem cmp_x0005 --render
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "runs" / "paper_eval"
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def leaf_diff(a: Any, b: Any) -> int:
    """Number of leaf values that differ (added / removed keys count once each)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(leaf_diff(a.get(k), b.get(k)) for k in set(a) | set(b))
    if isinstance(a, list) and isinstance(b, list):
        n = sum(leaf_diff(x, y) for x, y in zip(a, b))
        return n + abs(len(a) - len(b))
    return int(a != b)


def program(dsl: dict) -> dict:
    """The part a user edits: layout plus the type library."""
    return {
        "layout": {k: dsl["layout"].get(k) for k in ("floors", "bays", "placement", "placement_spans", "balconies")},
        "window_types": {t["name"]: t["structure_ir"] for t in dsl.get("window_types", [])},
        "balcony_types": {t["name"]: t["structure_ir"] for t in dsl.get("balcony_types", [])},
    }


def type_by_name(dsl: dict, key: str, name: str) -> dict:
    return next(t for t in dsl[key] if t["name"] == name)


def placed(dsl: dict) -> list[tuple[str, str]]:
    """Every placed element as (kind, type name), one entry per instance."""
    out = []
    for row in dsl["layout"]["placement"]:
        out += [("window", t) for t in row if t is not None]
    out += [("balcony", b["type"]) for b in dsl["layout"].get("balconies", [])]
    return out


def changed_instances(before: dict, after: dict) -> int:
    """Instances whose resolved definition (type IR at its slot) differs."""
    def resolve(dsl: dict) -> list[str]:
        wt = {t["name"]: t["structure_ir"] for t in dsl.get("window_types", [])}
        bt = {t["name"]: t["structure_ir"] for t in dsl.get("balcony_types", [])}
        rows = []
        for r, row in enumerate(dsl["layout"]["placement"]):
            for c, t in enumerate(row):
                rows.append(f"w{r},{c}:" + json.dumps(wt.get(t), sort_keys=True) if t else f"w{r},{c}:-")
        for b in dsl["layout"].get("balconies", []):
            rows.append(f"b{b['floor']},{b['bay_start']}:" + json.dumps(bt.get(b["type"]), sort_keys=True))
        return rows

    a, b = resolve(before), resolve(after)
    n = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return n


# ---------------------------------------------------------------- edits


def edit_window_arch(dsl: dict) -> str:
    """Asset level: the most repeated window type gets an arched head and a 2x2 muntin grid."""
    t = max(dsl["window_types"], key=lambda t: t["n_instances"])
    ir = t["structure_ir"]
    ir["boundary"]["shape"] = "arch_head"
    ir["boundary"]["params"] = {"width": 1.0, "height": 1.6}
    ir["panes"]["rules"]["root"] = {"op": "grid", "vertical": 1, "horizontal": 1}
    return f"{t['name']}: rectangle -> arch_head, 2x2 panes"


def edit_railing_solid(dsl: dict) -> str:
    """Asset level: every open-work balcony type gets a solid parapet."""
    names = []
    for t in dsl.get("balcony_types", []):
        rail = t["structure_ir"].get("railing")
        if rail and rail.get("kind") == "open_work":
            rail["kind"] = "solid"
            rail.pop("material", None)
            names.append(t["name"])
    return f"{', '.join(names)}: railing open_work -> solid"


def edit_enclosure_punched(dsl: dict) -> str:
    """Asset level: the enclosed volumes become rectangular, punched-window oriels."""
    t = type_by_name(dsl, "balcony_types", "balc_enclosed")
    t["structure_ir"]["walls"] = {"infill": "wall"}
    t["structure_ir"]["floor"]["shape"] = "rectangle"
    return "balc_enclosed: circle curtain wall -> rectangle punched window"


def edit_floor_type(dsl: dict) -> str:
    """Facade level: one floor reuses the type of the floor below (layout tokens only)."""
    rows = dsl["layout"]["placement"]
    k = max(
        range(len(rows) - 1),
        key=lambda r: sum(a is not None and b is not None and a != b for a, b in zip(rows[r], rows[r + 1])),
    )
    below = rows[k + 1]
    rows[k] = [b if (a is not None and b is not None) else a for a, b in zip(rows[k], below)]
    return f"layout row {k}: window types copied from row {k + 1}"


EDITS: dict[str, Callable[[dict], str]] = {
    "window_arch": edit_window_arch,
    "railing_solid": edit_railing_solid,
    "enclosure_punched": edit_enclosure_punched,
    "floor_retype": edit_floor_type,
}


def render(dsl_path: Path, out_dir: Path) -> None:
    env = {**os.environ, "BLENDER": BLENDER, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_facade.py"), "--recovery", str(dsl_path),
         "--out-dir", str(out_dir), "--render"],
        check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="cmp_x0005")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    base_path = PAPER / args.stem / "balcony" / "facade_dsl_with_balconies.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    out_root = PAPER / "edits" / args.stem
    rows = []
    for name, fn in EDITS.items():
        dsl = copy.deepcopy(base)
        what = fn(dsl)
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        path = d / "facade_dsl_with_balconies.json"
        path.write_text(json.dumps(dsl, indent=1), encoding="utf-8")
        fields = leaf_diff(program(base), program(dsl))
        inst = changed_instances(base, dsl)
        rows.append({"edit": name, "what": what, "dsl_fields": fields, "instances_changed": inst,
                     "instances_total": len(placed(dsl))})
        print(f"{name:<18} fields={fields:<3} instances={inst:<3} / {len(placed(dsl))}  {what}", flush=True)
        if args.render:
            render(path, d / "blender")
    (out_root / "edits.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
