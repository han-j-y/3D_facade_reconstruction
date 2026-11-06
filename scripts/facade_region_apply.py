#!/usr/bin/env python3
"""Put the facade region from runs_facade_region/regions.json into the validation DSLs, windows only and with balconies.
Writes runs_region/<id>/facade_dsl.json and <id>/balcony/facade_dsl_with_balconies.json."""
import json, sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(E / "facade_recovery"))
from facade_region import choose_facade_region
R = json.load(open(E / "runs_facade_region/regions.json")); O = E / "runs_region"; n = 0
for t, r in R.items():
    D = json.load(open(E / "runs_val30_combined" / t / "facade_dsl.json"))
    box, source = choose_facade_region([(d["box"], d["score"]) for d in r["facade"]], [u["box_xyxy"] for u in D["instances"]], tuple(r["size"]))
    print(t, source, [round(v) for v in box], "grid", r["grid"]); n += source == "sam3"
    D["meta"]["facade_region_xyxy"] = box; D["meta"]["facade_region_source"] = source
    d = O / t; d.mkdir(parents=True, exist_ok=True); json.dump(D, open(d / "facade_dsl.json", "w"), indent=1)
    b = E / "runs_window_map" / t / "balcony/facade_dsl_with_balconies.json"
    if b.exists():
        B = json.load(open(b)); B["meta"]["facade_region_xyxy"] = box; B["meta"]["facade_region_source"] = source
        (d / "balcony").mkdir(exist_ok=True); json.dump(B, open(d / "balcony/facade_dsl_with_balconies.json", "w"), indent=1)
print("sam3", n, "of", len(R))
