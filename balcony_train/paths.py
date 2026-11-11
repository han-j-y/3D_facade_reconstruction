"""Default paths for railing classifier data and checkpoint."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops"
DEFAULT_LABELS_JSONL = ROOT / "runs" / "balcony_clf" / "labels.jsonl"
DEFAULT_SPLIT_JSON = ROOT / "runs" / "balcony_clf" / "split.json"
# v5 (2026-11-10, user): = railing_curve30_lowres_x0005_20261110.pt, v4's recipe plus
# 30 curved-front user crops (127 px wide) and the 16 cmp_x0005 unit crops. Trained
# without the test100 photos and without FLUX.2. Its skin head saw no curtain-wall
# crop and always answers punched_window. Previous defaults: railing_test100_v4.pt
# (2026-11-02), railing_best.pt.
DEFAULT_RAILING_CKPT = ROOT / "checkpoints" / "railing_test100_v5.pt"
# Structure head (projecting / juliet) on the same frozen DINOv2. It is loaded
# only when the railing checkpoint has none (railing_test100_v5.pt has its own).
DEFAULT_STRUCTURE_CKPT = ROOT / "checkpoints" / "railing_structure.pt"
BATCH_RUNS_DIR = ROOT / "runs" / "batch"
# CMP extended (cmp_x) balcony boxes cut from the official XML, for labeling.
EXT_PHOTO_DIR = ROOT / "data" / "extended"
EXT_XML_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops_ext_xml"
EXT_XML_LABELS = EXT_XML_CROPS_DIR / "labels.jsonl"
EXT_XML_MANIFEST = EXT_XML_CROPS_DIR / "manifest.jsonl"
EXT_XML_NONE = EXT_XML_CROPS_DIR / "none.jsonl"
