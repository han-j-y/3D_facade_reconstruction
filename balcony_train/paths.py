"""Default paths for railing classifier data and checkpoint."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops"
DEFAULT_LABELS_JSONL = ROOT / "runs" / "balcony_clf" / "labels.jsonl"
DEFAULT_SPLIT_JSON = ROOT / "runs" / "balcony_clf" / "split.json"
DEFAULT_RAILING_CKPT = ROOT / "checkpoints" / "railing_best.pt"
# Structure head (projecting / juliet) on the same frozen DINOv2. The pipeline
# loads only this head from it, so railing_best.pt keeps the other attributes.
DEFAULT_STRUCTURE_CKPT = ROOT / "checkpoints" / "railing_structure.pt"
BATCH_RUNS_DIR = ROOT / "runs" / "batch"
# CMP extended (cmp_x) balcony boxes cut from the official XML, for labeling.
EXT_PHOTO_DIR = ROOT / "data" / "extended"
EXT_XML_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops_ext_xml"
EXT_XML_LABELS = EXT_XML_CROPS_DIR / "labels.jsonl"
EXT_XML_MANIFEST = EXT_XML_CROPS_DIR / "manifest.jsonl"
EXT_XML_NONE = EXT_XML_CROPS_DIR / "none.jsonl"
