"""Default paths for railing classifier data and checkpoint."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops"
DEFAULT_LABELS_JSONL = ROOT / "runs" / "balcony_clf" / "labels.jsonl"
DEFAULT_SPLIT_JSON = ROOT / "runs" / "balcony_clf" / "split.json"
DEFAULT_RAILING_CKPT = ROOT / "checkpoints" / "railing_best.pt"
BATCH_RUNS_DIR = ROOT / "runs" / "batch"
