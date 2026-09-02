"""Stratified train/val/test splits for railing classifier crops."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from balcony_train.labels import CLASSES, class_counts, iter_labeled_samples

SplitName = Literal["train", "val", "test"]

DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1


def _class_sizes(n: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """Per-class train/val/test counts (at least one train; val/test when n allows)."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        n_test = min(1, n - 1)
        n_val = min(1, n - n_test - 1)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
    return n_train, n_val, n_test


def stratified_split_indices(
    samples: list[tuple[Path, int]],
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    seed: int = 42,
) -> dict[SplitName, list[int]]:
    """Stratified split indices into train / val / test."""
    by_label: dict[int, list[int]] = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(samples):
        by_label[int(label)].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    rng = random.Random(int(seed))

    for label in sorted(by_label):
        indices = by_label[label]
        if not indices:
            continue
        ordered = list(indices)
        rng.shuffle(ordered)
        n_train, n_val, n_test = _class_sizes(
            len(ordered), train_frac, val_frac, test_frac
        )
        train_idx.extend(ordered[:n_train])
        val_idx.extend(ordered[n_train : n_train + n_val])
        test_idx.extend(ordered[n_train + n_val : n_train + n_val + n_test])

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _rel_path(path: Path, base: Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(rel: str, base: Path) -> Path:
    p = Path(rel)
    return p if p.is_file() else base / rel


def build_split_record(
    samples: list[tuple[Path, int]],
    indices: dict[SplitName, list[int]],
    *,
    data_dir: Path,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, Any]:
    """JSON-serializable split manifest."""
    base = Path(data_dir).resolve()
    splits: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "val", "test"):
        rows: list[dict[str, str]] = []
        for idx in indices[name]:
            path, label = samples[idx]
            rows.append(
                {
                    "path": _rel_path(path, base),
                    "label": CLASSES[int(label)],
                }
            )
        splits[name] = rows
    return {
        "version": 1,
        "seed": int(seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "data_dir": str(base),
        "counts": class_counts(samples),
        "split_counts": {
            name: class_counts([(samples[i][0], samples[i][1]) for i in indices[name]])
            for name in ("train", "val", "test")
        },
        "splits": splits,
    }


def save_split(record: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_split(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def samples_from_split_record(
    record: dict[str, Any],
    split: SplitName,
) -> list[tuple[Path, int]]:
    """Load (path, label_idx) for one fold from a saved split.json."""
    base = Path(record["data_dir"])
    rows = record["splits"][split]
    out: list[tuple[Path, int]] = []
    for row in rows:
        path = _resolve_path(row["path"], base)
        label = CLASSES.index(str(row["label"]))
        out.append((path, label))
    return out


def make_or_load_split(
    data_dir: Path,
    split_path: Path,
    *,
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    refresh: bool = False,
) -> tuple[dict[str, Any], list[tuple[Path, int]]]:
    """Create stratified split (or load existing) and return record + all samples."""
    data_dir = Path(data_dir)
    split_path = Path(split_path)
    samples = iter_labeled_samples(data_dir)
    if not samples:
        raise ValueError(f"no labeled images under {data_dir}")

    if split_path.is_file() and not refresh:
        record = load_split(split_path)
        return record, samples

    indices = stratified_split_indices(
        samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    record = build_split_record(
        samples,
        indices,
        data_dir=data_dir,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    save_split(record, split_path)
    return record, samples


def indices_from_record(
    record: dict[str, Any],
    samples: list[tuple[Path, int]],
) -> dict[SplitName, list[int]]:
    """Map saved split paths back to indices in ``samples`` (order may differ)."""
    path_to_idx = {Path(p).resolve(): i for i, (p, _) in enumerate(samples)}
    base = Path(record["data_dir"])
    out: dict[SplitName, list[int]] = {"train": [], "val": [], "test": []}
    for name in ("train", "val", "test"):
        for row in record["splits"][name]:
            path = _resolve_path(row["path"], base).resolve()
            if path not in path_to_idx:
                raise ValueError(f"split path not in current dataset: {path}")
            out[name].append(path_to_idx[path])
    return out
