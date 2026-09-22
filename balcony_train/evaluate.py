"""Accuracy / confusion for multitask kind + material checkpoint.

By default only real facade crops whose filename starts with ``cmp`` are
scored (excludes FLUX.2 synth ``flux2_*``). Use ``--include-synth`` to score
everything in the fold. ``--list`` / ``--list-out`` / ``--copy-to`` print,
save, or copy the images that would be evaluated (no GPU / checkpoint needed).
``--errors-to`` scores the checkpoint, then copies mistakes under a new
``YYYY-MM-DD_HHMMSS`` folder there, as ``kind/true-{label}__pred-{label}/``
and ``material/true-{label}__pred-{label}/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.dataset import MultitaskCropDataset  # noqa: E402
from balcony_train.labels import (  # noqa: E402
    CLASSES,
    MATERIAL_IGNORE_INDEX,
    MATERIALS,
    MultitaskSample,
    class_counts,
    material_counts,
)
from balcony_train.model import load_classifier  # noqa: E402
from balcony_train.paths import (  # noqa: E402
    DEFAULT_CROPS_DIR,
    DEFAULT_LABELS_JSONL,
    DEFAULT_RAILING_CKPT,
    DEFAULT_SPLIT_JSON,
)
from balcony_train.splits import (  # noqa: E402
    SplitName,
    indices_from_record,
    load_split,
    make_or_load_split,
)


def filter_indices_by_name_prefix(
    samples: list[MultitaskSample],
    indices: list[int],
    prefix: str,
) -> list[int]:
    """Keep indices whose image filename starts with ``prefix`` (case-insensitive).

    Empty ``prefix`` keeps all indices.
    """
    if not str(prefix):
        return list(indices)
    needle = str(prefix).lower()
    kept: list[int] = []
    for i in indices:
        name = Path(samples[i][0]).name.lower()
        if name.startswith(needle):
            kept.append(i)
    return kept


def eval_image_paths(
    samples: list[MultitaskSample],
    indices: list[int],
) -> list[Path]:
    """Absolute paths for the images that would be scored."""
    return [Path(samples[i][0]).resolve() for i in indices]


def write_eval_list(
    paths: list[Path],
    *,
    list_out: Path | None,
    also_print: bool,
) -> None:
    """Print and/or write one absolute path per line."""
    lines = [str(p) for p in paths]
    if also_print:
        for line in lines:
            print(line)
        print(f"n={len(lines)}", flush=True)
    if list_out is not None:
        out = Path(list_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"wrote {len(lines)} paths -> {out.resolve()}", flush=True)


class ScoredCrop(NamedTuple):
    """One eval crop with kind and material predictions."""

    path: Path
    kind_true: int
    kind_pred: int
    mat_true: int
    mat_pred: int


def misclass_folder_name(true_name: str, pred_name: str) -> str:
    """Folder name that states the label and the model's call."""
    return f"true-{true_name}__pred-{pred_name}"


def error_run_stamp(when: datetime | None = None) -> str:
    """Local date and time, safe as a Windows folder name."""
    when = when or datetime.now()
    return when.strftime("%Y-%m-%d_%H%M%S")


def error_run_dir(dest_dir: Path, stamp: str | None = None) -> Path:
    """``dest_dir/<stamp>``, with ``_2``, ``_3``, ... if that name exists."""
    dest_dir = Path(dest_dir)
    name = stamp or error_run_stamp()
    run_dir = dest_dir / name
    if not run_dir.exists():
        return run_dir
    n = 2
    while (dest_dir / f"{name}_{n}").exists():
        n += 1
    return dest_dir / f"{name}_{n}"


def copy_eval_images(paths: list[Path], dest_dir: Path) -> int:
    """Copy eval images into ``dest_dir`` (flat). Returns number copied.

    Name collisions get a ``{parent}__`` prefix (e.g. ``open_work__foo.png``).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in paths:
        src = Path(src)
        if not src.is_file():
            print(f"skip missing: {src}", flush=True)
            continue
        dest = dest_dir / src.name
        if dest.exists():
            try:
                if dest.resolve() == src.resolve():
                    n += 1
                    continue
            except OSError:
                pass
            dest = dest_dir / f"{src.parent.name}__{src.name}"
        shutil.copy2(src, dest)
        n += 1
    print(f"copied {n} images -> {dest_dir.resolve()}", flush=True)
    return n


def copy_misclassified_images(
    rows: list[ScoredCrop],
    dest_dir: Path,
    *,
    stamp: str | None = None,
) -> int:
    """Copy kind and material mistakes under a new timestamp folder.

    Layout::

        dest_dir/YYYY-MM-DD_HHMMSS/kind/true-{label}__pred-{label}/<image>
        dest_dir/YYYY-MM-DD_HHMMSS/material/true-{label}__pred-{label}/<image>

    ``stamp`` overrides the clock (tests). A second run in the same second
    gets a ``_2`` suffix. Material is skipped when the true kind is not
    open_work (``mat_true == MATERIAL_IGNORE_INDEX``). An image wrong on
    both heads is copied into both trees. Returns the number of file copies.
    """
    buckets: dict[Path, list[Path]] = defaultdict(list)
    for row in rows:
        if int(row.kind_true) != int(row.kind_pred):
            rel = Path("kind") / misclass_folder_name(
                CLASSES[int(row.kind_true)],
                CLASSES[int(row.kind_pred)],
            )
            buckets[rel].append(Path(row.path))
        if int(row.mat_true) == MATERIAL_IGNORE_INDEX:
            continue
        if int(row.mat_true) != int(row.mat_pred):
            rel = Path("material") / misclass_folder_name(
                MATERIALS[int(row.mat_true)],
                MATERIALS[int(row.mat_pred)],
            )
            buckets[rel].append(Path(row.path))

    run_dir = error_run_dir(dest_dir, stamp)
    n = 0
    for rel in sorted(buckets):
        n += copy_eval_images(buckets[rel], run_dir / rel)
    if not buckets:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"no misclassifications -> {run_dir.resolve()}", flush=True)
    else:
        print(f"misclassified copies={n} -> {run_dir.resolve()}", flush=True)
    return n


def resolve_eval_subset(
    *,
    data_dir: Path,
    split_file: Path,
    labels: Path,
    split: str,
    stem_prefix: str,
    include_synth: bool,
) -> tuple[list[MultitaskSample], list[int], str]:
    """Return samples, eval indices, and split_label."""
    _, samples = make_or_load_split(
        data_dir,
        split_file,
        labels_jsonl=labels,
        refresh=False,
    )
    if not samples:
        raise SystemExit(f"no labeled images under {data_dir}")

    if split == "all":
        eval_idx = list(range(len(samples)))
        split_label = "all"
    else:
        if not split_file.is_file():
            raise SystemExit(
                f"split file not found: {split_file}\n"
                "run train.py first (or train.py --refresh-split after adding images)"
            )
        record = load_split(split_file)
        fold = indices_from_record(record, samples)
        split_name: SplitName = split  # type: ignore[assignment]
        eval_idx = fold[split_name]
        split_label = split_name
        if not eval_idx:
            raise SystemExit(f"split {split_name!r} is empty in {split_file}")

    prefix = "" if include_synth else str(stem_prefix)
    n_before = len(eval_idx)
    eval_idx = filter_indices_by_name_prefix(samples, eval_idx, prefix)
    if prefix and not eval_idx:
        raise SystemExit(
            f"no images with filename prefix {prefix!r} in split={split_label!r} "
            f"(had {n_before} before filter). "
            "Use --include-synth to include synth, or label more cmp_* crops."
        )
    if prefix:
        print(f"filter filename prefix={prefix!r}: {n_before} -> {len(eval_idx)}")
        split_label = f"{split_label}|prefix={prefix}"
    return samples, eval_idx, split_label


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_JSON)
    ap.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="which fold to evaluate (default: test held-out set)",
    )
    ap.add_argument(
        "--stem-prefix",
        default="cmp",
        help=(
            "only score images whose filename starts with this prefix "
            "(default: cmp = real facade crops; use with --include-synth for all)"
        ),
    )
    ap.add_argument(
        "--include-synth",
        action="store_true",
        help="score all sources in the split (ignore --stem-prefix; includes flux2_*)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print absolute paths of images in this eval set; skip accuracy",
    )
    ap.add_argument(
        "--list-out",
        type=Path,
        default=None,
        help="write those paths to a text file (one per line); implies listing mode",
    )
    ap.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help=(
            "copy eval images into this folder (flat) for browsing; "
            "implies listing mode (skips accuracy)"
        ),
    )
    ap.add_argument(
        "--errors-to",
        type=Path,
        default=None,
        help=(
            "after scoring, copy mistakes under this folder in a new "
            "YYYY-MM-DD_HHMMSS directory, as "
            "kind/true-{label}__pred-{label}/ and "
            "material/true-{label}__pred-{label}/"
        ),
    )
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples, eval_idx, split_label = resolve_eval_subset(
        data_dir=args.data_dir,
        split_file=args.split_file,
        labels=args.labels,
        split=str(args.split),
        stem_prefix=str(args.stem_prefix),
        include_synth=bool(args.include_synth),
    )

    list_only = (
        bool(args.list)
        or args.list_out is not None
        or args.copy_to is not None
    )
    if list_only:
        paths = eval_image_paths(samples, eval_idx)
        # --list prints; --list-out / --copy-to alone avoid dumping every path.
        write_eval_list(
            paths,
            list_out=Path(args.list_out) if args.list_out is not None else None,
            also_print=bool(args.list)
            or (args.list_out is None and args.copy_to is None),
        )
        if args.copy_to is not None:
            copy_eval_images(paths, Path(args.copy_to))
        return

    subset = [samples[i] for i in eval_idx]
    counts = class_counts(subset)
    mat_counts = material_counts(subset)
    device = torch.device(args.device)
    model = load_classifier(args.ckpt, device)
    loader = DataLoader(
        Subset(MultitaskCropDataset(samples), eval_idx),
        batch_size=min(args.batch_size, len(eval_idx)),
        shuffle=False,
    )
    kind_conf = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    mat_conf = torch.zeros((len(MATERIALS), len(MATERIALS)), dtype=torch.int64)
    scored: list[ScoredCrop] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for images, kind_y, mat_y in loader:
            kind_logits, mat_logits = model(images.to(device))
            kind_pred = kind_logits.argmax(dim=1).cpu()
            mat_pred = mat_logits.argmax(dim=1).cpu()
            kt_list = kind_y.tolist()
            kp_list = kind_pred.tolist()
            mt_list = mat_y.tolist()
            mp_list = mat_pred.tolist()
            for i, (t, p) in enumerate(zip(kt_list, kp_list)):
                kind_conf[t, p] += 1
                src_i = eval_idx[cursor + i]
                scored.append(
                    ScoredCrop(
                        Path(samples[src_i][0]),
                        int(t),
                        int(p),
                        int(mt_list[i]),
                        int(mp_list[i]),
                    )
                )
            for t, p in zip(mt_list, mp_list):
                if int(t) == MATERIAL_IGNORE_INDEX:
                    continue
                mat_conf[t, p] += 1
            cursor += len(kt_list)

    kind_total = int(kind_conf.sum().item())
    kind_correct = int(kind_conf.diag().sum().item())
    mat_total = int(mat_conf.sum().item())
    mat_correct = int(mat_conf.diag().sum().item())
    print(f"ckpt={args.ckpt}")
    print(
        f"split={split_label}  kind_n={kind_total}  counts={counts}  "
        f"kind_acc={kind_correct / max(kind_total, 1):.3f}"
    )
    print("kind rows=true cols=pred  " + "  ".join(CLASSES))
    for i, name in enumerate(CLASSES):
        row = " ".join(f"{int(v):4d}" for v in kind_conf[i].tolist())
        print(f"  {name:16s} {row}")

    print(
        f"material(open_work) n={mat_total}  counts={mat_counts}  "
        f"mat_acc={mat_correct / max(mat_total, 1):.3f}"
    )
    print("material rows=true cols=pred  " + "  ".join(MATERIALS))
    for i, name in enumerate(MATERIALS):
        row = " ".join(f"{int(v):4d}" for v in mat_conf[i].tolist())
        print(f"  {name:16s} {row}")

    if args.errors_to is not None:
        copy_misclassified_images(scored, Path(args.errors_to))


if __name__ == "__main__":
    main()
