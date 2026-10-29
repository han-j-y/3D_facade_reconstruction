"""Accuracy / confusion for multitask kind + material checkpoint.

By default only real facade crops whose filename starts with ``cmp`` are
scored (excludes FLUX.2 synth ``flux2_*``). Use ``--include-synth`` to score
everything in the fold. ``--list`` / ``--list-out`` / ``--copy-to`` print,
save, or copy the images that would be evaluated (no GPU / checkpoint needed).
``--errors-to`` scores the checkpoint, then copies mistakes under a new
``YYYY-MM-DD_HHMMSS`` folder there, as ``kind/true-{label}__pred-{label}/``,
``material/true-{label}__pred-{label}/``,
``floor/true-{label}__pred-{label}/``,
``enclosure/true-{label}__pred-{label}/``, and
``skin/true-{label}__pred-{label}/``.
``mistakes.txt`` in that folder lists every copy.

``--photo-dir`` keeps only crops whose photo stem is in that folder
(``cmp_x0001_bal_000.png`` belongs to ``cmp_x0001``). Use it to score
``data/extended_selected`` from the CMP-XML crop labels::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate.py --device cuda --ckpt checkpoints/railing_flux640.pt --data-dir runs/balcony_clf/crops_ext_xml --labels runs/balcony_clf/crops_ext_xml/labels.jsonl --split all --include-synth --photo-dir data/extended_selected --errors-to runs/balcony_clf/eval_view_ext_selected/Wrong
"""

from __future__ import annotations

import argparse
import re
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
    ENCLOSURE_IGNORE_INDEX,
    ENCLOSURES,
    ENCLOSURE_SKINS,
    FLOOR_IGNORE_INDEX,
    FLOOR_SHAPES,
    KIND_IGNORE_INDEX,
    LABEL_STRUCTURES,
    SKIN_IGNORE_INDEX,
    STRUCTURE_IGNORE_INDEX,
    MATERIAL_IGNORE_INDEX,
    MATERIALS,
    MultitaskSample,
    class_counts,
    enclosure_counts,
    floor_counts,
    material_counts,
    skin_counts,
    structure_counts,
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


_CROP_PHOTO_STEM = re.compile(r"^(?P<stem>.+)_bal_\d+$", re.IGNORECASE)
_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".xml"}


def photo_stem_from_crop(path: Path) -> str | None:
    """Photo id for a balcony crop named ``{photo}_bal_000.png``."""
    match = _CROP_PHOTO_STEM.match(Path(path).stem)
    if match is None:
        return None
    return match.group("stem")


def photo_stems_in_dir(photo_dir: Path) -> set[str]:
    """Lowercase stems of facade files in ``photo_dir``."""
    photo_dir = Path(photo_dir)
    if not photo_dir.is_dir():
        return set()
    return {
        path.stem.lower()
        for path in photo_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _PHOTO_SUFFIXES
    }


def filter_indices_by_photo_stems(
    samples: list[MultitaskSample],
    indices: list[int],
    photo_stems: set[str],
) -> list[int]:
    """Keep crops whose ``{photo}_bal_NNN`` name is in ``photo_stems``."""
    needles = {stem.lower() for stem in photo_stems}
    kept: list[int] = []
    for i in indices:
        stem = photo_stem_from_crop(Path(samples[i][0]))
        if stem is not None and stem.lower() in needles:
            kept.append(i)
    return kept


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
    """One eval crop with kind, material, floor, enclosure, and skin predictions."""

    path: Path
    kind_true: int
    kind_pred: int
    mat_true: int
    mat_pred: int
    floor_true: int = FLOOR_IGNORE_INDEX
    floor_pred: int = FLOOR_IGNORE_INDEX
    enc_true: int = ENCLOSURE_IGNORE_INDEX
    enc_pred: int = ENCLOSURE_IGNORE_INDEX
    skin_true: int = SKIN_IGNORE_INDEX
    skin_pred: int = SKIN_IGNORE_INDEX


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
    """Copy kind, material, floor, enclosure, and skin mistakes.

    Layout::

        dest_dir/YYYY-MM-DD_HHMMSS/kind/true-{label}__pred-{label}/<image>
        dest_dir/YYYY-MM-DD_HHMMSS/material/true-{label}__pred-{label}/<image>
        dest_dir/YYYY-MM-DD_HHMMSS/floor/true-{label}__pred-{label}/<image>
        dest_dir/YYYY-MM-DD_HHMMSS/enclosure/true-{label}__pred-{label}/<image>
        dest_dir/YYYY-MM-DD_HHMMSS/skin/true-{label}__pred-{label}/<image>

    ``mistakes.txt`` lists ``head true pred filename`` for every copy.
    ``stamp`` overrides the clock (tests). A second run in the same second
    gets a ``_2`` suffix. Material is skipped when the true kind is not
    open_work (``mat_true == MATERIAL_IGNORE_INDEX``). Floor, enclosure, and
    skin are skipped when that label is ignore. An image wrong on more than
    one head is copied into each tree. Returns the number of file copies.
    """
    buckets: dict[Path, list[Path]] = defaultdict(list)
    lines: list[str] = []

    def take(head: str, true_name: str, pred_name: str, path: Path) -> None:
        rel = Path(head) / misclass_folder_name(true_name, pred_name)
        buckets[rel].append(path)
        lines.append(f"{head}\t{true_name}\t{pred_name}\t{path.name}")

    for row in rows:
        if int(row.kind_true) != KIND_IGNORE_INDEX and int(row.kind_true) != int(row.kind_pred):
            take(
                "kind",
                CLASSES[int(row.kind_true)],
                CLASSES[int(row.kind_pred)],
                Path(row.path),
            )
        if int(row.mat_true) != MATERIAL_IGNORE_INDEX and int(row.mat_true) != int(
            row.mat_pred
        ):
            take(
                "material",
                MATERIALS[int(row.mat_true)],
                MATERIALS[int(row.mat_pred)],
                Path(row.path),
            )
        if int(row.floor_true) != FLOOR_IGNORE_INDEX and int(row.floor_true) != int(
            row.floor_pred
        ):
            take(
                "floor",
                FLOOR_SHAPES[int(row.floor_true)],
                FLOOR_SHAPES[int(row.floor_pred)],
                Path(row.path),
            )
        if int(row.enc_true) != ENCLOSURE_IGNORE_INDEX and int(row.enc_true) != int(
            row.enc_pred
        ):
            take(
                "enclosure",
                ENCLOSURES[int(row.enc_true)],
                ENCLOSURES[int(row.enc_pred)],
                Path(row.path),
            )
        if int(row.skin_true) != SKIN_IGNORE_INDEX and int(row.skin_true) != int(
            row.skin_pred
        ):
            take(
                "skin",
                ENCLOSURE_SKINS[int(row.skin_true)],
                ENCLOSURE_SKINS[int(row.skin_pred)],
                Path(row.path),
            )

    run_dir = error_run_dir(dest_dir, stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel in sorted(buckets):
        n += copy_eval_images(buckets[rel], run_dir / rel)
    list_path = run_dir / "mistakes.txt"
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not buckets:
        print(f"no misclassifications -> {run_dir.resolve()}", flush=True)
    else:
        print(f"misclassified copies={n} -> {run_dir.resolve()}", flush=True)
        print(f"list -> {list_path.resolve()}", flush=True)
    return n


def resolve_eval_subset(
    *,
    data_dir: Path,
    split_file: Path,
    labels: Path,
    split: str,
    stem_prefix: str,
    include_synth: bool,
    photo_dir: Path | None = None,
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
    if photo_dir is not None:
        stems = photo_stems_in_dir(photo_dir)
        if not stems:
            raise SystemExit(f"no facade files in {photo_dir}")
        n_before = len(eval_idx)
        eval_idx = filter_indices_by_photo_stems(samples, eval_idx, stems)
        print(
            f"filter photos in {photo_dir}: {len(stems)} photos, "
            f"crops {n_before} -> {len(eval_idx)}"
        )
        if not eval_idx:
            raise SystemExit(
                f"no labeled crops from photos in {photo_dir} "
                f"(had {n_before} before the photo filter)"
            )
        split_label = f"{split_label}|photos={Path(photo_dir).name}"
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
        "--photo-dir",
        type=Path,
        default=None,
        help=(
            "score only crops from facade photos in this folder "
            "(crop cmp_x0001_bal_000 belongs to photo cmp_x0001)"
        ),
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
            "kind/true-{label}__pred-{label}/, "
            "material/true-{label}__pred-{label}/, and "
            "floor/true-{label}__pred-{label}/"
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
        photo_dir=Path(args.photo_dir) if args.photo_dir is not None else None,
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
    shape_counts = floor_counts(subset)
    enc_counts = enclosure_counts(subset)
    device = torch.device(args.device)
    model = load_classifier(args.ckpt, device)
    loader = DataLoader(
        Subset(MultitaskCropDataset(samples), eval_idx),
        batch_size=min(args.batch_size, len(eval_idx)),
        shuffle=False,
    )
    kind_conf = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    mat_conf = torch.zeros((len(MATERIALS), len(MATERIALS)), dtype=torch.int64)
    floor_conf = torch.zeros((len(FLOOR_SHAPES), len(FLOOR_SHAPES)), dtype=torch.int64)
    enc_conf = torch.zeros((len(ENCLOSURES), len(ENCLOSURES)), dtype=torch.int64)
    skin_conf = torch.zeros((len(ENCLOSURE_SKINS), len(ENCLOSURE_SKINS)), dtype=torch.int64)
    floor_trained = bool(getattr(model, "_floor_trained", False))
    enc_trained = bool(getattr(model, "_enclosure_trained", False))
    skin_trained = bool(getattr(model, "_skin_trained", False))
    structure_conf = torch.zeros((len(LABEL_STRUCTURES), len(LABEL_STRUCTURES)), dtype=torch.int64)
    structure_trained = bool(getattr(model, "_structure_trained", False))
    scored: list[ScoredCrop] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for images, kind_y, mat_y, floor_y, enc_y, skin_y, structure_y in loader:
            kind_logits, mat_logits, floor_logits, enc_logits, skin_logits, structure_logits = model(
                images.to(device)
            )
            kind_pred = kind_logits.argmax(dim=1).cpu()
            mat_pred = mat_logits.argmax(dim=1).cpu()
            floor_pred = floor_logits.argmax(dim=1).cpu()
            enc_pred = enc_logits.argmax(dim=1).cpu()
            skin_pred = skin_logits.argmax(dim=1).cpu()
            kt_list = kind_y.tolist()
            kp_list = kind_pred.tolist()
            mt_list = mat_y.tolist()
            mp_list = mat_pred.tolist()
            ft_list = floor_y.tolist()
            fp_list = floor_pred.tolist()
            et_list = enc_y.tolist()
            ep_list = enc_pred.tolist()
            st_list = skin_y.tolist()
            sp_list = skin_pred.tolist()
            for i, (t, p) in enumerate(zip(kt_list, kp_list)):
                if int(t) != KIND_IGNORE_INDEX:
                    kind_conf[t, p] += 1
                src_i = eval_idx[cursor + i]
                if floor_trained:
                    floor_true = int(ft_list[i])
                    floor_pred_i = int(fp_list[i])
                else:
                    floor_true = FLOOR_IGNORE_INDEX
                    floor_pred_i = FLOOR_IGNORE_INDEX
                if enc_trained:
                    enc_true = int(et_list[i])
                    enc_pred_i = int(ep_list[i])
                    if enc_true != ENCLOSURE_IGNORE_INDEX:
                        enc_conf[enc_true, enc_pred_i] += 1
                else:
                    enc_true = ENCLOSURE_IGNORE_INDEX
                    enc_pred_i = ENCLOSURE_IGNORE_INDEX
                if skin_trained:
                    skin_true = int(st_list[i])
                    skin_pred_i = int(sp_list[i])
                else:
                    skin_true = SKIN_IGNORE_INDEX
                    skin_pred_i = SKIN_IGNORE_INDEX
                scored.append(
                    ScoredCrop(
                        Path(samples[src_i][0]),
                        int(t),
                        int(p),
                        int(mt_list[i]),
                        int(mp_list[i]),
                        floor_true,
                        floor_pred_i,
                        enc_true,
                        enc_pred_i,
                        skin_true,
                        skin_pred_i,
                    )
                )
            for t, p in zip(mt_list, mp_list):
                if int(t) == MATERIAL_IGNORE_INDEX:
                    continue
                mat_conf[t, p] += 1
            if floor_trained:
                for t, p in zip(ft_list, fp_list):
                    if int(t) == FLOOR_IGNORE_INDEX:
                        continue
                    floor_conf[int(t), int(p)] += 1
            if skin_trained:
                for t, p in zip(skin_y.tolist(), skin_pred.tolist()):
                    if int(t) == SKIN_IGNORE_INDEX:
                        continue
                    skin_conf[int(t), int(p)] += 1
            if structure_trained:
                structure_pred = structure_logits.argmax(dim=1).cpu()
                for t, p in zip(structure_y.tolist(), structure_pred.tolist()):
                    if int(t) == STRUCTURE_IGNORE_INDEX:
                        continue
                    structure_conf[int(t), int(p)] += 1
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

    floor_total = int(floor_conf.sum().item())
    floor_correct = int(floor_conf.diag().sum().item())
    if floor_trained:
        print(
            f"floor n={floor_total}  counts={shape_counts}  "
            f"floor_acc={floor_correct / max(floor_total, 1):.3f}"
        )
        print("floor rows=true cols=pred  " + "  ".join(FLOOR_SHAPES))
        for i, name in enumerate(FLOOR_SHAPES):
            row = " ".join(f"{int(v):4d}" for v in floor_conf[i].tolist())
            print(f"  {name:16s} {row}")
    else:
        print(f"floor head missing or not trained on {FLOOR_SHAPES}; retrain to evaluate floor")

    if enc_trained:
        enc_total = int(enc_conf.sum().item())
        enc_correct = int(enc_conf.diag().sum().item())
        print(
            f"enclosure n={enc_total}  counts={enc_counts}  "
            f"enc_acc={enc_correct / max(enc_total, 1):.3f}"
        )
        print("enclosure rows=true cols=pred  " + "  ".join(ENCLOSURES))
        for i, name in enumerate(ENCLOSURES):
            row = " ".join(f"{int(v):4d}" for v in enc_conf[i].tolist())
            print(f"  {name:16s} {row}")
    else:
        print("enclosure head not in checkpoint")

    if skin_trained:
        skin_total = int(skin_conf.sum().item())
        skin_correct = int(skin_conf.diag().sum().item())
        print(
            f"skin n={skin_total}  counts={skin_counts(subset)}  "
            f"skin_acc={skin_correct / max(skin_total, 1):.3f}"
        )
        print("skin rows=true cols=pred  " + "  ".join(ENCLOSURE_SKINS))
        for i, name in enumerate(ENCLOSURE_SKINS):
            row = " ".join(f"{int(v):4d}" for v in skin_conf[i].tolist())
            print(f"  {name:16s} {row}")
    else:
        print("skin head not in checkpoint")

    if structure_trained:
        structure_total = int(structure_conf.sum().item())
        structure_correct = int(structure_conf.diag().sum().item())
        print(
            f"structure n={structure_total}  counts={structure_counts(subset)}  "
            f"structure_acc={structure_correct / max(structure_total, 1):.3f}"
        )
        print("structure rows=true cols=pred  " + "  ".join(LABEL_STRUCTURES))
        for i, name in enumerate(LABEL_STRUCTURES):
            row = " ".join(f"{int(v):4d}" for v in structure_conf[i].tolist())
            print(f"  {name:16s} {row}")
    else:
        print("structure head not in checkpoint")

    if args.errors_to is not None:
        copy_misclassified_images(scored, Path(args.errors_to))


if __name__ == "__main__":
    main()
