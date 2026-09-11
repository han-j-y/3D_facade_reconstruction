"""Accuracy / confusion for multitask kind + material checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _, samples = make_or_load_split(
        args.data_dir,
        args.split_file,
        labels_jsonl=args.labels,
        refresh=False,
    )
    if not samples:
        raise SystemExit(f"no labeled images under {args.data_dir}")

    if args.split == "all":
        eval_idx = list(range(len(samples)))
        split_label = "all"
    else:
        if not args.split_file.is_file():
            raise SystemExit(
                f"split file not found: {args.split_file}\n"
                "run train.py first (or train.py --refresh-split after adding images)"
            )
        record = load_split(args.split_file)
        fold = indices_from_record(record, samples)
        split_name: SplitName = args.split
        eval_idx = fold[split_name]
        split_label = split_name
        if not eval_idx:
            raise SystemExit(f"split {split_name!r} is empty in {args.split_file}")

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
    model.eval()
    with torch.no_grad():
        for images, kind_y, mat_y in loader:
            kind_logits, mat_logits = model(images.to(device))
            kind_pred = kind_logits.argmax(dim=1).cpu()
            mat_pred = mat_logits.argmax(dim=1).cpu()
            for t, p in zip(kind_y.tolist(), kind_pred.tolist()):
                kind_conf[t, p] += 1
            for t, p in zip(mat_y.tolist(), mat_pred.tolist()):
                if int(t) == MATERIAL_IGNORE_INDEX:
                    continue
                mat_conf[t, p] += 1

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


if __name__ == "__main__":
    main()
