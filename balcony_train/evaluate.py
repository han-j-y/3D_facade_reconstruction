"""Accuracy / confusion on labeled crops using checkpoints/railing_best.pt."""

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

from balcony_train.dataset import RailingCropDataset  # noqa: E402
from balcony_train.labels import CLASSES, class_counts, iter_labeled_samples  # noqa: E402
from balcony_train.model import load_classifier  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_RAILING_CKPT, DEFAULT_SPLIT_JSON  # noqa: E402
from balcony_train.splits import SplitName, indices_from_record, load_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_CROPS_DIR)
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
    samples = iter_labeled_samples(args.data_dir)
    if not samples:
        raise SystemExit(f"no labeled images under {args.data_dir}/{CLASSES}")

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

    counts = class_counts([samples[i] for i in eval_idx])
    device = torch.device(args.device)
    model = load_classifier(args.ckpt, device)
    loader = DataLoader(
        Subset(RailingCropDataset(samples), eval_idx),
        batch_size=min(args.batch_size, len(eval_idx)),
        shuffle=False,
    )
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(dim=1).cpu()
            for t, p in zip(labels.tolist(), pred.tolist()):
                confusion[t, p] += 1
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    print(f"ckpt={args.ckpt}")
    print(f"split={split_label}  n={total}  counts={counts}  acc={correct / max(total, 1):.3f}")
    print("rows=true cols=pred  " + "  ".join(CLASSES))
    for i, name in enumerate(CLASSES):
        row = " ".join(f"{int(v):4d}" for v in confusion[i].tolist())
        print(f"  {name:16s} {row}")


if __name__ == "__main__":
    main()
