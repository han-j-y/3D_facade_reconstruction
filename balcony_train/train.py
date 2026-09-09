"""Train Linear(384, n_classes) on frozen DINOv2 features. Saves checkpoints/railing_best.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.dataset import RailingCropDataset  # noqa: E402
from balcony_train.labels import CLASSES, class_counts, iter_labeled_samples  # noqa: E402
from balcony_train.model import RailingClassifier, load_backbone, save_checkpoint  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_RAILING_CKPT, DEFAULT_SPLIT_JSON  # noqa: E402
from balcony_train.splits import (  # noqa: E402
    DEFAULT_TEST_FRAC,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
    indices_from_record,
    make_or_load_split,
)


@torch.no_grad()
def _accuracy(model: RailingClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    ap.add_argument(
        "--refresh-split",
        action="store_true",
        help="recompute stratified split.json (e.g. after adding labeled images)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples = iter_labeled_samples(args.data_dir)
    counts = class_counts(samples)
    print(f"labeled: {counts}  dir={args.data_dir}")
    if min(counts.values()) < 1:
        raise SystemExit(
            f"need at least one labeled image in each of {CLASSES} under {args.data_dir}"
        )

    record, samples = make_or_load_split(
        args.data_dir,
        args.split_file,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        refresh=args.refresh_split,
    )
    fold = indices_from_record(record, samples)
    train_idx = fold["train"]
    val_idx = fold["val"]
    test_idx = fold["test"]
    print(f"split: {args.split_file}")
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  per-fold: {record['split_counts']}")

    device = torch.device(args.device)
    ds = RailingCropDataset(samples)
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=min(args.batch_size, max(len(train_idx), 1)),
        shuffle=True,
    )
    val_loader = None
    if val_idx:
        val_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=min(args.batch_size, len(val_idx)),
            shuffle=False,
        )

    train_counts = class_counts([samples[i] for i in train_idx])
    model = RailingClassifier(pretrained=True, freeze_backbone=True)
    load_backbone(model, device)
    model.head.train()
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    n = max(sum(int(train_counts[c]) for c in CLASSES), 1)
    k = max(len(CLASSES), 1)
    weight = torch.tensor(
        [n / (k * max(int(train_counts[c]), 1)) for c in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.head.train()
        model.backbone.eval()
        running = 0.0
        n_batch = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batch += 1
        train_loss = running / max(n_batch, 1)
        if val_loader is not None:
            acc = _accuracy(model, val_loader, device)
            metric = acc
            extra = f"val_acc={acc:.3f}"
        else:
            acc = _accuracy(model, train_loader, device)
            metric = acc
            extra = f"train_acc={acc:.3f}"
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  {extra}")
        if metric >= best_acc:
            best_acc = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}

    if best_state is not None:
        model.head.load_state_dict(best_state)
    save_checkpoint(
        model,
        args.out,
        val_acc=best_acc,
        class_counts=counts,
        train_counts=train_counts,
        split_file=str(args.split_file),
        epochs=args.epochs,
    )
    print(f"wrote {args.out}  best_val_acc={best_acc:.3f}")
    print(f"held-out test n={len(test_idx)} — run evaluate.py --split test")


if __name__ == "__main__":
    main()
