"""Train multitask kind + material heads on frozen DINOv2.

Saves ``checkpoints/railing_best.pt``. Material loss applies only to open_work.
"""

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

from balcony_train.dataset import MultitaskCropDataset  # noqa: E402
from balcony_train.labels import (  # noqa: E402
    CLASSES,
    MATERIAL_IGNORE_INDEX,
    MATERIALS,
    class_counts,
    material_counts,
)
from balcony_train.model import RailingClassifier, load_backbone, save_checkpoint  # noqa: E402
from balcony_train.paths import (  # noqa: E402
    DEFAULT_CROPS_DIR,
    DEFAULT_LABELS_JSONL,
    DEFAULT_RAILING_CKPT,
    DEFAULT_SPLIT_JSON,
)
from balcony_train.splits import (  # noqa: E402
    DEFAULT_TEST_FRAC,
    DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
    indices_from_record,
    make_or_load_split,
)


@torch.no_grad()
def _metrics(
    model: RailingClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    kind_correct = 0
    kind_total = 0
    mat_correct = 0
    mat_total = 0
    for images, kind_y, mat_y in loader:
        images = images.to(device)
        kind_y = kind_y.to(device)
        mat_y = mat_y.to(device)
        kind_logits, mat_logits = model(images)
        kind_pred = kind_logits.argmax(dim=1)
        kind_correct += int((kind_pred == kind_y).sum().item())
        kind_total += int(kind_y.numel())
        mask = mat_y != MATERIAL_IGNORE_INDEX
        if bool(mask.any()):
            mat_pred = mat_logits.argmax(dim=1)
            mat_correct += int((mat_pred[mask] == mat_y[mask]).sum().item())
            mat_total += int(mask.sum().item())
    return {
        "kind_acc": kind_correct / max(kind_total, 1),
        "material_acc": mat_correct / max(mat_total, 1) if mat_total else float("nan"),
        "n_kind": float(kind_total),
        "n_material": float(mat_total),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--material-loss-weight", type=float, default=1.0)
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
    record, samples = make_or_load_split(
        args.data_dir,
        args.split_file,
        labels_jsonl=args.labels,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        refresh=args.refresh_split,
    )
    counts = class_counts(samples)
    mat_counts = material_counts(samples)
    print(f"labeled kind: {counts}  material(open_work): {mat_counts}")
    print(f"labels={args.labels}  dir={args.data_dir}")
    if min(counts.values()) < 1:
        raise SystemExit(
            f"need at least one labeled image in each of {CLASSES} under {args.data_dir}"
        )
    if min(mat_counts.values()) < 1:
        print(
            f"warn: missing open_work material class in {MATERIALS}; "
            "material head will be weak until both metal and masonry exist"
        )

    fold = indices_from_record(record, samples)
    train_idx = fold["train"]
    val_idx = fold["val"]
    test_idx = fold["test"]
    print(f"split: {args.split_file}")
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  per-fold kind: {record.get('split_counts')}")
    print(f"  per-fold material: {record.get('split_material_counts')}")

    device = torch.device(args.device)
    ds = MultitaskCropDataset(samples)
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

    train_rows = [samples[i] for i in train_idx]
    train_counts = class_counts(train_rows)
    train_mat_counts = material_counts(train_rows)
    model = RailingClassifier(pretrained=True, freeze_backbone=True)
    load_backbone(model, device)
    model.kind_head.train()
    model.material_head.train()
    opt = torch.optim.Adam(
        list(model.kind_head.parameters()) + list(model.material_head.parameters()),
        lr=args.lr,
    )
    n = max(sum(int(train_counts[c]) for c in CLASSES), 1)
    k = max(len(CLASSES), 1)
    kind_weight = torch.tensor(
        [n / (k * max(int(train_counts[c]), 1)) for c in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    kind_loss_fn = nn.CrossEntropyLoss(weight=kind_weight)

    mat_n = max(sum(int(train_mat_counts[m]) for m in MATERIALS), 1)
    mat_k = max(len(MATERIALS), 1)
    mat_weight = torch.tensor(
        [mat_n / (mat_k * max(int(train_mat_counts[m]), 1)) for m in MATERIALS],
        dtype=torch.float32,
        device=device,
    )
    mat_loss_fn = nn.CrossEntropyLoss(
        weight=mat_weight, ignore_index=MATERIAL_IGNORE_INDEX
    )
    mat_w = float(args.material_loss_weight)

    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.kind_head.train()
        model.material_head.train()
        model.backbone.eval()
        running = 0.0
        n_batch = 0
        for images, kind_y, mat_y in train_loader:
            images = images.to(device)
            kind_y = kind_y.to(device)
            mat_y = mat_y.to(device)
            kind_logits, mat_logits = model(images)
            loss = kind_loss_fn(kind_logits, kind_y)
            if bool((mat_y != MATERIAL_IGNORE_INDEX).any()):
                loss = loss + mat_w * mat_loss_fn(mat_logits, mat_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batch += 1
        train_loss = running / max(n_batch, 1)
        loader = val_loader if val_loader is not None else train_loader
        metrics = _metrics(model, loader, device)
        kind_acc = metrics["kind_acc"]
        mat_acc = metrics["material_acc"]
        # Prefer kind; blend material when available.
        if mat_acc == mat_acc:  # not NaN
            score = 0.7 * kind_acc + 0.3 * mat_acc
            extra = f"kind_acc={kind_acc:.3f}  mat_acc={mat_acc:.3f}"
        else:
            score = kind_acc
            extra = f"kind_acc={kind_acc:.3f}  mat_acc=n/a"
        tag = "val" if val_loader is not None else "train"
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  {tag}_{extra}")
        if score >= best_score:
            best_score = score
            best_state = {
                "kind_head": {
                    k: v.detach().cpu().clone()
                    for k, v in model.kind_head.state_dict().items()
                },
                "material_head": {
                    k: v.detach().cpu().clone()
                    for k, v in model.material_head.state_dict().items()
                },
            }

    if best_state is not None:
        model.kind_head.load_state_dict(best_state["kind_head"])
        model.material_head.load_state_dict(best_state["material_head"])
    save_checkpoint(
        model,
        args.out,
        val_score=best_score,
        class_counts=counts,
        material_counts=mat_counts,
        train_counts=train_counts,
        train_material_counts=train_mat_counts,
        split_file=str(args.split_file),
        epochs=args.epochs,
    )
    print(f"wrote {args.out}  best_val_score={best_score:.3f}")
    print(f"held-out test n={len(test_idx)} — run evaluate.py --split test")


if __name__ == "__main__":
    main()
