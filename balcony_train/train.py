"""Train multitask kind, material, floor, enclosure, and skin heads on frozen DINOv2.

Saves ``checkpoints/railing_best.pt``. Material loss applies only to open_work.
Floor / enclosure / kind / skin losses apply only where that label is set.

``--init-ckpt`` loads an existing checkpoint.
``--heads-only floor,skin`` trains just those heads. ``--floor-only`` is the
same as ``--heads-only floor``.
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
    ENCLOSURE_IGNORE_INDEX,
    ENCLOSURE_SKINS,
    ENCLOSURES,
    FLOOR_IGNORE_INDEX,
    FLOOR_SHAPES,
    KIND_IGNORE_INDEX,
    MATERIAL_IGNORE_INDEX,
    MATERIALS,
    SKIN_IGNORE_INDEX,
    class_counts,
    enclosure_counts,
    floor_counts,
    material_counts,
    skin_counts,
)
from balcony_train.model import (  # noqa: E402
    RailingClassifier,
    apply_skin_checkpoint,
    load_backbone,
    save_checkpoint,
)
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
    floor_correct = 0
    floor_total = 0
    enc_correct = 0
    enc_total = 0
    skin_correct = 0
    skin_total = 0
    for images, kind_y, mat_y, floor_y, enc_y, skin_y in loader:
        images = images.to(device)
        kind_y = kind_y.to(device)
        mat_y = mat_y.to(device)
        floor_y = floor_y.to(device)
        enc_y = enc_y.to(device)
        skin_y = skin_y.to(device)
        kind_logits, mat_logits, floor_logits, enc_logits, skin_logits = model(images)
        kind_mask = kind_y != KIND_IGNORE_INDEX
        kind_pred = kind_logits.argmax(dim=1)
        kind_correct += int((kind_pred[kind_mask] == kind_y[kind_mask]).sum().item())
        kind_total += int(kind_mask.sum().item())
        enc_mask = enc_y != ENCLOSURE_IGNORE_INDEX
        if bool(enc_mask.any()):
            enc_pred = enc_logits.argmax(dim=1)
            enc_correct += int((enc_pred[enc_mask] == enc_y[enc_mask]).sum().item())
            enc_total += int(enc_mask.sum().item())
        mask = mat_y != MATERIAL_IGNORE_INDEX
        if bool(mask.any()):
            mat_pred = mat_logits.argmax(dim=1)
            mat_correct += int((mat_pred[mask] == mat_y[mask]).sum().item())
            mat_total += int(mask.sum().item())
        floor_mask = floor_y != FLOOR_IGNORE_INDEX
        if bool(floor_mask.any()):
            floor_pred = floor_logits.argmax(dim=1)
            floor_correct += int((floor_pred[floor_mask] == floor_y[floor_mask]).sum().item())
            floor_total += int(floor_mask.sum().item())
        skin_mask = skin_y != SKIN_IGNORE_INDEX
        if bool(skin_mask.any()):
            skin_pred = skin_logits.argmax(dim=1)
            skin_correct += int((skin_pred[skin_mask] == skin_y[skin_mask]).sum().item())
            skin_total += int(skin_mask.sum().item())
    return {
        "kind_acc": kind_correct / max(kind_total, 1),
        "material_acc": mat_correct / max(mat_total, 1) if mat_total else float("nan"),
        "floor_acc": floor_correct / max(floor_total, 1) if floor_total else float("nan"),
        "enclosure_acc": enc_correct / max(enc_total, 1) if enc_total else float("nan"),
        "skin_acc": skin_correct / max(skin_total, 1) if skin_total else float("nan"),
        "n_kind": float(kind_total),
        "n_material": float(mat_total),
        "n_floor": float(floor_total),
        "n_enclosure": float(enc_total),
        "n_skin": float(skin_total),
    }


def _print_confusion(model: RailingClassifier, loader: DataLoader, device: torch.device) -> None:
    """Print floor and skin counts so a small enclosed set is readable."""
    floor_conf = torch.zeros((len(FLOOR_SHAPES), len(FLOOR_SHAPES)), dtype=torch.int64)
    skin_conf = torch.zeros((len(ENCLOSURE_SKINS), len(ENCLOSURE_SKINS)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for images, _kind_y, _mat_y, floor_y, _enc_y, skin_y in loader:
            _k, _m, floor_logits, _e, skin_logits = model(images.to(device))
            floor_pred = floor_logits.argmax(dim=1).cpu()
            skin_pred = skin_logits.argmax(dim=1).cpu()
            for t, p in zip(floor_y.tolist(), floor_pred.tolist()):
                if int(t) != FLOOR_IGNORE_INDEX:
                    floor_conf[int(t), int(p)] += 1
            for t, p in zip(skin_y.tolist(), skin_pred.tolist()):
                if int(t) != SKIN_IGNORE_INDEX:
                    skin_conf[int(t), int(p)] += 1
    if int(floor_conf.sum().item()):
        print("floor rows=true cols=pred  " + "  ".join(FLOOR_SHAPES))
        for i, name in enumerate(FLOOR_SHAPES):
            row = " ".join(f"{int(v):4d}" for v in floor_conf[i].tolist())
            print(f"  {name:16s} {row}")
    if int(skin_conf.sum().item()):
        print("skin rows=true cols=pred  " + "  ".join(ENCLOSURE_SKINS))
        for i, name in enumerate(ENCLOSURE_SKINS):
            row = " ".join(f"{int(v):4d}" for v in skin_conf[i].tolist())
            print(f"  {name:16s} {row}")


HEAD_ONLY_NAMES = ("floor", "skin")


def selected_head_names(args: argparse.Namespace) -> set[str]:
    """Heads that should receive gradients. Empty means train every head."""
    names = {part.strip() for part in str(args.heads_only or "").split(",") if part.strip()}
    if args.floor_only:
        names.add("floor")
    unknown = names - set(HEAD_ONLY_NAMES)
    if unknown:
        raise SystemExit(f"unknown --heads-only {sorted(unknown)}; expected floor,skin")
    return names


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
    ap.add_argument("--floor-loss-weight", type=float, default=1.0)
    ap.add_argument("--enclosure-loss-weight", type=float, default=1.0)
    ap.add_argument("--skin-loss-weight", type=float, default=1.0)
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
        "--init-ckpt",
        type=Path,
        default=None,
        help="load kind and material heads from this checkpoint before training",
    )
    ap.add_argument(
        "--floor-only",
        action="store_true",
        help="alias for --heads-only floor (requires --init-ckpt)",
    )
    ap.add_argument(
        "--heads-only",
        default="",
        help="comma-separated heads to train: floor,skin (requires --init-ckpt)",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def load_init_heads(model: RailingClassifier, ckpt_path: Path) -> bool:
    """Load kind and material heads. Return True when a matching floor head loads."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"init checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = tuple(ckpt.get("classes") or ())
    if saved and saved != tuple(model.classes):
        raise SystemExit(f"init ckpt classes {saved} != {tuple(model.classes)}")
    if "kind_head" in ckpt:
        model.kind_head.load_state_dict(ckpt["kind_head"])
    elif "head" in ckpt:
        model.kind_head.load_state_dict(ckpt["head"])
    else:
        raise SystemExit(f"init checkpoint missing kind head: {ckpt_path}")
    mats = tuple(ckpt.get("materials") or ())
    if "material_head" not in ckpt or mats != tuple(model.materials):
        raise SystemExit(f"init checkpoint missing material head: {ckpt_path}")
    model.material_head.load_state_dict(ckpt["material_head"])
    encs = tuple(ckpt.get("enclosures") or ())
    if "enclosure_head" in ckpt and encs == tuple(model.enclosures):
        model.enclosure_head.load_state_dict(ckpt["enclosure_head"])
    floors = tuple(ckpt.get("floor_shapes") or ())
    had_floor = False
    if "floor_head" in ckpt and floors == tuple(model.floor_shapes):
        model.floor_head.load_state_dict(ckpt["floor_head"])
        had_floor = True
    apply_skin_checkpoint(model, ckpt)
    return had_floor


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
    shape_counts = floor_counts(samples)
    enc_counts = enclosure_counts(samples)
    print(
        f"labeled kind: {counts}  material(open_work): {mat_counts}  "
        f"floor: {shape_counts}  enclosure: {enc_counts}"
    )
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
    print(f"  per-fold enclosure: {record.get('split_enclosure_counts')}")

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
    train_floor_counts = floor_counts(train_rows)
    train_enc_counts = enclosure_counts(train_rows)
    train_skin_counts = skin_counts(train_rows)
    only = selected_head_names(args)
    if only and args.init_ckpt is None:
        raise SystemExit("--heads-only requires --init-ckpt")
    if "floor" in only and sum(int(train_floor_counts[name]) for name in FLOOR_SHAPES) < 1:
        raise SystemExit(
            "no floor_shape labels in the train split; label floors before training the floor head"
        )
    if "skin" in only and sum(int(train_skin_counts[name]) for name in ENCLOSURE_SKINS) < 1:
        raise SystemExit(
            "no skin labels in the train split; label enclosed skins before training the skin head"
        )

    model = RailingClassifier(pretrained=True, freeze_backbone=True)
    load_backbone(model, device)
    if args.init_ckpt is not None:
        had_floor = load_init_heads(model, args.init_ckpt)
        print(
            f"init={args.init_ckpt}  "
            f"floor_head={'loaded' if had_floor else 'new'}"
        )
    head_params = {
        "kind": model.kind_head,
        "material": model.material_head,
        "floor": model.floor_head,
        "enclosure": model.enclosure_head,
        "skin": model.skin_head,
    }
    if only:
        for name, head in head_params.items():
            if name not in only:
                for p in head.parameters():
                    p.requires_grad = False
        train_params = [p for name in only for p in head_params[name].parameters()]
        print(f"training {','.join(sorted(only))} only; other heads stay fixed")
    else:
        train_params = [p for head in head_params.values() for p in head.parameters()]
        if sum(int(train_skin_counts[name]) for name in ENCLOSURE_SKINS) > 0:
            model._skin_trained = True  # type: ignore[attr-defined]
    if "skin" in only:
        model._skin_trained = True  # type: ignore[attr-defined]
    heads = tuple(head_params.values())
    for head in heads:
        head.train()
    opt = torch.optim.Adam(train_params, lr=args.lr)
    n = max(sum(int(train_counts[c]) for c in CLASSES), 1)
    k = max(len(CLASSES), 1)
    kind_weight = torch.tensor(
        [n / (k * max(int(train_counts[c]), 1)) for c in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    kind_loss_fn = nn.CrossEntropyLoss(weight=kind_weight, ignore_index=KIND_IGNORE_INDEX)
    enc_n = max(sum(int(train_enc_counts[e]) for e in ENCLOSURES), 1)
    enc_weight = torch.tensor(
        [enc_n / (len(ENCLOSURES) * max(int(train_enc_counts[e]), 1)) for e in ENCLOSURES],
        dtype=torch.float32,
        device=device,
    )
    enc_loss_fn = nn.CrossEntropyLoss(weight=enc_weight, ignore_index=ENCLOSURE_IGNORE_INDEX)
    enc_w = float(args.enclosure_loss_weight)

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
    floor_n = max(sum(int(train_floor_counts[name]) for name in FLOOR_SHAPES), 1)
    floor_k = max(len(FLOOR_SHAPES), 1)
    floor_weight = torch.tensor(
        [floor_n / (floor_k * max(int(train_floor_counts[name]), 1)) for name in FLOOR_SHAPES],
        dtype=torch.float32,
        device=device,
    )
    floor_loss_fn = nn.CrossEntropyLoss(
        weight=floor_weight, ignore_index=FLOOR_IGNORE_INDEX
    )
    skin_n = max(sum(int(train_skin_counts[name]) for name in ENCLOSURE_SKINS), 1)
    skin_k = max(len(ENCLOSURE_SKINS), 1)
    skin_weight = torch.tensor(
        [skin_n / (skin_k * max(int(train_skin_counts[name]), 1)) for name in ENCLOSURE_SKINS],
        dtype=torch.float32,
        device=device,
    )
    skin_loss_fn = nn.CrossEntropyLoss(weight=skin_weight, ignore_index=SKIN_IGNORE_INDEX)
    mat_w = float(args.material_loss_weight)
    floor_w = float(args.floor_loss_weight)
    skin_w = float(args.skin_loss_weight)

    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        for head in heads:
            head.train()
        model.backbone.eval()
        running = 0.0
        n_batch = 0
        for images, kind_y, mat_y, floor_y, enc_y, skin_y in train_loader:
            images = images.to(device)
            kind_y = kind_y.to(device)
            mat_y = mat_y.to(device)
            floor_y = floor_y.to(device)
            enc_y = enc_y.to(device)
            skin_y = skin_y.to(device)
            kind_logits, mat_logits, floor_logits, enc_logits, skin_logits = model(images)
            has_floor = bool((floor_y != FLOOR_IGNORE_INDEX).any())
            has_skin = bool((skin_y != SKIN_IGNORE_INDEX).any())
            terms = []
            if (not only or "kind" in only) and bool((kind_y != KIND_IGNORE_INDEX).any()):
                terms.append(kind_loss_fn(kind_logits, kind_y))
            if (not only or "material" in only) and bool((mat_y != MATERIAL_IGNORE_INDEX).any()):
                terms.append(mat_w * mat_loss_fn(mat_logits, mat_y))
            if (not only or "floor" in only) and has_floor:
                terms.append(floor_w * floor_loss_fn(floor_logits, floor_y))
            if (not only or "enclosure" in only) and bool((enc_y != ENCLOSURE_IGNORE_INDEX).any()):
                terms.append(enc_w * enc_loss_fn(enc_logits, enc_y))
            if (not only or "skin" in only) and has_skin:
                terms.append(skin_w * skin_loss_fn(skin_logits, skin_y))
            if not terms:
                continue
            loss = terms[0] if len(terms) == 1 else sum(terms)
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
        floor_acc = metrics["floor_acc"]
        enc_acc = metrics["enclosure_acc"]
        skin_acc = metrics["skin_acc"]
        parts = [f"kind_acc={kind_acc:.3f}"]
        weights = [(kind_acc, 0.4)]
        if mat_acc == mat_acc:
            parts.append(f"mat_acc={mat_acc:.3f}")
            weights.append((mat_acc, 0.2))
        else:
            parts.append("mat_acc=n/a")
        if floor_acc == floor_acc:
            parts.append(f"floor_acc={floor_acc:.3f}")
            weights.append((floor_acc, 0.2))
        else:
            parts.append("floor_acc=n/a")
        if enc_acc == enc_acc:
            parts.append(f"enc_acc={enc_acc:.3f}")
            weights.append((enc_acc, 0.2))
        else:
            parts.append("enc_acc=n/a")
        if skin_acc == skin_acc:
            parts.append(f"skin_acc={skin_acc:.3f} n={int(metrics['n_skin'])}")
            weights.append((skin_acc, 0.2))
        else:
            parts.append("skin_acc=n/a")
        weight_sum = sum(w for _acc, w in weights)
        score = sum(acc * w for acc, w in weights) / weight_sum
        extra = "  ".join(parts)
        tag = "val" if val_loader is not None else "train"
        print(f"epoch {epoch:03d}  loss={train_loss:.4f}  {tag}_{extra}")
        if score >= best_score:
            best_score = score
            best_state = [
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                for head in heads
            ]

    if best_state is not None:
        for head, state in zip(heads, best_state):
            head.load_state_dict(state)
    _print_confusion(model, loader, device)
    save_checkpoint(
        model,
        args.out,
        val_score=best_score,
        class_counts=counts,
        material_counts=mat_counts,
        floor_counts=shape_counts,
        enclosure_counts=enc_counts,
        skin_counts=skin_counts(samples),
        train_counts=train_counts,
        train_material_counts=train_mat_counts,
        train_floor_counts=train_floor_counts,
        train_enclosure_counts=train_enc_counts,
        split_file=str(args.split_file),
        epochs=args.epochs,
    )
    print(f"wrote {args.out}  best_val_score={best_score:.3f}")
    print(f"held-out test n={len(test_idx)} — run evaluate.py --split test")


if __name__ == "__main__":
    main()
