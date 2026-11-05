"""Score railing checkpoints on the Base + LSAA 100-photo test set (``runs/test100/crops``).

The crops are the hand-labeled pipeline crops (``label_pipeline_crops_ui.py``),
all open. ``not_balcony.jsonl`` and unsaved crops are left out. The test set
has no triangle floor, so a triangle prediction counts as rectangle.

Per head: confusion block (rows true, cols predicted), then Markdown tables of
per-class precision / recall / F1 and macro, pooled over all crops. ``--by-source`` adds the
cmp_b and LSAA parts as supplementary tables.

Only checkpoints trained without these photos give test scores
(``train.py --exclude-prefix`` with ``runs/test100/train_exclude_prefixes.txt``).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_test100.py checkpoints/railing_test100_heldout.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.dataset import MultitaskCropDataset  # noqa: E402
from balcony_train.labels import CLASSES, FLOOR_SHAPES, LABEL_STRUCTURES, MATERIALS, iter_multitask_samples  # noqa: E402
from balcony_train.model import load_classifier  # noqa: E402

DEFAULT_DIR = ROOT / "runs" / "test100" / "crops"
TEST_FLOORS = ("rectangle", "circle")
HEADS = {"kind": CLASSES, "material": MATERIALS, "floor": TEST_FLOORS, "structure": LABEL_STRUCTURES}


TITLES = {"kind": "Kind", "material": "Material (open_work only)", "floor": "Floor", "structure": "Structure"}


def confusion(head: str, names: tuple[str, ...], c: torch.Tensor) -> list[str]:
    """Plain confusion block, e.g. ``floor n=384  floor_acc=0.938``."""
    n = int(c.sum())
    acc = int(c.diag().sum()) / max(1, n)
    if head == "kind":
        out = ["kind rows=true cols=pred  " + "  ".join(names)]
    else:
        label = "material(open_work)" if head == "material" else head
        out = [f"{label} n={n}  {head[:3] if head == 'material' else head}_acc={acc:.3f}"]
    out += [f"  {nm:14s} " + " ".join(f"{int(v):5d}" for v in c[i].tolist()) for i, nm in enumerate(names)]
    return out


def table(head: str, names: tuple[str, ...], c: torch.Tensor) -> list[str]:
    """Markdown precision / recall / F1 table with a bold macro row."""
    out = [f"**{TITLES[head]}**", "", "| class | precision | recall | F1 |", "|---|---|---|---|"]
    ps, rs, fs = [], [], []
    for i, nm in enumerate(names):
        tp, col, row = int(c[i, i]), int(c[:, i].sum()), int(c[i].sum())
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p); rs.append(r); fs.append(f)
        out.append(f"| {nm} | {tp}/{col} = {p:.3f} | {tp}/{row} = {r:.3f} | {f:.3f} |")
    k = len(names)
    out.append(f"| **macro** | **{sum(ps) / k:.3f}** | **{sum(rs) / k:.3f}** | **{sum(fs) / k:.3f}** |")
    return out + [""]


def score(ckpt: Path, samples, device, by_source: bool) -> None:
    model = load_classifier(ckpt, device)
    sources = ("all", "cmp_b", "lsaa") if by_source else ("all",)
    conf = {(s, h): torch.zeros((len(n), len(n)), dtype=torch.int64) for s in sources for h, n in HEADS.items()}
    tri, rect = FLOOR_SHAPES.index("triangle"), FLOOR_SHAPES.index("rectangle")
    loader = DataLoader(MultitaskCropDataset(samples), batch_size=32, shuffle=False)
    k = 0
    with torch.no_grad():
        for images, ky, my, fy, _ey, _sy, sty in loader:
            outs = model(images.to(device))
            names = [Path(samples[k + i][0]).name for i in range(len(ky))]
            k += len(ky)
            for head, y, logit in (("kind", ky, outs[0]), ("material", my, outs[1]),
                                   ("floor", fy, outs[2]), ("structure", sty, outs[5])):
                for name, t, p in zip(names, y.tolist(), logit.argmax(1).cpu().tolist()):
                    if t < 0:
                        continue
                    if head == "floor":
                        t = TEST_FLOORS.index(FLOOR_SHAPES[rect if t == tri else t])
                        p = TEST_FLOORS.index(FLOOR_SHAPES[rect if p == tri else p])
                    src = "cmp_b" if name.startswith("cmp_b") else "lsaa"
                    for s in ("all", src) if by_source else ("all",):
                        conf[s, head][t, p] += 1
    print(f"===== {ckpt}  sha256={hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]}")
    for s in sources:
        print(f"\n## {s}\n\n```")
        for head, names in HEADS.items():
            print("\n".join(confusion(head, names, conf[s, head])))
        print("```\n")
        for head, names in HEADS.items():
            print("\n".join(table(head, names, conf[s, head])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpts", type=Path, nargs="+")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--by-source", action="store_true", help="also cmp_b and LSAA tables (supplementary)")
    args = ap.parse_args()
    samples = iter_multitask_samples(args.dir, args.dir / "labels.jsonl")
    print(f"samples {len(samples)} from {args.dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ck in args.ckpts:
        score(ck, samples, device, args.by_source)


if __name__ == "__main__":
    main()
