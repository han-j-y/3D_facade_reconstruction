"""Score external (Astra) crop labels against the test100 crop labels, same tables as evaluate_test100.

Reads every ``<out_root>/partNN/answers.json``, maps ids back through
``crop_set_mapping.json`` and scores each head on its own:

- kind, structure: every crop with that ground-truth label. A missing / null answer
  is reported (``no answer``) and left out of the matrix.
- material: crops whose truth is open_work and where Astra gave a material
  (a crop Astra did not call open_work already counts as a kind error).
- floor: crops with a floor label and where Astra gave a floor (a projecting crop
  called juliet already counts as a structure error).

With ``--ckpt`` the classifier is also scored on exactly the same crops per head
(``classifier, same crops``) besides its full table.

Usage:
  python balcony_train/score_astra_crops.py <out_root> <crop_set_mapping.json>
      [--ckpt checkpoints/railing_test100_v4.pt] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import CLASSES, LABEL_STRUCTURES, MATERIALS  # noqa: E402

TEST_FLOORS = ("rectangle", "circle")
HEADS = {"kind": CLASSES, "material": MATERIALS, "floor": TEST_FLOORS, "structure": LABEL_STRUCTURES}
FIELD = {"kind": "kind", "material": "material", "floor": "floor_shape", "structure": "structure"}


def truth_of(head: str, lab: dict):
    v = lab.get(FIELD[head])
    if head == "material" and lab.get("kind") != "open_work":
        return None
    if head == "floor" and v == "triangle":
        v = "rectangle"
    return v if v in HEADS[head] else None


def table(title: str, names, pairs) -> list[str]:
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    c = [[0] * k for _ in range(k)]
    for t, p in pairs:
        c[idx[t]][idx[p]] += 1
    n = len(pairs)
    acc = sum(c[i][i] for i in range(k)) / n if n else 0.0
    out = [f"**{title}** n={n} acc={acc:.3f}", "", "rows=true cols=pred: " + " / ".join(names)]
    for i, nm in enumerate(names):
        out.append(f"- {nm}: " + " ".join(str(v) for v in c[i]))
    out += ["", "| class | precision | recall | F1 |", "|---|---|---|---|"]
    ps, rs, fs = [], [], []
    for i, nm in enumerate(names):
        tp = c[i][i]
        col = sum(c[r][i] for r in range(k))
        row = sum(c[i])
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        ps.append(p); rs.append(r); fs.append(f)
        out.append(f"| {nm} | {tp}/{col} = {p:.3f} | {tp}/{row} = {r:.3f} | {f:.3f} |")
    out.append(f"| **macro** | **{sum(ps) / k:.3f}** | **{sum(rs) / k:.3f}** | **{sum(fs) / k:.3f}** |")
    return out + [""]


def summary(who: str, pairs_by_head: dict) -> list[str]:
    """Confusion block + P/R/F1 table + accuracy lines, laid out like the v4 classifier table."""
    label = {"kind": "Kind", "material": "Material", "floor": "Floor", "structure": "Structure"}
    short = {"kind": "kind", "material": "mat", "floor": "floor", "structure": "struct"}
    block, rows, accs = [], [], []
    for head, (pairs, no_answer) in pairs_by_head.items():
        names = HEADS[head]
        k = len(names)
        idx = {n: i for i, n in enumerate(names)}
        c = [[0] * k for _ in range(k)]
        for t, p in pairs:
            c[idx[t]][idx[p]] += 1
        n = len(pairs)
        diag = [c[i][i] for i in range(k)]
        acc = sum(diag) / n if n else 0.0
        extra = f"  (no answer {no_answer})" if no_answer else ""
        if head == "kind":
            block.append("kind rows=true cols=pred  " + "  ".join(names) + extra)
        else:
            sub = "material(open_work)" if head == "material" else head
            tag = {"material": "mat_acc", "floor": "floor_acc", "structure": "structure_acc"}[head]
            block.append(f"{sub} n={n}  {tag}={acc:.3f}{extra}")
        for i, nm in enumerate(names):
            block.append(f"  {nm:<16}" + "".join(f"{v:>6}" for v in c[i]))
        ps, rs, fs = [], [], []
        for i, nm in enumerate(names):
            col = sum(c[r][i] for r in range(k))
            row = sum(c[i])
            p = c[i][i] / col if col else 0.0
            r = c[i][i] / row if row else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            ps.append(p); rs.append(r); fs.append(f)
            rows.append(f"| {label[head] if i == 0 else ''} | {nm} | {c[i][i]}/{col} = {p:.3f} | "
                        f"{c[i][i]}/{row} = {r:.3f} | {f:.3f} |")
        rows.append(f"| | **macro** | **{sum(ps) / k:.3f}** | **{sum(rs) / k:.3f}** | **{sum(fs) / k:.3f}** |")
        rowsums = [sum(c[i]) for i in range(k)]
        accs.append(f"- Acc_{short[head]} = ({' + '.join(map(str, diag))}) / ({' + '.join(map(str, rowsums))})"
                    f" = {sum(diag)} / {n} = **{acc:.3f}**")
    return ([f"# {who}: Balcony Type Classification", "", "```"] + block + ["```", "",
            "| Head | class | precision | recall | F1 |", "|---|---|---|---|---|"] + rows
            + ["", "Accuracy", ""] + accs + ["", "---", ""])


def classifier_preds(ckpt: Path, files: dict[str, Path]) -> dict[str, dict]:
    import torch
    from PIL import Image

    from balcony_train.labels import FLOOR_SHAPES
    from balcony_train.model import load_classifier, make_transform

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(ckpt, dev)
    tfm = make_transform()
    out = {}
    with torch.no_grad():
        for cid, f in files.items():
            x = tfm(Image.open(f).convert("RGB")).unsqueeze(0).to(dev)
            o = model(x)
            fl = FLOOR_SHAPES[int(o[2].argmax(1))]
            out[cid] = {
                "kind": CLASSES[int(o[0].argmax(1))],
                "material": MATERIALS[int(o[1].argmax(1))],
                "floor": "rectangle" if fl == "triangle" else fl,
                "structure": LABEL_STRUCTURES[int(o[5].argmax(1))],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root", type=Path)
    ap.add_argument("mapping", type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--answers", type=Path, default=None,
                    help="one answers.json for all crops (instead of <out_root>/partNN/answers.json)")
    args = ap.parse_args()

    mp = json.loads(args.mapping.read_text(encoding="utf-8"))
    base = Path(mp["dir"])
    labels = {}
    for l in (base / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            labels[r["path"]] = r
    answers: dict[str, dict] = {}
    parts = sorted({m["part"] for m in mp["crops"]})
    missing_parts = []
    files = [args.answers] if args.answers else []
    if not args.answers:
        for part in parts:
            f = args.out_root / f"part{part:02d}" / "answers.json"
            if f.is_file():
                files.append(f)
            else:
                missing_parts.append(part)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for r in data["answers"] if isinstance(data, dict) else data:
            answers[str(r["id"])] = r
    crops = [m for m in mp["crops"] if m["part"] not in missing_parts]
    lines = [f"# Astra on test100 crops ({len(crops)} of {len(mp['crops'])} crops)", ""]
    if missing_parts:
        lines += [f"Parts without answers.json (left out): {missing_parts}", ""]

    cls = None
    if args.ckpt is not None:
        cls = classifier_preds(args.ckpt, {m["id"]: ROOT / m["src"] for m in crops})

    pairs_by_head = {}
    for head, names in HEADS.items():
        astra, clf_same, clf_all, no_answer = [], [], [], 0
        for m in crops:
            t = truth_of(head, labels[m["path"]])
            if t is None:
                continue
            a = answers.get(m["id"], {})
            p = a.get(FIELD[head])
            if head == "floor" and p == "triangle":
                p = "rectangle"
            if cls is not None:
                clf_all.append((t, cls[m["id"]][head]))
            if p not in names:
                no_answer += 1
                continue
            astra.append((t, p))
            if cls is not None:
                clf_same.append((t, cls[m["id"]][head]))
        pairs_by_head[head] = (astra, no_answer)
        lines += [f"## {head}", "", f"no answer (null / missing / invalid): {no_answer}", ""]
        lines += table("Astra", names, astra)
        if cls is not None:
            lines += table(f"classifier {args.ckpt.name}, same crops", names, clf_same)
            lines += table(f"classifier {args.ckpt.name}, all crops", names, clf_all)
    lines = summary("Astra", pairs_by_head) + lines
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")
    print(text)


if __name__ == "__main__":
    main()
