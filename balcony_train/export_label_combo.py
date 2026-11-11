r"""Copy the railing-classifier training crops of one kind (optionally one structure too).

Uses the same loader as ``train.py`` (kind from the crop folder, other heads from
``labels.jsonl``) and the fold from ``split.json``. Crops whose photo is in
test100 or whose name starts with ``flux2_`` are marked: ``railing_test100_v2.pt``
did not train on them.

Output: ``<out>/<fold>/<crop>.png``, ``<out>/index.csv`` and contact sheets per
fold, 64 crops a page (``<out>/sheet_<fold>_NN.jpg``, crop file name under each tile).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/export_label_combo.py --kind solid
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import CLASSES, LABEL_STRUCTURES, iter_multitask_samples, structure_target  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL, DEFAULT_SPLIT_JSON  # noqa: E402
from balcony_train.splits import indices_from_record, load_split  # noqa: E402

T100_PHOTOS = ROOT / "data" / "test100" / "photos"


def sheet(paths: list[Path], out: Path, tile: int = 220, cols: int = 8) -> None:
    font = ImageFont.truetype("arial.ttf", 11)
    rows = (len(paths) + cols - 1) // cols
    img = Image.new("RGB", (cols * tile, rows * (tile + 30)), "white")
    d = ImageDraw.Draw(img)
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        f = min((tile - 6) / im.width, (tile - 6) / im.height)
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        x, y = (k % cols) * tile, (k // cols) * (tile + 30)
        img.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        name = p.stem
        d.text((x + 3, y + tile + 1), name[:34], fill="black", font=font)
        if len(name) > 34:
            d.text((x + 3, y + tile + 14), name[34:68], fill="black", font=font)
    img.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="solid", choices=CLASSES)
    ap.add_argument("--structure", default="any", choices=(*LABEL_STRUCTURES, "any"),
                    help="any = every crop of --kind, whatever its structure label")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_JSON)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tag = args.kind if args.structure == "any" else f"{args.kind}_{args.structure}"
    out = args.out or ROOT / "runs" / "balcony_clf" / f"review_{tag}"

    samples = iter_multitask_samples(args.data_dir, args.labels)
    fold_of: dict[int, str] = {}
    if args.split_file.is_file():
        for name, idx in indices_from_record(load_split(args.split_file), samples).items():
            for i in idx:
                fold_of[i] = name
    test100 = {p.stem for p in T100_PHOTOS.glob("*.jpg")}
    k_idx = CLASSES.index(args.kind)
    s_idx = None if args.structure == "any" else LABEL_STRUCTURES.index(args.structure)

    if out.exists():
        shutil.rmtree(out)
    hits: dict[str, list[Path]] = {}
    rows = []
    for i, row in enumerate(samples):
        if int(row[1]) != k_idx or (s_idx is not None and structure_target(row) != s_idx):
            continue
        src = Path(row[0])
        held = "flux2_" if src.name.startswith("flux2_") else (
            "test100" if any(t in src.stem for t in test100) else "")
        fold = fold_of.get(i, "unsplit")
        folder = fold if not held else f"not_trained_{held.strip('_')}"
        dst = out / folder / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hits.setdefault(folder, []).append(dst)
        st = structure_target(row)
        rows.append({"crop": src.name, "folder": folder, "fold": fold,
                     "structure": LABEL_STRUCTURES[st] if 0 <= st < len(LABEL_STRUCTURES) else "",
                     "source": str(src.relative_to(ROOT))})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "folder", "fold", "structure", "source"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["folder"], r["crop"])))
    for folder, paths in sorted(hits.items()):
        paths = sorted(paths)
        for k in range(0, len(paths), 64):
            sheet(paths[k:k + 64], out / f"sheet_{folder}_{k // 64 + 1:02d}.jpg")
        print(f"{folder:22s} {len(paths)}")
    print(f"{len(rows)} crops -> {out}")


if __name__ == "__main__":
    main()
