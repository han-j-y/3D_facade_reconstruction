"""Cut cmp_b open-balcony crops as the pipeline does and keep only ones the training set lacks.

Boxes are the pipeline's, same as ``export_pipeline_crops.py``: SAM3 ``balcony``
at score >= 0.45, clamped to the mask, sides under 16 px dropped, then
``filter_balcony_boxes``. The crop is ``photo.crop(box)`` with no margin.
Photos are ``data/base_jpg`` (cmp_b only; never cmp_x).

Old crops have no saved boxes. Every ``cmp_b*`` image under
``runs/balcony_clf/crops`` is found again in its photo by template match
(old crops are exact sub-images, sometimes with a 12% pad). A new box is then:

* ``duplicate``  IoU >= 0.5 with a found old box, or the image matches an old
                 crop that was not found (dHash distance <= 6 or DINOv2 cosine >= 0.95);
* ``check``      IoU 0.3-0.5, or cosine 0.85-0.95;
* ``new``        anything else.

Default is a dry run. It writes only ``--out-dir``:

* ``manifest.jsonl``           crop, photo, box, result, best old crop, IoU, cosine
* ``old_boxes.jsonl``          each old crop, its found box and match score
* ``context/<crop stem>.jpg``  photo, new box red, old boxes green; also one per
                               old crop found in its photo, for the structure pass
* ``crops/<crop stem>.png``    the crops
* ``dup_board.jpg``            new | old for every ``check`` and 30 random ``duplicate``

``--commit`` copies the ``new`` crops (and ``--accept`` names) to
``runs/balcony_clf/crops/unlabeled/<stem>_pbal_<NNN>.png``. An existing file stops the run.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/export_train_pipeline_crops.py --device cuda
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/export_train_pipeline_crops.py --commit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filter import filter_balcony_boxes  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_sam3_pipeline_stages import detect_stages  # noqa: E402

ROWS = ROOT / "runs" / "batch" / "_rows"
DEFAULT_IMAGE_DIR = ROOT / "data" / "base_jpg"
DEFAULT_CACHE = ROWS / "base_jpg_sam3_stage_boxes.json"
DEFAULT_OUT = ROOT / "runs" / "balcony_clf" / "crops_pipeline_train"
CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops"
LABELS = ROOT / "runs" / "balcony_clf" / "labels.jsonl"
OLD_STEM = re.compile(r"^(cmp_b\d{4})")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
MATCH_MIN = 0.98
DUP_IOU, CHECK_IOU = 0.5, 0.3
DUP_COS, CHECK_COS = 0.95, 0.85
DUP_DHASH = 6
CONTEXT_SIDE = 900
BOARD_TILE = 160
BOARD_RANDOM_DUP = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iter_old_crops() -> dict[str, list[Path]]:
    """Photo stem -> every cmp_b crop image under crops/ (any subfolder)."""
    out: dict[str, list[Path]] = {}
    for path in sorted(CROPS_DIR.rglob("cmp_b*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        m = OLD_STEM.match(path.name)
        if m and "_" in path.stem:
            out.setdefault(m.group(1), []).append(path)
    return out


def find_in_photo(gray_photo: np.ndarray, crop: Image.Image) -> tuple[list[int] | None, float]:
    """Best template-match box of ``crop`` in the photo, and its score."""
    tpl = np.asarray(crop.convert("L"))
    th, tw = tpl.shape
    ph, pw = gray_photo.shape
    if th > ph or tw > pw or th < 2 or tw < 2:
        return None, 0.0
    if float(tpl.std()) < 1e-3:
        return None, 0.0
    res = cv2.matchTemplate(gray_photo, tpl, cv2.TM_CCOEFF_NORMED)
    _lo, hi, _lo_at, at = cv2.minMaxLoc(res)
    x, y = at
    return [int(x), int(y), int(x + tw), int(y + th)], float(hi)


def dhash(im: Image.Image) -> int:
    g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Dino:
    """DINOv2 ``dinov2_vits14`` CLS embedding, loaded on first use."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None

    def embed(self, im: Image.Image) -> np.ndarray:
        import torch

        from balcony_train.model import BACKBONE, make_transform

        if self.model is None:
            self.model = torch.hub.load("facebookresearch/dinov2", BACKBONE).to(self.device).eval()
            self.tf = make_transform()
        with torch.no_grad():
            x = self.tf(im.convert("RGB")).unsqueeze(0).to(self.device)
            v = self.model(x)[0].float().cpu().numpy()
        return v / max(float(np.linalg.norm(v)), 1e-8)


def context_image(photo: Image.Image, box: list[int], old_boxes: list[list[int]]) -> Image.Image:
    scale = CONTEXT_SIDE / max(photo.size)
    im = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    for ob in old_boxes:
        d.rectangle([round(v * scale) for v in ob], outline=(40, 200, 60), width=2)
    b = [round(v * scale) for v in box]
    d.rectangle(b, outline=(255, 255, 255), width=6)
    d.rectangle(b, outline=(230, 30, 30), width=3)
    return im


def _tile(im: Image.Image | None) -> Image.Image:
    tile = Image.new("RGB", (BOARD_TILE, BOARD_TILE), (40, 40, 40))
    if im is not None:
        im = im.convert("RGB").copy()
        im.thumbnail((BOARD_TILE, BOARD_TILE))
        tile.paste(im, ((BOARD_TILE - im.width) // 2, (BOARD_TILE - im.height) // 2))
    return tile


def dup_board(rows: list[dict], out_dir: Path, path: Path) -> int:
    """Rows of new | old | caption for the given manifest rows."""
    if not rows:
        return 0
    cap_w = 360
    board = Image.new("RGB", (2 * BOARD_TILE + cap_w, BOARD_TILE * len(rows)), (20, 20, 20))
    d = ImageDraw.Draw(board)
    for i, r in enumerate(rows):
        y = i * BOARD_TILE
        board.paste(_tile(Image.open(out_dir / "crops" / r["crop"])), (0, y))
        old = Image.open(ROOT / r["old_crop"]) if r.get("old_crop") else None
        board.paste(_tile(old), (BOARD_TILE, y))
        lines = [
            f"{r['result']}  {r['crop']}",
            f"old: {Path(r['old_crop']).name if r.get('old_crop') else '-'}",
            f"iou={r['iou']}  cos={r['cos']}  dhash={r['dhash']}",
        ]
        for k, t in enumerate(lines):
            d.text((2 * BOARD_TILE + 8, y + 10 + 18 * k), t, fill=(230, 230, 230))
    board.save(path, quality=88)
    return len(rows)


def classify(iou: float, cos: float | None, dh: int | None) -> str:
    if iou >= DUP_IOU or (cos is not None and cos >= DUP_COS) or (dh is not None and dh <= DUP_DHASH):
        return "duplicate"
    if iou >= CHECK_IOU or (cos is not None and cos >= CHECK_COS):
        return "check"
    return "new"


def dry_run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} is not empty. Use --force to rebuild, or --commit to copy.")
    images = sorted(p for p in args.image_dir.glob("*.jpg") if p.stem.startswith("cmp_b"))
    stages = detect_stages(images, args.cache, args)
    old_by_stem = iter_old_crops()
    dino = Dino(args.device)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for sub in ("crops", "context"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    filter_sha = sha256(ROOT / "balcony_pipeline" / "filter.py")
    manifest, old_rows = [], []
    for k, path in enumerate(images, 1):
        photo = Image.open(path).convert("RGB")
        kept, _log = filter_balcony_boxes(stages[path.stem]["sized"], photo, [])
        boxes = [[int(v) for v in b] for b in kept]
        gray = np.asarray(photo.convert("L"))

        found, unfound = [], []
        for op in old_by_stem.get(path.stem, []):
            crop = Image.open(op)
            box, score = find_in_photo(gray, crop)
            ok = box is not None and score >= MATCH_MIN
            old_rows.append({"old_crop": str(op.relative_to(ROOT)), "photo": path.name,
                             "box_xyxy": box if ok else None, "match": round(score, 4)})
            (found if ok else unfound).append((op, box, crop))

        unfound_feats = [(op, dhash(c), dino.embed(c)) for op, _b, c in unfound] if boxes else []
        for j, box in enumerate(boxes):
            name = f"{path.stem}_pbal_{j:03d}"
            crop = photo.crop(tuple(box))
            crop.save(args.out_dir / "crops" / f"{name}.png")
            best_iou, best_old = 0.0, None
            for op, ob, _c in found:
                v = box_iou(box, ob)
                if v > best_iou:
                    best_iou, best_old = v, op
            best_cos, best_dh, cos_old = None, None, None
            if unfound_feats:
                h, e = dhash(crop), dino.embed(crop)
                for op, oh, oe in unfound_feats:
                    c = float(np.dot(e, oe))
                    if best_cos is None or c > best_cos:
                        best_cos, cos_old = c, op
                    dh = hamming(h, oh)
                    best_dh = dh if best_dh is None else min(best_dh, dh)
            result = classify(best_iou, best_cos, best_dh)
            old_ref = best_old if best_iou >= CHECK_IOU or cos_old is None else cos_old
            context_image(photo, box, [ob for _op, ob, _c in found]).save(
                args.out_dir / "context" / f"{name}.jpg", quality=88
            )
            manifest.append({
                "crop": f"{name}.png", "photo": path.name, "box_xyxy": box, "photo_size": list(photo.size),
                "result": result, "old_crop": str(old_ref.relative_to(ROOT)) if old_ref else None,
                "iou": round(best_iou, 3), "cos": None if best_cos is None else round(best_cos, 3),
                "dhash": best_dh, "cache": str(args.cache.relative_to(ROOT)), "filter_sha256": filter_sha,
            })
        if k % 25 == 0:
            print(f"[{k}/{len(images)}] crops so far={len(manifest)}", flush=True)

    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    (args.out_dir / "old_boxes.jsonl").write_text("".join(json.dumps(m) + "\n" for m in old_rows), encoding="utf-8")

    checks = [m for m in manifest if m["result"] == "check"]
    dups = [m for m in manifest if m["result"] == "duplicate"]
    rng = random.Random(0)
    sample = rng.sample(dups, min(BOARD_RANDOM_DUP, len(dups)))
    n_board = dup_board(checks + sample, args.out_dir, args.out_dir / "dup_board.jpg")

    counts = {r: sum(m["result"] == r for m in manifest) for r in ("new", "check", "duplicate")}
    n_found = sum(r["box_xyxy"] is not None for r in old_rows)
    print(f"photos={len(images)}  boxes={len(manifest)}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"old crops={len(old_rows)}  found in photo={n_found}  not found={len(old_rows) - n_found}")
    print(f"old crop context images={write_old_context(args.out_dir, args.image_dir)}")
    print(f"dup_board rows={n_board}  filter.py sha256={filter_sha}")
    print(f"-> {args.out_dir}")


def write_old_context(out_dir: Path, image_dir: Path) -> int:
    """context/<old crop stem>.jpg for every old crop found in its photo (structure pass)."""
    rows = [json.loads(l) for l in (out_dir / "old_boxes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    photos: dict[str, Image.Image] = {}
    n = 0
    for r in rows:
        if r["box_xyxy"] is None:
            continue
        if r["photo"] not in photos:
            photos = {r["photo"]: Image.open(image_dir / r["photo"]).convert("RGB")}
        context_image(photos[r["photo"]], r["box_xyxy"], []).save(
            out_dir / "context" / f"{Path(r['old_crop']).stem}.jpg", quality=88
        )
        n += 1
    return n


def commit(args: argparse.Namespace) -> None:
    manifest_path = args.out_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"no dry run at {args.out_dir}; run without --commit first")
    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    accept = {a.strip() for a in (args.accept or "").split(",") if a.strip()}
    picked = [r for r in rows if r["result"] == "new" or Path(r["crop"]).stem in accept or r["crop"] in accept]
    dest_dir = CROPS_DIR / "unlabeled"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # A crop committed earlier may since have moved to a kind folder; skip it anywhere under crops/.
    present = {p.name for p in CROPS_DIR.rglob("cmp_b*_pbal_*")}
    skipped = [r for r in picked if r["crop"] in present]
    picked = [r for r in picked if r["crop"] not in present]
    for r in picked:
        shutil.copy2(args.out_dir / "crops" / r["crop"], dest_dir / r["crop"])
    print(f"copied {len(picked)} crops -> {dest_dir}  (accepted: {len(accept)}, already present: {len(skipped)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="rebuild an existing dry run")
    ap.add_argument("--commit", action="store_true", help="copy new crops into crops/unlabeled")
    ap.add_argument("--accept", default="", help="comma-separated check crops to copy with --commit")
    ap.add_argument("--old-context", action="store_true", help="only redraw context for old crops")
    args = ap.parse_args()
    if args.old_context:
        print(f"old crop context images={write_old_context(args.out_dir, args.image_dir)}")
    elif args.commit:
        commit(args)
    else:
        dry_run(args)


if __name__ == "__main__":
    main()
