"""Label UI for pipeline crops of extended_selected (the attribute test set).

Same keys as ``label_ui.py`` (1/2/3 kind, M/N material, R/T/C floor, O/A/E
enclosure, P/G skin, P/J structure, Enter save, S skip, Left previous, U undo,
Q quit), plus:

* the photo with the crop's box drawn in red, beside the crop;
* starting values from earlier hand labels (``prefill.jsonl``) on crops not
  yet saved. They are never model predictions;
* ``X``: not a balcony (``not_balcony.jsonl``). These are left out of the
  attribute accuracy and counted separately.

A crop counts as reviewed once it is saved to ``labels.jsonl`` or marked X.

Labeling rules:

* label the real balcony seen in the photo, not only what the crop shows;
* a box over two balconies gets the larger one;
* a box that is mostly wall, window, cornice or a sill gets X;
* a crop of an enclosed oriel gets E (enclosed) and its skin.

``--train`` labels the cmp_b training crops instead: ``runs/balcony_clf/crops``
and ``labels.jsonl``, with context from ``crops_pipeline_train/context``
(``export_train_pipeline_crops.py``). There is no prefill. ``--structure-pass``
works as in ``label_ui.py``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_pipeline_crops_ui.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_pipeline_crops_ui.py --train
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import _place_in_unlabeled, find_image, write_labels_jsonl  # noqa: E402
from balcony_train.label_ui import LabelApp, _fit_image  # noqa: E402
from balcony_train.labels import CLASSES, ENCLOSURES, FLOOR_SHAPES, LABEL_STRUCTURES, MATERIALS  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL  # noqa: E402

DEFAULT_DIR = ROOT / "runs" / "balcony_clf" / "crops_pipeline_ext"
TRAIN_DIR = ROOT / "runs" / "balcony_clf" / "crops_pipeline_train"


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[Path(rec["path"]).stem] = rec
    return out


class PipelineCropApp(LabelApp):
    def __init__(
        self,
        root: tk.Tk,
        *,
        crops_dir: Path,
        labels_path: Path,
        context_dir: Path,
        prefill_path: Path,
        not_balcony_path: Path,
        review_all: bool = False,
        structure_pass: bool = False,
    ) -> None:
        self.context_dir = Path(context_dir)
        self.is_train = Path(prefill_path).parent.resolve() == TRAIN_DIR.resolve()
        self.prefill = load_jsonl(Path(prefill_path))
        self.not_balcony_path = Path(not_balcony_path)
        self.not_balcony = set(load_jsonl(self.not_balcony_path))
        self._ctx_pil: Image.Image | None = None
        self._ctx_photo: ImageTk.PhotoImage | None = None
        super().__init__(
            root,
            crops_dir=crops_dir,
            labels_path=labels_path,
            review_all=review_all,
            structure_pass=structure_pass,
        )
        if not review_all:
            self.queue = [s for s in self.queue if s not in self.not_balcony]
            self.index = 0
            self._show_current()

    def _build(self) -> None:
        super()._build()
        title = "training crops" if self.is_train else "attribute test set"
        self.root.title(f"Pipeline crop label UI ({title})")
        self.root.geometry("1500x950")
        mid = self.image_frame.master
        self.image_frame.pack_forget()
        self.ctx_frame = ttk.Frame(mid)
        self.ctx_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ctx_label = ttk.Label(self.ctx_frame, anchor=tk.CENTER)
        self.ctx_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.ctx_frame.bind("<Configure>", self._on_image_frame_configure)
        ttk.Label(
            self.root,
            text="X = not a balcony. Label the real balcony in the red box (left), not only the crop (right).",
            padding=8,
        ).pack(anchor=tk.W)

    def _render_preview(self) -> None:
        super()._render_preview()
        if self._ctx_pil is None:
            self.ctx_label.configure(image="")
            return
        self.ctx_frame.update_idletasks()
        w = max(64, self.ctx_frame.winfo_width() - 16)
        h = max(64, self.ctx_frame.winfo_height() - 16)
        self._ctx_photo = ImageTk.PhotoImage(_fit_image(self._ctx_pil, w, h))
        self.ctx_label.configure(image=self._ctx_photo)

    def _show_current(self) -> None:
        self._ctx_pil = None
        super()._show_current()
        if not self.queue:
            self._render_preview()
            return
        stem = self.queue[self.index]
        note = ""
        if stem in self.not_balcony:
            note = "   [marked NOT A BALCONY]"
        elif stem not in self.labels and stem in self.prefill:
            pre = self.prefill[stem]
            kind = pre.get("kind")
            self.kind_var.set(kind if kind in CLASSES else "")
            self._on_kind_change()
            if kind == "open_work" and pre.get("material") in MATERIALS:
                self.material_var.set(pre["material"])
            floor = pre.get("floor_shape")
            self.floor_var.set(floor if floor in FLOOR_SHAPES else "")
            enc = pre.get("enclosure")
            self.enclosure_var.set(enc if enc in ENCLOSURES else "")
            self._sync_skin()
            # cmp_x labels saved projecting as null.
            structure = pre.get("structure") or "projecting"
            if enc == "open" and structure in LABEL_STRUCTURES:
                self.structure_var.set(structure)
            note = f"   [starting values from {pre['prefill_from']}, IoU {pre['prefill_iou']}]"
        elif stem not in self.labels:
            note = "   [no starting values]"
        ctx = self.context_dir / f"{stem}.jpg"
        if ctx.is_file():
            self._ctx_pil = Image.open(ctx).convert("RGB")
        self._render_preview()
        self.path_var.set(self.path_var.get() + note)

    def _counts(self) -> str:
        total = len(self.labels) + len(self.not_balcony)
        return f"reviewed={total}  not_balcony={len(self.not_balcony)}   " + super()._counts()

    def _write_not_balcony(self) -> None:
        lines = [json.dumps({"path": f"{s}.png"}) + "\n" for s in sorted(self.not_balcony)]
        self.not_balcony_path.write_text("".join(lines), encoding="utf-8")

    def mark_not_balcony(self) -> None:
        if not self.queue:
            return
        stem = self.queue[self.index]
        if stem in self.labels and not messagebox.askyesno(
            "Not a balcony", f"{stem} already has labels. Remove them and mark it X?"
        ):
            return
        if stem in self.labels:
            self.labels.pop(stem)
            write_labels_jsonl(self.labels_path, self.labels)
            path = find_image(self.crops_dir, stem)
            if path is not None:
                _place_in_unlabeled(self.crops_dir, path)
        self.not_balcony.add(stem)
        self._write_not_balcony()
        self.back_stack.append(stem)
        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1
        self._show_current()

    def save(self) -> None:
        stem = self.queue[self.index] if self.queue else None
        super().save()
        if stem in self.not_balcony and stem in self.labels:
            self.not_balcony.discard(stem)
            self._write_not_balcony()

    def _on_key(self, event: tk.Event) -> None:
        if (event.char or "").lower() == "x":
            self.mark_not_balcony()
            return
        super()._on_key(event)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="crops, context, prefill, labels live here")
    ap.add_argument("--review-all", action="store_true", help="also show reviewed crops (for editing)")
    ap.add_argument("--train", action="store_true", help="label the cmp_b training crops (crops/, labels.jsonl)")
    ap.add_argument(
        "--structure-pass", action="store_true", help="open crops with no structure yet (cmp_b names)"
    )
    args = ap.parse_args()
    root = tk.Tk()
    if args.train:
        side = TRAIN_DIR if args.dir == DEFAULT_DIR else args.dir
        crops_dir, labels_path = DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL
    else:
        side = args.dir
        crops_dir, labels_path = args.dir, args.dir / "labels.jsonl"
    PipelineCropApp(
        root,
        crops_dir=crops_dir,
        labels_path=labels_path,
        context_dir=side / "context",
        prefill_path=side / "prefill.jsonl",
        not_balcony_path=side / "not_balcony.jsonl",
        review_all=args.review_all,
        structure_pass=args.structure_pass,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
