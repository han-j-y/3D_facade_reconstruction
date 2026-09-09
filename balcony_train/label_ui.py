"""Tkinter Label UI for railing kind (+ material when open_work).

Usage::

    python balcony_train/label_ui.py
    python balcony_train/label_ui.py --crops-dir runs/balcony_clf/crops

Keys: 1/2/3 kind, M/N material (open_work), Enter save, S skip, U undo, Q quit.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import (  # noqa: E402
    find_image,
    merge_label_sources,
    queue_stems,
    save_annotation,
    write_labels_jsonl,
)
from balcony_train.labels import CLASSES, MATERIALS, ensure_crop_dirs  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL  # noqa: E402

KIND_KEYS = {"1": "open_work", "2": "surface_panel", "3": "solid"}
MATERIAL_KEYS = {"m": "metal", "n": "masonry"}
# Fallback when the image pane has not been laid out yet.
PREVIEW_FALLBACK = (900, 700)


def _fit_image(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale (up or down) to fit inside max_w×max_h, keeping aspect ratio."""
    max_w = max(64, int(max_w))
    max_h = max(64, int(max_h))
    w, h = im.size
    if w < 1 or h < 1:
        return im
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


class LabelApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        crops_dir: Path,
        labels_path: Path,
        review_all: bool = False,
    ) -> None:
        self.root = root
        self.crops_dir = Path(crops_dir)
        self.labels_path = Path(labels_path)
        ensure_crop_dirs(self.crops_dir)
        self.labels = merge_label_sources(self.crops_dir, self.labels_path)
        self.queue = queue_stems(
            self.crops_dir,
            self.labels,
            unlabeled_only=not review_all,
            need_material=True,
        )
        self.index = 0
        self.undo_stack: list[dict] = []
        self.kind_var = tk.StringVar(value="")
        self.material_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self._photo: ImageTk.PhotoImage | None = None
        self._pil: Image.Image | None = None
        self._current_path: Path | None = None
        self._resize_after: str | None = None
        self._build()
        self._bind_keys()
        self._show_current()

    def _build(self) -> None:
        self.root.title("Balcony railing label UI")
        self.root.geometry("1100x900")
        self.root.minsize(900, 700)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.path_var, font=("Segoe UI", 10)).pack(
            anchor=tk.W
        )
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W
        )

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill=tk.BOTH, expand=True)
        self.image_frame = ttk.Frame(mid)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.image_frame, anchor=tk.CENTER)
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.image_frame.bind("<Configure>", self._on_image_frame_configure)

        kind_fr = ttk.LabelFrame(self.root, text="Kind (1/2/3)", padding=8)
        kind_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(CLASSES, ("1", "2", "3")):
            ttk.Radiobutton(
                kind_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.kind_var,
                command=self._on_kind_change,
            ).pack(side=tk.LEFT, padx=8)

        self.mat_fr = ttk.LabelFrame(
            self.root,
            text="Material (open_work only) — M metal / N masonry",
            padding=8,
        )
        self.mat_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(MATERIALS, ("M", "N")):
            ttk.Radiobutton(
                self.mat_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.material_var,
            ).pack(side=tk.LEFT, padx=8)

        btns = ttk.Frame(self.root, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Save (Enter)", command=self.save).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btns, text="Skip (S)", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Quit (Q)", command=self.root.destroy).pack(
            side=tk.RIGHT, padx=4
        )

        help_txt = (
            "Material is required only for open_work. "
            "Labels → labels.jsonl; images move into crops/{kind}/."
        )
        ttk.Label(self.root, text=help_txt, padding=8).pack(anchor=tk.W)

    def _preview_size(self) -> tuple[int, int]:
        self.image_frame.update_idletasks()
        w = int(self.image_frame.winfo_width())
        h = int(self.image_frame.winfo_height())
        if w < 80 or h < 80:
            return PREVIEW_FALLBACK
        # Leave a little margin inside the pane.
        return (max(64, w - 16), max(64, h - 16))

    def _render_preview(self) -> None:
        if self._pil is None:
            return
        fitted = _fit_image(self._pil, *self._preview_size())
        self._photo = ImageTk.PhotoImage(fitted)
        self.image_label.configure(image=self._photo)

    def _on_image_frame_configure(self, _event: tk.Event | None = None) -> None:
        if self._resize_after is not None:
            self.root.after_cancel(self._resize_after)
        self._resize_after = self.root.after(50, self._render_preview)

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()

    def _on_key(self, event: tk.Event) -> None:
        ch = (event.char or "").lower()
        keysym = event.keysym
        if ch in KIND_KEYS:
            self.kind_var.set(KIND_KEYS[ch])
            self._on_kind_change()
            return
        if ch in MATERIAL_KEYS and self.kind_var.get() == "open_work":
            self.material_var.set(MATERIAL_KEYS[ch])
            return
        if keysym in ("Return", "KP_Enter"):
            self.save()
            return
        if ch == "s":
            self.skip()
            return
        if ch == "u":
            self.undo()
            return
        if ch == "q":
            self.root.destroy()

    def _on_kind_change(self) -> None:
        kind = self.kind_var.get()
        if kind == "open_work":
            self._set_material_enabled(True)
            if self.material_var.get() not in MATERIALS:
                self.material_var.set("metal")
        else:
            self.material_var.set("")
            self._set_material_enabled(False)

    def _set_material_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for child in self.mat_fr.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _counts(self) -> str:
        n_kind = {k: 0 for k in CLASSES}
        n_mat = {m: 0 for m in MATERIALS}
        n_open_no_mat = 0
        for rec in self.labels.values():
            kind = rec.get("kind")
            if kind in n_kind:
                n_kind[kind] += 1
            if kind == "open_work":
                mat = rec.get("material")
                if mat in n_mat:
                    n_mat[mat] += 1
                else:
                    n_open_no_mat += 1
        parts = [f"{k}={n_kind[k]}" for k in CLASSES]
        parts.append(
            f"open_work materials metal={n_mat['metal']} masonry={n_mat['masonry']}"
            + (f" missing={n_open_no_mat}" if n_open_no_mat else "")
        )
        return "  ".join(parts)

    def _show_current(self) -> None:
        if not self.queue:
            self.path_var.set("No images left to label.")
            self.status_var.set(self._counts())
            self.image_label.configure(image="")
            self._photo = None
            self._pil = None
            self._current_path = None
            self.kind_var.set("")
            self.material_var.set("")
            self._set_material_enabled(False)
            return

        self.index = max(0, min(self.index, len(self.queue) - 1))
        stem = self.queue[self.index]
        path = find_image(self.crops_dir, stem)
        if path is None:
            self.queue.pop(self.index)
            self._show_current()
            return

        rec = self.labels.get(stem) or {}
        kind = rec.get("kind") or ""
        material = rec.get("material") or ""
        self.kind_var.set(kind if kind in CLASSES else "")
        self._on_kind_change()
        if kind == "open_work" and material in MATERIALS:
            self.material_var.set(material)

        try:
            im = Image.open(path).convert("RGB")
        except OSError as exc:
            messagebox.showerror("Image error", f"{path}\n{exc}")
            self.skip()
            return
        self._pil = im
        self._current_path = path
        self._render_preview()

        left = len(self.queue) - self.index
        self.path_var.set(f"[{self.index + 1}/{len(self.queue)}]  {path.name}")
        self.status_var.set(f"remaining≈{left}   {self._counts()}")

    def save(self) -> None:
        if not self.queue:
            return
        stem = self.queue[self.index]
        path = find_image(self.crops_dir, stem)
        if path is None:
            messagebox.showwarning("Missing", f"Image not found for {stem}")
            return
        kind = self.kind_var.get()
        if kind not in CLASSES:
            messagebox.showwarning("Kind required", "Select kind (1/2/3).")
            return
        material = self.material_var.get() if kind == "open_work" else None
        if kind == "open_work" and material not in MATERIALS:
            messagebox.showwarning(
                "Material required",
                "open_work needs material: metal (M) or masonry (N).",
            )
            return

        prev = dict(self.labels.get(stem) or {})
        prev_path = str(path)
        try:
            self.labels = save_annotation(
                crops_dir=self.crops_dir,
                jsonl_path=self.labels_path,
                image_path=path,
                kind=kind,
                material=material,
                labels=self.labels,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.undo_stack.append(
            {
                "stem": stem,
                "prev_label": prev,
                "prev_path": prev_path,
                "new_kind": kind,
            }
        )
        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1
        self._show_current()

    def skip(self) -> None:
        if not self.queue:
            return
        self.index = (self.index + 1) % len(self.queue)
        self._show_current()

    def undo(self) -> None:
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        item = self.undo_stack.pop()
        stem = item["stem"]
        prev = item["prev_label"]
        # Move file back toward previous location when possible.
        cur = find_image(self.crops_dir, stem)
        prev_path = Path(item["prev_path"])
        if cur is not None and prev_path.parent != cur.parent:
            prev_path.parent.mkdir(parents=True, exist_ok=True)
            if prev_path.exists() and prev_path != cur:
                prev_path.unlink()
            cur.rename(prev_path)

        if prev.get("kind"):
            self.labels[stem] = {
                "path": prev.get("path") or f"{stem}.png",
                "kind": prev.get("kind"),
                "material": prev.get("material"),
            }
        else:
            self.labels.pop(stem, None)
        write_labels_jsonl(self.labels_path, self.labels)
        if stem not in self.queue:
            self.queue.insert(self.index, stem)
        self._show_current()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument(
        "--review-all",
        action="store_true",
        help="include already complete labels (for editing)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    LabelApp(
        root,
        crops_dir=args.crops_dir,
        labels_path=args.labels,
        review_all=args.review_all,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
