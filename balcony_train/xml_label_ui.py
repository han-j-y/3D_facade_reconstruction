"""Shared Tk UI for CMP-XML balcony crops.

Two entry points:

- ``label_open_ui.py`` — open balconies (kind, material, floor, structure).
  Structure is projecting or juliet. Key E sends the crop to the enclosed UI.
- ``label_enclosed_ui.py`` — enclosed balconies (floor, skin). Key A is
  half_enclosed (railing kind still required). Key O sends it back.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import (  # noqa: E402
    merge_label_sources,
    save_annotation,
    write_labels_jsonl,
)
from balcony_train.labels import (  # noqa: E402
    CLASSES,
    ENCLOSURE_SKINS,
    FLOOR_SHAPES,
    LABEL_STRUCTURES,
    MATERIALS,
)
from balcony_train.paths import (  # noqa: E402
    EXT_PHOTO_DIR,
    EXT_XML_CROPS_DIR,
    EXT_XML_LABELS,
    EXT_XML_MANIFEST,
)
from balcony_train.xml_balcony_labels import (  # noqa: E402
    enclosed_queue,
    facade_stem_from_crop,
    find_crop_image,
    load_manifest,
    open_queue,
    send_back_to_open,
    send_to_enclosed,
)

KIND_KEYS = {"1": "open_work", "2": "surface_panel", "3": "solid"}
MATERIAL_KEYS = {"m": "metal", "n": "masonry"}
FLOOR_KEYS = {"r": "rectangle", "t": "triangle", "c": "circle"}
SKIN_KEYS = {"p": "punched_window", "g": "curtain_wall"}
STRUCTURE_KEYS = {"p": "projecting", "j": "juliet"}


def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
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


def facade_with_box(
    photo_path: Path,
    box_xyxy: list[int] | None,
) -> Image.Image | None:
    if not photo_path.is_file():
        return None
    im = Image.open(photo_path).convert("RGB")
    if box_xyxy and len(box_xyxy) == 4:
        draw = ImageDraw.Draw(im)
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        for delta in range(4):
            draw.rectangle((x0 - delta, y0 - delta, x1 + delta, y1 + delta), outline=(220, 30, 30))
    return im


class XmlLabelApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        mode: str,
        crops_dir: Path,
        labels_path: Path,
        manifest_path: Path,
        photo_dir: Path,
        review_all: bool = False,
    ) -> None:
        if mode not in ("open", "enclosed"):
            raise ValueError(mode)
        self.root = root
        self.mode = mode
        self.crops_dir = Path(crops_dir)
        self.labels_path = Path(labels_path)
        self.photo_dir = Path(photo_dir)
        self.manifest = load_manifest(manifest_path)
        self.labels = merge_label_sources(self.crops_dir, self.labels_path)
        self.review_all = review_all
        self.queue = self._rebuild_queue()
        self.index = 0
        self.undo_stack: list[dict] = []
        self.back_stack: list[str] = []
        self.kind_var = tk.StringVar(value="")
        self.material_var = tk.StringVar(value="")
        self.floor_var = tk.StringVar(value="")
        self.structure_var = tk.StringVar(value="projecting")
        self.skin_var = tk.StringVar(value="")
        self.half_var = tk.BooleanVar(value=False)
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._crop_photo: ImageTk.PhotoImage | None = None
        self._facade_photo: ImageTk.PhotoImage | None = None
        self._pil: Image.Image | None = None
        self._facade_pil: Image.Image | None = None
        self._resize_after: str | None = None
        self._build()
        self._bind_keys()
        self._show_current()

    def _rebuild_queue(self) -> list[str]:
        if self.mode == "open":
            return open_queue(self.crops_dir, self.labels, unlabeled_only=not self.review_all)
        return enclosed_queue(self.crops_dir, self.labels, unlabeled_only=not self.review_all)

    def _build(self) -> None:
        title = "Open balcony" if self.mode == "open" else "Enclosed balcony"
        self.root.title(title)
        self.root.geometry("1200x820")
        self.root.minsize(960, 640)

        top = ttk.Frame(self.root, padding=(8, 4))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, textvariable=self.path_var, font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        # Pack the key rows first so a large photo cannot push them off screen.
        self.controls = ttk.Frame(self.root)
        self.controls.pack(side=tk.BOTTOM, fill=tk.X)

        mid = ttk.Frame(self.root, padding=(8, 4))
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.facade_frame = ttk.LabelFrame(mid, text="Facade (XML box)", padding=4, width=340)
        self.facade_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.facade_frame.pack_propagate(False)
        self.crop_frame = ttk.LabelFrame(mid, text="Crop", padding=4)
        self.crop_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.crop_frame.pack_propagate(False)
        self.crop_label = ttk.Label(self.crop_frame, anchor=tk.CENTER)
        self.crop_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.facade_label = ttk.Label(self.facade_frame, anchor=tk.CENTER)
        self.facade_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.crop_frame.bind("<Configure>", self._on_resize)
        self.facade_frame.bind("<Configure>", self._on_resize)

        if self.mode == "open":
            self._build_open_controls()
            help_txt = (
                "Open balcony.  1/2/3 kind, M/N material (open_work only), "
                "R/T/C floor, P projecting / J juliet, Enter save.  "
                "E = this is enclosed (switch to the other UI).  "
                "S skip, Left previous, U undo, Q quit."
            )
        else:
            self._build_enclosed_controls()
            help_txt = (
                "Enclosed balcony.  R/T/C floor, P punched_window, G curtain_wall, Enter save.  "
                "A = half_enclosed (then 1/2/3 kind, M/N if open_work).  "
                "O = send back to the open UI.  S skip, Left previous, U undo, Q quit."
            )
        ttk.Label(self.controls, text=help_txt, padding=(8, 2), wraplength=1100).pack(anchor=tk.W)

    def _build_open_controls(self) -> None:
        kind_fr = ttk.LabelFrame(self.controls, text="Kind — 1 open_work / 2 surface_panel / 3 solid", padding=4)
        kind_fr.pack(fill=tk.X, padx=8, pady=1)
        for name, key in zip(CLASSES, ("1", "2", "3")):
            ttk.Radiobutton(
                kind_fr, text=f"{key}: {name}", value=name, variable=self.kind_var,
                command=self._on_kind_change,
            ).pack(side=tk.LEFT, padx=8)
        self.mat_fr = ttk.LabelFrame(self.controls, text="Material — M metal / N masonry", padding=4)
        self.mat_fr.pack(fill=tk.X, padx=8, pady=1)
        for name, key in zip(MATERIALS, ("M", "N")):
            ttk.Radiobutton(
                self.mat_fr, text=f"{key}: {name}", value=name, variable=self.material_var,
            ).pack(side=tk.LEFT, padx=8)
        self._floor_row()
        self._structure_row()
        self._button_row("Save open (Enter)", "Enclosed (E)")

    def _build_enclosed_controls(self) -> None:
        self._floor_row()
        self.skin_fr = ttk.LabelFrame(
            self.controls, text="Skin — P punched_window / G curtain_wall", padding=4
        )
        self.skin_fr.pack(fill=tk.X, padx=8, pady=1)
        for name, key in zip(ENCLOSURE_SKINS, ("P", "G")):
            ttk.Radiobutton(
                self.skin_fr, text=f"{key}: {name}", value=name, variable=self.skin_var,
            ).pack(side=tk.LEFT, padx=8)
        self.kind_fr = ttk.LabelFrame(
            self.controls, text="Half-enclosed railing — 1/2/3 (only when A is on)", padding=4
        )
        for name, key in zip(CLASSES, ("1", "2", "3")):
            ttk.Radiobutton(
                self.kind_fr, text=f"{key}: {name}", value=name, variable=self.kind_var,
                command=self._on_kind_change,
            ).pack(side=tk.LEFT, padx=8)
        self.mat_fr = ttk.LabelFrame(self.controls, text="Material — M metal / N masonry", padding=4)
        for name, key in zip(MATERIALS, ("M", "N")):
            ttk.Radiobutton(
                self.mat_fr, text=f"{key}: {name}", value=name, variable=self.material_var,
            ).pack(side=tk.LEFT, padx=8)
        self._button_row("Save (Enter)", "Half (A)")

    def _floor_row(self) -> None:
        floor_fr = ttk.LabelFrame(
            self.controls, text="Floor — R rectangle / T triangle / C circle", padding=4
        )
        floor_fr.pack(fill=tk.X, padx=8, pady=1)
        for name, key in zip(FLOOR_SHAPES, ("R", "T", "C")):
            ttk.Radiobutton(
                floor_fr, text=f"{key}: {name}", value=name, variable=self.floor_var,
            ).pack(side=tk.LEFT, padx=8)

    def _structure_row(self) -> None:
        structure_fr = ttk.LabelFrame(
            self.controls, text="Structure — P projecting / J juliet", padding=4
        )
        structure_fr.pack(fill=tk.X, padx=8, pady=1)
        for name, key in zip(LABEL_STRUCTURES, ("P", "J")):
            ttk.Radiobutton(
                structure_fr, text=f"{key}: {name}", value=name, variable=self.structure_var,
            ).pack(side=tk.LEFT, padx=8)

    def _button_row(self, save_label: str, extra_label: str) -> None:
        btns = ttk.Frame(self.controls, padding=(8, 4))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text=save_label, command=self.save).pack(side=tk.LEFT, padx=4)
        if self.mode == "open":
            ttk.Button(btns, text=extra_label, command=self.mark_enclosed).pack(side=tk.LEFT, padx=4)
        else:
            ttk.Button(btns, text=extra_label, command=self.toggle_half).pack(side=tk.LEFT, padx=4)
            ttk.Button(btns, text="Back to open (O)", command=self.mark_open).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Skip (S)", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Previous", command=self.previous).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Quit (Q)", command=self.root.destroy).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()

    def _on_key(self, event: tk.Event) -> None:
        ch = (event.char or "").lower()
        keysym = event.keysym
        if keysym in ("Return", "KP_Enter"):
            self.save()
            return
        if keysym == "Left":
            self.previous()
            return
        if ch == "s" or keysym == "Right":
            self.skip()
            return
        if ch == "u":
            self.undo()
            return
        if ch == "q":
            self.root.destroy()
            return
        if ch in FLOOR_KEYS:
            self.floor_var.set(FLOOR_KEYS[ch])
            return
        if self.mode == "open":
            self._on_open_key(ch)
            return
        self._on_enclosed_key(ch)

    def _on_open_key(self, ch: str) -> None:
        if ch in KIND_KEYS:
            self.kind_var.set(KIND_KEYS[ch])
            self._on_kind_change()
            return
        if ch in MATERIAL_KEYS and self.kind_var.get() == "open_work":
            self.material_var.set(MATERIAL_KEYS[ch])
            return
        if ch in STRUCTURE_KEYS:
            self.structure_var.set(STRUCTURE_KEYS[ch])
            return
        if ch == "e":
            self.mark_enclosed()

    def _on_enclosed_key(self, ch: str) -> None:
        if ch == "a":
            self.toggle_half()
            return
        if ch == "o":
            self.mark_open()
            return
        if ch == "e":
            self.half_var.set(False)
            self._sync_half_fields()
            return
        if self.half_var.get() and ch in KIND_KEYS:
            self.kind_var.set(KIND_KEYS[ch])
            self._on_kind_change()
            return
        if self.half_var.get() and self.kind_var.get() == "open_work" and ch in MATERIAL_KEYS:
            self.material_var.set(MATERIAL_KEYS[ch])
            return
        if not self.half_var.get() and ch in SKIN_KEYS:
            self.skin_var.set(SKIN_KEYS[ch])

    def _on_kind_change(self) -> None:
        if self.kind_var.get() == "open_work":
            self._set_children_state(self.mat_fr, True)
        else:
            self.material_var.set("")
            self._set_children_state(self.mat_fr, False)

    def _set_children_state(self, frame: ttk.LabelFrame, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _sync_half_fields(self) -> None:
        if self.mode != "enclosed":
            return
        if self.half_var.get():
            if not self.kind_fr.winfo_ismapped():
                self.kind_fr.pack(fill=tk.X, padx=8, pady=2, before=self.skin_fr)
            if not self.mat_fr.winfo_ismapped():
                self.mat_fr.pack(fill=tk.X, padx=8, pady=2, before=self.skin_fr)
            self.skin_var.set("")
            self._set_children_state(self.skin_fr, False)
            self._on_kind_change()
            return
        if self.kind_fr.winfo_ismapped():
            self.kind_fr.pack_forget()
        if self.mat_fr.winfo_ismapped():
            self.mat_fr.pack_forget()
        self.kind_var.set("")
        self.material_var.set("")
        self._set_children_state(self.skin_fr, True)

    def toggle_half(self) -> None:
        self.half_var.set(not self.half_var.get())
        self._sync_half_fields()

    def _on_resize(self, _event: tk.Event | None = None) -> None:
        if self._resize_after is not None:
            self.root.after_cancel(self._resize_after)
        self._resize_after = self.root.after(50, self._render)

    def _pane_size(self, frame: ttk.Frame, fallback: tuple[int, int]) -> tuple[int, int]:
        frame.update_idletasks()
        w = int(frame.winfo_width())
        h = int(frame.winfo_height())
        if w < 80 or h < 80:
            return fallback
        return (max(64, w - 16), max(64, h - 16))

    def _render(self) -> None:
        if self._pil is not None:
            fitted = _fit(self._pil, *self._pane_size(self.crop_frame, (700, 520)))
            self._crop_photo = ImageTk.PhotoImage(fitted)
            self.crop_label.configure(image=self._crop_photo)
        if self._facade_pil is not None:
            fitted = _fit(self._facade_pil, *self._pane_size(self.facade_frame, (440, 520)))
            self._facade_photo = ImageTk.PhotoImage(fitted)
            self.facade_label.configure(image=self._facade_photo)

    def _counts(self) -> str:
        n_open = sum(1 for rec in self.labels.values() if rec.get("enclosure") == "open")
        n_enc = sum(1 for rec in self.labels.values() if rec.get("enclosure") == "enclosed")
        n_half = sum(1 for rec in self.labels.values() if rec.get("enclosure") == "half_enclosed")
        return f"labeled open={n_open}  enclosed={n_enc}  half={n_half}  this queue={len(self.queue)}"

    def _show_current(self) -> None:
        if not self.queue:
            empty = (
                "Open queue is empty. Enclosed crops are in the other UI."
                if self.mode == "open"
                else "Enclosed queue is empty. Mark crops with E in the open UI first."
            )
            self.path_var.set(empty)
            self.status_var.set(self._counts())
            self.crop_label.configure(image="")
            self.facade_label.configure(image="")
            self._crop_photo = None
            self._facade_photo = None
            self._pil = None
            self._facade_pil = None
            return

        self.index = max(0, min(self.index, len(self.queue) - 1))
        stem = self.queue[self.index]
        path = find_crop_image(self.crops_dir, stem)
        if path is None:
            self.queue.pop(self.index)
            self._show_current()
            return
        rec = self.labels.get(stem) or {}
        self._load_fields(rec)
        try:
            self._pil = Image.open(path).convert("RGB")
        except OSError as exc:
            messagebox.showerror("Image error", f"{path}\n{exc}")
            self.skip()
            return
        meta = self.manifest.get(stem) or {}
        photo_name = str(meta.get("photo") or f"{facade_stem_from_crop(stem)}.jpg")
        box = meta.get("box_xyxy")
        self._facade_pil = facade_with_box(self.photo_dir / photo_name, box if isinstance(box, list) else None)
        self._render()
        self.path_var.set(f"[{self.index + 1}/{len(self.queue)}]  {path.name}   facade {photo_name}")
        half = "   HALF" if self.mode == "enclosed" and self.half_var.get() else ""
        self.status_var.set(self._counts() + half)

    def _load_fields(self, rec: dict) -> None:
        kind = rec.get("kind") or ""
        floor = rec.get("floor_shape") or ""
        skin = rec.get("skin") or ""
        self.floor_var.set(floor if floor in FLOOR_SHAPES else "")
        structure = rec.get("structure") or "projecting"
        self.structure_var.set(structure if structure in LABEL_STRUCTURES else "projecting")
        self.kind_var.set(kind if kind in CLASSES else "")
        material = rec.get("material") or ""
        if self.mode == "enclosed":
            self.half_var.set(rec.get("enclosure") == "half_enclosed")
            self.skin_var.set(skin if skin in ENCLOSURE_SKINS else "")
            self._sync_half_fields()
        self._on_kind_change()
        if kind == "open_work" and material in MATERIALS:
            self.material_var.set(material)

    def _current_path(self) -> Path | None:
        if not self.queue:
            return None
        return find_crop_image(self.crops_dir, self.queue[self.index])

    def _remember(self, stem: str, prev: dict, prev_path: str) -> None:
        self.undo_stack.append({"stem": stem, "prev_label": prev, "prev_path": prev_path})
        self.back_stack.append(stem)

    def _advance_after_save(self) -> None:
        if self.queue:
            self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1
        self._show_current()

    def save(self) -> None:
        path = self._current_path()
        if path is None:
            return
        stem = path.stem
        floor = self.floor_var.get()
        if floor not in FLOOR_SHAPES:
            messagebox.showwarning("Floor", "Choose a floor: R rectangle, T triangle, C circle.")
            return
        if self.mode == "open":
            kind = self.kind_var.get()
            if kind not in CLASSES:
                messagebox.showwarning("Kind", "Choose kind: 1 open_work, 2 surface_panel, 3 solid.")
                return
            material = self.material_var.get() if kind == "open_work" else None
            if kind == "open_work" and material not in MATERIALS:
                messagebox.showwarning("Material", "open_work needs M metal or N masonry.")
                return
            structure = self.structure_var.get()
            if structure not in LABEL_STRUCTURES:
                messagebox.showwarning("Structure", "Choose P projecting or J juliet.")
                return
            enclosure, skin = "open", None
        else:
            if self.half_var.get():
                kind = self.kind_var.get()
                if kind not in CLASSES:
                    messagebox.showwarning("Kind", "half_enclosed needs a railing kind (1/2/3).")
                    return
                material = self.material_var.get() if kind == "open_work" else None
                if kind == "open_work" and material not in MATERIALS:
                    messagebox.showwarning("Material", "open_work needs M metal or N masonry.")
                    return
                enclosure, skin, structure = "half_enclosed", None, None
            else:
                skin = self.skin_var.get()
                if skin not in ENCLOSURE_SKINS:
                    messagebox.showwarning("Skin", "Choose P punched_window or G curtain_wall.")
                    return
                kind, material, enclosure, structure = None, None, "enclosed", None
        prev = dict(self.labels.get(stem) or {})
        try:
            self.labels = save_annotation(
                crops_dir=self.crops_dir,
                jsonl_path=self.labels_path,
                image_path=path,
                kind=kind,
                material=material,
                floor_shape=floor,
                enclosure=enclosure,
                skin=skin,
                structure=structure,
                labels=self.labels,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._remember(stem, prev, str(path))
        self._advance_after_save()

    def mark_enclosed(self) -> None:
        path = self._current_path()
        if path is None or self.mode != "open":
            return
        stem = path.stem
        prev = dict(self.labels.get(stem) or {})
        try:
            self.labels = send_to_enclosed(
                crops_dir=self.crops_dir,
                jsonl_path=self.labels_path,
                image_path=path,
                labels=self.labels,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._remember(stem, prev, str(path))
        self._advance_after_save()

    def mark_open(self) -> None:
        path = self._current_path()
        if path is None or self.mode != "enclosed":
            return
        stem = path.stem
        prev = dict(self.labels.get(stem) or {})
        try:
            self.labels = send_back_to_open(
                crops_dir=self.crops_dir,
                jsonl_path=self.labels_path,
                image_path=path,
                labels=self.labels,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._remember(stem, prev, str(path))
        self._advance_after_save()

    def skip(self) -> None:
        if not self.queue:
            return
        self.back_stack.append(self.queue[self.index])
        self.index = (self.index + 1) % len(self.queue)
        self._show_current()

    def previous(self) -> None:
        if not self.back_stack:
            return
        stem = self.back_stack.pop()
        if stem in self.queue:
            self.index = self.queue.index(stem)
        else:
            self.queue.insert(min(self.index, len(self.queue)), stem)
            self.index = self.queue.index(stem)
        self._show_current()

    def undo(self) -> None:
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        item = self.undo_stack.pop()
        stem = item["stem"]
        prev = item["prev_label"]
        cur = find_crop_image(self.crops_dir, stem)
        prev_path = Path(item["prev_path"])
        if cur is not None and prev_path.parent != cur.parent:
            prev_path.parent.mkdir(parents=True, exist_ok=True)
            if prev_path.exists() and prev_path != cur:
                prev_path.unlink()
            cur.rename(prev_path)
        if prev.get("kind") or prev.get("enclosure"):
            self.labels[stem] = {
                "path": prev.get("path") or f"{stem}.png",
                "kind": prev.get("kind"),
                "material": prev.get("material"),
                "floor_shape": prev.get("floor_shape"),
                "enclosure": prev.get("enclosure"),
                "skin": prev.get("skin"),
                "structure": prev.get("structure"),
            }
        else:
            self.labels.pop(stem, None)
        write_labels_jsonl(self.labels_path, self.labels)
        if stem not in self.queue:
            self.queue.insert(self.index, stem)
        self._show_current()


def _parser(mode: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=EXT_XML_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=EXT_XML_LABELS)
    ap.add_argument("--manifest", type=Path, default=EXT_XML_MANIFEST)
    ap.add_argument("--photo-dir", type=Path, default=EXT_PHOTO_DIR)
    ap.add_argument("--review-all", action="store_true")
    ap.set_defaults(mode=mode)
    return ap


def main_open(argv: list[str] | None = None) -> None:
    _main("open", argv)


def main_enclosed(argv: list[str] | None = None) -> None:
    _main("enclosed", argv)


def _main(mode: str, argv: list[str] | None) -> None:
    args = _parser(mode).parse_args(argv)
    root = tk.Tk()
    XmlLabelApp(
        root,
        mode=mode,
        crops_dir=args.crops_dir,
        labels_path=args.labels,
        manifest_path=args.manifest,
        photo_dir=args.photo_dir,
        review_all=args.review_all,
    )
    root.mainloop()
