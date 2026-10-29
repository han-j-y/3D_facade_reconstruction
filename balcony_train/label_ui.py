"""Tkinter Label UI for railing kind, material, floor shape, enclosure, skin, and structure.

Usage::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_ui.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_ui.py --crops-dir runs/balcony_clf/crops
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_ui.py --structure-pass

Keys: 1/2/3 kind (0 = no railing, enclosed only), M/N material (open_work),
R/T/C floor shape, O/A/E enclosure (open / half_enclosed / enclosed),
P/G skin when enclosed (punched_window / curtain_wall),
P/J structure when open (projecting / juliet),
Enter save, S skip, Left previous, U undo, Q quit. Juliet has no floor, so
the floor row is disabled and saved as null.

``--structure-pass`` shows open crops that have no structure yet, only names
starting with ``--structure-prefix`` (default ``cmp_b``; flux2 and web crops
stay unlabeled for structure).
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
    floor_required,
    kind_optional,
    merge_label_sources,
    queue_stems,
    save_annotation,
    write_labels_jsonl,
)
from balcony_train.labels import (  # noqa: E402
    CLASSES,
    ENCLOSURE_SKINS,
    ENCLOSURES,
    FLOOR_SHAPES,
    LABEL_STRUCTURES,
    MATERIALS,
    ensure_crop_dirs,
)
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL  # noqa: E402

KIND_KEYS = {"0": "", "1": "open_work", "2": "surface_panel", "3": "solid"}
ENCLOSURE_KEYS = {"o": "open", "a": "half_enclosed", "e": "enclosed"}
MATERIAL_KEYS = {"m": "metal", "n": "masonry"}
FLOOR_KEYS = {"r": "rectangle", "t": "triangle", "c": "circle"}
SKIN_KEYS = {"p": "punched_window", "g": "curtain_wall"}
# P is skin when enclosed and structure when open; only one row is shown.
STRUCTURE_KEYS = {"p": "projecting", "j": "juliet"}
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
        structure_pass: bool = False,
        structure_prefix: str = "cmp_b",
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
            need_floor_shape=True,
            need_enclosure=True,
            need_skin=True,
            need_structure=structure_pass,
        )
        if structure_pass:
            self.queue = [s for s in self.queue if s.startswith(structure_prefix)]
        self.index = 0
        self.undo_stack: list[dict] = []
        self.back_stack: list[str] = []
        self.kind_var = tk.StringVar(value="")
        self.material_var = tk.StringVar(value="")
        self.floor_var = tk.StringVar(value="")
        self.enclosure_var = tk.StringVar(value="")
        self.skin_var = tk.StringVar(value="")
        self.structure_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")
        self._photo: ImageTk.PhotoImage | None = None
        self._pil: Image.Image | None = None
        self._current_path: Path | None = None
        self._resize_after: str | None = None
        self._build()
        self.structure_var.trace_add("write", lambda *_: self._sync_floor())
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

        kind_fr = ttk.LabelFrame(
            self.root, text="Kind (1/2/3, 0 = no railing for enclosed)", padding=8
        )
        kind_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(("", *CLASSES), ("0", "1", "2", "3")):
            ttk.Radiobutton(
                kind_fr,
                text=f"{key}: {name or 'none'}",
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

        self.floor_fr = ttk.LabelFrame(
            self.root,
            text="Floor shape — R rectangle / T triangle / C circle (none for juliet)",
            padding=8,
        )
        self.floor_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(FLOOR_SHAPES, ("R", "T", "C")):
            ttk.Radiobutton(
                self.floor_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.floor_var,
            ).pack(side=tk.LEFT, padx=8)

        enc_fr = ttk.LabelFrame(
            self.root,
            text="Enclosure — O open / A half_enclosed / E enclosed",
            padding=8,
        )
        enc_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(ENCLOSURES, ("O", "A", "E")):
            ttk.Radiobutton(
                enc_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.enclosure_var,
                command=self._sync_skin,
            ).pack(side=tk.LEFT, padx=8)

        self.skin_fr = ttk.LabelFrame(
            self.root,
            text="Enclosed skin — P punched_window / G curtain_wall",
            padding=8,
        )
        for name, key in zip(ENCLOSURE_SKINS, ("P", "G")):
            ttk.Radiobutton(
                self.skin_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.skin_var,
            ).pack(side=tk.LEFT, padx=8)

        # Always shown; disabled for enclosed / half_enclosed. Picking one sets open.
        self.structure_fr = ttk.LabelFrame(
            self.root,
            text="Structure (open) — P projecting / J juliet",
            padding=8,
        )
        self.structure_fr.pack(fill=tk.X, padx=8, pady=4)
        for name, key in zip(LABEL_STRUCTURES, ("P", "J")):
            ttk.Radiobutton(
                self.structure_fr,
                text=f"{key}: {name}",
                value=name,
                variable=self.structure_var,
                command=self._on_structure_pick,
            ).pack(side=tk.LEFT, padx=8)

        self.btns = ttk.Frame(self.root, padding=8)
        self.btns.pack(fill=tk.X)
        btns = self.btns
        ttk.Button(btns, text="Save (Enter)", command=self.save).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btns, text="Previous", command=self.previous).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Skip (S)", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Quit (Q)", command=self.root.destroy).pack(
            side=tk.RIGHT, padx=4
        )

        help_txt = (
            "Material is required only for open_work. Floor shape (except juliet) and "
            "enclosure are required. Kind may be none only for enclosed. "
            "Skin is required only for enclosed, structure only for open. "
            "Labels → labels.jsonl; images move into crops/{kind}/ (no kind: unlabeled/)."
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
        if ch in FLOOR_KEYS and floor_required(self.structure_var.get()):
            self.floor_var.set(FLOOR_KEYS[ch])
            return
        if ch in ENCLOSURE_KEYS:
            self.enclosure_var.set(ENCLOSURE_KEYS[ch])
            self._sync_skin()
            return
        if ch in SKIN_KEYS and self.enclosure_var.get() == "enclosed":
            self.skin_var.set(SKIN_KEYS[ch])
            return
        if ch in STRUCTURE_KEYS and self.enclosure_var.get() not in ("enclosed", "half_enclosed"):
            self.structure_var.set(STRUCTURE_KEYS[ch])
            self._on_structure_pick()
            return
        if keysym in ("Return", "KP_Enter"):
            self.save()
            return
        if keysym == "Left":
            self.previous()
            return
        if keysym == "Right":
            self.skip()
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

    def _sync_skin(self) -> None:
        """Show the skin row for enclosed; enable the structure row unless enclosed."""
        self._sync_structure()
        if self.enclosure_var.get() == "enclosed":
            if not self.skin_fr.winfo_manager():
                self.skin_fr.pack(fill=tk.X, padx=8, pady=4, before=self.btns)
            return
        self.skin_var.set("")
        if self.skin_fr.winfo_manager():
            self.skin_fr.pack_forget()

    def _sync_structure(self) -> None:
        enabled = self.enclosure_var.get() not in ("enclosed", "half_enclosed")
        if not enabled:
            self.structure_var.set("")
        for child in self.structure_fr.winfo_children():
            child.configure(state="normal" if enabled else "disabled")

    def _sync_floor(self) -> None:
        enabled = floor_required(self.structure_var.get())
        if not enabled:
            self.floor_var.set("")
        for child in self.floor_fr.winfo_children():
            child.configure(state="normal" if enabled else "disabled")

    def _on_structure_pick(self) -> None:
        """A structure is only for open balconies, so an unset enclosure becomes open."""
        if self.enclosure_var.get() not in ENCLOSURES:
            self.enclosure_var.set("open")
            self._sync_skin()

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
        n_floor = {name: 0 for name in FLOOR_SHAPES}
        n_enc = {name: 0 for name in ENCLOSURES}
        n_skin = {name: 0 for name in ENCLOSURE_SKINS}
        n_structure = {name: 0 for name in LABEL_STRUCTURES}
        n_open_no_mat = 0
        for rec in self.labels.values():
            enc = rec.get("enclosure")
            if enc in n_enc:
                n_enc[enc] += 1
            kind = rec.get("kind")
            if kind in n_kind:
                n_kind[kind] += 1
            if kind == "open_work":
                mat = rec.get("material")
                if mat in n_mat:
                    n_mat[mat] += 1
                else:
                    n_open_no_mat += 1
            floor = rec.get("floor_shape")
            if floor in n_floor:
                n_floor[floor] += 1
            skin = rec.get("skin")
            if skin in n_skin:
                n_skin[skin] += 1
            structure = rec.get("structure")
            if structure in n_structure:
                n_structure[structure] += 1
        parts = [f"{k}={n_kind[k]}" for k in CLASSES]
        parts.append(
            f"open_work materials metal={n_mat['metal']} masonry={n_mat['masonry']}"
            + (f" missing={n_open_no_mat}" if n_open_no_mat else "")
        )
        parts.append("floor " + " ".join(f"{name}={n_floor[name]}" for name in FLOOR_SHAPES))
        parts.append("enclosure " + " ".join(f"{name}={n_enc[name]}" for name in ENCLOSURES))
        parts.append("skin " + " ".join(f"{name}={n_skin[name]}" for name in ENCLOSURE_SKINS))
        parts.append(
            "structure " + " ".join(f"{name}={n_structure[name]}" for name in LABEL_STRUCTURES)
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
            self.floor_var.set("")
            self.enclosure_var.set("")
            self.skin_var.set("")
            self.structure_var.set("")
            self._sync_skin()
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
        floor = rec.get("floor_shape") or ""
        self.kind_var.set(kind if kind in CLASSES else "")
        self._on_kind_change()
        if kind == "open_work" and material in MATERIALS:
            self.material_var.set(material)
        self.floor_var.set(floor if floor in FLOOR_SHAPES else "")
        enclosure = rec.get("enclosure") or ""
        self.enclosure_var.set(enclosure if enclosure in ENCLOSURES else "")
        skin = rec.get("skin") or ""
        self.skin_var.set(skin if skin in ENCLOSURE_SKINS else "")
        self._sync_skin()
        structure = rec.get("structure") or ""
        self.structure_var.set(structure if structure in LABEL_STRUCTURES else "")

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
        enclosure = self.enclosure_var.get()
        if enclosure not in ENCLOSURES:
            messagebox.showwarning(
                "Enclosure required",
                "Select enclosure: open (O), half_enclosed (A), or enclosed (E).",
            )
            return
        kind = self.kind_var.get()
        if kind not in CLASSES and not kind_optional(enclosure):
            messagebox.showwarning(
                "Kind required", "Select kind (1/2/3). Only enclosed may have none (0)."
            )
            return
        kind = kind if kind in CLASSES else None
        material = self.material_var.get() if kind == "open_work" else None
        if kind == "open_work" and material not in MATERIALS:
            messagebox.showwarning(
                "Material required",
                "open_work needs material: metal (M) or masonry (N).",
            )
            return
        skin = self.skin_var.get() if enclosure == "enclosed" else ""
        if enclosure == "enclosed" and skin not in ENCLOSURE_SKINS:
            messagebox.showwarning(
                "Skin required",
                "Enclosed needs a skin: punched_window (P) or curtain_wall (G).",
            )
            return
        structure = self.structure_var.get() if enclosure == "open" else ""
        if enclosure == "open" and structure not in LABEL_STRUCTURES:
            messagebox.showwarning(
                "Structure required",
                "Open needs a structure: projecting (P) or juliet (J).",
            )
            return
        floor = self.floor_var.get() if floor_required(structure) else ""
        if floor_required(structure) and floor not in FLOOR_SHAPES:
            messagebox.showwarning(
                "Floor shape required",
                "Select floor shape: rectangle (R), triangle (T), or circle (C).",
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
                floor_shape=floor or None,
                enclosure=enclosure,
                skin=skin or None,
                structure=structure or None,
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
        self.back_stack.append(stem)
        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1
        self._show_current()

    def skip(self) -> None:
        if not self.queue:
            return
        self.back_stack.append(self.queue[self.index])
        self.index = (self.index + 1) % len(self.queue)
        self._show_current()

    def previous(self) -> None:
        """Show the crop visited before this one. A saved crop comes back for editing."""
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
        # Move file back toward previous location when possible.
        cur = find_image(self.crops_dir, stem)
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument(
        "--review-all",
        action="store_true",
        help="include already complete labels (for editing)",
    )
    ap.add_argument(
        "--structure-pass",
        action="store_true",
        help="show open crops with no structure yet (names starting with --structure-prefix)",
    )
    ap.add_argument("--structure-prefix", default="cmp_b")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    LabelApp(
        root,
        crops_dir=args.crops_dir,
        labels_path=args.labels,
        review_all=args.review_all,
        structure_pass=args.structure_pass,
        structure_prefix=args.structure_prefix,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
