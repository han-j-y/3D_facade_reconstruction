"""Click window-grid cells that are enclosed volumes.

Selected cells are the ground truth. Next marks the open photo done:
painted cells are enclosed, and no cells means no enclosure. Skip leaves
the photo unlabeled. Finish closes the window.

The default queue is the 133-photo test set: labeled facades in
``labels_ext_all.jsonl`` whose photo is in ``data/extended_selected``.
Grid-failure rows are left out. A photo with no window grid yet is skipped
until ``facade_dsl.json`` exists.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/enclosed_grid_ui.py

The grid is the per-floor bay layout (``meta.floor_bays``): each floor is
drawn with its own bays. A bay that spans several union columns shows them as
thin inner lines; labels stay stored per union column. A red outline marks a
half-labeled bay. ``--review-partial`` queues only photos that have one, and
``--stems`` queues the named photos.

Keys: click toggles a whole floor bay, Shift+click toggles one column of it,
Next or Enter marks this photo done, S skips, C clears the selection, Left
goes back, U undo, Q or Finish closes.

Ignore cells (grey hatching) are cells whose label is ambiguous from the photo.
Right-click (Shift+right-click: one column) or the Ignore mode button (I)
toggles them. They are saved as ``ignore`` next to ``cells`` and do not change
the oriel cells.

The eval labels file also asks for one floor shape and one skin on each
photo that has cells. R rectangle, T triangle, C circle (Clear is Delete),
P punched window, G curtain wall. An empty selection still means no enclosure
and stores neither. Train cell labels (``--no-plan``) keep C as clear.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_CROP_LABELS,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    Cell,
    FloorBays,
    GridRecord,
    bay_cells,
    cell_at,
    facade_stems_from_crop_labels,
    GRID_FAILURE,
    find_photo,
    list_grid_stems,
    load_floor_bays,
    load_grid,
    load_labels,
    load_records,
    partial_bays,
    record_for_save,
    write_records,
)

Fill = tuple[int, int, int, int]


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _hatch(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int) -> None:
    """Grey fill with diagonal lines: an ignore cell."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(90, 90, 90, 120))
    step = max(10, 6 * width)
    for k in range(x0 - (y1 - y0), x1, step):
        # segment of the line x - y = k - y0 inside the box
        a0, a1 = max(x0, k), min(x1, k + (y1 - y0))
        if a1 > a0:
            draw.line((a0, y0 + a0 - k, a1, y0 + a1 - k), fill=(230, 230, 230, 200), width=max(1, width // 2))


def render_grid(
    photo: Image.Image,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    selected: set[Cell],
    floor_bays: FloorBays | None = None,
    ignore: set[Cell] | None = None,
) -> Image.Image:
    """Photo with every floor bay outlined; selected cells filled.

    Without ``floor_bays`` every column is its own bay. A half-labeled bay gets
    a red outline. ``ignore`` cells get grey hatching.
    """
    ignore = ignore or set()
    if floor_bays is None:
        floor_bays = {f: [(c, c + 1) for c in sorted(bay_x)] for f in floor_y}
    base = photo.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line = max(2, base.width // 280)
    label = _font(max(13, base.width // 52))
    for fid, (y0, y1) in sorted(floor_y.items()):
        y0, y1 = sorted((int(y0), int(y1)))
        if y1 <= y0:
            continue
        for c0, c1 in floor_bays.get(fid, []):
            span = [c for c in range(c0, c1) if c in bay_x]
            if not span:
                continue
            x0 = int(min(bay_x[c][0] for c in span))
            x1 = int(max(bay_x[c][1] for c in span))
            hit = [c for c in span if (int(fid), c) in selected]
            for c in hit:
                draw.rectangle((int(bay_x[c][0]), y0, int(bay_x[c][1]), y1), fill=(255, 140, 30, 110))
            for c in span:
                if (int(fid), c) in ignore:
                    _hatch(draw, (int(bay_x[c][0]), y0, int(bay_x[c][1]), y1), line)
            for c in span[1:]:
                xc = int(bay_x[c][0])
                draw.line((xc, y0, xc, y1), fill=(40, 220, 120, 120), width=1)
            if len(hit) == len(span):
                outline: Fill = (255, 170, 40, 255)
            elif hit:
                outline = (255, 50, 50, 255)
            else:
                outline = (40, 220, 120, 230)
            draw.rectangle((x0, y0, x1, y1), outline=outline, width=line)
            name = f"b{c0}" if c1 - c0 == 1 else f"b{c0}-{c1 - 1}"
            draw.text((x0 + 4, y0 + 2), f"f{fid} {name}", fill=(255, 255, 255, 235), font=label)
    return Image.alpha_composite(base, layer).convert("RGB")


class EnclosedGridApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        photo_dir: Path,
        dsl_dir: Path,
        labels_path: Path,
        stems: list[str] | None = None,
        plan: bool = False,
    ) -> None:
        self.root = root
        self.photo_dir = Path(photo_dir)
        self.dsl_dir = Path(dsl_dir)
        self.labels_path = Path(labels_path)
        self.plan = bool(plan)
        ready, missing = list_grid_stems(self.photo_dir, self.dsl_dir, stems=stems)
        self.queue = ready
        self.missing = missing
        self.labels = load_records(self.labels_path)
        self.index = 0
        for i, stem in enumerate(self.queue):
            rec = self.labels.get(stem)
            if rec is None:
                self.index = i
                break
            if self.plan and rec.cells and (rec.floor_shape is None or rec.skin is None):
                self.index = i
                break
        self.selected: set[Cell] = set()
        self.ignored: set[Cell] = set()
        self.ignore_mode = tk.BooleanVar(value=False)
        self.undo_stack: list[tuple[str, GridRecord | None]] = []
        self.floor_var = tk.StringVar(value="")
        self.skin_var = tk.StringVar(value="")
        self._photo: Image.Image | None = None
        self._floor_y: dict[int, tuple[int, int]] = {}
        self._bay_x: dict[int, tuple[int, int]] = {}
        self._floor_bays: FloorBays = {}
        self._tk_image: ImageTk.PhotoImage | None = None
        self._geom: tuple[int, int, float] = (0, 0, 1.0)
        self._resize_after: str | None = None
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._build()
        self._bind_keys()
        self._show_current()

    def _build(self) -> None:
        self.root.title("Enclosed grid label UI")
        self.root.geometry("1100x920")
        self.root.minsize(800, 640)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.path_var, font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10)).pack(anchor=tk.W)
        hint = (
            "Click a floor bay to mark it enclosed (Shift+click: one column of it). "
            "Next marks this photo done (no cells = no enclosure). Skip leaves it unlabeled. "
            "Red outline = half-labeled bay. Right-click or Ignore mode (I) marks an ambiguous cell as ignore."
        )
        if self.plan:
            hint += " Enclosure photos also need a floor shape and a skin."
        ttk.Label(top, text=hint).pack(anchor=tk.W)
        if self.plan:
            plan = ttk.Frame(top)
            plan.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(plan, text="Floor").pack(side=tk.LEFT, padx=(0, 6))
            for name, key in (("rectangle", "R"), ("triangle", "T"), ("circle", "C")):
                ttk.Radiobutton(
                    plan, text=f"{name} ({key})", value=name, variable=self.floor_var
                ).pack(side=tk.LEFT, padx=4)
            ttk.Label(plan, text="   Skin").pack(side=tk.LEFT, padx=(12, 6))
            for name, key in (("punched_window", "P"), ("curtain_wall", "G")):
                ttk.Radiobutton(
                    plan, text=f"{name} ({key})", value=name, variable=self.skin_var
                ).pack(side=tk.LEFT, padx=4)
            self.floor_var.trace_add("write", lambda *_a: self._render())
            self.skin_var.trace_add("write", lambda *_a: self._render())

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Shift-Button-1>", lambda e: self._on_click(e, single=True))
        self.canvas.bind("<Button-3>", lambda e: self._on_click(e, ignore=True))
        self.canvas.bind("<Shift-Button-3>", lambda e: self._on_click(e, single=True, ignore=True))
        self.canvas.bind("<Configure>", self._on_configure)

        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Save (Enter)", command=self.save).pack(side=tk.LEFT, padx=4)
        clear_key = "Delete" if self.plan else "C"
        ttk.Button(bar, text=f"Clear ({clear_key})", command=self.clear_selection).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Skip (S)", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Prev", command=self.prev).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Next", command=self.next).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            bar,
            text="Ignore mode (I): click marks ignore",
            variable=self.ignore_mode,
            command=self._render,
        ).pack(side=tk.LEFT, padx=12)
        ttk.Button(bar, text="Finish", command=self.finish).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _e: self.save())
        self.root.bind("<Key-s>", lambda _e: self.skip())
        self.root.bind("<Left>", lambda _e: self.prev())
        self.root.bind("<Right>", lambda _e: self.next())
        self.root.bind("<Key-u>", lambda _e: self.undo())
        self.root.bind("<Key-q>", lambda _e: self.finish())
        self.root.bind("<Key-i>", lambda _e: self._toggle_ignore_mode())
        if self.plan:
            self.root.bind("<Delete>", lambda _e: self.clear_selection())
            self.root.bind("<BackSpace>", lambda _e: self.clear_selection())
            self.root.bind("<Key-r>", lambda _e: self.floor_var.set("rectangle"))
            self.root.bind("<Key-t>", lambda _e: self.floor_var.set("triangle"))
            self.root.bind("<Key-c>", lambda _e: self.floor_var.set("circle"))
            self.root.bind("<Key-p>", lambda _e: self.skin_var.set("punched_window"))
            self.root.bind("<Key-g>", lambda _e: self.skin_var.set("curtain_wall"))
        else:
            self.root.bind("<Key-c>", lambda _e: self.clear_selection())

    def _on_configure(self, _event: tk.Event) -> None:
        if self._resize_after is not None:
            self.root.after_cancel(self._resize_after)
        self._resize_after = self.root.after(80, self._render)

    def _show_current(self) -> None:
        if not self.queue:
            self.path_var.set("No photos with a window grid.")
            extra = f"  missing DSL: {', '.join(self.missing)}" if self.missing else ""
            self.status_var.set(extra.strip())
            self.canvas.delete("all")
            self._photo = None
            return
        self.index = max(0, min(self.index, len(self.queue) - 1))
        stem = self.queue[self.index]
        photo = find_photo(self.photo_dir, stem)
        dsl = self.dsl_dir / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            self.queue.pop(self.index)
            self._show_current()
            return
        try:
            self._photo = Image.open(photo).convert("RGB")
            self._floor_y, self._bay_x = load_grid(dsl)
            self._floor_bays = load_floor_bays(dsl, self._floor_y, self._bay_x)
        except OSError as exc:
            messagebox.showerror("Image error", f"{photo}\n{exc}")
            self.skip()
            return
        saved = self.labels.get(stem)
        self.selected = set(saved.cells) if saved is not None else set()
        self.ignored = set(saved.ignore) if saved is not None else set()
        if self.plan:
            self.floor_var.set(saved.floor_shape if saved is not None and saved.floor_shape else "")
            self.skin_var.set(saved.skin if saved is not None and saved.skin else "")
        n_saved = sum(1 for s in self.queue if s in self.labels)
        state = "not done"
        if saved is not None and saved.status == GRID_FAILURE:
            state = "grid failure"
        elif saved is not None:
            if not saved.cells:
                state = "done, no enclosure"
            else:
                plan = ""
                if self.plan:
                    plan = (
                        f", {saved.floor_shape}, {saved.skin}"
                        if saved.floor_shape and saved.skin
                        else ", plan missing"
                    )
                state = f"done, {len(saved.cells)} cells{plan}"
            if saved.ignore:
                state += f", {len(saved.ignore)} ignore"
        self.path_var.set(f"[{self.index + 1}/{len(self.queue)}]  {stem}   labeled {n_saved}/{len(self.queue)}")
        miss = f"   no DSL: {len(self.missing)}" if self.missing else ""
        self.status_var.set(f"{state}{miss}")
        self._render()

    def _render(self) -> None:
        self._resize_after = None
        if self._photo is None:
            return
        painted = render_grid(
            self._photo, self._floor_y, self._bay_x, self.selected, self._floor_bays, self.ignored
        )
        cw = max(64, int(self.canvas.winfo_width()))
        ch = max(64, int(self.canvas.winfo_height()))
        scale = min(cw / painted.width, ch / painted.height)
        dw = max(1, int(round(painted.width * scale)))
        dh = max(1, int(round(painted.height * scale)))
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self._geom = (ox, oy, scale)
        shown = painted.resize((dw, dh), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._tk_image)
        n = len(self.selected)
        choice = "no enclosure" if n == 0 else f"{n} cell(s)"
        if self.plan and n:
            floor = self.floor_var.get() or "shape?"
            skin = self.skin_var.get() or "skin?"
            choice = f"{choice}, {floor}, {skin}"
        n_partial = len(partial_bays(self.selected, self._floor_bays))
        if n_partial:
            choice = f"{choice}   ! {n_partial} half-labeled bay(s) (red)"
        if self.ignored:
            choice = f"{choice}   ignore {len(self.ignored)} cell(s)"
        if self.ignore_mode.get():
            choice = f"{choice}   [IGNORE MODE]"
        self.canvas.create_text(
            12,
            ch - 16,
            anchor=tk.W,
            fill="#f2f2f2",
            text=f"current: {choice}",
        )

    def _toggle_ignore_mode(self) -> None:
        self.ignore_mode.set(not self.ignore_mode.get())
        self._render()

    def _on_click(self, event: tk.Event, single: bool = False, ignore: bool = False) -> None:
        """Toggle the floor bay under the pointer, or one column with ``single``.

        ``ignore`` (right-click, or Ignore mode) toggles ignore cells instead.
        """
        if self._photo is None:
            return
        ox, oy, scale = self._geom
        if scale <= 0:
            return
        x = (event.x - ox) / scale
        y = (event.y - oy) / scale
        hit = cell_at(x, y, self._floor_y, self._bay_x)
        if hit is None:
            return
        cells = [hit] if single else bay_cells(hit, self._floor_bays)
        target = self.ignored if ignore or self.ignore_mode.get() else self.selected
        if all(c in target for c in cells):
            target.difference_update(cells)
        else:
            target.update(cells)
        self._render()

    def clear_selection(self) -> None:
        """Clear the oriel cells and the ignore cells."""
        self.selected.clear()
        self.ignored.clear()
        self._render()

    def _write_current(self) -> bool:
        """Save the open photo, including an empty selection (no enclosure)."""
        if not self.queue:
            return False
        stem = self.queue[self.index]
        prev = self.labels.get(stem)
        prev_copy = prev.copy() if prev is not None else None
        try:
            saved = record_for_save(
                self.selected,
                self.floor_var.get(),
                self.skin_var.get(),
                plan=self.plan,
                previous=prev,
                ignore=self.ignored,
            )
        except ValueError as exc:
            messagebox.showinfo(
                "Need shape and skin",
                f"{exc}\n\nR rectangle, T triangle, C circle, P punched window, G curtain wall.",
            )
            return False
        self.labels[stem] = saved
        try:
            write_records(self.labels_path, self.labels)
        except OSError as exc:
            if prev_copy is None:
                self.labels.pop(stem, None)
            else:
                self.labels[stem] = prev_copy
            messagebox.showerror("Save failed", str(exc))
            return False
        self.undo_stack.append((stem, prev_copy))
        return True

    def save(self) -> None:
        self.next()

    def finish(self) -> None:
        """Close. Keep unsaved clicks; do not mark an untouched photo as empty."""
        if self.queue and self._photo is not None:
            stem = self.queue[self.index]
            prev = self.labels.get(stem)
            try:
                current = record_for_save(
                    self.selected,
                    self.floor_var.get(),
                    self.skin_var.get(),
                    plan=self.plan,
                    previous=prev,
                    ignore=self.ignored,
                )
            except ValueError as exc:
                messagebox.showinfo(
                    "Need shape and skin",
                    f"{exc}\n\nR rectangle, T triangle, C circle, P punched window, G curtain wall.",
                )
                return
            if prev is None:
                changed = bool(current.cells or current.ignore)
            else:
                changed = (
                    prev.cells != current.cells
                    or prev.floor_shape != current.floor_shape
                    or prev.skin != current.skin
                    or prev.ignore != current.ignore
                )
            if changed and not self._write_current():
                return
        self.root.destroy()

    def _move(self, step: int) -> None:
        if not self.queue:
            return
        self.index = (self.index + step) % len(self.queue)
        self._show_current()

    def skip(self) -> None:
        """Leave this photo unlabeled and show the next one."""
        self._move(1)

    def next(self) -> None:
        """Mark this photo evaluated, then show the next one.

        Painted cells are enclosed. No painted cells means no enclosure.
        """
        if self._write_current():
            self._move(1)

    def prev(self) -> None:
        self._move(-1)

    def undo(self) -> None:
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        stem, prev = self.undo_stack.pop()
        if prev is None:
            self.labels.pop(stem, None)
        else:
            self.labels[stem] = prev.copy()
        write_records(self.labels_path, self.labels)
        if stem in self.queue:
            self.index = self.queue.index(stem)
        self._show_current()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument(
        "--crop-labels",
        type=Path,
        default=DEFAULT_EVAL_CROP_LABELS,
        help="Balcony-crop labels, or the cell-label file. A cell-label file keeps photos in --in-dir that are not grid failures.",
    )
    ap.add_argument(
        "--all-photos",
        action="store_true",
        help="Every photo in --in-dir that has a window grid (ignore --crop-labels).",
    )
    ap.add_argument(
        "--plan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask for floor shape and skin on enclosure photos. Default: on for the eval labels file.",
    )
    ap.add_argument("--stems", nargs="+", default=None, help="Queue only these photos.")
    ap.add_argument(
        "--review-partial",
        action="store_true",
        help="Queue only labeled photos with a half-labeled floor bay.",
    )
    return ap.parse_args()


def partial_stems(stems: list[str], labels_path: Path, dsl_dir: Path) -> list[str]:
    """Labeled photos in ``stems`` with at least one half-labeled floor bay."""
    records = load_records(labels_path)
    out = []
    for stem in stems:
        rec = records.get(stem)
        dsl = Path(dsl_dir) / stem / "facade_dsl.json"
        if rec is None or not rec.cells or not dsl.is_file():
            continue
        floor_y, bay_x = load_grid(dsl)
        if partial_bays(rec.cells, load_floor_bays(dsl, floor_y, bay_x)):
            out.append(stem)
    return out


def main() -> None:
    args = parse_args()
    stems = None
    if not args.all_photos:
        if Path(args.crop_labels).resolve() == Path(args.labels).resolve():
            stems = [
                stem
                for stem in sorted(load_labels(args.labels))
                if find_photo(args.in_dir, stem) is not None
            ]
        else:
            stems = facade_stems_from_crop_labels(args.crop_labels)
    if args.stems:
        stems = list(args.stems)
    if args.review_partial:
        stems = partial_stems(stems if stems is not None else sorted(load_records(args.labels)), args.labels, args.dsl_dir)
        print(f"photos with a half-labeled bay {len(stems)}", flush=True)
    ready, missing = list_grid_stems(args.in_dir, args.dsl_dir, stems=stems)
    plan = (
        args.plan
        if args.plan is not None
        else Path(args.labels).resolve() == DEFAULT_EVAL_LABELS_JSONL.resolve()
    )
    print(f"grid ready {len(ready)}  waiting for window grid {len(missing)}", flush=True)
    if missing:
        print("no facade_dsl.json: " + ", ".join(missing), flush=True)
    print(f"plan labels {'on' if plan else 'off'}", flush=True)
    root = tk.Tk()
    EnclosedGridApp(
        root,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        labels_path=args.labels,
        stems=stems,
        plan=plan,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
