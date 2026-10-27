"""Click window-grid cells that are enclosed volumes.

Selected cells are the ground truth. Next marks the open photo done:
painted cells are enclosed, and no cells means no enclosure. Skip leaves
the photo unlabeled. Finish closes the window.

The default queue is the held-out eval set: one facade per building in
``labels_ext_eval_balanced.jsonl`` (about 100 balcony crops, fewer facades).
A photo with no window grid yet is skipped until ``facade_dsl.json`` exists.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/enclosed_grid_ui.py

Keys: click toggles a cell, Next or Enter marks this photo done, S skips,
C clears the selection, Left goes back, U undo, Q or Finish closes.

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
    GridRecord,
    cell_at,
    facade_stems_from_crop_labels,
    GRID_FAILURE,
    find_photo,
    list_grid_stems,
    load_grid,
    load_records,
    record_for_save,
    write_records,
)

Fill = tuple[int, int, int, int]


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_grid(
    photo: Image.Image,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    selected: set[Cell],
) -> Image.Image:
    """Photo with every floor×bay cell outlined; selected cells filled."""
    base = photo.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line = max(2, base.width // 280)
    label = _font(max(13, base.width // 52))
    for fid, (y0, y1) in sorted(floor_y.items()):
        for bid, (x0, x1) in sorted(bay_x.items()):
            x0, x1 = sorted((int(x0), int(x1)))
            y0, y1 = sorted((int(y0), int(y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            on = (int(fid), int(bid)) in selected
            fill: Fill | None = (255, 140, 30, 110) if on else None
            outline: Fill = (255, 170, 40, 255) if on else (40, 220, 120, 230)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=line)
            draw.text((x0 + 4, y0 + 2), f"f{fid} b{bid}", fill=(255, 255, 255, 235), font=label)
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
        self.undo_stack: list[tuple[str, GridRecord | None]] = []
        self.floor_var = tk.StringVar(value="")
        self.skin_var = tk.StringVar(value="")
        self._photo: Image.Image | None = None
        self._floor_y: dict[int, tuple[int, int]] = {}
        self._bay_x: dict[int, tuple[int, int]] = {}
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
            "Click a cell to mark it enclosed. Next marks this photo done "
            "(no cells = no enclosure). Skip leaves it unlabeled."
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
        ttk.Button(bar, text="Finish", command=self.finish).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _e: self.save())
        self.root.bind("<Key-s>", lambda _e: self.skip())
        self.root.bind("<Left>", lambda _e: self.prev())
        self.root.bind("<Right>", lambda _e: self.next())
        self.root.bind("<Key-u>", lambda _e: self.undo())
        self.root.bind("<Key-q>", lambda _e: self.finish())
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
        except OSError as exc:
            messagebox.showerror("Image error", f"{photo}\n{exc}")
            self.skip()
            return
        saved = self.labels.get(stem)
        self.selected = set(saved.cells) if saved is not None else set()
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
        self.path_var.set(f"[{self.index + 1}/{len(self.queue)}]  {stem}   labeled {n_saved}/{len(self.queue)}")
        miss = f"   no DSL: {len(self.missing)}" if self.missing else ""
        self.status_var.set(f"{state}{miss}")
        self._render()

    def _render(self) -> None:
        self._resize_after = None
        if self._photo is None:
            return
        painted = render_grid(self._photo, self._floor_y, self._bay_x, self.selected)
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
        self.canvas.create_text(
            12,
            ch - 16,
            anchor=tk.W,
            fill="#f2f2f2",
            text=f"current: {choice}",
        )

    def _on_click(self, event: tk.Event) -> None:
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
        if hit in self.selected:
            self.selected.remove(hit)
        else:
            self.selected.add(hit)
        self._render()

    def clear_selection(self) -> None:
        self.selected.clear()
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
                )
            except ValueError as exc:
                messagebox.showinfo(
                    "Need shape and skin",
                    f"{exc}\n\nR rectangle, T triangle, C circle, P punched window, G curtain wall.",
                )
                return
            if prev is None:
                changed = bool(current.cells)
            else:
                changed = (
                    prev.cells != current.cells
                    or prev.floor_shape != current.floor_shape
                    or prev.skin != current.skin
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
        help="Balcony-crop labels. Facade id is the filename before _bal_.",
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stems = None
    if not args.all_photos:
        stems = facade_stems_from_crop_labels(args.crop_labels)
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
