"""Label enclosed floor plan on cells cropped from ``data/base``.

Only cells already marked enclosed in ``labels_train.jsonl`` are shown.
One choice (rectangle or circle) applies to every cell of that photo.
Crops and labels are written so ``train.py --floor-only`` can read them:
``runs/enclosed_floor/crops/unlabeled/<stem>_f<floor>b<bay>.png`` and
``runs/enclosed_floor/labels.jsonl`` (enclosure=enclosed, no railing kind).

``data/base`` jpgs are the photos. The same-stem ``.png`` files are CMP
label maps and are not used when a jpg is present.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/enclosed_floor_ui.py

Keys: R rectangle, C circle, Enter saves, S skips, Left/Right move, Q closes.
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

from balcony_train.depth_cells import cell_box  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    Cell,
    find_photo,
    load_grid,
)
from balcony_train.enclosed_grid_store import load_labels as load_cell_labels  # noqa: E402
from balcony_train.label_store import load_labels as load_crop_labels  # noqa: E402
from balcony_train.label_store import write_labels_jsonl  # noqa: E402

DEFAULT_PHOTO_DIR = ROOT / "data" / "base"
DEFAULT_CELL_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train.jsonl"
DEFAULT_CROPS_DIR = ROOT / "runs" / "enclosed_floor" / "crops"
DEFAULT_LABELS = ROOT / "runs" / "enclosed_floor" / "labels.jsonl"
SHAPES = ("rectangle", "circle")


def crop_stem(facade: str, cell: Cell) -> str:
    """Filename stem for one enclosed cell. Unique across facades."""
    floor, bay = cell
    return f"{facade}_f{int(floor)}b{int(bay)}"


def enclosed_queue(
    cell_labels: dict[str, list[Cell]],
    photo_dir: Path,
    dsl_dir: Path,
) -> list[str]:
    """Facades that have at least one enclosed cell, a photo, and a window grid."""
    ready: list[str] = []
    for stem in sorted(cell_labels):
        if not cell_labels[stem]:
            continue
        if find_photo(photo_dir, stem) is None:
            continue
        if not (Path(dsl_dir) / stem / "facade_dsl.json").is_file():
            continue
        ready.append(stem)
    return ready


def photo_shape(records: dict[str, dict], facade: str, cells: list[Cell]) -> str | None:
    """The shared plan if every cell of this facade is already labeled the same way."""
    found: list[str] = []
    for cell in cells:
        rec = records.get(crop_stem(facade, cell)) or {}
        shape = rec.get("floor_shape")
        if shape not in SHAPES:
            return None
        found.append(str(shape))
    if not found or len(set(found)) != 1:
        return None
    return found[0]


def assign_photo_shape(
    records: dict[str, dict],
    facade: str,
    cells: list[Cell],
    shape: str,
) -> None:
    """Write one floor plan onto each cell and drop stale cells of this facade."""
    if shape not in SHAPES:
        raise ValueError(f"floor shape must be one of {SHAPES}")
    keep = {crop_stem(facade, cell) for cell in cells}
    prefix = f"{facade}_f"
    for stem in list(records):
        if stem.startswith(prefix) and stem not in keep:
            records.pop(stem, None)
    for cell in cells:
        stem = crop_stem(facade, cell)
        records[stem] = {
            "path": f"{stem}.png",
            "kind": None,
            "material": None,
            "floor_shape": shape,
            "enclosure": "enclosed",
        }


def extract_cells(
    photo: Image.Image,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    cells: list[Cell],
) -> list[tuple[Cell, Image.Image]]:
    """Crop each labeled cell. Cells missing from the grid are skipped."""
    out: list[tuple[Cell, Image.Image]] = []
    for cell in cells:
        floor, bay = int(cell[0]), int(cell[1])
        if floor not in floor_y or bay not in bay_x:
            continue
        box = cell_box((floor, bay), floor_y, bay_x)
        out.append(((floor, bay), photo.crop(tuple(box)).convert("RGB")))
    return out


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_sheet(
    photo: Image.Image,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    cells: list[Cell],
    shape: str,
) -> Image.Image:
    """Facade with the enclosed cells marked, and those crops underneath."""
    base = photo.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line = max(2, base.width // 280)
    label = _font(max(13, base.width // 52))
    chosen = set(cells)
    for fid, (y0, y1) in sorted(floor_y.items()):
        for bid, (x0, x1) in sorted(bay_x.items()):
            on = (int(fid), int(bid)) in chosen
            fill = (255, 140, 30, 110) if on else None
            outline = (255, 170, 40, 255) if on else (80, 80, 80, 120)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=line if on else 1)
            if on:
                draw.text((x0 + 4, y0 + 2), f"f{fid} b{bid}", fill=(255, 255, 255, 235), font=label)
    facade = Image.alpha_composite(base, layer).convert("RGB")
    crops = extract_cells(photo, floor_y, bay_x, cells)
    thumb_h = 180
    gap = 8
    thumbs: list[Image.Image] = []
    font = _font(16)
    for cell, crop in crops:
        scale = thumb_h / max(1, crop.height)
        tw = max(1, int(round(crop.width * scale)))
        thumb = crop.resize((tw, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tw, thumb_h + 24), (24, 24, 24))
        tile.paste(thumb, (0, 0))
        ImageDraw.Draw(tile).text((4, thumb_h + 2), f"f{cell[0]}b{cell[1]}", fill=(240, 240, 240), font=font)
        thumbs.append(tile)
    width = max(facade.width, sum(t.width for t in thumbs) + gap * max(0, len(thumbs) - 1), 320)
    height = facade.height + 36 + (thumbs[0].height if thumbs else 0)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    sheet.paste(facade, ((width - facade.width) // 2, 0))
    banner = f"floor: {shape}" if shape in SHAPES else "floor: not set"
    ImageDraw.Draw(sheet).text((12, facade.height + 6), banner, fill=(255, 220, 80), font=_font(20))
    x = 0
    y = facade.height + 36
    for thumb in thumbs:
        sheet.paste(thumb, (x, y))
        x += thumb.width + gap
    return sheet


class EnclosedFloorApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        photo_dir: Path,
        dsl_dir: Path,
        cell_labels: Path,
        crops_dir: Path,
        labels_path: Path,
        review_all: bool = False,
    ) -> None:
        self.root = root
        self.photo_dir = Path(photo_dir)
        self.dsl_dir = Path(dsl_dir)
        self.crops_dir = Path(crops_dir)
        self.labels_path = Path(labels_path)
        self.cell_labels = load_cell_labels(cell_labels)
        self.records = load_crop_labels(self.labels_path) if self.labels_path.is_file() else {}
        self.queue = enclosed_queue(self.cell_labels, self.photo_dir, self.dsl_dir)
        if not review_all:
            self.queue = [
                stem
                for stem in self.queue
                if photo_shape(self.records, stem, self.cell_labels[stem]) is None
            ]
        self.index = 0
        self.shape = ""
        self._photo: Image.Image | None = None
        self._floor_y: dict[int, tuple[int, int]] = {}
        self._bay_x: dict[int, tuple[int, int]] = {}
        self._cells: list[Cell] = []
        self._tk: ImageTk.PhotoImage | None = None
        self._resize_after: str | None = None
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self._build()
        self._bind_keys()
        self._show_current()

    def _build(self) -> None:
        self.root.title("Enclosed floor label UI")
        self.root.geometry("1200x980")
        self.root.minsize(860, 680)
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.path_var, font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10)).pack(anchor=tk.W)
        ttk.Label(
            top,
            text="One plan for every enclosed cell of this photo. R rectangle, C circle, Enter saves.",
        ).pack(anchor=tk.W)
        self.canvas = tk.Canvas(self.root, bg="#121212", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>", self._on_configure)
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Rectangle (R)", command=lambda: self._choose("rectangle")).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Circle (C)", command=lambda: self._choose("circle")).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Save (Enter)", command=self.save).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Skip (S)", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Prev", command=self.prev).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Next", command=self.next_photo).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Finish", command=self.finish).pack(side=tk.RIGHT, padx=4)

    def _bind_keys(self) -> None:
        self.root.bind("<Key-r>", lambda _e: self._choose("rectangle"))
        self.root.bind("<Key-c>", lambda _e: self._choose("circle"))
        self.root.bind("<Return>", lambda _e: self.save())
        self.root.bind("<Key-s>", lambda _e: self.skip())
        self.root.bind("<Left>", lambda _e: self.prev())
        self.root.bind("<Right>", lambda _e: self.next_photo())
        self.root.bind("<Key-q>", lambda _e: self.finish())

    def _on_configure(self, _event: tk.Event) -> None:
        if self._resize_after is not None:
            self.root.after_cancel(self._resize_after)
        self._resize_after = self.root.after(80, self._render)

    def _show_current(self) -> None:
        if not self.queue:
            self.path_var.set("No enclosed photos left to label.")
            self.status_var.set(str(self.labels_path))
            self.canvas.delete("all")
            self._photo = None
            return
        self.index = max(0, min(self.index, len(self.queue) - 1))
        stem = self.queue[self.index]
        photo_path = find_photo(self.photo_dir, stem)
        dsl = self.dsl_dir / stem / "facade_dsl.json"
        if photo_path is None or not dsl.is_file():
            self.queue.pop(self.index)
            self._show_current()
            return
        try:
            self._photo = Image.open(photo_path).convert("RGB")
            self._floor_y, self._bay_x = load_grid(dsl)
        except OSError as exc:
            messagebox.showerror("Image error", f"{photo_path}\n{exc}")
            self.skip()
            return
        self._cells = list(self.cell_labels.get(stem) or [])
        saved = photo_shape(self.records, stem, self._cells)
        self.shape = saved or ""
        n_done = sum(
            1
            for s in self.queue
            if photo_shape(self.records, s, self.cell_labels[s]) is not None
        )
        self.path_var.set(
            f"[{self.index + 1}/{len(self.queue)}]  {stem}   "
            f"cells={len(self._cells)}   labeled {n_done}/{len(self.queue)}"
        )
        self.status_var.set(f"current: {self.shape or 'not set'}")
        self._render()

    def _render(self) -> None:
        self._resize_after = None
        if self._photo is None:
            return
        sheet = render_sheet(self._photo, self._floor_y, self._bay_x, self._cells, self.shape)
        cw = max(64, int(self.canvas.winfo_width()))
        ch = max(64, int(self.canvas.winfo_height()))
        scale = min(cw / sheet.width, ch / sheet.height)
        dw = max(1, int(round(sheet.width * scale)))
        dh = max(1, int(round(sheet.height * scale)))
        shown = sheet.resize((dw, dh), Image.Resampling.LANCZOS)
        self._tk = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image((cw - dw) // 2, (ch - dh) // 2, anchor=tk.NW, image=self._tk)

    def _choose(self, shape: str) -> None:
        self.shape = shape
        self.status_var.set(f"current: {shape}")
        self._render()

    def save(self) -> None:
        if not self.queue or self._photo is None:
            return
        if self.shape not in SHAPES:
            messagebox.showwarning("Floor shape", "Choose rectangle (R) or circle (C).")
            return
        stem = self.queue[self.index]
        crops = extract_cells(self._photo, self._floor_y, self._bay_x, self._cells)
        if not crops:
            messagebox.showwarning("No cells", f"{stem} has no cells on the window grid.")
            return
        folder = self.crops_dir / "unlabeled"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for cell, crop in crops:
                name = crop_stem(stem, cell)
                crop.save(folder / f"{name}.png")
            assign_photo_shape(self.records, stem, [cell for cell, _crop in crops], self.shape)
            write_labels_jsonl(self.labels_path, self.records)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1
        self._show_current()

    def skip(self) -> None:
        if not self.queue:
            return
        self.index = (self.index + 1) % len(self.queue)
        self._show_current()

    def prev(self) -> None:
        if not self.queue:
            return
        self.index = (self.index - 1) % len(self.queue)
        self._show_current()

    def next_photo(self) -> None:
        self.skip()

    def finish(self) -> None:
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_PHOTO_DIR, help="facade photos (data/base jpgs)")
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--cell-labels", type=Path, default=DEFAULT_CELL_LABELS)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--review-all", action="store_true", help="include photos that already have a floor plan")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    EnclosedFloorApp(
        root,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        cell_labels=args.cell_labels,
        crops_dir=args.crops_dir,
        labels_path=args.labels,
        review_all=args.review_all,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
