"""Draw, edit and delete open-balcony ground-truth boxes on the test100 photos.

Each photo opens with its prefill boxes (CMP label boxes for cmp_b / cmp_x,
pipeline boxes for LSAA; see ``gt_box_store``). What you leave on screen is
saved as that photo's ground truth when you move to another photo, press S, or
close the window. Saved photos reopen with the saved boxes.

Mouse:
  left-drag on empty area   draw a new box
  left-click a box          select it (yellow); drag inside to move it
  drag an edge / corner     resize the selected box
  right-click a box         delete it
  wheel                     zoom at the cursor
  Shift + left-drag, middle-drag   pan

Keys:
  Delete / BackSpace   delete the selected box
  Ctrl+Z               undo (this photo)
  N / Right / Enter    save, next photo        P / Left   save, previous photo
  J                    save, next photo not yet saved
  S                    save          R   reset to prefill (undoable)
  F                    fit to window H   hide / show boxes
  O                    show / hide the prefill as grey dashes
  Esc                  deselect
  I                    switch what a new box is: balcony (red) / ignore region (grey)
  T                    switch the selected box between balcony and ignore

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/gt_box_ui.py
    ... balcony_train/gt_box_ui.py --only cmp --start cmp_b0171
"""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.gt_box_store import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_OUT,
    DEFAULT_PHOTOS,
    clean_box,
    is_cmp,
    load_gt,
    make_row,
    prefill_boxes,
    save_gt,
)

HANDLE_PX = 7
MIN_BOX_PX = 4


class GtBoxApp:
    def __init__(self, root: tk.Tk, photos: list[Path], out: Path, cache: dict, start: int) -> None:
        self.root = root
        self.photos = photos
        self.out = out
        self.cache = cache
        self.rows = load_gt(out)
        self.idx = start

        self.canvas = tk.Canvas(root, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.status = tk.Label(root, anchor="w", font=("Segoe UI", 11), bg="#f0f0f0")
        self.toolbar = tk.Frame(root, bg="#e4e4e4", pady=3)
        self.draw_kind = "balcony"
        self.mode_btn = tk.Button(self.toolbar, command=lambda: self.toggle_draw_kind(), takefocus=0,
                                  font=("Segoe UI", 10, "bold"), padx=8, width=24)
        self.mode_btn.pack(side="left", padx=3)
        tk.Button(self.toolbar, text="Switch selected type  (T)", command=lambda: self.toggle_selected_kind(),
                  takefocus=0, font=("Segoe UI", 10), padx=8).pack(side="left", padx=3)
        for text, fn in (
            ("◀ Previous  (P / ←)", lambda: self.go(self.idx - 1)),
            ("Next ▶  (N / → / Enter)", lambda: self.go(self.idx + 1)),
            ("Next unsaved  (J)", lambda: self.next_unsaved()),
            ("Save  (S)", lambda: self.save()),
            ("Undo  (Ctrl+Z)", lambda: self.undo()),
            ("Delete box  (Del)", lambda: self.delete_selected()),
            ("Reset to prefill  (R)", lambda: self.reset()),
            ("Fit  (F)", lambda: self.fit()),
            ("Hide boxes  (H)", lambda: self.toggle_hide()),
            ("Show prefill  (O)", lambda: self.toggle_prefill()),
        ):
            tk.Button(self.toolbar, text=text, command=fn, takefocus=0, font=("Segoe UI", 10),
                      padx=8).pack(side="left", padx=3)
        self.help = tk.Label(
            root, anchor="w", font=("Segoe UI", 9), fg="#444", bg="#f0f0f0",
            text="Mouse: drag on empty area = new box | click = select, drag = move | drag edge/corner = resize | "
                 "right-click = delete | wheel = zoom | Shift+drag or middle-drag = pan | Esc = deselect",
        )
        self.status.pack(side="top", fill="x")
        self.toolbar.pack(side="top", fill="x")
        self.help.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)

        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<ButtonPress-3>", self.on_right)
        c.bind("<ButtonPress-2>", self.on_pan_start)
        c.bind("<B2-Motion>", self.on_pan_move)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Configure>", self.on_configure)
        for keys, fn in (
            (("<Delete>", "<BackSpace>"), self.delete_selected),
            (("<Control-z>", "<Control-Z>"), self.undo),
            (("n", "N", "<Right>", "<Return>"), lambda: self.go(self.idx + 1)),
            (("p", "P", "<Left>"), lambda: self.go(self.idx - 1)),
            (("j", "J"), self.next_unsaved),
            (("s", "S"), self.save),
            (("r", "R"), self.reset),
            (("f", "F"), self.fit),
            (("h", "H"), self.toggle_hide),
            (("o", "O"), self.toggle_prefill),
            (("<Escape>",), self.deselect),
            (("i", "I"), self.toggle_draw_kind),
            (("t", "T"), self.toggle_selected_kind),
        ):
            for k in keys:
                root.bind(k, lambda e, fn=fn: fn())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.show_prefill = False
        self.hidden = False
        self.load(self.idx)

    # ---------- photo state ----------
    def load(self, idx: int) -> None:
        self.idx = idx
        path = self.photos[idx]
        self.stem = path.stem
        self.img = Image.open(path).convert("RGB")
        self.size = self.img.size
        self.prefill_name, self.prefill = prefill_boxes(self.stem, self.size, self.cache)
        row = self.rows.get(self.stem)
        if row:
            ign = row.get("ignore", [])
            self.boxes = [list(b) for b in row["boxes"]] + [list(b) for b in ign]
            self.kinds = ["balcony"] * len(row["boxes"]) + ["ignore"] * len(ign)
        else:
            self.boxes = [list(b) for b in self.prefill]
            self.kinds = ["balcony"] * len(self.boxes)
        self.saved_state = self.state() if row else None
        self.history: list[tuple] = []
        self.selected: int | None = None
        self.drag: dict | None = None
        self.auto_fit = True
        self.fit()

    def state(self) -> tuple:
        return ([list(b) for b in self.boxes], list(self.kinds))

    def dirty(self) -> bool:
        return self.saved_state is None or self.saved_state != self.state()

    def save(self) -> None:
        balc = [b for b, k in zip(self.boxes, self.kinds) if k == "balcony"]
        ign = [b for b, k in zip(self.boxes, self.kinds) if k == "ignore"]
        self.rows[self.stem] = make_row(
            self.stem, self.photos[self.idx].name, self.size, balc, self.prefill_name, self.prefill, ignore=ign
        )
        save_gt(self.out, self.rows)
        self.saved_state = self.state()
        self.redraw()

    def go(self, idx: int) -> None:
        if not 0 <= idx < len(self.photos):
            self.root.bell()
            return
        self.save()
        self.load(idx)

    def next_unsaved(self) -> None:
        self.save()
        n = len(self.photos)
        for k in range(1, n + 1):
            j = (self.idx + k) % n
            if self.photos[j].stem not in self.rows:
                self.load(j)
                return
        self.root.bell()

    def on_close(self) -> None:
        self.save()
        self.root.destroy()

    def push(self) -> None:
        self.history.append(self.state())
        if len(self.history) > 200:
            self.history.pop(0)

    def undo(self) -> None:
        if self.history:
            self.boxes, self.kinds = self.history.pop()
            self.selected = None
            self.redraw()

    def reset(self) -> None:
        self.push()
        self.boxes = [list(b) for b in self.prefill]
        self.kinds = ["balcony"] * len(self.boxes)
        self.selected = None
        self.redraw()

    def remove(self, i: int) -> None:
        del self.boxes[i]
        del self.kinds[i]

    def delete_selected(self) -> None:
        if self.selected is not None:
            self.push()
            self.remove(self.selected)
            self.selected = None
            self.redraw()

    def toggle_draw_kind(self) -> None:
        self.draw_kind = "ignore" if self.draw_kind == "balcony" else "balcony"
        self.redraw()

    def toggle_selected_kind(self) -> None:
        if self.selected is None:
            self.root.bell()
            return
        self.push()
        i = self.selected
        self.kinds[i] = "ignore" if self.kinds[i] == "balcony" else "balcony"
        self.redraw()

    def deselect(self) -> None:
        self.selected = None
        self.redraw()

    def toggle_hide(self) -> None:
        self.hidden = not self.hidden
        self.redraw()

    def toggle_prefill(self) -> None:
        self.show_prefill = not self.show_prefill
        self.redraw()

    # ---------- view ----------
    def on_configure(self, e) -> None:
        # Refit while the user has not zoomed or panned (window resize / maximize).
        if self.auto_fit:
            self.fit()
        else:
            self.redraw()

    def fit(self) -> None:
        self.auto_fit = True
        self.root.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 100 or ch < 100:  # not laid out yet; <Configure> will refit
            cw, ch = 1400, 880
        w, h = self.size
        self.scale = min(cw / w, ch / h) * 0.98
        self.ox = (cw - w * self.scale) / 2
        self.oy = (ch - h * self.scale) / 2
        self.redraw()

    def to_img(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) / self.scale, (y - self.oy) / self.scale

    def to_scr(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.ox, y * self.scale + self.oy

    def on_wheel(self, e) -> None:
        f = 1.25 if e.delta > 0 else 0.8
        self.auto_fit = False
        new = min(max(self.scale * f, 0.05), 20.0)
        ix, iy = self.to_img(e.x, e.y)
        self.scale = new
        self.ox = e.x - ix * new
        self.oy = e.y - iy * new
        self.redraw()

    def on_pan_start(self, e) -> None:
        self.auto_fit = False
        self.pan = (e.x, e.y, self.ox, self.oy)

    def on_pan_move(self, e) -> None:
        x0, y0, ox, oy = self.pan
        self.ox, self.oy = ox + e.x - x0, oy + e.y - y0
        self.redraw()

    # ---------- hit tests ----------
    def box_at(self, ix: float, iy: float) -> int | None:
        hits = [i for i, b in enumerate(self.boxes) if b[0] <= ix <= b[2] and b[1] <= iy <= b[3]]
        if not hits:
            return None
        return min(hits, key=lambda i: (self.boxes[i][2] - self.boxes[i][0]) * (self.boxes[i][3] - self.boxes[i][1]))

    def edges_at(self, i: int, sx: float, sy: float) -> set[str]:
        x0, y0 = self.to_scr(*self.boxes[i][:2])
        x1, y1 = self.to_scr(*self.boxes[i][2:])
        t = HANDLE_PX
        if not (x0 - t <= sx <= x1 + t and y0 - t <= sy <= y1 + t):
            return set()
        edges = set()
        if abs(sx - x0) <= t:
            edges.add("l")
        elif abs(sx - x1) <= t:
            edges.add("r")
        if abs(sy - y0) <= t:
            edges.add("t")
        elif abs(sy - y1) <= t:
            edges.add("b")
        return edges

    # ---------- mouse ----------
    def on_motion(self, e) -> None:
        cur = "crosshair"
        if self.selected is not None:
            ed = self.edges_at(self.selected, e.x, e.y)
            if ed in ({"l", "t"}, {"r", "b"}):
                cur = "size_nw_se"
            elif ed in ({"r", "t"}, {"l", "b"}):
                cur = "size_ne_sw"
            elif ed & {"l", "r"}:
                cur = "sb_h_double_arrow"
            elif ed & {"t", "b"}:
                cur = "sb_v_double_arrow"
            elif self.box_at(*self.to_img(e.x, e.y)) == self.selected:
                cur = "fleur"
        self.canvas.configure(cursor=cur)

    def on_press(self, e) -> None:
        if e.state & 0x0001:  # Shift: pan
            self.drag = {"mode": "pan"}
            self.on_pan_start(e)
            return
        ix, iy = self.to_img(e.x, e.y)
        if self.selected is not None:
            ed = self.edges_at(self.selected, e.x, e.y)
            if ed:
                self.push()
                self.drag = {"mode": "resize", "edges": ed}
                return
        hit = self.box_at(ix, iy)
        if hit is not None:
            self.selected = hit
            self.push()
            self.drag = {"mode": "move", "start": (ix, iy), "orig": list(self.boxes[hit]), "moved": False}
            self.redraw()
            return
        self.selected = None
        self.drag = {"mode": "new", "start": (ix, iy), "cur": (ix, iy)}
        self.redraw()

    def on_drag(self, e) -> None:
        d = self.drag
        if not d:
            return
        if d["mode"] == "pan":
            self.on_pan_move(e)
            return
        ix, iy = self.to_img(e.x, e.y)
        w, h = self.size
        ix, iy = min(max(ix, 0), w), min(max(iy, 0), h)
        if d["mode"] == "new":
            d["cur"] = (ix, iy)
        elif d["mode"] == "move":
            o = d["orig"]
            dx = min(max(ix - d["start"][0], -o[0]), w - o[2])
            dy = min(max(iy - d["start"][1], -o[1]), h - o[3])
            self.boxes[self.selected] = [round(o[0] + dx), round(o[1] + dy), round(o[2] + dx), round(o[3] + dy)]
            d["moved"] = True
        elif d["mode"] == "resize":
            b = self.boxes[self.selected]
            if "l" in d["edges"]:
                b[0] = round(ix)
            if "r" in d["edges"]:
                b[2] = round(ix)
            if "t" in d["edges"]:
                b[1] = round(iy)
            if "b" in d["edges"]:
                b[3] = round(iy)
        self.redraw()

    def on_release(self, e) -> None:
        d, self.drag = self.drag, None
        if not d or d["mode"] == "pan":
            return
        if d["mode"] == "new":
            (x0, y0), (x1, y1) = d["start"], d["cur"]
            if abs(x1 - x0) * self.scale >= MIN_BOX_PX and abs(y1 - y0) * self.scale >= MIN_BOX_PX:
                box = clean_box([x0, y0, x1, y1], *self.size)
                if box:
                    self.push()
                    self.boxes.append(box)
                    self.kinds.append(self.draw_kind)
                    self.selected = len(self.boxes) - 1
        elif d["mode"] == "move" and not d["moved"]:
            self.history.pop()  # a click only selected the box
        elif d["mode"] == "resize":
            box = clean_box(self.boxes[self.selected], *self.size)
            if box is None:
                self.remove(self.selected)
                self.selected = None
            else:
                self.boxes[self.selected] = box
        self.redraw()

    def on_right(self, e) -> None:
        hit = self.box_at(*self.to_img(e.x, e.y))
        if hit is not None:
            self.push()
            self.remove(hit)
            self.selected = None
            self.redraw()

    # ---------- drawing ----------
    def redraw(self) -> None:
        if not hasattr(self, "scale"):
            return
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        w, h = self.size
        ix0, iy0 = self.to_img(0, 0)
        ix1, iy1 = self.to_img(cw, ch)
        ix0, iy0 = max(0, int(ix0)), max(0, int(iy0))
        ix1, iy1 = min(w, int(ix1) + 1), min(h, int(iy1) + 1)
        if ix1 > ix0 and iy1 > iy0:
            crop = self.img.crop((ix0, iy0, ix1, iy1))
            dw = max(1, round((ix1 - ix0) * self.scale))
            dh = max(1, round((iy1 - iy0) * self.scale))
            resample = Image.NEAREST if self.scale > 3 else Image.BILINEAR
            self.tk_img = ImageTk.PhotoImage(crop.resize((dw, dh), resample))
            sx, sy = self.to_scr(ix0, iy0)
            c.create_image(sx, sy, image=self.tk_img, anchor="nw")

        if self.show_prefill:
            for b in self.prefill:
                c.create_rectangle(*self.to_scr(*b[:2]), *self.to_scr(*b[2:]), outline="#bbbbbb", dash=(4, 3), width=1)
        if not self.hidden:
            for i, b in enumerate(self.boxes):
                x0, y0 = self.to_scr(*b[:2])
                x1, y1 = self.to_scr(*b[2:])
                if self.kinds[i] == "ignore":
                    c.create_rectangle(x0, y0, x1, y1, outline="#c8c8c8", width=2, dash=(6, 3),
                                       fill="#808080", stipple="gray25")
                    c.create_text(x0 + 3, y0 + 2, text="IGNORE", anchor="nw", fill="#ffffff",
                                  font=("Segoe UI", 8, "bold"))
                elif i != self.selected:
                    c.create_rectangle(x0, y0, x1, y1, outline="#ff2020", width=2)
            if self.selected is not None and self.selected < len(self.boxes):
                b = self.boxes[self.selected]
                x0, y0 = self.to_scr(*b[:2])
                x1, y1 = self.to_scr(*b[2:])
                c.create_rectangle(x0, y0, x1, y1, outline="#ffe000", width=2,
                                   dash=(6, 3) if self.kinds[self.selected] == "ignore" else None)
                for hx in (x0, (x0 + x1) / 2, x1):
                    for hy in (y0, (y0 + y1) / 2, y1):
                        if hx == (x0 + x1) / 2 and hy == (y0 + y1) / 2:
                            continue
                        c.create_rectangle(hx - 3, hy - 3, hx + 3, hy + 3, fill="#ffe000", outline="black")
        d = self.drag
        if d and d["mode"] == "new":
            color = "#c8c8c8" if self.draw_kind == "ignore" else "#00e0ff"
            c.create_rectangle(*self.to_scr(*d["start"]), *self.to_scr(*d["cur"]), outline=color, width=2, dash=(5, 3))

        if self.draw_kind == "ignore":
            self.mode_btn.configure(text="Drawing: IGNORE  (I)", bg="#9a9a9a", fg="white", activebackground="#9a9a9a")
        else:
            self.mode_btn.configure(text="Drawing: BALCONY  (I)", bg="#d03030", fg="white", activebackground="#d03030")
        done = sum(1 for p in self.photos if p.stem in self.rows)
        src = "CMP label" if self.prefill_name == "cmp_label" else "pipeline"
        state = "unsaved changes" if self.dirty() else "saved"
        if self.stem not in self.rows:
            state = "not saved yet"
        n_ign = self.kinds.count("ignore")
        self.status.configure(
            text=f"  {self.idx + 1}/{len(self.photos)}   {self.stem}   |   balconies {len(self.boxes) - n_ign}  "
                 f"ignore {n_ign} "
                 f"(prefill {src}: {len(self.prefill)})   |   {state}   |   saved photos {done}/{len(self.photos)}"
                 f"{'   |   BOXES HIDDEN' if self.hidden else ''}",
            fg="#b00000" if self.dirty() else "#006000",
        )
        self.root.title(f"Balcony GT boxes - {self.stem}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="SAM3 stage box cache (LSAA prefill)")
    ap.add_argument("--only", choices=("all", "cmp", "lsaa"), default="all")
    ap.add_argument("--start", default=None, help="stem to open first (default: first photo not yet saved)")
    args = ap.parse_args()

    photos = sorted(args.photos_dir.glob("*.jpg"))
    if args.only == "cmp":
        photos = [p for p in photos if is_cmp(p.stem)]
    elif args.only == "lsaa":
        photos = [p for p in photos if not is_cmp(p.stem)]
    if not photos:
        sys.exit(f"no photos in {args.photos_dir}")
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.is_file() else {}

    stems = [p.stem for p in photos]
    if args.start:
        if args.start not in stems:
            sys.exit(f"--start {args.start} not in the photo list")
        start = stems.index(args.start)
    else:
        saved = load_gt(args.out)
        start = next((i for i, s in enumerate(stems) if s not in saved), 0)

    root = tk.Tk()
    root.geometry("1400x950")
    try:
        root.state("zoomed")  # Windows: start maximized
    except tk.TclError:
        pass
    GtBoxApp(root, photos, args.out, cache, start)
    root.mainloop()


if __name__ == "__main__":
    main()
