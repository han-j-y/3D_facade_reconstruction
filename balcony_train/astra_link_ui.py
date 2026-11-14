"""Link Astra balcony crops to the test100 ground-truth crops by hand, then score.

Left: the test100 photo with its hand-labelled balcony crops (``runs/test100/crops``
and ``runs/test100/add5_crops``; the hand GT boxes of ``balcony_gt_boxes.jsonl``
are drawn thin grey for reference). Right: Astra's front view
(``test_Astra Result/Front View``) with the labelled Astra crops
(``test_Astra Result/Cropped Balcony``). ``Output/index.csv`` gives the photo of
each ``facade_NN``.

Mouse:
* left click a box: select it. One GT box + one Astra box selected = one link.
  A box may take part in several links (Astra merging two balconies, etc.).
* right click a box: remove every link of that box.

Keys: Right / Left or N / P next / previous facade, D mark the facade reviewed,
Delete remove the link selected in the list, Escape clear the selection.

Links are saved on every change to ``--links``
(default ``test_Astra Result/Cropped Balcony/gt_links.jsonl``), one row per facade.

Score (button, or ``--score`` without the UI): GT boxes and Astra boxes left
unlinked on reviewed facades count as missed / extra. Attributes on the linked
pairs, same rule as the test100 classifier score: kind and structure on every
pair; material only where both are open_work; floor only where both are
projecting.

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/astra_link_ui.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/astra_link_ui.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tkinter as tk
from collections import Counter, defaultdict
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
ASTRA = ROOT / "test_Astra Result"
GT_CROP_DIRS = [ROOT / "runs/test100/crops", ROOT / "runs/test100/add5_crops"]
GT_BOXES = ROOT / "runs/test100/balcony_gt_boxes.jsonl"
PHOTOS = ROOT / "data/test100/photos"

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#469990", "#9a6324", "#800000", "#808000", "#000075", "#ffe119"]
UNLINKED = "#ffffff"
SELECTED = "#ffff00"


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def facade_id(index: str) -> str:
    i = int(index)
    return f"facade_{i:02d}" if i <= 50 else f"facade_{i:03d}"


def short(lab: dict) -> str:
    parts = [lab.get("kind"), lab.get("material"), lab.get("structure"), lab.get("floor_shape")]
    return " / ".join(p for p in parts if p)


def load_crops(dirs: list[Path]) -> dict[str, list[dict]]:
    """photo stem -> labelled crops [{id, box, label}] (not_balcony and unlabelled left out)."""
    out: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for d in dirs:
        labels = {r["path"]: r for r in read_jsonl(d / "labels.jsonl")}
        bad = {r["path"] for r in read_jsonl(d / "not_balcony.jsonl")}
        for m in read_jsonl(d / "manifest.jsonl"):
            c = m["crop"]
            if c in seen or c in bad or c not in labels:
                continue
            seen.add(c)
            out[Path(m["photo"]).stem].append(
                {"id": c, "box": m["box_xyxy"], "size": m.get("photo_size"), "label": labels[c]})
    return out


class Data:
    def __init__(self, links_path: Path):
        rows = list(csv.DictReader(open(ASTRA / "Output/index.csv", encoding="utf-8-sig")))
        self.facades = [{"fid": facade_id(r["index"]), "stem": r["stem"], "tid": r.get("test_id", "")}
                        for r in rows]
        self.gt = load_crops(GT_CROP_DIRS)
        self.astra = load_crops([ASTRA / "Cropped Balcony"])
        self.gt_boxes = {r["stem"]: r for r in read_jsonl(GT_BOXES)}
        self.links_path = links_path
        self.state = {r["facade"]: r for r in read_jsonl(links_path)}

    def entry(self, fid: str, stem: str) -> dict:
        return self.state.setdefault(fid, {"facade": fid, "stem": stem, "links": [], "done": False})

    def save(self):
        tmp = self.links_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for fac in self.facades:
                r = self.state.get(fac["fid"])
                if r and (r["links"] or r["done"]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.links_path)


def score(data: Data) -> str:
    n_gt = n_a = hit_gt = hit_a = 0
    heads = {h: Counter() for h in ("kind", "structure", "material", "floor_shape")}
    conf = {h: Counter() for h in heads}
    n_done = 0
    for fac in data.facades:
        r = data.state.get(fac["fid"])
        if not r:
            continue
        gt = {c["id"]: c["label"] for c in data.gt.get(fac["stem"], [])}
        ast = {c["id"]: c["label"] for c in data.astra.get(fac["fid"], [])}
        links = [(g, a) for g, a in r["links"] if g in gt and a in ast]
        if r["done"]:
            n_done += 1
            n_gt += len(gt)
            n_a += len(ast)
            hit_gt += len({g for g, _ in links})
            hit_a += len({a for _, a in links})
        for g, a in links:
            t, p = gt[g], ast[a]
            for h in heads:
                if h == "material" and not (t["kind"] == p["kind"] == "open_work"):
                    continue
                if h == "floor_shape" and not (t["structure"] == p["structure"] == "projecting"):
                    continue
                heads[h]["n"] += 1
                heads[h]["ok"] += t[h] == p[h]
                conf[h][(t[h], p[h])] += 1
    lines = [f"Reviewed facades: {n_done} / {len(data.facades)}"]
    if n_done:
        lines.append(f"Detection (reviewed): GT {n_gt}, Astra {n_a}, "
                     f"recall {hit_gt / max(n_gt, 1):.3f} ({hit_gt}), precision {hit_a / max(n_a, 1):.3f} ({hit_a})")
    for h, c in heads.items():
        if c["n"]:
            lines.append(f"{h:12s} acc {c['ok'] / c['n']:.3f}  ({c['ok']} / {c['n']})")
            lines.append("    " + ", ".join(f"{t}->{p}: {k}" for (t, p), k in sorted(conf[h].items(), key=str)))
    return "\n".join(lines)


class App:
    def __init__(self, root: tk.Tk, data: Data, start: int):
        self.root, self.data = root, data
        self.i = start
        self.sel = {"gt": None, "astra": None}
        self.colors: dict[tuple[str, str], str] = {}

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.cw, self.ch = (sw - 60) // 2, max(sh - 380, 250)
        root.geometry(f"{sw - 20}x{sh - 80}+0+0")
        self._resize_job = None

        top = ttk.Frame(root)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Button(top, text="◀ Prev", command=lambda: self.go(-1)).pack(side="left")
        ttk.Button(top, text="Next ▶", command=lambda: self.go(1)).pack(side="left")
        self.combo = ttk.Combobox(top, width=40, state="readonly")
        self.combo.pack(side="left", padx=6)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.jump(self.combo.current()))
        self.done_var = tk.BooleanVar()
        ttk.Checkbutton(top, text="Reviewed (D)", variable=self.done_var, command=self.toggle_done).pack(side="left")
        ttk.Button(top, text="Score", command=self.show_score).pack(side="left", padx=6)
        self.title = ttk.Label(top, text="")
        self.title.pack(side="left", padx=10)

        bot = ttk.Frame(root)
        bot.pack(side="bottom", fill="x", padx=6, pady=4)

        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True)
        self.canvas = {}
        for side, txt in (("gt", "test100 photo + GT crops"), ("astra", "Astra front view + Astra crops")):
            f = ttk.Frame(mid)
            f.pack(side="left", padx=4, fill="both", expand=True)
            ttk.Label(f, text=txt).pack()
            c = tk.Canvas(f, width=self.cw, height=self.ch, bg="#333", highlightthickness=0)
            c.pack(fill="both", expand=True)
            c.bind("<Configure>", self.on_resize)
            c.bind("<Button-1>", lambda e, s=side: self.click(s, e))
            c.bind("<Button-3>", lambda e, s=side: self.unlink_box(s, e))
            self.canvas[side] = c

        self.info = {}
        self.thumb = {}
        for side in ("gt", "list", "astra"):
            f = ttk.Frame(bot)
            f.pack(side="left", fill="both", expand=True)
            if side == "list":
                self.listbox = tk.Listbox(f, height=8, font=("Consolas", 9))
                self.listbox.pack(fill="both", expand=True)
                ttk.Button(f, text="Delete link (Del)", command=self.delete_link).pack()
            else:
                self.thumb[side] = ttk.Label(f)
                self.thumb[side].pack(side="left")
                self.info[side] = ttk.Label(f, text="", justify="left", font=("Consolas", 10))
                self.info[side].pack(side="left", padx=6)

        for k in ("<Right>", "n"):
            root.bind(k, lambda e: self.go(1))
        for k in ("<Left>", "p"):
            root.bind(k, lambda e: self.go(-1))
        root.bind("d", lambda e: (self.done_var.set(not self.done_var.get()), self.toggle_done()))
        root.bind("<Delete>", lambda e: self.delete_link())
        root.bind("<Escape>", lambda e: self.clear_sel())
        self.load()

    # ---- facade ----
    @property
    def fac(self):
        return self.data.facades[self.i]

    @property
    def entry(self):
        return self.data.entry(self.fac["fid"], self.fac["stem"])

    def go(self, d):
        self.jump((self.i + d) % len(self.data.facades))

    def jump(self, i):
        self.i = i
        self.clear_sel(redraw=False)
        self.load()

    def refresh_combo(self):
        vals = []
        for f in self.data.facades:
            r = self.data.state.get(f["fid"], {})
            mark = "✓" if r.get("done") else ("…" if r.get("links") else " ")
            vals.append(f"{mark} {f['fid']}  {f['tid'] or '----'}  GT {len(self.data.gt.get(f['stem'], []))}"
                        f" / A {len(self.data.astra.get(f['fid'], []))}")
        self.combo["values"] = vals
        self.combo.current(self.i)

    def load(self):
        fac = self.fac
        self.items = {"gt": self.data.gt.get(fac["stem"], []), "astra": self.data.astra.get(fac["fid"], [])}
        photo = next(PHOTOS.glob(fac["stem"] + ".*"))
        self.img = {"gt": Image.open(photo).convert("RGB"),
                    "astra": Image.open(ASTRA / "Front View" / f"{fac['fid']}.png").convert("RGB")}
        self.rescale()
        self.done_var.set(self.entry["done"])
        self.title.config(text=f"{fac['fid']}  {fac['tid']}  {fac['stem']}")
        self.refresh_combo()
        self.redraw()

    def rescale(self):
        self.scale, self.tk_img, self.off = {}, {}, {}
        for s, im in self.img.items():
            c = self.canvas[s]
            cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
            if cw <= 50:  # not mapped yet
                cw, ch = self.cw, self.ch
            k = min(cw / im.width, ch / im.height)
            w, h = max(1, int(im.width * k)), max(1, int(im.height * k))
            self.scale[s] = k
            self.off[s] = ((cw - w) // 2, (ch - h) // 2)
            self.tk_img[s] = ImageTk.PhotoImage(im.resize((w, h)))

    def on_resize(self, _e):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self._do_resize)

    def _do_resize(self):
        self._resize_job = None
        if hasattr(self, "img"):
            self.rescale()
            self.redraw()

    def toggle_done(self):
        self.entry["done"] = self.done_var.get()
        self.data.save()
        self.refresh_combo()

    # ---- drawing ----
    def box_scaled(self, side, item):
        b = item["box"]
        im = self.img[side]
        sx = sy = 1.0
        if item.get("size"):  # manifest coords are in photo_size pixels
            sx, sy = im.width / item["size"][0], im.height / item["size"][1]
        k = self.scale[side]
        ox, oy = self.off[side]
        return b[0] * sx * k + ox, b[1] * sy * k + oy, b[2] * sx * k + ox, b[3] * sy * k + oy

    def link_color(self, link):
        key = tuple(link)
        if key not in self.colors:
            self.colors[key] = PALETTE[len(self.colors) % len(PALETTE)]
        return self.colors[key]

    def redraw(self):
        links = self.entry["links"]
        self.colors = {}
        box_color = {"gt": {}, "astra": {}}
        for l in links:
            col = self.link_color(l)
            box_color["gt"].setdefault(l[0], col)
            box_color["astra"].setdefault(l[1], col)
        for side, c in self.canvas.items():
            c.delete("all")
            c.create_image(*self.off[side], anchor="nw", image=self.tk_img[side])
            if side == "gt" and self.fac["stem"] in self.data.gt_boxes:
                r = self.data.gt_boxes[self.fac["stem"]]
                im = self.img["gt"]
                sx, sy = im.width / r["width"], im.height / r["height"]
                k = self.scale["gt"]
                ox, oy = self.off["gt"]
                for b in r["boxes"]:
                    c.create_rectangle(b[0] * sx * k + ox, b[1] * sy * k + oy, b[2] * sx * k + ox,
                                       b[3] * sy * k + oy, outline="#888888", dash=(2, 2))
            for n, it in enumerate(self.items[side]):
                x0, y0, x1, y1 = self.box_scaled(side, it)
                col = box_color[side].get(it["id"], UNLINKED)
                w = 2
                if self.sel[side] == it["id"]:
                    col, w = SELECTED, 4
                c.create_rectangle(x0, y0, x1, y1, outline="black", width=w + 2)
                c.create_rectangle(x0, y0, x1, y1, outline=col, width=w,
                                   dash=() if it["id"] in box_color[side] or w == 4 else (4, 3))
                tag = ("G" if side == "gt" else "A") + str(n)
                t = c.create_text(x0 + 2, y0 + 1, anchor="nw", text=tag, fill=col, font=("Arial", 9, "bold"))
                c.create_rectangle(c.bbox(t), fill="black", outline="")
                c.tag_raise(t)
        self.listbox.delete(0, "end")
        idx = {s: {it["id"]: n for n, it in enumerate(self.items[s])} for s in self.items}
        for n, (g, a) in enumerate(links):
            gl = next((it["label"] for it in self.items["gt"] if it["id"] == g), {})
            al = next((it["label"] for it in self.items["astra"] if it["id"] == a), {})
            same = "=" if (gl.get("kind"), gl.get("structure")) == (al.get("kind"), al.get("structure")) else "≠"
            self.listbox.insert("end", f"G{idx['gt'].get(g, '?')} ↔ A{idx['astra'].get(a, '?')}  {same}  "
                                       f"{short(gl)}  |  {short(al)}")
            self.listbox.itemconfig(n, fg=self.link_color((g, a)))
        nl_g = len(self.items["gt"]) - len({g for g, _ in links})
        nl_a = len(self.items["astra"]) - len({a for _, a in links})
        self.listbox.insert("end", f"-- unlinked: GT {nl_g}, Astra {nl_a}")
        for side in ("gt", "astra"):
            self.show_sel(side)

    def show_sel(self, side):
        cid = self.sel[side]
        it = next((x for x in self.items[side] if x["id"] == cid), None)
        if it is None:
            self.thumb[side].config(image="")
            self.info[side].config(text="(click a box)")
            return
        im = self.img[side]
        ox, oy = self.off[side]
        bx = self.box_scaled(side, it)
        x0, y0, x1, y1 = ((bx[0] - ox) / self.scale[side], (bx[1] - oy) / self.scale[side],
                          (bx[2] - ox) / self.scale[side], (bx[3] - oy) / self.scale[side])
        pad_x, pad_y = (x1 - x0) * 0.3, (y1 - y0) * 0.3
        crop = im.crop((max(0, x0 - pad_x), max(0, y0 - pad_y), min(im.width, x1 + pad_x), min(im.height, y1 + pad_y)))
        k = min(320 / crop.width, 200 / crop.height)
        crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))))
        self.thumb_img = getattr(self, "thumb_img", {})
        self.thumb_img[side] = ImageTk.PhotoImage(crop)
        self.thumb[side].config(image=self.thumb_img[side])
        lab = it["label"]
        self.info[side].config(text="\n".join(
            [("GT " if side == "gt" else "Astra ") + it["id"][-16:]] +
            [f"{k}: {lab.get(k)}" for k in ("kind", "material", "structure", "floor_shape")]))

    # ---- interaction ----
    def hit(self, side, e):
        best = None
        for it in self.items[side]:
            x0, y0, x1, y1 = self.box_scaled(side, it)
            if x0 - 3 <= e.x <= x1 + 3 and y0 - 3 <= e.y <= y1 + 3:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best[0]:
                    best = (area, it["id"])
        return best and best[1]

    def click(self, side, e):
        cid = self.hit(side, e)
        if cid is None:
            return
        self.sel[side] = None if self.sel[side] == cid else cid
        if self.sel["gt"] and self.sel["astra"]:
            link = [self.sel["gt"], self.sel["astra"]]
            if link not in self.entry["links"]:
                self.entry["links"].append(link)
                self.data.save()
                self.refresh_combo()
            self.sel = {"gt": None, "astra": None}
        self.redraw()

    def unlink_box(self, side, e):
        cid = self.hit(side, e)
        if cid is None:
            return
        j = 0 if side == "gt" else 1
        self.entry["links"] = [l for l in self.entry["links"] if l[j] != cid]
        self.data.save()
        self.refresh_combo()
        self.redraw()

    def delete_link(self):
        s = self.listbox.curselection()
        if not s or s[0] >= len(self.entry["links"]):
            return
        del self.entry["links"][s[0]]
        self.data.save()
        self.refresh_combo()
        self.redraw()

    def clear_sel(self, redraw=True):
        self.sel = {"gt": None, "astra": None}
        if redraw:
            self.redraw()

    def show_score(self):
        txt = score(self.data)
        print(txt)
        w = tk.Toplevel(self.root)
        w.title("Astra vs GT")
        t = tk.Text(w, width=110, height=24, font=("Consolas", 10))
        t.insert("1.0", txt)
        t.pack()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", type=Path, default=ASTRA / "Cropped Balcony/gt_links.jsonl")
    ap.add_argument("--score", action="store_true", help="print the score and exit")
    ap.add_argument("--start", default=None, help="facade id to open first (e.g. facade_04)")
    a = ap.parse_args()
    data = Data(a.links)
    if a.score:
        txt = score(data)
        print(txt)
        out = a.links.with_name("gt_links_score.txt")
        out.write_text(txt + f"\n({time.strftime('%Y-%m-%d %H:%M')})\n", encoding="utf-8")
        return
    start = 0
    if a.start:
        start = next((i for i, f in enumerate(data.facades) if f["fid"] == a.start), 0)
    else:  # first facade not reviewed yet
        start = next((i for i, f in enumerate(data.facades) if not data.state.get(f["fid"], {}).get("done")), 0)
    try:  # sharp images on scaled Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    root.title("Astra ↔ GT balcony links")
    App(root, data, start)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
