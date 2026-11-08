r"""Ground truth on the val30 pipeline units: balcony groups (types) and the four railing labels.

Each photo shows the open-balcony units the pipeline typed (``runs/val30/<stem>/balcony/balcony_units.json``).
Two modes, switched with Tab:

* GROUP mode: give every unit a group number. Units of one group are the same balcony design and should
  become one type in the DSL.
* LABEL mode: set kind / material / floor / structure, once per group (click a box of the group) or for one
  unit (Shift+click). Keys as in ``label_ui.py``. Unlabeled members of a group take the group's labels
  automatically, field by field (``fill_group_labels``: from the group's most common complete label set);
  values already set are kept.

Saved per photo to ``data/val30/type_gt.jsonl`` (one JSON row per photo, boxes included, so the file stays
valid when the run is redone). No prefill: the pipeline's types and labels are never shown.

Rules (user, 2026-11-07):

* same group = the same balcony design (railing kind and pattern, material, floor shape, projecting /
  juliet); a clearly different width class (2x or more) is another group; small differences from the view
  angle or the box are the same group; mirror twins are the same group;
* projecting / juliet: the reference plane is the wall; a slab or frame box sticking out of the wall is
  projecting. An inset balcony (loggia, recess, a glass box set into the wall) is measured against the
  outer wall right beside it: not sticking out of it = juliet. A railing on a small slab or planter shelf
  that sticks out = projecting. Juliet has no floor shape. Material only for open work;
* a box that is not an open balcony (sill, cornice, sign, shop grille, wall) gets X;
* a box over two balconies of one design keeps that design's group and labels.

Mouse:
  GROUP mode   left-click a box: assign the current group (again: unassign); left-drag: assign every box
               whose centre is inside; right-click: unassign
  LABEL mode   left-click a box: target = its whole group (or the box when it has no group);
               Shift+left-click: target = that box only

Keys (both modes):
  Tab        switch GROUP / LABEL mode
  X          toggle "not a balcony" on the box under the mouse
  Ctrl+Z     undo (this photo)
  Right / Enter   save, next photo        Left   save, previous photo
  S save     H hide / show tags          Q / Esc   save and quit
GROUP mode:
  1-9, 0     current group 1-9, 10 (Shift+1-9: 11-19)
  A          assign the current group to every unassigned box
  N / P      next / previous photo       J   next photo not yet complete
LABEL mode (on the target):
  1 / 2 / 3  kind: open_work / surface_panel / solid
  M / N      material: metal / masonry (open_work only)
  R / C      floor shape: rectangle / circle (projecting only)
  P / J      structure: projecting / juliet

A photo is complete when every box has X, or a group, kind, structure, material (open work) and floor
shape (projecting).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/type_gt_ui.py
    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/type_gt_ui.py --start v16
    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/type_gt_ui.py --check
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "val30"
PHOTOS = ROOT / "data" / "val30"
OUT = ROOT / "data" / "val30" / "type_gt.jsonl"
COLORS = ["#00aaff", "#ff00c8", "#ffdc00", "#00e678", "#965aff", "#ff7800", "#00ffff", "#ff5050",
          "#aaff00", "#7aa0ff", "#ff9ec8", "#b4783c", "#80ffd4", "#ffffff", "#c8c800", "#ff3cff",
          "#3cc8ff", "#ffb43c", "#78ff78", "#dc78ff"]
LABEL_FIELDS = ("kind", "material", "floor_shape", "structure")
ABBR = {"open_work": "OW", "surface_panel": "SP", "solid": "SO", "metal": "me", "masonry": "ma",
        "rectangle": "R", "circle": "C", "projecting": "P", "juliet": "J"}
SHIFT_DIGITS = ["exclam", "at", "numbersign", "dollar", "percent", "asciicircum", "ampersand", "asterisk",
                "parenleft"]


def empty_unit() -> dict:
    return {"group": None, "not_balcony": False, "kind": None, "material": None, "floor_shape": None,
            "structure": None}


def load_photos(run_dir: Path, photos: Path) -> list[dict]:
    """Photos with at least one pipeline unit, in val30 id order."""
    ids = json.loads((photos / "val30_ids.json").read_text(encoding="utf-8"))
    out = []
    for vid, stem in sorted(ids.items()):
        uj = run_dir / stem / "balcony" / "balcony_units.json"
        if not uj.is_file():
            continue
        units = json.loads(uj.read_text(encoding="utf-8")).get("units") or []
        if not units:
            continue
        img = next((p for p in (photos / f"{stem}.jpg", photos / f"{stem}.png") if p.is_file()), None)
        if img is None:
            continue
        out.append({"id": vid, "stem": stem, "image": img, "source": str(uj.relative_to(ROOT)),
                    "boxes": [[int(v) for v in u["box_xyxy"]] for u in units]})
    return out


def load_gt(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["stem"]: r for r in rows}


def save_gt(path: Path, gt: dict[str, dict], order: list[str]) -> None:
    rows = [gt[s] for s in order if s in gt] + [r for s, r in gt.items() if s not in order]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def state_for(photo: dict, gt: dict[str, dict]) -> list[dict]:
    """Per box the saved unit (group, not_balcony, labels) when the saved boxes match, else empty."""
    row = gt.get(photo["stem"])
    if row and [u["box"] for u in row["units"]] == photo["boxes"]:
        return [{**empty_unit(), **{k: u.get(k) for k in empty_unit()}} for u in row["units"]]
    if row:
        print(f"warn {photo['id']}: saved boxes differ from the run; starting empty (old row kept until save)")
    return [empty_unit() for _ in photo["boxes"]]


def unit_complete(u: dict) -> bool:
    if u["not_balcony"]:
        return True
    if u["group"] is None or u["kind"] is None or u["structure"] is None:
        return False
    if u["kind"] == "open_work" and u["material"] is None:
        return False
    if u["structure"] == "projecting" and u["floor_shape"] is None:
        return False
    return True


def complete(state: list[dict]) -> bool:
    return all(unit_complete(u) for u in state)


def set_label(u: dict, field: str, value: str) -> None:
    """Set one label; material only on open work and floor only on projecting (ignored otherwise)."""
    if field == "material" and u["kind"] not in (None, "open_work"):
        return
    if field == "floor_shape" and u["structure"] == "juliet":
        return
    u[field] = value
    if field == "kind" and value != "open_work":
        u["material"] = None
    if field == "structure" and value == "juliet":
        u["floor_shape"] = None


def fill_group_labels(state: list[dict]) -> int:
    """Fill the empty label fields of every group member from the group's reference labels.

    Reference = the most common label set among the group's complete members; without a complete member,
    the most common value of each field among the members that have it. Only empty fields are filled, so
    values already set (per-unit exceptions) are kept; material only when the kind is open work, floor
    only when projecting. Returns the number of units changed.
    """
    n = 0
    groups = {u["group"] for u in state if u["group"] is not None and not u["not_balcony"]}
    for g in groups:
        mem = [u for u in state if u["group"] == g and not u["not_balcony"]]
        full = [tuple(u[f] for f in LABEL_FIELDS) for u in mem if unit_complete(u)]
        if full:
            ref = dict(zip(LABEL_FIELDS, Counter(full).most_common(1)[0][0]))
        else:
            ref = {}
            for f in LABEL_FIELDS:
                vals = [u[f] for u in mem if u[f] is not None]
                ref[f] = Counter(vals).most_common(1)[0][0] if vals else None
        for u in mem:
            before = tuple(u[f] for f in LABEL_FIELDS)
            for f in ("kind", "structure"):
                if u[f] is None and ref.get(f) is not None:
                    u[f] = ref[f]
            if u["kind"] == "open_work" and u["material"] is None and ref.get("material") is not None:
                u["material"] = ref["material"]
            if u["kind"] != "open_work":
                u["material"] = None
            if u["structure"] == "projecting" and u["floor_shape"] is None and ref.get("floor_shape") is not None:
                u["floor_shape"] = ref["floor_shape"]
            if u["structure"] == "juliet":
                u["floor_shape"] = None
            n += tuple(u[f] for f in LABEL_FIELDS) != before
    return n


def label_text(u: dict) -> str:
    parts = [ABBR.get(u["kind"], "?")]
    if u["kind"] == "open_work":
        parts.append(ABBR.get(u["material"], "?"))
    parts.append(ABBR.get(u["structure"], "?"))
    if u["structure"] == "projecting":
        parts.append(ABBR.get(u["floor_shape"], "?"))
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--photos", type=Path, default=PHOTOS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", default=None, help="val30 id (v16) or stem to open first")
    ap.add_argument("--check", action="store_true", help="print the photo list and progress, no window")
    args = ap.parse_args()

    photos = load_photos(args.run_dir, args.photos)
    gt = load_gt(args.out)
    if args.check:
        n_units = sum(len(p["boxes"]) for p in photos)
        done = sum(1 for p in photos if complete(state_for(p, gt)))
        print(f"{len(photos)} photos, {n_units} units, complete {done} -> {args.out}")
        for p in photos:
            st = state_for(p, gt)
            print(f"  {p['id']} {len(p['boxes']):3d} units  {sum(unit_complete(u) for u in st):3d} done  "
                  f"{'COMPLETE' if complete(st) else '        '}  {p['stem']}")
        return


    import tkinter as tk

    from PIL import Image, ImageTk

    if sys.platform == "win32":  # sharp text and true pixel sizes on scaled displays
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    root.title("val30 balcony groups + labels")
    W, H = min(1600, root.winfo_screenwidth() - 40), min(980, root.winfo_screenheight() - 90)
    PANEL_BG, FG = "#303030", "white"
    SMALL = ("Yu Gothic UI", 9)
    SMALL_BOLD = ("Yu Gothic UI", 9, "bold")
    canvas = tk.Canvas(root, width=W - 460, height=H, bg="#202020", highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    side = tk.Frame(root, width=460, bg=PANEL_BG)
    side.pack(side=tk.RIGHT, fill=tk.Y)
    side.pack_propagate(False)

    s: dict = {"i": 0, "group": 1, "mode": "GROUP", "show": True, "undo": [], "drag": None, "scale": 1.0,
               "off": (0, 0), "target": []}
    if args.start:
        for k, p in enumerate(photos):
            if args.start in (p["id"], p["stem"]):
                s["i"] = k
    order = [p["stem"] for p in photos]
    buttons: dict = {}

    # ---------------- actions (shared by buttons and keys) ----------------
    def cur():
        return photos[s["i"]]

    def push():
        s["undo"].append(copy.deepcopy(s["state"]))

    def save():
        """Write only this photo's row: re-read the file first, so edits made elsewhere are kept."""
        p = cur()
        gt.clear()
        gt.update(load_gt(args.out))
        gt[p["stem"]] = {
            "id": p["id"], "stem": p["stem"], "source": p["source"],
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "complete": complete(s["state"]),
            "units": [{"box": b, **u} for b, u in zip(p["boxes"], s["state"])],
        }
        save_gt(args.out, gt, order)

    def go(k):
        save()
        s["i"] = k % len(photos)
        load()

    def next_incomplete():
        save()
        for d in range(1, len(photos) + 1):
            k = (s["i"] + d) % len(photos)
            if not complete(state_for(photos[k], gt)):
                s["i"] = k
                break
        load()

    def set_mode(mode=None):
        s["mode"] = mode or ("LABEL" if s["mode"] == "GROUP" else "GROUP")
        s["target"] = []
        draw()

    def set_group(n):
        s["group"] = n
        draw()

    def apply_label(field, value):
        if not s["target"]:
            return
        push()
        for k in s["target"]:
            if not s["state"][k]["not_balcony"]:
                set_label(s["state"][k], field, value)
        draw()

    def toggle_x(k=None):
        ks = [k] if k is not None else list(s["target"])
        if not ks:
            return
        push()
        for j in ks:
            u = s["state"][j]
            s["state"][j] = empty_unit() if u["not_balcony"] else {**empty_unit(), "not_balcony": True}
        draw()

    def assign_rest():
        push()
        for u in s["state"]:
            if u["group"] is None and not u["not_balcony"]:
                u["group"] = s["group"]
        draw()

    def undo():
        if s["undo"]:
            s["state"] = s["undo"].pop()
            draw()

    def toggle_tags():
        s["show"] = not s["show"]
        draw()

    def save_now():
        save()
        draw()

    def quit_ui():
        save()
        root.destroy()

    # ---------------- side panel ----------------
    def section(title):
        if title:
            tk.Label(side, text=title, bg=PANEL_BG, fg="#ffd24a", font=SMALL_BOLD, anchor="w").pack(
                fill=tk.X, padx=8, pady=(6, 0))
        f = tk.Frame(side, bg=PANEL_BG)
        f.pack(fill=tk.X, padx=8)
        return f

    def button(parent, key, text, cmd, width=None, col=None, row=None):
        b = tk.Button(parent, text=text, command=cmd, font=SMALL, takefocus=0, relief=tk.RAISED,
                      bg="#4a4a4a", fg=FG, activebackground="#6a6a6a", activeforeground=FG,
                      width=width or 0, padx=1, pady=0, bd=1)
        if row is None:
            b.pack(side=tk.LEFT, padx=1, pady=1)
        else:
            b.grid(row=row, column=col, padx=1, pady=1, sticky="we")
        buttons[key] = b
        return b

    info = tk.Label(side, justify=tk.LEFT, anchor="nw", bg=PANEL_BG, fg=FG, font=("Consolas", 9),
                    wraplength=440)
    info.pack(fill=tk.X, padx=8, pady=(6, 0))

    f = section("Photo")
    button(f, "prev", "◀ Prev [←]", lambda: go(s["i"] - 1))
    button(f, "next", "Next ▶ [→]", lambda: go(s["i"] + 1))
    button(f, "next_incomplete", "Next incomplete [J]", next_incomplete)
    button(f, "save", "Save [S]", save_now)
    f = section("")
    button(f, "tags", "Tags [H]", toggle_tags)
    button(f, "undo", "Undo [Ctrl+Z]", undo)
    button(f, "quit", "Save + quit [Q]", quit_ui)

    f = section("Mode [Tab]")
    button(f, "mode_GROUP", "Group", lambda: set_mode("GROUP"), width=13)
    button(f, "mode_LABEL", "Label", lambda: set_mode("LABEL"), width=13)

    f = section("Group number [1-9, 0, Shift+1-9]")
    for n in range(1, 21):
        button(f, f"group_{n}", str(n), lambda n=n: set_group(n), width=2, row=(n - 1) // 10, col=(n - 1) % 10)
    f = section("")
    button(f, "assign_rest", "Assign rest to group [A]", assign_rest)
    button(f, "x", "Not a balcony [X]", lambda: toggle_x())

    f = section("Labels (selected box / group)")
    rows = [("kind [1/2/3]", [("kind_open_work", "OW open", "kind", "open_work"),
                               ("kind_surface_panel", "SP panel", "kind", "surface_panel"),
                               ("kind_solid", "SO solid", "kind", "solid")]),
            ("material [M/N]", [("material_metal", "me metal", "material", "metal"),
                                 ("material_masonry", "ma masonry", "material", "masonry")]),
            ("structure [P/J]", [("structure_projecting", "P projecting", "structure", "projecting"),
                                  ("structure_juliet", "J juliet", "structure", "juliet")]),
            ("floor [R/C]", [("floor_shape_rectangle", "R rectangle", "floor_shape", "rectangle"),
                              ("floor_shape_circle", "C circle", "floor_shape", "circle")])]
    for r, (title, items) in enumerate(rows):
        tk.Label(f, text=title, bg=PANEL_BG, fg="#dddddd", font=SMALL, anchor="w", width=12).grid(
            row=r, column=0, sticky="w")
        for c, (key, text, field, value) in enumerate(items):
            button(f, key, text, lambda field=field, value=value: apply_label(field, value), width=9, row=r,
                   col=c + 1)

    help_lbl = tk.Label(side, justify=tk.LEFT, anchor="nw", bg=PANEL_BG, fg="#bbbbbb", font=SMALL,
                        wraplength=440)
    help_lbl.pack(fill=tk.X, padx=8, pady=(8, 4))

    # ---------------- drawing ----------------
    def load():
        p = cur()
        gt.clear()
        gt.update(load_gt(args.out))
        s["state"] = state_for(p, gt)
        s["undo"] = []
        s["target"] = []
        im = Image.open(p["image"]).convert("RGB")
        cw, ch = max(200, canvas.winfo_width()), max(200, canvas.winfo_height())
        sc = min(cw / im.width, ch / im.height)
        s["scale"] = sc
        s["off"] = ((cw - im.width * sc) / 2, (ch - im.height * sc) / 2)
        s["tk"] = ImageTk.PhotoImage(im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc)))))
        draw()

    def to_canvas(b):
        sc, (ox, oy) = s["scale"], s["off"]
        return [b[0] * sc + ox, b[1] * sc + oy, b[2] * sc + ox, b[3] * sc + oy]

    def target_values(field):
        return {s["state"][k][field] for k in s["target"] if not s["state"][k]["not_balcony"]}

    def refresh_buttons():
        for key, b in buttons.items():
            on = False
            if key.startswith("mode_"):
                on = s["mode"] == key[5:]
            elif key.startswith("group_"):
                on = s["group"] == int(key[6:])
                b.config(fg=COLORS[(int(key[6:]) - 1) % len(COLORS)])
            else:
                for field in LABEL_FIELDS:
                    if key.startswith(field + "_"):
                        on = target_values(field) == {key[len(field) + 1:]}
            b.config(bg="#2e7d32" if on else "#4a4a4a", relief=tk.SUNKEN if on else tk.RAISED)
        has_target = bool(s["target"])
        for field in LABEL_FIELDS:
            for key, b in buttons.items():
                if key.startswith(field + "_"):
                    b.config(state=tk.NORMAL if has_target else tk.DISABLED)
        buttons["x"].config(state=tk.NORMAL if has_target else tk.DISABLED)

    def draw():
        p = cur()
        fill_group_labels(s["state"])
        canvas.delete("all")
        canvas.create_image(*s["off"], image=s["tk"], anchor="nw")
        st = s["state"]
        for k, (b, u) in enumerate(zip(p["boxes"], st)):
            x0, y0, x1, y1 = to_canvas(b)
            if k in s["target"]:
                canvas.create_rectangle(x0 - 4, y0 - 4, x1 + 4, y1 + 4, outline="yellow", width=2)
            if u["not_balcony"]:
                canvas.create_rectangle(x0, y0, x1, y1, outline="#888888", width=2, dash=(4, 3))
                canvas.create_line(x0, y0, x1, y1, fill="#888888", width=2)
                canvas.create_line(x0, y1, x1, y0, fill="#888888", width=2)
                continue
            col = "white" if u["group"] is None else COLORS[(u["group"] - 1) % len(COLORS)]
            canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=2 if u["group"] is None else 3,
                                    dash=(5, 3) if u["group"] is None else None)
            if s["show"]:
                tag = "" if u["group"] is None else str(u["group"])
                if s["mode"] == "LABEL" or any(u[f] for f in LABEL_FIELDS):
                    tag = (tag + " " + label_text(u)).strip()
                if not unit_complete(u):
                    tag += " !"
                if tag:
                    tw = 7 * len(tag) + 8
                    canvas.create_rectangle(x0, y0 - 18, x0 + tw, y0, fill=col, outline=col)
                    canvas.create_text(x0 + 4, y0 - 9, text=tag, fill="black", anchor="w",
                                       font=("Arial", 10, "bold"))
        groups = sorted({u["group"] for u in st if u["group"] is not None and not u["not_balcony"]})
        done = sum(1 for q in photos if complete(state_for(q, gt)))
        lines = [f"{p['id']}  ({s['i'] + 1}/{len(photos)})   complete {done}/{len(photos)} photos", p["stem"][:48],
                 f"boxes {len(st)}  done {sum(unit_complete(u) for u in st)}  X {sum(u['not_balcony'] for u in st)}"
                 f"   {'COMPLETE' if complete(st) else 'incomplete'}  saved: {'yes' if p['stem'] in gt else 'no'}",
                 f"mode: {s['mode']}   current group: {s['group']}"]
        for g in groups:
            mem = [u for u in st if u["group"] == g and not u["not_balcony"]]
            labs = Counter(label_text(u) for u in mem)
            lines.append(f" g{g:<2d} {len(mem):2d} units  " + ", ".join(f"{k} x{v}" for k, v in labs.most_common()))
        info.config(text="\n".join(lines))
        if s["mode"] == "GROUP":
            help_lbl.config(text="Group mode: click a box to give it the current group (click again to clear). "
                                 "Drag to assign every box inside. Right-click to clear.")
        else:
            help_lbl.config(text="Label mode: click a box to select its whole group, Shift+click to select only "
                                 "that box. Then set labels with the buttons or keys.")
        refresh_buttons()

    # ---------------- mouse ----------------
    def box_at(x, y):
        hits = []
        for k, b in enumerate(cur()["boxes"]):
            x0, y0, x1, y1 = to_canvas(b)
            if x0 <= x <= x1 and y0 <= y <= y1:
                hits.append(((x1 - x0) * (y1 - y0), k))
        return min(hits)[1] if hits else None

    def mouse_box():
        return box_at(canvas.winfo_pointerx() - canvas.winfo_rootx(), canvas.winfo_pointery() - canvas.winfo_rooty())

    def on_press(ev):
        s["drag"] = (ev.x, ev.y, box_at(ev.x, ev.y), bool(ev.state & 0x0001))

    def on_release(ev):
        if s["drag"] is None:
            return
        x, y, k, shift = s["drag"]
        s["drag"] = None
        canvas.delete("rubber")
        click = abs(ev.x - x) < 5 and abs(ev.y - y) < 5
        if s["mode"] == "LABEL":
            if click and k is not None:
                g = s["state"][k]["group"]
                if shift or g is None:
                    s["target"] = [k]
                else:
                    s["target"] = [j for j, u in enumerate(s["state"]) if u["group"] == g and not u["not_balcony"]]
                    s["group"] = g
            elif click:
                s["target"] = []
            draw()
            return
        if click:
            if k is None:
                s["target"] = []
                draw()
                return
            push()
            u = s["state"][k]
            u["not_balcony"] = False
            u["group"] = None if u["group"] == s["group"] else s["group"]
            s["target"] = [k]
        else:
            x0, x1 = sorted((x, ev.x))
            y0, y1 = sorted((y, ev.y))
            push()
            hit = []
            for j, (b, u) in enumerate(zip(cur()["boxes"], s["state"])):
                bx0, by0, bx1, by1 = to_canvas(b)
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    u["group"], u["not_balcony"] = s["group"], False
                    hit.append(j)
            s["target"] = hit
        draw()

    def on_motion(ev):
        if s["drag"] is None or s["mode"] == "LABEL":
            return
        x, y = s["drag"][:2]
        canvas.delete("rubber")
        canvas.create_rectangle(x, y, ev.x, ev.y, outline="yellow", dash=(3, 3), tags="rubber")

    def on_right(ev):
        if s["mode"] != "GROUP":
            return
        k = box_at(ev.x, ev.y)
        if k is not None:
            push()
            s["state"][k]["group"] = None
            s["state"][k]["not_balcony"] = False
            draw()

    # ---------------- keys ----------------
    LABEL_KEYS = {"1": ("kind", "open_work"), "2": ("kind", "surface_panel"), "3": ("kind", "solid"),
                  "m": ("material", "metal"), "n": ("material", "masonry"),
                  "r": ("floor_shape", "rectangle"), "c": ("floor_shape", "circle"),
                  "p": ("structure", "projecting"), "j": ("structure", "juliet")}

    def on_key(ev):
        key = ev.keysym
        low = key.lower()
        shift = bool(ev.state & 0x0001)
        ctrl = bool(ev.state & 0x0004)
        if key == "Tab":
            set_mode()
            return "break"
        if low == "z" and ctrl:
            undo()
            return
        if low == "x":
            k = mouse_box()
            toggle_x(k) if k is not None else toggle_x()
            return
        if key in ("Right", "Return"):
            go(s["i"] + 1)
            return
        if key == "Left":
            go(s["i"] - 1)
            return
        if low == "s" and not ctrl:
            save_now()
            return
        if low == "h":
            toggle_tags()
            return
        if low == "q" or key == "Escape":
            quit_ui()
            return
        if s["mode"] == "LABEL":
            if low in LABEL_KEYS:
                apply_label(*LABEL_KEYS[low])
            return
        if key in "1234567890" and len(key) == 1:
            n = 10 if key == "0" else int(key)
            set_group(n + 10 if shift and key != "0" else n)
        elif key in SHIFT_DIGITS:
            set_group(11 + SHIFT_DIGITS.index(key))
        elif low == "a":
            assign_rest()
        elif low == "n":
            go(s["i"] + 1)
        elif low == "p":
            go(s["i"] - 1)
        elif low == "j":
            next_incomplete()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>", on_right)
    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", quit_ui)
    root.geometry(f"{W}x{H}")
    root.after(100, load)
    root.mainloop()


if __name__ == "__main__":
    main()
