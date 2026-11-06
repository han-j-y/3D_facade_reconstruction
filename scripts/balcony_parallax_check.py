#!/usr/bin/env python3
"""Measurement only. Do balcony-to-window offsets on LSAA validation photos follow the camera position?

LSAA photos are crops of a rectilinear projection with y=10, mpp=0.025 (focal 400 px), whose optical axis faces the wall
(lsaa-dataset/step2 + project_panoramas). So a photo's camera foot point is (2268.5 - box_x, 2268.5 - box_y), and the
camera stands 400 photo pixels from the wall. Something d px in front of the wall shifts by (p - c) * d / (400 - d).

Compares three models of the offset: none, shift about the camera foot point, shift about the photo centre (control).
Writes runs_val30_seated/compare/parallax_check.json and parallax_sheet.jpg."""
import csv, json, math, random, statistics as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; DS = E.parents[1] / "datasets"; L = DS / "lsaa-dataset/annotations"; RUN = E / "runs_val30_combined"
O = E / "runs_val30_seated/compare"; FOCAL = 400.0; CENTER = 56.71281292110204 / 0.025  # n/2 and the horizon row, both tan(80 deg)*10/mpp
FLOOR_M = 3.2
ids = {t: s for t, s in json.load(open(DS / "valset/val30_ids.json")).items() if not s.startswith("cmp")}
F = json.load(open(L / "facade_detection_result.json")); VA = {}
for r in csv.DictReader(open(L / "Properties23K.csv", encoding="utf8")): VA[r["name"]] = float(r["view_angle"])
for r in csv.DictReader(open(L / "Properties200K.csv", encoding="utf8")): VA.setdefault(r["name"], float(r["view_angle"]))
pairs, faces, angle_err = [], {}, []
for t, s in ids.items():
    p = RUN / t / "balcony/facade_dsl_with_balconies.json"; box = F[s + ".jpg"]["box"]
    cx, cy = CENTER - box[0], CENTER - box[1]
    angle_err.append(abs(math.degrees(math.atan((box[0] + box[2] / 2 - CENTER) / FOCAL)) - VA[s + ".jpg"]))
    if not p.exists(): continue
    D = json.load(open(p)); W = [u for u in D["instances"] if u.get("kind", "window") == "window"]; iw, ih = D["meta"]["image_size"]
    by = {}
    for u in W: by.setdefault(u["floor"], []).append(u["box_xyxy"])
    ys = sorted(st.median((b[1] + b[3]) / 2 for b in v) for v in by.values()); pitch = st.median(b - a for a, b in zip(ys, ys[1:])) if len(ys) > 1 else None
    if not pitch: continue
    ppm = pitch / FLOOR_M; faces[t] = dict(cx=cx, cy=cy, iw=iw, ih=ih, ppm=ppm, cam_m=FOCAL / ppm, boxes=[], n=0)
    for b in D["layout"].get("balconies", []):
        if "width_norm" not in b or b.get("structure") == "rooftop": continue
        B = b["box_xyxy"]; hit = [w for w in by.get(b["floor"], []) if min(B[2], w[2]) - max(B[0], w[0]) > 0.5 * min(B[2] - B[0], w[2] - w[0])]
        if len(hit) != 1: continue
        w = hit[0]; ww = w[2] - w[0]
        if not 0.5 * ww <= B[2] - B[0] <= 2.5 * ww: continue  # one balcony, one window: long galleries have no single centre
        x = (w[0] + w[2]) / 2
        pairs.append(dict(t=t, floor=b["floor"], ww=ww, ppm=ppm, cx=cx, cy=cy, iw=iw, ih=ih, x=x, dx=(B[0] + B[2]) / 2 - x, yb=w[3], d_bot=B[3] - w[3], d_top=B[1] - w[3]))
        faces[t]["boxes"].append((B, w)); faces[t]["n"] += 1
def pred(p, model):  # shift per metre of depth, photo px
    c = p["cx"] if model == "camera" else p["iw"] / 2
    return (p["x"] - c) * p["ppm"] / FOCAL
def fit(ps, model):  # robust one-parameter fit through the origin, metres
    q = sorted((p["dx"] / pred(p, model), abs(pred(p, model))) for p in ps if abs(pred(p, model)) > 1e-6); tot = sum(a for _, a in q); s = 0
    for k, a in q:
        s += a
        if s >= tot / 2: return k
def err(ps, model, k):  # median absolute offset left over, in window widths
    return st.median(abs(p["dx"] - (k * pred(p, model) if model else 0)) / p["ww"] for p in ps)
def row_slope(ps, key):  # shift with height, floor-to-floor within a facade: offset against (y - horizon), per window of depth in m
    xs, ys = [], []
    for t in {p["t"] for p in ps}:
        q = [p for p in ps if p["t"] == t]
        if len({p["floor"] for p in q}) < 2: continue
        mx, my = st.mean(p["yb"] for p in q), st.mean(p[key] for p in q)
        for p in q: xs.append((p["yb"] - mx) * p["ppm"] / FOCAL); ys.append(p[key] - my)
    return (sum(a * b for a, b in zip(xs, ys)) / sum(a * a for a in xs), len(xs)) if xs else (None, 0)
res = {}
kc, kp = fit(pairs, "camera"), fit(pairs, "centre")
res["photos"] = len(ids); res["max_view_angle_error_deg"] = round(max(angle_err), 2); res["pairs"] = len(pairs); res["facades_with_pairs"] = len({p["t"] for p in pairs})
res["implied_camera_distance_m_median"] = round(st.median(f["cam_m"] for f in faces.values()), 1)
res["depth_m"] = dict(camera=round(kc, 2), photo_centre=round(kp, 2))
res["left_over_window_widths"] = dict(no_shift=round(err(pairs, None, 0), 3), camera=round(err(pairs, "camera", kc), 3), photo_centre=round(err(pairs, "centre", kp), 3))
random.seed(0); T = sorted({p["t"] for p in pairs}); bt = []
for _ in range(2000):
    s = [p for t in random.choices(T, k=len(T)) for p in pairs if p["t"] == t]; a, b = fit(s, "camera"), fit(s, "centre")
    bt.append((a, err(s, None, 0) - err(s, "camera", a), err(s, "centre", b) - err(s, "camera", a)))
ci = lambda v: [round(sorted(v)[50], 3), round(sorted(v)[1949], 3)]
res["ci95_by_facade"] = dict(depth_m=ci([x[0] for x in bt]), gain_vs_no_shift=ci([x[1] for x in bt]), gain_vs_photo_centre=ci([x[2] for x in bt]))
res["vertical_depth_m"] = {k: (round(v[0], 2), v[1]) for k, v in (("balcony_bottom", row_slope(pairs, "d_bot")), ("balcony_top", row_slope(pairs, "d_top")))}
per = []
for t in T:
    q = [p for p in pairs if p["t"] == t]
    if len(q) < 4: continue
    per.append(dict(t=t, n=len(q), va=round(VA[ids[t] + ".jpg"], 1), none=round(err(q, None, 0), 3), cam=round(err(q, "camera", kc), 3), cen=round(err(q, "centre", kp), 3), depth=round(fit(q, "camera"), 2)))
res["per_facade"] = per
json.dump(res, open(O / "parallax_check.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "per_facade"}, indent=1))
for r in sorted(per, key=lambda r: r["va"]): print(r)
# Sheet: windows green, balconies orange, line from window centre to balcony centre; magenta line = camera side.
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20); H = 560; rows = []
for r in sorted(per, key=lambda r: -abs(r["va"]))[:8]:
    t = r["t"]; f = faces[t]; im = Image.open(DS / "valset/photos" / (ids[t] + ".jpg")).convert("RGB"); d = ImageDraw.Draw(im); lw = max(2, im.height // 300)
    for B, w in f["boxes"]:
        d.rectangle(w, outline=(40, 220, 60), width=lw); d.rectangle(B, outline=(255, 140, 0), width=lw); ym = (B[1] + B[3]) / 2
        d.line([(w[0] + w[2]) / 2, ym, (B[0] + B[2]) / 2, ym], fill=(255, 0, 255), width=lw * 2)
    s = H / im.height; A = im.resize((round(im.width * s), H)); R = Image.new("RGB", (A.width, H + 30), "white"); D2 = ImageDraw.Draw(R)
    side = "camera right" if f["cx"] > f["iw"] else "camera left" if f["cx"] < 0 else "camera in front"
    D2.text((4, 3), f"{t} {r['va']:+.0f} deg, {side}", fill="black", font=Ft); R.paste(A, (0, 30)); rows.append(R)
page = Image.new("RGB", (sum(sorted((q.width for q in rows))[-4:]) + 40, 2 * (H + 40)), "white"); x = y = 0
for i, q in enumerate(rows):
    if i == 4: x, y = 0, H + 40
    page.paste(q, (x, y)); x += q.width + 10
page.save(O / "parallax_sheet.jpg", quality=88); print(O / "parallax_sheet.jpg", page.size)
