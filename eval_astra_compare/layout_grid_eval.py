"""Layout F1 of the paper's main table (tab:main): unified grid-to-grid layout F1 for windows, balconies and oriels (Astra's 46 test facades).

Each image (photo, and each method's rendering placed on the photo) gets its own floor x bay grid from its own windows
(floors = window rows, bays = window columns). A balcony or oriel box far from every floor / outside every bay adds a
floor / bay of its own. Photo and render floors are matched one-to-one in order, bays likewise; occupied cells on
unmatched floors / bays count as missed (photo) or extra (render).
Occupancy, the same in photo and render:
  window  : a window centre lies in the cell;
  balcony : floor = the window row whose sill is nearest the box centre, bays covered >= 50 % of their window span;
  oriel   : bays covered >= 50 %, floors whose window-row centre lies inside the box.
Don't care: cells whose photo windows are all unusable (windows only); cells under a balcony ignore region (balconies).
Inputs: photo windows = SAM3 labels; photo balconies = hand boxes; photo oriels = labelled cells (one box per cell on the
grid they were labelled on). Render windows = SAM3 on the rendering. Render balconies / oriels = read from each
method's 3D model: ours by object name (Balconies collection, enclosed shells = oriels), Astra balconies by a name
rule, Astra oriels = regions of its depth grid > 0.3 m in front of the wall (oriel_objects_current.py)."""
import json, re, sys, os
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
A = paths.ASTRA_DIR
E2E = paths.REPO
LBL = paths.LABELS_DIR
HM = paths.HITMAP_DIR  # front-camera object hit maps of both 3D models (hitmap.py)
sys.path[:0] = [str(paths.HERE), str(E2E)]
os.chdir(A)
import layout_balcony as LB, layout_floorwise as lf, eval_testset as ev
from eval_astra import init_alignment, refine_alignment
from eval_astra_protocol import grid_map, _bands
from balcony_train.enclosed_detect import _bands_from_windows_dsl as dsl_bands

RULES = {"strict": r"balcon|juliet|terrace|loggia",
         "medium": r"balcon|juliet|terrace|loggia|balust|railing|guard|handrail",
         "broad": r"balcon|juliet|terrace|loggia|balust|railing|guard|handrail|rail|baluster|console|bracket|slab|parapet"}


def boxes_from(hm, sel):
    g = np.array(hm["grid"]); GH, GW = g.shape; W, H = hm["size"]
    mask = np.isin(g, list(sel)) if sel else np.zeros_like(g, bool)
    lab, n = ndimage.label(ndimage.binary_dilation(mask, np.ones((3, 3))), np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        ys, xs = np.where((lab == k) & mask)
        if len(xs) < 4: continue
        out.append([xs.min() * W / GW, ys.min() * H / GH, (xs.max() + 1) * W / GW, (ys.max() + 1) * H / GH])
    return out


def object_boxes(hm, sel, key=lambda n: n, pad=0.01, gapx=None):
    """One box per object group (all hits of the selected objects sharing key(name)), then boxes that overlap
    (after padding by pad x image size) are merged. Same procedure for both methods."""
    g = np.array(hm["grid"]); GH, GW = g.shape; W, H = hm["size"]
    groups = {}
    for i in sel:
        ys, xs = np.where(g == i)
        if len(xs) < 2: continue
        b = [xs.min() * W / GW, ys.min() * H / GH, (xs.max() + 1) * W / GW, (ys.max() + 1) * H / GH]
        k = key(hm["names"][i]); o = groups.get(k)
        groups[k] = b if o is None else [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
    boxes = list(groups.values()); px, py = pad * W, pad * H
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if gapx is not None:  # row merge: same row (vertical overlap >= half the shorter), horizontal gap <= gapx
                    vo = min(a[3], b[3]) - max(a[1], b[1]); ok = vo >= 0.5 * min(a[3] - a[1], b[3] - b[1]) and max(a[0], b[0]) - min(a[2], b[2]) <= gapx
                else:
                    ok = a[0] - px <= b[2] and b[0] - px <= a[2] and a[1] - py <= b[3] and b[1] - py <= a[3]
                if ok:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]; boxes.pop(j); merged = True; break
            if merged: break
    return boxes


def nearest(y, vals): return int(np.argmin([abs(y - v) for v in vals]))


def grid(W, Bc, Oc):
    boxes = list(Bc) + list(Oc)
    ref = W if W else boxes
    mh = float(np.median([b[3] - b[1] for b in ref])) * (1 if W else 0.5)
    mw = float(np.median([b[2] - b[0] for b in ref]))
    fl = _bands(W, 1, 0.5 * mh) if W else []; ba = _bands(W, 0, 0.5 * mw) if W else []
    floors = [dict(yc=lf.ycentre(W, r), bottom=float(np.median([W[i][3] for i in r])), wins=r) for r in fl]
    bays = [dict(xc=float(np.mean([(W[i][0] + W[i][2]) / 2 for i in r])),
                 span=(float(np.median([W[i][0] for i in r])), float(np.median([W[i][2] for i in r]))), wins=r) for r in ba]
    if os.environ.get("BAY_MERGE") == "1":  # one window column whose windows jitter stays one bay
        bays.sort(key=lambda B: B["xc"]); mb = []
        for B in bays:
            if mb:
                p_ = mb[-1]; lo, hi = max(p_["span"][0], B["span"][0]), min(p_["span"][1], B["span"][1])
                if hi - lo >= 0.5 * min(p_["span"][1] - p_["span"][0], B["span"][1] - B["span"][0]):
                    r = p_["wins"] + B["wins"]
                    mb[-1] = dict(xc=float(np.mean([(W[i][0] + W[i][2]) / 2 for i in r])),
                                  span=(float(np.median([W[i][0] for i in r])), float(np.median([W[i][2] for i in r]))), wins=r)
                    continue
            mb.append(B)
        bays = mb
    fp = float(np.median(np.diff(sorted(f["bottom"] for f in floors)))) if len(floors) > 1 else 2 * mh
    gapy = lambda b, y: max(0.0, b[1] - y, y - b[3])
    for b in sorted(Bc, key=lambda b: b[3]):  # balcony far from every sill line: own floor
        if not floors or min(gapy(b, f["bottom"]) for f in floors) > 0.5 * fp:
            c = (b[1] + b[3]) / 2; floors.append(dict(yc=c - mh / 2, bottom=c, wins=[]))
    for b in sorted(Oc, key=lambda b: b[3]):  # oriel containing no window-row centre: own floor at its centre
        if not any(b[1] <= f["yc"] <= b[3] for f in floors):
            c = (b[1] + b[3]) / 2; floors.append(dict(yc=c, bottom=c + mh / 2, wins=[]))
    minw = 0.5 * mw if os.environ.get("MIN_BAY_W", "1") == "1" else 0.0  # a fragment narrower than half a window adds no bay
    for b in sorted(boxes, key=lambda b: b[0] + b[2]):  # covers no bay by half: own bay
        if b[2] - b[0] >= minw and not any(lf.cover_1d(B["span"], [(b[0], b[2])]) >= 0.5 for B in bays):
            bays.append(dict(xc=(b[0] + b[2]) / 2, span=(b[0], b[2]), wins=[]))
    floors.sort(key=lambda f: f["yc"]); bays.sort(key=lambda B: B["xc"])
    f_of = {i: k for k, f in enumerate(floors) for i in f["wins"]}; b_of = {i: k for k, B in enumerate(bays) for i in B["wins"]}
    bottoms = [f["bottom"] for f in floors]; ycs = [f["yc"] for f in floors]; bspan = [B["span"] for B in bays]
    win = {}
    for i in range(len(W)): win.setdefault((f_of[i], b_of[i]), []).append(i)
    covered = lambda b: [k for k, sp in enumerate(bspan) if lf.cover_1d(sp, [(b[0], b[2])]) >= 0.5]
    bal = {(nearest((b[1] + b[3]) / 2, bottoms), k) for b in Bc for k in covered(b)}
    ori = {(f, k) for b in Oc for k in covered(b) for f, y in enumerate(ycs) if b[1] <= y <= b[3]}
    return dict(fy=ycs, bx=[B["xc"] for B in bays], win=win, bal=bal, ori=ori, nf=len(floors), nb=len(bays),
                bottoms=bottoms, bspan=bspan)


def match_pos(p, r, tol):
    """One-to-one pairing of photo / render positions (after alignment) by ascending distance within tol."""
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(p) for j, y in enumerate(r) if abs(x - y) <= tol)
    out, used = {}, set()
    for _, i, j in pairs:
        if i in out or j in used: continue
        out[i] = j; used.add(j)
    return out

def match_seq(p, r, p_ext, r_ext):
    """Order-preserving subsequence matching of photo / render positions, each normalised to its own extent
    (no global alignment). DP minimises sum of matched |p - r| + gap cost for every unmatched position;
    gap = half the photo's normalised pitch, so a pair is matched only if closer than one pitch."""
    if not p or not r: return {}
    pn = [(x - p_ext[0]) / max(1e-6, p_ext[1] - p_ext[0]) for x in p]
    rn = [(x - r_ext[0]) / max(1e-6, r_ext[1] - r_ext[0]) for x in r]
    g = 0.5 * (float(np.median(np.diff(pn))) if len(pn) > 1 else 0.5)
    n, m = len(pn), len(rn); D = np.zeros((n + 1, m + 1)); B = np.zeros((n + 1, m + 1), int)
    D[1:, 0] = g * np.arange(1, n + 1); D[0, 1:] = g * np.arange(1, m + 1); B[1:, 0] = 1; B[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = [D[i - 1, j - 1] + abs(pn[i - 1] - rn[j - 1]), D[i - 1, j] + g, D[i, j - 1] + g]
            B[i, j] = int(np.argmin(c)); D[i, j] = min(c)
    out, i, j = {}, n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and B[i, j] == 0:
            if abs(pn[i - 1] - rn[j - 1]) < 2 * g: out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or B[i, j] == 1): i -= 1
        else: j -= 1
    return out

def dp_order(p, r, g, wc=None):
    """Order-preserving matching of p and r (same frame); unmatched costs g each, a pair |p-r| (+ wc[i][j], a
    width term in the same units; taken if < 2g)."""
    n, m = len(p), len(r); D = np.zeros((n + 1, m + 1)); B = np.zeros((n + 1, m + 1), int)
    D[1:, 0] = g * np.arange(1, n + 1); D[0, 1:] = g * np.arange(1, m + 1); B[1:, 0] = 1; B[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pc = abs(p[i - 1] - r[j - 1]) + (wc[i - 1][j - 1] if wc is not None else 0.0)
            c = [D[i - 1, j - 1] + min(pc, 2 * g), D[i - 1, j] + g, D[i, j - 1] + g]
            B[i, j] = int(np.argmin(c)); D[i, j] = min(c)
    out, i, j = {}, n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and B[i, j] == 0:
            if abs(p[i - 1] - r[j - 1]) + (wc[i - 1][j - 1] if wc is not None else 0.0) < 2 * g: out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or B[i, j] == 1): i -= 1
        else: j -= 1
    return out, D[n, m]

def match_refit(p, r0, start, g, iters=5, wc=None):
    iters = 1 if len(start) > 50 else iters  # exhaustive starts: one refit step each is enough
    """Subsequence matching with a refitted 1-D scale + shift: start from each initial mapping of the render
    positions r0, match in order (dp_order), refit scale / shift on the matched pairs, repeat; keep the result
    with the lowest DP cost. start: list of (scale, shift) candidates."""
    best = None
    for sc, sh in start:
        for _ in range(iters):
            rm = [sc * x + sh for x in r0]; m, cost = dp_order(p, rm, g, wc)
            if len(m) >= 2:
                xs = np.array([r0[j] for j in m.values()]); ys = np.array([p[i] for i in m])
                if np.ptp(xs) > 1e-6:
                    sc2, sh2 = np.polyfit(xs, ys, 1)
                    if sc2 > 0: sc, sh = float(sc2), float(sh2)
        rm = [sc * x + sh for x in r0]; m, cost = dp_order(p, rm, g, wc)
        if best is None or cost < best[1]: best = (m, cost)
    return best[0] if best else {}

def dp_groups(pc, ps, rc, rs, g, mpen):
    """Ordered matching with groups: 1:1, 1 photo bay : 2 adjacent render bays, 2 adjacent photo bays : 1 render bay,
    or skip (cost g). A group's position is the centre of its union span; a merge is allowed only when the two union
    spans overlap by >= 50 % of the narrower one; merges pay mpen. Positions / spans in the photo frame."""
    def union(sp, a, b): return (min(x[0] for x in sp[a:b]), max(x[1] for x in sp[a:b]))
    def ov(u, v): return max(0.0, min(u[1], v[1]) - max(u[0], v[0])) / max(1e-6, min(u[1] - u[0], v[1] - v[0]))
    n, m = len(pc), len(rc); INF = 1e18
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0; B = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i, j] >= INF: continue
            cands = []
            if i < n: cands.append((i + 1, j, g, None))
            if j < m: cands.append((i, j + 1, g, None))
            for a, b in ((1, 1), (1, 2), (2, 1)):
                if i + a > n or j + b > m: continue
                up, ur = union(ps, i, i + a), union(rs, j, j + b)
                if (a, b) != (1, 1) and ov(up, ur) < 0.5: continue
                c = abs((up[0] + up[1]) / 2 - (ur[0] + ur[1]) / 2) + (0 if (a, b) == (1, 1) else mpen)
                if c >= 2 * g: continue
                cands.append((i + a, j + b, c, (list(range(i, i + a)), list(range(j, j + b)))))
            for ni, nj, c, grp in cands:
                if D[i, j] + c < D[ni, nj]: D[ni, nj] = D[i, j] + c; B[(ni, nj)] = (i, j, grp)
    out, cur = [], (n, m)
    while cur != (0, 0):
        i, j, grp = B[cur]
        if grp: out.append(grp)
        cur = (i, j)
    return out[::-1], D[n, m]

def match_groups(pc, ps, r0, rs0, start, g, iters=3):
    best = None
    for sc, sh in start:
        for it in range(iters + 1):
            rc = [sc * x + sh for x in r0]; rs = [(sc * a + sh, sc * b + sh) for a, b in rs0]
            grp, cost = dp_groups(pc, ps, rc, rs, g, 0.25 * g)
            if it == iters or len(grp) < 2: break
            xs = np.array([np.mean([(rs0[j][0] + rs0[j][1]) / 2 for j in gr]) for _, gr in grp])
            ys = np.array([np.mean([(ps[i][0] + ps[i][1]) / 2 for i in gp]) for gp, _ in grp])
            if np.ptp(xs) < 1e-6: break
            sc2, sh2 = np.polyfit(xs, ys, 1)
            if sc2 <= 0: break
            sc, sh = float(sc2), float(sh2)
        if best is None or cost < best[1]: best = (grp, cost)
    return best[0] if best else []


def pitch(v, fb): return float(np.median(np.diff(v))) if len(v) > 1 else fb


def no_roof(B, W):
    """Balcony rule: railings in front of openings. Drop boxes whose centre lies above the top window row or below
    the lowest one (roof parapets, cornice and plinth rails, ground fences). Same for photo and renders."""
    if not W or os.environ.get("ROOF_FILTER", "1") != "1": return B
    top, bot = min(w[1] for w in W), max(w[3] for w in W)
    return [b for b in B if top <= (b[1] + b[3]) / 2 <= bot]

def upper_only(O, W):
    """Oriel rule: an oriel sits on the upper floors. Drop rendered oriel boxes that reach down to the lowest window
    row (risalits, entrance portals) or whose centre lies above the top window row (dormers). Both methods."""
    if not W or os.environ.get("ORIEL_UPPER", "1") != "1": return O
    rows = sorted(_bands(W, 1, 0.5 * float(np.median([b[3] - b[1] for b in W]))), key=lambda r: lf.ycentre(W, r))
    top = min(W[i][1] for i in rows[0]); low_c = lf.ycentre(W, rows[-1])
    return [b for b in O if b[3] < low_c and (b[1] + b[3]) / 2 >= top]

def score(gW, gU, gB, gI, gO, rw, rb, ro, fmap):
    gB, rb = no_roof(gB, gW), no_roof(rb, rw)
    ro = upper_only(ro, rw)
    P = grid(gW, gB, gO)
    R = grid(rw, rb, ro) if (rw or rb or ro) else None
    if R:
        R["fy0"], R["bx0"] = list(R["fy"]), list(R["bx"])
    if R:  # render grid built in render coordinates, positions placed on the photo for matching
        R["fy"] = [fmap([0, y, 1, y + 1e-3])[1] for y in R["fy"]]
        R["bx"] = [fmap([x, 0, x + 1e-3, 1])[0] for x in R["bx"]]
    fp_ = 0.5 * pitch(P["fy"], 2 * float(np.median([b[3] - b[1] for b in gW])))
    bp_ = 0.5 * pitch(P["bx"], 2 * float(np.median([b[2] - b[0] for b in gW])))
    ext = lambda W_: ((min(b[0] for b in W_), max(b[2] for b in W_)), (min(b[1] for b in W_), max(b[3] for b in W_)))
    def starts(axis, P_ext, R_ext, mapped0, raw0):
        c = []
        if len(raw0) >= 2 and np.ptp(raw0) > 1e-6:  # the global alignment, as a linear map of raw positions
            sc, sh = np.polyfit(raw0, mapped0, 1); c.append((float(sc), float(sh)))
        sc = (P_ext[1] - P_ext[0]) / max(1e-6, R_ext[1] - R_ext[0]); c.append((sc, P_ext[0] - sc * R_ext[0]))  # extents
        return c
    if R and os.environ.get("FLOOR_MATCH") == "refit":
        fm = match_refit(P["fy"], R["fy0"], starts(1, ext(gW)[1], ext(rw)[1], R["fy"], R["fy0"]), fp_)
    elif R and os.environ.get("FLOOR_MATCH") == "seq":
        fm = match_seq(P["fy"], R["fy0"], ext(gW)[1], ext(rw)[1])
    else:
        fm = dict(lf.match_floors(P["fy"], R["fy"], fp_)) if R else {}
    if R and os.environ.get("BAY_MATCH", "m2m") == "refit":
        wc = None
        if os.environ.get("BAY_WIDTH"):  # width term: |log| ratio of bay widths, each image normalised by its median
            al = float(os.environ["BAY_WIDTH"])
            pw = np.array([b[1] - b[0] for b in P["bspan"]]); rw_ = np.array([b[1] - b[0] for b in R["bspan"]])
            pw, rw_ = pw / max(1e-6, np.median(pw)), rw_ / max(1e-6, np.median(rw_))
            wc = [[al * bp_ * abs(np.log(max(1e-3, a_) / max(1e-3, b__))) for b__ in rw_] for a_ in pw]
        st = starts(0, ext(gW)[0], ext(rw)[0], R["bx"], R["bx0"])
        if os.environ.get("BAY_ANCHORS") == "1":  # exhaustive starts: every pair of anchor matches (i1<i2, j1<j2)
            px_, rx_ = P["bx"], R["bx0"]
            for i1 in range(len(px_)):
                for i2 in range(i1 + 1, len(px_)):
                    for j1 in range(len(rx_)):
                        for j2 in range(j1 + 1, len(rx_)):
                            sc_ = (px_[i2] - px_[i1]) / max(1e-6, rx_[j2] - rx_[j1])
                            if sc_ > 0: st.append((sc_, px_[i1] - sc_ * rx_[j1]))
        if os.environ.get("BAY_SIZESCALE") == "1":  # scale from window widths, every bay-to-bay anchor as shift
            ss = float(np.median([b_[2] - b_[0] for b_ in gW])) / max(1e-6, float(np.median([b_[2] - b_[0] for b_ in rw])))
            st += [(ss, px - ss * rx) for px in P["bx"] for rx in R["bx0"]]
        bm = match_refit(P["bx"], R["bx0"], st, bp_, wc=wc)
    elif R and os.environ.get("BAY_MATCH", "m2m") == "m2m":
        st = starts(0, ext(gW)[0], ext(rw)[0], R["bx"], R["bx0"])
        ss = float(np.median([b_[2] - b_[0] for b_ in gW])) / max(1e-6, float(np.median([b_[2] - b_[0] for b_ in rw])))
        st += [(ss, px - ss * rx) for px in P["bx"] for rx in R["bx0"]]
        groups = match_groups(P["bx"], P["bspan"], R["bx0"], R["bspan"], st, bp_)
        bm = {}; Rb = {}; GR = [(list(gp), list(gr)) for gp, gr in groups]
        for gp, gr in groups:
            for i in gp: Rb[i] = list(gr)
    elif R and os.environ.get("BAY_MATCH", "m2m") == "seq":
        bm = match_seq(P["bx"], R["bx0"], ext(gW)[0], ext(rw)[0])
    else:
        bm = (match_pos(P["bx"], R["bx"], bp_) if os.environ.get("BAY_MATCH", "m2m") == "pos" else dict(lf.match_floors(P["bx"], R["bx"], bp_))) if R else {}
    if not (R and os.environ.get("BAY_MATCH", "m2m") == "m2m"):
        Rb = {k: [v] for k, v in bm.items()}; GR = [([k], [v]) for k, v in sorted(bm.items())]
    GR = GR + [([k], []) for k in range(P["nb"]) if k not in Rb]  # unmatched photo bays: groups without partner
    matched_r = {j for v in Rb.values() for j in v}
    def dc_win(c):
        ws = P["win"].get(c, []); return bool(ws) and all(gU[i] for i in ws)
    def dc_bal(c):
        f, k = c
        return any(nearest((g[1] + g[3]) / 2, P["bottoms"]) == f and lf.cover_1d(P["bspan"][k], [(g[0], g[2])]) >= 0.5 for g in gI)
    res = {}
    global LAST
    LAST = dict(P=P, R=R, fm=dict(fm), GR=GR, gB=gB, rb=rb)
    for kind, dc in (("win", dc_win), ("bal", dc_bal), ("ori", lambda c: False)):
        po = set(P[kind]); ro_ = set(R[kind]) if R else set()
        T = dict(tp=0, fn=0, fp=0, tn=0)
        for f in range(P["nf"]):
            for gp, gr in GR:  # one comparison per floor and bay group (a group is 1 or 2 bays on each side)
                cs = [(f, k) for k in gp]
                if all(dc(c) and (kind == "win" or c not in po) for c in cs): continue
                hp = any(c in po for c in cs)
                hr = any((fm[f], j) in ro_ for j in gr) if f in fm else False
                T["tp" if hp and hr else "fn" if hp else "fp" if hr else "tn"] += 1
        inv_f, inv_b = set(fm.values()), matched_r
        T["fp"] += sum(1 for c in ro_ if not (c[0] in inv_f and c[1] in inv_b))
        res[kind] = T
    return res, (P["nf"], R["nf"] if R else 0, len(fm)), (P["nb"], R["nb"] if R else 0, len(Rb))


def astra_oriels(t, stem, aw, protrude=0.3, min_h=0.6, opening=0.4):
    """oriel_objects_current.py extraction, boxes kept in Astra's render frame."""
    D = json.loads((A / "oriel_depth" / f"{t}.json").read_text()); N = D["grid_n"]
    G = np.array([[np.nan if x is None else D["wall"] - x for x in row] for row in D["grid"]])
    maps = {r["test_id"]: r for r in json.load(open(A / "light/mapping.json")) + json.load(open(A / "light/output/mapping.json"))}
    lab = A / ("light_labels2/labels" if maps[t]["astra_folder"].startswith("facade_") else "light_labels/labels")
    Wr, Hr = Image.open(next((lab.parent / "photos").glob(f"{stem}.*"))).size
    rows = _bands(aw, 1, 0.5 * float(np.median([b[3] - b[1] for b in aw])))
    rc = [np.mean([(aw[i][1] + aw[i][3]) / 2 for i in rr]) for rr in rows]
    p = float(np.median(np.diff(rc))) if len(rc) > 1 else Hr / 5
    mask = np.nan_to_num(G, nan=-9) > protrude
    k = max(1, int(round(opening * p / Hr * N)))
    mask = ndimage.binary_opening(mask, structure=np.ones((k, 1), bool))
    lbl, n = ndimage.label(mask); out = []
    for k in range(1, n + 1):
        ys, xs = np.where(lbl == k)
        y0, y1, x0, x1 = ys.min() / N * Hr, (ys.max() + 1) / N * Hr, xs.min() / N * Wr, (xs.max() + 1) / N * Wr
        if y1 - y0 < min_h * p: continue
        if not any(x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1 and
                   lbl[min(N - 1, int((b[1] + b[3]) / 2 / Hr * N)), min(N - 1, int((b[0] + b[2]) / 2 / Wr * N))] == k for b in aw):
            continue
        out.append([x0, y0, x1, y1])
    return out, lab


def main():
    gt = ev.load_gt(ev.TESTSET / "labels")
    bgt = {json.loads(l)["stem"]: json.loads(l) for l in (LBL / "balcony_gt_boxes.jsonl").open()}
    ol = {json.loads(l)["stem"]: json.loads(l) for l in (LBL / "oriel_labels.jsonl").open()}
    tids = json.load(open(A / "tids46.json"))["tids"]
    rule = sys.argv[1] if len(sys.argv) > 1 else "medium"  # Astra balcony name rule (medium = most favourable to Astra)
    kw = re.compile(RULES[rule], re.I)
    tot = {m: {k: dict(tp=0, fn=0, fp=0, tn=0) for k in ("win", "bal", "ori")} for m in ("ours_cur", "astra")}
    per = {}
    for t in tids:
        stem = gt[t]["stem"]; ws = gt[t]["windows"]
        gW = [w["box"] for w in ws]; gU = [w["unusable"] for w in ws]; gb = [w["box"] for w in ws if not w["unusable"]]
        B = bgt[stem]; gB, gI = B["boxes"], B.get("ignore") or []
        fy, bx = dsl_bands(json.loads((LBL / "window_grid" / stem / "facade_dsl.json").read_text()))
        gO = [[bx[b][0], fy[f][0], bx[b][1], fy[f][1]] for f, b in map(tuple, ol[stem].get("cells") or []) if f in fy and b in bx]
        per[t] = {}
        for m in ("ours_cur", "astra"):
            rw = [x["box_xyxy"] for x in json.loads((A / "eval_layout_dets" / m / f"{stem}.json").read_text())["windows"]]
            h = json.load(open(HM / ("ours" if m == "ours_cur" else "astra") / f"{t}.json"))
            GX = 0.5 * float(np.median([b_[2] - b_[0] for b_ in rw])) if os.environ.get("BOX_MERGE") == "row" else None
            if m == "ours_cur":
                rb = object_boxes(h, {i for i, (n, c) in enumerate(zip(h["names"], h["colls"])) if any(x.startswith("Balconies") for x in c) and "enclosed" not in n}, key=lambda n: re.match(r"(Balc_\d+)", n).group(1) if re.match(r"Balc_\d+", n) else n, gapx=GX)
                ro = object_boxes(h, {i for i, n in enumerate(h["names"]) if "balc_enclosed" in n}, key=lambda n: re.match(r"(Balc_\d+)", n).group(1) if re.match(r"Balc_\d+", n) else n)
            else:
                sel = os.environ.get("ASTRA_SEL", "name")
                NOT = re.compile(r"window|glaz|mullion|transom|reveal|sill|door|frame|cornice|coping|fascia|wall|shutter|pane|lintel|arch|pilaster|column|string|course|plinth|base course|roof(?!.*terrace)", re.I)
                if sel == "name":  # object name only
                    pick = {i for i, n in enumerate(h["names"]) if kw.search(n)}
                elif sel == "namecoll_not":  # name, or collection unless the object name says it is something else
                    pick = {i for i, (n, c) in enumerate(zip(h["names"], h["colls"])) if kw.search(n) or (any(kw.search(x) for x in c) and not NOT.search(n))}
                else:
                    pick = {i for i, (n, c) in enumerate(zip(h["names"], h["colls"])) if kw.search(n) or any(kw.search(x) for x in c)}
                rb = object_boxes(h, pick, gapx=GX)
                maps = {r["test_id"]: r for r in json.load(open(A / "light/mapping.json")) + json.load(open(A / "light/output/mapping.json"))}
                lab = A / ("light_labels2/labels" if maps[t]["astra_folder"].startswith("facade_") else "light_labels/labels")
                aw = [w["box"] for w in ev.load_gt(lab)[t]["windows"]]
                ro, _ = astra_oriels(t, stem, aw, protrude=float(os.environ.get("ORIEL_PROTRUDE", 0.3)))
                if os.environ.get("ASTRA_ORIEL_NAMES", "1") == "1":  # add objects named as oriels / projecting bays
                    okw = re.compile(r"oriel|projecting bay|erker|bow window", re.I)
                    ro = ro + object_boxes(h, {i for i, n in enumerate(h["names"]) if okw.search(n)})
            a, b_ = init_alignment(rw, gb); a, b_ = refine_alignment(rw, gb, a, b_); fmap = grid_map(rw, gb, a, b_)
            res, fl, ba = score(gW, gU, gB, gI, gO, rw, rb, ro, fmap)
            per[t][m] = {k: res[k] for k in res}; per[t][m]["floors"] = fl; per[t][m]["bays"] = ba
            for k in res:
                for x in res[k]: tot[m][k][x] += res[k][x]
        print(t, {m: {k: (per[t][m][k]["tp"], per[t][m][k]["fn"], per[t][m][k]["fp"]) for k in ("win", "bal", "ori")} for m in per[t]}, flush=True)
    print(f"\nAstra balcony name rule: {rule}")
    for m in tot:
        for k, nm in (("win", "window"), ("bal", "balcony"), ("ori", "oriel")):
            c = tot[m][k]; Pp = c["tp"] / max(1, c["tp"] + c["fp"]); Rr = c["tp"] / max(1, c["tp"] + c["fn"])
            print(f"{m:9s} {nm:8s} TP {c['tp']:4d} FN {c['fn']:4d} FP {c['fp']:4d}  P {Pp:.3f} R {Rr:.3f} F1 {2 * c['tp'] / max(1, 2 * c['tp'] + c['fn'] + c['fp']):.3f}")
    json.dump({"total": tot, "per": per}, open(paths.OUT_DIR / f"layout_grid_46_{rule}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
