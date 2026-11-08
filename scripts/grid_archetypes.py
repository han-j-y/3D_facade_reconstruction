#!/usr/bin/env python3
"""Floor archetypes and bay symbols with overrides, from a recovery DSL (prototype).

A cell's value is everything in it: window / door token(s) with span markers, the open balcony type on it and an
oriel flag. Bays and floors are grouped by majority, alternately, until stable:

  * a column joins a symbol when it agrees with the symbol's majority column on at least ``t`` of the floors,
  * a row joins an archetype when it agrees with the archetype's majority row on at least ``t`` of the bays,
  * agreement is read on the grid denoised by the other grouping (the per-(archetype, symbol) majority), so one
    missed window does not split its column and, through it, every floor.

The facade is then two order lists of names plus one table cell per (archetype, symbol); every cell where the
recovered grid differs from its table value is an override on that floor. Expanding gives the recovered grid back
exactly (checked). ``t = 1.0`` is exact matching, as in ``facade_template.py``.

  python scripts/grid_archetypes.py --dsl runs_merged/r25/balcony/facade_dsl_with_balconies.json --t 0.8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

EMPTY = (None, "", "empty", "spacer", "s", "null")


def cell_grid(dsl: dict) -> list[list[str]]:
    """Recovered grid, one string per (floor row, bay column)."""
    lay = dsl["layout"]
    floors, bays = lay["floors"], lay["bays"]
    n_f, n_b = len(floors), len(bays)
    row_of = {int(f["id"]): i for i, f in enumerate(floors)}
    col_of = {int(b["id"]): j for j, b in enumerate(bays)}
    place = lay.get("placement") or []
    spans = lay.get("placement_spans") or []
    win = [["" for _ in range(n_b)] for _ in range(n_f)]
    for r in range(n_f):
        c = 0
        while c < n_b:
            tok = place[r][c] if r < len(place) and c < len(place[r]) else None
            toks = [t for t in (tok if isinstance(tok, list) else [tok]) if t not in EMPTY]
            sp = int(spans[r][c]) if r < len(spans) and c < len(spans[r]) and spans[r][c] else 1
            if toks:
                win[r][c] = "+".join(str(t) for t in toks) + (f"*{sp}" if sp > 1 else "")
                for k in range(1, sp):
                    if c + k < n_b:
                        win[r][c + k] = "^"
                c += max(1, sp)
            else:
                c += 1
    balc = [["" for _ in range(n_b)] for _ in range(n_f)]
    for b in lay.get("balconies") or []:
        r = row_of.get(int(b["floor"]))
        c0, c1 = col_of.get(int(b["bay_start"])), col_of.get(int(b["bay_end"]))
        if r is None or c0 is None or c1 is None:
            continue
        mark = "o" if "enclosed" in str(b["type"]) else "b:" + str(b["type"]).replace("balc_type_", "T")
        for c in range(min(c0, c1), max(c0, c1) + 1):
            balc[r][c] = mark if not balc[r][c] else balc[r][c] if balc[r][c] == mark else balc[r][c] + "&" + mark
    return [[(win[r][c] or "-") + ("|" + balc[r][c] if balc[r][c] else "") for c in range(n_b)] for r in range(n_f)]


def _majority(values: list[str]) -> str:
    cnt = Counter(values)
    best = max(cnt.values())
    return next(v for v in values if cnt[v] == best)  # ties: first in order


def _group(lines: list[list[str]], t: float) -> list[int]:
    """Greedy, in order: a line joins the first group whose majority line it matches on >= t of positions."""
    groups: list[list[int]] = []
    out: list[int] = []
    for i, line in enumerate(lines):
        best, best_s = -1, -1.0
        for g, members in enumerate(groups):
            rep = [_majority([lines[m][k] for m in members]) for k in range(len(line))]
            s = sum(a == b for a, b in zip(line, rep)) / max(1, len(line))
            if s >= t - 1e-9 and s > best_s:
                best, best_s = g, s
        if best < 0:
            groups.append([i])
            out.append(len(groups) - 1)
        else:
            groups[best].append(i)
            out.append(best)
    return out


def _table(grid: list[list[str]], arch: list[int], sym: list[int]) -> dict[tuple[int, int], str]:
    cells: dict[tuple[int, int], list[str]] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            cells.setdefault((arch[r], sym[c]), []).append(v)
    return {k: _majority(v) for k, v in cells.items()}


def fit(grid: list[list[str]], t: float, iters: int = 6) -> tuple[list[int], list[int], dict]:
    n_f, n_b = len(grid), len(grid[0]) if grid else 0
    cols = [[grid[r][c] for r in range(n_f)] for c in range(n_b)]
    sym = _group(cols, t)
    arch = _group(grid, t)
    for _ in range(iters):
        tab = _table(grid, arch, sym)
        den = [[tab[(arch[r], sym[c])] for c in range(n_b)] for r in range(n_f)]
        # regroup each axis on the grid denoised by the other one, but keep the raw cells where they agree
        new_sym = _group([[grid[r][c] if grid[r][c] == den[r][c] else den[r][c] for r in range(n_f)] for c in range(n_b)], t)
        new_arch = _group([[grid[r][c] if grid[r][c] == den[r][c] else den[r][c] for c in range(n_b)] for r in range(n_f)], t)
        if new_sym == sym and new_arch == arch:
            break
        sym, arch = new_sym, new_arch
    return arch, sym, _table(grid, arch, sym)


def names(grid: list[list[str]], arch: list[int], sym: list[int]) -> tuple[list[str], list[str]]:
    n_f, n_b = len(arch), len(sym)
    cnt = Counter(arch)
    typical = max(cnt, key=lambda a: (cnt[a], -arch.index(a))) if arch else None
    fname: dict[int, str] = {}
    if typical is not None and cnt[typical] > 1:
        fname[typical] = "typical"
    if arch and arch[0] not in fname:
        fname[arch[0]] = "top"
    if arch and arch[-1] not in fname:
        fname[arch[-1]] = "ground"
    k = 1
    for a in arch:
        if a not in fname:
            k += 1
            fname[a] = f"mid{k}"
    door = lambda c: any(v.startswith("d") and v[1:3].isdigit() for v in (grid[r][c] for r in range(n_f)))
    sname: dict[int, str] = {}
    if n_b >= 3 and sym[0] == sym[-1]:
        sname[sym[0]] = "E"
    door_syms = {sym[c] for c in range(n_b) if door(c)} - set(sname)
    if len(door_syms) == 1:
        sname[door_syms.pop()] = "D"
    letters = iter("ABCFGHIJKLMNOPQRSTUVWXYZ")
    for s in sym:
        if s not in sname:
            sname[s] = next(letters)
    return [fname[a] for a in arch], [sname[s] for s in sym]


def compact(dsl: dict, t: float) -> dict[str, Any]:
    grid = cell_grid(dsl)
    arch, sym, tab = fit(grid, t)
    fn, sn = names(grid, arch, sym)
    n_f, n_b = len(arch), len(sym)
    overrides = [{c: grid[r][c] for c in range(n_b) if grid[r][c] != tab[(arch[r], sym[c])]} for r in range(n_f)]
    # lossless check
    back = [[overrides[r].get(c, tab[(arch[r], sym[c])]) for c in range(n_b)] for r in range(n_f)]
    assert back == grid
    table = {}
    for (a, s), v in tab.items():
        table.setdefault(fn[arch.index(a)], {})[sn[sym.index(s)]] = v
    n_ov = sum(len(o) for o in overrides)
    return {
        "floors": fn,
        "bays": sn,
        "table": table,
        "overrides": {f"{r}:{fn[r]}": o for r, o in enumerate(overrides) if o},
        "n_floors": n_f,
        "n_bays": n_b,
        "n_archetypes": len(set(arch)),
        "n_symbols": len(set(sym)),
        "n_overrides": n_ov,
        # words: order lists + table values (+ one size per archetype / symbol) + overrides (column, value)
        "words": n_f + n_b + len(tab) + len(set(arch)) + len(set(sym)) + 2 * n_ov,
        "words_grid_now": 2 * n_f * n_b + 4 * (n_f + n_b),  # placement + spans cells, floor / bay entries
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsl", type=Path, required=True)
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = compact(json.loads(a.dsl.read_text(encoding="utf-8")), a.t)
    txt = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
