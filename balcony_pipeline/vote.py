"""Majority-vote balcony IR — same algorithm as run_pipeline.vote_cluster_ir,
keyed by balcony_view instead of window structure_view.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from heuristic_ir import balcony_view


def balcony_fingerprint(ir: dict[str, Any], *, profile_name: str | None = None) -> str:
    return json.dumps(
        balcony_view(ir, profile_name=profile_name),
        sort_keys=True,
        separators=(",", ":"),
    )


def vote_cluster_ir(
    member_preds: list[dict[str, Any]],
    *,
    prefer_unit_id: int | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    for p in member_preds:
        ir = p.get("ir")
        if not isinstance(ir, dict) or p.get("parse_error"):
            continue
        try:
            key = balcony_fingerprint(ir, profile_name=profile_name)
        except Exception:
            continue
        valid.append({**p, "structure_key": key})

    tallies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in valid:
        tallies[p["structure_key"]].append(p)

    vote_summary = {
        "n_members": len(member_preds),
        "n_valid": len(valid),
        "n_unique": len(tallies),
        "recovery_profile": profile_name,
        "counts": {
            k: len(v) for k, v in sorted(tallies.items(), key=lambda kv: -len(kv[1]))
        },
    }

    if not valid:
        return {
            "structure_ir": None,
            "structure_tokens": None,
            "vote": {**vote_summary, "winner_key": None, "winner_count": 0},
            "members": member_preds,
        }

    ranked = sorted(tallies.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    best_count = len(ranked[0][1])
    tied = [k for k, vs in ranked if len(vs) == best_count]
    winner_key = tied[0]
    if prefer_unit_id is not None and len(tied) > 1:
        for k in tied:
            if any(int(p["unit_id"]) == int(prefer_unit_id) for p in tallies[k]):
                winner_key = k
                break
    winners = tallies[winner_key]
    chosen = winners[0]
    if prefer_unit_id is not None:
        for p in winners:
            if int(p["unit_id"]) == int(prefer_unit_id):
                chosen = p
                break

    members_out = []
    for p in member_preds:
        key = None
        ir = p.get("ir")
        if isinstance(ir, dict) and not p.get("parse_error"):
            try:
                key = balcony_fingerprint(ir, profile_name=profile_name)
            except Exception:
                key = None
        members_out.append(
            {
                "unit_id": int(p["unit_id"]),
                "structure_key": key,
                "tokens": p.get("tokens"),
                "parse_error": p.get("parse_error"),
                "agrees_with_vote": key is not None and key == winner_key,
            }
        )

    return {
        "structure_ir": chosen.get("ir"),
        "structure_tokens": chosen.get("tokens"),
        "vote": {
            **vote_summary,
            "winner_key": winner_key,
            "winner_count": best_count,
            "winner_unit_id": int(chosen["unit_id"]),
            "unanimous": best_count == len(valid) and len(tallies) == 1,
        },
        "members": members_out,
    }


SOFT_KINDS = ("open_work", "surface_panel", "solid")       # balcony_train.labels.CLASSES order
SOFT_MATERIALS = ("metal", "masonry")                      # MATERIALS order


def soft_type_labels(probs: list[dict], floor_voters: list[int]) -> dict | None:
    """Soft vote (pipeline default since 2026-11-07, user; chosen on val30): kind = argmax of the members' mean
    kind softmax; material = argmax of the material softmax weighted by each member's P(open_work); floor =
    argmax of the mean [rectangle + triangle, circle] over ``floor_voters`` (indices of members that are not
    juliet), None without voters. ``probs``: ``RailingPredictor.predict_probs`` per member. None when a member
    has no probabilities (e.g. external labels): the caller keeps the majority vote."""
    import numpy as np

    if not probs or any(p is None or p.get("kind") is None or p.get("material") is None for p in probs):
        return None
    kind = SOFT_KINDS[int(np.argmax(np.mean([p["kind"] for p in probs], axis=0)))]
    w = np.array([p["kind"][0] for p in probs], dtype=float) + 1e-6
    material = SOFT_MATERIALS[int(np.argmax(np.average([p["material"] for p in probs], axis=0, weights=w)))]
    floor = None
    fv = [i for i in floor_voters if probs[i].get("floor") is not None]
    if fv:
        fl = np.mean([[probs[i]["floor"][0] + probs[i]["floor"][1], probs[i]["floor"][2]] for i in fv], axis=0)
        floor = "rectangle" if fl[0] >= fl[1] else "circle"
    return {"kind": kind, "material": material if kind == "open_work" else None, "floor": floor}


def apply_soft_labels(ir: dict | None, labels: dict | None) -> dict | None:
    """Copy ``ir`` with the soft-voted railing kind / material and (when given) floor shape."""
    import copy

    if not isinstance(ir, dict) or not labels:
        return ir
    out = copy.deepcopy(ir)
    rail = dict(out.get("railing") or {})
    rail["kind"] = labels["kind"]
    rail["material"] = labels["material"]
    out["railing"] = rail
    if labels.get("floor"):
        fl = dict(out.get("floor") or {})
        fl["shape"] = labels["floor"]
        out["floor"] = fl
    return out

