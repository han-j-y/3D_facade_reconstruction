"""Pane-tree constants and token assembly used by the bundled structure model."""

from __future__ import annotations

from typing import Any

MAX_CHILDREN = 8
NODE_CLASSES = ["eps", "grid"] + [f"{axis}{count}" for axis in "vh" for count in range(2, MAX_CHILDREN + 1)]
CLS_INDEX = {name: index for index, name in enumerate(NODE_CLASSES)}
MAX_COUNT = 16
MAX_CUTS = MAX_CHILDREN - 1


def split_prefix(tokens: list[str]) -> tuple[list[str], str | None]:
    if "PANES" in tokens:
        index = tokens.index("PANES")
        return list(tokens[:index]), tokens[index + 1].split("=", 1)[1]
    return [token for token in tokens if token != "END"], None


def parse_prefix_key(key: str) -> tuple[list[str], str | None]:
    prefix, region = key.rsplit(" | ", 1)
    return prefix.split(), None if region == "none" else region


def _child_id(parent: str, index: int) -> str:
    return f"{parent}_{index}"


def _canonical_panes(rules: dict[str, Any], region: str) -> dict[str, Any]:
    return {"region": region, "root": "root", "rules": rules}


def assemble_tokens(prefix: list[str], region: str | None, nodes: list[dict[str, Any]]) -> list[str]:
    """Assemble model output without importing another experiment checkout."""
    if region is None or not nodes:
        return list(prefix) + ["END"]

    rules: dict[str, Any] = {}

    def build(index: int, node_id: str) -> None:
        node = nodes[index]
        cls = NODE_CLASSES[node["cls"]]
        if cls == "eps":
            rules[node_id] = {"op": "eps"}
        elif cls == "grid":
            rules[node_id] = {"op": "grid", "vertical": int(node["V"]), "horizontal": int(node["H"])}
        else:
            children = list(reversed(node["children"]))
            child_ids = [_child_id(node_id, child_index) for child_index in range(len(children))]
            cuts = [round(1.0 - value, 4) for value in reversed(node["cuts"])]
            rules[node_id] = {
                "op": "split",
                "axis": cls[0],
                "children": child_ids,
                "at": cuts[0] if len(cuts) == 1 else cuts,
            }
            for child_id, child in zip(child_ids, children):
                build(child, child_id)

    build(0, "root")
    from window_ast.structure import ir_to_structure_tokens

    shell = {
        "boundary": {"shape": "rectangle"},
        "regions": [],
        "program": [],
        "panes": _canonical_panes(rules, region),
    }
    tokens = ir_to_structure_tokens(shell)
    return list(prefix) + tokens[tokens.index("PANES") :]
