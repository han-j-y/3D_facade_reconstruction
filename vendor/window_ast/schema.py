"""Special tokens and vocabulary helpers for AST sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS, UNK)

# Section / structural keywords emitted by the serializer
KEYWORDS: tuple[str, ...] = (
    "BOUNDARY",
    "FRAME",
    "REGION",
    "PROGRAM",
    "OP",
    "OUTPUT",
    "PANES",
    "RULE",
)


class Vocab:
    """String token ↔ integer id map."""

    def __init__(self, tokens: Iterable[str]) -> None:
        ordered: list[str] = []
        seen: set[str] = set()
        for t in list(SPECIAL_TOKENS) + list(tokens):
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        self.itos: list[str] = ordered
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], *, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> list[str]:
        out: list[str] = []
        special = {self.pad_id, self.bos_id, self.eos_id}
        for i in ids:
            if i == self.eos_id:
                break
            if skip_special and i in special:
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append(UNK)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.itos}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocab:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["tokens"])
