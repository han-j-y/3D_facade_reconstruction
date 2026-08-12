"""Standalone facade recovery package (photo → DSL; no renderer dependency)."""

from facade_recovery.paths import (
    PKG_ROOT,
    default_structure_ckpt,
    default_train_up,
    default_vocab,
)

__all__ = [
    "PKG_ROOT",
    "default_structure_ckpt",
    "default_train_up",
    "default_vocab",
]
