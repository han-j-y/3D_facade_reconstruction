"""Simple SDXL prompts for balcony crops (label solid vs baluster by hand)."""

from __future__ import annotations

NEGATIVE_PROMPT = "blurry, cartoon, illustration"

BALCONY_PROMPTS: tuple[str, ...] = (
    "a photo of a balcony",
    "a photo of an apartment balcony",
    "a photo of a balcony railing",
    "a photo of a concrete balcony",
    "a photo of a glass balcony",
    "a photo of a metal balcony railing",
    "a photo of a stone balcony",
    "a photo of a wooden balcony",
    "a photo of a Juliet balcony",
    "a photo of a projecting balcony",
)

SOLID_PROMPTS = BALCONY_PROMPTS
