"""2D t-SNE of frozen DINOv2 features, colored by floor shape.

Uses labeled crops under ``open_work``, ``surface_panel``, and ``solid``.
Real (``cmp*``) and synth (``flux2*``) are both included. Crops with no
``floor_shape`` are skipped. The plot uses backbone features, not the
trained floor head.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/tsne_floor_shape.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import load_labels  # noqa: E402
from balcony_train.labels import CLASSES, FLOOR_SHAPES, LABELED_IMAGE_SUFFIXES  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL  # noqa: E402
from balcony_train.tsne_real_vs_synth import (  # noqa: E402
    extract_features,
    run_tsne,
    source_label,
)

DEFAULT_OUT = ROOT / "runs" / "balcony_clf" / "tsne_floor_shape.png"

FLOOR_COLORS = {
    "rectangle": "#1f77b4",
    "triangle": "#ff7f0e",
    "circle": "#2ca02c",
    "hexagon": "#9467bd",
    "trapezoid": "#d62728",
}


def iter_floor_rows(
    crops_dir: Path,
    labels_path: Path,
) -> list[tuple[Path, str, str, str]]:
    """Return (path, source, kind, floor_shape) for labeled real and synth crops."""
    crops_dir = Path(crops_dir)
    labels = load_labels(labels_path)
    rows: list[tuple[Path, str, str, str]] = []
    for kind in CLASSES:
        folder = crops_dir / kind
        if not folder.is_dir():
            continue
        for suffix in LABELED_IMAGE_SUFFIXES:
            for pattern in (f"*{suffix}", f"*{suffix.upper()}"):
                for path in folder.glob(pattern):
                    if not path.is_file():
                        continue
                    src = source_label(path)
                    if src is None:
                        continue
                    floor = (labels.get(path.stem) or {}).get("floor_shape")
                    if floor not in FLOOR_SHAPES:
                        continue
                    rows.append((path.resolve(), src, kind, str(floor)))
    uniq: dict[Path, tuple[Path, str, str, str]] = {}
    for row in rows:
        uniq[row[0]] = row
    return sorted(uniq.values(), key=lambda r: (r[3], r[1], r[2], r[0].name.lower()))


def plot_floor_embedding(
    emb: np.ndarray,
    sources: list[str],
    floors: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for floor in FLOOR_SHAPES:
        for src, marker in (("real", "o"), ("synth", "^")):
            mask = np.array(
                [(f == floor and s == src) for f, s in zip(floors, sources)],
                dtype=bool,
            )
            if not bool(mask.any()):
                continue
            pts = emb[mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=FLOOR_COLORS.get(floor, "#333333"),
                marker=marker,
                s=36,
                alpha=0.75,
                label=f"{floor}/{src} (n={int(mask.sum())})",
            )
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA dims before t-SNE (0 = skip PCA)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = iter_floor_rows(args.crops_dir, args.labels)
    if len(rows) < 2:
        print(
            f"need at least 2 crops with floor_shape under {args.crops_dir}",
            file=sys.stderr,
        )
        return 1
    n_real = sum(1 for _, src, _, _ in rows if src == "real")
    n_synth = sum(1 for _, src, _, _ in rows if src == "synth")
    print(f"floor-labeled real={n_real} synth={n_synth} under {args.crops_dir}")
    for floor in FLOOR_SHAPES:
        n = sum(1 for *_, shape in rows if shape == floor)
        print(f"  {floor}={n}")
    if n_real == 0 or n_synth == 0:
        print("warn: one of real/synth is empty; plot still uses the other")

    device = torch.device(args.device)
    feats = extract_features(
        [(path, src, kind) for path, src, kind, _floor in rows],
        device=device,
        batch_size=int(args.batch_size),
    )
    emb = run_tsne(
        feats,
        n_components=2,
        perplexity=float(args.perplexity),
        seed=int(args.seed),
        pca_dim=int(args.pca_dim),
    )
    plot_floor_embedding(
        emb,
        [src for _, src, _, _ in rows],
        [floor for *_, floor in rows],
        Path(args.out),
        title="DINOv2 ViT-S/14 — floor shape (color), real=o synth=^",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
