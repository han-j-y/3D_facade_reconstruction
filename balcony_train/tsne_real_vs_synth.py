"""t-SNE of frozen DINOv2 features: real facade (cmp*) vs FLUX.2 (flux2_*).

Shows whether backbone embeddings separate synthetic crops from real ones
(domain gap), independent of the trained kind/material heads.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe -m pip install scikit-learn matplotlib
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/tsne_real_vs_synth.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import (  # noqa: E402
    CLASSES,
    LABELED_IMAGE_SUFFIXES,
)
from balcony_train.model import RailingClassifier, load_backbone, make_transform  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402

DEFAULT_OUT = ROOT / "runs" / "balcony_clf" / "tsne_real_vs_synth.png"


def source_label(path: Path) -> str | None:
    """Return ``real`` / ``synth`` from filename, or None if neither."""
    name = Path(path).name.lower()
    if name.startswith("cmp"):
        return "real"
    if name.startswith("flux2"):
        return "synth"
    return None


def iter_crop_paths(crops_dir: Path) -> list[tuple[Path, str, str]]:
    """List (path, source, kind_folder) under kind class folders."""
    crops_dir = Path(crops_dir)
    rows: list[tuple[Path, str, str]] = []
    for kind in CLASSES:
        folder = crops_dir / kind
        if not folder.is_dir():
            continue
        for suffix in LABELED_IMAGE_SUFFIXES:
            for path in folder.glob(f"*{suffix}"):
                if not path.is_file():
                    continue
                src = source_label(path)
                if src is None:
                    continue
                rows.append((path.resolve(), src, kind))
            for path in folder.glob(f"*{suffix.upper()}"):
                if not path.is_file():
                    continue
                src = source_label(path)
                if src is None:
                    continue
                rows.append((path.resolve(), src, kind))
    # de-dupe (Windows case)
    uniq: dict[Path, tuple[Path, str, str]] = {}
    for row in rows:
        uniq[row[0]] = row
    return sorted(uniq.values(), key=lambda r: (r[1], r[2], r[0].name.lower()))


def subsample_balanced(
    rows: list[tuple[Path, str, str]],
    *,
    max_per_source: int,
    seed: int,
) -> list[tuple[Path, str, str]]:
    """Cap each source (real/synth) at ``max_per_source`` (0 = no cap)."""
    if max_per_source <= 0:
        return list(rows)
    rng = random.Random(int(seed))
    by_src: dict[str, list[tuple[Path, str, str]]] = {"real": [], "synth": []}
    for row in rows:
        by_src.setdefault(row[1], []).append(row)
    out: list[tuple[Path, str, str]] = []
    for src, items in by_src.items():
        items = list(items)
        rng.shuffle(items)
        out.extend(items[: int(max_per_source)])
    out.sort(key=lambda r: (r[1], r[2], r[0].name.lower()))
    return out


@torch.no_grad()
def extract_features(
    rows: list[tuple[Path, str, str]],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model = RailingClassifier(pretrained=True, freeze_backbone=True)
    load_backbone(model, device)
    model.eval()
    tfm = make_transform()
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    for i, (path, _, _) in enumerate(rows, start=1):
        img = Image.open(path).convert("RGB")
        batch.append(tfm(img))
        if len(batch) >= batch_size or i == len(rows):
            x = torch.stack(batch, dim=0).to(device)
            f = model.encode(x).float().cpu().numpy()
            feats.append(f)
            batch = []
            if i % 100 == 0 or i == len(rows):
                print(f"features {i}/{len(rows)}", flush=True)
    return np.concatenate(feats, axis=0)


def run_tsne(
    features: np.ndarray,
    *,
    n_components: int,
    perplexity: float,
    seed: int,
    pca_dim: int,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x = np.asarray(features, dtype=np.float64)
    if pca_dim > 0 and x.shape[1] > pca_dim and x.shape[0] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
        print(f"PCA -> {x.shape[1]} dims", flush=True)
    n = x.shape[0]
    perp = float(perplexity)
    if n <= 1:
        raise SystemExit("need at least 2 images for t-SNE")
    # sklearn requires perplexity < n_samples
    perp = min(perp, max(5.0, (n - 1) / 3.0))
    perp = max(2.0, min(perp, float(n - 1)))
    print(f"t-SNE n={n} perplexity={perp:.1f} components={n_components}", flush=True)
    return TSNE(
        n_components=int(n_components),
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    ).fit_transform(x)


def plot_embedding(
    emb: np.ndarray,
    sources: list[str],
    kinds: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_comp = emb.shape[1]
    fig = plt.figure(figsize=(10, 8))
    if n_comp >= 3:
        ax = fig.add_subplot(111, projection="3d")
        for src, marker, color in (
            ("real", "o", "#1f77b4"),
            ("synth", "^", "#ff7f0e"),
        ):
            mask = [s == src for s in sources]
            if not any(mask):
                continue
            pts = emb[mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=color,
                marker=marker,
                s=28,
                alpha=0.75,
                label=f"{src} (n={int(np.sum(mask))})",
            )
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.set_zlabel("t-SNE-3")
    else:
        ax = fig.add_subplot(111)
        for src, marker, color in (
            ("real", "o", "#1f77b4"),
            ("synth", "^", "#ff7f0e"),
        ):
            mask = [s == src for s in sources]
            if not any(mask):
                continue
            pts = emb[mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                marker=marker,
                s=36,
                alpha=0.75,
                label=f"{src} (n={int(np.sum(mask))})",
            )
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path.resolve()}", flush=True)

    # Secondary plot: color by kind, marker by source
    out2 = out_path.with_name(out_path.stem + "_by_kind" + out_path.suffix)
    kind_colors = {
        "open_work": "#2ca02c",
        "surface_panel": "#9467bd",
        "solid": "#d62728",
    }
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    for kind in CLASSES:
        for src, marker in (("real", "o"), ("synth", "^")):
            mask = [(k == kind and s == src) for k, s in zip(kinds, sources)]
            if not any(mask):
                continue
            pts = emb[mask]
            ax2.scatter(
                pts[:, 0],
                pts[:, 1] if n_comp >= 2 else np.zeros(len(pts)),
                c=kind_colors.get(kind, "#333333"),
                marker=marker,
                s=36,
                alpha=0.75,
                label=f"{kind}/{src} (n={int(np.sum(mask))})",
            )
    ax2.set_xlabel("t-SNE-1")
    ax2.set_ylabel("t-SNE-2")
    ax2.set_title(title + " (color=kind, marker=source)")
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out2, dpi=160)
    plt.close(fig2)
    print(f"saved {out2.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--max-per-source",
        type=int,
        default=400,
        help="max images per source real/synth (0 = use all)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument(
        "--components",
        type=int,
        choices=(2, 3),
        default=2,
        help="t-SNE output dims (2 or 3)",
    )
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
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="also write .npz with embedding + labels next to --out",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = iter_crop_paths(args.crops_dir)
    if not rows:
        print(f"no cmp*/flux2* images under {args.crops_dir}", file=sys.stderr)
        return 1
    n_real = sum(1 for _, s, _ in rows if s == "real")
    n_synth = sum(1 for _, s, _ in rows if s == "synth")
    print(f"found real={n_real} synth={n_synth} under {args.crops_dir}")
    rows = subsample_balanced(
        rows, max_per_source=int(args.max_per_source), seed=int(args.seed)
    )
    print(
        f"using real={sum(1 for _, s, _ in rows if s == 'real')} "
        f"synth={sum(1 for _, s, _ in rows if s == 'synth')}"
    )
    if len({s for _, s, _ in rows}) < 2:
        print(
            "need both real (cmp*) and synth (flux2*) images for comparison",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    feats = extract_features(rows, device=device, batch_size=int(args.batch_size))
    emb = run_tsne(
        feats,
        n_components=int(args.components),
        perplexity=float(args.perplexity),
        seed=int(args.seed),
        pca_dim=int(args.pca_dim),
    )
    sources = [s for _, s, _ in rows]
    kinds = [k for _, _, k in rows]
    title = "DINOv2 ViT-S/14 features — real (cmp) vs synth (flux2)"
    plot_embedding(emb, sources, kinds, Path(args.out), title=title)

    if args.save_npz:
        npz_path = Path(args.out).with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            embedding=emb,
            features=feats,
            source=np.array(sources),
            kind=np.array(kinds),
            path=np.array([str(p) for p, _, _ in rows]),
        )
        print(f"saved {npz_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
