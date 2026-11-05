"""Domain classifier AUC: logistic regression on DINOv2 features (real vs synth).

Quantifies how separable frozen DINOv2 embeddings are between real facade
crops (``cmp*``) and FLUX.2 synth (``flux2_*``). AUC near 1.0 ⇒ strong domain
gap; near 0.5 ⇒ features do not encode source.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/domain_auc_real_vs_synth.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402
from balcony_train.tsne_real_vs_synth import (  # noqa: E402
    extract_features,
    iter_crop_paths,
    subsample_balanced,
)

DEFAULT_OUT = ROOT / "runs" / "balcony_clf" / "domain_auc_real_vs_synth.json"


def labels_from_sources(sources: list[str]) -> np.ndarray:
    """Map real→0, synth→1."""
    out = []
    for s in sources:
        if s == "real":
            out.append(0)
        elif s == "synth":
            out.append(1)
        else:
            raise ValueError(f"unknown source {s!r}")
    return np.asarray(out, dtype=np.int64)


def domain_auc_from_features(
    features: np.ndarray,
    y: np.ndarray,
    *,
    test_frac: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
) -> dict[str, float | int]:
    """Stratified train/test logistic regression; return AUC and counts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("need both real and synth labels")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features/labels length mismatch")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_frac),
        random_state=int(seed),
        stratify=y,
    )
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    C=float(C),
                    random_state=int(seed),
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = clf.predict(x_test)
    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, pred))
    return {
        "auc": auc,
        "accuracy": acc,
        "n_total": int(y.shape[0]),
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
        "n_real": int(np.sum(y == 0)),
        "n_synth": int(np.sum(y == 1)),
        "test_frac": float(test_frac),
        "C": float(C),
        "seed": int(seed),
    }


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
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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
        print("need both real (cmp*) and synth (flux2*) images", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    feats = extract_features(rows, device=device, batch_size=int(args.batch_size))
    y = labels_from_sources([s for _, s, _ in rows])
    metrics = domain_auc_from_features(
        feats,
        y,
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        C=float(args.C),
    )
    metrics["crops_dir"] = str(Path(args.crops_dir).resolve())
    metrics["max_per_source"] = int(args.max_per_source)
    metrics["backbone"] = "dinov2_vits14"

    print(
        f"domain AUC={metrics['auc']:.4f}  acc={metrics['accuracy']:.4f}  "
        f"test_n={metrics['n_test']}  (synth=1, real=0)"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
