"""Generate synthetic open_work+masonry balcony crops with FLUX.2 [dev].

Writes PNGs into ``runs/balcony_clf/crops/unlabeled/`` for Label UI review
(label as open_work + masonry), then train with ``--refresh-split``.

Setup (once)::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe -m pip install -r requirements-flux2.txt
    # Accept https://huggingface.co/black-forest-labs/FLUX.2-dev terms
    huggingface-cli login

Examples::

    # Quick test (consumer GPU: quantized + offload)
    python balcony_train/generate_flux2.py -n 2 --quantized --cpu-offload --device cuda

    # Batch for labeling
    python balcony_train/generate_flux2.py --count 40 --quantized --cpu-offload --device cuda
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import ensure_crop_dirs  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402
from balcony_train.prompts_masonry import MASONRY_OPENWORK_PROMPTS  # noqa: E402

DEFAULT_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_QUANTIZED_MODEL = "diffusers/FLUX.2-dev-bnb-4bit"
DEFAULT_PREFIX = "flux2_masonry_"


def _next_index(dest_dir: Path, prefix: str) -> int:
    """Next free numeric suffix for ``{prefix}NNN.png`` (does not overwrite)."""
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        return 0
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.(?:png|jpg|jpeg|webp)$", re.I)
    best = -1
    for p in dest_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def _load_pipeline(model_id: str, device: str, *, cpu_offload: bool):
    try:
        import torch
        from diffusers import Flux2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing FLUX.2 dependencies. Install with:\n"
            "  C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe "
            "-m pip install -r requirements-flux2.txt"
        ) from exc

    # bf16 is the documented dtype for FLUX.2; fall back on CPU.
    if device.startswith("cuda") and torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe, torch


def generate_masonry_crops(
    dest_dir: Path,
    *,
    count: int,
    seed: int,
    prefix: str = DEFAULT_PREFIX,
    model_id: str = DEFAULT_MODEL,
    width: int = 640,
    height: int = 384,
    steps: int = 28,
    guidance_scale: float = 4.0,
    device: str = "cuda",
    cpu_offload: bool = False,
) -> dict[str, int | str]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    pipe, torch = _load_pipeline(model_id, device, cpu_offload=cpu_offload)
    # Generators for offloaded pipes are safer on CPU.
    gen_device = "cpu" if cpu_offload or not device.startswith("cuda") else "cuda"

    start = _next_index(dest_dir, prefix)
    saved = 0
    for i in range(count):
        idx = start + i
        prompt = MASONRY_OPENWORK_PROMPTS[i % len(MASONRY_OPENWORK_PROMPTS)]
        generator = torch.Generator(device=gen_device).manual_seed(seed + i)
        image = pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        out = dest_dir / f"{prefix}{idx:03d}.png"
        image.save(out)
        saved += 1
        print(f"[{saved}/{count}] {out.name}")

    return {"saved": saved, "dest": str(dest_dir), "start_index": start}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        metavar="N",
        help="number of images to generate (default: 10)",
    )
    ap.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="root with unlabeled/ and kind folders",
    )
    ap.add_argument(
        "--out-subdir",
        default="unlabeled",
        help="subdir under crops-dir (default: unlabeled for Label UI)",
    )
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="output filename prefix")
    ap.add_argument("--seed", type=int, default=3000, help="base RNG seed")
    ap.add_argument(
        "--model",
        default=None,
        help=f"HF model id (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--quantized",
        action="store_true",
        help=f"use {DEFAULT_QUANTIZED_MODEL} (consumer GPU)",
    )
    ap.add_argument("--width", type=int, default=640, help="image width (wide crop)")
    ap.add_argument("--height", type=int, default=384, help="image height")
    ap.add_argument(
        "--steps",
        type=int,
        default=28,
        help="inference steps (default 28; 50 is higher quality)",
    )
    ap.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="guidance scale (FLUX.2 docs often use ~2.5–4)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu (cuda strongly recommended)",
    )
    ap.add_argument(
        "--cpu-offload",
        action="store_true",
        help="model CPU offload (needed for full / 4-bit on many consumer GPUs)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.model:
        model_id = str(args.model)
    elif args.quantized:
        model_id = DEFAULT_QUANTIZED_MODEL
    else:
        model_id = DEFAULT_MODEL

    ensure_crop_dirs(args.crops_dir)
    dest_dir = Path(args.crops_dir) / args.out_subdir
    stats = generate_masonry_crops(
        dest_dir,
        count=max(1, int(args.count)),
        seed=int(args.seed),
        prefix=str(args.prefix),
        model_id=model_id,
        width=int(args.width),
        height=int(args.height),
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        device=str(args.device),
        cpu_offload=bool(args.cpu_offload),
    )
    print(f"saved={stats['saved']} -> {stats['dest']}")
    print(
        "Label in Label UI as open_work + masonry, then: "
        "python balcony_train/train.py --device cuda --refresh-split"
    )


if __name__ == "__main__":
    main()
