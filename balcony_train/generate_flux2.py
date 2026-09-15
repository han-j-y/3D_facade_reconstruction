"""Generate synthetic balcony railing crops with FLUX.2 [dev].

Writes PNGs into ``runs/balcony_clf/crops/unlabeled/`` for Label UI review,
then train with ``--refresh-split``.

Prompt sets (``--prompt-set``):
  - ``masonry`` — open_work + masonry (default)
  - ``metal`` — open_work + metal (bars / wrought iron / open patterns)
  - ``surface_panel`` — modern frame + glass/metal/privacy panels
  - ``solid`` — opaque wall / parapet mass

Setup (once)::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe -m pip install -r requirements-flux2.txt
    # Accept https://huggingface.co/black-forest-labs/FLUX.2-dev terms
    huggingface-cli login

Examples::

    python balcony_train/generate_flux2.py -n 2 --quantized --device cuda --prompt-set masonry
    python balcony_train/generate_flux2.py -n 300 --quantized --device cuda --prompt-set metal
    python balcony_train/generate_flux2.py -n 40 --quantized --device cuda --prompt-set surface_panel
    python balcony_train/generate_flux2.py -n 40 --quantized --device cuda --prompt-set solid
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
from balcony_train.prompts_metal import METAL_OPENWORK_PROMPTS  # noqa: E402
from balcony_train.prompts_solid import SOLID_PROMPTS  # noqa: E402
from balcony_train.prompts_surface_panel import SURFACE_PANEL_PROMPTS  # noqa: E402

DEFAULT_MODEL = "black-forest-labs/FLUX.2-dev"
DEFAULT_QUANTIZED_MODEL = "diffusers/FLUX.2-dev-bnb-4bit"

PROMPT_SETS: dict[str, dict[str, object]] = {
    "masonry": {
        "prompts": MASONRY_OPENWORK_PROMPTS,
        "prefix": "flux2_masonry_",
        "label_hint": "Label as open_work + masonry",
    },
    "metal": {
        "prompts": METAL_OPENWORK_PROMPTS,
        "prefix": "flux2_metal_",
        "label_hint": "Label as open_work + metal",
    },
    "surface_panel": {
        "prompts": SURFACE_PANEL_PROMPTS,
        "prefix": "flux2_surface_",
        "label_hint": "Label as surface_panel",
    },
    "solid": {
        "prompts": SOLID_PROMPTS,
        "prefix": "flux2_solid_",
        "label_hint": "Label as solid",
    },
}

DEFAULT_PREFIX = str(PROMPT_SETS["masonry"]["prefix"])


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


def _load_pipeline_cpu_first(model_id: str, dtype):
    """Load weights onto CPU first so bnb-4bit does not spike VRAM during init.

    Quantized FLUX.2 checkpoints otherwise place shards on CUDA inside
    ``from_pretrained``, which OOMs on ~24–32 GB cards before offload starts.
    """
    from diffusers import Flux2Pipeline

    # Newer diffusers: whole pipe on CPU, then enable_model_cpu_offload().
    try:
        return Flux2Pipeline.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cpu"
        )
    except (TypeError, ValueError):
        pass

    # Explicit component load (BFL / DataCamp pattern for bnb-4bit).
    from transformers import Mistral3ForConditionalGeneration

    try:
        from diffusers import Flux2Transformer2DModel as TransformerCls
    except ImportError:  # pragma: no cover
        from diffusers import AutoModel as TransformerCls

    text_kwargs = {
        "subfolder": "text_encoder",
        "device_map": "cpu",
    }
    # transformers APIs differ slightly across versions.
    try:
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, **text_kwargs
        )
    except TypeError:
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, **text_kwargs
        )

    transformer = TransformerCls.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
    )
    return Flux2Pipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    )


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

    if cpu_offload:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Loading {model_id} on CPU first (cpu-offload)...", flush=True)
        pipe = _load_pipeline_cpu_first(model_id, dtype)
        pipe.enable_model_cpu_offload()
    else:
        print(f"Loading {model_id} onto {device}...", flush=True)
        pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to(device)
    return pipe, torch


def generate_balcony_crops(
    dest_dir: Path,
    *,
    count: int,
    seed: int,
    prompts: list[str],
    prefix: str = DEFAULT_PREFIX,
    model_id: str = DEFAULT_MODEL,
    width: int = 640,
    height: int = 384,
    steps: int = 28,
    guidance_scale: float = 4.0,
    device: str = "cuda",
    cpu_offload: bool = False,
) -> dict[str, int | str]:
    if not prompts:
        raise ValueError("prompts must be non-empty")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    pipe, torch = _load_pipeline(model_id, device, cpu_offload=cpu_offload)
    # Generators for offloaded pipes are safer on CPU.
    gen_device = "cpu" if cpu_offload or not device.startswith("cuda") else "cuda"

    start = _next_index(dest_dir, prefix)
    saved = 0
    for i in range(count):
        idx = start + i
        prompt = prompts[i % len(prompts)]
        generator = torch.Generator(device=gen_device).manual_seed(seed + i)
        print(f"prompt[{i % len(prompts)}]: {prompt}", flush=True)
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


# Back-compat alias.
generate_masonry_crops = generate_balcony_crops


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
        "--prompt-set",
        choices=sorted(PROMPT_SETS.keys()),
        default="masonry",
        help="which prompt bank + default filename prefix to use",
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
    ap.add_argument(
        "--prefix",
        default=None,
        help="output filename prefix (default depends on --prompt-set)",
    )
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
        help="model CPU offload (auto-on with --quantized; needed on ~24–32 GB GPUs)",
    )
    ap.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="force full GPU residency (large VRAM only; overrides --cpu-offload)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    spec = PROMPT_SETS[str(args.prompt_set)]
    prompts = list(spec["prompts"])  # type: ignore[arg-type]
    prefix = str(args.prefix) if args.prefix else str(spec["prefix"])
    label_hint = str(spec["label_hint"])

    if args.model:
        model_id = str(args.model)
    elif args.quantized:
        model_id = DEFAULT_QUANTIZED_MODEL
    else:
        model_id = DEFAULT_MODEL

    # 4-bit still OOMs if shards land on CUDA during from_pretrained; default offload on.
    cpu_offload = bool(args.cpu_offload) or bool(args.quantized)
    if args.no_cpu_offload:
        cpu_offload = False

    ensure_crop_dirs(args.crops_dir)
    dest_dir = Path(args.crops_dir) / args.out_subdir
    stats = generate_balcony_crops(
        dest_dir,
        count=max(1, int(args.count)),
        seed=int(args.seed),
        prompts=prompts,
        prefix=prefix,
        model_id=model_id,
        width=int(args.width),
        height=int(args.height),
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        device=str(args.device),
        cpu_offload=cpu_offload,
    )
    print(f"prompt_set={args.prompt_set}  prefix={prefix}")
    print(f"saved={stats['saved']} -> {stats['dest']}")
    print(
        f"{label_hint}, then: "
        "python balcony_train/train.py --device cuda --refresh-split"
    )


if __name__ == "__main__":
    main()
