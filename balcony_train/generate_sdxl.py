"""Generate synthetic balcony crops with Stable Diffusion XL.



Writes mixed railing styles into ``runs/balcony_clf/crops/unlabeled/``.

Review each PNG and move to ``solid/`` or ``baluster/`` by hand.



Requires optional deps::



    pip install -r requirements-sdxl.txt

    huggingface-cli login   # accept SDXL license once



Example::



    python balcony_train/generate_sdxl.py -n 5 --device cuda

    python balcony_train/generate_sdxl.py --count 100 --device cuda

"""



from __future__ import annotations



import argparse

import sys

from pathlib import Path



HERE = Path(__file__).resolve().parent

ROOT = HERE.parent

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



from balcony_train.labels import ensure_crop_dirs  # noqa: E402

from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402

from balcony_train.prompts_balcony import BALCONY_PROMPTS, NEGATIVE_PROMPT  # noqa: E402



DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

DEFAULT_PREFIX = "sdxl_balcony_"





def _next_index(dest_dir: Path, prefix: str) -> int:

    """First unused index for ``{prefix}{idx:03d}.png``."""

    idx = 0

    while (dest_dir / f"{prefix}{idx:03d}.png").is_file():

        idx += 1

    return idx





def _load_pipeline(model_id: str, device: str, *, cpu_offload: bool):

    try:

        import torch

        from diffusers import StableDiffusionXLPipeline

    except ImportError as exc:

        raise SystemExit(

            "Missing SDXL dependencies. Install with:\n"

            "  pip install -r requirements-sdxl.txt"

        ) from exc



    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(

        model_id,

        torch_dtype=dtype,

        variant="fp16" if dtype == torch.float16 else None,

        use_safetensors=True,

    )

    if cpu_offload:

        pipe.enable_model_cpu_offload()

    else:

        pipe.to(device)

    return pipe, torch





def generate_balcony_crops(

    dest_dir: Path,

    *,

    count: int,

    seed: int,

    prefix: str = DEFAULT_PREFIX,

    model_id: str = DEFAULT_MODEL,

    width: int = 640,

    height: int = 384,

    steps: int = 35,

    guidance_scale: float = 6.0,

    device: str = "cuda",

    cpu_offload: bool = False,

) -> dict[str, int]:

    dest_dir = Path(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)



    pipe, torch = _load_pipeline(model_id, device, cpu_offload=cpu_offload)

    gen_device = "cuda" if device.startswith("cuda") and not cpu_offload else "cpu"



    start = _next_index(dest_dir, prefix)

    saved = 0

    for i in range(count):

        idx = start + i

        prompt = BALCONY_PROMPTS[i % len(BALCONY_PROMPTS)]

        generator = torch.Generator(device=gen_device).manual_seed(seed + i)

        image = pipe(

            prompt=prompt,

            negative_prompt=NEGATIVE_PROMPT,

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

generate_solid_crops = generate_balcony_crops





def parse_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(

        "-n",

        "--count",

        type=int,

        default=10,

        metavar="N",

        help="number of images to generate (default: 10; e.g. -n 5 for a quick test)",

    )

    ap.add_argument(

        "--crops-dir",

        type=Path,

        default=DEFAULT_CROPS_DIR,

        help="root with unlabeled/ solid/ baluster/",

    )

    ap.add_argument(

        "--out-subdir",

        default="unlabeled",

        help="subdir under crops-dir (default: unlabeled for manual review)",

    )

    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="output filename prefix")

    ap.add_argument("--seed", type=int, default=2000, help="base RNG seed")

    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")

    ap.add_argument("--width", type=int, default=640, help="image width (wide crop)")

    ap.add_argument("--height", type=int, default=384, help="image height")

    ap.add_argument("--steps", type=int, default=35, help="inference steps (default 35)")

    ap.add_argument(

        "--guidance-scale",

        type=float,

        default=6.0,

        help="CFG scale; lower (5–7) tends to look more photographic",

    )

    ap.add_argument(

        "--device",

        default="cuda",

        help="cuda or cpu (cuda strongly recommended)",

    )

    ap.add_argument(

        "--cpu-offload",

        action="store_true",

        help="enable model CPU offload (slower, helps 8 GB VRAM)",

    )

    return ap.parse_args()





def main() -> None:

    args = parse_args()

    ensure_crop_dirs(args.crops_dir)

    dest_dir = Path(args.crops_dir) / args.out_subdir

    stats = generate_balcony_crops(

        dest_dir,

        count=max(1, int(args.count)),

        seed=int(args.seed),

        prefix=str(args.prefix),

        model_id=str(args.model),

        width=int(args.width),

        height=int(args.height),

        steps=int(args.steps),

        guidance_scale=float(args.guidance_scale),

        device=str(args.device),

        cpu_offload=bool(args.cpu_offload),

    )

    print(f"saved={stats['saved']} -> {stats['dest']}")

    print("Review in unlabeled/, move to solid/ or baluster/, then run train.py")





if __name__ == "__main__":

    main()


