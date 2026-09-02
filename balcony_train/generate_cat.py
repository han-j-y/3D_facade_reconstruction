"""Generate a cat image with SDXL (pipeline sanity check).

    pip install -r requirements-sdxl.txt
    python balcony_train/generate_cat.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.prompts_cat import CAT_PROMPT, NEGATIVE_PROMPT  # noqa: E402

OUT_DIR = ROOT / "runs" / "sdxl_test"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for i in range(args.count):
        img = pipe(
            prompt=CAT_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_inference_steps=30,
            guidance_scale=7.0,
            generator=gen,
        ).images[0]
        path = OUT_DIR / f"cat_{i:03d}.png"
        img.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
