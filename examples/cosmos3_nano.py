"""Run one Cosmos3-Nano-16B image-conditioned sample with SparsePR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

from sparsepr.adapters.cosmos3 import Cosmos3Config, install_cosmos3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("cosmos3_sparsepr.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Cosmos3Config()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=7.0
    )
    pipe.to("cuda")
    install_cosmos3(pipe, config=config)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            image=load_image(args.image),
            height=720,
            width=1280,
            num_frames=121,
            fps=24,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            enable_safety_check=False,
            generator=generator,
        )
    frames = result.frames[0] if hasattr(result, "frames") else result[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=24)


if __name__ == "__main__":
    main()
