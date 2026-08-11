"""Run one HunyuanVideo-13B text-to-video sample with SparsePR."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.utils import export_to_video

from sparsepr.adapters.hunyuanvideo import (
    HunyuanVideoConfig,
    dense_warmup_threshold,
    install_hunyuanvideo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("hunyuan_sparsepr.mp4"))
    parser.add_argument("--model", default="tencent/HunyuanVideo")
    parser.add_argument("--revision", default="refs/pr/18")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = HunyuanVideoConfig()
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        scheduler=scheduler,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(50)
    config = replace(
        config,
        first_timestep=dense_warmup_threshold(
            pipe.scheduler, num_steps=50, fraction=0.10
        ),
    )
    install_hunyuanvideo(pipe, args.prompt, config=config)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            guidance_scale=6.0,
            num_inference_steps=50,
            generator=generator,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(args.output), fps=24)


if __name__ == "__main__":
    main()
