"""Unified command-line inference for the supported SparsePR integrations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence


MODEL_IDS = {
    "hunyuanvideo-13b": "tencent/HunyuanVideo",
    "wan2.2-i2v-a14b": "Wan-AI/Wan2.2-I2V-A14B",
    "cosmos-predict2.5-14b": "nvidia/Cosmos-Predict2.5-14B",
    "cosmos3-nano-16b": "nvidia/Cosmos3-Nano",
}
IMAGE_MODELS = {
    "wan2.2-i2v-a14b",
    "cosmos-predict2.5-14b",
    "cosmos3-nano-16b",
}


@dataclass(frozen=True)
class InferenceRequest:
    """Validated, model-independent inference request."""

    model: str
    prompt: str
    output: Path
    image: Path | None
    attention: str
    model_id: str
    source_root: Path | None
    checkpoint: Path | None
    revision: str | None
    seed: int
    height: int
    width: int
    frames: int
    steps: int
    fps: int
    guidance_scale: float
    size: str
    offload: bool

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        for key in ("output", "image", "source_root", "checkpoint"):
            value = values[key]
            values[key] = str(value) if value is not None else None
        return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparsepr-infer",
        description="Run dense or SparsePR inference on a supported video model.",
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_IDS))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention", choices=("dense", "sparsepr"), default="sparsepr"
    )
    parser.add_argument("--model-id")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Official Wan2.2 or Cosmos-Predict2.5 source checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local Wan2.2 checkpoint directory.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--size", default="832*480")
    parser.add_argument(
        "--offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model CPU offloading where the upstream runtime supports it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved request without loading a model.",
    )
    return parser


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def resolve_request(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> InferenceRequest:
    defaults = {
        "hunyuanvideo-13b": (544, 960, 129, 50, 6.0),
        "wan2.2-i2v-a14b": (480, 832, 81, 40, 5.0),
        "cosmos-predict2.5-14b": (704, 1280, 77, 35, 7.0),
        "cosmos3-nano-16b": (480, 832, 81, 35, 7.0),
    }
    height, width, frames, steps, guidance = defaults[args.model]
    image = args.image.expanduser().resolve() if args.image else None
    if args.model in IMAGE_MODELS and image is None:
        parser.error(f"--image is required for {args.model}")
    if image is not None and not image.is_file():
        parser.error(f"input image does not exist: {image}")

    source_root = args.source_root
    checkpoint = args.checkpoint
    if args.model == "wan2.2-i2v-a14b":
        source_root = source_root or _environment_path("SPARSEPR_WAN_ROOT")
        checkpoint = checkpoint or _environment_path("SPARSEPR_WAN_CHECKPOINT")
        if source_root is None:
            parser.error("set --source-root or SPARSEPR_WAN_ROOT for Wan2.2")
        if checkpoint is None:
            parser.error("set --checkpoint or SPARSEPR_WAN_CHECKPOINT for Wan2.2")
    elif args.model == "cosmos-predict2.5-14b":
        source_root = source_root or _environment_path("SPARSEPR_COSMOS25_ROOT")
        if source_root is None:
            parser.error(
                "set --source-root or SPARSEPR_COSMOS25_ROOT for Cosmos-Predict2.5"
            )

    source_root = source_root.expanduser().resolve() if source_root else None
    checkpoint = checkpoint.expanduser().resolve() if checkpoint else None
    if source_root is not None and not source_root.is_dir():
        parser.error(f"source checkout does not exist: {source_root}")
    if checkpoint is not None and not checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint}")

    request = InferenceRequest(
        model=args.model,
        prompt=args.prompt,
        output=args.output.expanduser().resolve(),
        image=image,
        attention=args.attention,
        model_id=args.model_id or MODEL_IDS[args.model],
        source_root=source_root,
        checkpoint=checkpoint,
        revision=args.revision,
        seed=args.seed,
        height=args.height or height,
        width=args.width or width,
        frames=args.frames or frames,
        steps=args.steps or steps,
        fps=args.fps,
        guidance_scale=(
            args.guidance_scale if args.guidance_scale is not None else guidance
        ),
        size=args.size,
        offload=args.offload,
    )
    for name in ("height", "width", "frames", "steps", "fps"):
        if getattr(request, name) <= 0:
            parser.error(f"--{name} must be positive")
    if request.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    return request


def _place_pipeline(pipe: object, offload: bool) -> None:
    if offload:
        enable = getattr(pipe, "enable_model_cpu_offload", None)
        if enable is None:
            raise RuntimeError("This pipeline does not support model CPU offloading.")
        enable()
    else:
        pipe.to("cuda")


def _run_hunyuanvideo(request: InferenceRequest) -> None:
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

    revision = request.revision or "refs/pr/18"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        request.model_id,
        subfolder="transformer",
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        request.model_id,
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.enable_tiling()
    _place_pipeline(pipe, request.offload)
    pipe.scheduler.set_timesteps(request.steps)
    if request.attention == "sparsepr":
        config = HunyuanVideoConfig(
            height=request.height,
            width=request.width,
            num_frames=request.frames,
        )
        config = replace(
            config,
            first_timestep=dense_warmup_threshold(
                pipe.scheduler, num_steps=request.steps, fraction=0.10
            ),
        )
        install_hunyuanvideo(pipe, request.prompt, config=config)

    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_frames=request.frames,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.steps,
            generator=generator,
        )
    export_to_video(result.frames[0], str(request.output), fps=request.fps)


def _run_wan22(request: InferenceRequest) -> None:
    from PIL import Image

    assert request.source_root is not None
    assert request.checkpoint is not None
    assert request.image is not None
    sys.path.insert(0, str(request.source_root))
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils import utils as wan_utils

    task = "i2v-A14B"
    model_config = WAN_CONFIGS[task]
    steps = request.steps or int(model_config.sample_steps)
    if request.attention == "sparsepr":
        from sparsepr.adapters.wan22 import (
            Wan22TI2VSparseConfig,
            install_wan22_ti2v_sparse_patch,
        )

        total_layers = int(model_config.num_layers)
        install_wan22_ti2v_sparse_patch(
            Wan22TI2VSparseConfig(
                pattern="N8_custom_v6",
                first_layers_fp=max(1, math.floor(0.03 * total_layers)),
                first_sparse_forward=math.floor(0.20 * steps),
                n8_target_density=0.22,
                n8_probe_rows=64,
                n8_repair_rank=16,
                kmeans_iter_init=25,
                kmeans_iter_step=2,
            )
        )
    if request.size not in MAX_AREA_CONFIGS:
        choices = ", ".join(sorted(MAX_AREA_CONFIGS))
        raise ValueError(f"Unsupported Wan size {request.size!r}; choose one of: {choices}")
    pipe = wan.WanI2V(
        config=model_config,
        checkpoint_dir=str(request.checkpoint),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=request.offload,
        convert_model_dtype=True,
    )
    video = pipe.generate(
        request.prompt,
        Image.open(request.image).convert("RGB"),
        max_area=MAX_AREA_CONFIGS[request.size],
        frame_num=request.frames,
        shift=float(model_config.sample_shift),
        sample_solver="unipc",
        sampling_steps=steps,
        guide_scale=request.guidance_scale,
        seed=request.seed,
        offload_model=request.offload,
    )
    save = getattr(wan_utils, "save_video", None) or wan_utils.cache_video
    save(
        tensor=video[None],
        save_file=str(request.output),
        fps=request.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


def _run_cosmos_predict2(request: InferenceRequest) -> None:
    assert request.source_root is not None
    assert request.image is not None
    entrypoint = request.source_root / "examples" / "inference.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Missing official Cosmos entrypoint: {entrypoint}")
    source_root = Path(__file__).resolve().parents[1]
    hook_root = source_root / "sparsepr" / "hooks" / "cosmos_predict2"

    with tempfile.TemporaryDirectory(prefix="sparsepr_cosmos25_") as temp:
        temp_root = Path(temp)
        manifest = temp_root / "input.jsonl"
        output_dir = temp_root / "output"
        manifest.write_text(
            json.dumps(
                {
                    "name": "sparsepr_sample",
                    "prompt": request.prompt,
                    "inference_type": "image2world",
                    "input_path": str(request.image),
                    "seed": request.seed,
                    "guidance": request.guidance_scale,
                    "num_output_frames": request.frames,
                    "num_steps": request.steps,
                }
            )
            + "\n"
        )
        env = os.environ.copy()
        if request.attention == "sparsepr":
            config = temp_root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "total_layers": 36,
                        "total_steps": request.steps,
                        "forwards_per_step": 2,
                        "dense_first_steps": 2,
                        "dense_first_layers": 1,
                        "target_density": 0.22,
                        "num_q_centroids": 300,
                        "num_k_centroids": 1000,
                        "probe_rows": 64,
                        "repair_rank": 16,
                        "route_refresh_every": 4,
                    }
                )
            )
            env["SPARSEPR_COSMOS_PREDICT2_CONFIG"] = str(config)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(hook_root), str(source_root), env.get("PYTHONPATH", ""))
            )
        command = [
            sys.executable,
            "-u",
            str(entrypoint),
            "-i",
            str(manifest),
            "-o",
            str(output_dir),
            "--model=14B/post-trained",
            "--context-parallel-size=1",
            "--disable-guardrails",
        ]
        if request.offload:
            command.extend(
                (
                    "--offload-text-encoder",
                    "--offload-tokenizer",
                    "--offload-diffusion-model",
                )
            )
        subprocess.run(command, cwd=request.source_root, env=env, check=True)
        videos = sorted(output_dir.rglob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(
                f"Expected one Cosmos output MP4, found {len(videos)} under {output_dir}"
            )
        shutil.copy2(videos[0], request.output)


def _run_cosmos3(request: InferenceRequest) -> None:
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )
    from diffusers.utils import export_to_video, load_image

    from sparsepr.adapters.cosmos3 import Cosmos3Config, install_cosmos3

    assert request.image is not None
    pipe = Cosmos3OmniPipeline.from_pretrained(
        request.model_id,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=7.0
    )
    _place_pipeline(pipe, request.offload)
    if request.attention == "sparsepr":
        install_cosmos3(
            pipe,
            config=Cosmos3Config(
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.steps,
            ),
        )
    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            image=load_image(str(request.image)),
            height=request.height,
            width=request.width,
            num_frames=request.frames,
            fps=request.fps,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            enable_safety_check=False,
            generator=generator,
        )
    frames = result.frames[0] if hasattr(result, "frames") else result[0]
    export_to_video(frames, str(request.output), fps=request.fps)


RUNNERS = {
    "hunyuanvideo-13b": _run_hunyuanvideo,
    "wan2.2-i2v-a14b": _run_wan22,
    "cosmos-predict2.5-14b": _run_cosmos_predict2,
    "cosmos3-nano-16b": _run_cosmos3,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    request = resolve_request(args, parser)
    if args.dry_run:
        print(json.dumps(request.public_dict(), indent=2, sort_keys=True))
        return

    request.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    RUNNERS[request.model](request)
    if not request.output.is_file() or request.output.stat().st_size == 0:
        raise RuntimeError(f"Inference did not produce a non-empty file: {request.output}")
    print(
        json.dumps(
            {
                "attention": request.attention,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "model": request.model,
                "output": str(request.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
