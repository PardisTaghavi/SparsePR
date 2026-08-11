"""Run one official Wan2.2-I2V-A14B sample with SparsePR."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

from sparsepr.adapters.wan22 import (
    Wan22TI2VSparseConfig,
    install_wan22_ti2v_sparse_patch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("wan22_sparsepr.mp4"))
    parser.add_argument("--size", default="1280*720")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.wan_root.expanduser().resolve()))
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils import utils as wan_utils

    task = "i2v-A14B"
    model_config = WAN_CONFIGS[task]
    steps = int(model_config.sample_steps)
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
    pipe = wan.WanI2V(
        config=model_config,
        checkpoint_dir=str(args.checkpoint.expanduser().resolve()),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    video = pipe.generate(
        args.prompt,
        Image.open(args.image).convert("RGB"),
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=int(model_config.frame_num),
        shift=float(model_config.sample_shift),
        sample_solver="unipc",
        sampling_steps=steps,
        guide_scale=model_config.sample_guide_scale,
        seed=args.seed,
        offload_model=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = getattr(wan_utils, "save_video", None) or wan_utils.cache_video
    save(
        tensor=video[None],
        save_file=str(args.output),
        fps=int(model_config.sample_fps),
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


if __name__ == "__main__":
    main()
