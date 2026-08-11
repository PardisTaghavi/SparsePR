"""Launch official Cosmos-Predict2.5-14B image-to-world inference with SparsePR."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cosmos25_sparsepr"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.cosmos_root.expanduser().resolve()
    entrypoint = root / "examples" / "inference.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Missing official Cosmos entrypoint: {entrypoint}")
    package_root = Path(__file__).resolve().parents[1]
    hook_root = package_root / "src" / "sparsepr" / "hooks" / "cosmos_predict2"

    with tempfile.TemporaryDirectory(prefix="sparsepr_cosmos25_") as temp:
        temp_root = Path(temp)
        manifest = temp_root / "input.jsonl"
        config = temp_root / "config.json"
        manifest.write_text(
            json.dumps(
                {
                    "name": "sparsepr_sample",
                    "prompt": args.prompt,
                    "inference_type": "image2world",
                    "input_path": str(args.image.expanduser().resolve()),
                    "seed": args.seed,
                    "guidance": 7,
                    "num_output_frames": 77,
                    "num_steps": 35,
                }
            )
            + "\n"
        )
        config.write_text(
            json.dumps(
                {
                    "total_layers": 36,
                    "total_steps": 35,
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
        env = os.environ.copy()
        env["SPARSEPR_COSMOS_PREDICT2_CONFIG"] = str(config)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(hook_root), str(package_root / "src"), env.get("PYTHONPATH", ""))
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(entrypoint),
                "-i",
                str(manifest),
                "-o",
                str(args.output_dir.expanduser().resolve()),
                "--model=14B/post-trained",
                "--context-parallel-size=1",
                "--offload-text-encoder",
                "--offload-tokenizer",
                "--offload-diffusion-model",
                "--disable-guardrails",
            ],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
