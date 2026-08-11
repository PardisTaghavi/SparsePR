# SparsePR

SparsePR is the reference implementation of training-free sparse attention with
response-coupled partitioning and probe-fitted residual reconstruction.

Supported integrations:

- HunyuanVideo-13B (text-to-video)
- Wan2.2-I2V-A14B
- Cosmos-Predict2.5-14B
- Cosmos3-Nano-16B

The repository is private while release packaging and reproducibility checks are
completed.

## Install

Use Python 3.11+, PyTorch 2.5+, CUDA 12.4/12.8, Triton, and a compatible
FlashInfer build. Install SparsePR in editable mode:

```bash
pip install -e ".[test,cuda,inference]"
```

Build the optional fused CUDA kernels with:

```bash
cmake -S src/sparsepr/kernels/n8_ext -B src/sparsepr/kernels/n8_ext/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build src/sparsepr/kernels/n8_ext/build -j
```

## Inference

All integrations use one interface. Checkpoints come from their official Hugging
Face repositories and are never stored here.

```bash
sparsepr-infer \
  --model cosmos3-nano-16b \
  --prompt "A robot walking through a forest" \
  --image input.png \
  --output output.mp4
```

Select `--attention dense` for the baseline or `--attention sparsepr` for the
method. `--offload` is enabled by default. Wan2.2 and Cosmos-Predict2.5 use their
official source trees:

```bash
export SPARSEPR_WAN_ROOT=/path/to/Wan2.2
export SPARSEPR_WAN_CHECKPOINT=/path/to/Wan2.2-I2V-A14B
export SPARSEPR_COSMOS25_ROOT=/path/to/cosmos-predict2.5
```

On one A100 40 GB, start with Cosmos-Predict2.5 or Cosmos3-Nano. HunyuanVideo
requires reduced resolution and Wan2.2 full inference may require an 80 GB GPU.
Use `--dry-run` to validate a command without loading weights.

## Layout

```text
src/sparsepr/models/common/  shared routing and residual reconstruction
src/sparsepr/adapters/       model-specific attention integrations
src/sparsepr/kernels/        Triton and optional CUDA kernels
src/flash_kmeans/            batched Triton k-means runtime
configs/                     frozen per-model inference settings
examples/                    inference entrypoints
tests/                       CPU contracts and CUDA parity tests
```

## Attribution

Parts of the attention runtime derive from Sparse-VideoGen and Flash k-Means.
Both are distributed under Apache-2.0; see `THIRD_PARTY_NOTICES.md` and
`LICENSES/Apache-2.0.txt`.
