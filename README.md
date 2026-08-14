# SparsePR

SparsePR is the reference implementation of training-free sparse attention with
response-coupled partitioning and probe-fitted residual reconstruction.

<p align="center">
  <strong>arXiv</strong> &nbsp;·&nbsp;
  <strong>Website</strong> &nbsp;·&nbsp;
  <strong>Hugging Face</strong>
</p>

https://github.com/user-attachments/assets/6d27e6a6-a0e6-43f0-bccb-ee3be4161f9b

Supported integrations:

- HunyuanVideo-13B (text-to-video)
- Wan2.2-I2V-A14B
- Cosmos-Predict2.5-14B
- Cosmos3-Nano-16B

## Install

Linux is required, and an NVIDIA H100 GPU is recommended. Two CUDA 12.8 wheel
environments are used because Cosmos3 requires newer Diffusers and Transformers
versions:

```bash
# HunyuanVideo, Wan2.2, and Cosmos-Predict2.5
python3.10 -m venv .venv-official
source .venv-official/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints-official-cu128.txt \
  -e ".[test,cuda,inference]"
python -m pip install flashinfer-python==0.6.2 --no-deps
python -m pip install flash-attn==2.7.3 --no-build-isolation

# Cosmos3
deactivate
python3.11 -m venv .venv-cosmos3
source .venv-cosmos3/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints-cosmos3-cu128.txt \
  -e ".[test,cuda,inference]"
python -m pip install flashinfer-python==0.6.2 --no-deps
```

Build the optional fused CUDA kernels with:

```bash
cmake -S src/sparsepr/kernels/n8_ext -B src/sparsepr/kernels/n8_ext/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DPython_ROOT_DIR="$(python -c 'import sys; print(sys.prefix)')"
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
git clone https://github.com/Wan-Video/Wan2.2.git
git -C Wan2.2 checkout --detach 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
hf download Wan-AI/Wan2.2-I2V-A14B \
  --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3 \
  --local-dir Wan2.2-I2V-A14B

git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git -C cosmos-predict2.5 checkout --detach aacbf4865d69c43a891cfda88e1a9bfcfd31d094

export SPARSEPR_WAN_ROOT=/path/to/Wan2.2
export SPARSEPR_WAN_CHECKPOINT=/path/to/Wan2.2-I2V-A14B
export SPARSEPR_COSMOS25_ROOT=/path/to/cosmos-predict2.5
```

Use `--dry-run` to validate a command without loading weights. The TOML files
under `configs/` record the final generation and SparsePR settings.

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
