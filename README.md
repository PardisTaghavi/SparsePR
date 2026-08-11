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
FlashInfer build. Then install SparsePR in editable mode:

```bash
pip install -e ".[test,cuda]"
```

Build the optional fused CUDA kernels with:

```bash
cmake -S src/sparsepr/kernels/n8_ext -B src/sparsepr/kernels/n8_ext/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build src/sparsepr/kernels/n8_ext/build -j
```

Model-specific inference commands will live in `examples/`. Checkpoints are
downloaded from their official providers and are never stored in this repository.

```bash
python examples/hunyuanvideo_t2v.py --prompt "..." --output out.mp4
python examples/wan22_i2v.py --wan-root /path/to/Wan2.2 \
  --checkpoint /path/to/Wan2.2-I2V-A14B --image input.jpg --prompt "..."
python examples/cosmos_predict2.py --cosmos-root /path/to/cosmos-predict2.5 \
  --image input.jpg --prompt "..."
python examples/cosmos3_nano.py --model /path/to/Cosmos3-Nano-16B \
  --image input.jpg --prompt "..."
```

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
