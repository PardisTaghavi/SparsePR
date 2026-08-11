"""Fused role-coordinate projection candidates for SWM/N8.

These kernels preserve FP32 role features.  BF16 model activations are promoted
to FP32 and the projection uses the same TF32 tensor-core contract as the
compiled PyTorch implementation.  The Q projection and per-row RMS
normalization are emitted in one launch, avoiding a token-sized intermediate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _role_projection_rms_kernel(
    X,
    FACTOR,
    OUT,
    TOKENS: tl.constexpr,
    INPUT_DIM: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    r = tl.arange(0, BLOCK_R)
    row_mask = rows < TOKENS
    k_mask = k < INPUT_DIM
    r_mask = r < RANK

    x = tl.load(
        X + (pid_bh * TOKENS + rows[:, None]) * INPUT_DIM + k[None, :],
        mask=row_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    factor = tl.load(
        FACTOR + (pid_bh * INPUT_DIM + k[:, None]) * RANK + r[None, :],
        mask=k_mask[:, None] & r_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    projected = tl.dot(x, factor, input_precision="tf32")
    sum_sq = tl.sum(tl.where(r_mask[None, :], projected * projected, 0.0), axis=1)
    inv_rms = tl.rsqrt(tl.maximum(sum_sq / RANK, 1.0e-12))
    normalized = projected * inv_rms[:, None]
    tl.store(
        OUT + (pid_bh * TOKENS + rows[:, None]) * RANK + r[None, :],
        normalized,
        mask=row_mask[:, None] & r_mask[None, :],
    )


def role_projection_rms_triton(
    x: torch.Tensor,
    factor: torch.Tensor,
    *,
    block_m: int = 16,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project ``[BH,N,D]`` through ``[BH,D,R]`` and RMS-normalize rows."""
    if not x.is_cuda or not factor.is_cuda:
        raise RuntimeError("Role projection requires CUDA tensors.")
    if x.ndim != 3 or factor.ndim != 3:
        raise RuntimeError(f"Expected X[BH,N,D], F[BH,D,R], got {x.shape}, {factor.shape}.")
    bh, tokens, dim = x.shape
    if factor.shape[0] != bh or factor.shape[1] != dim:
        raise RuntimeError(f"Role projection shape mismatch: X={x.shape}, F={factor.shape}.")
    rank = int(factor.shape[2])
    if dim > 256 or rank > 128:
        raise RuntimeError("Fused role projection currently supports D<=256 and R<=128.")
    if block_m not in {16, 32, 64}:
        raise RuntimeError("Role projection tensor-core BLOCK_M must be one of 16,32,64.")
    if num_warps not in {2, 4, 8}:
        raise RuntimeError("Role projection num_warps must be one of 2,4,8.")
    x = x.contiguous()
    factor = factor.contiguous()
    if out is None:
        out = torch.empty((bh, tokens, rank), device=x.device, dtype=torch.float32)
    elif out.shape != (bh, tokens, rank) or out.device != x.device or out.dtype != torch.float32:
        raise RuntimeError(
            f"Role projection output must be {(bh, tokens, rank)} FP32 on {x.device}, got "
            f"{tuple(out.shape)} {out.dtype} {out.device}."
        )
    _role_projection_rms_kernel[(triton.cdiv(tokens, block_m), bh)](
        x,
        factor,
        out,
        TOKENS=tokens,
        INPUT_DIM=dim,
        RANK=rank,
        BLOCK_M=block_m,
        BLOCK_K=triton.next_power_of_2(dim),
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=num_warps,
        num_stages=1,
    )
    return out
