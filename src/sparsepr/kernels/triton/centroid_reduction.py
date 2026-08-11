"""FP32 atomic original-space centroid reductions for fixed N8 shapes."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _centroid_sum_kernel(
    X,
    LABELS,
    SUMS,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    CLUSTERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    n_mask = n < TOKENS
    d_mask = d < DIM
    labels = tl.load(LABELS + pid_b * TOKENS + n, mask=n_mask, other=0).to(tl.int32)
    valid = n_mask[:, None] & d_mask[None, :] & (labels[:, None] < CLUSTERS)
    values = tl.load(
        X + (pid_b * TOKENS + n[:, None]) * DIM + d[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    destinations = SUMS + (pid_b * CLUSTERS + labels[:, None]) * DIM + d[None, :]
    tl.atomic_add(destinations, values, mask=valid)


@triton.jit
def _joint_centroid_sum_kernel(
    X,
    Y,
    LABELS,
    X_SUMS,
    Y_SUMS,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    CLUSTERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    n_mask = n < TOKENS
    d_mask = d < DIM
    labels = tl.load(LABELS + pid_b * TOKENS + n, mask=n_mask, other=0).to(tl.int32)
    valid = n_mask[:, None] & d_mask[None, :] & (labels[:, None] < CLUSTERS)
    offsets = (pid_b * TOKENS + n[:, None]) * DIM + d[None, :]
    x = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)
    y = tl.load(Y + offsets, mask=valid, other=0.0).to(tl.float32)
    destinations = (pid_b * CLUSTERS + labels[:, None]) * DIM + d[None, :]
    tl.atomic_add(X_SUMS + destinations, x, mask=valid)
    tl.atomic_add(Y_SUMS + destinations, y, mask=valid)


@triton.jit
def _centroid_finalize_kernel(
    SUMS,
    SIZES,
    FALLBACK,
    OUT,
    DIM: tl.constexpr,
    CLUSTERS: tl.constexpr,
    HAS_FALLBACK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < DIM
    size = tl.load(SIZES + pid_b * CLUSTERS + pid_c).to(tl.float32)
    offsets = (pid_b * CLUSTERS + pid_c) * DIM + d
    sums = tl.load(SUMS + offsets, mask=d_mask, other=0.0)
    mean = sums / tl.maximum(size, 1.0)
    if HAS_FALLBACK:
        fallback = tl.load(FALLBACK + offsets, mask=d_mask, other=0.0).to(tl.float32)
        mean = tl.where(size > 0.0, mean, fallback)
    tl.store(OUT + offsets, mean, mask=d_mask)


def _validate(
    x: torch.Tensor,
    labels: torch.Tensor,
    sizes: torch.Tensor,
    fallback: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if not x.is_cuda or not labels.is_cuda or not sizes.is_cuda:
        raise RuntimeError("Triton centroid reduction requires CUDA tensors.")
    if x.ndim != 3 or labels.shape != x.shape[:2] or sizes.ndim != 2:
        raise RuntimeError(
            f"Invalid centroid shapes x={tuple(x.shape)}, labels={tuple(labels.shape)}, sizes={tuple(sizes.shape)}."
        )
    batch, tokens, dim = (int(value) for value in x.shape)
    clusters = int(sizes.shape[1])
    if sizes.shape[0] != batch:
        raise RuntimeError("Centroid batch dimensions do not match.")
    if fallback is not None and fallback.shape != (batch, clusters, dim):
        raise RuntimeError(f"Centroid fallback must be {(batch, clusters, dim)}, got {tuple(fallback.shape)}.")
    return batch, tokens, dim, clusters


def _finalize(
    sums: torch.Tensor,
    sizes: torch.Tensor,
    fallback: torch.Tensor | None,
) -> torch.Tensor:
    batch, clusters, dim = sums.shape
    out = torch.empty_like(sums)
    fallback_tensor = fallback if fallback is not None else sums
    _centroid_finalize_kernel[(clusters, batch)](
        sums,
        sizes,
        fallback_tensor,
        out,
        DIM=dim,
        CLUSTERS=clusters,
        HAS_FALLBACK=fallback is not None,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return out


def original_centroids_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    sizes: torch.Tensor,
    *,
    empty_fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, tokens, dim, clusters = _validate(x, labels, sizes, empty_fallback)
    sums = torch.zeros((batch, clusters, dim), device=x.device, dtype=torch.float32)
    block_n = 32
    block_d = 32
    _centroid_sum_kernel[(triton.cdiv(tokens, block_n), triton.cdiv(dim, block_d), batch)](
        x.contiguous(),
        labels.contiguous(),
        sums,
        TOKENS=tokens,
        DIM=dim,
        CLUSTERS=clusters,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return _finalize(sums, sizes, empty_fallback)


def joint_original_centroids_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    sizes: torch.Tensor,
    *,
    x_empty_fallback: torch.Tensor | None = None,
    y_empty_fallback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, dim, clusters = _validate(x, labels, sizes, x_empty_fallback)
    _validate(y, labels, sizes, y_empty_fallback)
    if y.shape != x.shape:
        raise RuntimeError(f"Joint centroid inputs differ: X={tuple(x.shape)}, Y={tuple(y.shape)}.")
    x_sums = torch.zeros((batch, clusters, dim), device=x.device, dtype=torch.float32)
    y_sums = torch.zeros_like(x_sums)
    block_n = 32
    block_d = 32
    _joint_centroid_sum_kernel[(triton.cdiv(tokens, block_n), triton.cdiv(dim, block_d), batch)](
        x.contiguous(),
        y.contiguous(),
        labels.contiguous(),
        x_sums,
        y_sums,
        TOKENS=tokens,
        DIM=dim,
        CLUSTERS=clusters,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return (
        _finalize(x_sums, sizes, x_empty_fallback),
        _finalize(y_sums, sizes, y_empty_fallback),
    )
