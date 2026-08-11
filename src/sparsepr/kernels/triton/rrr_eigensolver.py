"""Fixed-shape top-16 subspace solver for the N8 64x64 RRR row Gram."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_Q0_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


@triton.jit
def _top16_subspace_kernel(
    A,
    Q0,
    Q_OUT,
    N: tl.constexpr,
    R: tl.constexpr,
    POWER_ITERS: tl.constexpr,
    INV_SQRT_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    r0 = tl.arange(0, R)
    r1 = tl.arange(0, R)
    a = tl.load(A + (pid * N + i[:, None]) * N + j[None, :]).to(tl.float32)
    q = tl.load(Q0 + i[:, None] * R + r0[None, :]).to(tl.float32)
    eye = (r0[:, None] == r1[None, :]).to(tl.float32)

    for _ in range(POWER_ITERS):
        y = tl.dot(a, q, input_precision="ieee")
        gram = tl.dot(tl.trans(y), y, input_precision="ieee")
        scale = tl.maximum(tl.sqrt(tl.sum(gram * gram)), 1.0e-20)
        normalized = gram / scale
        inv_sqrt = eye
        for _ in range(INV_SQRT_ITERS):
            squared = tl.dot(inv_sqrt, inv_sqrt, input_precision="ieee")
            update = 3.0 * eye - tl.dot(normalized, squared, input_precision="ieee")
            inv_sqrt = 0.5 * tl.dot(inv_sqrt, update, input_precision="ieee")
        q = tl.dot(y, inv_sqrt, input_precision="ieee") / tl.sqrt(scale)

    tl.store(Q_OUT + (pid * N + i[:, None]) * R + r0[None, :], q)


@triton.jit
def _orthonormalize_basis_kernel(
    BASIS,
    OUT,
    D: tl.constexpr,
    R: tl.constexpr,
    INV_SQRT_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, D)
    r0 = tl.arange(0, R)
    r1 = tl.arange(0, R)
    basis = tl.load(BASIS + (pid * D + d[:, None]) * R + r0[None, :]).to(tl.float32)
    gram = tl.dot(tl.trans(basis), basis, input_precision="ieee")
    eye = (r0[:, None] == r1[None, :]).to(tl.float32)
    scale = tl.maximum(tl.sqrt(tl.sum(gram * gram)), 1.0e-20)
    normalized = gram / scale
    inv_sqrt = eye
    for _ in range(INV_SQRT_ITERS):
        squared = tl.dot(inv_sqrt, inv_sqrt, input_precision="ieee")
        update = 3.0 * eye - tl.dot(normalized, squared, input_precision="ieee")
        inv_sqrt = 0.5 * tl.dot(inv_sqrt, update, input_precision="ieee")
    result = tl.dot(basis, inv_sqrt, input_precision="ieee") / tl.sqrt(scale)
    tl.store(OUT + (pid * D + d[:, None]) * R + r0[None, :], result)


def _initial_subspace(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    cached = _Q0_CACHE.get(key)
    if cached is None:
        rows = torch.arange(64, device=device, dtype=torch.float32).unsqueeze(1)
        cols = torch.arange(16, device=device, dtype=torch.float32).unsqueeze(0)
        seed = torch.cos((rows + 0.5) * (cols + 1.0) * (math.pi / 64.0))
        cached, _ = torch.linalg.qr(seed, mode="reduced")
        _Q0_CACHE[key] = cached.contiguous()
    return cached


def top16_right_basis_triton(
    weighted_fitted: torch.Tensor,
    *,
    power_iters: int = 8,
    inv_sqrt_iters: int = 6,
) -> torch.Tensor:
    """Approximate the top-16 right singular subspace of [BH,64,128]."""
    if not weighted_fitted.is_cuda:
        raise RuntimeError("top16_right_basis_triton requires CUDA tensors.")
    if tuple(weighted_fitted.shape[1:]) != (64, 128):
        raise RuntimeError(
            "N8 top-16 kernel requires [BH,64,128], got "
            f"{tuple(weighted_fitted.shape)}."
        )
    if power_iters <= 0 or inv_sqrt_iters <= 0:
        raise RuntimeError("Subspace iteration counts must be positive.")
    values = weighted_fitted.float().contiguous()
    bh = int(values.shape[0])
    row_gram = torch.bmm(values, values.transpose(1, 2)).contiguous()
    q = torch.empty((bh, 64, 16), device=values.device, dtype=torch.float32)
    _top16_subspace_kernel[(bh,)](
        row_gram,
        _initial_subspace(values.device),
        q,
        N=64,
        R=16,
        POWER_ITERS=int(power_iters),
        INV_SQRT_ITERS=int(inv_sqrt_iters),
        num_warps=8,
        num_stages=1,
    )
    raw_basis = torch.bmm(values.transpose(1, 2), q).contiguous()
    basis = torch.empty_like(raw_basis)
    _orthonormalize_basis_kernel[(bh,)](
        raw_basis,
        basis,
        D=128,
        R=16,
        INV_SQRT_ITERS=int(inv_sqrt_iters),
        num_warps=8,
        num_stages=1,
    )
    return basis
