"""Fused fixed-shape CUDA epilogue for Cosmos3 N8 residual repair."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _n8_repair_apply_kernel(
    BASE,
    ROLE,
    X_MEAN,
    X_SCALE,
    Y_MEAN,
    LEFT,
    BASIS,
    OUT,
    TOKENS: tl.constexpr,
    BASE_DIM: tl.constexpr,
    ROLE_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ALPHA: tl.constexpr,
    NORM_CAP: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BASE_DIM)
    r = tl.arange(0, RANK)
    p = tl.arange(0, ROLE_DIM)
    n_mask = n < TOKENS

    base_ptrs = BASE + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :]
    base = tl.load(base_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    base_mean = tl.load(X_MEAN + pid_bh * (BASE_DIM + ROLE_DIM) + d).to(tl.float32)
    base_scale = tl.load(X_SCALE + pid_bh * (BASE_DIM + ROLE_DIM) + d).to(tl.float32)
    left_base = tl.load(
        LEFT + (pid_bh * (BASE_DIM + ROLE_DIM) + d[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    standardized_base = (base - base_mean[None, :]) / base_scale[None, :]
    if USE_TF32:
        coordinates = tl.dot(standardized_base, left_base, input_precision="tf32")
    else:
        coordinates = tl.dot(standardized_base, left_base, input_precision="ieee")

    role_ptrs = ROLE + (pid_bh * TOKENS + n[:, None]) * ROLE_DIM + p[None, :]
    role = tl.load(role_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    role_mean = tl.load(
        X_MEAN + pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p
    ).to(tl.float32)
    role_scale = tl.load(
        X_SCALE + pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p
    ).to(tl.float32)
    left_role = tl.load(
        LEFT
        + (pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p[:, None]) * RANK
        + r[None, :]
    ).to(tl.float32)
    standardized_role = (role - role_mean[None, :]) / role_scale[None, :]
    if USE_TF32:
        coordinates += tl.dot(standardized_role, left_role, input_precision="tf32")
    else:
        coordinates += tl.dot(standardized_role, left_role, input_precision="ieee")

    basis = tl.load(
        BASIS + (pid_bh * BASE_DIM + d[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    if USE_TF32:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="tf32")
    else:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="ieee")
    correction += tl.load(Y_MEAN + pid_bh * BASE_DIM + d)[None, :].to(tl.float32)

    if NORM_CAP > 0.0:
        base_norm = tl.sqrt(tl.sum(base * base, axis=1))
        correction_norm = tl.sqrt(tl.sum(correction * correction, axis=1))
        scale = tl.minimum(
            1.0,
            NORM_CAP * base_norm / tl.maximum(correction_norm, 1.0e-20),
        )
        correction *= scale[:, None]

    out = base + ALPHA * correction
    tl.store(
        OUT + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :],
        out,
        mask=n_mask[:, None],
    )


@triton.jit
def _n8_repair_apply_output_kernel(
    BASE,
    X_MEAN,
    X_SCALE,
    Y_MEAN,
    LEFT,
    BASIS,
    OUT,
    TOKENS: tl.constexpr,
    BASE_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ALPHA: tl.constexpr,
    NORM_CAP: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply output-only residual repair without loading role features."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BASE_DIM)
    r = tl.arange(0, RANK)
    n_mask = n < TOKENS

    base_ptrs = BASE + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :]
    base = tl.load(base_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    base_mean = tl.load(X_MEAN + pid_bh * BASE_DIM + d).to(tl.float32)
    base_scale = tl.load(X_SCALE + pid_bh * BASE_DIM + d).to(tl.float32)
    left = tl.load(
        LEFT + (pid_bh * BASE_DIM + d[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    standardized = (base - base_mean[None, :]) / base_scale[None, :]
    if USE_TF32:
        coordinates = tl.dot(standardized, left, input_precision="tf32")
    else:
        coordinates = tl.dot(standardized, left, input_precision="ieee")

    basis = tl.load(
        BASIS + (pid_bh * BASE_DIM + d[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    if USE_TF32:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="tf32")
    else:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="ieee")
    correction += tl.load(Y_MEAN + pid_bh * BASE_DIM + d)[None, :].to(tl.float32)

    if NORM_CAP > 0.0:
        base_norm = tl.sqrt(tl.sum(base * base, axis=1))
        correction_norm = tl.sqrt(tl.sum(correction * correction, axis=1))
        scale = tl.minimum(
            1.0,
            NORM_CAP * base_norm / tl.maximum(correction_norm, 1.0e-20),
        )
        correction *= scale[:, None]

    tl.store(
        OUT + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :],
        base + ALPHA * correction,
        mask=n_mask[:, None],
    )


@triton.jit
def _n8_repair_apply_exact_kernel(
    BASE,
    ROLE,
    X_MEAN,
    X_SCALE,
    Y_MEAN,
    LEFT,
    BASIS,
    PROBE_ROWS,
    DENSE_PROBES,
    OUT,
    TOKENS: tl.constexpr,
    PROBES: tl.constexpr,
    BASE_DIM: tl.constexpr,
    ROLE_DIM: tl.constexpr,
    RANK: tl.constexpr,
    ALPHA: tl.constexpr,
    NORM_CAP: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Apply repair and overwrite exact probe rows in the same launch."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.arange(0, BASE_DIM)
    r = tl.arange(0, RANK)
    p = tl.arange(0, ROLE_DIM)
    probe = tl.arange(0, BLOCK_P)
    n_mask = n < TOKENS

    base = tl.load(
        BASE + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    base_mean = tl.load(X_MEAN + pid_bh * (BASE_DIM + ROLE_DIM) + d).to(tl.float32)
    base_scale = tl.load(X_SCALE + pid_bh * (BASE_DIM + ROLE_DIM) + d).to(tl.float32)
    left_base = tl.load(
        LEFT + (pid_bh * (BASE_DIM + ROLE_DIM) + d[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    standardized_base = (base - base_mean[None, :]) / base_scale[None, :]
    if USE_TF32:
        coordinates = tl.dot(standardized_base, left_base, input_precision="tf32")
    else:
        coordinates = tl.dot(standardized_base, left_base, input_precision="ieee")

    role = tl.load(
        ROLE + (pid_bh * TOKENS + n[:, None]) * ROLE_DIM + p[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    role_mean = tl.load(X_MEAN + pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p).to(tl.float32)
    role_scale = tl.load(X_SCALE + pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p).to(tl.float32)
    left_role = tl.load(
        LEFT + (pid_bh * (BASE_DIM + ROLE_DIM) + BASE_DIM + p[:, None]) * RANK + r[None, :]
    ).to(tl.float32)
    standardized_role = (role - role_mean[None, :]) / role_scale[None, :]
    if USE_TF32:
        coordinates += tl.dot(standardized_role, left_role, input_precision="tf32")
    else:
        coordinates += tl.dot(standardized_role, left_role, input_precision="ieee")

    basis = tl.load(BASIS + (pid_bh * BASE_DIM + d[:, None]) * RANK + r[None, :]).to(tl.float32)
    if USE_TF32:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="tf32")
    else:
        correction = tl.dot(coordinates, tl.trans(basis), input_precision="ieee")
    correction += tl.load(Y_MEAN + pid_bh * BASE_DIM + d)[None, :].to(tl.float32)
    if NORM_CAP > 0.0:
        base_norm = tl.sqrt(tl.sum(base * base, axis=1))
        correction_norm = tl.sqrt(tl.sum(correction * correction, axis=1))
        scale = tl.minimum(1.0, NORM_CAP * base_norm / tl.maximum(correction_norm, 1.0e-20))
        correction *= scale[:, None]
    output = base + ALPHA * correction

    rows = tl.load(
        PROBE_ROWS + pid_bh * PROBES + probe,
        mask=probe < PROBES,
        other=-1,
    ).to(tl.int32)
    matches = (n[:, None] == rows[None, :]) & n_mask[:, None] & (probe[None, :] < PROBES)
    probe_index = tl.max(tl.where(matches, probe[None, :], -1), axis=1)
    is_probe = probe_index >= 0
    dense = tl.load(
        DENSE_PROBES + (pid_bh * PROBES + probe_index[:, None]) * BASE_DIM + d[None, :],
        mask=is_probe[:, None],
        other=0.0,
    ).to(tl.float32)
    output = tl.where(is_probe[:, None], dense, output)
    tl.store(
        OUT + (pid_bh * TOKENS + n[:, None]) * BASE_DIM + d[None, :],
        output,
        mask=n_mask[:, None],
    )


@triton.jit
def _n8_exact_probe_scatter_kernel(
    OUT,
    ROWS,
    DENSE_PROBES,
    TOKENS: tl.constexpr,
    PROBES: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    mask = m < PROBES
    rows = tl.load(ROWS + pid_bh * PROBES + m, mask=mask, other=0).to(tl.int32)
    values = tl.load(
        DENSE_PROBES + (pid_bh * PROBES + m[:, None]) * DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    tl.store(
        OUT + (pid_bh * TOKENS + rows[:, None]) * DIM + d[None, :],
        values,
        mask=mask[:, None],
    )


@triton.jit
def _n8_prepare_fit_kernel(
    BASE,
    ROLE,
    DENSE,
    ROWS,
    X_FIT,
    DENSE_FLOAT,
    Y_FIT,
    TOKENS: tl.constexpr,
    PROBES: tl.constexpr,
    BASE_DIM: tl.constexpr,
    ROLE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BASE_DIM)
    p = tl.arange(0, ROLE_DIM)
    mask = m < PROBES
    rows = tl.load(ROWS + pid_bh * PROBES + m, mask=mask, other=0).to(tl.int32)
    base = tl.load(
        BASE + (pid_bh * TOKENS + rows[:, None]) * BASE_DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    role = tl.load(
        ROLE + (pid_bh * TOKENS + rows[:, None]) * ROLE_DIM + p[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    dense = tl.load(
        DENSE + (pid_bh * PROBES + m[:, None]) * BASE_DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_base = X_FIT + (pid_bh * PROBES + m[:, None]) * (BASE_DIM + ROLE_DIM)
    tl.store(x_base + d[None, :], base, mask=mask[:, None])
    tl.store(x_base + BASE_DIM + p[None, :], role, mask=mask[:, None])
    target_offsets = (pid_bh * PROBES + m[:, None]) * BASE_DIM + d[None, :]
    tl.store(DENSE_FLOAT + target_offsets, dense, mask=mask[:, None])
    tl.store(Y_FIT + target_offsets, dense - base, mask=mask[:, None])


@triton.jit
def _n8_prepare_output_fit_kernel(
    BASE,
    DENSE,
    ROWS,
    X_FIT,
    DENSE_FLOAT,
    Y_FIT,
    TOKENS: tl.constexpr,
    PROBES: tl.constexpr,
    BASE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Prepare output-only fit tensors without packing role features."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BASE_DIM)
    mask = m < PROBES
    rows = tl.load(ROWS + pid_bh * PROBES + m, mask=mask, other=0).to(tl.int32)
    base = tl.load(
        BASE + (pid_bh * TOKENS + rows[:, None]) * BASE_DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    dense = tl.load(
        DENSE + (pid_bh * PROBES + m[:, None]) * BASE_DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    offsets = (pid_bh * PROBES + m[:, None]) * BASE_DIM + d[None, :]
    tl.store(X_FIT + offsets, base, mask=mask[:, None])
    tl.store(DENSE_FLOAT + offsets, dense, mask=mask[:, None])
    tl.store(Y_FIT + offsets, dense - base, mask=mask[:, None])


@triton.jit
def _n8_output_norm_partials_kernel(
    OUTPUT,
    PARTIALS,
    TOTAL_ROWS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Accumulate row L2 norms without materializing OUTPUT.float()."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    mask = rows < TOTAL_ROWS
    values = tl.load(
        OUTPUT + rows[:, None] * DIM + d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    norms = tl.sqrt(tl.sum(values * values, axis=1))
    tl.store(PARTIALS + pid, tl.sum(tl.where(mask, norms, 0.0), axis=0))


def n8_repair_apply_triton(
    base: torch.Tensor,
    role: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    left: torch.Tensor,
    basis: torch.Tensor,
    alpha: float,
    norm_cap: float,
    *,
    block_n: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    input_precision: str = "ieee",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the fixed Cosmos3 N8 P=192,D=128,r=16 repair in one launch."""
    if not base.is_cuda:
        raise RuntimeError("n8_repair_apply_triton requires CUDA tensors.")
    bh, tokens, base_dim = base.shape
    role_dim = int(role.shape[2])
    rank = int(left.shape[2])
    if (base_dim, role_dim, rank) != (128, 64, 16):
        raise RuntimeError(
            "The specialized N8 repair kernel requires D=128, role=64, rank=16; "
            f"got D={base_dim}, role={role_dim}, rank={rank}."
        )
    expected = {
        "role": (bh, tokens, role_dim),
        "x_mean": (bh, 1, base_dim + role_dim),
        "x_scale": (bh, 1, base_dim + role_dim),
        "y_mean": (bh, 1, base_dim),
        "left": (bh, base_dim + role_dim, rank),
        "basis": (bh, base_dim, rank),
    }
    actual = {
        "role": tuple(role.shape),
        "x_mean": tuple(x_mean.shape),
        "x_scale": tuple(x_scale.shape),
        "y_mean": tuple(y_mean.shape),
        "left": tuple(left.shape),
        "basis": tuple(basis.shape),
    }
    mismatched = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if mismatched:
        raise RuntimeError(f"Invalid specialized N8 repair shapes: {mismatched}.")
    tensors = (base, role, x_mean, x_scale, y_mean, left, basis)
    if any(not value.is_contiguous() for value in tensors):
        raise RuntimeError("The specialized N8 repair kernel requires contiguous tensors.")

    if block_n not in {8, 16, 32, 64}:
        raise RuntimeError(f"N8 repair BLOCK_N must be one of 8,16,32,64, got {block_n}.")
    if num_warps not in {2, 4, 8}:
        raise RuntimeError(f"N8 repair num_warps must be one of 2,4,8, got {num_warps}.")
    if num_stages not in {1, 2, 3, 4}:
        raise RuntimeError("N8 repair num_stages must be one of 1,2,3,4.")
    if input_precision not in {"ieee", "tf32"}:
        raise RuntimeError("N8 repair input_precision must be 'ieee' or 'tf32'.")
    if out is None:
        out = torch.empty_like(base)
    elif tuple(out.shape) != tuple(base.shape) or out.device != base.device or out.dtype != base.dtype:
        raise RuntimeError(
            f"N8 repair output must match base, got {tuple(out.shape)} {out.dtype} {out.device}."
        )
    elif not out.is_contiguous():
        raise RuntimeError("N8 repair output must be contiguous.")
    _n8_repair_apply_kernel[(triton.cdiv(tokens, block_n), bh)](
        base,
        role,
        x_mean,
        x_scale,
        y_mean,
        left,
        basis,
        out,
        TOKENS=tokens,
        BASE_DIM=base_dim,
        ROLE_DIM=role_dim,
        RANK=rank,
        ALPHA=float(alpha),
        NORM_CAP=float(norm_cap),
        USE_TF32=input_precision == "tf32",
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def n8_repair_apply_output_triton(
    base: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    left: torch.Tensor,
    basis: torch.Tensor,
    alpha: float,
    norm_cap: float,
    *,
    block_n: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    input_precision: str = "ieee",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the fixed D=128,r=16 output-only N8 repair in one launch."""
    if not base.is_cuda:
        raise RuntimeError("n8_repair_apply_output_triton requires CUDA tensors.")
    bh, tokens, base_dim = base.shape
    rank = int(left.shape[2])
    if (base_dim, rank) != (128, 16):
        raise RuntimeError(
            "The output-only N8 repair kernel requires D=128, rank=16; "
            f"got D={base_dim}, rank={rank}."
        )
    expected = {
        "x_mean": (bh, 1, base_dim),
        "x_scale": (bh, 1, base_dim),
        "y_mean": (bh, 1, base_dim),
        "left": (bh, base_dim, rank),
        "basis": (bh, base_dim, rank),
    }
    actual = {
        "x_mean": tuple(x_mean.shape),
        "x_scale": tuple(x_scale.shape),
        "y_mean": tuple(y_mean.shape),
        "left": tuple(left.shape),
        "basis": tuple(basis.shape),
    }
    mismatched = {
        name: (actual[name], shape)
        for name, shape in expected.items()
        if actual[name] != shape
    }
    if mismatched:
        raise RuntimeError(f"Invalid output-only N8 repair shapes: {mismatched}.")
    tensors = (base, x_mean, x_scale, y_mean, left, basis)
    if any(not value.is_contiguous() for value in tensors):
        raise RuntimeError("The output-only N8 repair kernel requires contiguous tensors.")
    if block_n not in {8, 16, 32, 64}:
        raise RuntimeError(f"N8 repair BLOCK_N must be one of 8,16,32,64, got {block_n}.")
    if num_warps not in {2, 4, 8}:
        raise RuntimeError(f"N8 repair num_warps must be one of 2,4,8, got {num_warps}.")
    if num_stages not in {1, 2, 3, 4}:
        raise RuntimeError("N8 repair num_stages must be one of 1,2,3,4.")
    if input_precision not in {"ieee", "tf32"}:
        raise RuntimeError("N8 repair input_precision must be 'ieee' or 'tf32'.")
    if out is None:
        out = torch.empty_like(base)
    elif tuple(out.shape) != tuple(base.shape) or out.device != base.device or out.dtype != base.dtype:
        raise RuntimeError(
            f"N8 repair output must match base, got {tuple(out.shape)} {out.dtype} {out.device}."
        )
    elif not out.is_contiguous():
        raise RuntimeError("N8 repair output must be contiguous.")
    _n8_repair_apply_output_kernel[(triton.cdiv(tokens, block_n), bh)](
        base,
        x_mean,
        x_scale,
        y_mean,
        left,
        basis,
        out,
        TOKENS=tokens,
        BASE_DIM=base_dim,
        RANK=rank,
        ALPHA=float(alpha),
        NORM_CAP=float(norm_cap),
        USE_TF32=input_precision == "tf32",
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def n8_repair_apply_exact_triton(
    base: torch.Tensor,
    role: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    y_mean: torch.Tensor,
    left: torch.Tensor,
    basis: torch.Tensor,
    probe_rows: torch.Tensor,
    dense_probes: torch.Tensor,
    alpha: float,
    norm_cap: float,
    *,
    block_n: int = 16,
    num_warps: int = 2,
    num_stages: int = 1,
    input_precision: str = "tf32",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply N8 repair and exact probe replacement in a single launch."""
    if not base.is_cuda:
        raise RuntimeError("n8_repair_apply_exact_triton requires CUDA tensors.")
    bh, tokens, base_dim = base.shape
    role_dim = int(role.shape[2])
    rank = int(left.shape[2])
    probes = int(probe_rows.shape[1])
    if (base_dim, role_dim, rank) != (128, 64, 16):
        raise RuntimeError("Fused N8 repair/exact replacement requires D=128, role=64, rank=16.")
    expected = {
        "role": (bh, tokens, role_dim),
        "x_mean": (bh, 1, base_dim + role_dim),
        "x_scale": (bh, 1, base_dim + role_dim),
        "y_mean": (bh, 1, base_dim),
        "left": (bh, base_dim + role_dim, rank),
        "basis": (bh, base_dim, rank),
        "probe_rows": (bh, probes),
        "dense_probes": (bh, probes, base_dim),
    }
    actual = {
        "role": tuple(role.shape),
        "x_mean": tuple(x_mean.shape),
        "x_scale": tuple(x_scale.shape),
        "y_mean": tuple(y_mean.shape),
        "left": tuple(left.shape),
        "basis": tuple(basis.shape),
        "probe_rows": tuple(probe_rows.shape),
        "dense_probes": tuple(dense_probes.shape),
    }
    mismatched = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if mismatched:
        raise RuntimeError(f"Invalid fused N8 repair shapes: {mismatched}.")
    if probes <= 0 or probes > 128:
        raise RuntimeError("Fused exact repair supports 1..128 probes per head.")
    if block_n not in {16, 32} or num_warps not in {2, 4, 8} or num_stages not in {1, 2, 3}:
        raise RuntimeError("Unsupported fused repair launch configuration.")
    if input_precision not in {"ieee", "tf32"}:
        raise RuntimeError("input_precision must be ieee or tf32.")
    tensors = (base, role, x_mean, x_scale, y_mean, left, basis, probe_rows, dense_probes)
    tensors = tuple(value.contiguous() for value in tensors)
    base, role, x_mean, x_scale, y_mean, left, basis, probe_rows, dense_probes = tensors
    if out is None:
        out = torch.empty_like(base)
    elif out.shape != base.shape or out.device != base.device or out.dtype != base.dtype:
        raise RuntimeError("Fused repair output must match base.")
    _n8_repair_apply_exact_kernel[(triton.cdiv(tokens, block_n), bh)](
        base,
        role,
        x_mean,
        x_scale,
        y_mean,
        left,
        basis,
        probe_rows,
        dense_probes,
        out,
        TOKENS=tokens,
        PROBES=probes,
        BASE_DIM=base_dim,
        ROLE_DIM=role_dim,
        RANK=rank,
        ALPHA=float(alpha),
        NORM_CAP=float(norm_cap),
        USE_TF32=input_precision == "tf32",
        BLOCK_N=block_n,
        BLOCK_P=triton.next_power_of_2(probes),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def n8_exact_probe_scatter_triton(
    output: torch.Tensor,
    rows: torch.Tensor,
    dense_probes: torch.Tensor,
    *,
    block_m: int = 16,
    num_warps: int = 4,
    convert_rows: bool = True,
) -> torch.Tensor:
    """Replace exact probe rows without constructing an expanded scatter index."""
    if not output.is_cuda:
        raise RuntimeError("n8_exact_probe_scatter_triton requires CUDA tensors.")
    if output.ndim != 3 or rows.ndim != 2 or dense_probes.ndim != 3:
        raise RuntimeError(
            "Expected output [BH,N,D], rows [BH,M], and dense probes [BH,M,D]."
        )
    bh, tokens, dim = output.shape
    probes = int(rows.shape[1])
    if tuple(rows.shape) != (bh, probes) or tuple(dense_probes.shape) != (bh, probes, dim):
        raise RuntimeError(
            f"Exact-scatter shape mismatch: output={tuple(output.shape)}, "
            f"rows={tuple(rows.shape)}, probes={tuple(dense_probes.shape)}."
        )
    if rows.device != output.device or dense_probes.device != output.device:
        raise RuntimeError("Exact-scatter tensors must share a CUDA device.")
    if not dense_probes.dtype.is_floating_point:
        raise RuntimeError("Exact-scatter dense probes must be floating point.")
    if block_m not in {8, 16, 32, 64}:
        raise RuntimeError("Exact-scatter BLOCK_M must be one of 8,16,32,64.")
    if num_warps not in {1, 2, 4, 8}:
        raise RuntimeError("Exact-scatter num_warps must be one of 1,2,4,8.")
    rows_input = rows.to(torch.int32).contiguous() if convert_rows else rows.contiguous()
    probes_contiguous = dense_probes.contiguous()
    _n8_exact_probe_scatter_kernel[(triton.cdiv(probes, block_m), bh)](
        output,
        rows_input,
        probes_contiguous,
        TOKENS=tokens,
        PROBES=probes,
        DIM=dim,
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def n8_output_norm_mean_triton(
    output: torch.Tensor,
    *,
    block_m: int = 32,
    partials: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean row L2 norm without a full FP32 output temporary."""
    if not output.is_cuda or output.ndim != 3 or not output.is_contiguous():
        raise RuntimeError("Output-norm candidate requires contiguous CUDA [BH,N,D].")
    bh, tokens, dim = output.shape
    if dim != 128:
        raise RuntimeError(f"Output-norm candidate requires D=128, got {dim}.")
    if block_m not in {8, 16, 32, 64}:
        raise RuntimeError("Output-norm BLOCK_M must be one of 8,16,32,64.")
    total_rows = bh * tokens
    programs = triton.cdiv(total_rows, block_m)
    if partials is None:
        partials = torch.empty(programs, device=output.device, dtype=torch.float32)
    elif tuple(partials.shape) != (programs,) or partials.device != output.device:
        raise RuntimeError(
            f"Output-norm partials must have shape {(programs,)}, got {tuple(partials.shape)}."
        )
    _n8_output_norm_partials_kernel[(programs,)](
        output,
        partials,
        TOTAL_ROWS=total_rows,
        DIM=dim,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return partials.sum() / float(total_rows)


def n8_prepare_repair_fit_triton(
    base: torch.Tensor,
    role: torch.Tensor,
    dense_probes: torch.Tensor,
    rows: torch.Tensor,
    x_fit: torch.Tensor,
    dense_float: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    block_m: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse probe gathers, feature packing, FP32 conversion, and target subtraction."""
    bh, tokens, base_dim = base.shape
    role_dim = int(role.shape[2])
    probes = int(rows.shape[1])
    expected = {
        "role": (bh, tokens, role_dim),
        "dense": (bh, probes, base_dim),
        "rows": (bh, probes),
        "x_fit": (bh, probes, base_dim + role_dim),
        "dense_float": (bh, probes, base_dim),
        "y_fit": (bh, probes, base_dim),
    }
    actual = {
        "role": tuple(role.shape),
        "dense": tuple(dense_probes.shape),
        "rows": tuple(rows.shape),
        "x_fit": tuple(x_fit.shape),
        "dense_float": tuple(dense_float.shape),
        "y_fit": tuple(y_fit.shape),
    }
    mismatched = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if mismatched:
        raise RuntimeError(f"Invalid fused fit-preparation shapes: {mismatched}.")
    if not base.is_cuda or any(
        value.device != base.device
        for value in (role, dense_probes, rows, x_fit, dense_float, y_fit)
    ):
        raise RuntimeError("Fused fit preparation requires one CUDA device.")
    if any(not value.is_contiguous() for value in (base, role, dense_probes, x_fit, dense_float, y_fit)):
        raise RuntimeError("Fused fit preparation requires contiguous tensors.")
    rows_i32 = rows.to(torch.int32).contiguous()
    _n8_prepare_fit_kernel[(triton.cdiv(probes, block_m), bh)](
        base,
        role,
        dense_probes,
        rows_i32,
        x_fit,
        dense_float,
        y_fit,
        TOKENS=tokens,
        PROBES=probes,
        BASE_DIM=base_dim,
        ROLE_DIM=role_dim,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return x_fit, dense_float, y_fit


def n8_prepare_output_repair_fit_triton(
    base: torch.Tensor,
    dense_probes: torch.Tensor,
    rows: torch.Tensor,
    x_fit: torch.Tensor,
    dense_float: torch.Tensor,
    y_fit: torch.Tensor,
    *,
    block_m: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse output-only probe gathers, conversion, and target subtraction."""
    bh, tokens, base_dim = base.shape
    probes = int(rows.shape[1])
    expected = {
        "dense": (bh, probes, base_dim),
        "rows": (bh, probes),
        "x_fit": (bh, probes, base_dim),
        "dense_float": (bh, probes, base_dim),
        "y_fit": (bh, probes, base_dim),
    }
    actual = {
        "dense": tuple(dense_probes.shape),
        "rows": tuple(rows.shape),
        "x_fit": tuple(x_fit.shape),
        "dense_float": tuple(dense_float.shape),
        "y_fit": tuple(y_fit.shape),
    }
    mismatched = {
        name: (actual[name], shape)
        for name, shape in expected.items()
        if actual[name] != shape
    }
    if mismatched:
        raise RuntimeError(f"Invalid output-only fit-preparation shapes: {mismatched}.")
    if not base.is_cuda or any(
        value.device != base.device
        for value in (dense_probes, rows, x_fit, dense_float, y_fit)
    ):
        raise RuntimeError("Output-only fit preparation requires one CUDA device.")
    if any(
        not value.is_contiguous()
        for value in (base, dense_probes, x_fit, dense_float, y_fit)
    ):
        raise RuntimeError("Output-only fit preparation requires contiguous tensors.")
    rows_i32 = rows.to(torch.int32).contiguous()
    _n8_prepare_output_fit_kernel[(triton.cdiv(probes, block_m), bh)](
        base,
        dense_probes,
        rows_i32,
        x_fit,
        dense_float,
        y_fit,
        TOKENS=tokens,
        PROBES=probes,
        BASE_DIM=base_dim,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return x_fit, dense_float, y_fit
