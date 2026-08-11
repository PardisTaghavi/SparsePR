"""Triton candidates for native Wan2.2 block normalization/modulation work.

The public functions accept and return regular ``torch.Tensor`` objects.
Triton only vectorizes their rows while the kernel is executing.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _wan_layer_norm_modulate_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows_per_batch: tl.constexpr,
    scale_batch_stride: tl.constexpr,
    scale_token_stride: tl.constexpr,
    shift_batch_stride: tl.constexpr,
    shift_token_stride: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block_width: tl.constexpr,
    has_affine: tl.constexpr,
    has_modulation: tl.constexpr,
    round_norm_to_input: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_width)
    mask = columns < width
    row_offsets = row * width + columns

    values = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(values, axis=0) / width
    centered = values - mean
    variance = tl.sum(centered * centered, axis=0) / width
    normalized = centered * tl.rsqrt(variance + epsilon)

    if has_affine:
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        bias = tl.load(bias_ptr + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        normalized = normalized * weight + bias

    # Native WanLayerNorm returns type_as(x) before the following explicit
    # float() modulation. Preserve that rounding boundary for BF16/FP16 input.
    if round_norm_to_input:
        normalized = normalized.to(x_ptr.type.element_ty).to(tl.float32)

    if has_modulation:
        # Native Wan stores modulation as [B, tokens, 6, dim]. A component
        # view therefore has a batch stride above INT32_MAX at 720p. Keep
        # modulation pointer arithmetic in int64 even though the row count
        # itself fits in int32.
        row_i64 = row.to(tl.int64)
        batch = row_i64 // rows_per_batch
        token = row_i64 % rows_per_batch
        scale_offsets = (
            batch * scale_batch_stride
            + token * scale_token_stride
            + columns
        )
        shift_offsets = (
            batch * shift_batch_stride
            + token * shift_token_stride
            + columns
        )
        scale = tl.load(
            scale_ptr + scale_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        shift = tl.load(
            shift_ptr + shift_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        normalized = normalized * (1.0 + scale) + shift

    tl.store(output_ptr + row_offsets, normalized, mask=mask)


def wan_layer_norm_modulate(
    x: torch.Tensor,
    *,
    epsilon: float,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fuse native Wan LayerNorm and optional AdaLN scale/shift."""
    if not x.is_cuda:
        raise ValueError("Wan Triton block fusion requires a CUDA tensor.")
    if x.ndim != 3 or not x.is_contiguous():
        raise ValueError("x must be contiguous with shape [batch, tokens, dim].")
    batch, tokens, width = x.shape
    has_modulation = scale is not None or shift is not None
    if has_modulation and (scale is None or shift is None):
        raise ValueError("scale and shift must either both be set or both be None.")
    has_affine = weight is not None or bias is not None
    if has_affine and (weight is None or bias is None):
        raise ValueError("weight and bias must either both be set or both be None.")

    if has_modulation:
        batch_modulation_elements = batch * width
        token_modulation_elements = batch * tokens * width
        if scale.numel() == batch_modulation_elements:
            tokenwise_modulation = False
        elif scale.numel() == token_modulation_elements:
            tokenwise_modulation = True
        else:
            raise ValueError(
                "scale and shift must contain either one vector per batch "
                f"({batch_modulation_elements} elements) or one vector per "
                f"token ({token_modulation_elements} elements)."
            )
        if shift.numel() != scale.numel():
            raise ValueError("scale and shift must have identical element counts.")
        (
            scale_batch_stride,
            scale_token_stride,
        ) = _modulation_strides(
            scale,
            batch=batch,
            tokens=tokens,
            width=width,
            tokenwise=tokenwise_modulation,
            name="scale",
        )
        (
            shift_batch_stride,
            shift_token_stride,
        ) = _modulation_strides(
            shift,
            batch=batch,
            tokens=tokens,
            width=width,
            tokenwise=tokenwise_modulation,
            name="shift",
        )
    else:
        tokenwise_modulation = False
        scale = x
        shift = x
        scale_batch_stride = scale_token_stride = 0
        shift_batch_stride = shift_token_stride = 0
    if has_affine:
        weight = weight.reshape(width).contiguous()
        bias = bias.reshape(width).contiguous()
    else:
        weight = x
        bias = x

    output = torch.empty_like(x, dtype=output_dtype)
    block_width = triton.next_power_of_2(width)
    if block_width > 65536:
        raise ValueError(f"Wan hidden dimension {width} is too large for fusion.")
    _wan_layer_norm_modulate_kernel[(batch * tokens,)](
        x,
        scale,
        shift,
        weight,
        bias,
        output,
        rows_per_batch=tokens,
        scale_batch_stride=scale_batch_stride,
        scale_token_stride=scale_token_stride,
        shift_batch_stride=shift_batch_stride,
        shift_token_stride=shift_token_stride,
        width=width,
        epsilon=float(epsilon),
        block_width=block_width,
        has_affine=has_affine,
        has_modulation=has_modulation,
        round_norm_to_input=x.dtype != torch.float32,
        num_warps=8,
    )
    return output


def _modulation_strides(
    tensor: torch.Tensor,
    *,
    batch: int,
    tokens: int,
    width: int,
    tokenwise: bool,
    name: str,
) -> tuple[int, int]:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if tensor.shape[-1] != width or tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must have a contiguous final dimension of width {width}."
        )
    if tokenwise:
        if tensor.ndim != 3 or tensor.shape[:2] != (batch, tokens):
            raise ValueError(
                f"tokenwise {name} must have shape "
                f"[{batch}, {tokens}, {width}], got {tuple(tensor.shape)}."
            )
        return int(tensor.stride(0)), int(tensor.stride(1))
    if tensor.ndim == 2 and tensor.shape == (batch, width):
        return int(tensor.stride(0)), 0
    if tensor.ndim == 3 and tensor.shape == (batch, 1, width):
        return int(tensor.stride(0)), 0
    raise ValueError(
        f"batchwise {name} must have shape [{batch}, {width}] or "
        f"[{batch}, 1, {width}], got {tuple(tensor.shape)}."
    )


@triton.jit
def _wan_gate_residual_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    output_ptr,
    rows_per_batch: tl.constexpr,
    gate_batch_stride: tl.constexpr,
    gate_token_stride: tl.constexpr,
    width: tl.constexpr,
    block_width: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_width)
    mask = columns < width
    row_offsets = row * width + columns
    row_i64 = row.to(tl.int64)
    batch = row_i64 // rows_per_batch
    token = row_i64 % rows_per_batch
    gate_offsets = (
        batch * gate_batch_stride + token * gate_token_stride + columns
    )

    residual = tl.load(
        residual_ptr + row_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    update = tl.load(update_ptr + row_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        output_ptr + row_offsets,
        residual + update * gate,
        mask=mask,
    )


def wan_gate_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse ``residual.float() + update.float() * gate.float()``."""
    if not residual.is_cuda or not update.is_cuda or not gate.is_cuda:
        raise ValueError("Wan Triton block fusion requires CUDA tensors.")
    if (
        residual.ndim != 3
        or update.shape != residual.shape
        or not residual.is_contiguous()
        or not update.is_contiguous()
    ):
        raise ValueError(
            "residual and update must be contiguous tensors with identical "
            "shape [batch, tokens, dim]."
        )
    batch, tokens, width = residual.shape
    batch_gate_elements = batch * width
    token_gate_elements = batch * tokens * width
    if gate.numel() == batch_gate_elements:
        tokenwise_gate = False
    elif gate.numel() == token_gate_elements:
        tokenwise_gate = True
    else:
        raise ValueError(
            "gate must contain either one vector per batch "
            f"({batch_gate_elements} elements) or one vector per token "
            f"({token_gate_elements} elements)."
        )
    gate_batch_stride, gate_token_stride = _modulation_strides(
        gate,
        batch=batch,
        tokens=tokens,
        width=width,
        tokenwise=tokenwise_gate,
        name="gate",
    )
    output = torch.empty_like(residual, dtype=torch.float32)
    block_width = triton.next_power_of_2(width)
    if block_width > 65536:
        raise ValueError(f"Wan hidden dimension {width} is too large for fusion.")
    _wan_gate_residual_kernel[(batch * tokens,)](
        residual,
        update,
        gate,
        output,
        rows_per_batch=tokens,
        gate_batch_stride=gate_batch_stride,
        gate_token_stride=gate_token_stride,
        width=width,
        block_width=block_width,
        num_warps=8,
    )
    return output
