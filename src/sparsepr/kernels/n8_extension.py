"""Lazy loader and guarded Python facade for ``n8_kernels.so``."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import torch

_MODULE: ModuleType | None = None
_LOAD_ERROR: Exception | None = None


def build_directory() -> Path:
    override = os.environ.get("SPARSEPR_N8_CUDA_EXT_BUILD", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "n8_ext" / "build"


def load_n8_kernels(*, required: bool = False) -> ModuleType | None:
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None and not required:
        return None
    directory = build_directory()
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    try:
        _MODULE = importlib.import_module("n8_kernels")
    except Exception as exc:
        _LOAD_ERROR = exc
        if required:
            raise RuntimeError(
                f"Could not load n8_kernels.so from {directory}: {exc}"
            ) from exc
        return None
    return _MODULE


def n8_cuda_extension_available() -> bool:
    return load_n8_kernels(required=False) is not None


def wan_qkv_norm_rope_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freq_real: torch.Tensor,
    freq_imag: torch.Tensor,
    *,
    heads: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = load_n8_kernels(required=True)
    assert module is not None
    return tuple(
        module.wan_qkv_norm_rope_layout(
            q,
            k,
            v,
            q_weight,
            k_weight,
            freq_real,
            freq_imag,
            int(heads),
            float(epsilon),
        )
    )


def cosmos25_qkv_norm_rope_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_angles: torch.Tensor,
    *,
    heads: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Cosmos2.5 per-head RMSNorm and TE-style RoPE.

    Projected Q/K/V enter as ``[B,S,H*D]`` and leave as official Cosmos
    ``[B,S,H,D]`` tensors. Transformer Engine keeps RMSNorm parameters in
    FP32, so the facade normalizes that small parameter representation before
    entering the CUDA extension.
    """

    module = load_n8_kernels(required=True)
    assert module is not None
    if not hasattr(module, "cosmos25_qkv_norm_rope_bshd"):
        raise RuntimeError(
            "Loaded n8_kernels.so predates the Cosmos2.5 QKV entry point; "
            "rebuild the extension in the Cosmos2.5 environment."
        )
    return tuple(
        module.cosmos25_qkv_norm_rope_bshd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            q_weight.float().contiguous(),
            k_weight.float().contiguous(),
            rope_angles.float().contiguous(),
            int(heads),
            float(epsilon),
        )
    )


def role_cluster_step_q64(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = load_n8_kernels(required=True)
    assert module is not None
    return tuple(module.role_cluster_step_q64(features, centroids))


def role_cluster_step_k64(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = load_n8_kernels(required=True)
    assert module is not None
    return tuple(module.role_cluster_step_k64(features, centroids))


def svg_ear_select_budget(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    module = load_n8_kernels(required=True)
    assert module is not None
    return module.svg_ear_select_budget(
        scores.contiguous(), cluster_sizes, budgets
    )


def compact_selected_clusters(
    selected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ascending fixed-stride cluster IDs and a count for every row."""
    module = load_n8_kernels(required=True)
    assert module is not None
    if not hasattr(module, "compact_selected_clusters"):
        raise RuntimeError(
            "Loaded n8_kernels.so predates compact selected-cluster schedules; "
            "rebuild the extension."
        )
    return tuple(module.compact_selected_clusters(selected_mask.contiguous()))


def svg_ear_select_budget_schedule(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the exact support and emit its executor schedule in one call."""
    module = load_n8_kernels(required=True)
    assert module is not None
    if not hasattr(module, "svg_ear_select_budget_schedule"):
        raise RuntimeError(
            "Loaded n8_kernels.so predates fused selector schedules; "
            "rebuild the extension."
        )
    return tuple(
        module.svg_ear_select_budget_schedule(
            scores.contiguous(), cluster_sizes, budgets
        )
    )
