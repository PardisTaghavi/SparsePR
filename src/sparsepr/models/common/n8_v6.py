"""Model-independent N8-V6 role routing and current-step repair.

This module deliberately knows nothing about Cosmos or Hunyuan transformer
blocks.  It consumes already-normalized ``[B,H,N,D]`` Q/K/V tensors and owns
the N8 method state: role metrics/factors, role-space Flash K-means, exact
fixed-density or SVG-EAR TopP-budgeted routing, selectable current-step probes,
and rank-16 residual repair. Model adapters remain responsible for QKV
projection/RoPE, special token layout, the sparse attention call, and output
projection.

The operator and RRR primitives predate this common facade and currently live
under ``models.cosmos3``.  They are shape-generic pure tensor routines; this
facade is the only interface new model adapters should import.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...kernels.triton.permute import permute_tensor_by_labels_triton
from ...kernels.triton.residual_repair import (
    n8_exact_probe_scatter_triton,
    n8_prepare_output_repair_fit_triton,
    n8_prepare_repair_fit_triton,
    n8_repair_apply_output_triton,
    n8_repair_apply_triton,
)
from ...kernels.triton.role_projection import role_projection_rms_triton
from ...kernels.triton.selector_budget import exact_budget_dynamic_map_triton
from ...kernels.triton.svg_ear import (
    attention_mass_scores_triton,
    svg_ear_grouped_value_error_scores_triton,
    svg_ear_value_error_scores_triton,
)
from ...kmeans_utils import batch_kmeans_Euclid, pop_last_kmeans_timings
from ..cosmos3.operator_kmeans import (
    gqa_metric_sources,
    joint_original_centroids,
    key_metric_from_queries,
    metric_factor,
    normalize_feature_scale,
    normalize_rows_rms,
    original_centroids,
    project_tokens,
    query_metric_from_key_centroids,
    value_metric_from_values,
    warm_subspace_metric_factor,
)
from ..cosmos3.residual_repair import (
    apply_reduced_rank_residual,
    fit_reduced_rank_residual,
    recover_nonfinite_rrr_state,
    select_first_token_largest_groups,
    select_role_equal_probes_vectorized,
)

_COMPILED_ROLE_KV_PROJECTION = None
_COMPILED_ROLE_KV_INIT_PROJECTION = None
_COMPILED_ROLE_K_PROJECTION = None
_COMPILED_ROLE_K_INIT_PROJECTION = None
_N8_CUDA_EXT_SELECTOR = os.environ.get("SPARSEPR_N8_CUDA_EXT_SELECTOR", "0").lower() not in {
    "", "0", "false", "no", "off"
}
_N8_CUDA_EXT_STRICT = os.environ.get("SPARSEPR_N8_CUDA_EXT_STRICT", "0").lower() not in {
    "", "0", "false", "no", "off"
}
_N8_CUDA_EXT_SELECTOR_WARNED = False


def _run_with_fp32_matmul_precision(precision: str, fn, *args):
    """Launch an isolated FP32 GEMM path with the offline-gated precision."""
    previous = torch.get_float32_matmul_precision()
    if previous == precision:
        return fn(*args)
    torch.set_float32_matmul_precision(precision)
    try:
        return fn(*args)
    finally:
        torch.set_float32_matmul_precision(previous)


def _role_kv_projection_normalize_fp32(
    k: torch.Tensor,
    v: torch.Tensor,
    k_factor: torch.Tensor,
    v_factor: torch.Tensor,
    v_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse exact FP32 K/V projections, normalization, and feature packing."""
    key = torch.bmm(k.float(), k_factor.float())
    value = torch.bmm(v.float(), v_factor.float())
    key_scale = key.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    value_scale = value.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    features = torch.cat((key / key_scale, float(v_weight) * value / value_scale), dim=2)
    return features, key_scale, value_scale


def _compiled_role_kv_projection_normalize(*args):
    global _COMPILED_ROLE_KV_PROJECTION
    if _COMPILED_ROLE_KV_PROJECTION is None:
        _COMPILED_ROLE_KV_PROJECTION = torch.compile(
            _role_kv_projection_normalize_fp32,
            fullgraph=True,
            dynamic=False,
        )
    return _COMPILED_ROLE_KV_PROJECTION(*args)


def _role_kv_init_projection_fp32(
    k_centroids: torch.Tensor,
    v_centroids: torch.Tensor,
    k_factor: torch.Tensor,
    v_factor: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    v_weight: float,
) -> torch.Tensor:
    key = torch.bmm(k_centroids.float(), k_factor.float()) / key_scale
    value = torch.bmm(v_centroids.float(), v_factor.float()) / value_scale
    return torch.cat((key, float(v_weight) * value), dim=2)


def _compiled_role_kv_init_projection(*args):
    global _COMPILED_ROLE_KV_INIT_PROJECTION
    if _COMPILED_ROLE_KV_INIT_PROJECTION is None:
        _COMPILED_ROLE_KV_INIT_PROJECTION = torch.compile(
            _role_kv_init_projection_fp32,
            fullgraph=True,
            dynamic=False,
        )
    return _COMPILED_ROLE_KV_INIT_PROJECTION(*args)


def _role_k_projection_normalize_fp32(
    k: torch.Tensor,
    k_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project and normalize K without constructing the removed V branch."""
    key = torch.bmm(k.float(), k_factor.float())
    key_scale = key.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return key / key_scale, key_scale


def _compiled_role_k_projection_normalize(*args):
    global _COMPILED_ROLE_K_PROJECTION
    if _COMPILED_ROLE_K_PROJECTION is None:
        _COMPILED_ROLE_K_PROJECTION = torch.compile(
            _role_k_projection_normalize_fp32,
            fullgraph=True,
            dynamic=False,
        )
    return _COMPILED_ROLE_K_PROJECTION(*args)


def _role_k_init_projection_fp32(
    k_centroids: torch.Tensor,
    k_factor: torch.Tensor,
    key_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(k_centroids.float(), k_factor.float()) / key_scale


def _compiled_role_k_init_projection(*args):
    global _COMPILED_ROLE_K_INIT_PROJECTION
    if _COMPILED_ROLE_K_INIT_PROJECTION is None:
        _COMPILED_ROLE_K_INIT_PROJECTION = torch.compile(
            _role_k_init_projection_fp32,
            fullgraph=True,
            dynamic=False,
        )
    return _COMPILED_ROLE_K_INIT_PROJECTION(*args)


def _svg_top_p_key_budgets(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Derive SVG2-style adaptive key-token budgets for each query cluster.

    The cumulative-mass mask includes the first cluster that crosses ``top_p``,
    matching the released SVG/SVG-EAR budget convention. SVG-EAR routing later
    spends these budgets according to error-to-cost scores rather than attention
    mass.
    """
    if q_centroids.ndim != 3 or k_centroids.ndim != 3:
        raise ValueError("TopP budget centroids must be [BH,C,D].")
    if q_centroids.shape[0] != k_centroids.shape[0]:
        raise ValueError("TopP Q/K centroids must have the same head count.")
    if q_centroids.shape[2] != k_centroids.shape[2]:
        raise ValueError("TopP Q/K centroids must have the same head dimension.")
    if tuple(k_cluster_sizes.shape) != tuple(k_centroids.shape[:2]):
        raise ValueError(
            "TopP K-cluster sizes must match K centroids, got "
            f"sizes={tuple(k_cluster_sizes.shape)} centroids={tuple(k_centroids.shape)}."
        )
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"svg_ear_top_p must be in (0,1], got {top_p}.")

    dim = int(q_centroids.shape[2])
    logits = torch.bmm(
        q_centroids.float(),
        k_centroids.float().transpose(1, 2),
    ) / math.sqrt(dim)
    weights = k_cluster_sizes.float().unsqueeze(1)
    logits = logits - logits.amax(dim=2, keepdim=True)
    weighted_exp = logits.exp() * weights
    probabilities = weighted_exp / weighted_exp.sum(dim=2, keepdim=True).clamp_min(
        1.0e-12
    )
    sorted_probabilities, order = torch.sort(
        probabilities, dim=2, descending=True
    )
    cumulative = sorted_probabilities.cumsum(dim=2)
    keep = cumulative <= float(top_p)
    keep = F.pad(keep[:, :, :-1], (1, 0), value=True)
    sorted_sizes = torch.gather(
        k_cluster_sizes.long().unsqueeze(1).expand(-1, q_centroids.shape[1], -1),
        2,
        order,
    )
    return (sorted_sizes * keep.long()).sum(dim=2)


@dataclass(frozen=True)
class N8V6Config:
    """Immutable, portable N8-V6 method configuration."""

    num_q_centroids: int = 300
    num_k_centroids: int = 1000
    target_density: float = 0.22
    role_q_rank: int = 64
    role_k_rank: int = 48
    role_v_rank: int = 16
    role_v_weight: float = 0.0
    metric_sample_tokens: int = 2048
    kmeans_iter_init: int = 25
    kmeans_iter_step: int = 2
    kmeans_backend: str = "flash"
    factor_power_iters: int = 2
    factor_orthogonalization: str = "each_cholesky_qr2"
    factor_ritz_mode: str = "rayleigh_cholesky"
    factor_rayleigh_ridge: float = 1e-4
    factor_cholesky_qr_ridge: float = 1e-5
    factor_refresh_every: int = 1
    probe_rows: int = 64
    repair_rank: int = 16
    repair_ridge: float = 0.1
    repair_alpha: float = 1.0
    repair_norm_cap: float = 0.0
    repair_features: str = "output"
    repair_basis_source: str = "raw_residual"
    probe_selection_policy: str = "role_equal_centroid"
    selector_block_q: int = 32
    selector_block_n: int = 16
    selector_num_warps: int = 2
    selector_policy: str = "svg_ear_value"
    selector_budget_mode: str = "fixed_density"
    svg_ear_top_p: float = 0.85
    adaptive_budget_min_density: float = 0.0
    adaptive_budget_max_density: float = 1.0
    grouped_svg_ear_router: bool = False
    role_projection_block_m: int = 32
    role_projection_num_warps: int = 2
    repair_block_n: int = 16
    repair_num_warps: int = 2
    repair_num_stages: int = 1
    repair_input_precision: str = "tf32"
    compile_role_kv_projection: bool = True
    role_kv_matmul_precision: str = "high"
    profile_breakdown: bool = False

    def validate(self, head_dim: int | None = None) -> None:
        if not 0.0 < self.target_density <= 1.0:
            raise ValueError(f"target_density must be in (0,1], got {self.target_density}.")
        if self.probe_rows <= 0 or self.repair_rank <= 0:
            raise ValueError("N8 probe_rows and repair_rank must be positive.")
        if self.factor_refresh_every <= 0:
            raise ValueError("factor_refresh_every must be positive.")
        if self.role_kv_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "role_kv_matmul_precision must be highest, high, or medium; got "
                f"{self.role_kv_matmul_precision!r}."
            )
        if self.probe_rows > self.num_q_centroids:
            raise ValueError("Vectorized role-equal probes require M <= Q clusters.")
        if self.role_v_weight < 0.0:
            raise ValueError("role_v_weight must be non-negative.")
        if self.repair_norm_cap < 0.0:
            raise ValueError("repair_norm_cap must be non-negative.")
        if self.repair_features not in {"output", "output_role"}:
            raise ValueError(
                "repair_features must be output or output_role; got "
                f"{self.repair_features!r}."
            )
        if self.repair_basis_source not in {"raw_residual", "fitted_residual"}:
            raise ValueError(
                "repair_basis_source must be raw_residual or fitted_residual; got "
                f"{self.repair_basis_source!r}."
            )
        if self.selector_policy not in {"svg_ear_value", "attention_mass"}:
            raise ValueError(
                "selector_policy must be svg_ear_value or attention_mass; got "
                f"{self.selector_policy!r}."
            )
        if self.selector_budget_mode not in {
            "fixed_density",
            "svg_ear_top_p",
        }:
            raise ValueError(
                "selector_budget_mode must be fixed_density or svg_ear_top_p; got "
                f"{self.selector_budget_mode!r}."
            )
        if not 0.0 < self.svg_ear_top_p <= 1.0:
            raise ValueError(
                f"svg_ear_top_p must be in (0,1], got {self.svg_ear_top_p}."
            )
        if not (
            0.0
            <= self.adaptive_budget_min_density
            <= self.adaptive_budget_max_density
            <= 1.0
        ):
            raise ValueError(
                "Adaptive budget densities must satisfy "
                "0 <= minimum <= maximum <= 1; got "
                f"minimum={self.adaptive_budget_min_density}, "
                f"maximum={self.adaptive_budget_max_density}."
            )
        if self.probe_selection_policy not in {
            "role_equal_centroid",
            "first_token_largest_groups",
        }:
            raise ValueError(
                "probe_selection_policy must be role_equal_centroid or "
                "first_token_largest_groups; got "
                f"{self.probe_selection_policy!r}."
            )
        if head_dim is not None and max(self.role_q_rank, self.role_k_rank, self.role_v_rank) > head_dim:
            raise ValueError(f"N8 role rank exceeds head dimension {head_dim}.")


@dataclass
class N8V6Route:
    """Portable routing result consumed by a model-specific attention adapter."""

    q_labels: torch.Tensor
    k_labels: torch.Tensor
    q_cluster_sizes: torch.Tensor
    k_cluster_sizes: torch.Tensor
    video_dynamic_map: torch.Tensor
    q_role_features: torch.Tensor
    k_permuted: torch.Tensor
    v_permuted: torch.Tensor
    k_sorted_indices: torch.Tensor
    base_video_density: float
    q_kmeans_iters: int
    k_kmeans_iters: int
    permuted_includes_suffix: bool = False
    probe_rows: torch.Tensor | None = None
    probe_weights: torch.Tensor | None = None


class N8V6Core:
    """Stateful per-attention-layer N8-V6 method core."""

    def __init__(self, config: N8V6Config):
        config.validate()
        self.config = config
        self.q_centroids: torch.Tensor | None = None
        self.k_centroids: torch.Tensor | None = None
        self.v_centroids: torch.Tensor | None = None
        self.q_factor_state: torch.Tensor | None = None
        self.k_factor_state: torch.Tensor | None = None
        self.v_factor_state: torch.Tensor | None = None
        self.q_factor_current: torch.Tensor | None = None
        self.k_factor_current: torch.Tensor | None = None
        self.v_factor_current: torch.Tensor | None = None
        self.q_factor_recovery_mask: torch.Tensor | None = None
        self.k_factor_recovery_mask: torch.Tensor | None = None
        self.v_factor_recovery_mask: torch.Tensor | None = None
        self.repair_state_current: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self.repair_recovery_mask: torch.Tensor | None = None
        self.route_call_index = 0
        self._fit_feature_buffer: torch.Tensor | None = None
        self._dense_probe_buffer: torch.Tensor | None = None
        self._fit_target_buffer: torch.Tensor | None = None
        self._full_k_order: torch.Tensor | None = None
        self.last_route_profile: dict[str, float | int | bool | str] = {}
        self.last_repair_profile: dict[str, float | int | bool] = {}
        self.diagnostic_callback = None

    def _full_key_order(
        self,
        video_order: torch.Tensor,
        video_tokens: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Append an identity-ordered suffix without allocating every call."""
        bh = int(video_order.shape[0])
        expected = (bh, total_tokens)
        if (
            self._full_k_order is None
            or tuple(self._full_k_order.shape) != expected
            or self._full_k_order.device != video_order.device
        ):
            self._full_k_order = torch.empty(
                expected, device=video_order.device, dtype=torch.int32
            )
            suffix = torch.arange(
                video_tokens,
                total_tokens,
                device=video_order.device,
                dtype=torch.int32,
            )
            self._full_k_order[:, video_tokens:].copy_(suffix.unsqueeze(0))
        self._full_k_order[:, :video_tokens].copy_(video_order)
        return self._full_k_order

    @property
    def initialized(self) -> bool:
        return self.q_centroids is not None

    def reset(self) -> None:
        self.__init__(self.config)

    @staticmethod
    def _distribution_scalars(
        prefix: str,
        values: torch.Tensor,
    ) -> dict[str, float | int]:
        """Small synchronized summaries used only by an enabled diagnostic."""
        detached = values.detach().float()
        if detached.numel() == 0:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_min": float(detached.amin().item()),
            f"{prefix}_mean": float(detached.mean().item()),
            f"{prefix}_max": float(detached.amax().item()),
        }

    def _emit_diagnostic(
        self,
        stage: str,
        *,
        tensors: dict[str, torch.Tensor],
        scalars: dict[str, object],
    ) -> None:
        callback = self.diagnostic_callback
        if callback is not None:
            callback(
                stage,
                tensors,
                {
                    "core_route_call": int(self.route_call_index),
                    **scalars,
                },
            )

    def _updated_factor(
        self,
        metric: torch.Tensor,
        previous: torch.Tensor | None,
        current: torch.Tensor | None,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if previous is None:
            factor = metric_factor(metric, rank)
            recovered = torch.zeros(
                metric.shape[0], device=metric.device, dtype=torch.bool
            )
            return factor, factor, recovered
        if current is None:
            raise RuntimeError("Warm role-factor state is missing its current factor.")
        result = warm_subspace_metric_factor(
            metric,
            previous,
            rank,
            power_iters=cfg.factor_power_iters,
            orthogonalization=cfg.factor_orthogonalization,
            ritz_mode=cfg.factor_ritz_mode,
            rayleigh_ridge=cfg.factor_rayleigh_ridge,
            cholesky_qr_ridge=cfg.factor_cholesky_qr_ridge,
            return_basis=True,
            return_recovery_mask=True,
        )
        factor, update_state, recovered = result
        recovered = (
            recovered
            | ~torch.isfinite(factor).all(dim=(1, 2))
            | ~torch.isfinite(update_state).all(dim=(1, 2))
        )
        select_previous = recovered[:, None, None]
        # A failed warm solve is normally isolated to one ill-conditioned head.
        # Reuse that head's immediately previous finite geometry, while every
        # healthy head keeps its newly updated factor. This is device-only,
        # does not change routing density, and avoids a full batched EIGH/SVD.
        factor = torch.where(select_previous, current.float(), factor)
        update_state = torch.where(select_previous, previous.float(), update_state)
        return factor, update_state, recovered

    def route(
        self,
        q_video: torch.Tensor,
        k_video: torch.Tensor,
        v_video: torch.Tensor,
        *,
        gqa_group_size: int = 1,
        always_attended_keys: int = 0,
        total_key_tokens: int | None = None,
        k_full_source: torch.Tensor | None = None,
        v_full_source: torch.Tensor | None = None,
    ) -> N8V6Route:
        """Build the exact V6 video-token mask without any tail compensation."""
        if q_video.shape != k_video.shape or q_video.shape != v_video.shape or q_video.ndim != 4:
            raise RuntimeError(
                "N8-V6 expects matching [B,H,N,D] video Q/K/V; got "
                f"Q={tuple(q_video.shape)}, K={tuple(k_video.shape)}, V={tuple(v_video.shape)}."
            )
        batch, heads, tokens, dim = q_video.shape
        self.config.validate(dim)
        bh = batch * heads
        q = q_video.contiguous().view(bh, tokens, dim)
        cfg = self.config
        profile_breakdown = bool(
            cfg.profile_breakdown
            or getattr(self, "profile_breakdown_active", False)
        )
        initialized_before = self.initialized
        group = int(gqa_group_size)
        if group <= 0 or heads % group != 0:
            raise ValueError(
                f"gqa_group_size must divide the query-head count; got H={heads}, G={group}."
            )
        kv_heads = heads // group
        if group > 1:
            k_cluster_video = k_video[:, ::group, :, :].contiguous()
            v_cluster_video = v_video[:, ::group, :, :].contiguous()
            k = k_cluster_video.view(batch * kv_heads, tokens, dim)
            v = v_cluster_video.view(batch * kv_heads, tokens, dim)
            _, q_metric_source = gqa_metric_sources(
                q_video,
                k_video,
                q_cluster_heads=heads,
                k_cluster_heads=kv_heads,
                gqa_group_size=group,
            )
            metric_sample_tokens = cfg.metric_sample_tokens * group
        else:
            k = k_video.contiguous().view(bh, tokens, dim)
            v = v_video.contiguous().view(bh, tokens, dim)
            q_metric_source = q
            metric_sample_tokens = cfg.metric_sample_tokens

        self.last_route_profile = {
            "n8_profile_enabled": profile_breakdown,
            "n8_gqa_key_sharing": bool(group > 1),
            "n8_gqa_group_size": group,
            "n8_q_cluster_heads": heads,
            "n8_k_cluster_heads": kv_heads,
            "n8_k_runtime_heads": heads,
            "n8_selector_budget_mode": cfg.selector_budget_mode,
            "n8_svg_ear_top_p": float(cfg.svg_ear_top_p),
            "n8_adaptive_budget_min_density": float(
                cfg.adaptive_budget_min_density
            ),
            "n8_adaptive_budget_max_density": float(
                cfg.adaptive_budget_max_density
            ),
        }
        profile_started = time.perf_counter()

        def profile_mark(name: str) -> None:
            nonlocal profile_started
            if not profile_breakdown:
                return
            if q_video.is_cuda:
                torch.cuda.synchronize(q_video.device)
            now = time.perf_counter()
            self.last_route_profile[f"n8_route_{name}_ms"] = (
                now - profile_started
            ) * 1000.0
            profile_started = now

        if profile_breakdown and q_video.is_cuda:
            torch.cuda.synchronize(q_video.device)
            profile_started = time.perf_counter()

        # Symmetric heads use one K/V clustering problem per query head. For GQA,
        # cluster only native K/V heads. The key metric sees all query heads in
        # each GQA group, while the resulting Q factor is broadcast back to the
        # group's query heads so Q token assignment remains per query head.
        refresh_factors = (
            self.q_factor_current is None
            or self.k_factor_current is None
            or (cfg.role_v_weight > 0.0 and self.v_factor_current is None)
            or self.route_call_index % cfg.factor_refresh_every == 0
        )
        if refresh_factors:
            k_metric = key_metric_from_queries(q_metric_source, metric_sample_tokens)
            v_metric = (
                value_metric_from_values(v, cfg.metric_sample_tokens)
                if cfg.role_v_weight > 0.0
                else None
            )
            if self.diagnostic_callback is not None:
                metric_tensors = {"k_role_metric": k_metric}
                if v_metric is not None:
                    metric_tensors["v_role_metric"] = v_metric
                self._emit_diagnostic(
                    "clustering",
                    tensors=metric_tensors,
                    scalars={"phase": "kv_metrics_before_factorization"},
                )
            k_factor, self.k_factor_state, self.k_factor_recovery_mask = self._updated_factor(
                k_metric,
                self.k_factor_state,
                self.k_factor_current,
                cfg.role_k_rank,
            )
            self.k_factor_current = k_factor
            if v_metric is not None:
                v_factor, self.v_factor_state, self.v_factor_recovery_mask = self._updated_factor(
                    v_metric,
                    self.v_factor_state,
                    self.v_factor_current,
                    cfg.role_v_rank,
                )
                self.v_factor_current = v_factor
            else:
                v_factor = None
                self.v_factor_recovery_mask = torch.zeros(
                    k_factor.shape[0], device=k_factor.device, dtype=torch.bool
                )
        else:
            assert self.k_factor_current is not None
            k_factor = self.k_factor_current
            v_factor = self.v_factor_current if cfg.role_v_weight > 0.0 else None
        profile_mark("kv_metric_factor")
        if cfg.role_v_weight == 0.0 and k.is_cuda and cfg.compile_role_kv_projection:
            k_features, k_scale = _run_with_fp32_matmul_precision(
                cfg.role_kv_matmul_precision,
                _compiled_role_k_projection_normalize,
                k,
                k_factor,
            )
            v_scale = None
        elif cfg.role_v_weight == 0.0:
            k_features, k_scale = normalize_feature_scale(project_tokens(k, k_factor))
            v_scale = None
        elif k.is_cuda and cfg.compile_role_kv_projection:
            assert v_factor is not None
            k_features, k_scale, v_scale = _run_with_fp32_matmul_precision(
                cfg.role_kv_matmul_precision,
                _compiled_role_kv_projection_normalize,
                k, v, k_factor, v_factor, cfg.role_v_weight,
            )
        else:
            assert v_factor is not None
            k_features_raw, k_scale = normalize_feature_scale(project_tokens(k, k_factor))
            v_features_raw, v_scale = normalize_feature_scale(project_tokens(v, v_factor))
            k_features = torch.cat((k_features_raw, cfg.role_v_weight * v_features_raw), dim=2)

        k_init = None
        if self.k_centroids is not None:
            if cfg.role_v_weight == 0.0 and k.is_cuda and cfg.compile_role_kv_projection:
                k_init = _run_with_fp32_matmul_precision(
                    cfg.role_kv_matmul_precision,
                    _compiled_role_k_init_projection,
                    self.k_centroids,
                    k_factor,
                    k_scale,
                ).contiguous()
            elif cfg.role_v_weight == 0.0:
                k_init = (
                    project_tokens(self.k_centroids, k_factor) / k_scale
                ).contiguous()
            elif k.is_cuda and cfg.compile_role_kv_projection:
                assert self.v_centroids is not None and v_factor is not None and v_scale is not None
                k_init = _run_with_fp32_matmul_precision(
                    cfg.role_kv_matmul_precision,
                    _compiled_role_kv_init_projection,
                    self.k_centroids, self.v_centroids, k_factor, v_factor,
                    k_scale, v_scale, cfg.role_v_weight,
                ).contiguous()
            else:
                assert self.v_centroids is not None and v_factor is not None and v_scale is not None
                k_init = torch.cat(
                    (
                        project_tokens(self.k_centroids, k_factor) / k_scale,
                        cfg.role_v_weight * project_tokens(self.v_centroids, v_factor) / v_scale,
                    ),
                    dim=2,
                ).contiguous()
        profile_mark("kv_projection_init")
        k_iters = cfg.kmeans_iter_step if self.initialized else cfg.kmeans_iter_init
        k_backend = cfg.kmeans_backend
        if k_backend == "n8_cuda_ext" and k_init is None:
            k_backend = "flash"
        k_labels, _k_feature_centroids, k_sizes, k_iter = batch_kmeans_Euclid(
            k_features,
            cfg.num_k_centroids,
            max_iters=k_iters,
            init_centroids=k_init,
            backend=k_backend,
        )
        if profile_breakdown:
            self.last_route_profile.update(
                {
                    f"n8_kmeans_k_{key}": value
                    for key, value in pop_last_kmeans_timings().items()
                }
            )
        profile_mark("kmeans_k")
        self.k_centroids, self.v_centroids = joint_original_centroids(
            k,
            v,
            k_labels,
            k_sizes,
            x_empty_fallback=self.k_centroids,
            y_empty_fallback=self.v_centroids,
        )
        profile_mark("kv_centroids")

        if refresh_factors:
            q_metric = query_metric_from_key_centroids(self.k_centroids, k_sizes)
            if self.diagnostic_callback is not None:
                self._emit_diagnostic(
                    "clustering",
                    tensors={"q_role_metric": q_metric},
                    scalars={"phase": "q_metric_before_factorization"},
                )
            q_factor, self.q_factor_state, self.q_factor_recovery_mask = self._updated_factor(
                q_metric,
                self.q_factor_state,
                self.q_factor_current,
                cfg.role_q_rank,
            )
            self.q_factor_current = q_factor
        else:
            assert self.q_factor_current is not None
            q_factor = self.q_factor_current
        if group > 1:
            q_factor_runtime = (
                q_factor.view(batch, kv_heads, dim, cfg.role_q_rank)
                .repeat_interleave(group, dim=1)
                .contiguous()
                .view(bh, dim, cfg.role_q_rank)
            )
        else:
            q_factor_runtime = q_factor
        profile_mark("q_metric_factor")
        if q.is_cuda:
            q_features = role_projection_rms_triton(
                q,
                q_factor_runtime,
                block_m=cfg.role_projection_block_m,
                num_warps=cfg.role_projection_num_warps,
            )
        else:
            q_features = normalize_rows_rms(project_tokens(q, q_factor_runtime))
        q_init = None
        if self.q_centroids is not None:
            q_init = normalize_rows_rms(
                project_tokens(self.q_centroids, q_factor_runtime)
            ).contiguous()
        profile_mark("q_projection_init")
        q_backend = cfg.kmeans_backend
        if q_backend == "n8_cuda_ext" and q_init is None:
            q_backend = "flash"
        q_labels, _q_feature_centroids, q_sizes, q_iter = batch_kmeans_Euclid(
            q_features,
            cfg.num_q_centroids,
            max_iters=k_iters,
            init_centroids=q_init,
            backend=q_backend,
        )
        if profile_breakdown:
            self.last_route_profile.update(
                {
                    f"n8_kmeans_q_{key}": value
                    for key, value in pop_last_kmeans_timings().items()
                }
            )
        profile_mark("kmeans_q")
        self.q_centroids = original_centroids(q, q_labels, q_sizes, empty_fallback=self.q_centroids)
        profile_mark("q_centroids")

        if group > 1:
            k_labels_runtime = (
                k_labels.view(batch, kv_heads, tokens)
                .repeat_interleave(group, dim=1)
                .contiguous()
                .view(bh, tokens)
            )
            k_sizes_runtime = (
                k_sizes.view(batch, kv_heads, cfg.num_k_centroids)
                .repeat_interleave(group, dim=1)
                .contiguous()
                .view(bh, cfg.num_k_centroids)
            )
            k_centroids_runtime = (
                self.k_centroids.view(batch, kv_heads, cfg.num_k_centroids, dim)
                .repeat_interleave(group, dim=1)
                .contiguous()
                .view(bh, cfg.num_k_centroids, dim)
            )
            v_centroids_runtime = (
                self.v_centroids.view(batch, kv_heads, cfg.num_k_centroids, dim)
                .repeat_interleave(group, dim=1)
                .contiguous()
                .view(bh, cfg.num_k_centroids, dim)
            )
        else:
            k_labels_runtime = k_labels
            k_sizes_runtime = k_sizes
            k_centroids_runtime = self.k_centroids
            v_centroids_runtime = self.v_centroids
        if self.diagnostic_callback is not None:
            cluster_tensors = {
                "q_factor": q_factor,
                "k_factor": k_factor,
                "q_centroids": self.q_centroids,
                "k_centroids": self.k_centroids,
                "v_centroids": self.v_centroids,
                "q_cluster_sizes": q_sizes,
                "k_cluster_sizes": k_sizes,
                "q_factor_recovery_mask": self.q_factor_recovery_mask,
                "k_factor_recovery_mask": self.k_factor_recovery_mask,
            }
            if v_factor is not None:
                cluster_tensors["v_factor"] = v_factor
            if self.v_factor_recovery_mask is not None:
                cluster_tensors["v_factor_recovery_mask"] = self.v_factor_recovery_mask
            v_recovery_mask = self.v_factor_recovery_mask
            if v_recovery_mask is None:
                v_recovery_mask = torch.zeros_like(self.k_factor_recovery_mask)
            self._emit_diagnostic(
                "clustering",
                tensors=cluster_tensors,
                scalars={
                    "phase": "cluster_result",
                    "initialized_before": bool(initialized_before),
                    "factor_refresh": bool(refresh_factors),
                    "q_kmeans_iters": int(q_iter),
                    "k_kmeans_iters": int(k_iter),
                    "q_empty_clusters": int((q_sizes == 0).sum().item()),
                    "k_empty_clusters": int((k_sizes == 0).sum().item()),
                    "q_factor_recovered_heads": int(
                        self.q_factor_recovery_mask.sum().item()
                    ),
                    "k_factor_recovered_heads": int(
                        self.k_factor_recovery_mask.sum().item()
                    ),
                    "v_factor_recovered_heads": int(
                        v_recovery_mask.sum().item()
                    ),
                    "q_factor_recovered_head_indices": [
                        int(index)
                        for index in torch.nonzero(
                            self.q_factor_recovery_mask, as_tuple=False
                        ).flatten().tolist()
                    ],
                    "k_factor_recovered_head_indices": [
                        int(index)
                        for index in torch.nonzero(
                            self.k_factor_recovery_mask, as_tuple=False
                        ).flatten().tolist()
                    ],
                    "v_factor_recovered_head_indices": [
                        int(index)
                        for index in torch.nonzero(
                            v_recovery_mask, as_tuple=False
                        ).flatten().tolist()
                    ],
                    "factor_recovery_policy": "reuse_previous_finite_factor",
                    **self._distribution_scalars("q_cluster_size", q_sizes),
                    **self._distribution_scalars("k_cluster_size", k_sizes),
                },
            )

        # K/V are permuted once and reused by both routing and the model
        # adapter's sparse attention call.
        # Hunyuan supplies the full video+text tensors here. Permuting the full
        # source once is exact and removes the later K/V torch.cat copies.
        k_order = torch.argsort(k_labels_runtime, dim=-1).to(torch.int32).contiguous()
        use_full_layout = k_full_source is not None or v_full_source is not None
        if use_full_layout:
            if k_full_source is None or v_full_source is None:
                raise ValueError("k_full_source and v_full_source must be provided together.")
            if k_full_source.shape != v_full_source.shape or k_full_source.ndim != 4:
                raise ValueError("Full K/V sources must have matching [B,H,N,D] shapes.")
            if tuple(k_full_source.shape[:2]) != (batch, heads) or k_full_source.shape[-1] != dim:
                raise ValueError("Full K/V sources do not match the video tensor head layout.")
            total_layout_tokens = int(k_full_source.shape[2])
            if total_layout_tokens < tokens:
                raise ValueError("Full K/V source is shorter than the video prefix.")
            full_order = self._full_key_order(k_order, tokens, total_layout_tokens)
            k_permuted, _ = permute_tensor_by_labels_triton(
                k_full_source, None, dim=2, sorted_indices=full_order
            )
            v_permuted, _ = permute_tensor_by_labels_triton(
                v_full_source, None, dim=2, sorted_indices=full_order
            )
            selector_k = k_permuted.reshape(bh, total_layout_tokens, dim)
            selector_v = v_permuted.reshape(bh, total_layout_tokens, dim)
        else:
            k_permuted, _ = permute_tensor_by_labels_triton(
                k_video, None, dim=2, sorted_indices=k_order
            )
            v_permuted, _ = permute_tensor_by_labels_triton(
                v_video, None, dim=2, sorted_indices=k_order
            )
            selector_k = k_permuted.reshape(bh, tokens, dim)
            selector_v = v_permuted.reshape(bh, tokens, dim)
        profile_mark("kv_permute")
        use_grouped_svg_ear = bool(
            cfg.grouped_svg_ear_router
            and cfg.selector_policy == "svg_ear_value"
            and group > 1
        )
        if use_grouped_svg_ear:
            if use_full_layout:
                selector_tokens = int(selector_k.shape[1])
                selector_k_native = (
                    selector_k.view(batch, heads, selector_tokens, dim)[:, ::group]
                    .contiguous()
                    .view(batch * kv_heads, selector_tokens, dim)
                )
                selector_v_native = (
                    selector_v.view(batch, heads, selector_tokens, dim)[:, ::group]
                    .contiguous()
                    .view(batch * kv_heads, selector_tokens, dim)
                )
            else:
                selector_k_native = k_permuted.view(
                    batch, heads, tokens, dim
                )[:, ::group].contiguous().view(batch * kv_heads, tokens, dim)
                selector_v_native = v_permuted.view(
                    batch, heads, tokens, dim
                )[:, ::group].contiguous().view(batch * kv_heads, tokens, dim)
            scores = svg_ear_grouped_value_error_scores_triton(
                qcentroids=self.q_centroids,
                k_sorted=selector_k_native,
                v_sorted=selector_v_native,
                kcentroids=self.k_centroids,
                vcentroids=self.v_centroids,
                kcluster_sizes=k_sizes,
                gqa_group=group,
                block_q=min(cfg.selector_block_q, 16),
                block_n=cfg.selector_block_n,
                num_warps=max(4, cfg.selector_num_warps),
            )
        elif cfg.selector_policy == "attention_mass":
            scores = attention_mass_scores_triton(
                qcentroids=self.q_centroids,
                k_sorted=selector_k,
                kcluster_sizes=k_sizes_runtime,
                gqa_group=1,
                block_q=cfg.selector_block_q,
                block_n=cfg.selector_block_n,
                num_warps=cfg.selector_num_warps,
                logical_tokens=tokens,
            )
        else:
            scores = svg_ear_value_error_scores_triton(
                qcentroids=self.q_centroids,
                k_sorted=selector_k,
                v_sorted=selector_v,
                kcentroids=k_centroids_runtime,
                vcentroids=v_centroids_runtime,
                kcluster_sizes=k_sizes_runtime,
                gqa_group=1,
                var_k_diag=None,
                jensen_var_cap=0.0,
                block_q=cfg.selector_block_q,
                block_n=cfg.selector_block_n,
                num_warps=cfg.selector_num_warps,
                logical_tokens=tokens,
            )
        profile_mark("selector_scores")
        if self.diagnostic_callback is not None:
            self._emit_diagnostic(
                "svg_ear_selection",
                # Preserve the established diagnostic schema; selector_policy
                # disambiguates SVG-EAR from attention mass.
                tensors={"svg_ear_scores": scores},
                scalars={
                    "phase": "scores_before_budget_selection",
                    "selector_policy": cfg.selector_policy,
                    "grouped_svg_ear_router": use_grouped_svg_ear,
                    "empty_key_clusters": int((k_sizes_runtime == 0).sum().item()),
                    "score_neg_inf_expected_for_empty_key_clusters": True,
                },
            )
        # Fixed density uses matched-additive accounting: exact M probes are
        # removed from the base sparse budget, then restored by dense probe
        # attention. In adaptive mode SVG2 TopP determines each query-cluster
        # video budget directly; repair and model-specific always-attended keys
        # are reported as additional measured density.
        total_keys = int(total_key_tokens) if total_key_tokens is not None else tokens
        if total_keys < tokens or always_attended_keys < 0 or always_attended_keys > total_keys:
            raise ValueError(
                f"Invalid key accounting: video={tokens}, total={total_keys}, "
                f"always={always_attended_keys}."
            )
        if cfg.selector_budget_mode == "fixed_density":
            base_total_density = max(
                0.0, cfg.target_density - cfg.probe_rows / float(tokens)
            )
            target_keys = int(math.ceil(base_total_density * total_keys))
            budget = max(0, target_keys - int(always_attended_keys))
            budgets = torch.full(
                (bh, cfg.num_q_centroids),
                budget,
                device=q_video.device,
                dtype=torch.long,
            )
            base_density = budget / float(tokens)
            diagnostic_budget = int(budget)
        else:
            budgets = _svg_top_p_key_budgets(
                self.q_centroids,
                k_centroids_runtime,
                k_sizes_runtime,
                cfg.svg_ear_top_p,
            )
            minimum_budget = int(
                math.ceil(cfg.adaptive_budget_min_density * tokens)
            )
            maximum_budget = int(
                math.floor(cfg.adaptive_budget_max_density * tokens)
            )
            budgets = budgets.clamp(
                min=minimum_budget,
                max=maximum_budget,
            )
            budget_pair_density = (
                budgets.long() * q_sizes.long()
            ).sum(dim=1).float() / float(tokens * tokens)
            base_density = float(budget_pair_density.mean().item())
            diagnostic_budget = -1
        if scores.is_cuda:
            global _N8_CUDA_EXT_SELECTOR_WARNED
            if _N8_CUDA_EXT_SELECTOR:
                try:
                    from ...kernels.n8_extension import svg_ear_select_budget

                    routed = svg_ear_select_budget(scores, k_sizes_runtime, budgets)
                except Exception as exc:
                    if _N8_CUDA_EXT_STRICT:
                        raise RuntimeError(
                            "Strict N8 CUDA selector execution failed."
                        ) from exc
                    if not _N8_CUDA_EXT_SELECTOR_WARNED:
                        warnings.warn(
                            f"n8_kernels selector unavailable; using Triton: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _N8_CUDA_EXT_SELECTOR_WARNED = True
                    routed = exact_budget_dynamic_map_triton(
                        scores, k_sizes_runtime, budgets
                    )
            else:
                routed = exact_budget_dynamic_map_triton(
                    scores, k_sizes_runtime, budgets
                )
        else:
            order = torch.argsort(scores, dim=2, descending=True)
            costs = (
                k_sizes_runtime.long()
                .unsqueeze(1)
                .expand(-1, cfg.num_q_centroids, -1)
            )
            sorted_costs = torch.gather(costs, 2, order)
            cumulative = sorted_costs.cumsum(dim=2)
            keep = (sorted_costs > 0) & (cumulative <= budgets.unsqueeze(2))
            selected_cost = (sorted_costs * keep.long()).sum(dim=2)
            count = keep.long().sum(dim=2)
            crossing = count.clamp_max(cfg.num_k_centroids - 1)
            need = selected_cost < budgets
            keep.scatter_(2, crossing.unsqueeze(2), need.unsqueeze(2))
            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected.scatter_(2, order, keep)
            routed = torch.zeros(
                bh, cfg.num_q_centroids + 1, cfg.num_k_centroids + 1,
                device=scores.device, dtype=torch.bool,
            )
            routed[:, 0, 0] = True
            routed[:, 1:, 0] = True
            routed[:, 1:, 1:] = selected
        video_map = routed[:, 1:, 1:]
        profile_mark("selector_budget")
        if self.diagnostic_callback is not None:
            selected_key_tokens = (
                video_map.long() * k_sizes_runtime.long().unsqueeze(1)
            ).sum(dim=2)
            selected_clusters = video_map.long().sum(dim=2)
            pair_density = (
                selected_key_tokens * q_sizes.long()
            ).sum(dim=1).float() / float(tokens * tokens)
            self._emit_diagnostic(
                "svg_ear_selection",
                tensors={
                    "selected_key_tokens": selected_key_tokens,
                    "selected_key_clusters": selected_clusters,
                    "pair_density_per_head": pair_density,
                },
                scalars={
                    "phase": "budget_selection_result",
                    "selector_policy": cfg.selector_policy,
                    "selector_budget_mode": cfg.selector_budget_mode,
                    "svg_ear_top_p": float(cfg.svg_ear_top_p),
                    "adaptive_budget_min_density": float(
                        cfg.adaptive_budget_min_density
                    ),
                    "adaptive_budget_max_density": float(
                        cfg.adaptive_budget_max_density
                    ),
                    "target_key_budget": diagnostic_budget,
                    "base_video_density": float(base_density),
                    **self._distribution_scalars(
                        "requested_key_budget", budgets
                    ),
                    **self._distribution_scalars(
                        "selected_key_tokens", selected_key_tokens
                    ),
                    **self._distribution_scalars(
                        "selected_key_clusters", selected_clusters
                    ),
                    **self._distribution_scalars(
                        "actual_pair_density", pair_density
                    ),
                },
            )
        result = N8V6Route(
            q_labels=q_labels,
            k_labels=k_labels_runtime,
            q_cluster_sizes=q_sizes,
            k_cluster_sizes=k_sizes_runtime,
            video_dynamic_map=video_map,
            q_role_features=q_features,
            k_permuted=k_permuted,
            v_permuted=v_permuted,
            k_sorted_indices=k_order,
            base_video_density=base_density,
            q_kmeans_iters=int(q_iter),
            k_kmeans_iters=int(k_iter),
            permuted_includes_suffix=use_full_layout,
        )
        self.route_call_index += 1
        return result

    def repair(
        self,
        base_video: torch.Tensor,
        q_video: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        route: N8V6Route,
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        """Apply current-step M64/r16 probe repair to video rows."""
        batch, heads, tokens, dim = base_video.shape
        bh = batch * heads
        cfg = self.config
        self.last_repair_profile = {
            "n8_profile_enabled": bool(cfg.profile_breakdown),
            "n8_probe_selection_policy": cfg.probe_selection_policy,
        }
        profile_started = time.perf_counter()

        def profile_mark(name: str) -> None:
            nonlocal profile_started
            if not cfg.profile_breakdown:
                return
            if base_video.is_cuda:
                torch.cuda.synchronize(base_video.device)
            now = time.perf_counter()
            self.last_repair_profile[f"n8_repair_{name}_ms"] = (
                now - profile_started
            ) * 1000.0
            profile_started = now

        if cfg.profile_breakdown and base_video.is_cuda:
            torch.cuda.synchronize(base_video.device)
            profile_started = time.perf_counter()

        base = base_video.contiguous().view(bh, tokens, dim)
        probe_cache_hit = bool(
            route.probe_rows is not None and route.probe_weights is not None
        )
        if probe_cache_hit:
            assert route.probe_rows is not None
            assert route.probe_weights is not None
            rows, weights = route.probe_rows, route.probe_weights
        else:
            probe_selector = (
                select_first_token_largest_groups
                if cfg.probe_selection_policy == "first_token_largest_groups"
                else select_role_equal_probes_vectorized
            )
            rows, weights = probe_selector(
                route.q_role_features,
                route.q_labels,
                num_clusters=cfg.num_q_centroids,
                rows=cfg.probe_rows,
            )
            route.probe_rows = rows
            route.probe_weights = weights
        self.last_repair_profile["n8_probe_selection_cache_hit"] = (
            probe_cache_hit
        )
        profile_mark("probe_select")
        count = int(rows.shape[1])
        q_flat = q_video.contiguous().view(bh, tokens, dim)
        q_probe = torch.gather(q_flat, 1, rows.unsqueeze(2).expand(-1, -1, dim))
        dense_probe = F.scaled_dot_product_attention(
            q_probe.view(batch, heads, count, dim),
            k_full,
            v_full,
            dropout_p=0.0,
            is_causal=False,
        ).reshape(bh, count, dim)
        profile_mark("dense_probe")

        role_feature_dim = (
            int(route.q_role_features.shape[2])
            if cfg.repair_features == "output_role"
            else 0
        )
        feature_dim = dim + role_feature_dim
        feature_shape = (bh, count, feature_dim)
        target_shape = (bh, count, dim)
        if self._fit_feature_buffer is None or self._fit_feature_buffer.shape != feature_shape:
            self._fit_feature_buffer = torch.empty(feature_shape, device=base.device, dtype=torch.float32)
            self._dense_probe_buffer = torch.empty(target_shape, device=base.device, dtype=torch.float32)
            self._fit_target_buffer = torch.empty(target_shape, device=base.device, dtype=torch.float32)
        assert self._dense_probe_buffer is not None and self._fit_target_buffer is not None
        if base.is_cuda and cfg.repair_features == "output_role":
            x_fit, dense_probe_f, y_fit = n8_prepare_repair_fit_triton(
                base,
                route.q_role_features,
                dense_probe,
                rows,
                self._fit_feature_buffer,
                self._dense_probe_buffer,
                self._fit_target_buffer,
            )
        elif base.is_cuda:
            x_fit, dense_probe_f, y_fit = n8_prepare_output_repair_fit_triton(
                base,
                dense_probe,
                rows,
                self._fit_feature_buffer,
                self._dense_probe_buffer,
                self._fit_target_buffer,
            )
        else:
            base_probe = torch.gather(base.float(), 1, rows.unsqueeze(2).expand(-1, -1, dim))
            if cfg.repair_features == "output_role":
                role_probe = torch.gather(
                    route.q_role_features,
                    1,
                    rows.unsqueeze(2).expand(-1, -1, route.q_role_features.shape[2]),
                )
                x_fit = torch.cat((base_probe, role_probe), dim=2)
            else:
                x_fit = base_probe
            dense_probe_f = dense_probe.float()
            y_fit = dense_probe_f - base_probe
        profile_mark("fit_prepare")
        if self.diagnostic_callback is not None:
            self._emit_diagnostic(
                "probe_fit",
                tensors={
                    "probe_rows": rows,
                    "probe_weights": weights,
                    "dense_probe": dense_probe_f,
                    "fit_features": x_fit,
                    "fit_targets": y_fit,
                },
                scalars={
                    "phase": "inputs_before_solver",
                    "probe_count": int(count),
                    **self._distribution_scalars("probe_row", rows),
                    **self._distribution_scalars("probe_weight", weights),
                },
            )
        state = fit_reduced_rank_residual(
            x_fit,
            y_fit,
            rank=min(cfg.repair_rank, count, dim),
            ridge=cfg.repair_ridge,
            weights=weights,
            output_basis_backend="skinny",
            cache_identity=True,
            fit_backend="dual_cholesky",
            decomposition_backend="triton_top16" if base.is_cuda else "eigh",
            basis_source=cfg.repair_basis_source,
        )
        state, self.repair_recovery_mask, repair_recovery_policy = (
            recover_nonfinite_rrr_state(
                state,
                self.repair_state_current,
            )
        )
        self.repair_state_current = state
        profile_mark("rrr_fit")
        x_mean, x_scale, y_mean, left, basis = state
        if base.is_cuda and cfg.repair_features == "output_role":
            corrected = n8_repair_apply_triton(
                base,
                route.q_role_features,
                x_mean,
                x_scale,
                y_mean,
                left,
                basis,
                cfg.repair_alpha,
                cfg.repair_norm_cap,
                block_n=cfg.repair_block_n,
                num_warps=cfg.repair_num_warps,
                num_stages=cfg.repair_num_stages,
                input_precision=cfg.repair_input_precision,
            )
        elif base.is_cuda:
            corrected = n8_repair_apply_output_triton(
                base,
                x_mean,
                x_scale,
                y_mean,
                left,
                basis,
                cfg.repair_alpha,
                cfg.repair_norm_cap,
                block_n=cfg.repair_block_n,
                num_warps=cfg.repair_num_warps,
                num_stages=cfg.repair_num_stages,
                input_precision=cfg.repair_input_precision,
            )
        if base.is_cuda:
            n8_exact_probe_scatter_triton(
                corrected,
                rows,
                dense_probe_f,
                block_m=32,
                num_warps=2,
                convert_rows=True,
            )
        else:
            if cfg.repair_features == "output_role":
                corrected = apply_reduced_rank_residual(
                    base,
                    route.q_role_features,
                    x_mean,
                    x_scale,
                    y_mean,
                    left,
                    basis,
                    cfg.repair_alpha,
                    cfg.repair_norm_cap,
                )
            else:
                correction = torch.bmm(
                    torch.bmm((base.float() - x_mean) / x_scale, left),
                    basis.transpose(1, 2),
                ) + y_mean
                if cfg.repair_norm_cap > 0.0:
                    base_norm = torch.linalg.vector_norm(base.float(), dim=2, keepdim=True)
                    correction_norm = torch.linalg.vector_norm(correction, dim=2, keepdim=True)
                    correction = correction * (
                        cfg.repair_norm_cap * base_norm / correction_norm.clamp_min(1e-20)
                    ).clamp_max(1.0)
                corrected = base + cfg.repair_alpha * correction.to(base.dtype)
            corrected.scatter_(1, rows.unsqueeze(2).expand(-1, -1, dim), dense_probe_f.to(corrected.dtype))
        profile_mark("apply_scatter")
        if self.diagnostic_callback is not None:
            sorted_rows = torch.sort(rows.long(), dim=1).values
            duplicate_rows = (
                (sorted_rows[:, 1:] == sorted_rows[:, :-1]).sum()
                if sorted_rows.shape[1] > 1
                else torch.zeros((), device=rows.device, dtype=torch.long)
            )
            self._emit_diagnostic(
                "probe_fit",
                tensors={
                    "fit_x_mean": x_mean,
                    "fit_x_scale": x_scale,
                    "fit_y_mean": y_mean,
                    "fit_left": left,
                    "fit_basis": basis,
                    "repair_recovery_mask": self.repair_recovery_mask,
                },
                scalars={
                    "phase": "solver_result",
                    "probe_count": int(count),
                    "probe_duplicate_rows": int(duplicate_rows.item()),
                    "repair_rank": int(min(cfg.repair_rank, count, dim)),
                    "repair_recovered_heads": int(
                        self.repair_recovery_mask.sum().item()
                    ),
                    "repair_recovered_head_indices": [
                        int(index)
                        for index in torch.nonzero(
                            self.repair_recovery_mask, as_tuple=False
                        ).flatten().tolist()
                    ],
                    "repair_recovery_policy": repair_recovery_policy,
                    **self._distribution_scalars("probe_row", rows),
                    **self._distribution_scalars("probe_weight", weights),
                    **self._distribution_scalars("fit_x_scale", x_scale),
                },
            )
        return corrected.view_as(base_video), {
            "probe_rows": count,
            "probe_selection_policy": cfg.probe_selection_policy,
            "repair_rank": min(cfg.repair_rank, count, dim),
            "repair_feature_dim": feature_dim,
            "repair_features": cfg.repair_features,
            "repair_basis_source": cfg.repair_basis_source,
            "repair_backend": "triton_top16" if base.is_cuda else "eigh_reference",
            "effective_video_density": min(1.0, route.base_video_density + count / float(tokens)),
            **self.last_repair_profile,
        }
