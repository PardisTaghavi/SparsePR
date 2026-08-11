"""N8-V6 sparse-attention adapter for official Cosmos-Predict2.5 inference.

The portable :class:`sparsepr.models.common.N8V6Core` consumes normalized,
RoPE-applied ``[B,H,N,D]`` Q/K/V tensors. Cosmos-Predict2.5 exposes those
tensors immediately before its attention operator, so this adapter always
patches ``Attention.compute_attention``. An independently gated CUDA extension
may also replace the self-attention Q/K RMSNorm and RoPE portion of
``Attention.compute_qkv`` while leaving the learned projections unchanged.
If that extension is absent or rejects a runtime shape, the adapter returns to
NVIDIA's official QKV implementation.

Cosmos classifier-free guidance evaluates the conditional and unconditional
denoisers in separate transformer forwards. Repair state remains branch-local,
while an explicitly gated route cache can reuse cluster labels, the sparse
map, and probe rows across nearby steps and the paired CFG forward. Current
Q/K/V tensors are still permuted, attended, and repaired on every call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any
import warnings

import torch


@dataclass(frozen=True)
class Cosmos25N8Config:
    """Production N8-V6 settings for Cosmos-Predict2.5-14B."""

    total_layers: int = 36
    total_steps: int = 35
    forwards_per_step: int = 2
    dense_first_steps: int = 2
    dense_first_layers: int = 1
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
    factor_power_iters: int = 2
    factor_refresh_every: int = 1
    route_refresh_every: int = 4
    share_cfg_route: bool = True
    flashinfer_plan_reuse: bool = True
    cross_attention_kv_cache: bool = True
    probe_rows: int = 64
    repair_rank: int = 16
    role_kv_matmul_precision: str = "high"
    selector_policy: str = "svg_ear_value"
    selector_budget_mode: str = "fixed_density"
    selector_top_p: float = 0.85
    adaptive_budget_min_density: float = 0.0
    adaptive_budget_max_density: float = 1.0
    profile_breakdown: bool = False
    profile_step: int = 3
    diffusion_offload_enabled: bool = True
    cuda_ext_build: str | None = None
    cuda_ext_cosmos_qkv: bool = False
    cuda_ext_role_cluster: bool = False
    cuda_ext_selector: bool = True
    run_manifest: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Cosmos25N8Config":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Cosmos2.5 N8 config keys: {unknown}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.total_layers <= 0 or self.total_steps <= 0:
            raise ValueError("Cosmos2.5 total_layers and total_steps must be positive.")
        if self.forwards_per_step <= 0:
            raise ValueError("Cosmos2.5 forwards_per_step must be positive.")
        if not 0 <= self.dense_first_steps <= self.total_steps:
            raise ValueError("dense_first_steps must lie in [0, total_steps].")
        if not 0 <= self.dense_first_layers <= self.total_layers:
            raise ValueError("dense_first_layers must lie in [0, total_layers].")
        if self.num_q_centroids <= 0 or self.num_k_centroids <= 0:
            raise ValueError("N8 centroid counts must be positive.")
        if not 0.0 < self.target_density <= 1.0:
            raise ValueError("N8 target_density must lie in (0, 1].")
        if self.probe_rows <= 0 or self.probe_rows > self.num_q_centroids:
            raise ValueError("N8 probe_rows must lie in [1, num_q_centroids].")
        if self.factor_refresh_every <= 0:
            raise ValueError("factor_refresh_every must be positive.")
        if self.route_refresh_every <= 0:
            raise ValueError("route_refresh_every must be positive.")
        if self.repair_rank <= 0:
            raise ValueError("N8 repair_rank must be positive.")
        if not 0 <= self.profile_step < self.total_steps:
            raise ValueError("profile_step must lie in [0, total_steps).")
        if self.role_kv_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "role_kv_matmul_precision must be highest, high, or medium."
            )
        if self.selector_policy not in {"svg_ear_value", "attention_mass"}:
            raise ValueError("selector_policy must be svg_ear_value or attention_mass.")
        if self.selector_budget_mode not in {"fixed_density", "svg_ear_top_p"}:
            raise ValueError(
                "selector_budget_mode must be fixed_density or svg_ear_top_p."
            )
        if not 0.0 < self.selector_top_p <= 1.0:
            raise ValueError("selector_top_p must lie in (0, 1].")
        if not (
            0.0
            <= self.adaptive_budget_min_density
            <= self.adaptive_budget_max_density
            <= 1.0
        ):
            raise ValueError(
                "adaptive budget densities must satisfy 0 <= min <= max <= 1."
            )
        if self.cuda_ext_build is not None and not self.cuda_ext_build.strip():
            raise ValueError("cuda_ext_build must be a nonempty path or null.")

    def provenance(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("run_manifest", None)
        return values


@dataclass
class _RouteCacheEntry:
    """Compact route metadata reused with current-step Q/K/V tensors."""

    origin_step: int
    last_step: int
    route: Any
    q_sorted_indices: torch.Tensor


def bshd_to_bhnd(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise RuntimeError(
            f"Cosmos2.5 {name} must be [B,S,H,D], got "
            f"{type(tensor).__name__} {getattr(tensor, 'shape', None)}."
        )
    return tensor.permute(0, 2, 1, 3).contiguous()


def bhnd_to_bshd_flat(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise RuntimeError("Cosmos2.5 N8 output must be [B,H,S,D].")
    batch, heads, tokens, dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim)


def _extension_components(module: Any) -> dict[str, bool]:
    """Report independently usable symbols from one loaded extension."""

    return {
        "cosmos_qkv": hasattr(module, "cosmos25_qkv_norm_rope_bshd"),
        "role_cluster": all(
            hasattr(module, name)
            for name in ("role_cluster_step_q64", "role_cluster_step_k64")
        ),
        "selector": hasattr(module, "svg_ear_select_budget"),
    }


class Cosmos25N8Patch:
    """One installed official-Cosmos N8 patch and its per-sample state."""

    def __init__(self, config: Cosmos25N8Config):
        config.validate()
        self.config = config
        self._original_compute_qkv = None
        self._original_i2v_compute_qkv = None
        self._original_compute_attention = None
        self._original_generate_sample = None
        self._original_dit_forward = None
        self._original_block_forward = None
        self._original_attention_forward = None
        self._original_i2v_cross_attention_forward = None
        self._original_mlp_forward = None
        self._clear_flashinfer_plan_cache = None
        self._torch_profiler_active = False
        self._torch_profiler_complete = False
        self._torch_profiler_outputs: dict[str, Any] = {}
        self._nonattention_fusions = None
        self._layer_by_module: dict[int, int] = {}
        self._modules: dict[int, Any] = {}
        self._cores: dict[tuple[int, int], Any] = {}
        self._route_cache: dict[tuple[int, int], _RouteCacheEntry] = {}
        self._cross_kv_cache: dict[tuple[int, int], tuple[Any, ...]] = {}
        self._cross_layer_by_module: dict[int, int] = {}
        self._cross_context_reference: dict[int, tuple[int, Any]] = {}
        self._cross_context_validated: set[int] = set()
        self._cross_cache_disabled_branches: set[int] = set()
        self._cross_cache_active_printed = False
        self._sample_name: str | None = None
        self._forward_index = -1
        self._dense_calls = 0
        self._sparse_calls = 0
        self._branch_dense_calls = [0 for _ in range(config.forwards_per_step)]
        self._branch_sparse_calls = [0 for _ in range(config.forwards_per_step)]
        self._cuda_qkv_calls = 0
        self._cuda_ext_cosmos_qkv = False
        self._cuda_ext_role_cluster = False
        self._cuda_ext_selector = False
        self._cuda_qkv_warned = False
        self._method_counts: dict[str, int] = {}
        self._profile_components_ms: dict[str, float] = {}
        self._profile_details_ms: dict[str, float] = {}
        self._profile_step_started: float | None = None
        self._profile_step_elapsed_ms: float | None = None
        self._density_sum = 0.0
        self._density_count = 0
        self._density_min = float("inf")
        self._density_max = 0.0

    def install(self) -> None:
        if self._original_compute_attention is not None:
            return

        # Match the production Hunyuan/Wan N8-V6 gates before importing SVG.
        os.environ.setdefault("SPARSEPR_COSMOS3_FLASH_KMEANS_INLINE_XSQ", "1")
        os.environ.setdefault(
            "SPARSEPR_COSMOS3_ORIGINAL_CENTROID_BACKEND", "triton_atomic_fp32"
        )
        if self.config.cuda_ext_build is not None:
            os.environ["SPARSEPR_N8_CUDA_EXT_BUILD"] = str(
                Path(self.config.cuda_ext_build).expanduser().resolve()
            )

        requested_extension = any(
            (
                self.config.cuda_ext_cosmos_qkv,
                self.config.cuda_ext_role_cluster,
                self.config.cuda_ext_selector,
            )
        )
        if requested_extension:
            from sparsepr.kernels.n8_extension import load_n8_kernels

            module = load_n8_kernels(required=False)
            if module is None:
                warnings.warn(
                    "n8_kernels.so was requested for Cosmos2.5 but could not "
                    "be loaded; all compiled components are disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                available = _extension_components(module)
                self._cuda_ext_cosmos_qkv = (
                    self.config.cuda_ext_cosmos_qkv
                    and available["cosmos_qkv"]
                )
                self._cuda_ext_role_cluster = (
                    self.config.cuda_ext_role_cluster
                    and available["role_cluster"]
                )
                self._cuda_ext_selector = (
                    self.config.cuda_ext_selector and available["selector"]
                )
                missing = [
                    name
                    for name, requested, enabled in (
                        (
                            "Cosmos2.5 QKV",
                            self.config.cuda_ext_cosmos_qkv,
                            self._cuda_ext_cosmos_qkv,
                        ),
                        (
                            "role clustering",
                            self.config.cuda_ext_role_cluster,
                            self._cuda_ext_role_cluster,
                        ),
                        (
                            "selector",
                            self.config.cuda_ext_selector,
                            self._cuda_ext_selector,
                        ),
                    )
                    if requested and not enabled
                ]
                if missing:
                    warnings.warn(
                        "Loaded n8_kernels.so is missing requested symbols for "
                        f"{', '.join(missing)}; those components remain on "
                        "their reference paths.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        # N8V6Core reads this gate when its module is imported below.
        os.environ["SPARSEPR_N8_CUDA_EXT_SELECTOR"] = (
            "1" if self._cuda_ext_selector else "0"
        )

        # Fail at installation rather than after the 14B checkpoint is loaded
        # when a sparse runtime dependency is absent.
        import flashinfer  # noqa: F401
        from flash_kmeans.kmeans_triton_impl import (  # noqa: F401
            batch_kmeans_Euclid,
        )
        from sparsepr.kmeans_utils import (
            clear_flashinfer_plan_cache,
            set_flashinfer_workspace_cache_enabled,
        )

        set_flashinfer_workspace_cache_enabled(True)
        self._clear_flashinfer_plan_cache = clear_flashinfer_plan_cache

        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Attention,
            I2VCrossAttention,
        )
        from cosmos_predict2.inference import Inference

        self._original_compute_qkv = Attention.compute_qkv
        self._original_i2v_compute_qkv = I2VCrossAttention.compute_qkv
        self._original_i2v_cache_forward = I2VCrossAttention.forward
        self._original_compute_attention = Attention.compute_attention
        self._original_generate_sample = Inference._generate_sample
        patch = self

        def patched_compute_qkv(attention, x, context=None, rope_emb=None):
            return patch.compute_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )

        def patched_i2v_compute_qkv(attention, x, context, rope_emb=None):
            return patch.compute_i2v_cross_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )

        def patched_i2v_forward(attention, x, context=None, rope_emb=None):
            q, k, v, k_img, v_img = patch.compute_i2v_cross_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )
            return attention.compute_attention(q, k, v, k_img, v_img)

        def patched_compute_attention(
            attention,
            q,
            k,
            v,
            video_size=None,
            kv_cache_cfg=None,
        ):
            return patch.compute_attention(
                attention,
                q,
                k,
                v,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
            )

        def patched_generate_sample(inference, sample, output_dir):
            return patch.generate_sample(inference, sample, output_dir)

        Attention.compute_qkv = patched_compute_qkv
        I2VCrossAttention.compute_qkv = patched_i2v_compute_qkv
        I2VCrossAttention.forward = patched_i2v_forward
        Attention.compute_attention = patched_compute_attention
        Inference._generate_sample = patched_generate_sample
        if any(
            os.environ.get(name, "0") == "1"
            for name in (
                "SPARSEPR_COSMOS25_COMPILE_MLP",
                "SPARSEPR_COSMOS25_FUSE_BLOCK_ELEMENTWISE",
            )
        ):
            if os.environ.get("SPARSEPR_COSMOS25_TORCH_PROFILE", "0") == "1":
                raise RuntimeError(
                    "Cosmos2.5 fusion gating and torch profiling must run "
                    "in separate samples."
                )
            from sparsepr.adapters.cosmos_predict2_fusions import install_from_environment

            self._nonattention_fusions = install_from_environment()
        if os.environ.get("SPARSEPR_COSMOS25_TORCH_PROFILE", "0") == "1":
            self._install_torch_profiler_ranges()
        print(
            "[N8-COSMOS25] installed shared N8-V6 on official "
            "Cosmos-Predict2.5 self-attention; CUDA extension "
            f"qkv={self._cuda_ext_cosmos_qkv} "
            f"role={self._cuda_ext_role_cluster} "
            f"selector={self._cuda_ext_selector}.",
            flush=True,
        )

    def restore(self) -> None:
        if self._original_compute_attention is None:
            return
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Attention,
            I2VCrossAttention,
        )
        from cosmos_predict2.inference import Inference

        Attention.compute_qkv = self._original_compute_qkv
        I2VCrossAttention.compute_qkv = self._original_i2v_compute_qkv
        I2VCrossAttention.forward = self._original_i2v_cache_forward
        Attention.compute_attention = self._original_compute_attention
        Inference._generate_sample = self._original_generate_sample
        if self._nonattention_fusions is not None:
            self._nonattention_fusions.restore()
            self._nonattention_fusions = None
        self._original_compute_qkv = None
        self._original_i2v_compute_qkv = None
        self._original_i2v_cache_forward = None
        self._original_compute_attention = None
        self._original_generate_sample = None
        if self._original_dit_forward is not None:
            from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
                Attention as ModelAttention,
                Block,
                GPT2FeedForward,
                I2VCrossAttention,
                MiniTrainDIT,
            )

            MiniTrainDIT.forward = self._original_dit_forward
            Block.forward = self._original_block_forward
            ModelAttention.forward = self._original_attention_forward
            I2VCrossAttention.forward = self._original_i2v_cache_forward
            GPT2FeedForward.forward = self._original_mlp_forward
            self._original_dit_forward = None
        self._original_i2v_cache_forward = None

    def _install_torch_profiler_ranges(self) -> None:
        """Profile one warm CFG transformer forward with semantic NVTX ranges."""

        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Attention,
            Block,
            GPT2FeedForward,
            I2VCrossAttention,
            MiniTrainDIT,
        )

        self._original_dit_forward = MiniTrainDIT.forward
        self._original_block_forward = Block.forward
        self._original_attention_forward = Attention.forward
        self._original_i2v_cross_attention_forward = I2VCrossAttention.forward
        self._original_mlp_forward = GPT2FeedForward.forward
        patch = self

        def ranged(name: str, fn, *args, **kwargs):
            if not patch._torch_profiler_active:
                return fn(*args, **kwargs)
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        def profiled_block(block, *args, **kwargs):
            return ranged(
                "cosmos25/block_total",
                patch._original_block_forward,
                block,
                *args,
                **kwargs,
            )

        def profiled_attention(attention, *args, **kwargs):
            name = (
                "cosmos25/self_attention_module"
                if bool(getattr(attention, "is_selfattn", False))
                else "cosmos25/cross_attention_module"
            )
            return ranged(
                name,
                patch._original_attention_forward,
                attention,
                *args,
                **kwargs,
            )

        def profiled_i2v_cross_attention(attention, *args, **kwargs):
            return ranged(
                "cosmos25/cross_attention_module",
                patch._original_i2v_cross_attention_forward,
                attention,
                *args,
                **kwargs,
            )

        def profiled_mlp(mlp, *args, **kwargs):
            return ranged(
                "cosmos25/mlp",
                patch._original_mlp_forward,
                mlp,
                *args,
                **kwargs,
            )

        def profiled_dit(model, *args, **kwargs):
            return patch._profile_dit_forward(model, *args, **kwargs)

        Block.forward = profiled_block
        Attention.forward = profiled_attention
        I2VCrossAttention.forward = profiled_i2v_cross_attention
        GPT2FeedForward.forward = profiled_mlp
        MiniTrainDIT.forward = profiled_dit

    @staticmethod
    def _profile_event_us(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _profile_dit_forward(self, model, *args, **kwargs):
        """Capture the last CFG branch of the configured profiling step."""

        assert self._original_dit_forward is not None
        next_forward = self._forward_index + 1
        target_forward = (
            self.config.profile_step * self.config.forwards_per_step
            + self.config.forwards_per_step
            - 1
        )
        should_profile = bool(
            self._sample_name is not None
            and not self._torch_profiler_complete
            and next_forward == target_forward
        )
        if not should_profile:
            return self._original_dit_forward(model, *args, **kwargs)

        from torch.profiler import ProfilerActivity, profile, record_function

        self._torch_profiler_active = True
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                with record_function("cosmos25/transformer_forward"):
                    output = self._original_dit_forward(model, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            self._torch_profiler_active = False

        root = (
            Path(self.config.run_manifest).expanduser().parent.parent
            if self.config.run_manifest
            else Path("/tmp/cosmos25_torch_profile")
        )
        profile_dir = root / "torch_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / "sparse_step_cfg_branch1_trace.json.gz"
        table_path = profile_dir / "cuda_operator_table.txt"
        summary_path = profile_dir / "cuda_operator_summary.json"
        prof.export_chrome_trace(str(trace_path))
        averages = list(prof.key_averages())
        rows = []
        for event in averages:
            self_device_us = self._profile_event_us(
                event, "self_device_time_total", "self_cuda_time_total"
            )
            device_total_us = self._profile_event_us(
                event, "device_time_total", "cuda_time_total"
            )
            rows.append(
                {
                    "key": str(event.key),
                    "count": int(event.count),
                    "self_device_ms": self_device_us / 1000.0,
                    "device_total_ms": device_total_us / 1000.0,
                    "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                    "cpu_total_ms": float(event.cpu_time_total) / 1000.0,
                }
            )
        rows.sort(key=lambda row: row["self_device_ms"], reverse=True)
        table_path.write_text(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=100,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._torch_profiler_outputs = {
            "profiled_forward_index": next_forward,
            "profiled_step_index_zero_based": self.config.profile_step,
            "profiled_cfg_branch": self.config.forwards_per_step - 1,
            "trace": str(trace_path),
            "operator_table": str(table_path),
            "operator_summary": str(summary_path),
        }
        self._torch_profiler_complete = True
        return output

    def _reset_sample(self, name: str) -> None:
        self._sample_name = name
        self._forward_index = -1
        self._dense_calls = 0
        self._sparse_calls = 0
        self._cuda_qkv_calls = 0
        self._method_counts = {
            "role_aligned_route_calls": 0,
            "q_kmeans_calls": 0,
            "k_kmeans_calls": 0,
            "factor_refresh_calls": 0,
            "route_refresh_calls": 0,
            "route_reuse_calls": 0,
            "cfg_route_reuse_calls": 0,
            "temporal_route_reuse_calls": 0,
            "probe_fitted_repair_calls": 0,
            "flashinfer_plan_calls": 0,
            "flashinfer_plan_reuse_calls": 0,
            "cross_attention_kv_cache_hits": 0,
            "cross_attention_kv_cache_misses": 0,
        }
        self._profile_components_ms = {}
        self._profile_details_ms = {}
        self._profile_step_started = None
        self._profile_step_elapsed_ms = None
        self._torch_profiler_active = False
        self._torch_profiler_complete = False
        self._torch_profiler_outputs = {}
        self._density_sum = 0.0
        self._density_count = 0
        self._density_min = float("inf")
        self._density_max = 0.0
        self._branch_dense_calls = [
            0 for _ in range(self.config.forwards_per_step)
        ]
        self._branch_sparse_calls = [
            0 for _ in range(self.config.forwards_per_step)
        ]
        # Centroids, role factors, and repair buffers must never cross videos.
        self._cores.clear()
        self._route_cache.clear()
        self._cross_kv_cache.clear()
        self._cross_layer_by_module.clear()
        self._cross_context_reference.clear()
        self._cross_context_validated.clear()
        self._cross_cache_disabled_branches.clear()
        self._cross_cache_active_printed = False
        if (
            self.config.flashinfer_plan_reuse
            and self._clear_flashinfer_plan_cache is not None
        ):
            self._clear_flashinfer_plan_cache()

    def _finish_sample(self, status: str, error: str | None = None) -> None:
        expected_forwards = self.config.total_steps * self.config.forwards_per_step
        observed_forwards = self._forward_index + 1
        profile: dict[str, Any] = {
            "enabled": bool(self.config.profile_breakdown),
            "step_index_zero_based": int(self.config.profile_step),
            "diffusion_offload_enabled": bool(
                self.config.diffusion_offload_enabled
            ),
            "components_ms": dict(sorted(self._profile_components_ms.items())),
            "details_ms": dict(sorted(self._profile_details_ms.items())),
            "torch_profiler": dict(self._torch_profiler_outputs),
        }
        if self._profile_step_elapsed_ms is not None:
            measured_attention_ms = sum(self._profile_components_ms.values())
            profile.update(
                {
                    "transformer_window_ms": self._profile_step_elapsed_ms,
                    "profiled_attention_ms": measured_attention_ms,
                    "model_and_step_remainder_ms": max(
                        0.0,
                        self._profile_step_elapsed_ms - measured_attention_ms,
                    ),
                }
            )
        record = {
            "sample": self._sample_name,
            "status": status,
            "error": error,
            "method": "N8_custom_v6",
            "observed_self_attention_modules": len(self._layer_by_module),
            "expected_self_attention_modules": self.config.total_layers,
            "observed_transformer_forwards": observed_forwards,
            "expected_transformer_forwards": expected_forwards,
            "dense_attention_calls": self._dense_calls,
            "sparse_attention_calls": self._sparse_calls,
            "cuda_qkv_calls": self._cuda_qkv_calls,
            "cuda_extension_effective": {
                "cosmos_qkv": self._cuda_ext_cosmos_qkv,
                "role_cluster": self._cuda_ext_role_cluster,
                "selector": self._cuda_ext_selector,
            },
            "branch_dense_attention_calls": self._branch_dense_calls,
            "branch_sparse_attention_calls": self._branch_sparse_calls,
            "method_execution": self._method_counts,
            "sparse_density": {
                "mean": (
                    self._density_sum / self._density_count
                    if self._density_count
                    else None
                ),
                "min": self._density_min if self._density_count else None,
                "max": self._density_max if self._density_count else None,
                "calls": self._density_count,
            },
            "method_implementation": {
                "role_aligned_routing": True,
                "q_role_rank": self.config.role_q_rank,
                "k_role_rank": self.config.role_k_rank,
                "q_and_k_kmeans_each_sparse_call": (
                    self.config.route_refresh_every == 1
                    and not self.config.share_cfg_route
                ),
                "route_refresh_every": self.config.route_refresh_every,
                "share_cfg_route": self.config.share_cfg_route,
                "flashinfer_plan_reuse": self.config.flashinfer_plan_reuse,
                "cross_attention_kv_cache": self.config.cross_attention_kv_cache,
                "cross_attention_context_validated_branches": sorted(
                    self._cross_context_validated
                ),
                "cross_attention_cache_disabled_branches": sorted(
                    self._cross_cache_disabled_branches
                ),
                "flash_kmeans_inline_xsq_arbitrary_d": True,
                "probe_fitted_residual_repair": True,
                "probe_rows": self.config.probe_rows,
                "repair_rank": self.config.repair_rank,
            },
            "profile": profile,
            "nonattention_fusions": (
                self._nonattention_fusions.report()
                if self._nonattention_fusions is not None
                else None
            ),
            "config": self.config.provenance(),
        }
        if status == "ok":
            if len(self._layer_by_module) != self.config.total_layers:
                raise RuntimeError(
                    "Cosmos2.5 N8 observed "
                    f"{len(self._layer_by_module)}/{self.config.total_layers} "
                    "self-attention modules."
                )
            if observed_forwards != expected_forwards:
                raise RuntimeError(
                    "Cosmos2.5 N8 observed "
                    f"{observed_forwards}/{expected_forwards} transformer forwards."
                )
        if self.config.run_manifest:
            path = Path(self.config.run_manifest).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "[N8-COSMOS25] "
            f"sample={self._sample_name} status={status} "
            f"dense_calls={self._dense_calls} sparse_calls={self._sparse_calls}",
            flush=True,
        )
        self._sample_name = None

    def _make_core(self):
        from sparsepr.models.common import N8V6Config, N8V6Core

        cfg = self.config
        return N8V6Core(
            N8V6Config(
                num_q_centroids=cfg.num_q_centroids,
                num_k_centroids=cfg.num_k_centroids,
                target_density=cfg.target_density,
                role_q_rank=cfg.role_q_rank,
                role_k_rank=cfg.role_k_rank,
                role_v_rank=cfg.role_v_rank,
                role_v_weight=cfg.role_v_weight,
                metric_sample_tokens=cfg.metric_sample_tokens,
                kmeans_iter_init=cfg.kmeans_iter_init,
                kmeans_iter_step=cfg.kmeans_iter_step,
                kmeans_backend=(
                    "n8_cuda_ext"
                    if self._cuda_ext_role_cluster
                    else "flash"
                ),
                factor_power_iters=cfg.factor_power_iters,
                factor_refresh_every=cfg.factor_refresh_every,
                selector_policy=cfg.selector_policy,
                selector_budget_mode=cfg.selector_budget_mode,
                svg_ear_top_p=cfg.selector_top_p,
                adaptive_budget_min_density=cfg.adaptive_budget_min_density,
                adaptive_budget_max_density=cfg.adaptive_budget_max_density,
                probe_rows=cfg.probe_rows,
                repair_rank=cfg.repair_rank,
                role_kv_matmul_precision=cfg.role_kv_matmul_precision,
                profile_breakdown=cfg.profile_breakdown,
            )
        )

    @staticmethod
    def _sync(tensor: torch.Tensor) -> None:
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _profile_call(self, name: str, tensor: torch.Tensor, fn):
        self._sync(tensor)
        started = time.perf_counter()
        output = fn()
        self._sync(tensor)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._profile_components_ms[name] = (
            self._profile_components_ms.get(name, 0.0) + elapsed_ms
        )
        return output

    def _accumulate_profile_details(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if name.endswith("_ms") and isinstance(value, (int, float)):
                self._profile_details_ms[name] = (
                    self._profile_details_ms.get(name, 0.0) + float(value)
                )

    def _route_cache_key(self, module_id: int, branch: int) -> tuple[int, int]:
        return (module_id, 0 if self.config.share_cfg_route else branch)

    def _route_action(
        self,
        entry: _RouteCacheEntry | None,
        *,
        step: int,
        branch: int,
    ) -> str:
        """Return refresh, cfg_reuse, or temporal_reuse for one sparse call."""

        if entry is None:
            return "refresh"
        if self.config.share_cfg_route and entry.last_step == step:
            return "cfg_reuse"
        sparse_step = step - self.config.dense_first_steps
        if sparse_step < 0 or sparse_step % self.config.route_refresh_every == 0:
            return "refresh"
        return "temporal_reuse"

    @staticmethod
    def _compact_cached_route(entry: _RouteCacheEntry, reference: torch.Tensor) -> None:
        """Drop step-sized tensors while retaining labels, mask, and probes."""

        empty = reference.new_empty((0,))
        entry.route.q_role_features = empty
        entry.route.k_permuted = empty
        entry.route.v_permuted = empty

    def compute_qkv(
        self,
        attention,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the gated Cosmos preprocessing kernel when its contract matches."""

        if self._original_compute_qkv is None:
            raise RuntimeError("Cosmos2.5 N8 patch is not installed.")
        use_official = (
            not self._cuda_ext_cosmos_qkv
            or self._sample_name is None
            or not bool(getattr(attention, "is_selfattn", False))
            or context is not None
            or rope_emb is None
            or not torch.is_tensor(rope_emb)
            or not torch.is_tensor(x)
            or x.ndim != 3
            or getattr(attention, "qkv_format", None) != "bshd"
            or bool(getattr(attention, "use_wan_fp32_strategy", False))
            or not isinstance(getattr(attention, "v_norm", None), torch.nn.Identity)
            or bool(
                getattr(
                    getattr(attention, "q_norm", None),
                    "zero_centered_gamma",
                    False,
                )
            )
            or bool(
                getattr(
                    getattr(attention, "k_norm", None),
                    "zero_centered_gamma",
                    False,
                )
            )
        )
        if use_official:
            return self._original_compute_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )

        q_weight = getattr(getattr(attention, "q_norm", None), "weight", None)
        k_weight = getattr(getattr(attention, "k_norm", None), "weight", None)
        if not torch.is_tensor(q_weight) or not torch.is_tensor(k_weight):
            return self._original_compute_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )
        q_epsilon = float(getattr(attention.q_norm, "eps", 1e-6))
        k_epsilon = float(getattr(attention.k_norm, "eps", 1e-6))
        if q_epsilon != k_epsilon:
            return self._original_compute_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )

        # Learned projections remain official. Only the model-specific
        # normalization, rotary transform, and BSHD shaping are compiled.
        q = attention.q_proj(x)
        k = attention.k_proj(x)
        v = attention.v_proj(x)
        try:
            from sparsepr.kernels.n8_extension import (
                cosmos25_qkv_norm_rope_bshd,
            )

            output = cosmos25_qkv_norm_rope_bshd(
                q,
                k,
                v,
                q_weight,
                k_weight,
                rope_emb,
                heads=int(attention.n_heads),
                epsilon=q_epsilon,
            )
        except Exception as exc:
            # Disable after the first contract/load failure so a long video
            # does not repeatedly pay exception and duplicate-projection cost.
            self._cuda_ext_cosmos_qkv = False
            if not self._cuda_qkv_warned:
                warnings.warn(
                    "Cosmos2.5 CUDA QKV preprocessing failed; disabling it "
                    f"and restoring official Transformer Engine QKV: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._cuda_qkv_warned = True
            return self._original_compute_qkv(
                attention,
                x,
                context=context,
                rope_emb=rope_emb,
            )
        self._cuda_qkv_calls += 1
        return output

    @staticmethod
    def _context_signature(context: Any) -> tuple[Any, ...]:
        """Shape contract for re-materialized, diffusion-invariant context."""

        if isinstance(context, (tuple, list)):
            return tuple(
                item
                for value in context
                for item in Cosmos25N8Patch._context_signature(value)
            )
        if not torch.is_tensor(context):
            return (type(context).__name__, id(context))
        return (
            tuple(context.shape),
            tuple(context.stride()),
            str(context.dtype),
            str(context.device),
        )

    @staticmethod
    def _clone_context(context: Any) -> Any:
        if isinstance(context, tuple):
            return tuple(Cosmos25N8Patch._clone_context(value) for value in context)
        if isinstance(context, list):
            return [Cosmos25N8Patch._clone_context(value) for value in context]
        if torch.is_tensor(context):
            return context.detach().clone()
        return context

    @staticmethod
    def _context_equal(left: Any, right: Any) -> bool:
        if isinstance(left, (tuple, list)):
            return bool(
                isinstance(right, type(left))
                and len(left) == len(right)
                and all(
                    Cosmos25N8Patch._context_equal(a, b)
                    for a, b in zip(left, right)
                )
            )
        if torch.is_tensor(left):
            return bool(torch.is_tensor(right) and torch.equal(left, right))
        return left == right

    def compute_i2v_cross_qkv(
        self,
        attention,
        x: torch.Tensor,
        context,
        rope_emb: torch.Tensor | None = None,
    ):
        """Reuse exact text/image K/V while recomputing the step-dependent Q."""

        if self._original_i2v_compute_qkv is None:
            raise RuntimeError("Cosmos2.5 N8 patch is not installed.")
        if (
            not self.config.cross_attention_kv_cache
            or self._sample_name is None
            or self._forward_index < 0
        ):
            return self._original_i2v_compute_qkv(
                attention, x, context, rope_emb
            )

        branch = self._forward_index % self.config.forwards_per_step
        if not self._cross_cache_active_printed:
            print(
                "[N8-COSMOS25] exact cross-attention text K/V cache active.",
                flush=True,
            )
            self._cross_cache_active_printed = True
        module_id = id(attention)
        if module_id not in self._cross_layer_by_module:
            self._cross_layer_by_module[module_id] = len(
                self._cross_layer_by_module
            )
        cross_layer = self._cross_layer_by_module[module_id]
        if branch in self._cross_cache_disabled_branches:
            return self._original_i2v_compute_qkv(
                attention, x, context, rope_emb
            )
        if cross_layer == 0 and branch not in self._cross_context_validated:
            reference = self._cross_context_reference.get(branch)
            if reference is None:
                self._cross_context_reference[branch] = (
                    self._forward_index,
                    self._clone_context(context),
                )
            elif reference[0] != self._forward_index:
                if self._context_equal(reference[1], context):
                    self._cross_context_validated.add(branch)
                    del self._cross_context_reference[branch]
                else:
                    self._cross_cache_disabled_branches.add(branch)
                    self._cross_kv_cache = {
                        key: value
                        for key, value in self._cross_kv_cache.items()
                        if key[1] != branch
                    }
                    warnings.warn(
                        "Cosmos2.5 cross-attention context changed across "
                        f"steps for CFG branch {branch}; disabling its K/V cache.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return self._original_i2v_compute_qkv(
                        attention, x, context, rope_emb
                    )
        cache_key = (id(attention), branch)
        signature = self._context_signature(context)
        cached = self._cross_kv_cache.get(cache_key)
        if cached is None or cached[0] != signature:
            output = self._original_i2v_compute_qkv(
                attention, x, context, rope_emb
            )
            # Image-context K/V is sequence-sized and retaining it for every
            # layer/CFG branch costs tens of GiB for Cosmos I2V. Cache only the
            # invariant text K/V; recompute image K/V exactly on every call.
            self._cross_kv_cache[cache_key] = (signature, output[1], output[2])
            self._method_counts["cross_attention_kv_cache_misses"] += 1
            return output

        from einops import rearrange

        q = attention.q_proj(x)
        q = rearrange(
            q,
            "b ... (h d) -> b ... h d",
            h=attention.n_heads,
            d=attention.head_dim,
        )
        q = attention.q_norm(q)
        _, img_context = context
        k_img = attention.k_img(img_context)
        v_img = attention.v_img(img_context)
        k_img, v_img = map(
            lambda tensor: rearrange(
                tensor,
                "b ... (h d) -> b ... h d",
                h=attention.n_heads,
                d=attention.head_dim,
            ),
            (k_img, v_img),
        )
        self._method_counts["cross_attention_kv_cache_hits"] += 1
        return q, cached[1], cached[2], attention.k_img_norm(k_img), v_img

    def _n8_attention(
        self,
        attention,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
        *,
        layer: int,
        step: int,
        branch: int,
    ) -> torch.Tensor:
        from sparsepr.kernels.triton.permute import (
            apply_inverse_permutation_triton,
            permute_tensor_by_labels_triton,
        )
        from sparsepr.kmeans_utils import (
            dynamic_block_sparse_fwd_flashinfer,
            pop_last_flashinfer_timings,
            set_flashinfer_kernel_profile_enabled,
            set_kmeans_kernel_profile_enabled,
        )

        source_dtype = v_bshd.dtype
        profile_this = bool(
            self.config.profile_breakdown and step == self.config.profile_step
        )
        kernel_dtype = (
            source_dtype
            if source_dtype in {torch.float16, torch.bfloat16}
            else torch.bfloat16
        )

        def prepare_layout():
            return (
                bshd_to_bhnd(q_bshd, "Q").to(kernel_dtype),
                bshd_to_bhnd(k_bshd, "K").to(kernel_dtype),
                bshd_to_bhnd(v_bshd, "V").to(kernel_dtype),
            )

        q, k, v = (
            self._profile_call("attention_input_layout", q_bshd, prepare_layout)
            if profile_this
            else prepare_layout()
        )
        if q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError(
                "Cosmos2.5 N8 requires matching self-attention Q/K/V; got "
                f"Q={tuple(q.shape)} K={tuple(k.shape)} V={tuple(v.shape)}."
            )
        batch, heads, tokens, _ = q.shape
        if batch != 1:
            raise RuntimeError(
                f"Cosmos2.5 N8 currently requires batch one, got {batch}."
            )

        repair_core_key = (id(attention), branch)
        repair_core = self._cores.get(repair_core_key)
        if repair_core is None:
            repair_core = self._make_core()
            self._cores[repair_core_key] = repair_core
        route_core_key = (
            (id(attention), 0)
            if self.config.share_cfg_route
            else repair_core_key
        )
        route_core = self._cores.get(route_core_key)
        if route_core is None:
            route_core = self._make_core()
            self._cores[route_core_key] = route_core

        cache_key = self._route_cache_key(id(attention), branch)
        entry = self._route_cache.get(cache_key)
        route_action = self._route_action(
            entry,
            step=step,
            branch=branch,
        )
        if route_action == "refresh":
            refresh_factors = bool(
                route_core.q_factor_current is None
                or route_core.k_factor_current is None
                or route_core.route_call_index
                % self.config.factor_refresh_every
                == 0
            )
            route_core.profile_breakdown_active = profile_this
            if profile_this:
                set_kmeans_kernel_profile_enabled(True)
                self._sync(q)
                route_started = time.perf_counter()
            try:
                route = route_core.route(q, k, v, total_key_tokens=tokens)
            finally:
                route_core.profile_breakdown_active = False
                if profile_this:
                    set_kmeans_kernel_profile_enabled(False)
            if profile_this:
                self._sync(q)
                route_elapsed_ms = (time.perf_counter() - route_started) * 1000.0
            q_order = (
                torch.argsort(route.q_labels, dim=-1)
                .to(torch.int32)
                .contiguous()
            )
            entry = _RouteCacheEntry(
                origin_step=step,
                last_step=step,
                route=route,
                q_sorted_indices=q_order,
            )
            self._route_cache[cache_key] = entry
            self._method_counts["role_aligned_route_calls"] += 1
            self._method_counts["q_kmeans_calls"] += 1
            self._method_counts["k_kmeans_calls"] += 1
            self._method_counts["factor_refresh_calls"] += int(refresh_factors)
            self._method_counts["route_refresh_calls"] += 1
            if profile_this:
                route_detail = dict(route_core.last_route_profile)
                self._accumulate_profile_details(route_detail)
                role_projection_ms = sum(
                    float(route_detail.get(name, 0.0))
                    for name in (
                        "n8_route_kv_projection_init_ms",
                        "n8_route_q_projection_init_ms",
                    )
                )
                role_factorization_ms = sum(
                    float(route_detail.get(name, 0.0))
                    for name in (
                        "n8_route_kv_metric_factor_ms",
                        "n8_route_q_metric_factor_ms",
                    )
                )
                kmeans_ms = sum(
                    float(route_detail.get(name, 0.0))
                    for name in (
                        "n8_route_kmeans_k_ms",
                        "n8_route_kmeans_q_ms",
                    )
                )
                centroid_ms = sum(
                    float(route_detail.get(name, 0.0))
                    for name in (
                        "n8_route_kv_centroids_ms",
                        "n8_route_q_centroids_ms",
                    )
                )
                selector_permutation_ms = sum(
                    float(route_detail.get(name, 0.0))
                    for name in (
                        "n8_route_kv_permute_ms",
                        "n8_route_selector_scores_ms",
                        "n8_route_selector_budget_ms",
                    )
                )
                for name, value in (
                    ("role_factorization", role_factorization_ms),
                    ("role_projection", role_projection_ms),
                    ("kmeans", kmeans_ms),
                    ("centroid_updates", centroid_ms),
                    ("selector_and_kv_permutation", selector_permutation_ms),
                ):
                    self._profile_components_ms[name] = (
                        self._profile_components_ms.get(name, 0.0) + value
                    )
                categorized_route_ms = (
                    role_projection_ms
                    + role_factorization_ms
                    + kmeans_ms
                    + centroid_ms
                    + selector_permutation_ms
                )
                self._profile_components_ms["route_other"] = (
                    self._profile_components_ms.get("route_other", 0.0)
                    + max(0.0, route_elapsed_ms - categorized_route_ms)
                )
                self._profile_details_ms["n8_route_total_ms"] = (
                    self._profile_details_ms.get("n8_route_total_ms", 0.0)
                    + route_elapsed_ms
                )
        else:
            assert entry is not None

            def current_kv_permutation():
                k_permuted, _ = permute_tensor_by_labels_triton(
                    k,
                    None,
                    dim=2,
                    sorted_indices=entry.route.k_sorted_indices,
                )
                v_permuted, _ = permute_tensor_by_labels_triton(
                    v,
                    None,
                    dim=2,
                    sorted_indices=entry.route.k_sorted_indices,
                )
                return k_permuted, v_permuted

            k_permuted, v_permuted = (
                self._profile_call(
                    "reused_kv_permutation",
                    q,
                    current_kv_permutation,
                )
                if profile_this
                else current_kv_permutation()
            )
            route = replace(
                entry.route,
                k_permuted=k_permuted,
                v_permuted=v_permuted,
            )
            q_order = entry.q_sorted_indices
            entry.last_step = step
            self._method_counts["route_reuse_calls"] += 1
            reuse_counter = {
                "cfg_reuse": "cfg_route_reuse_calls",
                "temporal_reuse": "temporal_route_reuse_calls",
            }[route_action]
            self._method_counts[reuse_counter] += 1

        def q_permute():
            q_permuted_local, _ = permute_tensor_by_labels_triton(
                q, None, dim=2, sorted_indices=q_order
            )
            return q_permuted_local

        q_permuted = (
            self._profile_call("q_permutation", q, q_permute)
            if profile_this
            else q_permute()
        )
        dynamic_map = route.video_dynamic_map.view(
            batch,
            heads,
            route_core.config.num_q_centroids,
            route_core.config.num_k_centroids,
        )
        q_sizes = route.q_cluster_sizes.view(
            batch, heads, route_core.config.num_q_centroids
        )
        k_sizes = route.k_cluster_sizes.view(
            batch, heads, route_core.config.num_k_centroids
        )
        def flash_attention():
            plan_key = None
            plan_token = None
            if self.config.flashinfer_plan_reuse:
                plan_key = self._route_cache_key(id(attention), branch)
                plan_token = (id(entry.route), entry.origin_step)
            return dynamic_block_sparse_fwd_flashinfer(
                q_permuted,
                route.k_permuted,
                route.v_permuted,
                dynamic_map,
                q_sizes,
                k_sizes,
                is_cpu=False,
                plan_cache_key=plan_key,
                plan_cache_token=plan_token,
            )

        if self.config.flashinfer_plan_reuse:
            plan_counter = (
                "flashinfer_plan_calls"
                if route_action == "refresh"
                else "flashinfer_plan_reuse_calls"
            )
            self._method_counts[plan_counter] += 1

        if profile_this:
            set_flashinfer_kernel_profile_enabled(True)
            try:
                output_permuted = self._profile_call(
                    "flashinfer_fa3", q, flash_attention
                )
                self._accumulate_profile_details(pop_last_flashinfer_timings())
            finally:
                set_flashinfer_kernel_profile_enabled(False)
        else:
            output_permuted = flash_attention()

        inverse = lambda: apply_inverse_permutation_triton(
            output_permuted, q_order, dim=2
        )
        base = (
            self._profile_call("inverse_permutation", q, inverse)
            if profile_this
            else inverse()
        )

        repair = lambda: repair_core.repair(base, q, k, v, route)
        if profile_this:
            repaired, repair_info = self._profile_call(
                "probe_fitted_repair", q, repair
            )
            self._accumulate_profile_details(repair_core.last_repair_profile)
            for key, value in repair_info.items():
                if isinstance(value, (str, int, float, bool)):
                    # Retain non-timing method evidence in the runtime record.
                    continue
        else:
            repaired, repair_info = repair()
        effective_density = float(repair_info["effective_video_density"])
        self._density_sum += effective_density
        self._density_count += 1
        self._density_min = min(self._density_min, effective_density)
        self._density_max = max(self._density_max, effective_density)
        assert entry is not None
        if route.probe_rows is not None and entry.route.probe_rows is None:
            entry.route.probe_rows = route.probe_rows
            entry.route.probe_weights = route.probe_weights
        self._compact_cached_route(entry, q)
        del repair_info
        self._method_counts["probe_fitted_repair_calls"] += 1

        def output_projection():
            result = bhnd_to_bshd_flat(repaired).to(source_dtype)
            return attention.output_dropout(attention.output_proj(result))

        return (
            self._profile_call("attention_output_projection", q, output_projection)
            if profile_this
            else output_projection()
        )

    def compute_attention(
        self,
        attention,
        q,
        k,
        v,
        video_size=None,
        kv_cache_cfg=None,
    ):
        if self._original_compute_attention is None:
            raise RuntimeError("Cosmos2.5 N8 patch is not installed.")
        if not bool(getattr(attention, "is_selfattn", False)):
            return self._original_compute_attention(
                attention,
                q,
                k,
                v,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
            )
        # Model initialization and other out-of-band calls remain official
        # dense. Only calls owned by Inference._generate_sample are counted.
        if self._sample_name is None:
            return self._original_compute_attention(
                attention,
                q,
                k,
                v,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
            )
        if kv_cache_cfg is not None:
            raise RuntimeError(
                "Cosmos2.5 N8 Image2World does not support a self-attention KV cache."
            )

        module_id = id(attention)
        if module_id not in self._layer_by_module:
            layer = len(self._layer_by_module)
            if layer >= self.config.total_layers:
                raise RuntimeError(
                    "Cosmos2.5 N8 encountered more self-attention modules than "
                    f"the configured {self.config.total_layers}."
                )
            self._layer_by_module[module_id] = layer
            self._modules[module_id] = attention
        layer = self._layer_by_module[module_id]
        if layer == 0:
            self._forward_index += 1
        if self._forward_index < 0:
            raise RuntimeError(
                "Cosmos2.5 self-attention modules were not called in layer order."
            )
        step = self._forward_index // self.config.forwards_per_step
        branch = self._forward_index % self.config.forwards_per_step
        if step >= self.config.total_steps:
            raise RuntimeError(
                f"Cosmos2.5 N8 received unexpected denoiser step {step}."
            )

        force_dense = (
            step < self.config.dense_first_steps
            or layer < self.config.dense_first_layers
        )
        profile_this = bool(
            self.config.profile_breakdown and step == self.config.profile_step
        )
        if profile_this and layer == 0 and branch == 0:
            self._sync(q)
            self._profile_step_started = time.perf_counter()
        if force_dense:
            self._dense_calls += 1
            self._branch_dense_calls[branch] += 1
            dense = lambda: self._original_compute_attention(
                    attention,
                    q,
                    k,
                    v,
                    video_size=video_size,
                    kv_cache_cfg=kv_cache_cfg,
                )
            result = (
                self._profile_call("dense_attention", q, dense)
                if profile_this
                else dense()
            )
        else:
            self._sparse_calls += 1
            self._branch_sparse_calls[branch] += 1
            result = self._n8_attention(
                attention,
                q,
                k,
                v,
                layer=layer,
                step=step,
                branch=branch,
            )
        if (
            profile_this
            and layer == self.config.total_layers - 1
            and branch == self.config.forwards_per_step - 1
            and self._profile_step_started is not None
        ):
            self._sync(q)
            self._profile_step_elapsed_ms = (
                time.perf_counter() - self._profile_step_started
            ) * 1000.0
        return result

    def generate_sample(self, inference, sample, output_dir):
        if self._original_generate_sample is None:
            raise RuntimeError("Cosmos2.5 N8 patch is not installed.")
        self._reset_sample(str(sample.name))
        error: str | None = None
        try:
            return self._original_generate_sample(inference, sample, output_dir)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            status = "ok" if error is None else "error"
            try:
                self._finish_sample(status, error)
            except Exception:
                if error is None:
                    raise
                print(
                    "[N8-COSMOS25] failed to write/validate error summary.",
                    flush=True,
                )


_INSTALLED_PATCH: Cosmos25N8Patch | None = None


def install_cosmos25_n8_patch(
    config: Cosmos25N8Config | dict[str, Any],
) -> Cosmos25N8Patch:
    """Install the N8 adapter once after ``cosmos_oss.init_environment``."""

    global _INSTALLED_PATCH
    if not isinstance(config, Cosmos25N8Config):
        config = Cosmos25N8Config.from_dict(config)
    if _INSTALLED_PATCH is not None:
        if _INSTALLED_PATCH.config != config:
            raise RuntimeError(
                "Cosmos2.5 N8 is already installed with a different configuration."
            )
        return _INSTALLED_PATCH
    patch = Cosmos25N8Patch(config)
    patch.install()
    _INSTALLED_PATCH = patch
    return patch
