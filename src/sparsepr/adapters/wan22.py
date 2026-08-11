"""SVG2/SAP sparse attention patch for native Wan2.2 video models.

This module targets the official Wan2.2 ``wan.WanI2V``/``wan.WanTI2V`` stacks, whose transformer
uses ``wan.modules.model.WanSelfAttention`` rather than Diffusers'
``WanAttention`` processor interface.  The patch mirrors the existing SVG2 Wan
Diffusers path: semantic k-means partition, FlashInfer dynamic block-sparse
attention, and optional reused low-rank residual correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import torch


@dataclass
class Wan22TI2VSparseConfig:
    pattern: str = "SAP"
    first_layers_fp: int = 0
    first_sparse_forward: int = 0
    num_q_centroids: int = 300
    num_k_centroids: int = 1000
    top_p_kmeans: float = 0.90
    min_kc_ratio: float = 0.10
    kmeans_iter_init: int = 50
    kmeans_iter_step: int = 2
    zero_step_kmeans_init: bool = False
    logging_file: str | None = None
    lowrank_rank: int = 64
    basis_refresh_every: int = 8
    lowrank_accum_dtype: str = "bf16"
    lowrank_fit_tokens: int = 4096
    lowrank_cache_device: str = "cuda_factors"
    lowrank_layer_ranges: str = "all"
    n8_target_density: float = 0.22
    n8_probe_rows: int = 64
    n8_repair_rank: int = 16
    n8_role_q_rank: int = 64
    n8_role_k_rank: int = 48
    n8_role_v_rank: int = 16
    n8_role_v_weight: float = 0.0
    n8_metric_sample_tokens: int = 2048
    n8_factor_power_iters: int = 2
    n8_factor_refresh_every: int = 1
    n8_compile_role_kv_projection: bool = True
    n8_role_kv_matmul_precision: str = "high"
    n8_selector_policy: str = "svg_ear_value"
    dense_attention_backend: str = "fa2"
    n8_flashinfer_backend: str = "auto"
    block_fusion: bool = False
    cfg_layer0_reuse: bool = False


_ORIGINALS: dict[str, Any] = {}
_CONFIG = Wan22TI2VSparseConfig()
_WAN_ROPE_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_N8_CUDA_EXT_WAN_WARNED = False
_DENOISING_PROFILE_STATE: dict[str, Any] = {}
_N8_ATTENTION_REPLAY_CAPTURED = False
_WAN_N8_FLOP_AUDIT: dict[str, Any] = {}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in {
        "", "0", "false", "no", "off"
    }


def reset_wan22_flop_audit_state() -> None:
    """Reset optional exact executed-pair counters for one generated video."""
    _WAN_N8_FLOP_AUDIT.clear()
    if not _env_enabled("SPARSEPR_WAN_N8_FLOP_AUDIT"):
        return
    _WAN_N8_FLOP_AUDIT.update(
        {
            "enabled": True,
            "dense_calls": 0,
            "dense_reused_calls": 0,
            "sparse_calls": 0,
            "dense_token_pairs": 0,
            "probe_token_pairs": 0,
            "selected_token_pairs_cuda": None,
            "q_kmeans_iterations": 0,
            "k_kmeans_iterations": 0,
            "module_flops": 0,
            "module_flops_by_category": {},
            "module_calls_by_category": {},
            "cross_attention_token_pairs": 0,
            "cross_attention_calls": 0,
            "cross_context_lens_cache": {},
            "calls_by_model": {},
            "sequence_lengths": set(),
            "heads": set(),
            "head_dims": set(),
        }
    )


def _flop_audit_call(model_name: str | None, kind: str) -> None:
    if not _WAN_N8_FLOP_AUDIT.get("enabled"):
        return
    model = str(model_name or "unknown")
    models = _WAN_N8_FLOP_AUDIT["calls_by_model"]
    row = models.setdefault(
        model,
        {"dense_calls": 0, "dense_reused_calls": 0, "sparse_calls": 0},
    )
    row[f"{kind}_calls"] += 1
    _WAN_N8_FLOP_AUDIT[f"{kind}_calls"] += 1


def wan22_flop_audit_report() -> dict[str, Any] | None:
    """Materialize exact counters once after generation has completed."""
    if not _WAN_N8_FLOP_AUDIT.get("enabled"):
        return None
    selected = _WAN_N8_FLOP_AUDIT.get("selected_token_pairs_cuda")
    selected_pairs = int(selected.item()) if selected is not None else 0
    dense_pairs = int(_WAN_N8_FLOP_AUDIT["dense_token_pairs"])
    probe_pairs = int(_WAN_N8_FLOP_AUDIT["probe_token_pairs"])
    head_dims = sorted(int(value) for value in _WAN_N8_FLOP_AUDIT["head_dims"])
    if len(head_dims) > 1:
        raise RuntimeError(
            f"WAN FLOP audit observed multiple head dimensions: {head_dims}"
        )
    head_dim = head_dims[0] if head_dims else 0
    return {
        "status": "observed",
        "counter_protocol": "exact_selected_token_pairs_counter_only",
        "multiply_add_flops": 2,
        "dense_calls": int(_WAN_N8_FLOP_AUDIT["dense_calls"]),
        "dense_reused_calls": int(
            _WAN_N8_FLOP_AUDIT["dense_reused_calls"]
        ),
        "sparse_calls": int(_WAN_N8_FLOP_AUDIT["sparse_calls"]),
        "dense_token_pairs": dense_pairs,
        "selected_sparse_token_pairs": selected_pairs,
        "dense_probe_token_pairs": probe_pairs,
        "q_kmeans_iterations": int(
            _WAN_N8_FLOP_AUDIT["q_kmeans_iterations"]
        ),
        "k_kmeans_iterations": int(
            _WAN_N8_FLOP_AUDIT["k_kmeans_iterations"]
        ),
        "n8_config": {
            "num_q_centroids": int(_CONFIG.num_q_centroids),
            "num_k_centroids": int(_CONFIG.num_k_centroids),
            "role_q_rank": int(_CONFIG.n8_role_q_rank),
            "role_k_rank": int(_CONFIG.n8_role_k_rank),
            "role_v_rank": int(_CONFIG.n8_role_v_rank),
            "metric_sample_tokens": int(
                _CONFIG.n8_metric_sample_tokens
            ),
            "probe_rows": int(_CONFIG.n8_probe_rows),
            "repair_rank": int(_CONFIG.n8_repair_rank),
            "kmeans_iter_init": int(_CONFIG.kmeans_iter_init),
            "kmeans_iter_step": int(_CONFIG.kmeans_iter_step),
        },
        "module_flops": int(_WAN_N8_FLOP_AUDIT["module_flops"]),
        "module_flops_by_category": dict(
            _WAN_N8_FLOP_AUDIT["module_flops_by_category"]
        ),
        "module_calls_by_category": dict(
            _WAN_N8_FLOP_AUDIT["module_calls_by_category"]
        ),
        "cross_attention_token_pairs": int(
            _WAN_N8_FLOP_AUDIT["cross_attention_token_pairs"]
        ),
        "cross_attention_calls": int(
            _WAN_N8_FLOP_AUDIT["cross_attention_calls"]
        ),
        "sequence_lengths": sorted(
            int(value) for value in _WAN_N8_FLOP_AUDIT["sequence_lengths"]
        ),
        "heads": sorted(int(value) for value in _WAN_N8_FLOP_AUDIT["heads"]),
        "head_dims": head_dims,
        "calls_by_model": _WAN_N8_FLOP_AUDIT["calls_by_model"],
        "qk_pv_flops": {
            "dense": 4 * head_dim * dense_pairs,
            "sparse_base": 4 * head_dim * selected_pairs,
            "dense_probes": 4 * head_dim * probe_pairs,
            "total_executed": 4
            * head_dim
            * (dense_pairs + selected_pairs + probe_pairs),
        },
        "notes": [
            "Selected sparse pairs are counted from the actual per-head cluster mask and cluster sizes.",
            "Dense M64 probe attention is charged additively because it is executed in addition to the sparse base.",
            "Reused CFG layer-0 calls are reported but assigned zero QK/PV FLOPs.",
            "The counter adds no CUDA timing events; one scalar is copied to CPU only when this report is materialized.",
            "Linear and convolution FLOPs are counted from the actual denoiser module input/output shapes.",
            "Cross-attention QK/PV pairs are counted manually because fused attention kernels are opaque to module hooks.",
        ],
    }


def _flop_module_category(name: str) -> str:
    if ".self_attn." in name:
        return "self_attention_projection"
    if ".cross_attn." in name:
        return "cross_attention_projection"
    if ".ffn." in name:
        return "ffn"
    return "embedding_head_other"


def install_wan22_flop_audit_hooks(model: Any, model_name: str) -> None:
    """Install shape-only FLOP hooks on one WAN denoiser expert."""
    if not _env_enabled("SPARSEPR_WAN_N8_FLOP_AUDIT"):
        return
    if hasattr(model, "_wan_n8_flop_audit_handles"):
        return

    handles = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            category = _flop_module_category(module_name)

            def linear_hook(
                owner: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _category: str = category,
            ) -> None:
                if not _WAN_N8_FLOP_AUDIT.get("enabled"):
                    return
                if not torch.is_tensor(output):
                    return
                flops = (
                    2
                    * int(output.numel())
                    * int(owner.in_features)
                )
                _WAN_N8_FLOP_AUDIT["module_flops"] += flops
                by_category = _WAN_N8_FLOP_AUDIT[
                    "module_flops_by_category"
                ]
                by_category[_category] = (
                    int(by_category.get(_category, 0)) + flops
                )
                calls = _WAN_N8_FLOP_AUDIT["module_calls_by_category"]
                calls[_category] = int(calls.get(_category, 0)) + 1

            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(
            module,
            (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
        ):
            category = _flop_module_category(module_name)

            def convolution_hook(
                owner: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _category: str = category,
            ) -> None:
                if not _WAN_N8_FLOP_AUDIT.get("enabled"):
                    return
                if not torch.is_tensor(output):
                    return
                kernel = 1
                for value in owner.kernel_size:
                    kernel *= int(value)
                per_output = (
                    int(owner.in_channels) // int(owner.groups)
                ) * kernel
                flops = 2 * int(output.numel()) * per_output
                _WAN_N8_FLOP_AUDIT["module_flops"] += flops
                by_category = _WAN_N8_FLOP_AUDIT[
                    "module_flops_by_category"
                ]
                by_category[_category] = (
                    int(by_category.get(_category, 0)) + flops
                )
                calls = _WAN_N8_FLOP_AUDIT["module_calls_by_category"]
                calls[_category] = int(calls.get(_category, 0)) + 1

            handles.append(module.register_forward_hook(convolution_hook))

    for layer, block in enumerate(getattr(model, "blocks", [])):
        cross = getattr(block, "cross_attn", None)
        if cross is None:
            continue

        def cross_pre_hook(
            owner: Any,
            inputs: tuple[Any, ...],
            *,
            _layer: int = layer,
        ) -> None:
            del _layer
            if not _WAN_N8_FLOP_AUDIT.get("enabled"):
                return
            if len(inputs) < 2:
                raise RuntimeError(
                    "WAN cross-attention FLOP hook expected x and context."
                )
            x, context = inputs[:2]
            context_lens = inputs[2] if len(inputs) > 2 else None
            batch = int(x.shape[0])
            query_tokens = int(x.shape[1])
            context_tokens = int(context.shape[1])
            image_tokens = min(257, context_tokens)
            padded_text_tokens = max(0, context_tokens - image_tokens)
            if context_lens is None:
                attended_text_tokens = batch * padded_text_tokens
            elif torch.is_tensor(context_lens):
                cache_key = (
                    str(context_lens.device),
                    int(context_lens.data_ptr()),
                    tuple(int(value) for value in context_lens.shape),
                )
                cache = _WAN_N8_FLOP_AUDIT["cross_context_lens_cache"]
                if cache_key not in cache:
                    cache[cache_key] = int(context_lens.sum().item())
                attended_text_tokens = int(cache[cache_key])
            elif isinstance(context_lens, (list, tuple)):
                attended_text_tokens = sum(int(value) for value in context_lens)
            else:
                attended_text_tokens = int(context_lens)
            key_rows = batch * image_tokens + attended_text_tokens
            pairs = (
                int(owner.num_heads)
                * query_tokens
                * key_rows
            )
            _WAN_N8_FLOP_AUDIT["cross_attention_token_pairs"] += pairs
            _WAN_N8_FLOP_AUDIT["cross_attention_calls"] += 1

        handles.append(cross.register_forward_pre_hook(cross_pre_hook))

    model._wan_n8_flop_audit_handles = handles
    model._wan_n8_flop_audit_name = str(model_name)


def _maybe_capture_n8_attention_replay(
    attn: Any,
    *,
    q_permuted: torch.Tensor,
    k_permuted: torch.Tensor,
    v_permuted: torch.Tensor,
    dynamic_map: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    output_permuted: torch.Tensor,
    flashinfer_timings: dict[str, Any],
) -> None:
    """Save one real post-routing FlashInfer call for an offline kernel gate."""
    global _N8_ATTENTION_REPLAY_CAPTURED

    raw_path = os.environ.get("WAN_N8_ATTENTION_REPLAY_CAPTURE", "").strip()
    if not raw_path or _N8_ATTENTION_REPLAY_CAPTURED:
        return
    step = int(getattr(attn, "_svg_generation_step", -1))
    layer = int(getattr(attn, "_svg_layer_idx", -1))
    branch = int(getattr(attn, "_svg_cfg_branch", -1))
    wanted_step = int(
        os.environ.get("WAN_N8_ATTENTION_CAPTURE_STEP", str(step))
    )
    wanted_layer = int(
        os.environ.get("WAN_N8_ATTENTION_CAPTURE_LAYER", str(layer))
    )
    wanted_branch = int(
        os.environ.get("WAN_N8_ATTENTION_CAPTURE_BRANCH", str(branch))
    )
    if (step, layer, branch) != (
        wanted_step,
        wanted_layer,
        wanted_branch,
    ):
        return

    total_heads = int(q_permuted.shape[1])
    raw_heads = os.environ.get(
        "WAN_N8_ATTENTION_CAPTURE_HEADS",
        "7,16,17,22",
    )
    if raw_heads.strip().lower() == "all":
        heads = list(range(total_heads))
    else:
        heads = sorted(
            {
                int(token.strip())
                for token in raw_heads.split(",")
                if token.strip()
            }
        )
    if not heads or heads[0] < 0 or heads[-1] >= total_heads:
        raise RuntimeError(
            "WAN_N8_ATTENTION_CAPTURE_HEADS must select valid attention "
            f"heads in [0, {total_heads}); got {heads}."
        )

    capture_path = Path(raw_path).expanduser()
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    head_index = torch.tensor(
        heads,
        device=q_permuted.device,
        dtype=torch.long,
    )
    payload = {
        "format": "wan22_n8_sparse_attention_replay_v1",
        "random_tensors": False,
        "model": getattr(attn, "_svg_model_name", None),
        "generation_step": step,
        "layer": layer,
        "cfg_branch": branch,
        "heads": heads,
        "total_heads": total_heads,
        "tokens": int(q_permuted.shape[2]),
        "head_dim": int(q_permuted.shape[3]),
        "q_clusters": int(q_sizes.shape[-1]),
        "k_clusters": int(k_sizes.shape[-1]),
        "flashinfer_timings": dict(flashinfer_timings),
        "q_sorted": q_permuted.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "k_sorted": k_permuted.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "v_sorted": v_permuted.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "selected_mask": dynamic_map.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "q_sizes": q_sizes.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "k_sizes": k_sizes.index_select(1, head_index)
        .detach()
        .cpu()
        .contiguous(),
        "flashinfer_output_sorted": output_permuted.index_select(
            1, head_index
        )
        .detach()
        .cpu()
        .contiguous(),
    }
    torch.save(payload, capture_path)
    _N8_ATTENTION_REPLAY_CAPTURED = True
    print(
        "[WAN-N8-ATTENTION-CAPTURE] "
        f"wrote {capture_path} heads={heads} step={step} "
        f"layer={layer} cfg={branch}",
        flush=True,
    )


def _profile_index_selected(
    raw: str,
    value: int,
    *,
    total: int | None = None,
) -> bool:
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if not tokens or tokens <= {"none", "off", "disabled"}:
        return False
    if "all" in tokens:
        return True
    selected: set[int] = set()
    for token in tokens:
        if token == "last":
            if total is None:
                raise RuntimeError("'last' requires a known profile dimension.")
            selected.add(total - 1)
            continue
        try:
            parsed = int(token)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid sampled timing selector {raw!r}; expected integers, "
                "'last', 'all', or 'none'."
            ) from error
        selected.add(total + parsed if parsed < 0 and total is not None else parsed)
    return int(value) in selected


def _timing_profile_selected(
    attn: Any,
    *,
    generation_step: int | None = None,
    cfg_branch: int | None = None,
) -> bool:
    """Select a few production calls for synchronized CUDA-event timing."""
    if _CONFIG.logging_file is None or not _env_enabled(
        "SPARSEPR_WAN_N8_TIMING_PROFILE"
    ):
        return False
    step = int(
        generation_step
        if generation_step is not None
        else getattr(attn, "_svg_generation_step", -1)
    )
    branch = int(
        cfg_branch
        if cfg_branch is not None
        else getattr(attn, "_svg_cfg_branch", -1)
    )
    layer = int(getattr(attn, "_svg_layer_idx", -1))
    total_layers = int(getattr(attn, "_svg_total_layers", 0))
    return (
        step >= int(_CONFIG.first_sparse_forward)
        and layer >= int(_CONFIG.first_layers_fp)
        and _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_STEPS", ""),
            step,
        )
        and _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_LAYERS", ""),
            layer,
            total=total_layers,
        )
        and _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_BRANCHES", "0,1"),
            branch,
        )
    )


def _model_timing_profile_selected(
    model: Any,
    *,
    generation_step: int,
    cfg_branch: int,
) -> bool:
    if _CONFIG.logging_file is None or not _env_enabled(
        "SPARSEPR_WAN_N8_TIMING_PROFILE"
    ):
        return False
    return (
        generation_step >= int(_CONFIG.first_sparse_forward)
        and _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_STEPS", ""),
            generation_step,
        )
        and _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_BRANCHES", "0,1"),
            cfg_branch,
        )
        and bool(getattr(model, "blocks", None))
    )


def _record_cuda_timing_event(enabled: bool):
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _elapsed_cuda_ms(start, end) -> float:
    if start is None or end is None:
        raise RuntimeError("CUDA timing event pair is incomplete.")
    return float(start.elapsed_time(end))


def reset_wan22_timing_profile_state() -> None:
    """Reset cross-model denoising-step timing before generating one video."""
    _DENOISING_PROFILE_STATE.clear()


def profile_wan22_denoising_step_boundary(generation_step: int) -> None:
    """Close the preceding denoising step and start the next wall-time interval."""
    if _CONFIG.logging_file is None or not _env_enabled(
        "SPARSEPR_WAN_N8_TIMING_PROFILE"
    ):
        return
    now = time.perf_counter()
    previous_step = _DENOISING_PROFILE_STATE.get("generation_step")
    previous_started = _DENOISING_PROFILE_STATE.get("started")
    if previous_step is not None and previous_started is not None:
        previous_step = int(previous_step)
        if _profile_index_selected(
            os.environ.get("SPARSEPR_WAN_N8_TIMING_STEPS", ""),
            previous_step,
        ):
            wall_ms = (now - float(previous_started)) * 1000.0
            branch_times = dict(
                _DENOISING_PROFILE_STATE.get("model_ms", {}).get(
                    previous_step, {}
                )
            )
            model_ms = sum(float(value) for value in branch_times.values())
            _log_row(
                {
                    "record_type": "wan_denoising_step_timing",
                    "timing_profiled_step": True,
                    "generation_step": previous_step,
                    "wan_denoising_step_wall_ms": wall_ms,
                    "wan_cfg_offload_scheduler_ms": max(
                        0.0, wall_ms - model_ms
                    ),
                    "wan_profiled_model_total_ms": model_ms,
                    "profiled_model_branches": sorted(
                        int(branch) for branch in branch_times
                    ),
                }
            )
    _DENOISING_PROFILE_STATE["generation_step"] = int(generation_step)
    _DENOISING_PROFILE_STATE["started"] = now
    _DENOISING_PROFILE_STATE.setdefault("model_ms", {})


def _diagnostic_steps() -> frozenset[int]:
    raw = os.environ.get("SPARSEPR_WAN_N8_DIAGNOSTIC_STEPS", "").strip()
    if not raw:
        return frozenset()
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if set(values) <= {"none", "off", "disabled"}:
        return frozenset()
    if "all" in values:
        return frozenset()
    try:
        return frozenset(int(value) for value in values)
    except ValueError as error:
        raise RuntimeError(
            "SPARSEPR_WAN_N8_DIAGNOSTIC_STEPS must be 'all' or comma-separated integers, "
            f"got {raw!r}."
        ) from error


def _diagnostic_step_selected(step: int) -> bool:
    raw = os.environ.get("SPARSEPR_WAN_N8_DIAGNOSTIC_STEPS", "").strip()
    if not raw:
        return False
    values = {value.strip().lower() for value in raw.split(",") if value.strip()}
    if values <= {"none", "off", "disabled"}:
        return False
    return "all" in values or int(step) in _diagnostic_steps()


def _diagnostics_requested() -> bool:
    raw = os.environ.get("SPARSEPR_WAN_N8_DIAGNOSTIC_STEPS", "").strip()
    if not raw:
        return False
    values = {value.strip().lower() for value in raw.split(",") if value.strip()}
    return not values <= {"none", "off", "disabled"}


def _trace_every_step() -> bool:
    return _env_enabled("SPARSEPR_WAN_N8_TRACE_EVERY_STEP")


def _sparse_phase_started(step: int) -> bool:
    return int(step) >= int(_CONFIG.first_sparse_forward)


def _fail_fast() -> bool:
    return _env_enabled("SPARSEPR_WAN_N8_FAIL_FAST")


def _diagnostic_stages() -> frozenset[str]:
    raw = os.environ.get(
        "SPARSEPR_WAN_N8_DIAGNOSTIC_STAGES",
        "clustering,svg_ear_selection,probe_fit",
    ).strip()
    return frozenset(value.strip() for value in raw.split(",") if value.strip())


def _diagnostic_layer_selected(attn: Any) -> bool:
    layer = int(getattr(attn, "_svg_layer_idx", -1))
    total_layers = int(getattr(attn, "_svg_total_layers", 0))
    raw = os.environ.get("SPARSEPR_WAN_N8_DIAGNOSTIC_LAYERS", "").strip()
    if not raw:
        return total_layers > 0 and layer == total_layers - 1
    selected: set[int] = set()
    for value in raw.split(","):
        token = value.strip().lower()
        if not token:
            continue
        if token == "last":
            selected.add(total_layers - 1)
            continue
        try:
            parsed = int(token)
        except ValueError as error:
            raise RuntimeError(
                "SPARSEPR_WAN_N8_DIAGNOSTIC_LAYERS must contain comma-separated "
                f"integers or 'last', got {raw!r}."
            ) from error
        selected.add(total_layers + parsed if parsed < 0 else parsed)
    return layer in selected


def _tensor_failure_stats(tensor: torch.Tensor) -> dict[str, object]:
    """Synchronize compact tensor statistics only while reporting a failure."""
    detached = tensor.detach()
    total = int(detached.numel())
    if torch.is_floating_point(detached):
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        finite_abs_max = (
            float(
                torch.where(
                    finite,
                    detached.abs(),
                    torch.zeros_like(detached),
                ).amax().item()
            )
            if total
            else 0.0
        )
        nan = int(torch.isnan(detached).sum().item())
        posinf = int(torch.isposinf(detached).sum().item())
        neginf = int(torch.isneginf(detached).sum().item())
    else:
        finite_count = total
        finite_abs_max = (
            float(detached.to(torch.float32).abs().amax().item()) if total else 0.0
        )
        nan = posinf = neginf = 0
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite_count,
        "total": total,
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "finite_abs_max": finite_abs_max,
    }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = _first_tensor(item)
            if result is not None:
                return result
    return None


def _emit_step_diagnostic(
    event: str,
    attn: Any,
    *,
    tensors: dict[str, torch.Tensor],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": event,
        "model": str(getattr(attn, "_svg_model_name", "unknown_model")),
        "layer": int(getattr(attn, "_svg_layer_idx", -1)),
        "generation_step": int(getattr(attn, "_svg_generation_step", -1)),
        "attention_call": int(getattr(attn, "_svg_call_index", 0)) - 1,
        "cfg_branch": int(getattr(attn, "_svg_cfg_branch", -1)),
        "tensors": {
            name: _tensor_failure_stats(tensor)
            for name, tensor in tensors.items()
        },
    }
    print("WAN_N8_STEP_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def _fail_fast_tensors(
    event: str,
    attn: Any,
    *,
    tensors: dict[str, torch.Tensor],
) -> None:
    if not _fail_fast():
        return
    failures = [
        name
        for name, tensor in tensors.items()
        if torch.is_floating_point(tensor)
        and not bool(torch.isfinite(tensor.detach()).all())
    ]
    if not failures:
        return
    payload = _emit_step_diagnostic(event, attn, tensors=tensors)
    raise RuntimeError(
        "WAN N8 fail-fast detected the first non-finite tensors "
        f"at {event}: {failures}\n" + json.dumps(payload, indent=2)
    )


def _n8_stage_diagnostic_callback(attn: Any):
    verbose = _diagnostic_step_selected(
        int(getattr(attn, "_svg_generation_step", -1))
    ) and _diagnostic_layer_selected(attn)
    fail_fast = _fail_fast()
    if not verbose and not fail_fast:
        return None
    enabled_stages = _diagnostic_stages()

    def emit(
        stage: str,
        tensors: dict[str, torch.Tensor],
        scalars: dict[str, object],
    ) -> None:
        if stage not in enabled_stages:
            return
        failures: list[str] = []
        if fail_fast:
            for name, tensor in tensors.items():
                if not torch.is_floating_point(tensor):
                    continue
                detached = tensor.detach()
                if name == "svg_ear_scores":
                    invalid = (
                        torch.isnan(detached).any()
                        or torch.isposinf(detached).any()
                        or not torch.isfinite(detached).any()
                    )
                else:
                    invalid = not torch.isfinite(detached).all()
                if bool(invalid):
                    failures.append(name)
        recovery_event = any(
            key.endswith("_recovered_heads") and int(value) > 0
            for key, value in scalars.items()
        )
        if not verbose and not failures and not recovery_event:
            return
        payload: dict[str, object] = {
            "event": stage,
            "model": str(getattr(attn, "_svg_model_name", "unknown_model")),
            "layer": int(getattr(attn, "_svg_layer_idx", -1)),
            "generation_step": int(getattr(attn, "_svg_generation_step", -1)),
            "attention_call": int(getattr(attn, "_svg_call_index", 0)) - 1,
            "cfg_branch": int(getattr(attn, "_svg_cfg_branch", -1)),
            "fail_fast_nonfinite_tensors": failures,
            **scalars,
            "tensors": {
                name: _tensor_failure_stats(tensor)
                for name, tensor in tensors.items()
            },
        }
        print(
            "WAN_N8_STAGE_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        if failures:
            raise RuntimeError(
                "WAN N8 fail-fast detected the first non-finite stage tensors: "
                f"stage={stage}, phase={scalars.get('phase')}, tensors={failures}"
            )

    return emit


def _install_model_step_diagnostic(model: Any, model_name: str | None) -> None:
    diagnostics_requested = _diagnostics_requested()
    if (
        not diagnostics_requested
        and not _trace_every_step()
        and not _fail_fast()
    ) or hasattr(model, "_svg_n8_diagnostic_hook"):
        return

    def model_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        blocks = list(getattr(module, "blocks", []))
        if not blocks or not hasattr(blocks[0], "self_attn"):
            return
        attn = blocks[0].self_attn
        step = int(getattr(attn, "_svg_generation_step", -1))
        selected = _diagnostic_step_selected(step)
        sparse_phase = _sparse_phase_started(step)
        trace_sparse_step = _trace_every_step() and sparse_phase
        fail_fast_sparse_step = _fail_fast() and sparse_phase
        if not selected and not trace_sparse_step and not fail_fast_sparse_step:
            return
        tensors: dict[str, torch.Tensor] = {}
        model_input = _first_tensor(inputs[0]) if inputs else None
        model_output = _first_tensor(output)
        if model_input is not None:
            tensors["model_input"] = model_input
        if model_output is not None:
            tensors["model_output"] = model_output
        payload = _emit_step_diagnostic(
            "model_trajectory",
            attn,
            tensors=tensors,
        )
        nonfinite_input = (
            model_input is not None and not torch.isfinite(model_input).all()
        )
        nonfinite_output = (
            model_output is not None and not torch.isfinite(model_output).all()
        )
        if (selected or fail_fast_sparse_step) and (
            nonfinite_input or nonfinite_output
        ):
            raise RuntimeError(
                "WAN model trajectory became non-finite at a configured diagnostic "
                "step:\n" + json.dumps(payload, indent=2)
            )

    model._svg_model_name = model_name
    model._svg_n8_diagnostic_hook = model.register_forward_hook(model_hook)


def _install_model_timing_hooks(model: Any, model_name: str | None) -> None:
    """Install sampled CUDA-event timers without replacing Wan block math."""
    model._svg_model_name = model_name
    model._svg_profile_call_index = 0
    model._svg_profile_active = False
    if _CONFIG.logging_file is None or not _env_enabled(
        "SPARSEPR_WAN_N8_TIMING_PROFILE"
    ):
        return
    if hasattr(model, "_svg_n8_timing_hooks"):
        return

    blocks = list(getattr(model, "blocks", []))
    if not blocks:
        return

    def model_pre_hook(module: Any, _inputs: tuple[Any, ...]) -> None:
        call_index = int(getattr(module, "_svg_profile_call_index", 0))
        branch = call_index % 2
        module._svg_profile_call_index = call_index + 1
        step = int(getattr(module, "_svg_generation_step", -1))
        active = _model_timing_profile_selected(
            module,
            generation_step=step,
            cfg_branch=branch,
        )
        module._svg_profile_active = active
        module._svg_profile_step = step
        module._svg_profile_branch = branch
        module._svg_profile_model_start = _record_cuda_timing_event(active)
        module._svg_profile_first_block_start = None
        module._svg_profile_last_block_end = None

    def model_post_hook(
        module: Any,
        _inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if not bool(getattr(module, "_svg_profile_active", False)):
            return
        model_end = _record_cuda_timing_event(True)
        model_end.synchronize()
        model_start = getattr(module, "_svg_profile_model_start", None)
        first_block_start = getattr(
            module, "_svg_profile_first_block_start", None
        )
        last_block_end = getattr(module, "_svg_profile_last_block_end", None)
        if (
            model_start is None
            or first_block_start is None
            or last_block_end is None
        ):
            raise RuntimeError(
                "WAN model timing hooks did not observe the first and last blocks."
            )
        embedding_ms = _elapsed_cuda_ms(model_start, first_block_start)
        head_ms = _elapsed_cuda_ms(last_block_end, model_end)
        total_ms = _elapsed_cuda_ms(model_start, model_end)
        step = int(module._svg_profile_step)
        branch = int(module._svg_profile_branch)
        _DENOISING_PROFILE_STATE.setdefault("model_ms", {}).setdefault(
            step, {}
        )[branch] = total_ms
        _log_row(
            {
                "record_type": "wan_model_timing",
                "timing_profiled_model": True,
                "model_name": getattr(module, "_svg_model_name", None),
                "generation_step": step,
                "cfg_branch": branch,
                "wan_model_embedding_ms": embedding_ms,
                "wan_model_head_ms": head_ms,
                "wan_model_embedding_and_head_ms": embedding_ms + head_ms,
                "wan_model_total_ms": total_ms,
            }
        )
        module._svg_profile_active = False

    handles = [
        model.register_forward_pre_hook(model_pre_hook),
        model.register_forward_hook(model_post_hook),
    ]

    for layer_idx, block in enumerate(blocks):
        block._svg_profile_layer_idx = layer_idx

        def block_pre_hook(
            block_module: Any,
            _inputs: tuple[Any, ...],
            *,
            owning_model: Any = model,
            index: int = layer_idx,
        ) -> None:
            if not bool(getattr(owning_model, "_svg_profile_active", False)):
                block_module._svg_profile_active = False
                return
            if index == 0:
                owning_model._svg_profile_first_block_start = (
                    _record_cuda_timing_event(True)
                )
            attn = block_module.self_attn
            selected = _timing_profile_selected(
                attn,
                generation_step=int(owning_model._svg_profile_step),
                cfg_branch=int(owning_model._svg_profile_branch),
            )
            block_module._svg_profile_active = selected
            block_module._svg_profile_start = _record_cuda_timing_event(
                selected
            )
            block_module._svg_profile_cross_start = None
            block_module._svg_profile_cross_end = None
            block_module._svg_profile_ffn_start = None
            block_module._svg_profile_ffn_end = None

        def block_post_hook(
            block_module: Any,
            _inputs: tuple[Any, ...],
            _output: Any,
            *,
            owning_model: Any = model,
            index: int = layer_idx,
            total_layers: int = len(blocks),
        ) -> None:
            block_end = None
            if bool(getattr(owning_model, "_svg_profile_active", False)) and (
                index == total_layers - 1
                or bool(getattr(block_module, "_svg_profile_active", False))
            ):
                block_end = _record_cuda_timing_event(True)
            if (
                bool(getattr(owning_model, "_svg_profile_active", False))
                and index == total_layers - 1
            ):
                owning_model._svg_profile_last_block_end = block_end
            if not bool(getattr(block_module, "_svg_profile_active", False)):
                return
            if block_end is None:
                raise RuntimeError("WAN sampled block timing has no end event.")
            block_end.synchronize()
            attn = block_module.self_attn
            step = int(owning_model._svg_profile_step)
            branch = int(owning_model._svg_profile_branch)
            attention_key = getattr(
                attn, "_svg_last_profiled_attention_key", None
            )
            if attention_key != (step, branch):
                raise RuntimeError(
                    "WAN block timer could not match its self-attention timing."
                )
            self_attention_ms = float(
                attn._svg_last_profiled_attention_ms
            )
            cross_ms = _elapsed_cuda_ms(
                block_module._svg_profile_cross_start,
                block_module._svg_profile_cross_end,
            )
            ffn_ms = _elapsed_cuda_ms(
                block_module._svg_profile_ffn_start,
                block_module._svg_profile_ffn_end,
            )
            block_ms = _elapsed_cuda_ms(
                block_module._svg_profile_start, block_end
            )
            norm_residual_ms = max(
                0.0,
                block_ms - self_attention_ms - cross_ms - ffn_ms,
            )
            _log_row(
                {
                    "record_type": "wan_block_timing",
                    "timing_profiled_block": True,
                    "model_name": getattr(
                        owning_model, "_svg_model_name", None
                    ),
                    "generation_step": step,
                    "layer": index,
                    "cfg_branch": branch,
                    "wan_self_attention_ms": self_attention_ms,
                    "wan_cross_attention_ms": cross_ms,
                    "wan_ffn_ms": ffn_ms,
                    "wan_norm_modulation_residual_ms": norm_residual_ms,
                    "wan_block_total_ms": block_ms,
                }
            )
            block_module._svg_profile_active = False

        def cross_pre_hook(
            _module: Any,
            _inputs: tuple[Any, ...],
            *,
            block_module: Any = block,
        ) -> None:
            block_module._svg_profile_cross_start = (
                _record_cuda_timing_event(
                    bool(getattr(block_module, "_svg_profile_active", False))
                )
            )

        def cross_post_hook(
            _module: Any,
            _inputs: tuple[Any, ...],
            _output: Any,
            *,
            block_module: Any = block,
        ) -> None:
            block_module._svg_profile_cross_end = _record_cuda_timing_event(
                bool(getattr(block_module, "_svg_profile_active", False))
            )

        def ffn_pre_hook(
            _module: Any,
            _inputs: tuple[Any, ...],
            *,
            block_module: Any = block,
        ) -> None:
            block_module._svg_profile_ffn_start = _record_cuda_timing_event(
                bool(getattr(block_module, "_svg_profile_active", False))
            )

        def ffn_post_hook(
            _module: Any,
            _inputs: tuple[Any, ...],
            _output: Any,
            *,
            block_module: Any = block,
        ) -> None:
            block_module._svg_profile_ffn_end = _record_cuda_timing_event(
                bool(getattr(block_module, "_svg_profile_active", False))
            )

        handles.extend(
            (
                block.register_forward_pre_hook(block_pre_hook),
                block.register_forward_hook(block_post_hook),
                block.cross_attn.register_forward_pre_hook(cross_pre_hook),
                block.cross_attn.register_forward_hook(cross_post_hook),
                block.ffn.register_forward_pre_hook(ffn_pre_hook),
                block.ffn.register_forward_hook(ffn_post_hook),
            )
        )
    model._svg_n8_timing_hooks = handles


def _raise_n8_attention_failure(
    error: Exception,
    attn: Any,
    *,
    generation_step: int,
    call_index: int,
    tensors: dict[str, torch.Tensor],
) -> None:
    context = {
        "model": str(getattr(attn, "_svg_model_name", "unknown_model")),
        "layer": int(getattr(attn, "_svg_layer_idx", -1)),
        "generation_step": int(generation_step),
        "attention_call": int(call_index),
        "tensors": {
            name: _tensor_failure_stats(tensor)
            for name, tensor in tensors.items()
        },
    }
    raise RuntimeError(
        "WAN N8 attention failed with tensor diagnostics:\n"
        + json.dumps(context, indent=2)
    ) from error


def _wan_block_layer_norm_modulate(
    x: torch.Tensor,
    *,
    epsilon: float,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    from sparsepr.kernels.triton.wan_block_fusion import (
        wan_layer_norm_modulate,
    )

    return wan_layer_norm_modulate(
        x.contiguous(),
        epsilon=epsilon,
        scale=scale,
        shift=shift,
    )


def _wan_block_gate_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    from sparsepr.kernels.triton.wan_block_fusion import wan_gate_residual

    return wan_gate_residual(
        residual.contiguous(),
        update.contiguous(),
        gate,
    )


def _patched_attention_block_forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    """Native WanAttentionBlock with only promoted vector stages replaced."""
    if e.dtype != torch.float32:
        raise RuntimeError(
            f"WAN block fusion requires FP32 modulation, got {e.dtype}."
        )
    with torch.amp.autocast("cuda", dtype=torch.float32):
        modulation = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    if modulation[0].dtype != torch.float32:
        raise RuntimeError(
            "WAN block fusion expected FP32 modulation components."
        )

    self_input = _wan_block_layer_norm_modulate(
        x,
        epsilon=float(self.norm1.eps),
        scale=modulation[1].squeeze(2),
        shift=modulation[0].squeeze(2),
    )
    self_update = self.self_attn(
        self_input,
        seq_lens,
        grid_sizes,
        freqs,
    )
    x = _wan_block_gate_residual(
        x,
        self_update,
        modulation[2].squeeze(2),
    )

    # Preserve native cross-attention math; this stage was not part of the
    # promoted block-fusion speed gate.
    x = x + self.cross_attn(
        self.norm3(x),
        context,
        context_lens,
    )
    ffn_input = _wan_block_layer_norm_modulate(
        x,
        epsilon=float(self.norm2.eps),
        scale=modulation[4].squeeze(2),
        shift=modulation[3].squeeze(2),
    )
    ffn_update = self.ffn(ffn_input)
    return _wan_block_gate_residual(
        x,
        ffn_update,
        modulation[5].squeeze(2),
    )


def install_wan22_ti2v_sparse_patch(config: Wan22TI2VSparseConfig) -> None:
    """Install sparse attention hooks for native Wan2.2 video models.

    Keep ``WanModel.__init__`` untouched: Diffusers inspects that signature while
    loading ``config.json`` in ``from_pretrained``. Wrapping it with a generic
    ``*args, **kwargs`` function makes Diffusers ignore model dimensions and load
    the checkpoint into the default 2B-ish shape.
    """

    if config.pattern not in {"SAP", "SAP_REUSED_LOW_RANK", "N8_custom_v6"}:
        raise ValueError(f"unsupported Wan2.2 TI2V sparse pattern: {config.pattern}")
    if config.basis_refresh_every <= 0:
        raise ValueError(f"basis_refresh_every must be positive, got {config.basis_refresh_every}.")
    if config.lowrank_accum_dtype not in {"bf16", "fp32"}:
        raise ValueError(f"lowrank_accum_dtype must be 'bf16' or 'fp32', got {config.lowrank_accum_dtype!r}.")
    if config.lowrank_cache_device not in {"cuda", "cpu", "none", "cuda_factors", "cpu_factors"}:
        raise ValueError(
            "lowrank_cache_device must be 'cuda', 'cpu', 'none', 'cuda_factors', or 'cpu_factors', "
            f"got {config.lowrank_cache_device!r}."
        )
    _parse_layer_ranges(config.lowrank_layer_ranges)
    if config.pattern == "N8_custom_v6":
        if not 0.0 < config.n8_target_density <= 1.0:
            raise ValueError("n8_target_density must be in (0,1].")
        if config.n8_selector_policy not in {
            "svg_ear_value",
            "attention_mass",
        }:
            raise ValueError(
                "n8_selector_policy must be svg_ear_value or attention_mass."
            )
        if config.n8_probe_rows <= 0 or config.n8_repair_rank <= 0:
            raise ValueError("N8 probes and repair rank must be positive.")
        if config.n8_flashinfer_backend not in {"auto", "fa2", "fa3"}:
            raise ValueError(
                "n8_flashinfer_backend must be auto, fa2, or fa3; got "
                f"{config.n8_flashinfer_backend!r}."
            )
    if config.dense_attention_backend not in {"fa2", "fa3"}:
        raise ValueError(
            "dense_attention_backend must be fa2 or fa3; got "
            f"{config.dense_attention_backend!r}."
        )
    if config.cfg_layer0_reuse and (
        config.pattern != "N8_custom_v6" or config.first_layers_fp < 1
    ):
        raise ValueError(
            "cfg_layer0_reuse requires N8_custom_v6 with permanently dense "
            "layer 0 (first_layers_fp >= 1)."
        )

    global _CONFIG
    _CONFIG = config
    if config.logging_file is not None:
        os.makedirs(os.path.dirname(config.logging_file), exist_ok=True)
        with open(config.logging_file, "w"):
            pass

    from wan.modules import model as wan_model
    from wan.modules import attention as wan_attention

    if not _ORIGINALS:
        _ORIGINALS["WanSelfAttention.forward"] = wan_model.WanSelfAttention.forward
        _ORIGINALS["WanAttentionBlock.forward"] = (
            wan_model.WanAttentionBlock.forward
        )
        _ORIGINALS["wan_attention.FLASH_ATTN_3_AVAILABLE"] = getattr(
            wan_attention,
            "FLASH_ATTN_3_AVAILABLE",
            False,
        )

    if config.dense_attention_backend == "fa3":
        if not getattr(wan_attention, "FLASH_ATTN_3_AVAILABLE", False):
            raise RuntimeError(
                "FA3 was requested for Wan dense attention/dense warmup, but "
                "the official Wan attention module did not detect FA3."
            )
    elif getattr(wan_attention, "FLASH_ATTN_2_AVAILABLE", False):
        wan_attention.FLASH_ATTN_3_AVAILABLE = False

    if config.pattern == "N8_custom_v6":
        from sparsepr.kmeans_utils import (
            set_flashinfer_backend,
            set_flashinfer_workspace_cache_enabled,
        )

        set_flashinfer_backend(config.n8_flashinfer_backend)
        set_flashinfer_workspace_cache_enabled(True)

    wan_model.WanSelfAttention.forward = _patched_self_attention_forward
    wan_model.WanAttentionBlock.forward = (
        _patched_attention_block_forward
        if config.block_fusion
        else _ORIGINALS["WanAttentionBlock.forward"]
    )


def restore_wan22_ti2v_sparse_patch() -> None:
    if not _ORIGINALS:
        return
    from wan.modules import model as wan_model

    wan_model.WanSelfAttention.forward = _ORIGINALS["WanSelfAttention.forward"]
    wan_model.WanAttentionBlock.forward = _ORIGINALS[
        "WanAttentionBlock.forward"
    ]
    try:
        from wan.modules import attention as wan_attention

        wan_attention.FLASH_ATTN_3_AVAILABLE = _ORIGINALS.get(
            "wan_attention.FLASH_ATTN_3_AVAILABLE",
            getattr(wan_attention, "FLASH_ATTN_3_AVAILABLE", False),
        )
    except Exception:
        pass


def reset_wan22_ti2v_sparse_state(model: Any, model_name: str | None = None) -> None:
    """Clear semantic-clustering and residual caches on an instantiated Wan DiT."""

    blocks = list(getattr(model, "blocks", []))
    for layer_idx, block in enumerate(blocks):
        if hasattr(block, "self_attn"):
            block.self_attn._svg_layer_idx = layer_idx
            block.self_attn._svg_total_layers = len(blocks)
            block.self_attn._svg_model_name = model_name
            _reset_attention_state(block.self_attn)
    _install_model_step_diagnostic(model, model_name)
    _install_model_timing_hooks(model, model_name)


def _reset_attention_state(attn: Any) -> None:
    attn._svg_call_index = 0
    attn._svg_centroids_init = False
    attn._svg_q_centroids = None
    attn._svg_k_centroids = None
    attn._svg_correction_cache = None
    attn._svg_density_cache = None
    attn._svg_sparse_call_index = 0
    attn._svg_generation_step = 0
    attn._svg_cfg_branch = 0
    attn._svg_last_profiled_attention_key = None
    attn._svg_last_profiled_attention_ms = None
    attn._n8_core = None
    attn._n8_cores = {}
    attn._svg_cfg_layer0_cache = None
    attn._svg_cfg_layer0_cache_stats = {
        "stores": 0,
        "hits": 0,
        "misses": 0,
    }


def _cfg_layer0_cache_key(
    attn: Any,
    *,
    generation_step: int,
    tensor: torch.Tensor,
) -> tuple[Any, ...]:
    """Identify the exact layer-0 result shared by Wan's two CFG passes."""
    return (
        str(getattr(attn, "_svg_model_name", "unknown_model")),
        int(generation_step),
        tuple(int(value) for value in tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _cfg_layer0_cache_stats(attn: Any) -> dict[str, int]:
    stats = getattr(attn, "_svg_cfg_layer0_cache_stats", None)
    if not isinstance(stats, dict):
        stats = {"stores": 0, "hits": 0, "misses": 0}
        attn._svg_cfg_layer0_cache_stats = stats
    return stats


def _tensor_version_or_none(tensor: torch.Tensor) -> int | None:
    """Read the mutation counter when available outside inference mode."""
    try:
        return int(tensor._version)
    except RuntimeError:
        # PyTorch inference tensors intentionally do not expose a counter.
        return None


def _store_cfg_layer0_result(
    attn: Any,
    *,
    generation_step: int,
    input_tensor: torch.Tensor,
    projected: torch.Tensor,
) -> None:
    """Retain branch-0's exact dense layer-0 self-attention projection."""
    attn._svg_cfg_layer0_cache = {
        "key": _cfg_layer0_cache_key(
            attn,
            generation_step=generation_step,
            tensor=input_tensor,
        ),
        "projected": projected,
        "version": _tensor_version_or_none(projected),
    }
    _cfg_layer0_cache_stats(attn)["stores"] += 1


def _take_cfg_layer0_result(
    attn: Any,
    *,
    generation_step: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Consume a matching branch-0 result, otherwise take the exact fallback."""
    cache = getattr(attn, "_svg_cfg_layer0_cache", None)
    attn._svg_cfg_layer0_cache = None
    stats = _cfg_layer0_cache_stats(attn)
    expected_key = _cfg_layer0_cache_key(
        attn,
        generation_step=generation_step,
        tensor=input_tensor,
    )
    if not isinstance(cache, dict) or cache.get("key") != expected_key:
        stats["misses"] += 1
        return None
    projected = cache.get("projected")
    if (
        not torch.is_tensor(projected)
        or _tensor_version_or_none(projected) != cache.get("version")
    ):
        stats["misses"] += 1
        return None
    stats["hits"] += 1
    return projected


def _make_n8_core():
    """Create the portable N8 core with Wan-specific immutable settings."""
    from sparsepr.models.common import N8V6Config, N8V6Core

    kmeans_backend = "flash"
    if _env_enabled("SPARSEPR_N8_CUDA_EXT_ROLE_CLUSTER"):
        from sparsepr.kernels.n8_extension import n8_cuda_extension_available

        if n8_cuda_extension_available():
            kmeans_backend = "n8_cuda_ext"
    return N8V6Core(
        N8V6Config(
            num_q_centroids=int(_CONFIG.num_q_centroids),
            num_k_centroids=int(_CONFIG.num_k_centroids),
            target_density=float(_CONFIG.n8_target_density),
            role_q_rank=int(_CONFIG.n8_role_q_rank),
            role_k_rank=int(_CONFIG.n8_role_k_rank),
            role_v_rank=int(_CONFIG.n8_role_v_rank),
            role_v_weight=float(_CONFIG.n8_role_v_weight),
            metric_sample_tokens=int(_CONFIG.n8_metric_sample_tokens),
            kmeans_iter_init=int(_CONFIG.kmeans_iter_init),
            kmeans_iter_step=int(_CONFIG.kmeans_iter_step),
            kmeans_backend=kmeans_backend,
            factor_power_iters=int(_CONFIG.n8_factor_power_iters),
            factor_refresh_every=int(_CONFIG.n8_factor_refresh_every),
            probe_rows=int(_CONFIG.n8_probe_rows),
            repair_rank=int(_CONFIG.n8_repair_rank),
            selector_policy=str(_CONFIG.n8_selector_policy),
            compile_role_kv_projection=bool(
                _CONFIG.n8_compile_role_kv_projection
            ),
            role_kv_matmul_precision=str(_CONFIG.n8_role_kv_matmul_precision),
        )
    )


def _n8_core_for_cfg_branch(attn: Any):
    """Return the branch-local N8 state for Wan's cond/uncond CFG passes."""
    branch = int(getattr(attn, "_svg_cfg_branch", 0))
    cores = getattr(attn, "_n8_cores", None)
    if not isinstance(cores, dict):
        cores = {}
        attn._n8_cores = cores
    core = cores.get(branch)
    if core is None:
        core = _make_n8_core()
        cores[branch] = core
    # Preserve the historical attribute as a read-only alias to the core used
    # by the current call. External gates inspect it, but state ownership lives
    # in _n8_cores.
    attn._n8_core = core
    return core


def _expanded_wan_rope_frequencies(
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tokens: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache native Wan's 3-axis complex frequency expansion as FP32 planes."""
    if grid_sizes.shape[0] != 1:
        raise RuntimeError("Fused Wan N8 QKV currently requires batch size one.")
    f, h, w = (int(value) for value in grid_sizes[0].tolist())
    sequence = f * h * w
    if sequence > tokens:
        raise RuntimeError(f"Wan RoPE grid has {sequence} positions for {tokens} tokens.")
    key = (
        str(freqs.device), int(freqs.data_ptr()), tokens, head_dim, f, h, w,
    )
    cached = _WAN_ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    half = head_dim // 2
    split = [half - 2 * (half // 3), half // 3, half // 3]
    time_freq, height_freq, width_freq = freqs.split(split, dim=1)
    expanded = torch.cat(
        (
            time_freq[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            height_freq[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            width_freq[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ),
        dim=-1,
    ).reshape(sequence, half)
    if sequence < tokens:
        expanded = torch.cat(
            (
                expanded,
                torch.ones(
                    (tokens - sequence, half),
                    device=expanded.device,
                    dtype=expanded.dtype,
                ),
            ),
            dim=0,
        )
    result = (
        expanded.real.to(torch.float32).contiguous(),
        expanded.imag.to(torch.float32).contiguous(),
    )
    _WAN_ROPE_CACHE[key] = result
    return result


def _wan_qkv_cuda_extension(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sparsepr.kernels.n8_extension import wan_qkv_norm_rope_layout

    real, imag = _expanded_wan_rope_frequencies(
        grid_sizes,
        freqs,
        tokens=int(q.shape[1]),
        head_dim=int(self.head_dim),
    )
    return wan_qkv_norm_rope_layout(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        self.norm_q.weight.contiguous(),
        self.norm_k.weight.contiguous(),
        real,
        imag,
        heads=int(self.num_heads),
        epsilon=float(self.norm_q.eps),
    )


def _sync_for_timing(tensor: torch.Tensor | None = None) -> None:
    if tensor is not None and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _dense_attention(
    attn: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens,
    window_size,
    *,
    apply_rope: bool = True,
):
    from wan.modules.model import rope_apply
    from wan.modules.attention import flash_attention

    if apply_rope:
        q = rope_apply(q, attn._svg_grid_sizes, attn._svg_freqs)
        k = rope_apply(k, attn._svg_grid_sizes, attn._svg_freqs)
    return flash_attention(
        q=q,
        k=k,
        v=v,
        k_lens=seq_lens,
        window_size=window_size,
    )


def _patched_self_attention_forward(self, x, seq_lens, grid_sizes, freqs):
    from wan.modules.model import rope_apply

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    layer_idx = int(getattr(self, "_svg_layer_idx", 0))
    call_index = int(getattr(self, "_svg_call_index", 0))
    generation_step = int(getattr(self, "_svg_generation_step", call_index))
    self._svg_call_index = call_index + 1
    # Native Wan invokes conditional then unconditional model forwards at each
    # sampling step. Their online cluster/factor histories must remain isolated.
    self._svg_cfg_branch = call_index % 2
    self._svg_grid_sizes = grid_sizes
    self._svg_freqs = freqs

    force_dense = layer_idx < _CONFIG.first_layers_fp or generation_step < _CONFIG.first_sparse_forward
    reuse_cfg_layer0 = (
        _CONFIG.pattern == "N8_custom_v6"
        and bool(_CONFIG.cfg_layer0_reuse)
        and layer_idx == 0
        and force_dense
    )
    if reuse_cfg_layer0:
        if self._svg_cfg_branch == 0:
            # A new conditional pass invalidates any unconsumed stale entry.
            self._svg_cfg_layer0_cache = None
        else:
            cached = _take_cfg_layer0_result(
                self,
                generation_step=generation_step,
                input_tensor=x,
            )
            if cached is not None:
                _flop_audit_call(
                    getattr(self, "_svg_model_name", None),
                    "dense_reused",
                )
                return cached
    timing_profile = (
        not force_dense
        and _timing_profile_selected(
            self,
            generation_step=generation_step,
            cfg_branch=self._svg_cfg_branch,
        )
    )
    attention_start_event = _record_cuda_timing_event(timing_profile)
    q_projected = self.q(x)
    k_projected = self.k(x)
    v_projected = self.v(x)
    qkv_linear_end_event = _record_cuda_timing_event(timing_profile)

    use_wan_cuda_ext = (
        _CONFIG.pattern == "N8_custom_v6"
        and not force_dense
        and _env_enabled("SPARSEPR_N8_CUDA_EXT_WAN_QKV")
    )
    if use_wan_cuda_ext:
        global _N8_CUDA_EXT_WAN_WARNED
        try:
            q_bhld, k_bhld, v_bhld = _wan_qkv_cuda_extension(
                self, q_projected, k_projected, v_projected, grid_sizes, freqs
            )
        except Exception as exc:
            if _env_enabled("SPARSEPR_N8_CUDA_EXT_STRICT"):
                raise RuntimeError(
                    "Strict WAN N8 CUDA QKV extension execution failed."
                ) from exc
            if not _N8_CUDA_EXT_WAN_WARNED:
                import warnings

                warnings.warn(
                    f"Wan N8 CUDA QKV extension unavailable; using PyTorch: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _N8_CUDA_EXT_WAN_WARNED = True
            use_wan_cuda_ext = False
    qkv_postprocess_end_event = _record_cuda_timing_event(
        timing_profile and use_wan_cuda_ext
    )

    if use_wan_cuda_ext:
        try:
            sparse_bhld, density, timings = _sparse_n8_attention_bhld(
                self, q_bhld, k_bhld, v_bhld, seq_lens
            )
        except Exception as exc:
            _raise_n8_attention_failure(
                exc,
                self,
                generation_step=generation_step,
                call_index=call_index,
                tensors={
                    "attention_input": x,
                    "q_projected": q_projected,
                    "k_projected": k_projected,
                    "v_projected": v_projected,
                    "q_fused_bhld": q_bhld,
                    "k_fused_bhld": k_bhld,
                    "v_fused_bhld": v_bhld,
                },
            )
        timings = {**timings, "wan_qkv_cuda_extension": True}
        sparse_out = sparse_bhld.transpose(1, 2).contiguous()
    else:
        q = self.norm_q(q_projected).view(b, s, n, d)
        k = self.norm_k(k_projected).view(b, s, n, d)
        v = v_projected.view(b, s, n, d)
    if force_dense:
        if _WAN_N8_FLOP_AUDIT.get("enabled"):
            _flop_audit_call(
                getattr(self, "_svg_model_name", None),
                "dense",
            )
            _WAN_N8_FLOP_AUDIT["dense_token_pairs"] += (
                int(b) * int(n) * int(s) * int(s)
            )
            _WAN_N8_FLOP_AUDIT["sequence_lengths"].add(int(s))
            _WAN_N8_FLOP_AUDIT["heads"].add(int(n))
            _WAN_N8_FLOP_AUDIT["head_dims"].add(int(d))
        out = _dense_attention(self, q, k, v, seq_lens, self.window_size)
        projected = self.o(out.flatten(2))
        if reuse_cfg_layer0 and self._svg_cfg_branch == 0:
            _store_cfg_layer0_result(
                self,
                generation_step=generation_step,
                input_tensor=x,
                projected=projected,
            )
        return projected

    if not use_wan_cuda_ext:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        qkv_postprocess_end_event = _record_cuda_timing_event(timing_profile)
        if _CONFIG.pattern == "N8_custom_v6":
            try:
                sparse_out, density, timings = _sparse_n8_attention(
                    self, q, k, v, seq_lens
                )
            except Exception as exc:
                _raise_n8_attention_failure(
                    exc,
                    self,
                    generation_step=generation_step,
                    call_index=call_index,
                    tensors={
                        "attention_input": x,
                        "q_projected": q_projected,
                        "k_projected": k_projected,
                        "v_projected": v_projected,
                        "q_norm_rope": q,
                        "k_norm_rope": k,
                        "v_prepared": v,
                    },
                )
        else:
            sparse_out, density, timings = _sparse_sap_attention(self, q, k, v, seq_lens)

    if _CONFIG.pattern == "SAP_REUSED_LOW_RANK" and _lowrank_enabled_for_layer(layer_idx):
        sparse_out, timings = _apply_reused_lowrank(self, q, k, v, seq_lens, sparse_out, density, timings)
    elif _CONFIG.pattern == "SAP_REUSED_LOW_RANK":
        timings = {
            **timings,
            "avg_density": density,
            "cache_refresh": False,
            "uses_dense_teacher": False,
            "lowrank_rank": int(_CONFIG.lowrank_rank),
            "basis_refresh_every": int(_CONFIG.basis_refresh_every),
            "lowrank_fit_tokens": int(_CONFIG.lowrank_fit_tokens),
            "lowrank_cache_device": _CONFIG.lowrank_cache_device,
            "lowrank_layer_ranges": _CONFIG.lowrank_layer_ranges,
            "lowrank_skipped": True,
            "lowrank_skip_reason": "layer_not_selected",
            "dense_teacher_ms": 0.0,
            "lowrank_svd_ms": 0.0,
            "cache_reuse_ms": 0.0,
            "correction_shape": None,
        }

    output_projection_start_event = _record_cuda_timing_event(timing_profile)
    projected = self.o(sparse_out.flatten(2))
    output_projection_end_event = _record_cuda_timing_event(timing_profile)
    if timing_profile:
        output_projection_end_event.synchronize()
        profiled_attention_ms = _elapsed_cuda_ms(
            attention_start_event, output_projection_end_event
        )
        timings = {
            **timings,
            "wan_qkv_linear_ms": _elapsed_cuda_ms(
                attention_start_event, qkv_linear_end_event
            ),
            "wan_qkv_postprocess_ms": _elapsed_cuda_ms(
                qkv_linear_end_event, qkv_postprocess_end_event
            ),
            "wan_output_projection_ms": _elapsed_cuda_ms(
                output_projection_start_event, output_projection_end_event
            ),
            "wan_profiled_attention_total_ms": profiled_attention_ms,
        }
        self._svg_last_profiled_attention_key = (
            generation_step,
            int(getattr(self, "_svg_cfg_branch", -1)),
        )
        self._svg_last_profiled_attention_ms = profiled_attention_ms
    _log_row(
        {
            "record_type": "wan_attention_timing",
            "mode": _CONFIG.pattern,
            "model_name": getattr(self, "_svg_model_name", None),
            "layer": layer_idx,
            "generation_step": generation_step,
            "forward_call_index": call_index,
            "cfg_branch": int(getattr(self, "_svg_cfg_branch", -1)),
            "sparse_call_index": int(getattr(self, "_svg_sparse_call_index", 0)),
            "avg_density": density,
            "seq_len": int(s),
            "num_heads": int(n),
            "head_dim": int(d),
            **timings,
        }
    )
    self._svg_sparse_call_index = int(getattr(self, "_svg_sparse_call_index", 0)) + 1
    _fail_fast_tensors(
        "attention_output_projection",
        self,
        tensors={
            "sparse_attention_output": sparse_out,
            "attention_output_projection": projected,
        },
    )
    if (
        _diagnostic_step_selected(generation_step)
        and _diagnostic_layer_selected(self)
    ):
        _emit_step_diagnostic(
            "last_attention_projection",
            self,
            tensors={
                "sparse_attention_output": sparse_out,
                "attention_output_projection": projected,
            },
        )
    return projected


def _sparse_n8_attention(
    self,
    q_blhd: torch.Tensor,
    k_blhd: torch.Tensor,
    v_blhd: torch.Tensor,
    seq_lens,
):
    """Run native Wan self-attention through the model-independent N8 core."""
    source_dtype = v_blhd.dtype
    kernel_dtype = source_dtype if source_dtype in {torch.float16, torch.bfloat16} else torch.bfloat16
    q_bhld = q_blhd.transpose(1, 2).to(kernel_dtype).contiguous()
    k_bhld = k_blhd.transpose(1, 2).to(kernel_dtype).contiguous()
    v_bhld = v_blhd.transpose(1, 2).to(kernel_dtype).contiguous()
    repaired, density, timings = _sparse_n8_attention_bhld(
        self, q_bhld, k_bhld, v_bhld, seq_lens
    )
    return repaired.transpose(1, 2).to(source_dtype).contiguous(), density, timings


def _sparse_n8_attention_bhld(
    self,
    q_bhld: torch.Tensor,
    k_bhld: torch.Tensor,
    v_bhld: torch.Tensor,
    seq_lens,
):
    """N8 core for already-prepared native Wan BHLD tensors."""
    del seq_lens  # Native benchmark uses batch one with a fully valid latent sequence.
    from sparsepr.kernels.triton.permute import (
        apply_inverse_permutation_triton,
        permute_tensor_by_labels_triton,
    )
    from sparsepr.kmeans_utils import (
        dynamic_block_sparse_fwd_flashinfer,
        pop_last_flashinfer_timings,
        set_flashinfer_kernel_profile_enabled,
    )

    kernel_dtype = (
        v_bhld.dtype
        if v_bhld.dtype in {torch.float16, torch.bfloat16}
        else torch.bfloat16
    )
    q_bhld = q_bhld.to(kernel_dtype).contiguous()
    k_bhld = k_bhld.to(kernel_dtype).contiguous()
    v_bhld = v_bhld.to(kernel_dtype).contiguous()
    batch, heads, tokens, dim = q_bhld.shape
    if batch != 1:
        raise RuntimeError("Wan2.2 N8 currently requires batch size one.")

    _fail_fast_tensors(
        "n8_attention_input",
        self,
        tensors={"q": q_bhld, "k": k_bhld, "v": v_bhld},
    )
    core = _n8_core_for_cfg_branch(self)
    core.diagnostic_callback = _n8_stage_diagnostic_callback(self)
    timing_profile = _timing_profile_selected(self)
    route_detail_profile = timing_profile and _env_enabled(
        "SPARSEPR_WAN_N8_ROUTE_PROFILE"
    )
    route_start_event = _record_cuda_timing_event(timing_profile)
    core.profile_breakdown_active = route_detail_profile
    try:
        route = core.route(
            q_bhld, k_bhld, v_bhld, total_key_tokens=tokens
        )
    finally:
        core.profile_breakdown_active = False
    route_detail: dict[str, Any] = {}
    if route_detail_profile:
        route_detail = dict(core.last_route_profile)
        route_detail.update(
            {
                "n8_route_role_projection_ms": float(
                    route_detail.get("n8_route_kv_projection_init_ms", 0.0)
                )
                + float(
                    route_detail.get("n8_route_q_projection_init_ms", 0.0)
                ),
                "n8_route_factorization_ms": float(
                    route_detail.get("n8_route_kv_metric_factor_ms", 0.0)
                )
                + float(
                    route_detail.get("n8_route_q_metric_factor_ms", 0.0)
                ),
                "n8_route_kmeans_ms": float(
                    route_detail.get("n8_route_kmeans_k_ms", 0.0)
                )
                + float(route_detail.get("n8_route_kmeans_q_ms", 0.0)),
                "n8_route_selector_scoring_ms": float(
                    route_detail.get("n8_route_selector_scores_ms", 0.0)
                ),
                # Compatibility alias for existing profile summarizers. This
                # records generic selector scoring when attention_mass is used.
                "n8_route_svg_ear_scoring_ms": float(
                    route_detail.get("n8_route_selector_scores_ms", 0.0)
                ),
                "n8_route_selection_ms": float(
                    route_detail.get("n8_route_selector_budget_ms", 0.0)
                ),
            }
        )
    route_end_event = _record_cuda_timing_event(timing_profile)

    q_order = torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous()
    q_permuted, _ = permute_tensor_by_labels_triton(
        q_bhld, None, dim=2, sorted_indices=q_order
    )
    dynamic_map = route.video_dynamic_map.view(
        batch, heads, core.config.num_q_centroids, core.config.num_k_centroids
    )
    q_sizes = route.q_cluster_sizes.view(batch, heads, core.config.num_q_centroids)
    k_sizes = route.k_cluster_sizes.view(batch, heads, core.config.num_k_centroids)
    if _WAN_N8_FLOP_AUDIT.get("enabled"):
        # Count exact executed Q/K token pairs. Keep the aggregate on GPU so
        # the audit does not introduce a synchronization on every attention
        # call. The temporary is small compared with WAN Q/K/V at this shape.
        selected_key_tokens = torch.where(
            dynamic_map,
            k_sizes.to(torch.int64).unsqueeze(2),
            0,
        ).sum(dim=3)
        selected_pairs = (
            selected_key_tokens * q_sizes.to(torch.int64)
        ).sum()
        running = _WAN_N8_FLOP_AUDIT["selected_token_pairs_cuda"]
        _WAN_N8_FLOP_AUDIT["selected_token_pairs_cuda"] = (
            selected_pairs if running is None else running + selected_pairs
        )
        _flop_audit_call(
            getattr(self, "_svg_model_name", None),
            "sparse",
        )
        _WAN_N8_FLOP_AUDIT["probe_token_pairs"] += (
            int(batch)
            * int(heads)
            * int(core.config.probe_rows)
            * int(tokens)
        )
        _WAN_N8_FLOP_AUDIT["q_kmeans_iterations"] += int(
            route.q_kmeans_iters
        )
        _WAN_N8_FLOP_AUDIT["k_kmeans_iterations"] += int(
            route.k_kmeans_iters
        )
        _WAN_N8_FLOP_AUDIT["sequence_lengths"].add(int(tokens))
        _WAN_N8_FLOP_AUDIT["heads"].add(int(heads))
        _WAN_N8_FLOP_AUDIT["head_dims"].add(int(dim))
    q_permute_end_event = _record_cuda_timing_event(timing_profile)
    flashinfer_timings: dict[str, Any] = {}
    if timing_profile:
        set_flashinfer_kernel_profile_enabled(True)
    try:
        output_permuted = dynamic_block_sparse_fwd_flashinfer(
            q_permuted,
            route.k_permuted,
            route.v_permuted,
            dynamic_map,
            q_sizes,
            k_sizes,
            is_cpu=False,
        )
        if timing_profile:
            flashinfer_timings = pop_last_flashinfer_timings()
    finally:
        if timing_profile:
            set_flashinfer_kernel_profile_enabled(False)
    sparse_attention_end_event = _record_cuda_timing_event(timing_profile)
    _maybe_capture_n8_attention_replay(
        self,
        q_permuted=q_permuted,
        k_permuted=route.k_permuted,
        v_permuted=route.v_permuted,
        dynamic_map=dynamic_map,
        q_sizes=q_sizes,
        k_sizes=k_sizes,
        output_permuted=output_permuted,
        flashinfer_timings=flashinfer_timings,
    )
    base = apply_inverse_permutation_triton(output_permuted, q_order, dim=2)
    inverse_permute_end_event = _record_cuda_timing_event(timing_profile)
    _fail_fast_tensors(
        "block_sparse_attention_output",
        self,
        tensors={"sparse_base": base},
    )
    repaired, repair_info = core.repair(base, q_bhld, k_bhld, v_bhld, route)
    repair_end_event = _record_cuda_timing_event(timing_profile)
    _fail_fast_tensors(
        "probe_repair_output",
        self,
        tensors={"sparse_base": base, "repaired": repaired},
    )
    if (
        _diagnostic_step_selected(
            int(getattr(self, "_svg_generation_step", -1))
        )
        and _diagnostic_layer_selected(self)
    ):
        _emit_step_diagnostic(
            "last_attention_repair",
            self,
            tensors={
                "q": q_bhld,
                "k": k_bhld,
                "v": v_bhld,
                "sparse_base": base,
                "repaired": repaired,
            },
        )

    density = float(core.config.target_density)
    timings: dict[str, Any] = {
        "avg_density": density,
        "kernel_dtype": str(kernel_dtype),
        "n8_target_density": density,
        "n8_selector_policy": str(core.config.selector_policy),
        "n8_cfg_branch": int(getattr(self, "_svg_cfg_branch", -1)),
        "n8_cfg_core_route_call": int(core.route_call_index),
        "n8_base_video_density": float(route.base_video_density),
        "q_kmeans_iter": int(route.q_kmeans_iters),
        "k_kmeans_iter": int(route.k_kmeans_iters),
        "uses_dense_teacher": False,
        "tail_compensation": False,
        **repair_info,
    }
    if timing_profile:
        repair_end_event.synchronize()
        timings.update(
            {
                "timing_profiled": True,
                "n8_route_ms": _elapsed_cuda_ms(
                    route_start_event, route_end_event
                ),
                "n8_q_sort_permute_ms": _elapsed_cuda_ms(
                    route_end_event, q_permute_end_event
                ),
                "n8_sparse_attention_ms": _elapsed_cuda_ms(
                    q_permute_end_event, sparse_attention_end_event
                ),
                "n8_inverse_permute_ms": _elapsed_cuda_ms(
                    sparse_attention_end_event, inverse_permute_end_event
                ),
                "n8_probe_repair_ms": _elapsed_cuda_ms(
                    inverse_permute_end_event, repair_end_event
                ),
                "n8_route_attention_repair_ms": _elapsed_cuda_ms(
                    route_start_event, repair_end_event
                ),
                **route_detail,
                **flashinfer_timings,
            }
        )
    else:
        timings["timing_profiled"] = False
    return repaired, density, timings


def _kmeans_clustering(self, query_bhld: torch.Tensor, key_bhld: torch.Tensor):
    from sparsepr.kmeans_utils import batch_kmeans_Euclid

    batch, heads, seq_len, dim = query_bhld.shape
    q_flat = query_bhld.reshape(batch * heads, seq_len, dim)
    k_flat = key_bhld.reshape(batch * heads, seq_len, dim)
    if not getattr(self, "_svg_centroids_init", False):
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            q_flat,
            n_clusters=_CONFIG.num_q_centroids,
            max_iters=_CONFIG.kmeans_iter_init,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            k_flat,
            n_clusters=_CONFIG.num_k_centroids,
            max_iters=_CONFIG.kmeans_iter_init,
        )
        self._svg_centroids_init = True
    else:
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            q_flat,
            n_clusters=_CONFIG.num_q_centroids,
            max_iters=_CONFIG.kmeans_iter_step,
            init_centroids=self._svg_q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            k_flat,
            n_clusters=_CONFIG.num_k_centroids,
            max_iters=_CONFIG.kmeans_iter_step,
            init_centroids=self._svg_k_centroids,
        )
    self._svg_q_centroids = qcentroids
    self._svg_k_centroids = kcentroids
    return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter


def _sparse_sap_attention(self, q_blhd: torch.Tensor, k_blhd: torch.Tensor, v_blhd: torch.Tensor, seq_lens):
    from sparsepr.kmeans_utils import density_calculation, dynamic_block_sparse_fwd_flashinfer, identify_dynamic_map
    from sparsepr.kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton

    sparse_started = time.perf_counter()
    q_bhld = q_blhd.transpose(1, 2).contiguous()
    k_bhld = k_blhd.transpose(1, 2).contiguous()
    v_bhld = v_blhd.transpose(1, 2).contiguous()
    batch, heads, seq_len, dim = q_bhld.shape
    assert batch == 1, "Wan2.2 TI2V SVG2 sparse path currently assumes batch size 1."

    _sync_for_timing(q_bhld)
    kmeans_started = time.perf_counter()
    qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = _kmeans_clustering(
        self, q_bhld, k_bhld
    )
    _sync_for_timing(q_bhld)
    kmeans_ms = (time.perf_counter() - kmeans_started) * 1000.0

    q_cluster_sizes = qcluster_sizes.view(batch, heads, _CONFIG.num_q_centroids)
    k_cluster_sizes = kcluster_sizes.view(batch, heads, _CONFIG.num_k_centroids)
    dynamic_map = identify_dynamic_map(
        qcentroids.view(batch, heads, _CONFIG.num_q_centroids, dim),
        kcentroids.view(batch, heads, _CONFIG.num_k_centroids, dim),
        q_cluster_sizes,
        k_cluster_sizes,
        _CONFIG.top_p_kmeans,
        _CONFIG.min_kc_ratio,
    )

    kernel_dtype = v_bhld.dtype if v_bhld.dtype in {torch.float16, torch.bfloat16} else torch.bfloat16
    q_kernel = q_bhld.to(kernel_dtype)
    k_kernel = k_bhld.to(kernel_dtype)
    v_kernel = v_bhld.to(kernel_dtype)

    q_perm, q_sorted_indices = permute_tensor_by_labels_triton(q_kernel, qlabels, dim=2)
    k_perm, k_sorted_indices = permute_tensor_by_labels_triton(k_kernel, klabels, dim=2)
    v_perm, _ = permute_tensor_by_labels_triton(v_kernel, klabels, dim=2, sorted_indices=k_sorted_indices)
    q_perm = q_perm.to(kernel_dtype).contiguous()
    k_perm = k_perm.to(kernel_dtype).contiguous()
    v_perm = v_perm.to(kernel_dtype).contiguous()
    if q_perm.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(f"FlashInfer sparse path requires half/bfloat16 Q, got {q_perm.dtype}.")

    del q_kernel, k_kernel, v_kernel
    del q_bhld, k_bhld, v_bhld
    del qlabels, klabels, qcentroids, kcentroids, k_sorted_indices

    _sync_for_timing(q_perm)
    kernel_started = time.perf_counter()
    output_permuted = dynamic_block_sparse_fwd_flashinfer(
        q_perm, k_perm, v_perm, dynamic_map, q_cluster_sizes, k_cluster_sizes, is_cpu=False
    )
    del q_perm, k_perm, v_perm
    sparse_bhld = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)
    del output_permuted, q_sorted_indices
    _sync_for_timing(sparse_bhld)
    sparse_kernel_ms = (time.perf_counter() - kernel_started) * 1000.0

    density_started = time.perf_counter()
    density = float(density_calculation(dynamic_map, q_cluster_sizes, k_cluster_sizes).mean().item())
    _sync_for_timing(sparse_bhld)
    density_ms = (time.perf_counter() - density_started) * 1000.0
    sparse_ms = (time.perf_counter() - sparse_started) * 1000.0

    timings = {
        "kmeans_ms": kmeans_ms,
        "q_kmeans_iter": int(qiter) if not isinstance(qiter, torch.Tensor) else int(qiter.item()),
        "k_kmeans_iter": int(kiter) if not isinstance(kiter, torch.Tensor) else int(kiter.item()),
        "sparse_kernel_ms": sparse_kernel_ms,
        "density_ms": density_ms,
        "sparse_sap_ms": sparse_ms,
        "kernel_dtype": str(kernel_dtype),
    }
    return sparse_bhld.transpose(1, 2).contiguous(), density, timings


def _apply_reused_lowrank(self, q, k, v, seq_lens, sparse_out, density, timings):
    sparse_index = int(getattr(self, "_svg_sparse_call_index", 0))
    cache_refresh = (
        _CONFIG.lowrank_cache_device == "none"
        or getattr(self, "_svg_correction_cache", None) is None
        or sparse_index % _CONFIG.basis_refresh_every == 0
    )
    dense_teacher_ms = 0.0
    lowrank_svd_ms = 0.0
    cache_reuse_ms = 0.0

    correction_cache = None
    if cache_refresh:
        _sync_for_timing(q)
        dense_started = time.perf_counter()
        dense_out = _dense_attention(self, q, k, v, seq_lens, self.window_size, apply_rope=False)
        _sync_for_timing(q)
        dense_teacher_ms = (time.perf_counter() - dense_started) * 1000.0

        lowrank_started = time.perf_counter()
        if _CONFIG.lowrank_cache_device in {"cuda_factors", "cpu_factors"}:
            coeff, basis = _optimal_rank_residual_factors(
                dense_out,
                sparse_out,
                _CONFIG.lowrank_rank,
                fit_tokens=_CONFIG.lowrank_fit_tokens,
                sample_seed=int(getattr(self, "_svg_layer_idx", 0)) * 10007 + sparse_index * 31,
            )
            out = _add_factor_correction_inplace(sparse_out, coeff, basis, _CONFIG.lowrank_accum_dtype)
            if _CONFIG.lowrank_cache_device == "cuda_factors":
                correction_cache = {
                    "kind": "factors",
                    "coeff": coeff.detach().to(dtype=sparse_out.dtype).contiguous(),
                    "basis": basis.detach().to(dtype=sparse_out.dtype).contiguous(),
                }
            else:
                correction_cache = {
                    "kind": "factors",
                    "coeff": coeff.detach().to(device="cpu", dtype=sparse_out.dtype).contiguous(),
                    "basis": basis.detach().to(device="cpu", dtype=sparse_out.dtype).contiguous(),
                }
            del coeff, basis
        else:
            correction = _optimal_rank_residual_correction(
                dense_out,
                sparse_out,
                _CONFIG.lowrank_rank,
                fit_tokens=_CONFIG.lowrank_fit_tokens,
                sample_seed=int(getattr(self, "_svg_layer_idx", 0)) * 10007 + sparse_index * 31,
            )
            if _CONFIG.lowrank_accum_dtype == "fp32":
                out = sparse_out.float() + correction.float()
            else:
                out = sparse_out + correction.to(dtype=sparse_out.dtype)
            if _CONFIG.lowrank_cache_device == "cuda":
                correction_cache = correction.detach().to(dtype=sparse_out.dtype).contiguous()
            elif _CONFIG.lowrank_cache_device == "cpu":
                correction_cache = correction.detach().to(device="cpu", dtype=sparse_out.dtype).contiguous()
            else:
                correction_cache = None
            del correction
        del dense_out
        _sync_for_timing(q)
        lowrank_svd_ms = (time.perf_counter() - lowrank_started) * 1000.0
        self._svg_correction_cache = correction_cache
        self._svg_density_cache = density
    else:
        _sync_for_timing(q)
        cache_started = time.perf_counter()
        correction_cache = self._svg_correction_cache
        if correction_cache is None:
            raise RuntimeError("Wan2.2 TI2V low-rank cache unexpectedly missing.")
        if isinstance(correction_cache, dict):
            coeff = correction_cache["coeff"]
            basis = correction_cache["basis"]
            if coeff.device != sparse_out.device:
                coeff = coeff.to(device=sparse_out.device, dtype=sparse_out.dtype, non_blocking=True)
            if basis.device != sparse_out.device:
                basis = basis.to(device=sparse_out.device, dtype=sparse_out.dtype, non_blocking=True)
            out = _add_factor_correction_inplace(sparse_out, coeff, basis, _CONFIG.lowrank_accum_dtype)
        else:
            correction = correction_cache
            if correction.shape != sparse_out.shape:
                raise RuntimeError(
                    "Cached Wan2.2 TI2V low-rank correction shape changed: "
                    f"cached={tuple(correction.shape)}, current={tuple(sparse_out.shape)}."
                )
            if correction.device != sparse_out.device:
                correction = correction.to(device=sparse_out.device, dtype=sparse_out.dtype, non_blocking=True)
            if _CONFIG.lowrank_accum_dtype == "fp32":
                out = sparse_out.float() + correction.float()
            else:
                out = sparse_out + correction.to(dtype=sparse_out.dtype)
        _sync_for_timing(q)
        cache_reuse_ms = (time.perf_counter() - cache_started) * 1000.0
        density = self._svg_density_cache if self._svg_density_cache is not None else density

    timings = {
        **timings,
        "avg_density": density,
        "cache_refresh": cache_refresh,
        "uses_dense_teacher": cache_refresh,
        "lowrank_rank": int(_CONFIG.lowrank_rank),
        "basis_refresh_every": int(_CONFIG.basis_refresh_every),
        "lowrank_fit_tokens": int(_CONFIG.lowrank_fit_tokens),
        "lowrank_cache_device": _CONFIG.lowrank_cache_device,
        "lowrank_layer_ranges": _CONFIG.lowrank_layer_ranges,
        "lowrank_skipped": False,
        "dense_teacher_ms": dense_teacher_ms,
        "lowrank_svd_ms": lowrank_svd_ms,
        "cache_reuse_ms": cache_reuse_ms,
        "correction_shape": _lowrank_cache_shape(correction_cache),
    }
    return out.to(dtype=sparse_out.dtype), timings


def _parse_layer_ranges(spec: str | None) -> tuple[tuple[int, int], ...] | None:
    if spec is None:
        return None
    normalized = str(spec).strip().lower()
    if normalized in {"", "all", "*"}:
        return None
    ranges: list[tuple[int, int]] = []
    for part in normalized.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(item)
        if start < 0 or end < 0 or end < start:
            raise ValueError(f"Invalid low-rank layer range {item!r} in {spec!r}.")
        ranges.append((start, end))
    if not ranges:
        return None
    return tuple(ranges)


def _lowrank_enabled_for_layer(layer_idx: int) -> bool:
    ranges = _parse_layer_ranges(_CONFIG.lowrank_layer_ranges)
    if ranges is None:
        return True
    return any(start <= layer_idx <= end for start, end in ranges)


def _gram_eigh_lowrank(
    residual: torch.Tensor,
    rank: int,
    fit_tokens: int = 0,
    sample_seed: int = 0,
    output_dtype: torch.dtype | None = None,
    chunk_tokens: int = 8192,
) -> torch.Tensor:
    batch_heads, tokens, dim = residual.shape
    r = min(rank, dim)
    fit_residual = residual
    if fit_tokens > 0 and fit_tokens < tokens:
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed) + tokens * 37 + dim * 101)
        sample_idx = torch.randperm(tokens, generator=generator)[: max(r, fit_tokens)].to(residual.device)
        fit_residual = residual[:, sample_idx, :].contiguous()
    out_dtype = output_dtype or residual.dtype
    correction = torch.empty_like(residual, dtype=out_dtype)
    with torch.amp.autocast(device_type=residual.device.type, enabled=False):
        fit_residual = fit_residual.float()
        gram = torch.bmm(fit_residual.transpose(1, 2), fit_residual).float()
        _, eigvecs = torch.linalg.eigh(gram)
        basis = eigvecs[:, :, -r:].float()
        basis_t = basis.transpose(1, 2).contiguous()
        for start in range(0, tokens, chunk_tokens):
            end = min(start + chunk_tokens, tokens)
            residual_chunk = residual[:, start:end, :].float()
            correction[:, start:end, :] = torch.bmm(torch.bmm(residual_chunk, basis), basis_t).to(out_dtype)
    return correction


def _optimal_rank_residual_factors(
    dense_out: torch.Tensor,
    sparse_out: torch.Tensor,
    rank: int,
    fit_tokens: int = 0,
    sample_seed: int = 0,
    output_dtype: torch.dtype | None = None,
    chunk_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, dim = dense_out.shape
    r = min(rank, dim)
    out_dtype = output_dtype or sparse_out.dtype
    fit_count = tokens
    sample_idx = None
    if fit_tokens > 0 and fit_tokens < tokens:
        fit_count = max(r, fit_tokens)
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed) + tokens * 37 + dim * 101)
        sample_idx = torch.randperm(tokens, generator=generator)[:fit_count].to(dense_out.device)

    with torch.amp.autocast(device_type=dense_out.device.type, enabled=False):
        if sample_idx is None:
            dense_fit = dense_out.permute(0, 2, 1, 3).reshape(batch * heads, tokens, dim).float()
            sparse_fit = sparse_out.permute(0, 2, 1, 3).reshape(batch * heads, tokens, dim).float()
        else:
            dense_fit = dense_out[:, sample_idx, :, :].permute(0, 2, 1, 3).reshape(batch * heads, fit_count, dim).float()
            sparse_fit = sparse_out[:, sample_idx, :, :].permute(0, 2, 1, 3).reshape(batch * heads, fit_count, dim).float()
        fit_residual = dense_fit - sparse_fit
        gram = torch.bmm(fit_residual.transpose(1, 2), fit_residual).float()
        _, eigvecs = torch.linalg.eigh(gram)
        basis_fp32 = eigvecs[:, :, -r:].float()
        del dense_fit, sparse_fit, fit_residual, gram, eigvecs

        coeff = torch.empty((batch * heads, tokens, r), device=dense_out.device, dtype=out_dtype)
        dense_bhld = dense_out.permute(0, 2, 1, 3)
        sparse_bhld = sparse_out.permute(0, 2, 1, 3)
        for start in range(0, tokens, chunk_tokens):
            end = min(start + chunk_tokens, tokens)
            residual_chunk = (
                dense_bhld[:, :, start:end, :] - sparse_bhld[:, :, start:end, :]
            ).reshape(batch * heads, end - start, dim).float()
            coeff[:, start:end, :] = torch.bmm(residual_chunk, basis_fp32).to(out_dtype)
        basis = basis_fp32.to(out_dtype).contiguous()
    return coeff.contiguous(), basis


def _add_factor_correction_inplace(
    sparse_out: torch.Tensor,
    coeff: torch.Tensor,
    basis: torch.Tensor,
    accum_dtype: str,
    chunk_tokens: int = 8192,
) -> torch.Tensor:
    batch, tokens, heads, dim = sparse_out.shape
    out = sparse_out.contiguous()
    out_bhld = out.permute(0, 2, 1, 3)
    basis_t = basis.transpose(1, 2).contiguous()
    with torch.amp.autocast(device_type=out.device.type, enabled=False):
        for start in range(0, tokens, chunk_tokens):
            end = min(start + chunk_tokens, tokens)
            correction = torch.bmm(
                coeff[:, start:end, :].float(),
                basis_t.float(),
            ).view(batch, heads, end - start, dim)
            target = out_bhld[:, :, start:end, :]
            if accum_dtype == "fp32":
                target.copy_((target.float() + correction).to(dtype=out.dtype))
            else:
                target.add_(correction.to(dtype=out.dtype))
    return out


def _lowrank_cache_shape(cache: Any) -> Any:
    if cache is None:
        return None
    if isinstance(cache, dict):
        return {
            "kind": cache.get("kind"),
            "coeff": list(cache["coeff"].shape),
            "basis": list(cache["basis"].shape),
        }
    return list(cache.shape)


def _optimal_rank_residual_correction(
    dense_out: torch.Tensor,
    sparse_out: torch.Tensor,
    rank: int,
    fit_tokens: int = 0,
    sample_seed: int = 0,
) -> torch.Tensor:
    batch, tokens, heads, dim = dense_out.shape
    residual = (dense_out - sparse_out).permute(0, 2, 1, 3).reshape(batch * heads, tokens, dim).contiguous()
    correction = _gram_eigh_lowrank(
        residual,
        rank,
        fit_tokens=fit_tokens,
        sample_seed=sample_seed,
        output_dtype=sparse_out.dtype,
    )
    return correction.view(batch, heads, tokens, dim).permute(0, 2, 1, 3).contiguous()


def _log_row(row: dict[str, Any]) -> None:
    if _CONFIG.logging_file is None:
        return
    with open(_CONFIG.logging_file, "a") as f:
        f.write(json.dumps(row) + "\n")
