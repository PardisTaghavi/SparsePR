"""SparsePR adapter for Cosmos3-Nano-16B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sparsepr.kernels.triton.permute import (
    apply_inverse_permutation_triton,
    permute_tensor_by_labels_triton,
)
from sparsepr.kmeans_utils import (
    dynamic_block_sparse_fwd_flashinfer,
    set_flashinfer_backend,
    set_flashinfer_workspace_cache_enabled,
)
from sparsepr.models.common import N8V6Config, N8V6Core


@dataclass(frozen=True)
class Cosmos3Config:
    """Frozen SparsePR settings for Cosmos3-Nano-16B."""

    guidance_scale: float = 7.0
    num_inference_steps: int = 35
    dense_first_layers: int = 1
    dense_first_steps: int = 2
    dense_last_steps: int = 0
    gqa_key_sharing: bool = True
    flashinfer_backend: str = "auto"
    logging_file: str | None = None
    n8: N8V6Config = N8V6Config()


def apply_cosmos3_rotary(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply Cosmos3's real-valued pairwise rotary embedding."""
    original_dtype = tensor.dtype
    tensor_f = tensor.float().reshape(*tensor.shape[:-1], -1, 2)
    rotated = torch.stack((-tensor_f[..., 1], tensor_f[..., 0]), dim=-1)
    rotated = rotated.flatten(-2)
    return (tensor.float() * cos + rotated * sin).to(original_dtype)


def _expand_gqa(
    tensor: torch.Tensor, query_heads: int, key_value_heads: int
) -> torch.Tensor:
    if query_heads == key_value_heads:
        return tensor
    group_size = query_heads // key_value_heads
    return tensor.repeat_interleave(group_size, dim=1)


def _nhd_to_bhnd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()


def _bhnd_to_nhd_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(1, 2).squeeze(0).flatten(-2, -1).contiguous()


def is_cosmos3_attention(module: torch.nn.Module) -> bool:
    required = (
        "processor",
        "set_processor",
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "norm_q",
        "norm_k",
        "norm_added_q",
        "norm_added_k",
        "to_out",
        "to_add_out",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    return all(hasattr(module, name) for name in required)


class _Runtime:
    def __init__(self, config: Cosmos3Config, module_count: int):
        self.config = config
        self.module_count = module_count
        self.calls = 0
        if config.logging_file:
            path = Path(config.logging_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def next_call(self, layer: int) -> dict[str, int]:
        self.calls += 1
        forward = (self.calls - 1) // self.module_count
        branches = 2 if self.config.guidance_scale > 1.0 else 1
        return {
            "layer": layer,
            "transformer_forward": forward,
            "step": forward // branches,
            "cfg_branch": forward % branches,
        }

    def use_dense(self, layer: int, step: int) -> bool:
        if layer < self.config.dense_first_layers:
            return True
        if step < self.config.dense_first_steps:
            return True
        return self.config.dense_last_steps > 0 and step >= (
            self.config.num_inference_steps - self.config.dense_last_steps
        )

    def log(self, row: dict[str, Any]) -> None:
        if self.config.logging_file:
            with Path(self.config.logging_file).open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class Cosmos3N8Processor:
    """Cosmos3 QKV/GQA/layout adapter around the shared SparsePR core."""

    def __init__(
        self,
        *,
        layer_idx: int,
        runtime: _Runtime,
        n8_config: N8V6Config,
        query_heads: int,
        key_value_heads: int,
        gqa_key_sharing: bool,
    ):
        if query_heads % key_value_heads:
            raise ValueError(
                f"Cosmos3 query heads ({query_heads}) must be divisible by "
                f"KV heads ({key_value_heads})."
            )
        self.layer_idx = layer_idx
        self.runtime = runtime
        self.n8_config = n8_config
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.gqa_group_size = (
            query_heads // key_value_heads if gqa_key_sharing else 1
        )
        self.cores: dict[int, N8V6Core] = {}

    def _core(self, cfg_branch: int) -> N8V6Core:
        core = self.cores.get(cfg_branch)
        if core is None:
            core = N8V6Core(self.n8_config)
            self.cores[cfg_branch] = core
        return core

    def _project_qkv(
        self,
        attn: torch.nn.Module,
        understanding: torch.Tensor,
        generation: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        heads = int(attn.num_attention_heads)
        kv_heads = int(attn.num_key_value_heads)
        head_dim = int(attn.head_dim)
        q_und = attn.to_q(understanding).view(-1, heads, head_dim)
        k_und = attn.to_k(understanding).view(-1, kv_heads, head_dim)
        v_und = attn.to_v(understanding).view(-1, kv_heads, head_dim)
        q_gen = attn.add_q_proj(generation).view(-1, heads, head_dim)
        k_gen = attn.add_k_proj(generation).view(-1, kv_heads, head_dim)
        v_gen = attn.add_v_proj(generation).view(-1, kv_heads, head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und = apply_cosmos3_rotary(q_und, cos_und, sin_und)
        k_und = apply_cosmos3_rotary(k_und, cos_und, sin_und)
        q_gen = apply_cosmos3_rotary(q_gen, cos_gen, sin_gen)
        k_gen = apply_cosmos3_rotary(k_gen, cos_gen, sin_gen)

        k_und = _expand_gqa(k_und, heads, kv_heads)
        v_und = _expand_gqa(v_und, heads, kv_heads)
        k_gen = _expand_gqa(k_gen, heads, kv_heads)
        v_gen = _expand_gqa(v_gen, heads, kv_heads)
        return tuple(
            _nhd_to_bhnd(tensor)
            for tensor in (q_und, k_und, v_und, q_gen, k_gen, v_gen)
        )

    @staticmethod
    def _append_understanding_cluster(
        video_map: torch.Tensor,
        query_sizes: torch.Tensor,
        key_sizes: torch.Tensor,
        *,
        batch: int,
        heads: int,
        understanding_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_clusters = video_map.shape[-2]
        key_clusters = video_map.shape[-1]
        block_map = F.pad(
            video_map.view(batch, heads, query_clusters, key_clusters),
            (0, 1, 0, 1),
            value=False,
        )
        block_map[..., -1] = True
        query_sizes = F.pad(
            query_sizes.view(batch, heads, query_clusters), (0, 1), value=0
        )
        key_sizes = F.pad(
            key_sizes.view(batch, heads, key_clusters), (0, 1), value=0
        )
        query_sizes[..., -1] = understanding_tokens
        key_sizes[..., -1] = understanding_tokens
        return block_map, query_sizes, key_sizes

    def __call__(
        self,
        attn: torch.nn.Module,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        call = self.runtime.next_call(self.layer_idx)
        q_und, k_und, v_und, q_gen, k_gen, v_gen = self._project_qkv(
            attn, und_seq, gen_seq, rotary_emb
        )
        und_out = F.scaled_dot_product_attention(
            q_und, k_und, v_und, dropout_p=0.0, is_causal=True
        )

        if self.runtime.use_dense(self.layer_idx, call["step"]):
            k_full = torch.cat((k_und, k_gen), dim=2)
            v_full = torch.cat((v_und, v_gen), dim=2)
            gen_out = F.scaled_dot_product_attention(
                q_gen, k_full, v_full, dropout_p=0.0, is_causal=False
            )
            density = 1.0
            mode = "dense"
        else:
            batch, heads, gen_tokens, _ = q_gen.shape
            understanding_tokens = k_und.shape[2]
            k_full = torch.cat((k_gen, k_und), dim=2)
            v_full = torch.cat((v_gen, v_und), dim=2)
            core = self._core(call["cfg_branch"])
            route = core.route(
                q_gen,
                k_gen,
                v_gen,
                gqa_group_size=self.gqa_group_size,
                always_attended_keys=understanding_tokens,
                total_key_tokens=gen_tokens + understanding_tokens,
                k_full_source=k_full,
                v_full_source=v_full,
            )
            if not route.permuted_includes_suffix:
                raise RuntimeError("SparsePR did not return full Cosmos3 K/V layout.")
            q_order = torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous()
            q_permuted, _ = permute_tensor_by_labels_triton(
                q_gen, None, dim=2, sorted_indices=q_order
            )
            block_map, q_sizes, k_sizes = self._append_understanding_cluster(
                route.video_dynamic_map,
                route.q_cluster_sizes,
                route.k_cluster_sizes,
                batch=batch,
                heads=heads,
                understanding_tokens=understanding_tokens,
            )
            q_full = torch.cat((q_permuted, q_und), dim=2)
            output = dynamic_block_sparse_fwd_flashinfer(
                q_full,
                route.k_permuted,
                route.v_permuted,
                block_map,
                q_sizes,
                k_sizes,
                is_cpu=False,
            )[:, :, :gen_tokens]
            base = apply_inverse_permutation_triton(output, q_order, dim=2)
            gen_out, _ = core.repair(base, q_gen, k_full, v_full, route)
            density = min(
                1.0,
                float(route.base_video_density)
                + core.config.probe_rows / float(gen_tokens),
            )
            mode = "sparsepr"

        self.runtime.log({**call, "mode": mode, "density": density})
        return (
            attn.to_out(_bhnd_to_nhd_flat(und_out)),
            attn.to_add_out(_bhnd_to_nhd_flat(gen_out)),
        )


def install_cosmos3(pipe: Any, *, config: Cosmos3Config) -> int:
    """Install SparsePR on every Cosmos3 generation-attention module."""
    if not hasattr(pipe, "transformer"):
        raise TypeError("Expected a Cosmos3 pipeline with a transformer attribute.")
    modules = [
        module
        for _, module in pipe.transformer.named_modules()
        if is_cosmos3_attention(module)
    ]
    if not modules:
        raise RuntimeError("No Cosmos3 generation-attention modules were found.")

    set_flashinfer_backend(config.flashinfer_backend)
    set_flashinfer_workspace_cache_enabled(True)
    runtime = _Runtime(config, len(modules))
    for layer_idx, module in enumerate(modules):
        if not hasattr(module, "_sparsepr_dense_processor"):
            module._sparsepr_dense_processor = module.processor
        module.set_processor(
            Cosmos3N8Processor(
                layer_idx=layer_idx,
                runtime=runtime,
                n8_config=config.n8,
                query_heads=int(module.num_attention_heads),
                key_value_heads=int(module.num_key_value_heads),
                gqa_key_sharing=config.gqa_key_sharing,
            )
        )
    return len(modules)
