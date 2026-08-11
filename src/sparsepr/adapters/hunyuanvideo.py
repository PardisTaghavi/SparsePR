"""SparsePR adapter for Diffusers HunyuanVideo-13B."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sparsepr.kernels.triton.permute import (
    apply_inverse_permutation_triton,
    permute_tensor_by_labels_triton,
)
from sparsepr.kmeans_utils import (
    dynamic_block_sparse_fwd_flashinfer,
    set_flashinfer_workspace_cache_enabled,
)
from sparsepr.models.common import N8V6Config, N8V6Core


@dataclass(frozen=True)
class HunyuanVideoConfig:
    """Frozen SparsePR settings for HunyuanVideo-13B."""

    height: int = 720
    width: int = 1280
    num_frames: int = 129
    context_length: int = 256
    first_layers: int = 1
    first_timestep: float = 0.0
    n8: N8V6Config = N8V6Config()


def prompt_length(pipe: Any, prompt: str, max_sequence_length: int = 256) -> int:
    """Return HunyuanVideo's real prompt-token count after template cropping."""
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
        DEFAULT_PROMPT_TEMPLATE,
    )

    template = DEFAULT_PROMPT_TEMPLATE
    rendered = [template["template"].format(prompt)]
    crop_start = template.get("crop_start")
    if crop_start is None:
        encoded_template = pipe.tokenizer(
            template["template"],
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
        )
        crop_start = encoded_template["input_ids"].shape[-1] - 2
    encoded = pipe.tokenizer(
        rendered,
        max_length=max_sequence_length + crop_start,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    mask = encoded.attention_mask[:, crop_start:]
    return int(mask.sum().item())


class HunyuanVideoN8Processor:
    """Hunyuan QKV/layout adapter around the model-independent SparsePR core."""

    def __init__(
        self,
        *,
        layer_idx: int,
        config: HunyuanVideoConfig,
        real_prompt_length: int,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SparsePR requires PyTorch 2.0 or newer.")
        self.layer_idx = layer_idx
        self.config = config
        self.prompt_length = int(real_prompt_length)
        self.num_frame_patches = 1 + config.num_frames // 4
        self.frame_size = config.height * config.width // 256
        self.n8 = N8V6Core(config.n8)
        self._q_full_order: torch.Tensor | None = None

    @staticmethod
    def _apply_rotary(
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        single_stream: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is None:
            return query, key
        from diffusers.models.embeddings import apply_rotary_emb

        if single_stream and encoder_hidden_states is not None:
            text_tokens = encoder_hidden_states.shape[1]
            query = torch.cat(
                (apply_rotary_emb(query[:, :, :-text_tokens], image_rotary_emb),
                 query[:, :, -text_tokens:]),
                dim=2,
            )
            key = torch.cat(
                (apply_rotary_emb(key[:, :, :-text_tokens], image_rotary_emb),
                 key[:, :, -text_tokens:]),
                dim=2,
            )
            return query, key
        return apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(
            key, image_rotary_emb
        )

    @staticmethod
    def _dense_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, tokens, _ = query.shape
        real_tokens = int(attention_mask.sum().item())
        if not 0 < real_tokens <= tokens:
            raise RuntimeError(
                f"Invalid Hunyuan attention mask length {real_tokens}/{tokens}."
            )
        padded_tokens = tokens - real_tokens
        if padded_tokens == 0:
            return F.scaled_dot_product_attention(query, key, value)
        sizes = torch.tensor(
            [real_tokens, padded_tokens], device=query.device, dtype=torch.int32
        ).view(1, 1, 2).expand(batch, heads, -1)
        block_map = torch.eye(2, device=query.device, dtype=torch.bool)
        block_map = block_map.view(1, 1, 2, 2).expand(batch, heads, -1, -1)
        return dynamic_block_sparse_fwd_flashinfer(
            query, key, value, block_map, sizes, sizes, is_cpu=False
        )

    def _full_query_order(
        self, video_order: torch.Tensor, video_tokens: int, text_tokens: int
    ) -> torch.Tensor:
        batch_heads = video_order.shape[0]
        total = video_tokens + text_tokens
        expected = (batch_heads, total)
        if (
            self._q_full_order is None
            or tuple(self._q_full_order.shape) != expected
            or self._q_full_order.device != video_order.device
        ):
            self._q_full_order = torch.empty(
                expected, device=video_order.device, dtype=torch.int32
            )
            suffix = torch.arange(
                video_tokens, total, device=video_order.device, dtype=torch.int32
            )
            self._q_full_order[:, video_tokens:].copy_(suffix.unsqueeze(0))
        self._q_full_order[:, :video_tokens].copy_(video_order)
        return self._q_full_order

    def _sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, tokens, _ = query.shape
        video_tokens = self.num_frame_patches * self.frame_size
        text_tokens = self.config.context_length
        if tokens != video_tokens + text_tokens:
            raise RuntimeError(
                f"Hunyuan sequence mismatch: {tokens} != {video_tokens}+{text_tokens}."
            )

        q_video = query[:, :, :video_tokens].contiguous()
        k_video = key[:, :, :video_tokens].contiguous()
        v_video = value[:, :, :video_tokens].contiguous()
        route = self.n8.route(
            q_video,
            k_video,
            v_video,
            always_attended_keys=self.prompt_length,
            total_key_tokens=video_tokens + self.prompt_length,
            k_full_source=key,
            v_full_source=value,
        )
        if not route.permuted_includes_suffix:
            raise RuntimeError("SparsePR did not return full Hunyuan K/V layout.")

        q_order = torch.argsort(route.q_labels, dim=-1).to(torch.int32).contiguous()
        q_full_order = self._full_query_order(q_order, video_tokens, text_tokens)
        q_permuted, _ = permute_tensor_by_labels_triton(
            query, None, dim=2, sorted_indices=q_full_order
        )

        q_clusters = route.video_dynamic_map.shape[-2]
        k_clusters = route.video_dynamic_map.shape[-1]
        block_map = F.pad(
            route.video_dynamic_map.view(batch, heads, q_clusters, k_clusters),
            (0, 2, 0, 2),
            value=False,
        )
        block_map[:, :, :-1, -2] = True
        block_map[:, :, -2, :-1] = True
        block_map[:, :, -1, -1] = True

        padding = text_tokens - self.prompt_length
        q_sizes = F.pad(
            route.q_cluster_sizes.view(batch, heads, q_clusters), (0, 2), value=0
        )
        k_sizes = F.pad(
            route.k_cluster_sizes.view(batch, heads, k_clusters), (0, 2), value=0
        )
        q_sizes[:, :, -2:] = torch.tensor(
            [self.prompt_length, padding], device=query.device, dtype=q_sizes.dtype
        )
        k_sizes[:, :, -2:] = torch.tensor(
            [self.prompt_length, padding], device=query.device, dtype=k_sizes.dtype
        )

        output_permuted = dynamic_block_sparse_fwd_flashinfer(
            q_permuted,
            route.k_permuted,
            route.v_permuted,
            block_map,
            q_sizes,
            k_sizes,
            is_cpu=False,
        )
        base_video = apply_inverse_permutation_triton(
            output_permuted[:, :, :video_tokens], q_order, dim=2
        )
        repaired_video, _ = self.n8.repair(
            base_video,
            q_video,
            key[:, :, : video_tokens + self.prompt_length],
            value[:, :, : video_tokens + self.prompt_length],
            route,
        )
        return torch.cat((repaired_video, output_permuted[:, :, video_tokens:]), dim=2)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        single_stream = attn.add_q_proj is None and encoder_hidden_states is not None
        if single_stream:
            hidden_states = torch.cat((hidden_states, encoder_hidden_states), dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query, key = self._apply_rotary(
            query, key, image_rotary_emb, encoder_hidden_states, single_stream
        )

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat((query, encoder_query), dim=2)
            key = torch.cat((key, encoder_key), dim=2)
            value = torch.cat((value, encoder_value), dim=2)

        dense = self.layer_idx < self.config.first_layers
        if timestep is not None:
            dense = dense or bool(timestep[0] > self.config.first_timestep)
        if dense:
            if attention_mask is None:
                raise RuntimeError("Hunyuan dense warmup requires attention_mask.")
            output = self._dense_attention(query, key, value, attention_mask)
        else:
            output = self._sparse_attention(query, key, value)

        output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            output, encoder_output = (
                output[:, : -encoder_hidden_states.shape[1]],
                output[:, -encoder_hidden_states.shape[1] :],
            )
            if getattr(attn, "to_out", None) is not None:
                output = attn.to_out[1](attn.to_out[0](output))
            if getattr(attn, "to_add_out", None) is not None:
                encoder_output = attn.to_add_out(encoder_output)
            return output, encoder_output
        return output, None


def install_hunyuanvideo(
    pipe: Any,
    prompt: str,
    *,
    config: HunyuanVideoConfig,
) -> int:
    """Install SparsePR processors and return the real prompt-token count."""
    real_prompt_length = prompt_length(pipe, prompt, config.context_length)
    set_flashinfer_workspace_cache_enabled(True)
    double_blocks = pipe.transformer.transformer_blocks
    single_blocks = pipe.transformer.single_transformer_blocks
    for layer_idx, block in enumerate(double_blocks):
        block.attn.processor = HunyuanVideoN8Processor(
            layer_idx=layer_idx,
            config=config,
            real_prompt_length=real_prompt_length,
        )
    for offset, block in enumerate(single_blocks, start=len(double_blocks)):
        block.attn.processor = HunyuanVideoN8Processor(
            layer_idx=offset,
            config=config,
            real_prompt_length=real_prompt_length,
        )
    return real_prompt_length


def dense_warmup_threshold(
    scheduler: Any, *, num_steps: int, fraction: float
) -> float:
    """Convert a leading dense-step fraction to Hunyuan's timestep threshold."""
    dense_steps = math.floor(fraction * num_steps)
    if dense_steps <= 0:
        return 1001.0
    return float(scheduler.timesteps[dense_steps - 1].item() - 1)
