"""Opt-in, runtime-gated non-attention fusions for Cosmos-Predict2.5.

The large MLP GEMMs remain on PyTorch/cuBLAS.  ``torch.compile`` is used only
to select/fuse their surrounding graph and the block elementwise primitives.
Every candidate is checked on the first real model tensor for numerical
agreement and H100 latency before it is enabled for the remainder of the run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F


@dataclass
class FusionGate:
    requested: bool
    tested: bool = False
    passed_quality: bool = False
    passed_speed: bool = False
    enabled: bool = False
    baseline_ms: float | None = None
    candidate_ms: float | None = None
    speedup: float | None = None
    max_abs_error: float | None = None
    relative_rmse: float | None = None
    cosine_similarity: float | None = None
    error: str | None = None


class Cosmos25NonAttentionFusions:
    """Install and gate compiled MLP and block-elementwise paths."""

    def __init__(self, *, compile_mlp: bool, fuse_block_elementwise: bool):
        self.mlp = FusionGate(requested=compile_mlp)
        self.modulated_norm = FusionGate(requested=fuse_block_elementwise)
        self.gated_residual = FusionGate(requested=fuse_block_elementwise)
        self._original_block_forward = None
        self._original_mlp_forward = None
        self._compiled_mlp = None
        self._compiled_modulated_norm = None
        self._compiled_gated_residual = None

    @staticmethod
    def _quality(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
        ref = reference.float()
        got = candidate.float()
        delta = got - ref
        ref_rms = torch.sqrt(torch.mean(ref.square())).clamp_min(1.0e-8)
        relative_rmse = torch.sqrt(torch.mean(delta.square())) / ref_rms
        cosine = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0)
        return {
            "max_abs_error": float(delta.abs().max().item()),
            "relative_rmse": float(relative_rmse.item()),
            "cosine_similarity": float(cosine.item()),
        }

    @staticmethod
    def _latency_ms(fn: Callable[[], torch.Tensor], iterations: int) -> float:
        # One untimed call warms the selected GEMM and generated Triton kernels.
        value = fn()
        torch.cuda.synchronize(value.device)
        del value
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            value = fn()
        end.record()
        end.synchronize()
        del value
        return float(start.elapsed_time(end)) / iterations

    def _run_gate(
        self,
        gate: FusionGate,
        *,
        baseline: Callable[[], torch.Tensor],
        candidate: Callable[[], torch.Tensor],
        iterations: int,
        minimum_speedup: float,
    ) -> None:
        if gate.tested or not gate.requested:
            return
        gate.tested = True
        try:
            reference = baseline()
            proposed = candidate()
            torch.cuda.synchronize(reference.device)
            metrics = self._quality(reference, proposed)
            gate.max_abs_error = metrics["max_abs_error"]
            gate.relative_rmse = metrics["relative_rmse"]
            gate.cosine_similarity = metrics["cosine_similarity"]
            gate.passed_quality = bool(
                torch.isfinite(proposed).all().item()
                and gate.relative_rmse <= 2.0e-3
                and gate.cosine_similarity >= 0.99999
            )
            del reference, proposed
            gate.baseline_ms = self._latency_ms(baseline, iterations)
            gate.candidate_ms = self._latency_ms(candidate, iterations)
            gate.speedup = gate.baseline_ms / gate.candidate_ms
            gate.passed_speed = gate.speedup >= minimum_speedup
            gate.enabled = gate.passed_quality and gate.passed_speed
        except Exception as exc:  # The official path remains the safe fallback.
            gate.error = f"{type(exc).__name__}: {exc}"
            gate.enabled = False
        print(
            "[C25-FUSION-GATE] "
            f"quality={gate.passed_quality} speed={gate.passed_speed} "
            f"enabled={gate.enabled} baseline_ms={gate.baseline_ms} "
            f"candidate_ms={gate.candidate_ms} speedup={gate.speedup} "
            f"relative_rmse={gate.relative_rmse} error={gate.error}",
            flush=True,
        )

    def install(self) -> None:
        if not any(
            gate.requested
            for gate in (self.mlp, self.modulated_norm, self.gated_residual)
        ):
            return
        if self._original_block_forward is not None:
            return

        from einops import rearrange
        from torch import amp
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Block,
            GPT2FeedForward,
            VideoSize,
        )

        self._original_block_forward = Block.forward
        self._original_mlp_forward = GPT2FeedForward.forward
        controller = self

        def mlp_graph(x, weight1, weight2):
            return F.linear(F.gelu(F.linear(x, weight1)), weight2)

        def modulated_norm_graph(x, scale, shift):
            normalized = F.layer_norm(x, (x.shape[-1],), eps=1.0e-6)
            return normalized * (1.0 + scale) + shift

        def gated_residual_graph(x, gate, result):
            return x + gate * result

        compile_options = {
            "fullgraph": True,
            "dynamic": False,
            "mode": "max-autotune-no-cudagraphs",
        }
        if self.mlp.requested:
            self._compiled_mlp = torch.compile(mlp_graph, **compile_options)
        if self.modulated_norm.requested:
            self._compiled_modulated_norm = torch.compile(
                modulated_norm_graph, **compile_options
            )
        if self.gated_residual.requested:
            self._compiled_gated_residual = torch.compile(
                gated_residual_graph, **compile_options
            )

        def patched_mlp(mlp, x):
            baseline = lambda: controller._original_mlp_forward(mlp, x)
            if not controller.mlp.requested:
                return baseline()
            candidate = lambda: controller._compiled_mlp(
                x, mlp.layer1.weight, mlp.layer2.weight
            )
            controller._run_gate(
                controller.mlp,
                baseline=baseline,
                candidate=candidate,
                iterations=2,
                minimum_speedup=1.02,
            )
            return candidate() if controller.mlp.enabled else baseline()

        def modulated_norm(x, scale, shift):
            baseline = lambda: F.layer_norm(
                x, (x.shape[-1],), eps=1.0e-6
            ) * (1.0 + scale) + shift
            if not controller.modulated_norm.requested:
                return baseline()
            candidate = lambda: controller._compiled_modulated_norm(x, scale, shift)
            controller._run_gate(
                controller.modulated_norm,
                baseline=baseline,
                candidate=candidate,
                iterations=4,
                minimum_speedup=1.05,
            )
            return candidate() if controller.modulated_norm.enabled else baseline()

        def gated_residual(x, gate, result):
            baseline = lambda: x + gate * result
            if not controller.gated_residual.requested:
                return baseline()
            candidate = lambda: controller._compiled_gated_residual(x, gate, result)
            controller._run_gate(
                controller.gated_residual,
                baseline=baseline,
                candidate=candidate,
                iterations=4,
                minimum_speedup=1.05,
            )
            return candidate() if controller.gated_residual.enabled else baseline()

        def patched_block(
            block,
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            rope_emb_L_1_1_D=None,
            adaln_lora_B_T_3D=None,
            extra_per_block_pos_emb=None,
            kv_cache_cfg=None,
        ):
            if extra_per_block_pos_emb is not None:
                x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

            with amp.autocast(
                "cuda",
                enabled=block.use_wan_fp32_strategy,
                dtype=torch.float32,
            ):
                if block.use_adaln_lora:
                    shift_sa, scale_sa, gate_sa = (
                        block.adaln_modulation_self_attn(emb_B_T_D)
                        + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                    shift_ca, scale_ca, gate_ca = (
                        block.adaln_modulation_cross_attn(emb_B_T_D)
                        + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                    shift_mlp, scale_mlp, gate_mlp = (
                        block.adaln_modulation_mlp(emb_B_T_D)
                        + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                else:
                    shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(
                        emb_B_T_D
                    ).chunk(3, dim=-1)
                    shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(
                        emb_B_T_D
                    ).chunk(3, dim=-1)
                    shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(
                        emb_B_T_D
                    ).chunk(3, dim=-1)

            def broadcast(value):
                return rearrange(value, "b t d -> b t 1 1 d").type_as(
                    x_B_T_H_W_D
                )

            shift_sa, scale_sa, gate_sa = map(
                broadcast, (shift_sa, scale_sa, gate_sa)
            )
            shift_ca, scale_ca, gate_ca = map(
                broadcast, (shift_ca, scale_ca, gate_ca)
            )
            shift_mlp, scale_mlp, gate_mlp = map(
                broadcast, (shift_mlp, scale_mlp, gate_mlp)
            )
            _, time_dim, height, width, _ = x_B_T_H_W_D.shape
            video_size = VideoSize(T=time_dim, H=height, W=width)
            if block.cp_size is not None and block.cp_size > 1:
                video_size = VideoSize(
                    T=time_dim * block.cp_size, H=height, W=width
                )

            normalized = modulated_norm(x_B_T_H_W_D, scale_sa, shift_sa)
            result = rearrange(
                block.self_attn(
                    rearrange(normalized, "b t h w d -> b (t h w) d"),
                    None,
                    rope_emb=rope_emb_L_1_1_D,
                    video_size=video_size,
                    kv_cache_cfg=kv_cache_cfg,
                ),
                "b (t h w) d -> b t h w d",
                t=time_dim,
                h=height,
                w=width,
            )
            x_B_T_H_W_D = gated_residual(x_B_T_H_W_D, gate_sa, result)

            normalized = modulated_norm(x_B_T_H_W_D, scale_ca, shift_ca)
            result = rearrange(
                block.cross_attn(
                    rearrange(normalized, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=time_dim,
                h=height,
                w=width,
            )
            x_B_T_H_W_D = gated_residual(x_B_T_H_W_D, gate_ca, result)

            normalized = modulated_norm(x_B_T_H_W_D, scale_mlp, shift_mlp)
            result = block.mlp(normalized)
            return gated_residual(x_B_T_H_W_D, gate_mlp, result)

        GPT2FeedForward.forward = patched_mlp
        Block.forward = patched_block
        print(
            "[C25-FUSION] installed gated compiled MLP and block elementwise paths.",
            flush=True,
        )

    def restore(self) -> None:
        if self._original_block_forward is None:
            return
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (
            Block,
            GPT2FeedForward,
        )

        Block.forward = self._original_block_forward
        GPT2FeedForward.forward = self._original_mlp_forward
        self._original_block_forward = None
        self._original_mlp_forward = None

    def report(self) -> dict[str, Any]:
        return {
            "mlp": asdict(self.mlp),
            "modulated_norm": asdict(self.modulated_norm),
            "gated_residual": asdict(self.gated_residual),
            "generated_at_unix": time.time(),
        }


def install_from_environment() -> Cosmos25NonAttentionFusions | None:
    compile_mlp = os.environ.get("SPARSEPR_COSMOS25_COMPILE_MLP", "0") == "1"
    fuse_block = os.environ.get("SPARSEPR_COSMOS25_FUSE_BLOCK_ELEMENTWISE", "0") == "1"
    if not compile_mlp and not fuse_block:
        return None
    controller = Cosmos25NonAttentionFusions(
        compile_mlp=compile_mlp,
        fuse_block_elementwise=fuse_block,
    )
    controller.install()
    return controller
