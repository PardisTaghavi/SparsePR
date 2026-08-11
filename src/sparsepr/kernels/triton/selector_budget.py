"""Exact full-width selector budget/scatter epilogue for SWM routing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _exact_budget_mask_kernel(
    ORDER,
    SIZES,
    BUDGETS,
    OUTPUT,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q_CLUSTERS
    q = row - bh * Q_CLUSTERS
    offsets = tl.arange(0, BLOCK_K)
    valid = offsets < K_CLUSTERS
    order = tl.load(
        ORDER + row * K_CLUSTERS + offsets,
        mask=valid,
        other=0,
    ).to(tl.int32)
    costs = tl.load(
        SIZES + bh * K_CLUSTERS + order,
        mask=valid,
        other=0,
    ).to(tl.int32)
    positive = valid & (costs > 0)
    cumulative = tl.cumsum(tl.where(positive, costs, 0), axis=0)
    budget = tl.load(BUDGETS + row).to(tl.int32)
    keep = positive & (cumulative <= budget)
    selected_cost = tl.sum(tl.where(keep, costs, 0), axis=0)
    selected_count = tl.sum(keep.to(tl.int32), axis=0)
    valid_count = tl.sum(positive.to(tl.int32), axis=0)
    need_crossing = (
        (budget > 0)
        & (selected_cost < budget)
        & (selected_count < valid_count)
    )
    keep = keep | ((offsets == selected_count) & need_crossing)

    row_base = (bh * (Q_CLUSTERS + 1) + q + 1) * (K_CLUSTERS + 1)
    tl.store(
        OUTPUT + row_base + order + 1,
        keep,
        mask=valid,
    )


@triton.jit
def _exact_budget_packed_sort_mask_kernel(
    SCORES,
    SIZES,
    BUDGETS,
    OUTPUT,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Bitonic sort and exact budget/scatter in one program per query row."""
    row = tl.program_id(0)
    bh = row // Q_CLUSTERS
    q = row - bh * Q_CLUSTERS
    offsets = tl.arange(0, BLOCK_K)
    valid = offsets < K_CLUSTERS
    scores = tl.load(
        SCORES + row * K_CLUSTERS + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Convert IEEE float ordering to monotonically increasing unsigned keys,
    # then append the inverse source index as a deterministic tie breaker.
    # Invalid power-of-two padding receives key zero and therefore sorts last.
    bits = scores.to(tl.uint32, bitcast=True)
    ordered_bits = bits ^ tl.where(
        (bits & 0x80000000) != 0,
        tl.full((BLOCK_K,), 0xFFFFFFFF, tl.uint32),
        tl.full((BLOCK_K,), 0x80000000, tl.uint32),
    )
    tie = tl.full((BLOCK_K,), 0xFFFFFFFF, tl.uint32) - offsets.to(tl.uint32)
    packed = (ordered_bits.to(tl.uint64) << 32) | tie.to(tl.uint64)
    packed = tl.where(valid, packed, tl.zeros((BLOCK_K,), tl.uint64))
    packed = tl.sort(packed, dim=0, descending=True)
    order = (
        tl.full((BLOCK_K,), 0xFFFFFFFF, tl.uint64)
        - (packed & tl.full((BLOCK_K,), 0xFFFFFFFF, tl.uint64))
    ).to(tl.int32)

    costs = tl.load(
        SIZES + bh * K_CLUSTERS + order,
        mask=valid,
        other=0,
    ).to(tl.int32)
    positive = valid & (costs > 0)
    cumulative = tl.cumsum(tl.where(positive, costs, 0), axis=0)
    budget = tl.load(BUDGETS + row).to(tl.int32)
    keep = positive & (cumulative <= budget)
    selected_cost = tl.sum(tl.where(keep, costs, 0), axis=0)
    selected_count = tl.sum(keep.to(tl.int32), axis=0)
    valid_count = tl.sum(positive.to(tl.int32), axis=0)
    need_crossing = (
        (budget > 0)
        & (selected_cost < budget)
        & (selected_count < valid_count)
    )
    keep = keep | ((offsets == selected_count) & need_crossing)

    row_base = (bh * (Q_CLUSTERS + 1) + q + 1) * (K_CLUSTERS + 1)
    tl.store(OUTPUT + row_base, True)
    tl.store(
        OUTPUT + row_base + order + 1,
        keep,
        mask=valid,
    )
    header_valid = offsets < (K_CLUSTERS + 1)
    tl.store(
        OUTPUT + bh * (Q_CLUSTERS + 1) * (K_CLUSTERS + 1) + offsets,
        offsets == 0,
        mask=header_valid & (q == 0),
    )


def exact_budget_dynamic_map_triton(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    """Exact argsort followed by fused gather/cumsum/budget/mask scatter."""
    if not scores.is_cuda:
        raise RuntimeError("The exact selector-budget kernel requires CUDA tensors.")
    if scores.ndim != 3 or cluster_sizes.ndim != 2 or budgets.ndim != 2:
        raise RuntimeError("Expected scores [BH,Q,K], sizes [BH,K], budgets [BH,Q].")
    bh, q_clusters, k_clusters = scores.shape
    if tuple(cluster_sizes.shape) != (bh, k_clusters):
        raise RuntimeError("Key-cluster sizes do not match selector scores.")
    if tuple(budgets.shape) != (bh, q_clusters):
        raise RuntimeError("Per-query budgets do not match selector scores.")
    if k_clusters > 2048:
        raise RuntimeError(
            f"Exact selector supports at most 2048 key clusters, got {k_clusters}."
        )
    order = torch.argsort(scores, dim=2, descending=True).to(torch.int32).contiguous()
    sizes_i32 = cluster_sizes.to(device=scores.device, dtype=torch.int32).contiguous()
    budgets_i32 = budgets.to(device=scores.device, dtype=torch.int32).contiguous()
    output = torch.zeros(
        (bh, q_clusters + 1, k_clusters + 1),
        device=scores.device,
        dtype=torch.bool,
    )
    output[:, :, 0] = True
    block_k = triton.next_power_of_2(k_clusters)
    _exact_budget_mask_kernel[(bh * q_clusters,)](
        order,
        sizes_i32,
        budgets_i32,
        output,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=1,
    )
    return output


def exact_budget_dynamic_map_torch_sort_triton(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    """Use torch.sort indices with the existing fused exact budget epilogue."""
    if not scores.is_cuda:
        raise RuntimeError("The exact selector-budget kernel requires CUDA tensors.")
    if scores.ndim != 3 or cluster_sizes.ndim != 2 or budgets.ndim != 2:
        raise RuntimeError("Expected scores [BH,Q,K], sizes [BH,K], budgets [BH,Q].")
    bh, q_clusters, k_clusters = scores.shape
    if tuple(cluster_sizes.shape) != (bh, k_clusters):
        raise RuntimeError("Key-cluster sizes do not match selector scores.")
    if tuple(budgets.shape) != (bh, q_clusters):
        raise RuntimeError("Per-query budgets do not match selector scores.")
    order = torch.sort(scores, dim=2, descending=True).indices.to(torch.int32).contiguous()
    sizes_i32 = cluster_sizes.to(device=scores.device, dtype=torch.int32).contiguous()
    budgets_i32 = budgets.to(device=scores.device, dtype=torch.int32).contiguous()
    output = torch.zeros(
        (bh, q_clusters + 1, k_clusters + 1),
        device=scores.device,
        dtype=torch.bool,
    )
    output[:, :, 0] = True
    block_k = triton.next_power_of_2(k_clusters)
    _exact_budget_mask_kernel[(bh * q_clusters,)](
        order,
        sizes_i32,
        budgets_i32,
        output,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=1,
    )
    return output


def exact_budget_dynamic_map_packed_sort_triton(
    scores: torch.Tensor,
    cluster_sizes: torch.Tensor,
    budgets: torch.Tensor,
    *,
    num_warps: int = 8,
) -> torch.Tensor:
    """Experimental exact fused bitonic sort/budget/scatter candidate."""
    if not scores.is_cuda:
        raise RuntimeError("The packed exact selector requires CUDA tensors.")
    if scores.ndim != 3 or cluster_sizes.ndim != 2 or budgets.ndim != 2:
        raise RuntimeError("Expected scores [BH,Q,K], sizes [BH,K], budgets [BH,Q].")
    bh, q_clusters, k_clusters = scores.shape
    if tuple(cluster_sizes.shape) != (bh, k_clusters):
        raise RuntimeError("Key-cluster sizes do not match selector scores.")
    if tuple(budgets.shape) != (bh, q_clusters):
        raise RuntimeError("Per-query budgets do not match selector scores.")
    if k_clusters > 1024:
        raise RuntimeError(
            f"Packed exact selector supports at most 1024 key clusters, got {k_clusters}."
        )
    if num_warps not in {4, 8, 16}:
        raise RuntimeError("Packed exact selector num_warps must be 4, 8, or 16.")
    sizes_i32 = cluster_sizes.to(device=scores.device, dtype=torch.int32).contiguous()
    budgets_i32 = budgets.to(device=scores.device, dtype=torch.int32).contiguous()
    output = torch.empty(
        (bh, q_clusters + 1, k_clusters + 1),
        device=scores.device,
        dtype=torch.bool,
    )
    block_k = triton.next_power_of_2(k_clusters + 1)
    _exact_budget_packed_sort_mask_kernel[(bh * q_clusters,)](
        scores.contiguous(),
        sizes_i32,
        budgets_i32,
        output,
        Q_CLUSTERS=q_clusters,
        K_CLUSTERS=k_clusters,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
