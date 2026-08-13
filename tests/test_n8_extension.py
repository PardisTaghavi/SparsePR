"""CUDA parity contracts for the optional N8 extension."""

import pytest
import torch

from sparsepr.kernels.n8_extension import load_n8_kernels


@pytest.fixture(scope="module")
def extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module = load_n8_kernels(required=False)
    if module is None:
        pytest.skip("n8_kernels is not built")
    return module


def test_wan_qkv_identity_rope_matches_torch(extension) -> None:
    torch.manual_seed(0)
    batch, tokens, heads, head_dim = 1, 5, 2, 4
    channels = heads * head_dim
    q = torch.randn(batch, tokens, channels, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_weight = torch.randn(channels, device="cuda")
    k_weight = torch.randn(channels, device="cuda")
    freq_real = torch.ones(tokens, head_dim // 2, device="cuda")
    freq_imag = torch.zeros_like(freq_real)
    epsilon = 1e-6

    actual_q, actual_k, actual_v = extension.wan_qkv_norm_rope_layout(
        q,
        k,
        v,
        q_weight,
        k_weight,
        freq_real,
        freq_imag,
        heads,
        epsilon,
    )
    expected_q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + epsilon)
    expected_k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + epsilon)
    expected_q = (expected_q * q_weight).view(batch, tokens, heads, head_dim)
    expected_k = (expected_k * k_weight).view(batch, tokens, heads, head_dim)
    expected_v = v.view(batch, tokens, heads, head_dim)

    torch.testing.assert_close(actual_q, expected_q.permute(0, 2, 1, 3))
    torch.testing.assert_close(actual_k, expected_k.permute(0, 2, 1, 3))
    torch.testing.assert_close(actual_v, expected_v.permute(0, 2, 1, 3))


def test_cosmos25_qkv_zero_angle_matches_torch(extension) -> None:
    torch.manual_seed(1)
    batch, tokens, heads, head_dim = 1, 5, 2, 4
    channels = heads * head_dim
    q = torch.randn(batch, tokens, channels, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_weight = torch.randn(head_dim, device="cuda")
    k_weight = torch.randn(head_dim, device="cuda")
    angles = torch.zeros(tokens, 1, 1, head_dim, device="cuda")
    epsilon = 1e-6

    actual_q, actual_k, actual_v = extension.cosmos25_qkv_norm_rope_bshd(
        q, k, v, q_weight, k_weight, angles, heads, epsilon
    )
    q_heads = q.view(batch, tokens, heads, head_dim)
    k_heads = k.view(batch, tokens, heads, head_dim)
    expected_q = q_heads * torch.rsqrt(q_heads.square().mean(dim=-1, keepdim=True) + epsilon)
    expected_k = k_heads * torch.rsqrt(k_heads.square().mean(dim=-1, keepdim=True) + epsilon)

    torch.testing.assert_close(actual_q, expected_q * q_weight)
    torch.testing.assert_close(actual_k, expected_k * k_weight)
    torch.testing.assert_close(actual_v, v.view(batch, tokens, heads, head_dim))


def test_selector_schedule_and_compaction_are_exact(extension) -> None:
    scores = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]],
        device="cuda",
    )
    sizes = torch.tensor([[2, 3, 5, 1]], device="cuda", dtype=torch.int32)
    budgets = torch.tensor([[6, 4]], device="cuda", dtype=torch.int32)
    block_map, selected_ids, selected_counts = extension.svg_ear_select_budget_schedule(
        scores, sizes, budgets
    )

    expected_mask = torch.tensor(
        [[True, True, True, False], [False, True, True, False]], device="cuda"
    )
    torch.testing.assert_close(block_map[0, 1:, 1:], expected_mask)
    assert block_map[0, 0, 0].item()
    assert block_map[0, 1:, 0].all().item()
    torch.testing.assert_close(selected_counts, selected_counts.new_tensor([[3, 2]]))
    torch.testing.assert_close(selected_ids[0, 0, :3], selected_ids.new_tensor([0, 1, 2]))
    torch.testing.assert_close(selected_ids[0, 1, :2], selected_ids.new_tensor([1, 2]))

    compact_ids, compact_counts = extension.compact_selected_clusters(expected_mask)
    torch.testing.assert_close(compact_counts, compact_counts.new_tensor([3, 2]))
    torch.testing.assert_close(compact_ids[0, :3], compact_ids.new_tensor([0, 1, 2]))
    torch.testing.assert_close(compact_ids[1, :2], compact_ids.new_tensor([1, 2]))
