"""Cosmos3 adapter parity contracts."""

import importlib

import pytest


torch = pytest.importorskip("torch")
apply_cosmos3_rotary = importlib.import_module(
    "sparsepr.adapters.cosmos3"
).apply_cosmos3_rotary


def test_rotary_matches_diffusers_half_split_layout() -> None:
    tensor = torch.randn(7, 4, 8, dtype=torch.float32)
    cos = torch.randn(7, 8, dtype=torch.float32)
    sin = torch.randn(7, 8, dtype=torch.float32)
    half = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
    expected = tensor * cos[:, None, :] + rotated * sin[:, None, :]
    torch.testing.assert_close(apply_cosmos3_rotary(tensor, cos, sin), expected)
