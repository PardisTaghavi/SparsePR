"""Lazy model-specific SparsePR integrations."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Cosmos25N8Config": ("cosmos_predict2", "Cosmos25N8Config"),
    "install_cosmos25_n8_patch": ("cosmos_predict2", "install_cosmos25_n8_patch"),
    "Cosmos3Config": ("cosmos3", "Cosmos3Config"),
    "install_cosmos3": ("cosmos3", "install_cosmos3"),
    "HunyuanVideoConfig": ("hunyuanvideo", "HunyuanVideoConfig"),
    "install_hunyuanvideo": ("hunyuanvideo", "install_hunyuanvideo"),
    "Wan22TI2VSparseConfig": ("wan22", "Wan22TI2VSparseConfig"),
    "install_wan22_ti2v_sparse_patch": ("wan22", "install_wan22_ti2v_sparse_patch"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
