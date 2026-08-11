"""SparsePR sparse-attention runtime."""

from typing import Any

__all__ = ["N8V6Config", "N8V6Core"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load the CUDA-facing core lazily so CLI metadata remains lightweight."""
    if name in __all__:
        from .models.common import N8V6Config, N8V6Core

        return {"N8V6Config": N8V6Config, "N8V6Core": N8V6Core}[name]
    raise AttributeError(name)
