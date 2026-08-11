"""Model-specific SparsePR integrations."""

from .cosmos_predict2 import Cosmos25N8Config, install_cosmos25_n8_patch
from .cosmos3 import Cosmos3Config, install_cosmos3
from .hunyuanvideo import HunyuanVideoConfig, install_hunyuanvideo
from .wan22 import Wan22TI2VSparseConfig, install_wan22_ti2v_sparse_patch

__all__ = [
    "Cosmos25N8Config",
    "Cosmos3Config",
    "HunyuanVideoConfig",
    "Wan22TI2VSparseConfig",
    "install_hunyuanvideo",
    "install_cosmos25_n8_patch",
    "install_cosmos3",
    "install_wan22_ti2v_sparse_patch",
]
