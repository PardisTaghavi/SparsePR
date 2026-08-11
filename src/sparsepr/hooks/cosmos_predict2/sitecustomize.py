"""Deferred SparsePR installation for official Cosmos-Predict2.5 inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = os.environ.get("SPARSEPR_COSMOS_PREDICT2_CONFIG")

if CONFIG_PATH:
    config = json.loads(Path(CONFIG_PATH).read_text())
    from cosmos_oss import init as cosmos_init

    _original_init_environment = cosmos_init.init_environment
    _installed = False

    def _init_then_install(*args, **kwargs):
        global _installed
        result = _original_init_environment(*args, **kwargs)
        if not _installed:
            from sparsepr.adapters.cosmos_predict2 import install_cosmos25_n8_patch

            install_cosmos25_n8_patch(config)
            _installed = True
        return result

    cosmos_init.init_environment = _init_then_install
