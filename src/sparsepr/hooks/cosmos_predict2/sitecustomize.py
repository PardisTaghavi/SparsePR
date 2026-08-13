"""Deferred SparsePR installation for official Cosmos-Predict2.5 inference."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

CONFIG_PATH = os.environ.get("SPARSEPR_COSMOS_PREDICT2_CONFIG")
CPU_MATERIALIZATION = os.environ.get("SPARSEPR_COSMOS25_CPU_MATERIALIZATION", "0") == "1"

if (CONFIG_PATH or CPU_MATERIALIZATION) and importlib.util.find_spec("cosmos_oss"):
    config = json.loads(Path(CONFIG_PATH).read_text()) if CONFIG_PATH else None
    from cosmos_oss import init as cosmos_init

    _original_init_environment = cosmos_init.init_environment
    _sparse_installed = False
    _cpu_installed = False

    def _install_cpu_materialization() -> None:
        from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
            Text2WorldModelRectifiedFlow,
        )

        cls = Text2WorldModelRectifiedFlow
        if getattr(cls, "_sparsepr_cpu_materialization", False):
            return
        original_build_net = cls.build_net
        original_on_train_start = cls.on_train_start

        def build_net_on_cpu(self, keep_on_cpu=False):
            del keep_on_cpu
            return original_build_net(self, keep_on_cpu=True)

        def convert_on_cpu(self, memory_format=None):
            import torch

            if memory_format is None:
                memory_format = torch.preserve_format
            tensor_kwargs = self.tensor_kwargs
            self.tensor_kwargs = {**tensor_kwargs, "device": "cpu"}
            try:
                return original_on_train_start(self, memory_format=memory_format)
            finally:
                self.tensor_kwargs = tensor_kwargs

        cls.build_net = build_net_on_cpu
        cls.on_train_start = convert_on_cpu
        cls._sparsepr_cpu_materialization = True

    def _init_then_install(*args, **kwargs):
        global _cpu_installed, _sparse_installed
        result = _original_init_environment(*args, **kwargs)
        if CPU_MATERIALIZATION and not _cpu_installed:
            _install_cpu_materialization()
            _cpu_installed = True
        if config is not None and not _sparse_installed:
            from sparsepr.adapters.cosmos_predict2 import install_cosmos25_n8_patch

            install_cosmos25_n8_patch(config)
            _sparse_installed = True
        return result

    cosmos_init.init_environment = _init_then_install
