"""Optional CUDA operator timing used by the runtime kernels."""

from __future__ import annotations

import os
from contextlib import ContextDecorator

import torch

ENABLE_LOGGING = int(os.getenv("SPARSEPR_TIME_BENCH", "0")) >= 1
operator_log_data: dict[str, float] = {}


class TimeLoggingContext(ContextDecorator):
    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if ENABLE_LOGGING:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if ENABLE_LOGGING:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)
            operator_log_data[self.operation_type] = (
                operator_log_data.get(self.operation_type, 0.0) + elapsed
            )
        return False


time_logging_decorator = TimeLoggingContext


def clear_operator_log_data() -> None:
    operator_log_data.clear()
