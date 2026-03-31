"""
NT_OPS debug tracing.

Set NT_OPS_DEBUG=1 in the worker environment to print a log line on every
operator call.  Off by default; no-overhead when disabled.

Usage in operator modules::

    from nt_ops.debug import trace
    trace("rms_norm", shape=x.shape, dtype=x.dtype)
"""
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

NT_OPS_DEBUG: bool = os.getenv("NT_OPS_DEBUG", "0") == "1"


def trace(op: str, **kwargs) -> None:
    """Log one line per call when NT_OPS_DEBUG=1.  No-op otherwise."""
    if NT_OPS_DEBUG:
        parts = "  ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info("\033[36m[NT_DEBUG] %s  %s\033[0m", op, parts)
