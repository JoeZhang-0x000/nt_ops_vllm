from __future__ import annotations

import torch

import ntops.torch
from vllm.logger import init_logger

from nt_ops.capabilities import record_hit

logger = init_logger(__name__)


def _normalized_shape(weight: torch.Tensor | None, x: torch.Tensor) -> tuple[int, ...]:
    if weight is None:
        return (x.shape[-1],)
    return tuple(weight.shape)


def rms_norm_helper(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    logger.info_once("\033[32mNT RMS is enabled.\033[0m")
    record_hit("rms_norm")
    return ntops.torch.rms_norm(
        x,
        _normalized_shape(weight, x),
        weight=weight,
        eps=variance_epsilon,
    )


def fused_add_rms_norm_helper(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info_once("\033[32mNT RMS is enabled.\033[0m")
    record_hit("fused_add_rms_norm")

    residual_out = ntops.torch.add(x, residual)
    output = ntops.torch.rms_norm(
        residual_out,
        _normalized_shape(weight, residual_out),
        weight=weight,
        eps=variance_epsilon,
    )
    x.copy_(output)
    residual.copy_(residual_out)
    return x, residual
