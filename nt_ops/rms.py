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


def build_rms_forward_oot(original):
    """Replace RMSNorm.forward_oot to intercept all MLU dispatch paths.

    ``dispatch_forward`` on out-of-tree platforms binds ``_forward_method``
    to ``self.forward_oot`` (which by default calls ``forward_native``).
    Patching this method ensures our kernels are reached regardless of
    whether the prior occupant was the base-class fallback or an MLU-
    specific override saved as ``original``.

    Falls back to ``original`` for unsupported edge cases
    (variance_size_override, no weight) so MLU correctness is preserved.
    """

    def forward_oot(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # Edge cases not supported by ntops kernels — delegate to whatever
        # was there before (MLU kernel or PyTorch-native fallback).
        if getattr(self, "variance_size_override", None) is not None:
            return original(self, x, residual)
        if not getattr(self, "has_weight", True):
            return original(self, x, residual)

        weight = self.weight.data
        eps = self.variance_epsilon

        if residual is not None:
            return fused_add_rms_norm_helper(x, residual, weight, eps)
        return rms_norm_helper(x, weight, eps)

    return forward_oot
