from __future__ import annotations

from ninetoothed import Tensor, make, Symbol
import ninetoothed.language as ntl
import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger
from typing import Tuple

from nt_ops.capabilities import record_hit
from nt_ops.debug import trace

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Ninetoothed kernel classes (from main branch — proven on MLU)
# ---------------------------------------------------------------------------

class RMSWithWeight:

    def arrangement(
        input,
        weight,
        output,
        eps,
    ):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, eps

    def application(input, weight, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    def premake(ndim):
        kernel = make(
            RMSWithWeight.arrangement,
            RMSWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSNoWeight:

    def arrangement(input, output, eps):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        return input_arranged, output_arranged, eps

    def application(input, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    def premake(ndim):
        kernel = make(
            RMSNoWeight.arrangement,
            RMSNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualWithWeight:

    def arrangement(
        input,
        weight,
        output,
        residual,
        eps,
    ):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        res_arranged = residual.tile(arrange_shape)
        res_arranged = _squeeze(res_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, res_arranged, eps

    def application(input, weight, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    def premake(ndim):
        kernel = make(
            RMSResidualWithWeight.arrangement,
            RMSResidualWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualNoWeight:

    def arrangement(input, output, residual, eps):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        res_arranged = residual.tile(arrange_shape)
        res_arranged = _squeeze(res_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        return input_arranged, output_arranged, res_arranged, eps

    def application(input, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    def premake(ndim):
        kernel = make(
            RMSResidualNoWeight.arrangement,
            RMSResidualNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


# ---------------------------------------------------------------------------
# Pre-compiled kernel table
# ---------------------------------------------------------------------------

_kernel = {}
_max_ndim = 4
_kernel["with_residual_with_weight"] = {
    i: RMSResidualWithWeight.premake(i) for i in range(2, _max_ndim)
}
_kernel["with_residual_no_weight"] = {
    i: RMSResidualNoWeight.premake(i) for i in range(2, _max_ndim)
}
_kernel["no_residual_with_weight"] = {
    i: RMSWithWeight.premake(i) for i in range(2, _max_ndim)
}
_kernel["no_residual_no_weight"] = {
    i: RMSNoWeight.premake(i) for i in range(2, _max_ndim)
}


# ---------------------------------------------------------------------------
# Low-level kernel wrappers (registered as torch custom ops)
# ---------------------------------------------------------------------------

def _rms_rw(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    _kernel["with_residual_with_weight"][input.ndim](
        input, weight, output, residual, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output, residual


def _fake_rms_rw(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input), residual


def _rms_r(
    input: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input)
    _kernel["with_residual_no_weight"][input.ndim](
        input, output, residual, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output, residual


def _fake_rms_r(
    input: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input), residual


def _rms_w(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    _kernel["no_residual_with_weight"][input.ndim](
        input, weight, output, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output


def _fake_rms_w(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.empty_like(input)


def _rms(
    input: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    output = torch.empty_like(input)
    _kernel["no_residual_no_weight"][input.ndim](
        input, output, eps, BLOCK_SIZE=input.shape[-1]
    )
    return output


def _fake_rms(
    input: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.empty_like(input)


direct_register_custom_op("nt_rms_rw", _rms_rw, fake_impl=_fake_rms_rw)
direct_register_custom_op("nt_rms_r", _rms_r, fake_impl=_fake_rms_r)
direct_register_custom_op("nt_rms_w", _rms_w, fake_impl=_fake_rms_w)
direct_register_custom_op("nt_rms", _rms, fake_impl=_fake_rms)


# ---------------------------------------------------------------------------
# High-level dispatch (calls registered custom ops)
# ---------------------------------------------------------------------------

def rms(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    residual: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    if weight is not None and residual is not None:
        return torch.ops.vllm.nt_rms_rw(
            input, weight.view((1,) * (input.ndim - 1) + (-1,)), residual, eps
        )
    elif weight is not None and residual is None:
        return torch.ops.vllm.nt_rms_w(
            input, weight.view((1,) * (input.ndim - 1) + (-1,)), eps
        )
    elif weight is None and residual is not None:
        return torch.ops.vllm.nt_rms_r(input, residual, eps)
    else:
        return torch.ops.vllm.nt_rms(input, eps)


# ---------------------------------------------------------------------------
# forward_oot adapter helpers (called by build_rms_forward_oot)
# ---------------------------------------------------------------------------

def rms_norm_helper(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    logger.info_once("\033[32mNT RMS is enabled.\033[0m")
    trace("rms_norm", shape=tuple(x.shape), dtype=x.dtype)
    record_hit("rms_norm")
    return rms(input=x, weight=weight, residual=None, eps=variance_epsilon)


def fused_add_rms_norm_helper(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info_once("\033[32mNT RMS is enabled.\033[0m")
    trace("fused_add_rms_norm", shape=tuple(x.shape), dtype=x.dtype)
    record_hit("fused_add_rms_norm")
    # rms() with residual returns (output, updated_residual) as new tensors.
    # vLLM's forward_oot contract expects x and residual to be mutated in-place.
    output, residual_out = rms(input=x, weight=weight, residual=residual, eps=variance_epsilon)
    x.copy_(output)
    residual.copy_(residual_out)
    return x, residual


# ---------------------------------------------------------------------------
# forward_oot builder — called by registry.py
# ---------------------------------------------------------------------------

def build_rms_forward_oot(original):
    """Replace RMSNorm.forward_oot to intercept all MLU dispatch paths.

    ``dispatch_forward`` on out-of-tree platforms binds ``_forward_method``
    to ``self.forward_oot``.  Patching this method ensures our kernels are
    reached regardless of whether the prior occupant was the base-class
    fallback or an MLU-specific override saved as ``original``.

    Falls back to ``original`` for unsupported edge cases
    (variance_size_override, no weight) so MLU correctness is preserved.
    """

    def forward_oot(self, x: torch.Tensor, residual: torch.Tensor | None = None):
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
