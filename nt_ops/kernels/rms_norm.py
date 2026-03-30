"""RMSNorm kernel and torch dispatch, inlined from ntops.

Includes MLU grid-size fix: when outer_numel > 65535 (MLU hardware limit),
fall back to GROUPED_ROWS which divides the grid by ROWS_PER_PROGRAM.
"""
from __future__ import annotations

import enum
import functools
import math

import ninetoothed
import ninetoothed.language as ntl
import torch
from ninetoothed import Tensor

from nt_ops.kernels.utils import _cached_make


# ---------------------------------------------------------------------------
# Kernel constants
# ---------------------------------------------------------------------------
ROWS_PER_PROGRAM = 8
BLOCK_SIZE = 128

# Conservative hardware grid limit (MLU caps at 65535; CUDA is much larger).
_MAX_GRID_SIZE = 65535


# ---------------------------------------------------------------------------
# Dispatch variant
# ---------------------------------------------------------------------------
class RMSNormVariant(enum.Enum):
    GENERIC = "generic"
    GROUPED_ROWS = "grouped_rows"
    ROWWISE = "rowwise"


# ---------------------------------------------------------------------------
# Arrangement helpers (adapted from ntops.kernels.reduction)
# ---------------------------------------------------------------------------
def _reduction_arrangement(*tensors, dim, block_size=None):
    dims = dim
    if isinstance(dims, int):
        dims = (dims,)
    if block_size is None:
        block_size = ninetoothed.block_size()

    ndim = max(tensor.ndim for tensor in tensors)
    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)
    dims = tuple(d if d >= 0 else d + ndim for d in dims)
    non_target_dims = tuple(i for i in range(ndim) if i not in dims)

    def _arrange(tensor):
        arranged = tensor.permute(non_target_dims + dims)
        arranged = arranged.flatten(start_dim=-len(dims))
        inner_block_shape = tuple(1 for _ in non_target_dims) + (block_size,)
        outer_block_shape = tuple(1 for _ in non_target_dims) + (-1,)
        non_target_dim_indices = tuple(range(len(non_target_dims)))
        arranged = arranged.tile(inner_block_shape)
        arranged = arranged.tile(outer_block_shape)
        arranged.dtype = arranged.dtype.squeeze(non_target_dim_indices)
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze(non_target_dim_indices)
        return arranged

    return tuple(_arrange(t) if t.ndim != 0 else t for t in tensors)


# ---------------------------------------------------------------------------
# GENERIC fallback (reduction-based, works for any shape)
# ---------------------------------------------------------------------------
def _application_generic(input, weight, eps, output, num_normalized_elements):
    _rms = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for i in range(input.shape[0]):
        input_i = ntl.cast(input[i], ntl.float32)
        _rms += input_i * input_i
    rms = ntl.sqrt(ntl.sum(_rms) / num_normalized_elements + eps)
    for i in range(input.shape[0]):
        output[i] = input[i] / rms * weight[i]


def _premake_generic(ndim, num_normalized_dims, input_dtype=None, weight_dtype=None, output_dtype=None, block_size=None):
    dims = tuple(-(d + 1) for d in range(num_normalized_dims))
    arrangement_ = functools.partial(_reduction_arrangement, dim=dims, block_size=block_size)
    tensors = (
        Tensor(ndim, other=0, dtype=input_dtype),
        Tensor(ndim, dtype=weight_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(ndim, dtype=output_dtype),
        Tensor(0, dtype=ninetoothed.int64),
    )
    return arrangement_, _application_generic, tensors


# ---------------------------------------------------------------------------
# ROWWISE variant — one program per row, tiles over hidden dim
# ---------------------------------------------------------------------------
def _arrangement_rowwise(input, weight, eps, output, normalized_numel):
    arranged_input = input.tile((1, BLOCK_SIZE))
    arranged_output = output.tile((1, BLOCK_SIZE))
    arranged_weight = weight[None, None, :]
    return arranged_input, arranged_weight, eps, arranged_output, normalized_numel


def _application_rowwise(input, weight, eps, output, normalized_numel):
    _sum_sq = ntl.zeros((1,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            val = ntl.cast(input[i, j], ntl.float32)
            _sum_sq[0] += val * val
    rms = ntl.sqrt(_sum_sq[0] / normalized_numel + eps)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i, j] = input[i, j] / rms * weight[0, 0, j]


def _premake_rowwise(input_dtype=None, normalized_numel=None):
    tensors = (
        Tensor(2, other=0, dtype=input_dtype),
        Tensor(1, dtype=input_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(2, dtype=input_dtype),
        Tensor(0, dtype=ninetoothed.int64),
    )
    return _arrangement_rowwise, _application_rowwise, tensors


# ---------------------------------------------------------------------------
# GROUPED_ROWS variant — ROWS_PER_PROGRAM rows per program, reduces grid size
# ---------------------------------------------------------------------------
def _arrangement_grouped_rows(input, weight, eps, output, normalized_numel):
    arranged_input = input.tile((ROWS_PER_PROGRAM, BLOCK_SIZE))
    arranged_output = output.tile((ROWS_PER_PROGRAM, BLOCK_SIZE))
    arranged_weight = weight[None, None, :]
    return arranged_input, arranged_weight, eps, arranged_output, normalized_numel


def _application_grouped_rows(input, weight, eps, output, normalized_numel):
    _sum_sq = ntl.zeros((ROWS_PER_PROGRAM,), dtype=ntl.float32)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            val = ntl.cast(input[i, j], ntl.float32)
            _sum_sq[i] += val * val
    rms = ntl.sqrt(_sum_sq / normalized_numel + eps)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i, j] = input[i, j] / rms[i] * weight[0, 0, j]


def _premake_grouped_rows(input_dtype=None, normalized_numel=None):
    tensors = (
        Tensor(2, other=0, dtype=input_dtype),
        Tensor(1, dtype=input_dtype),
        Tensor(0, dtype=ninetoothed.float64),
        Tensor(2, dtype=input_dtype),
        Tensor(0, dtype=ninetoothed.int64),
    )
    return _arrangement_grouped_rows, _application_grouped_rows, tensors


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------
def _can_view_as_2d_contiguous(input, normalized_numel):
    if input.dim() < 2:
        return False, None
    outer_numel = input.numel() // normalized_numel
    if outer_numel * normalized_numel != input.numel():
        return False, None
    if input.stride(-1) != 1:
        return False, None
    for dim in range(input.dim() - 1):
        if input.size(dim) > 1 and input.stride(dim) != input.stride(dim + 1) * input.size(dim + 1):
            return False, None
    return True, outer_numel


def _select_variant(input, normalized_shape, weight):
    if len(normalized_shape) != 1:
        return RMSNormVariant.GENERIC

    normalized_numel = normalized_shape[0]
    can_view_2d, outer_numel = _can_view_as_2d_contiguous(input, normalized_numel)
    if not can_view_2d:
        return RMSNormVariant.GENERIC
    if input.dtype not in (torch.float16, torch.bfloat16):
        return RMSNormVariant.GENERIC
    if weight is not None and weight.shape != (normalized_numel,):
        return RMSNormVariant.GENERIC

    # Use GROUPED_ROWS when outer_numel is large enough for perf OR when
    # ROWWISE grid would exceed hardware limits (e.g. MLU caps at 65535).
    if outer_numel >= 256 and normalized_numel <= 2048:
        return RMSNormVariant.GROUPED_ROWS
    if outer_numel > _MAX_GRID_SIZE:
        return RMSNormVariant.GROUPED_ROWS
    return RMSNormVariant.ROWWISE


# ---------------------------------------------------------------------------
# Public torch-level entry point
# ---------------------------------------------------------------------------
def rms_norm(input: torch.Tensor, normalized_shape, weight=None, eps=None) -> torch.Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    normalized_shape = tuple(normalized_shape)
    normalized_numel = math.prod(normalized_shape)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)
    variant = _select_variant(input, normalized_shape, weight)

    if variant == RMSNormVariant.GROUPED_ROWS:
        compact_weight = weight if weight is not None else torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
        kernel = _cached_make(_premake_grouped_rows, input.dtype)
        kernel(input, compact_weight, eps, output, normalized_numel)
    elif variant == RMSNormVariant.ROWWISE:
        compact_weight = weight if weight is not None else torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
        kernel = _cached_make(_premake_rowwise, input.dtype)
        kernel(input, compact_weight, eps, output, normalized_numel)
    else:
        expanded_weight = weight.expand_as(input) if weight is not None else torch.ones_like(input)
        kernel = _cached_make(_premake_generic, input.ndim, len(normalized_shape))
        kernel(input, expanded_weight, eps, output, normalized_numel)

    return output
