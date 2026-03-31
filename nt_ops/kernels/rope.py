"""Rotary position embedding kernel and torch dispatch, inlined from ntops."""
from __future__ import annotations

import functools

import ninetoothed
import torch
from ninetoothed import Tensor

from nt_ops.kernels.utils import _cached_make


# ---------------------------------------------------------------------------
# Ninetoothed kernel
# ---------------------------------------------------------------------------
def _arrangement(input, sin_table, cos_table, output, interleaved=True):
    emb_dim = input.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    def _arrange_input_or_output(tensor):
        tensor_arranged = tensor.tile(tile_shape, strides=strides, dilation=dilation)
        tensor_arranged = tensor_arranged.tile((1, 1, 1, -1))
        tensor_arranged.dtype = tensor_arranged.dtype.squeeze((0, 1, 2))
        tensor_arranged.dtype.dtype = tensor_arranged.dtype.dtype.squeeze((0, 1, 2))
        return tensor_arranged

    def _arrange_table(table):
        table_arranged = table.tile(tile_shape)
        table_arranged.dtype = table_arranged.dtype.squeeze((0, 1, 2))
        return table_arranged

    return (
        _arrange_input_or_output(input),
        _arrange_table(sin_table),
        _arrange_table(cos_table),
        _arrange_input_or_output(output),
    )


def _application(input, sin_table, cos_table, output):
    sin = sin_table
    cos = cos_table
    x0 = input[0]
    x1 = input[1]
    output[0] = x0 * cos - x1 * sin
    output[1] = x0 * sin + x1 * cos


def _premake(ndim, emb_dim=None, dtype=None, interleaved=True):
    arrangement_ = functools.partial(_arrangement, interleaved=interleaved)
    shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})
    tensors = (
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
        Tensor(ndim, dtype=dtype, shape_options=shape_options),
    )
    if emb_dim is not None:
        for tensor in tensors:
            tensor.shape = tensor.shape[:-1] + (emb_dim,)
    return arrangement_, _application, tensors


# ---------------------------------------------------------------------------
# Public torch-level entry point
# ---------------------------------------------------------------------------
def rotary_position_embedding(
    input: torch.Tensor,
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
    *,
    interleaved: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    output = input if inplace else torch.empty_like(input)

    batch_size, _, num_heads, _ = input.shape
    sin_table = sin_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)

    kernel = _cached_make(_premake, input.ndim, interleaved=interleaved, num_warps=1)
    kernel(input, sin_table, cos_table, output)

    return output
