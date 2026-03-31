"""Import-level coverage for the vendored ntops package.

These tests verify that the vendored copy is complete and all public symbols
are importable. They do NOT execute kernels (which requires ninetoothed JIT
compilation and hardware), so they run in any environment.
"""
from __future__ import annotations

import importlib

import pytest


ninetoothed = pytest.importorskip(
    "ninetoothed",
    reason="ninetoothed not installed — skipping ntops import tests",
)


def test_ntops_top_level_importable():
    import ntops  # noqa: F401


def test_ntops_kernels_subpackage_importable():
    import ntops.kernels  # noqa: F401


def test_ntops_torch_subpackage_importable():
    import ntops.torch  # noqa: F401


# Representative sample of kernel modules covering arithmetic, activations,
# normalization, attention, and position encoding.
_KERNEL_MODULES = [
    "ntops.kernels.rms_norm",
    "ntops.kernels.layer_norm",
    "ntops.kernels.rotary_position_embedding",
    "ntops.kernels.scaled_dot_product_attention",
    "ntops.kernels.softmax",
    "ntops.kernels.gelu",
    "ntops.kernels.silu",
    "ntops.kernels.mm",
    "ntops.kernels.bmm",
    "ntops.kernels.addmm",
    "ntops.kernels.abs",
    "ntops.kernels.add",
    "ntops.kernels.mul",
    "ntops.kernels.relu",
    "ntops.kernels.dropout",
    "ntops.kernels.conv2d",
    "ntops.kernels.max_pool2d",
    "ntops.kernels.avg_pool2d",
]

_TORCH_MODULES = [
    "ntops.torch.rms_norm",
    "ntops.torch.layer_norm",
    "ntops.torch.rotary_position_embedding",
    "ntops.torch.scaled_dot_product_attention",
    "ntops.torch.softmax",
    "ntops.torch.gelu",
    "ntops.torch.silu",
    "ntops.torch.mm",
    "ntops.torch.bmm",
    "ntops.torch.addmm",
    "ntops.torch.matmul",
    "ntops.torch.abs",
    "ntops.torch.add",
    "ntops.torch.mul",
    "ntops.torch.relu",
    "ntops.torch.dropout",
]


@pytest.mark.parametrize("module_name", _KERNEL_MODULES)
def test_kernel_module_importable(module_name: str):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _TORCH_MODULES)
def test_torch_module_importable(module_name: str):
    importlib.import_module(module_name)
