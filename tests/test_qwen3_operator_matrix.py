from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from nt_ops.rope import build_rotary_forward_oot
from nt_ops.rms import fused_add_rms_norm_helper, rms_norm_helper
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_rms_norm_helper_matches_vllm_forward_static():
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)

    output = rms_norm_helper(x, weight, 1e-5)
    reference = RMSNorm.forward_static(
        x,
        1e-5,
        hidden_size=64,
        orig_dtype=x.dtype,
        weight=weight,
    )

    assert torch.allclose(output, reference, rtol=0.01, atol=0.01)


def test_fused_add_rms_norm_helper_matches_vllm_forward_static():
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    residual = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)

    output, residual_out = fused_add_rms_norm_helper(
        x.clone(),
        residual.clone(),
        weight,
        1e-5,
    )
    reference, reference_residual = RMSNorm.forward_static(
        x,
        1e-5,
        hidden_size=64,
        orig_dtype=x.dtype,
        weight=weight,
        residual=residual,
    )

    assert torch.allclose(output, reference, rtol=0.01, atol=0.01)
    assert torch.allclose(residual_out, reference_residual, rtol=0.01, atol=0.01)


def test_rope_forward_oot_adapter_matches_vllm_forward_static():
    rotary = RotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float16,
    ).cuda()
    patched_forward = build_rotary_forward_oot(RotaryEmbedding.forward_oot)

    positions = torch.arange(0, 6, device="cuda", dtype=torch.long)
    query = torch.randn(6, 16, device="cuda", dtype=torch.float16)
    key = torch.randn(6, 16, device="cuda", dtype=torch.float16)

    patched_q, patched_k = patched_forward(
        rotary,
        positions,
        query.clone(),
        key.clone(),
    )
    cos_sin_cache = rotary._match_cos_sin_cache_dtype(query)
    reference_q, reference_k = RotaryEmbedding.forward_static(
        positions,
        query,
        key,
        rotary.head_size,
        rotary.rotary_dim,
        cos_sin_cache,
        rotary.is_neox_style,
    )

    assert torch.allclose(patched_q, reference_q, rtol=0.001, atol=0.001)
    assert torch.allclose(patched_k, reference_k, rtol=0.001, atol=0.001)
