from __future__ import annotations

from nt_ops.registry import get_profile


def test_qwen3_minimal_dense_profile_matches_qwen3_runtime_path():
    profile, specs = get_profile("qwen3_minimal_dense")

    assert profile.enabled == (
        "rms_norm",
        "fused_add_rms_norm",
        "silu_and_mul",
    )
    assert tuple(spec.patch_id for spec in specs) == (
        "rms_norm",
        "silu_and_mul",
    )
    assert "rope" in profile.fallback
    assert "gelu_and_mul" in profile.fallback
    assert "xielu" in profile.fallback
