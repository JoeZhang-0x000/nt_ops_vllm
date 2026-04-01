from __future__ import annotations

from nt_ops.registry import get_phase1_operator_matrix, get_profile


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
    assert specs[1].module_path == "vllm_mlu._mlu_ops"
    assert specs[1].attr_name == "active"
    assert "rope" in profile.fallback
    assert "gelu_and_mul" in profile.fallback
    assert "xielu" in profile.fallback


def test_phase1_operator_matrix_matches_profile_buckets():
    matrix = get_phase1_operator_matrix("qwen3_minimal_dense")

    assert matrix["must_have_nt_ops"] == [
        "rms_norm",
        "fused_add_rms_norm",
        "silu_and_mul",
    ]
    assert "rope" in matrix["backend_native_fallback"]
    assert "attention" in matrix["out_of_scope"]
