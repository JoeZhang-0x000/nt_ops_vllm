from __future__ import annotations

import os

import pytest
torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen3_basic_smoke_contract():
    model_path = os.environ.get("NT_OPS_VLLM_MODEL_PATH")
    if not model_path:
        pytest.skip("NT_OPS_VLLM_MODEL_PATH is not set")

    from vllm import LLM, SamplingParams

    import nt_ops

    nt_ops.install(process_scope="exclusive_qwen3")
    llm = LLM(model=model_path, enforce_eager=True)
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0.0, max_tokens=8),
    )

    report = nt_ops.get_capability_report()
    assert outputs
    assert report["status"] == "installed"
    assert report["hits"].get("rms_norm", 0) > 0
    assert report["hits"].get("rope", 0) > 0
    assert report["hits"].get("silu_and_mul", 0) > 0
