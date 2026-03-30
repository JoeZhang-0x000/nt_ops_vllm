This document describes the current correctness target for the first-phase
`qwen3_minimal_dense` profile.

## Scope

The profile only claims NT-backed correctness for:

- `RMSNorm`
- fused add + `RMSNorm` helper path
- `RoPE`
- `SiluAndMul`

It does not claim NT-backed correctness for:

- `attention`
- general `GEMM`
- `lm_head`
- `embedding`
- `logits_processor`
- `sampler`

Those components intentionally remain on vLLM fallbacks in this phase.

## Validation Layers

Correctness is checked in three layers:

1. Runtime installation tests
   These verify that patch application is explicit, transactional, and rollback-safe.

2. Operator matrix tests
   These compare the NT-backed adapters against vLLM or PyTorch references for:
   - `RMSNorm`
   - fused add + `RMSNorm`
   - `RoPE`

3. Qwen3 smoke test
   When `NT_OPS_VLLM_MODEL_PATH` is provided in a CUDA environment, the basic smoke
   test verifies that `LLM.generate()` completes and that the NT capability report
   records hits for the intended patched components.

## Manual Check

To manually validate the profile:

```bash
export NT_OPS_VLLM_MODEL_PATH=/path/to/Qwen3-0.6B
python examples/basic.py
```

At the end of the run, inspect the printed capability report. A valid first-phase
run should show non-zero hits for NT-backed Qwen3 components and keep fallback
components unpatched.
