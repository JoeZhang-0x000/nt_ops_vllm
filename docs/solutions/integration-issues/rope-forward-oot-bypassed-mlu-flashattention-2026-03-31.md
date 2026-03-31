---
title: "RotaryEmbedding.forward_oot Bypassed by MLU FlashAttentionBackend"
date: 2026-03-31
problem_type: integration_issue
component: tooling
root_cause: config_error
resolution_type: config_change
severity: high
project: nt_ops_vllm
vllm_version: "0.11.2"
platform: mlu-cambricon
tags:
  - vllm
  - mlu
  - monkey-patching
  - rope
  - flashattention
  - kernel-fusion
  - forward_oot
---

# RotaryEmbedding.forward_oot Bypassed by MLU FlashAttentionBackend

## Problem

Patching `RotaryEmbedding.forward_oot` to intercept RoPE computation has no
effect when running vLLM with the vllm_mlu (MLU/Cambricon) backend. The patch
installs successfully, but the patched method is never called during inference.

## Symptoms

- Install log confirms the patch was applied: `Installed nt_ops profile qwen3_minimal_dense with patches: rms_norm, silu_and_mul, rope`
- Expected log `NT RoPE is enabled` never appears during inference
- RMSNorm and SiluAndMul patches work correctly (their `forward_oot` logs fire normally)
- No errors raised; the system silently takes the MLU-fused path

## What Didn't Work

1. **Checking subclass overrides** — inspected every `RotaryEmbedding` subclass in
   `vllm.model_executor.layers.rotary_embedding`; none override `forward_oot`, so the
   base class patch should propagate via MRO. It does — but the method is still
   never called.

2. **Checking guard conditions** — Qwen3-0.6B: `head_dim=128` (even), `is_neox_style=True`,
   `rope_scaling=None`. None of the fallback conditions inside `forward_oot` should trigger.

3. **Adding a diagnostic log at the very top of `forward_oot` (before any guard)** —
   zero calls observed during a complete Qwen3-0.6B inference run. This proves the
   method body is never entered, not that it takes a wrong branch.

## Solution

Remove the `rope` PatchSpec from the active profile and move `rope` to `fallback`.

**`nt_ops/registry.py` — before:**
```python
QWEN3_MINIMAL_DENSE_PROFILE = CapabilityProfile(
    name="qwen3_minimal_dense",
    enabled=("rms_norm", "fused_add_rms_norm", "silu_and_mul", "rope"),
    fallback=("embedding", "logits_processor", "sampler"),
    disabled=("linear", "attention", "lm_head", "flash_attn"),
)

# ... inside _PROFILES:
PatchSpec(
    patch_id="rope",
    module_path="vllm.model_executor.layers.rotary_embedding.base",
    object_name="RotaryEmbedding",
    attr_name="forward_oot",
    builder=_rope_forward_oot_builder,
),
```

**After:**
```python
QWEN3_MINIMAL_DENSE_PROFILE = CapabilityProfile(
    name="qwen3_minimal_dense",
    enabled=("rms_norm", "fused_add_rms_norm", "silu_and_mul"),
    # rope: RotaryEmbedding.forward_oot is never called on MLU — the MLU
    # FlashAttentionBackend applies RoPE internally as a fused op. Patching
    # forward_oot has no effect; leave rope in fallback until a viable
    # hook point is identified inside the MLU attention backend.
    fallback=("rope", "embedding", "logits_processor", "sampler"),
    disabled=("linear", "attention", "lm_head", "flash_attn"),
)

# rope PatchSpec removed from _PROFILES entirely.
# _rope_forward_oot_builder is kept for future backends where forward_oot is viable.
```

## Why This Works

vllm_mlu selects `FlashAttentionBackend` (logged at startup: `[MLU-V1] Select FlashAttentionBackend`). This backend fuses RoPE + attention into a single MLU kernel. The `RotaryEmbedding` CustomOp is present in the model graph but its `forward_oot` is never dispatched — vLLM routes RoPE through a different internal mechanism.

Additional evidence: `splitting_ops: ['vllm.rope_forward']` in the compilation config shows that vLLM treats RoPE as a sub-operation of the attention op, not a standalone dispatch target.

By contrast, `RMSNorm` and `SiluAndMul` are **not** fused into a larger kernel; they remain independent CustomOps whose `forward_oot` is dispatched normally. This is why the same patching approach works for them but not for RoPE on this backend.

## Prevention

### 1. Confirm `forward_oot` is actually called before writing a patch

Place a log at the **very first line** of `forward_oot` — before any guard or
conditional logic. If it never fires during a real inference run, the hook point
is dead on this backend.

```python
def forward_oot(self, ...):
    logger.info_once("DIAGNOSTIC: forward_oot called for %s", type(self).__name__)
    # ... rest of implementation
```

### 2. Read the backend selection log at startup

```
[MLU-V1] Select FlashAttentionBackend.
```

If a fused attention backend is selected, assume RoPE (and potentially other
position-encoding ops) is handled internally. Only backends that execute RoPE
as a standalone CustomOp will call `RotaryEmbedding.forward_oot`.

### 3. Check `splitting_ops` in the compilation config

`splitting_ops: ['vllm.rope_forward']` means vLLM treats `rope_forward` as a
split boundary in the computation graph — a strong signal that RoPE is a
sub-operation rather than an independent dispatch target.

### 4. Probe with `NT_OPS_DEBUG=1` before concluding

The `NT_OPS_DEBUG=1` env var (added to this project) prints a log on every
operator call. If no `[NT_DEBUG] rope` line appears, the operator is not
being exercised — even if the patch installed without errors.

### 5. Document per-backend hook validity in the profile

Add a comment to each PatchSpec explaining which backends it is valid for.
This prevents future profiles from silently enabling patches that won't fire.

## Related

- `docs/plans/2026-03-31-001-refactor-hybrid-dispatch-main-kernels-plan.md` — implementation plan that surfaced this issue
- `docs/brainstorms/2026-03-30-hybrid-dispatch-infrastructure-requirements.md` — original requirements
- `nt_ops/registry.py` — profile configuration and PatchSpec definitions
- `nt_ops/rope.py` — rope kernel and `build_rotary_forward_oot` (retained for future use)
- `nt_ops/debug.py` — `NT_OPS_DEBUG=1` per-call trace logging
