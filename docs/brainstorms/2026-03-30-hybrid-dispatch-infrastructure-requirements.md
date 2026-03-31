---
date: 2026-03-30
topic: hybrid-dispatch-infrastructure
---

# Hybrid Dispatch Infrastructure: Main Kernels + Refactored Patch System

## Problem Frame

The `refactor/nt-ops-qwen3-minimal-dense-path` branch introduced two improvements simultaneously:
1. A new dispatch infrastructure (runtime, registry, capabilities, worker, profile system) with the correct patch target (`forward_oot`) for MLU.
2. Inlined Triton kernels in `nt_ops/kernels/` meant to replace the external ninetoothed dependency.

The inlined kernels have bugs on MLU hardware (grid-size overflow, shape dispatch issues). `main` works on MLU because it uses the original ninetoothed kernels, but patches the wrong method (`forward` instead of `forward_oot`), which is fragile for the MLU vllm dispatch path.

The goal of this work is to combine the two correctly: keep the working ninetoothed kernel implementations from `main`, and adopt the new infrastructure and `forward_oot` patch target from the current branch.

## Requirements

- R1. New branch is created from `main` (not from the broken refactor branch).
- R2. The new infrastructure files are brought in from the current branch unchanged: `runtime.py`, `registry.py`, `capabilities.py`, `worker.py`.
- R3. `nt_ops/kernels/` directory is **not** included — inlined Triton kernels are dropped entirely.
- R4. `nt_ops/rms.py` retains all ninetoothed kernel classes and the `rms()` dispatch function from `main`. The old `rms_forward` (patches `forward`) is removed. A new `build_rms_forward_oot(original)` adapter is added that wraps the existing `rms()` function and matches the `forward_oot` signature expected by `registry.py`.
- R5. `nt_ops/rope.py` retains the ninetoothed rope kernel from `main`. A new `build_rotary_forward_oot(original)` adapter is added, wrapping the existing kernel and matching the `forward_oot` signature expected by `registry.py`.
- R6. `nt_ops/activation.py` is kept as-is from `main` — `registry.py` only references `activation.silu_and_mul_forward`, which already exists.
- R7. `nt_ops/monkey_patch.py` is updated to the thin shim from the current branch (deprecated `apply_monkey_patches` delegates to `install(profile='qwen3_minimal_dense', ...)`).
- R8. `registry.py` builder functions (`_rms_forward_oot_builder`, `_rope_forward_oot_builder`, `_activation_silu_builder`) call the adapter functions in `rms.py` / `rope.py` / `activation.py` — no changes needed if R4–R6 are met.
- R9. The `forward_oot` patch target is used throughout (not `forward`), ensuring correct dispatch on MLU via vllm's `dispatch_forward` path.

## Success Criteria

- The new branch passes the same smoke tests as `main` (conceptually — no MLU hardware available locally, so tests are skipped).
- `nt_ops.install(profile='qwen3_minimal_dense')` succeeds without import errors.
- `apply_monkey_patches()` still works as a deprecated shim.
- No reference to `nt_ops.kernels.*` remains anywhere (R3 enforced).
- `rms.py` and `rope.py` have no `rms_forward` / old forward-patching functions.

## Scope Boundaries

- Do not fix or retain the inlined Triton kernels from `nt_ops/kernels/` — drop entirely.
- Do not change the kernel math or ninetoothed arrangements from `main`.
- Do not add new operators or patch new vllm modules beyond what `registry.py` already defines.
- Skip MLU hardware tests — no test environment available locally.

## Key Decisions

- **Base from `main`, not refactor branch**: Avoids carrying over any broken kernel code; cleanest starting point.
- **Drop `nt_ops/kernels/` entirely**: The inlined Triton kernels are the root cause of MLU failures; the ninetoothed kernels from `main` are proven.
- **Add adapter functions, not rewrite**: `build_rms_forward_oot` and `build_rotary_forward_oot` are thin wrappers around existing `main` implementations — minimal new code, low risk.
- **Patch `forward_oot` not `forward`**: This is the correct vllm MLU dispatch hook. `dispatch_forward` on out-of-tree platforms binds `_forward_method` to `forward_oot`.

## Dependencies / Assumptions

- `ninetoothed` package is installed in the MLU environment (same as `main` requires).
- vllm's `dispatch_forward` on MLU resolves `_forward_method` to `forward_oot` (established by current branch's investigation).
- `capabilities.py` `record_hit()` calls in `rms.py` / `rope.py` are fine to add or keep — they're called inside the new adapter functions.

## Outstanding Questions

### Deferred to Planning

- [Affects R4][Needs research] Does `rms.py`'s `direct_register_custom_op` registration (torch custom ops) need to remain, or can it be simplified now that we no longer go through `forward` patching? (Likely keep as-is for safety.)
- [Affects R5][Needs research] Does `rope.py` on `main` already have a structure compatible with wrapping into `build_rotary_forward_oot`, or does it need the cos/sin cache logic from the current branch's version?

## Next Steps

→ `/ce:plan` for structured implementation planning
