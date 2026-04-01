---
date: 2026-04-01
topic: minimal-ntops-mlu-backend-phase1
---

# Minimal NT Ops MLU Backend Phase 1

## Problem Frame

`nt_ops_vllm` currently behaves as an operator overlay on top of `vllm-mlu`. That is enough to prove selective operator replacement, but it does not yet provide a clean product story for a user who wants an nt-ops-enabled MLU path with a first-class backend identity. The long-term architectural question is whether `nt_ops_vllm` should remain a thin augmentation of `vllm-mlu` or grow into a standalone hardware backend.

The repo evidence suggests those two goals should be separated. A fully standalone nt-ops device backend would require ownership of platform behavior, worker/runtime behavior, attention backend behavior, cache/runtime assumptions, and packaging/distribution constraints. That is a larger problem than the current codebase is ready to solve in one phase. Phase 1 should therefore aim at a thinner, lower-risk outcome: prove a minimal nt-ops-enabled MLU execution path on top of the existing `vllm-mlu` backend while keeping the door open to a future carve-out. The value of phase 1 is not raw performance or backend independence; it is establishing a correct, testable, product-facing nt-ops-enabled MLU path with honest ownership boundaries.

## Requirements

- R1. Phase 1 must lock in a product-facing architecture of **NT Ops on top of `vllm-mlu`**, not a fully standalone nt-ops hardware backend.
- R2. The phase-1 backend story must be honest about ownership: `vllm-mlu` continues to own MLU platform/runtime bring-up, while `nt_ops_vllm` owns nt-ops activation, worker-time patch attachment, capability reporting, and the minimum nt-ops-facing plugin surface required to expose that path.
- R3. Phase 1 must target only **single-card bf16 inference**.
- R4. Phase 1 must explicitly exclude quantization, communication, Ray/distributed support, and performance optimization as goals.
- R5. Phase 1 must define a concrete operator matrix that classifies operators into: must-have nt-ops substitutions, backend-native fallbacks that remain acceptable, and explicit out-of-scope operators. Unsupported or impractical pieces may continue to rely on existing backend behavior from `vllm-mlu`.
- R6. Phase 1 must not promise a full attention/backend rewrite. Attention remains in scope only to the extent needed to make the thin nt-ops-on-MLU path work correctly.
- R7. Phase 1 must preserve the current runtime truth that vLLM still expects block/page-oriented KV cache contracts; it must not require replacing the paged KV runtime model.
- R8. The architecture should preserve a credible path to a future staged carve-out, where more `vllm-mlu` responsibilities are replaced only after the minimal path is stable and understood.
- R9. Packaging and installation goals must be stated conservatively. The phase-1 backend should optimize for a reliable install story on MLU machines, but must not assume that a true `vllm-cpu` base plus one plugin package is already realistic.
- R10. If existing upstream plugin seams prove insufficient for the thin path, phase 1 must narrow or stop rather than expanding into standalone-backend ownership by default.

## Success Criteria

- Planning can describe phase 1 as a **thin nt-ops-enabled MLU backend path** rather than as a full custom hardware backend.
- The doc set and future plan distinguish clearly between what `vllm-mlu` owns and what `nt_ops_vllm` owns.
- The scope is narrow enough that implementation can focus on correctness and ownership boundaries rather than premature independence.
- The resulting design leaves room for a later staged carve-out if the thin path proves valuable and stable.
- Phase 1 defines a minimum viable nt-ops surface that includes plugin activation, worker-time nt-ops attachment, capability reporting, and at least one supported operator subset on the target single-card bf16 path.
- Planning can identify an explicit operator matrix and fallback policy without inventing phase-1 behavior.

## Scope Boundaries

- Do not plan a fully standalone nt-ops MLU backend in this phase.
- Do not plan for quantization, communication, distributed executors, Ray integration, or multi-card support.
- Do not require replacing vLLM’s block/page KV cache runtime assumptions.
- Do not treat simplified packaging as proof that `vllm-mlu` can already be removed from the runtime stack.
- Do not expand phase 1 into “all operators must be owned by nt_ops” if some critical paths still need backend-native behavior.
- Do not respond to a thin-path seam limitation by silently expanding phase 1 into broader backend ownership; that requires an explicit follow-up decision.

## Key Decisions

- **Thin on `vllm-mlu` first**: phase 1 should optimize for a minimal, stable nt-ops-enabled MLU path on top of the existing backend rather than for immediate independence.
- **Separate backend direction from packaging ambition**: installation simplicity is a valid product goal, but it must not force an unrealistic backend boundary in phase 1.
- **Treat attention carefully**: replacing the existing paged-attention kernel is a different question from replacing the paged KV runtime contract. Phase 1 should not conflate those.
- **Use staged ownership growth**: only promote more responsibilities from `vllm-mlu` into `nt_ops_vllm` when the code proves they are necessary to ship the thin path and bounded enough to own without recreating a full backend.

## Dependencies / Assumptions

- `vllm-mlu` remains the active MLU hardware backend in phase 1.
- Upstream vLLM plugin seams appear sufficient for `nt_ops_vllm` to present a first-class nt-ops-aware MLU path without owning the entire backend stack, but planning must verify the exact required seams for plugin activation, worker-time attach, and capability reporting before implementation is locked.
- Some unsupported or fused paths will continue to require backend-native handling rather than nt-ops replacements in phase 1.

## Outstanding Questions

### Deferred to Planning

- [Affects R2][Technical] What is the smallest ownership surface `nt_ops_vllm` should expose in phase 1 beyond plugin activation, worker attach, and capability reporting?
- [Affects R5][Needs research] What is the phase-1 operator matrix: must-have nt-ops substitutions, backend-native fallbacks, and explicit out-of-scope operators?
- [Affects R6][Technical] Does any attention-related behavior require phase-1 ownership beyond preserving the current backend-native path?
- [Affects R9][Technical] What packaging/install story is realistic for phase 1 without misleading users into thinking `vllm-mlu` is no longer required?

## Next Steps

→ `/ce:plan` for structured implementation planning
