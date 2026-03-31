---
date: 2026-03-31
topic: mlu-first-platform-worker-architecture
---

# MLU-First Platform/Worker Architecture

## Problem Frame

`nt_ops_vllm` currently behaves like a worker-side patch package: callers inject `worker_cls="nt_ops.worker.NTVLLMWorker"`, the worker dynamically wraps a discovered backend worker, and runtime code monkey-patches operator entry points and some MLU-specific internals. That works as an integration tactic, but it treats MLU as an implementation detail rather than a first-class vLLM device.

Upstream vLLM already has a stronger architecture seam: platform plugins determine device identity and can select the appropriate worker class. `vllm-mlu` follows that seam by registering an MLU platform with `device_type = "mlu"` and choosing MLU worker implementations from platform config. The goal of this work is to realign `nt_ops_vllm` with that philosophy: make MLU a first-class device/worker path first, while keeping the door open for additional backends later. The intended value is lower integration friction for MLU enablement, clearer ownership of backend-specific behavior, and lower maintenance risk than the current wrapper-centric design.

## Requirements

- R1. The target architecture must treat MLU as a first-class vLLM device path, selected through upstream-style platform/device integration rather than requiring users to pass a custom `worker_cls` manually.
- R2. The implementation must provide a dedicated MLU worker integration point whose lifecycle is the primary place where NT Ops is installed or activated for MLU execution.
- R3. `nt_ops_vllm` must stop treating backend worker discovery and dynamic wrapping as the core abstraction for MLU support.
- R4. The design must follow upstream vLLM terminology and seams where possible: platform, device type, worker class, and backend selection.
- R5. NT Ops should remain a thin backend capability layer attached to the MLU path, not a new top-level device framework that duplicates vLLM’s existing platform model. In this phase, “thin” means NT Ops may attach operators and capability-specific patches during the chosen MLU worker lifecycle, but it must not own device selection or define a parallel plugin model.
- R6. Support for future devices must be preserved as an architectural direction, but generic multi-device abstractions inside `nt_ops_vllm` are explicitly deferred until at least one additional backend proves what is truly shared.
- R7. Any backend-specific patches that remain must be treated as targeted exceptions around the MLU path, not as the primary architecture.
- R8. This phase must define a migration posture for the current wrapper-based entry points: either explicitly deprecate manual `worker_cls` injection as a compatibility path, or declare it out of scope for this phase rather than leaving the transition implicit.

## Success Criteria

- Users can select MLU through the normal vLLM device/platform path rather than by manually injecting `nt_ops.worker.NTVLLMWorker` as the primary integration path.
- The main architectural story becomes "MLU platform/worker integration with NT Ops attached" rather than "NT Ops wraps arbitrary workers and patches runtime behavior."
- Planning and implementation can explain where MLU-specific behavior lives without inventing a premature generic plugin API.
- The resulting design documents the intended future extension seam without requiring the MLU solution to solve all future device abstraction upfront.
- The phase defines one primary ownership point for NT Ops activation in the MLU lifecycle and makes clear which residual patches, if any, remain outside that point.

## Scope Boundaries

- Do not design a generic cross-device plugin framework inside `nt_ops_vllm` in this phase.
- Do not require the first MLU redesign to solve every backend-specific hijack or upstream extensibility gap.
- Do not treat future device support as justification for keeping the current wrapper-based philosophy.
- Do not lock the project into MLU-only forever; the decision is to sequence generalization later, not to reject it.
- Do not let packaging or ownership debates expand the phase beyond the minimum deliverable: one upstream-shaped MLU device path with a defined NT Ops activation point.

## Key Decisions

- **MLU first, generic later**: The repo should first become correct for one real device seam before abstracting for many. This reduces abstraction risk and matches the evidence from upstream vLLM and `vllm-mlu`.
- **Use upstream seams, not custom ones**: vLLM already exposes platform/device/worker extension points. Reusing those is lower-risk than centering the design on `NTVLLMWorker`.
- **Keep NT Ops thin**: NT Ops should be a capability layer attached to the chosen backend path, not the owner of device selection itself.
- **Treat patches as exceptions**: Some MLU-specific patching may still be necessary, but that should be framed as backend-specific adaptation rather than the main architecture.
- **Prefer the smallest ownership surface**: If planning must choose between direct ownership in `nt_ops_vllm` and layering on `vllm-mlu`, prefer the option with the smallest new surface area in `nt_ops_vllm` that still satisfies R1-R8.

## Dependencies / Assumptions

- Upstream vLLM platform hooks appear sufficient to express `device='mlu'` selection and choose an MLU worker class, but planning must validate the minimum hook set needed for registration, worker selection, and NT Ops activation before locking implementation.
- `vllm-mlu` remains the strongest local reference for how an out-of-tree MLU backend should align with upstream vLLM.
- Some MLU behavior may still require backend-specific code outside the pure platform seam, as seen in `vllm-mlu`.

## Terminology

- **User-facing selection**: choosing `device='mlu'` or the equivalent normal vLLM device/platform path.
- **Platform registration**: the mechanism that makes MLU available as a first-class vLLM device.
- **Worker selection**: the platform-driven choice of the concrete MLU worker class.
- **NT Ops attachment**: the point where NT Ops operators and any bounded backend-specific adaptations are activated for the selected MLU worker path.

## Outstanding Questions

### Deferred to Planning

- [Affects R1][Technical] Should the platform registration and MLU worker live directly in `nt_ops_vllm`, or should `nt_ops_vllm` layer on top of `vllm-mlu` while changing the integration seam?
- [Affects R2][Needs research] What is the minimal worker lifecycle hook needed to install NT Ops cleanly for MLU without preserving the current dynamic worker-wrapper pattern?
- [Affects R7][Needs research] Which existing runtime patches are unavoidable backend adaptations, and which are artifacts of the current wrapper-centric architecture?
- [Affects R8][Technical] What migration posture is acceptable for existing manual `worker_cls`-based usage during the transition to the platform/device path?

## Next Steps

→ `/ce:plan` for structured implementation planning
