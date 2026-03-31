---
title: "refactor: MLU-first platform and worker integration"
type: refactor
status: active
date: 2026-03-31
origin: docs/brainstorms/2026-03-31-mlu-first-platform-worker-architecture-requirements.md
---

# refactor: MLU-first platform and worker integration

## Overview

This plan replaces the current wrapper-centric `worker_cls="nt_ops.worker.NTVLLMWorker"` integration with an upstream-shaped MLU path: MLU is selected through a platform/device seam, the chosen MLU worker owns NT Ops attachment during worker startup, and NT Ops remains a thin capability layer rather than the owner of device selection.

## Problem Frame

Today `nt_ops_vllm` activates by dynamically wrapping whichever MLU worker it can find and installing patches inside `NTVLLMWorker.__init__`. That works as a local hack, but it makes MLU look like an implementation detail, forces examples to inject `worker_cls` manually, and leaves correctness defined at patch-install time rather than at platform selection, worker construction, and real MLU execution time. The origin requirements explicitly redirect the architecture toward an MLU-first platform/worker path with generic multi-device abstraction deferred until later. (see origin: docs/brainstorms/2026-03-31-mlu-first-platform-worker-architecture-requirements.md)

## Requirements Trace

- R1. Select MLU through an upstream-style platform/device path instead of manual `worker_cls` injection.
- R2. Provide a dedicated MLU worker integration point whose lifecycle is the primary NT Ops attachment point.
- R3. Retire dynamic backend worker discovery/wrapping as the core MLU abstraction.
- R4. Follow upstream vLLM seams: platform, device type, worker class, backend selection.
- R5. Keep NT Ops thin: it may attach operators and bounded patches, but must not own device selection.
- R6. Preserve a future extension seam without introducing a generic multi-device framework now.
- R7. Treat backend-specific patches as explicit exceptions around the MLU path.
- R8. Define migration posture for existing wrapper-based entry points.

## Scope Boundaries

- Do not design a generic cross-device plugin framework in this phase.
- Do not reimplement `vllm_mlu` device runtime, memory management, communicator setup, or backend internals from scratch.
- Do not expand the phase into solving every fused-backend hook gap; known non-viable hooks such as RoPE under MLU `FlashAttentionBackend` remain fallback behavior.
- Do not remove all compatibility shims immediately if a narrow deprecation path keeps migration lower-risk.

## Context & Research

### Relevant Code and Patterns

- `nt_ops/worker.py`: current wrapper-centric integration. `NTVLLMWorker` subclasses a dynamically resolved base worker and calls `install(process_scope=REQUIRED_PROCESS_SCOPE)` in `__init__`.
- `nt_ops/runtime.py`: transactional attach/rollback state machine. This is the right home for NT Ops install semantics, but not for device selection.
- `nt_ops/registry.py`: capability profile and patch inventory. Today it patches `RMSNorm.forward_oot` and `vllm_mlu._mlu_ops.active`, with RoPE explicitly left in fallback.
- `nt_ops/mlu_dispatch.py`: example of a bounded MLU-specific exception; this stays valuable even after the platform/worker redesign.
- `nt_ops/vllm_utils.py`: current report retrieval path through `collective_rpc("get_nt_ops_report")` or `model_executor.execute_method("get_nt_ops_report")`.
- `examples/basic.py` and `examples/benchmark_throughput.py`: current user-facing proof that the repo still depends on manual `worker_cls` injection.
- `setup.py`: current package surface is minimal and has no vLLM plugin entry points yet.
- `tests/test_runtime_install.py`, `tests/test_registry_profile.py`, `tests/test_mlu_dispatch.py`, `tests/test_vllm_utils.py`: current non-hardware coverage. Useful, but they stop at patch install and helper behavior rather than platform-driven worker selection.

### Institutional Learnings

- Always patch `forward_oot`, not `forward`, on out-of-tree MLU paths; `forward` can be silently bypassed. (see origin and `docs/plans/2026-03-31-001-refactor-hybrid-dispatch-main-kernels-plan.md`)
- Patch timing matters: class-level patching must happen before the relevant worker/model path binds the callable.
- Patch installation success is not proof of runtime viability. `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md` shows that a patch can install cleanly and still be dead under fused `FlashAttentionBackend`.
- RoPE stays in fallback on MLU until a real backend hook exists; the redesign must preserve backend-specific hook validity as first-class planning context.

### External References

- Local adjacent references only: `/Users/bytedance/Desktop/nt_workspace/vllm-mlu` and `/Users/bytedance/Desktop/nt_workspace/vllm` were sufficient to establish the seam; no additional external research is needed.

## Key Technical Decisions

- **Direct but thin ownership in `nt_ops_vllm`**: add an explicit nt-ops MLU platform/worker integration in this repo, but keep it thin by building on the existing `vllm_mlu` worker/runtime path rather than reimplementing backend runtime concerns.
- **Worker lifecycle remains the attachment point**: NT Ops still attaches before model execution, but the primary hook moves from a generic wrapper worker to an explicit MLU worker path selected by the platform seam.
- **Compatibility shim, not primary path**: keep `nt_ops.worker.NTVLLMWorker` temporarily as a deprecated bridge to the new worker path so existing scripts do not hard-fail immediately, but remove it from examples and docs as the recommended entry point.
- **Patch inventory stays explicit**: `registry.py` remains the source of truth for what is enabled, fallback, or disabled, including comments about backend-specific hook validity.
- **Residual MLU exceptions are acceptable when named**: `vllm_mlu._mlu_ops.active` remains an allowed bounded exception if it is still the real execution seam for gated SiLU on MLU.

## Open Questions

### Resolved During Planning

- **Should ownership live directly in `nt_ops_vllm` or layer entirely on `vllm-mlu`?**
  Use direct but thin ownership in `nt_ops_vllm`: expose a first-class nt-ops MLU platform/worker entry in this repo, but implement it as a narrow shim over the real `vllm_mlu` worker path. This satisfies the user’s desired architecture while minimizing new surface area.

- **What is the primary NT Ops activation point?**
  The nt-ops-aware MLU worker startup path. Attachment should happen before model execution begins and remain once-per-worker-process semantics, preserving the current early-install requirement without relying on dynamic base-worker resolution.

- **Which existing patches are already known residual exceptions?**
  `vllm_mlu._mlu_ops.active` is the canonical retained exception for gated SiLU on MLU. RoPE remains explicit fallback because the known MLU fused attention path bypasses `RotaryEmbedding.forward_oot`.

- **What migration posture should this phase take?**
  Platform/device selection becomes the documented primary path immediately. Manual `worker_cls` injection remains as a deprecated compatibility shim for one phase, with tests ensuring it resolves to the same attach behavior rather than preserving a second architecture.

### Deferred to Implementation

- Exact module/class names for the new platform entrypoint if upstream vLLM plugin discovery constrains naming.
- Whether the compatibility shim should emit a warning at import time or first use.
- Whether any report-retrieval plumbing needs minor adaptation once the worker path changes under different vLLM executors.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```text
user/config selects device='mlu'
        │
        ▼
nt_ops_vllm MLU platform registration
        │
        ▼
platform chooses nt-ops-aware MLU worker class
        │
        ▼
worker startup attaches NT Ops once per worker process
        │
        ├── runtime.py applies transactional patch inventory from registry.py
        ├── core reachable hooks enabled (e.g. RMSNorm)
        └── bounded MLU-specific exceptions remain explicit (e.g. vllm_mlu._mlu_ops.active)
        │
        ▼
inference path
        │
        ├── exercised hooks record hits / reports
        └── known dead hooks on fused backends remain fallback (e.g. rope)
```

## Implementation Units

- [ ] **Unit 1: Add first-class MLU platform registration for nt_ops**

**Goal:** Make MLU selectable through the normal vLLM platform/device path and stop requiring examples to inject a custom `worker_cls` as the primary entrypoint.

**Requirements:** R1, R4, R5, R8

**Dependencies:** None

**Files:**
- Modify: `setup.py`
- Create: `nt_ops/platforms/__init__.py`
- Create: `nt_ops/platforms/mlu.py`
- Modify: `nt_ops/__init__.py`
- Test: `tests/test_platform_registration.py`

**Approach:**
- Add the package entry point(s) required for vLLM to discover an out-of-tree MLU platform from this repo.
- Keep the platform file thin: it should declare the nt-ops MLU device path and choose the nt-ops-aware MLU worker class, not absorb patch logic or backend-specific operator code.
- Preserve `nt_ops` as the public Python package surface; platform registration is an integration seam, not a new top-level framework.

**Patterns to follow:**
- Upstream/vllm-mlu shape established in research: platform plugin owns device type and worker class selection.
- Current repo rule from `runtime.py`: installation must remain explicit and transactional, not import-time side effects.

**Test scenarios:**
- Plugin registration resolves an nt-ops MLU platform without requiring manual worker injection.
- Platform selection chooses the expected nt-ops-aware MLU worker class when device type is MLU.
- Non-MLU paths do not accidentally select the nt-ops MLU worker.

**Verification:**
- The repo exposes a discoverable MLU platform entrypoint in packaging metadata.
- A focused test can assert the platform chooses the intended worker class without importing hardware-specific runtime paths.

---

- [ ] **Unit 2: Replace dynamic wrapper resolution with an explicit nt-ops-aware MLU worker path**

**Goal:** Move NT Ops attachment from a generic wrapper worker to an explicit MLU worker path while preserving early, once-per-worker-process attachment semantics.

**Requirements:** R2, R3, R4, R5, R8

**Dependencies:** Unit 1

**Files:**
- Create: `nt_ops/workers/__init__.py`
- Create: `nt_ops/workers/mlu.py`
- Modify: `nt_ops/worker.py`
- Modify: `nt_ops/runtime.py`
- Modify: `nt_ops/vllm_utils.py`
- Test: `tests/test_mlu_worker_integration.py`
- Test: `tests/test_runtime_install.py`
- Test: `tests/test_vllm_utils.py`

**Approach:**
- Introduce an explicit nt-ops MLU worker class whose job is to attach NT Ops at the correct worker lifecycle point, then delegate to the real MLU worker behavior.
- Retire `_resolve_base_worker_cls()` and its hard-coded candidate probing as the main architecture.
- Keep `nt_ops/worker.py` only as a compatibility bridge that resolves to the new worker path and clearly signals deprecation.
- Preserve process-global safety in `runtime.py`, but make duplicate-attach semantics explicit for worker restarts and migration overlap.

**Patterns to follow:**
- Current `runtime.install()` transactional semantics and explicit `InstallationError` path.
- Current `get_nt_ops_report` contract exposed through worker/executor plumbing.

**Test scenarios:**
- NT Ops attaches exactly once for the intended worker startup path.
- Duplicate attach with the same profile remains idempotent; mixed-mode attach attempts fail or warn in a controlled way.
- Deprecated `nt_ops.worker.NTVLLMWorker` still reaches the new path during the transition window.
- Report retrieval still works through both `collective_rpc` and executor fallback paths.

**Verification:**
- The worker surface no longer relies on dynamic probing of `vllm_mlu.*` classes as the primary design.
- Compatibility tests show old and new entry paths converge on the same runtime state and reporting behavior.

---

- [ ] **Unit 3: Reframe the patch profile around reachable MLU seams and bounded exceptions**

**Goal:** Keep NT Ops thin by preserving `registry.py` as the explicit capability inventory, while documenting which MLU hooks are valid core seams and which are backend-specific exceptions or fallback-only.

**Requirements:** R5, R6, R7

**Dependencies:** Unit 2

**Files:**
- Modify: `nt_ops/registry.py`
- Modify: `nt_ops/mlu_dispatch.py`
- Modify: `nt_ops/debug.py`
- Test: `tests/test_registry_profile.py`
- Test: `tests/test_mlu_dispatch.py`

**Approach:**
- Keep the active/fallback/disabled profile split, but tighten comments and structure around backend reachability so planning assumptions become executable review targets.
- Preserve `vllm_mlu._mlu_ops.active` as a named residual exception if still necessary after the worker migration.
- Preserve RoPE fallback on MLU fused attention, and make hook viability part of the profile story rather than an implementation footnote.

**Patterns to follow:**
- Existing `PatchSpec` builder pattern in `registry.py`.
- Existing solution doc guidance: dead hooks must stay fallback even if they are patchable in theory.

**Test scenarios:**
- Profile assertions still match the intended active/fallback split after the worker/platform redesign.
- The MLU active wrapper continues to intercept only the supported gated SiLU path and falls back otherwise.
- Debug/reporting surfaces can distinguish “installed” from “exercised” for reachable hooks.

**Verification:**
- `registry.py` remains the single source of truth for which hooks are active, fallback, or disabled on MLU.
- The plan’s known backend exceptions are reflected in code comments and tests, not only in docs.

---

- [ ] **Unit 4: Migrate public entrypoints, examples, and compatibility messaging**

**Goal:** Make the MLU platform/device path the documented primary workflow and remove wrapper-centric guidance from examples and docs.

**Requirements:** R1, R4, R8

**Dependencies:** Unit 1, Unit 2

**Files:**
- Modify: `examples/basic.py`
- Modify: `examples/benchmark_throughput.py`
- Modify: `README.md`
- Modify: `nt_ops/monkey_patch.py`
- Modify: `doc/cross_platform.md`
- Test: `tests/test_examples_config.py`

**Approach:**
- Remove manual `worker_cls` injection from examples as the recommended path.
- Update README and cross-platform docs so MLU-first architecture is reflected in both usage and project positioning.
- Keep `monkey_patch.py` only as a compatibility shim if still needed; it should not advertise the old architecture as current.

**Patterns to follow:**
- Existing public API style in `nt_ops/__init__.py`.
- Current examples’ capability-report output flow, but sourced from the new platform-selected worker path.

**Test scenarios:**
- Example configuration no longer depends on manual `worker_cls` to describe the primary MLU path.
- Legacy compatibility messages are clear and non-breaking during the transition.
- Docs do not claim generic cross-device maturity that the architecture still intentionally defers.

**Verification:**
- User-facing examples and README align with the new architecture.
- No primary documentation path still tells users to activate NT Ops via `worker_cls="nt_ops.worker.NTVLLMWorker"`.

---

- [ ] **Unit 5: Add cross-layer verification for selection, attachment timing, and hook viability**

**Goal:** Cover the gaps in the current test suite by verifying platform selection, worker attachment timing, migration behavior, and the difference between patch installation and real hook reachability.

**Requirements:** R1, R2, R7, R8

**Dependencies:** Unit 1, Unit 2, Unit 3, Unit 4

**Files:**
- Create: `tests/test_platform_registration.py`
- Create: `tests/test_mlu_worker_integration.py`
- Create: `tests/test_examples_config.py`
- Modify: `tests/test_runtime_install.py`
- Modify: `tests/test_registry_profile.py`
- Modify: `tests/test_vllm_utils.py`
- Modify: `tests/test_mlu_dispatch.py`

**Approach:**
- Add pure-Python tests for plugin/worker selection and compatibility shims.
- Extend runtime tests to cover repeated attach, same-profile idempotence, and mixed old/new activation paths.
- Keep hardware-free coverage focused on architectural invariants; real MLU inference remains a manual verification path.
- Encode the Rope/FlashAttention lesson as a verification expectation: installed hooks and exercised hooks are different signals.

**Execution note:** Start with characterization coverage for current runtime/install/report behavior before replacing the worker seam.

**Patterns to follow:**
- Existing test style in `tests/test_runtime_install.py`, `tests/test_registry_profile.py`, and `tests/test_vllm_utils.py`.

**Test scenarios:**
- `device='mlu'` path selects the nt-ops MLU worker without manual override.
- Deprecated wrapper path reaches the same runtime attach path as the new platform path.
- Duplicate worker/process initialization does not double-install a conflicting runtime profile.
- Known dead hooks remain fallback and are not reported as exercised simply because install succeeded.
- Capability/report surfaces still work across executor variants.

**Verification:**
- The test suite proves architectural invariants without needing MLU hardware.
- Manual MLU smoke verification has a clear checklist: startup backend selection, attach log, hit counters, and known fused-backend fallback behavior.

## System-Wide Impact

- **Interaction graph:** Packaging metadata, vLLM plugin discovery, worker startup, runtime patch install, capability reporting, and example invocation all change together.
- **Error propagation:** Worker attachment failures must surface as clear startup failures or bounded compatibility behavior; silent fallback to the old architecture should be avoided.
- **State lifecycle risks:** Runtime state is process-global today, so worker restart, multi-process startup, and mixed old/new entrypoints can cause duplicate-attach problems if not explicitly tested.
- **API surface parity:** Example scripts, compatibility shims, and report retrieval must all describe the same architecture.
- **Integration coverage:** Unit tests alone will not prove hook viability on real MLU fused backends; the plan must preserve manual smoke verification for exercised-versus-installed distinctions.

## Risks & Dependencies

- The redesign depends on vLLM’s out-of-tree plugin seam remaining sufficient for MLU platform registration and worker selection.
- The thin worker shim must not accidentally grow into a second copy of `vllm_mlu` runtime behavior.
- Process-global runtime state may require careful idempotence rules once attachment moves under a new worker path.
- Some backend-specific exceptions may remain necessary longer than ideal; the risk is not their existence, but letting them become the hidden primary architecture again.

## Documentation / Operational Notes

- Update docs to distinguish “installed” from “exercised” behavior, especially for operators blocked by fused backends.
- Preserve `NT_OPS_DEBUG=1` and startup backend-selection logs as part of the manual MLU verification story.
- If compatibility shims remain for one phase, document their deprecation clearly in README/examples rather than burying it in code.

## Sources & References

- **Origin document:** `docs/brainstorms/2026-03-31-mlu-first-platform-worker-architecture-requirements.md`
- Related code: `nt_ops/worker.py`, `nt_ops/runtime.py`, `nt_ops/registry.py`, `nt_ops/mlu_dispatch.py`, `nt_ops/vllm_utils.py`
- Related plan: `docs/plans/2026-03-31-001-refactor-hybrid-dispatch-main-kernels-plan.md`
- Institutional learning: `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md`
- Adjacent references: `/Users/bytedance/Desktop/nt_workspace/vllm-mlu`, `/Users/bytedance/Desktop/nt_workspace/vllm`
