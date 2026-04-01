---
title: "feat: Establish thin nt-ops-enabled MLU backend phase 1"
type: feat
status: active
date: 2026-04-01
origin: docs/brainstorms/2026-04-01-minimal-ntops-mlu-backend-phase1-requirements.md
---

# feat: Establish thin nt-ops-enabled MLU backend phase 1

## Overview

This plan turns the current MLU integration into an explicitly scoped phase-1 backend path: `nt_ops_vllm` remains thin on top of `vllm-mlu`, owns plugin activation, worker-time nt-ops attachment, capability reporting, and a defined operator matrix, while `vllm-mlu` continues to own MLU platform/runtime bring-up. The goal is a correct, testable, single-card bf16 MLU path with honest ownership boundaries, not a standalone backend or a one-package install story.

## Problem Frame

The codebase has already moved away from manual `worker_cls` injection and toward a first-class nt-ops-aware MLU path, but the current shape is still easy to misread as either a full backend or a purely local patch layer. The new origin document narrows phase 1 to a thinner and more honest outcome: prove a product-facing nt-ops-enabled MLU execution path on top of `vllm-mlu`, define exactly what `nt_ops_vllm` owns, and avoid quietly expanding into standalone-backend work because of packaging or seam limitations. (see origin: `docs/brainstorms/2026-04-01-minimal-ntops-mlu-backend-phase1-requirements.md`)

## Requirements Trace

- R1. Keep phase 1 as NT Ops on top of `vllm-mlu`, not a standalone hardware backend.
- R2. Make ownership explicit: `vllm-mlu` owns MLU runtime bring-up; `nt_ops_vllm` owns activation, attach, reporting, and the minimal plugin surface.
- R3. Target only single-card bf16 inference.
- R4. Exclude quantization, communication, Ray/distributed support, and performance optimization.
- R5. Define a concrete operator matrix: must-have nt-ops substitutions, backend-native fallbacks, explicit out-of-scope operators.
- R6. Keep attention backend-native unless a specific phase-1 correctness issue requires more ownership.
- R7. Preserve vLLM’s current block/page-oriented KV cache runtime assumptions.
- R8. Preserve a future staged carve-out path without implementing it now.
- R9. Keep packaging/install claims conservative and truthful about `vllm-mlu` dependency.
- R10. If plugin seams are insufficient, narrow or stop phase 1 rather than expanding backend ownership by default.

## Scope Boundaries

- Do not plan a fully standalone nt-ops MLU backend in this phase.
- Do not plan quantization, communication, distributed executors, Ray integration, or multi-card support.
- Do not replace vLLM’s paged/block KV runtime model.
- Do not promise that `vllm-mlu` is removable from the runtime stack.
- Do not require attention to move into nt-ops ownership unless a concrete phase-1 blocker proves it necessary.

## Context & Research

### Relevant Code and Patterns

- `setup.py` publishes the current nt-ops platform plugin and the `nt_ops_mlu_hijack` general plugin wrapper.
- `nt_ops/__init__.py` returns plugin entry points and delegates hijack registration to upstream `vllm_mlu.register_mlu_hijack()`.
- `nt_ops/platforms/mlu.py` is the current thin seam: it subclasses upstream `MLUPlatform` and rewrites `worker_cls` / `sd_worker_cls` only.
- `nt_ops/workers/mlu.py` performs worker-time `install(process_scope=REQUIRED_PROCESS_SCOPE)` and exposes `get_nt_ops_report()`.
- `nt_ops/runtime.py` is the transactional install state machine and still defines the authoritative attach lifecycle.
- `nt_ops/vllm_utils.py` is the report-retrieval compatibility layer across `collective_rpc`, `execute_method`, driver worker, and nested worker surfaces.
- `nt_ops/registry.py` is the operator matrix source of truth and already encodes active/fallback/disabled behavior for MLU.
- `tests/test_platform_registration.py`, `tests/test_mlu_worker_integration.py`, `tests/test_runtime_install.py`, `tests/test_vllm_utils.py`, `tests/test_registry_profile.py`, `tests/test_capability_report.py`, `tests/test_examples_config.py` provide the current hardware-free verification surface.
- `README.md` and `examples/basic.py` already describe the current explicit plugin-based MLU path and reveal where packaging/user contract language must stay conservative.

### Institutional Learnings

- Patch `forward_oot`, not `forward`, on out-of-tree MLU paths; `forward` can be bypassed.
- Patch installation success is not proof of runtime reachability; `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md` shows fused MLU attention can make a hook dead even when install succeeds.
- Plugin activation order matters: `nt_ops_mlu` and `nt_ops_mlu_hijack` are coupled for the real MLU path.
- Report retrieval must tolerate multiple vLLM v1 executor paths and missing method behavior.

### External References

- Local adjacent references only: `/Users/bytedance/Desktop/nt_workspace/vllm` and `/Users/bytedance/Desktop/nt_workspace/vllm-mlu` are sufficient for this phase-1 plan.

## Key Technical Decisions

- **Keep the platform seam thin**: `nt_ops_vllm` should continue to present a first-class nt-ops-aware MLU path by composing with `vllm-mlu`, not by replacing platform/runtime ownership.
- **Treat plugins as a coupled surface**: the phase-1 path includes both the nt-ops platform plugin and the nt-ops general plugin wrapper so upstream MLU hijack/startup behavior is preserved.
- **Keep worker-time install as the attach point**: phase 1 continues to attach nt-ops during worker construction, with tests expanded around `spawn`, idempotence, and compatibility edges.
- **Make the operator matrix explicit**: active nt-ops substitutions, backend-native fallbacks, and explicit out-of-scope operators must be encoded in `registry.py` and reflected in tests/docs.
- **Keep attention backend-native by default**: no new attention backend ownership is planned unless a correctness issue on the thin path proves it is required.
- **Keep packaging claims honest**: the user-facing story is a reliable nt-ops-enabled MLU path, not “MLU from a CPU build plus one plugin package” in phase 1.

## Open Questions

### Resolved During Planning

- **What is the minimum viable nt-ops surface for phase 1?**
  Plugin activation, worker-time attach, capability reporting, and a defined operator matrix. Anything broader must justify itself as necessary for the thin path.

- **Should attention move into nt-ops ownership in phase 1?**
  No. Attention remains backend-native unless a specific correctness blocker is found during implementation.

- **What packaging story should phase 1 promise?**
  A reliable MLU-machine installation path that is explicit about `vllm-mlu` remaining required. Simpler packaging may be pursued later, but it is not a phase-1 acceptance target.

### Deferred to Implementation

- Exact plugin activation ergonomics if environment-variable defaults prove awkward in real usage.
- Whether the current lazy platform-class resolution in `nt_ops/platforms/mlu.py` needs further import-timing hardening once exercised in more environments.
- Whether additional report-retrieval fallbacks are needed for executor variants beyond the currently tested ones.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```text
user selects MLU path
      │
      ├── platform plugin: nt_ops_mlu
      ├── general plugin: nt_ops_mlu_hijack
      ▼
vllm-mlu still owns MLU startup/runtime behavior
      │
      ▼
nt_ops platform shim rewrites worker selection
      │
      ▼
nt-ops-aware MLU worker attaches runtime profile once per worker process
      │
      ├── runtime.py installs explicit patch inventory from registry.py
      ├── active nt-ops substitutions are reported and hit-counted
      └── backend-native/fused paths remain explicit fallback
      │
      ▼
user-visible capability report reflects installed + exercised operators
```

## Implementation Units

- [ ] **Unit 1: Stabilize the coupled plugin activation surface**

**Goal:** Make the phase-1 backend path explicit and reliable by treating `nt_ops_mlu` and `nt_ops_mlu_hijack` as a coupled plugin surface, with clear packaging and documentation boundaries.

**Requirements:** R1, R2, R9, R10

**Dependencies:** None

**Files:**
- Modify: `setup.py`
- Modify: `nt_ops/__init__.py`
- Modify: `README.md`
- Modify: `examples/basic.py`
- Test: `tests/test_platform_registration.py`
- Test: `tests/test_examples_config.py`

**Approach:**
- Keep the current split between platform plugin and general plugin wrapper, but make the user-facing contract explicit in packaging/docs and verification.
- Decide whether phase 1 should keep explicit `VLLM_PLUGINS` pairing, add a repo helper/env convention, or document that pairing as the supported path without trying to hide it.
- Verify that the docs never imply `vllm-mlu` is optional in this phase.

**Patterns to follow:**
- Existing plugin entry points in `setup.py` and helper registration functions in `nt_ops/__init__.py`.
- Upstream plugin-system contract in `vllm/docs/design/plugin_system.md`.

**Test scenarios:**
- Packaging metadata exposes both nt-ops plugin entry points.
- Explicit plugin pairing is reflected in examples/docs.
- Missing general plugin path has a documented, understandable failure mode rather than silent misbehavior.

**Verification:**
- The phase-1 install story is concrete and truthful.
- Plugin activation behavior is documented and test-covered without claiming standalone-backend packaging.

---

- [ ] **Unit 2: Harden worker-time nt-ops attachment and lifecycle edges**

**Goal:** Make worker-time attachment the reliable phase-1 ownership point, including startup timing, idempotence, compatibility behavior, and MLU process-scope constraints.

**Requirements:** R2, R3, R4, R10

**Dependencies:** Unit 1

**Files:**
- Modify: `nt_ops/platforms/mlu.py`
- Modify: `nt_ops/workers/mlu.py`
- Modify: `nt_ops/runtime.py`
- Modify: `nt_ops/worker.py`
- Test: `tests/test_mlu_worker_integration.py`
- Test: `tests/test_runtime_install.py`

**Approach:**
- Keep worker-time `install()` as the primary attach point, but tighten lifecycle behavior around `spawn`, duplicate initialization, compatibility shims, and early plugin/worker interactions.
- Revisit whether any import timing or lazy-resolution behavior in the platform and worker wrappers needs further hardening now that phase 1 is explicitly a long-lived path, not an experiment.
- Preserve deprecated compatibility entry points only as narrow bridges, not as parallel architecture.

**Execution note:** Add characterization coverage around current worker attach/runtime state behavior before changing lifecycle semantics.

**Patterns to follow:**
- Current transactional `install()` semantics in `nt_ops/runtime.py`.
- Current lazy worker resolution and compatibility shape in `nt_ops/workers/mlu.py` and `nt_ops/worker.py`.

**Test scenarios:**
- Nt-ops attaches exactly once in the intended worker startup path.
- Same-profile repeated init is idempotent.
- Conflicting attach attempts fail clearly.
- Compatibility shim still routes into the thin path without becoming a second supported architecture.

**Verification:**
- The worker lifecycle has one clear attach point for phase 1.
- Spawn/process-scope constraints are tested and documented as part of the supported path.

---

- [ ] **Unit 3: Define and encode the phase-1 operator matrix**

**Goal:** Turn the current implicit patch inventory into an explicit phase-1 operator matrix that distinguishes required nt-ops substitutions, backend-native fallbacks, and out-of-scope operators.

**Requirements:** R5, R6, R7

**Dependencies:** Unit 2

**Files:**
- Modify: `nt_ops/registry.py`
- Modify: `nt_ops/mlu_dispatch.py`
- Modify: `nt_ops/capabilities.py`
- Modify: `nt_ops/debug.py`
- Test: `tests/test_registry_profile.py`
- Test: `tests/test_mlu_dispatch.py`
- Test: `tests/test_capability_report.py`

**Approach:**
- Make the phase-1 matrix explicit in code and docs: which operators are must-have nt-ops substitutions, which are backend-native but acceptable, and which are out of scope.
- Preserve the current “installed vs exercised” distinction as part of the matrix contract, not just a debugging nicety.
- Keep fused attention-related paths backend-native by default unless implementation uncovers a real correctness blocker.

**Patterns to follow:**
- Existing `CapabilityProfile` / `PatchSpec` structure in `nt_ops/registry.py`.
- Existing rope/fused-path learning in `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md`.

**Test scenarios:**
- Active operator list matches the intended phase-1 substitutions.
- Fallback list reflects known backend-native fused paths.
- Capability report exposes both `hits` and `exercised` correctly for active operators.
- Debug/report surfaces do not imply reachability when only installation succeeded.

**Verification:**
- Planning and implementation can point to one source of truth for the phase-1 operator matrix.
- The matrix is specific enough that users and developers can tell what phase 1 actually delivers.

---

- [ ] **Unit 4: Harden report retrieval across vLLM v1 execution paths**

**Goal:** Make capability reporting a reliable part of the thin backend path across the vLLM v1 surfaces that phase 1 actually encounters.

**Requirements:** R2, R3, R5

**Dependencies:** Unit 2

**Files:**
- Modify: `nt_ops/vllm_utils.py`
- Test: `tests/test_vllm_utils.py`
- Test: `tests/test_mlu_worker_integration.py`

**Approach:**
- Keep the current layered report retrieval strategy, but expand it to cover the executor/object shapes phase 1 actually sees in practice.
- Treat report retrieval as part of the product-facing backend contract, not as a best-effort debug helper.
- Explicitly handle “method not implemented”, nested executor/worker objects, and pre-attach or empty-report states in a way that yields clear behavior.

**Patterns to follow:**
- Current fallback ordering in `nt_ops/vllm_utils.py`.
- Current worker-provided `get_nt_ops_report()` surface in `nt_ops/workers/mlu.py`.

**Test scenarios:**
- `collective_rpc` success path returns the expected report.
- Executor fallback path returns the expected report.
- Missing-method RPC failure falls back cleanly when the worker still exposes the report.
- Empty, missing, or pre-attach report states fail in a predictable and debuggable way.

**Verification:**
- Capability reporting works as a stable phase-1 feature across the supported vLLM v1 paths.
- The repo’s tests cover the actual failure modes already seen in MLU runs.

---

- [ ] **Unit 5: Align user-facing docs and examples with the supported phase-1 story**

**Goal:** Make the public story match the real phase-1 backend: thin on `vllm-mlu`, single-card bf16, conservative packaging claims, and explicit plugin/runtime expectations.

**Requirements:** R1, R4, R8, R9

**Dependencies:** Unit 1, Unit 3, Unit 4

**Files:**
- Modify: `README.md`
- Modify: `doc/cross_platform.md`
- Modify: `examples/basic.py`
- Modify: `examples/benchmark_throughput.py`
- Test: `tests/test_examples_config.py`

**Approach:**
- Update docs so they describe the supported phase-1 MLU path honestly: what is required, what is owned by `nt_ops_vllm`, what still belongs to `vllm-mlu`, and what is intentionally out of scope.
- Keep the examples aligned with the supported phase-1 plugin/runtime posture and capability report semantics.
- Avoid any language that implies standalone-backend independence, generic multi-device maturity, or a finished packaging story beyond what the code really supports.

**Patterns to follow:**
- Current README plugin guidance and capability report output style.
- The origin requirements doc’s packaging and ownership guardrails.

**Test scenarios:**
- Examples reflect the supported plugin/runtime path.
- Docs do not imply `vllm-mlu` is removable in phase 1.
- Public wording matches the phase-1 operator/fallback matrix and reporting semantics.

**Verification:**
- A reader can understand what phase 1 does and does not provide without reading the code.
- Docs and examples no longer over-promise packaging or backend independence.

## System-Wide Impact

- **Interaction graph:** platform plugin registration, general plugin loading, worker startup, runtime attach, operator patching, capability reporting, and example startup remain coupled surfaces in phase 1.
- **Error propagation:** plugin misconfiguration, attach failure, or missing report methods must surface as explicit startup/reporting failures, not as silent partial activation.
- **State lifecycle risks:** runtime state is still process-global, so worker restarts, repeated init, and mixed old/new entry paths are the main correctness risks.
- **API surface parity:** capability reporting must remain consistent across `collective_rpc`, executor fallback, and direct worker access.
- **Integration coverage:** pure unit tests will not prove MLU runtime behavior on their own; manual MLU validation must remain part of the acceptance path for plugin activation, attach logs, and exercised operator hits.

## Risks & Dependencies

- The thin path still depends on `vllm-mlu` ownership of MLU startup/runtime behavior.
- Plugin-coupling behavior remains a source of user-facing complexity if docs and tests drift.
- Fused backend behavior can make a patch appear installed while remaining dead at runtime.
- Attempting to simplify packaging or grow ownership surface prematurely risks turning phase 1 into a hidden standalone-backend effort.

## Documentation / Operational Notes

- Preserve clear user guidance for the supported phase-1 MLU command path and required plugins.
- Keep the capability report centered on `installed` plus `exercised` operator visibility.
- If runtime validation still depends on explicit env vars or startup constraints, document those as part of the supported phase-1 contract rather than as temporary hacks.

## Sources & References

- **Origin document:** `docs/brainstorms/2026-04-01-minimal-ntops-mlu-backend-phase1-requirements.md`
- Prior plan: `docs/plans/2026-03-31-003-refactor-mlu-platform-worker-plan.md`
- Institutional learning: `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md`
- Related code: `setup.py`, `nt_ops/__init__.py`, `nt_ops/platforms/mlu.py`, `nt_ops/workers/mlu.py`, `nt_ops/runtime.py`, `nt_ops/vllm_utils.py`, `nt_ops/registry.py`
- Adjacent references: `/Users/bytedance/Desktop/nt_workspace/vllm`, `/Users/bytedance/Desktop/nt_workspace/vllm-mlu`
