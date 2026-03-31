---
title: "refactor: Hybrid Dispatch — Main Kernels + Refactored Patch Infrastructure"
type: refactor
status: active
date: 2026-03-31
origin: docs/brainstorms/2026-03-30-hybrid-dispatch-infrastructure-requirements.md
---

# refactor: Hybrid Dispatch — Main Kernels + Refactored Patch Infrastructure

## Overview

The `refactor/nt-ops-qwen3-minimal-dense-path` branch introduced a correct dispatch architecture
(patching `forward_oot` instead of `forward` for MLU) and good infrastructure (runtime, registry,
capabilities, worker), but broke MLU operation by inlining Triton kernels with grid-size overflow
bugs. This plan creates a new branch from `main` that takes the working ninetoothed kernel
implementations and wires them into the new dispatch infrastructure.

## Problem Frame

`main` works on MLU but has two problems: (1) patches `forward` instead of `forward_oot`, which is
silently bypassed by vLLM's `dispatch_forward` on out-of-tree platforms, and (2) applies patches
imperatively on import with no lifecycle management. The refactor branch fixed both, but introduced
broken inlined Triton kernels that fail at runtime on MLU. Neither branch alone is correct; the
fix is to combine them precisely.

(see origin: docs/brainstorms/2026-03-30-hybrid-dispatch-infrastructure-requirements.md)

## Requirements Trace

- R1. New branch from `main` — not from the broken refactor branch.
- R2. Infrastructure files (`runtime.py`, `registry.py`, `capabilities.py`, `worker.py`) from
  refactor branch, unchanged.
- R3. `nt_ops/kernels/rms_norm.py` is dropped — it is the broken inlined Triton RMSNorm kernel.
  `nt_ops/kernels/rope.py` and `nt_ops/kernels/utils.py` are kept: the rope kernel is
  ninetoothed-based (no Triton grid-size dispatch logic) and has no known MLU failure. No
  main-branch rope implementation exists, so there is no proven alternative.
- R4. `nt_ops/rms.py` retains main's ninetoothed kernel classes and `rms()` dispatch; adds
  `build_rms_forward_oot(original)` adapter; removes `rms_forward`.
- R5. `nt_ops/rope.py` retains refactor branch's ninetoothed rope kernel and
  `build_rotary_forward_oot(original)` adapter. (No rope.py exists on main.)
- R6. `nt_ops/activation.py` — refactor branch version (main kernel math + `record_hit` call).
- R7. `nt_ops/monkey_patch.py` updated to thin shim from refactor branch.
- R8. `registry.py` builders wire to adapter functions in `rms.py` / `rope.py` / `activation.py`.
- R9. `forward_oot` is the patch target throughout; no `forward`-patching remains.

## Scope Boundaries

- Do not modify kernel math (ninetoothed `arrangement` / `application` lambdas) from `main`.
- Do not add new operators or PatchSpec entries beyond what the refactor branch defines.
- Drop `nt_ops/kernels/rms_norm.py` only — it is the broken Triton kernel.
- Keep `nt_ops/kernels/rope.py` and `nt_ops/kernels/utils.py` — required by the rope adapter.
- Skip MLU hardware validation — no test environment available locally.

## Context & Research

### Relevant Code and Patterns

- `nt_ops/rms.py` (main): inline ninetoothed kernel classes `RMSWithWeight`, `RMSNoWeight`,
  `RMSResidualWithWeight`, `RMSResidualNoWeight`; `rms()` dispatch over `torch.ops.vllm.nt_rms_*`;
  `direct_register_custom_op` registrations for all four variants.
- `nt_ops/rms.py` (refactor): `build_rms_forward_oot(original)` builder pattern; `record_hit`;
  delegates to `nt_ops.kernels.rms_norm` (to be replaced with main's inline kernel).
- `nt_ops/rope.py` (refactor only): `build_rotary_forward_oot(original)`; delegates to
  `nt_ops.kernels.rope.rotary_position_embedding`; strips `is_cuda` guard intentionally.
- `nt_ops/kernels/rope.py` (refactor): ninetoothed-based; uses `_cached_make`; no Triton
  grid-size dispatch logic — safe to keep.
- `nt_ops/kernels/utils.py` (refactor): `_cached_make` memoization helper used by `kernels/rope.py`.
- `nt_ops/registry.py` (refactor): `_rms_forward_oot_builder` → `rms.build_rms_forward_oot`;
  `_rope_forward_oot_builder` → `rope.build_rotary_forward_oot`;
  `_activation_silu_builder` → `activation.silu_and_mul_forward`. No changes needed.
- `nt_ops/activation.py` (refactor): same ninetoothed kernel math as main; adds `record_hit("silu_and_mul")`.
- `nt_ops/__init__.py` (refactor): side-effect-free; exports `install`, `uninstall`,
  `get_runtime_state`, `get_capability_report`, `reset_hits`, `InstallationError`, `RuntimeState`.
- Tests: `test_runtime_install.py` and `test_capability_report.py` require no GPU and should pass
  on the new branch. `test_qwen3_operator_matrix.py` and `test_qwen3_basic_smoke.py` require GPU.

### Institutional Learnings

- **Always patch `forward_oot`, not `forward`** — `dispatch_forward` on out-of-tree platforms
  binds `_forward_method` to `self.forward_oot`, not `forward`. Patching `forward` is a no-op on MLU.
- **Always close over `original` in adapters** — the original may be a vendor MLU kernel, not
  `forward_native`. Falling back preserves vendor correctness for edge cases.
- **Strip `is_cuda` guards when porting to MLU** — MLU tensors return `False` for `.is_cuda`.
- **Patch at class level before model instantiation** — after instantiation `_forward_method` is
  already bound; class-level patching has no effect.

### External References

- None required — approach and patterns are fully established by the two local branches.

## Key Technical Decisions

- **Base from `main`, not refactor branch**: Avoids carrying any broken kernel code;
  `main` is the last known-good MLU baseline. (see origin)
- **Drop `kernels/rms_norm.py` + `kernels/utils.py`, keep `kernels/rope.py`**: The rms_norm
  inlined Triton kernel is the identified failure point (grid-size overflow). The rope kernel is
  ninetoothed-based with no Triton grid dispatch logic; no MLU failure has been attributed to it.
- **Keep `direct_register_custom_op` in rms.py**: Main's `rms()` dispatch calls
  `torch.ops.vllm.nt_rms_*`. Those ops are only available after registration; removing the calls
  would break `rms()` at runtime.
- **Adapter functions, not rewrites**: `build_rms_forward_oot` wraps the existing `rms()` function
  with the `forward_oot(self, x, residual=None)` signature. Minimal new code, no kernel changes.
- **Rope from refactor branch only**: `rope.py` does not exist on `main`. The refactor branch's
  ninetoothed rope kernel is the only available implementation and has no known MLU failure.

## Open Questions

### Resolved During Planning

- **Does `direct_register_custom_op` need to stay in rms.py?**
  Yes — `rms()` calls `torch.ops.vllm.nt_rms_*` which require the ops to be registered. The
  registration must happen at module import time before any call to `rms()`.
- **Does rope.py exist on main?**
  No. `git ls-tree main nt_ops/` confirms only `rms.py` and `activation.py` have main counterparts.
  The refactor branch's rope.py and `kernels/rope.py` are the authoritative implementations.
- **Does `kernels/utils.py` need to be kept?**
  Yes — `kernels/rope.py` imports `_cached_make` from `nt_ops.kernels.utils`. Drop `utils.py`
  only if the rope kernel is also dropped.

### Deferred to Implementation

- **Residual path correctness in `build_rms_forward_oot`**: The adapter must match the residual
  in-place update semantics of vLLM's `fused_add_rms_norm` path. Main's `rms_rw` returns
  `(output, residual)` but modifies tensors differently than the refactor branch's helper. Verify
  that `x.copy_` / `residual.copy_` semantics are correct when integrating.
- **`direct_register_custom_op` and double-registration risk**: If `rms.py` is imported multiple
  times across workers, re-registration might raise. Check whether vLLM's custom op system is
  idempotent or needs a registration guard.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not code to
> reproduce.*

```
nt_ops.install(profile="qwen3_minimal_dense")
    │
    ▼
runtime.py → registry.py
    │              │
    │    ┌──────────────────────┐
    │    │  PatchSpec entries   │
    │    │  rms_norm  ──────────┼──► _rms_forward_oot_builder(original)
    │    │                      │        → rms.build_rms_forward_oot(original)
    │    │                      │              ↳ wraps rms() [main ninetoothed]
    │    │  silu_and_mul ───────┼──► _activation_silu_builder(original)
    │    │                      │        → activation.silu_and_mul_forward
    │    │                      │              ↳ [main ninetoothed]
    │    │  rope ───────────────┼──► _rope_forward_oot_builder(original)
    │    │                      │        → rope.build_rotary_forward_oot(original)
    │    └──────────────────────┘              ↳ wraps kernels/rope.py [ninetoothed]
    │
    ▼
setattr(TargetClass, "forward_oot", replacement)
  ↳ RMSNorm.forward_oot
  ↳ SiluAndMul.forward_oot
  ↳ RotaryEmbedding.forward_oot
```

## Implementation Units

---

- [ ] **Unit 1: Create new branch and copy infrastructure from refactor branch**

**Goal:** Establish the new branch baseline from `main` and bring in all infrastructure files that
have no main counterpart and need no modification.

**Requirements:** R1, R2, R7

**Dependencies:** None

**Files:**
- Create branch: `refactor/hybrid-dispatch-main-kernels` from `main`
- Copy from refactor branch (git checkout or manual copy):
  - `nt_ops/runtime.py`
  - `nt_ops/registry.py`
  - `nt_ops/capabilities.py`
  - `nt_ops/worker.py`
  - `nt_ops/__init__.py`
  - `nt_ops/monkey_patch.py`

**Approach:**
- Use `git checkout refactor/nt-ops-qwen3-minimal-dense-path -- <file>` for each file, or copy
  manually. No edits to these files.
- Verify that none of these files import from `nt_ops.kernels.rms_norm` or `nt_ops.kernels.utils`.
  (They do not, per research; `registry.py` imports from `nt_ops.rms`, `nt_ops.rope`,
  `nt_ops.activation` only.)

**Patterns to follow:**
- `registry.py` builder pattern: each `_X_builder(original)` calls into the matching module.
- `runtime.py` install/uninstall lifecycle: transactional, sets `_STATE` on success.

**Test scenarios:**
- `from nt_ops import install, uninstall, get_runtime_state` succeeds without error.
- `apply_monkey_patches()` is importable and emits a deprecation warning (not an error).

**Verification:**
- `python -c "from nt_ops import install; print('ok')"` exits 0 after the copy.
- No `nt_ops.kernels` import in any of the copied files.

---

- [ ] **Unit 2: Add rope kernel support (`kernels/` partial copy)**

**Goal:** Bring in only the ninetoothed rope kernel and its `_cached_make` utility from the refactor
branch. Explicitly exclude the broken `rms_norm.py`.

**Requirements:** R3, R5

**Dependencies:** Unit 1

**Files:**
- Create: `nt_ops/kernels/__init__.py` (empty or minimal, from refactor branch)
- Create: `nt_ops/kernels/rope.py` (from refactor branch, unchanged)
- Create: `nt_ops/kernels/utils.py` (from refactor branch, unchanged)
- **Do not create**: `nt_ops/kernels/rms_norm.py`

**Approach:**
- Copy `nt_ops/kernels/__init__.py`, `nt_ops/kernels/rope.py`, `nt_ops/kernels/utils.py` from the
  refactor branch without modification.
- Confirm that `kernels/rope.py` imports from `ninetoothed` (not `triton` directly) and uses
  `_cached_make` from `kernels/utils.py`.
- Confirm that neither `kernels/rope.py` nor `kernels/utils.py` reference `rms_norm`.

**Patterns to follow:**
- `_cached_make` in `utils.py` wraps `@functools.cache` around `premake_fn` calls for JIT amortization.
- `rotary_position_embedding` in `kernels/rope.py` is the entry point called by `rope.py`'s adapter.

**Test scenarios:**
- `from nt_ops.kernels.rope import rotary_position_embedding` imports without error.
- `from nt_ops.kernels import rms_norm` raises `ImportError` (file must not exist).

**Verification:**
- `python -c "from nt_ops.kernels.rope import rotary_position_embedding"` exits 0.
- `ls nt_ops/kernels/` shows only `__init__.py`, `rope.py`, `utils.py` — not `rms_norm.py`.

---

- [ ] **Unit 3: Merge rms.py — main ninetoothed kernel + new adapter**

**Goal:** Produce `nt_ops/rms.py` that contains main's working ninetoothed kernel classes and
`rms()` dispatch function, wired to the new `build_rms_forward_oot(original)` adapter expected
by `registry.py`. Remove `rms_forward` (the old `forward`-patching function).

**Requirements:** R4, R9

**Dependencies:** Unit 1

**Files:**
- Modify: `nt_ops/rms.py` (start from `main`'s version)
- Test: `tests/test_qwen3_operator_matrix.py` (existing, covers `rms_norm_helper` path)

**Approach:**
- Start from `main`'s `rms.py` (all 4 ninetoothed kernel classes, `rms()` dispatch, all
  `direct_register_custom_op` registrations).
- Add import: `from nt_ops.capabilities import record_hit`.
- Remove: `rms_forward(self, x, residual=None)` function entirely.
- Add helper functions `rms_norm_helper` and `fused_add_rms_norm_helper` that call `rms()` and
  include `record_hit("rms_norm")` / `record_hit("fused_add_rms_norm")` — mirroring the refactor
  branch's helper shape but delegating to `rms()` (not `_rms_norm_kernel`).
- Add `build_rms_forward_oot(original)`: closure over `original`, guards for
  `variance_size_override` and `has_weight=False` fall back to `original`; happy path calls
  `fused_add_rms_norm_helper` or `rms_norm_helper`.
- Verify the residual in-place update semantics: `rms_rw` on main returns `(output, residual)`.
  `build_rms_forward_oot` must ensure `x` and `residual` tensors are mutated in-place as vLLM
  expects from `forward_oot` (check against `test_qwen3_operator_matrix.py` expected behavior).

**Patterns to follow:**
- `build_rms_forward_oot` pattern: `nt_ops/rms.py` on refactor branch (closure shape, guard
  conditions, `record_hit` placement).
- `rms()` dispatch: `nt_ops/rms.py` on main (four-case dispatch by weight/residual presence).

**Test scenarios:**
- `rms_norm_helper(x, weight, eps)` returns a tensor of the same shape/dtype as `x`.
- `fused_add_rms_norm_helper(x, residual, weight, eps)` returns `(x, residual)` with values
  matching the reference (`RMSNorm.forward_static` or equivalent).
- `build_rms_forward_oot(original)` with `variance_size_override` set delegates to `original`.
- `build_rms_forward_oot(original)` with `has_weight=False` delegates to `original`.
- `record_hit("rms_norm")` is called on each non-delegated invocation.

**Verification:**
- `python -c "from nt_ops.rms import build_rms_forward_oot"` exits 0.
- No import of `nt_ops.kernels.rms_norm` anywhere in the file.
- `rms_forward` symbol is absent from `nt_ops/rms.py`.
- `torch.ops.vllm.nt_rms_rw` is registered after `import nt_ops.rms`.

---

- [ ] **Unit 4: Add rope.py and verify activation.py**

**Goal:** Copy `rope.py` adapter from refactor branch (it delegates to `kernels/rope.py`, which is
now in place from Unit 2). Confirm `activation.py` is the refactor branch version with
`silu_and_mul_forward` and `record_hit`.

**Requirements:** R5, R6

**Dependencies:** Unit 2

**Files:**
- Create: `nt_ops/rope.py` (from refactor branch, unchanged)
- Verify/update: `nt_ops/activation.py` (compare main vs refactor; use refactor version)

**Approach:**
- Copy `nt_ops/rope.py` from refactor branch without modification. It imports
  `nt_ops.kernels.rope.rotary_position_embedding` (available after Unit 2) and exports
  `build_rotary_forward_oot(original)`.
- For `activation.py`: the refactor branch version is substantively identical to main's but adds
  `from nt_ops.capabilities import record_hit` and `record_hit("silu_and_mul")` inside
  `silu_and_mul_forward`. Use the refactor branch version. All other activation classes (11 total)
  are unchanged. No other edits.
- No changes to `registry.py` — its `_rope_forward_oot_builder` already calls
  `rope.build_rotary_forward_oot`.

**Patterns to follow:**
- `build_rotary_forward_oot` pattern in the refactor branch's `rope.py`:
  guard for `not is_neox_style or rotary_dim % 2` falls back to `original`; strips `is_cuda` guard.

**Test scenarios:**
- `from nt_ops.rope import build_rotary_forward_oot` imports without error.
- `activation.silu_and_mul_forward` is a callable matching `forward_oot(self, x)` signature.
- `record_hit("silu_and_mul")` is called when `silu_and_mul_forward` executes.

**Verification:**
- `python -c "from nt_ops.rope import build_rotary_forward_oot; from nt_ops.activation import silu_and_mul_forward; print('ok')"` exits 0.
- `activation.py` contains `record_hit` import and call.

---

- [ ] **Unit 5: End-to-end import smoke test and cleanup sweep**

**Goal:** Confirm the new branch assembles correctly: no broken imports, no reference to the
dropped `rms_norm` kernel, `install()` runs through the profile without error in a dry-run context.

**Requirements:** All success criteria from origin document.

**Dependencies:** Units 1–4

**Files:**
- Run: `tests/test_runtime_install.py` (no GPU required)
- Run: `tests/test_capability_report.py` (no GPU required)
- Check: all `nt_ops/*.py` files for any remaining `nt_ops.kernels.rms_norm` references

**Approach:**
- Grep for `kernels.rms_norm` and `kernels/rms_norm` across the entire repo — must return zero hits.
- Grep for `rms_forward` across all files — must return zero hits (the old `forward` patcher).
- Confirm `nt_ops/__init__.py` `__all__` contains exactly the 7 symbols from the refactor branch.
- Run `pytest tests/test_runtime_install.py tests/test_capability_report.py -v`.
- Confirm `apply_monkey_patches()` emits a `DeprecationWarning` (logged) and returns a
  `RuntimeState` without raising (mocked profile context or real import).

**Test scenarios:**
- All tests in `test_runtime_install.py` pass (install lifecycle, process_scope enforcement,
  transactional rollback on partial failure).
- All tests in `test_capability_report.py` pass (hit counter tracking).
- `python -c "import nt_ops; print(nt_ops.__all__)"` prints the expected 7-symbol list.

**Verification:**
- `pytest tests/test_runtime_install.py tests/test_capability_report.py` exits 0.
- Zero grep hits for `rms_norm` and `rms_forward` across `nt_ops/`.

---

## System-Wide Impact

- **Interaction graph:** `NTVLLMWorker.__init__` calls `nt_ops.install()` at worker startup.
  `install()` calls `setattr` on `RMSNorm`, `SiluAndMul`, `RotaryEmbedding` at class level. This
  affects all instances created after install — timing must be before model construction.
- **Error propagation:** `install()` is transactional in `runtime.py`; a failed patch rolls back
  all previously applied patches and raises `InstallationError`. Callers (e.g. `NTVLLMWorker`)
  should not swallow this exception.
- **State lifecycle risks:** `direct_register_custom_op` in `rms.py` runs at module import time.
  If `rms.py` is imported in multiple worker processes, registration may be called multiple times.
  Confirm vLLM's `direct_register_custom_op` is idempotent or add a module-level guard.
- **API surface parity:** `apply_monkey_patches()` shim must remain callable for any existing
  code that calls it directly. The shim must not raise; it should log a deprecation warning and
  delegate to `install()`.
- **Integration coverage:** The GPU-requiring tests (`test_qwen3_operator_matrix.py`,
  `test_qwen3_basic_smoke.py`) are the authoritative validation but cannot run locally. They should
  be noted as the acceptance gate when run on the MLU machine.

## Risks & Dependencies

- **Residual in-place semantics (medium risk):** `build_rms_forward_oot` must mutate `x` and
  `residual` in-place to match vLLM's `forward_oot` contract. Main's `rms_rw` returns a new
  tensor; the adapter must call `.copy_()` or restructure. Verify against
  `test_qwen3_operator_matrix.py` reference behavior before running on MLU.
- **`direct_register_custom_op` double-registration (low risk):** Module-level registration at
  import time can collide in multi-process environments. Wrap with a try/except or check if the op
  is already registered.
- **Rope kernel MLU validation (medium risk):** `kernels/rope.py` is ninetoothed-based and has no
  known MLU failure, but it has never been tested on MLU independently. It may need adjustment.
  If rope fails on MLU, the fallback is to remove the rope `PatchSpec` from the profile, making
  it rms + silu only.
- **`ninetoothed` version compatibility:** Both branches use local workspace installs of
  `ninetoothed` and `ntops`. The new branch stays on `main`'s baseline, which is the known-good
  version.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-03-30-hybrid-dispatch-infrastructure-requirements.md](../brainstorms/2026-03-30-hybrid-dispatch-infrastructure-requirements.md)
- Related code — main kernels: `nt_ops/rms.py` (main branch), `nt_ops/activation.py` (main branch)
- Related code — infrastructure: `nt_ops/runtime.py`, `nt_ops/registry.py`, `nt_ops/capabilities.py`,
  `nt_ops/worker.py` (refactor branch)
- Related code — rope: `nt_ops/kernels/rope.py`, `nt_ops/rope.py` (refactor branch)
- Current broken branch: `refactor/nt-ops-qwen3-minimal-dense-path`
