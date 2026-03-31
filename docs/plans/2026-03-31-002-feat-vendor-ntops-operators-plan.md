---
title: "feat: Vendor ntops operators into nt_ops_vllm"
type: feat
status: active
date: 2026-03-31
---

# feat: Vendor ntops operators into nt_ops_vllm

## Overview

The project currently relies on a sibling workspace directory (`../ntops/`) for its kernel implementations. That upstream repo is no longer actively maintained. This plan brings the entire `ntops` package into this repo so the team owns and can modify all operator code without external dependencies.

## Problem Frame

`nt_ops_vllm` kernels are built on top of `ninetoothed` using operator implementations that originated in the InfiniTensor `ntops` project. Currently `tests/conftest.py` patches `sys.path` to point at the sibling workspace directory. This means:
- Operators can silently change under our feet if someone pulls `ntops`
- We cannot modify ntops operators without touching a separate repo
- The dependency is invisible to package consumers (not listed in `setup.py`)

Vendoring brings the code inside the boundary of this repo, making it auditable, modifiable, and version-controlled alongside the vLLM operator adapters.

## Requirements Trace

- R1. All 41 ntops operators (kernels + torch wrappers) live inside this repo
- R2. `import ntops` continues to work with no changes to existing source files
- R3. `sys.path` manipulation in `conftest.py` for ntops is removed
- R4. `setup.py` installs the vendored `ntops` package alongside `nt_ops`, with `python_requires>=3.10` and `install_requires` including torch and ninetoothed
- R5. Tests pass without the external `ntops/` sibling directory being present
- R6. The upstream `ntops` pip distribution must be absent before installing this repo — one supported install state, no coexistence

## Scope Boundaries

- **Not in scope**: Merging ntops operators into the `nt_ops/` namespace — the two packages remain independent
- **Not in scope**: Modifying any ntops operator logic during the copy
- **Not in scope**: Adding new tests for ntops operators (ntops has its own test suite; we can adopt it separately)
- **Not in scope**: Removing `ninetoothed` from the `sys.path` hack — that dependency stays as-is

## Context & Research

### Relevant Code and Patterns

- Source to copy: `../ntops/src/ntops/` → `ntops/` (repo root)
- `setup.py`: uses `find_packages()` — will auto-discover `ntops/` with no explicit change needed
- `tests/conftest.py:16`: `_append_path(WORKSPACE_ROOT / "ntops" / "src")` — the one line to remove
- `nt_ops/kernels/rope.py:1`: docstring says "inlined from ntops" — this is a separate copy that already lives in the repo; no conflict

### ntops Package Structure

```
ntops/src/ntops/
  __init__.py
  kernels/          ← 41 ninetoothed kernel implementations
    __init__.py
    element_wise.py, reduction.py, pooling.py   ← shared arrangement patterns
    abs.py, add.py, sub.py, mul.py, div.py, pow.py, neg.py
    exp.py, sin.py, cos.py, sigmoid.py, silu.py, tanh.py, relu.py
    rsqrt.py, clamp.py, dropout.py
    eq.py, ne.py, lt.py, le.py, gt.py, ge.py, isnan.py, isinf.py
    bitwise_and.py, bitwise_or.py, bitwise_not.py
    mm.py, bmm.py, addmm.py, conv2d.py
    max_pool2d.py, avg_pool2d.py, pooling.py
    gelu.py, softmax.py, layer_norm.py, rms_norm.py
    rotary_position_embedding.py, scaled_dot_product_attention.py
  torch/            ← PyTorch wrapper functions (42 files incl. matmul dispatcher)
    __init__.py
    utils.py        ← _cached_make, config helpers
    <one .py per operator>
```

### Institutional Learnings

- `nt_ops/kernels/rope.py` was previously inlined from ntops — the vendored copy will sit at `ntops/kernels/rotary_position_embedding.py`. No deduplication needed at this stage; the two implementations serve different call paths.
- `_cached_make` exists in both `ntops/torch/utils.py` and `nt_ops/kernels/utils.py` — these remain independent; no refactor needed.

## Key Technical Decisions

- **Top-level package, not sub-package**: Place the vendored code at `ntops/` (repo root) rather than `nt_ops/ntops/`. This keeps `import ntops` working with zero source changes and matches the original package namespace.
- **Verbatim copy**: Copy all files without modification. The goal is ownership, not refactoring.
- **`setup.py` carries upstream metadata as a breaking change**: The upstream `ntops` declares `requires-python = ">=3.10"` and `dependencies = ["ninetoothed>=0.16.0", "torch"]`. These constraints are deliberately carried into this repo's `setup.py`. Raising `python_requires` to `>=3.10` is an explicit breaking change — this workspace is the sole consumer of `nt_ops` and already targets MLU hardware requiring Python 3.10+, so the break is acceptable.
- **One supported install state — no coexistence**: The migration path is strictly: `pip uninstall ntops` (if present), then `pip install -e .`. Running with both the upstream `ntops` distribution and this repo installed simultaneously is an unsupported state. The README must document this.
- **Remove only the ntops sys.path line**: Leave the `vllm` and `ninetoothed` path additions in `conftest.py` untouched — those are separate workspace dependencies.

## Open Questions

### Resolved During Planning

- **Do any runtime files import from ntops?** No. Grep confirms zero runtime imports; only `conftest.py` adds ntops to sys.path.
- **Does `find_packages()` need a `where` argument?** No — the vendored `ntops/` will be at the repo root, same level as `nt_ops/`, which `find_packages()` already discovers.
- **Are there conflicting operator names between nt_ops and ntops?** The two packages are independent namespaces (`nt_ops.*` vs `ntops.*`). No conflicts.

### Deferred to Implementation

- Whether to adopt ntops's own test suite into this repo's `tests/` — deferred; can be done as follow-up.
- Whether to eventually merge ntops kernel implementations with `nt_ops/kernels/` — deferred; requires careful review of diverged implementations (e.g., rope).

## Implementation Units

- [ ] **Unit 1: Copy ntops source tree into repo**

**Goal:** Add `ntops/` as a first-class package in the repo, mirroring the structure from `ntops/src/ntops/`

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Create: `ntops/` (full directory tree — ~86 Python files)
  - `ntops/__init__.py`
  - `ntops/kernels/__init__.py` + all 41 kernel files
  - `ntops/torch/__init__.py` + all 42 torch wrapper files
  - `ntops/torch/utils.py`

**Approach:**
- Copy the contents of `../ntops/src/ntops/` verbatim into `ntops/` at the repo root
- Preserve all file contents, subdirectory layout, and `__init__.py` exports exactly as-is
- No modifications to any copied file

**Patterns to follow:** Existing `nt_ops/` package layout (top-level package, `__init__.py` at root)

**Test scenarios:**
- `python -c "import ntops; print(ntops.__version__)"` succeeds after `pip install -e .`
- `python -c "from ntops.torch import rms_norm"` succeeds
- `python -c "from ntops.kernels import rotary_position_embedding"` succeeds

**Verification:**
- All 41 kernel modules importable under `ntops.kernels.*`
- All 42 torch wrapper modules importable under `ntops.torch.*`

---

- [ ] **Unit 2: Remove ntops sys.path hack from conftest.py**

**Goal:** Stop relying on the external workspace sibling directory for tests

**Requirements:** R3, R5

**Dependencies:** Unit 1 (vendored package must be installed first)

**Files:**
- Modify: `tests/conftest.py`

**Approach:**
- Delete the single line: `_append_path(WORKSPACE_ROOT / "ntops" / "src")`
- Leave all other `_append_path` calls untouched (vllm, ninetoothed)
- The `_append_path` helper itself can stay unless it becomes unused — check if vllm and ninetoothed lines remain

**Test scenarios:**
- Tests run successfully with the external `ntops/` workspace directory removed from `sys.path`
- No `ModuleNotFoundError: No module named 'ntops'` errors

**Verification:**
- `pytest tests/` passes (or fails only for unrelated reasons such as missing vllm/MLU hardware)
- `python -c "import ntops"` works from the repo root without any sys.path manipulation

---

- [ ] **Unit 3: Update setup.py with upstream metadata**

**Goal:** Ensure `setup.py` correctly declares all dependencies inherited from ntops so the installed package is safe to use

**Requirements:** R4, R6

**Dependencies:** Unit 1

**Files:**
- Modify: `setup.py`

**Approach:**
- Raise `python_requires` from `>=3.6` to `>=3.10` (upstream ntops requires 3.10)
- Add `install_requires=["torch", "ninetoothed>=0.16.0"]` to make the dependency on both packages explicit
- `find_packages()` already discovers `ntops/` automatically — no package list changes needed

**Test scenarios:**
- `pip install -e .` completes without errors
- `pip show nt-ops` lists torch and ninetoothed as requirements
- Attempting install in a Python 3.9 environment produces a version mismatch error

**Verification:**
- `setup.py` `python_requires` and `install_requires` match the upstream ntops `pyproject.toml`

---

- [ ] **Unit 4: Add automated import coverage test for vendored ntops**

**Goal:** Ensure CI catches missing files, broken exports, or packaging failures in the vendored package — not just manual smoke checks

**Requirements:** R4, R5

**Dependencies:** Unit 1, Unit 2, Unit 3

**Files:**
- Create: `tests/test_ntops_import.py`

**Approach:**
- Add a pytest test file that imports a representative cross-section of the vendored package: top-level `ntops`, at least one kernel (`ntops.kernels.rotary_position_embedding`), and at least one torch wrapper (`ntops.torch.rms_norm`)
- The test does not execute kernels (that requires ninetoothed JIT + hardware) — import-level coverage is sufficient to catch missing files and export drift
- The test should be skippable if ninetoothed is not installed (use `pytest.importorskip` or a skip marker), so CI without the full ML stack still runs the rest of the suite

**Test scenarios:**
- `import ntops` succeeds
- `from ntops.kernels import rotary_position_embedding, rms_norm` succeeds
- `from ntops.torch import rms_norm, rotary_position_embedding` succeeds
- Test fails if any of the ~86 vendored files were accidentally omitted during the copy

**Verification:**
- `pytest tests/test_ntops_import.py` passes after vendoring
- `pytest tests/` passes with the ntops sys.path line removed from conftest.py

---

- [ ] **Unit 5: Validate install from outside the repo directory**

**Goal:** Confirm the package is importable from a working directory that is not the repo root, where the source tree is not implicitly on `sys.path`

**Requirements:** R4, R6

**Dependencies:** Unit 1, Unit 2, Unit 3

**Files:**
- No source changes — manual verification step

**Approach:**
- `pip uninstall ntops` to remove any upstream distribution
- `pip install -e <repo_path>` from a fresh shell
- `cd /tmp && python -c "import ntops; from ntops.torch import rms_norm; from ntops.kernels import rotary_position_embedding"`
- Confirm the import resolves to the vendored copy, not a sys.path-injected source tree

**Verification:**
- All imports succeed from outside the repo directory with no manual `sys.path` manipulation
- `pip show nt-ops` lists `torch` and `ninetoothed` in Requires

## System-Wide Impact

- **Import graph:** No runtime code changes. The only import-path change is removing the conftest.py sys.path line, which only affects test discovery.
- **Error propagation:** If Unit 1 is incomplete (missing files), Unit 3 will surface `ImportError` at import time.
- **State lifecycle risks:** None — this is a pure file-copy operation with no logic changes.
- **API surface parity:** `ntops` public API remains byte-for-byte identical to the upstream source.
- **Integration coverage:** Existing `tests/test_runtime_install.py` and `tests/test_capability_report.py` do not import from `ntops` directly, so they are unaffected.

## Risks & Dependencies

- **ninetoothed version**: `ntops` requires `ninetoothed>=0.16.0`. If the workspace `ninetoothed/` install is older, imports may fail at kernel compile time. Verify version before running.
- **Namespace collision with upstream pip package**: Only one supported install state exists — upstream `ntops` must be absent. The migration procedure is `pip uninstall ntops && pip install -e .`. Coexistence is unsupported and untested; document this in the README.
- **`python_requires` alignment**: This repo currently declares `>=3.6` but the vendored ntops code uses Python 3.10+ features (`X | Y` union syntax, `match` statements, `list[T]` generics). Install into Python 3.9 will succeed but fail at import. Unit 3 fixes this.
- **Diverged rope implementation**: `nt_ops/kernels/rope.py` was inlined from ntops at some point and may have diverged. The vendored `ntops/kernels/rotary_position_embedding.py` is a different file at a different import path — no conflict at copy time, but worth noting for future consolidation.
- **Kernel compile on import**: ntops kernels call `premake()` at module import time. First import after vendoring may be slow (ninetoothed JIT compilation). This is expected behavior, not a regression.

## Sources & References

- External ntops repo: https://github.com/InfiniTensor/ntops
- Source to copy: `/Users/bytedance/Desktop/nt_workspace/ntops/src/ntops/`
- Related solution: `docs/solutions/integration-issues/rope-forward-oot-bypassed-mlu-flashattention-2026-03-31.md`
- Related plan: `docs/plans/2026-03-31-001-refactor-hybrid-dispatch-main-kernels-plan.md`
