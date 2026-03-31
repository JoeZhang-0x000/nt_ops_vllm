from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from vllm.logger import init_logger

from nt_ops.capabilities import build_capability_report, reset_hits
from nt_ops.registry import PatchSpec, get_profile

logger = init_logger(__name__)

REQUIRED_PROCESS_SCOPE = "exclusive_qwen3"


class InstallationError(RuntimeError):
    pass


@dataclass
class RuntimeState:
    status: str = "not_installed"
    profile: str | None = None
    process_scope: str | None = None
    capability_report: dict[str, object] = field(default_factory=dict)
    applied_patch_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class _PreparedPatch:
    spec: PatchSpec
    target_obj: object
    replacement: object
    original: object


@dataclass
class _AppliedPatch:
    spec: PatchSpec
    target_obj: object
    original: object


_STATE = RuntimeState()
_APPLIED_PATCHES: list[_AppliedPatch] = []


def get_capability_report() -> dict[str, object]:
    if _STATE.profile is None:
        return {"status": _STATE.status, "hits": {}, "errors": list(_STATE.errors)}

    capability_profile, _ = get_profile(_STATE.profile)
    return build_capability_report(
        capability_profile,
        status=_STATE.status,
        errors=tuple(_STATE.errors),
    )


def get_runtime_state() -> RuntimeState:
    return RuntimeState(
        status=_STATE.status,
        profile=_STATE.profile,
        process_scope=_STATE.process_scope,
        capability_report=get_capability_report(),
        applied_patch_ids=tuple(_STATE.applied_patch_ids),
        errors=tuple(_STATE.errors),
    )


def _resolve_target(spec: PatchSpec) -> tuple[object, object]:
    module = importlib.import_module(spec.module_path)
    target_obj = getattr(module, spec.object_name) if spec.object_name else module
    original = getattr(target_obj, spec.attr_name)
    return target_obj, original


def _prepare_patches(specs: tuple[PatchSpec, ...]) -> tuple[list[_PreparedPatch], list[str]]:
    prepared: list[_PreparedPatch] = []
    errors: list[str] = []
    for spec in specs:
        try:
            target_obj, original = _resolve_target(spec)
            if spec.builder is None:
                replacement = original
            else:
                replacement = spec.builder(original)
            prepared.append(
                _PreparedPatch(
                    spec=spec,
                    target_obj=target_obj,
                    replacement=replacement,
                    original=original,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive path
            if spec.required:
                errors.append(f"{spec.patch_id}: {exc}")
            else:
                logger.warning("Skipping optional nt_ops patch %s: %s", spec.patch_id, exc)
    return prepared, errors


def _apply_prepared_patches(prepared: list[_PreparedPatch]) -> list[_AppliedPatch]:
    applied: list[_AppliedPatch] = []
    try:
        for patch in prepared:
            setattr(patch.target_obj, patch.spec.attr_name, patch.replacement)
            applied.append(
                _AppliedPatch(
                    spec=patch.spec,
                    target_obj=patch.target_obj,
                    original=patch.original,
                )
            )
    except Exception:
        for patch in reversed(applied):
            setattr(patch.target_obj, patch.spec.attr_name, patch.original)
        raise
    return applied


def uninstall() -> RuntimeState:
    global _APPLIED_PATCHES, _STATE
    for patch in reversed(_APPLIED_PATCHES):
        setattr(patch.target_obj, patch.spec.attr_name, patch.original)
    _APPLIED_PATCHES = []
    reset_hits()
    _STATE = RuntimeState()
    return get_runtime_state()


def install(
    *,
    profile: str = "qwen3_minimal_dense",
    process_scope: str | None = None,
) -> RuntimeState:
    global _APPLIED_PATCHES, _STATE

    if process_scope != REQUIRED_PROCESS_SCOPE:
        raise InstallationError(
            f"nt_ops profile '{profile}' requires process_scope={REQUIRED_PROCESS_SCOPE!r}."
        )

    if _STATE.status == "installed":
        if _STATE.profile == profile and _STATE.process_scope == process_scope:
            return get_runtime_state()
        raise InstallationError(
            "nt_ops is already installed with a different runtime profile."
        )

    uninstall()
    capability_profile, specs = get_profile(profile)
    prepared, errors = _prepare_patches(specs)
    if errors:
        _STATE = RuntimeState(
            status="failed_rolled_back",
            profile=profile,
            process_scope=process_scope,
            capability_report=build_capability_report(
                capability_profile,
                status="failed_rolled_back",
                errors=tuple(errors),
            ),
            errors=tuple(errors),
        )
        raise InstallationError("; ".join(errors))

    try:
        _APPLIED_PATCHES = _apply_prepared_patches(prepared)
    except Exception as exc:  # pragma: no cover - defensive path
        errors = [f"apply_failed: {exc}"]
        _STATE = RuntimeState(
            status="failed_rolled_back",
            profile=profile,
            process_scope=process_scope,
            capability_report=build_capability_report(
                capability_profile,
                status="failed_rolled_back",
                errors=tuple(errors),
            ),
            errors=tuple(errors),
        )
        raise InstallationError("; ".join(errors)) from exc

    reset_hits()
    _STATE = RuntimeState(
        status="installed",
        profile=profile,
        process_scope=process_scope,
        capability_report=build_capability_report(
            capability_profile,
            status="installed",
        ),
        applied_patch_ids=tuple(patch.spec.patch_id for patch in _APPLIED_PATCHES),
    )
    logger.info(
        "Installed nt_ops profile %s with patches: %s",
        profile,
        ", ".join(_STATE.applied_patch_ids),
    )
    return get_runtime_state()
