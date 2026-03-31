from __future__ import annotations

import sys
from types import ModuleType

import pytest

import nt_ops
from nt_ops.registry import PatchSpec
from nt_ops.runtime import InstallationError


@pytest.fixture(autouse=True)
def clean_runtime():
    nt_ops.uninstall()
    yield
    nt_ops.uninstall()


def _install_fake_module(module_name: str, *, value: str) -> ModuleType:
    module = ModuleType(module_name)
    module.target = value
    sys.modules[module_name] = module
    return module


def test_import_does_not_install_runtime():
    state = nt_ops.get_runtime_state()
    assert state.status == "not_installed"
    assert state.applied_patch_ids == ()


def test_install_requires_explicit_process_scope():
    with pytest.raises(InstallationError):
        nt_ops.install()


def test_install_applies_profile_transactionally(monkeypatch):
    module_name = "fake_nt_runtime_ok"
    module = _install_fake_module(module_name, value="original")

    def fake_get_profile(profile_name: str):
        del profile_name
        return (
            type(
                "Profile",
                (),
                {
                    "name": "fake",
                    "enabled": ("fake_patch",),
                    "fallback": (),
                    "disabled": (),
                },
            )(),
            (
                PatchSpec(
                    patch_id="fake_patch",
                    module_path=module_name,
                    attr_name="target",
                    builder=lambda original: f"{original}_patched",
                ),
            ),
        )

    monkeypatch.setattr("nt_ops.runtime.get_profile", fake_get_profile)

    state = nt_ops.install(profile="fake", process_scope="exclusive_qwen3")

    assert state.status == "installed"
    assert module.target == "original_patched"
    assert "fake_patch" in state.applied_patch_ids


def test_install_rolls_back_when_patch_fails(monkeypatch):
    module_name = "fake_nt_runtime_fail"
    module = _install_fake_module(module_name, value="original")

    def fake_get_profile(profile_name: str):
        del profile_name
        return (
            type(
                "Profile",
                (),
                {
                    "name": "fake",
                    "enabled": ("ok", "broken"),
                    "fallback": (),
                    "disabled": (),
                },
            )(),
            (
                PatchSpec(
                    patch_id="ok",
                    module_path=module_name,
                    attr_name="target",
                    builder=lambda original: f"{original}_patched",
                ),
                PatchSpec(
                    patch_id="broken",
                    module_path=module_name,
                    attr_name="target",
                    builder=lambda original: (_ for _ in ()).throw(
                        RuntimeError("boom")
                    ),
                ),
            ),
        )

    monkeypatch.setattr("nt_ops.runtime.get_profile", fake_get_profile)

    with pytest.raises(InstallationError):
        nt_ops.install(profile="fake", process_scope="exclusive_qwen3")

    state = nt_ops.get_runtime_state()
    assert state.status == "failed_rolled_back"
    assert module.target == "original"
