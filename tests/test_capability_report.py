from __future__ import annotations

import sys
from types import ModuleType

import nt_ops
from nt_ops.registry import PatchSpec


def _install_fake_module(module_name: str, *, value: str) -> ModuleType:
    module = ModuleType(module_name)
    module.target = value
    sys.modules[module_name] = module
    return module


def test_capability_report_tracks_hits(monkeypatch):
    module_name = "fake_nt_runtime_hits"
    _install_fake_module(module_name, value="original")

    def fake_get_profile(profile_name: str):
        del profile_name
        return (
            type(
                "Profile",
                (),
                {
                    "name": "fake",
                    "enabled": ("fake_patch",),
                    "fallback": ("embedding",),
                    "disabled": ("attention",),
                },
            )(),
            (
                PatchSpec(
                    patch_id="fake_patch",
                    module_path=module_name,
                    attr_name="target",
                    builder=lambda original: original,
                ),
            ),
        )

    monkeypatch.setattr("nt_ops.runtime.get_profile", fake_get_profile)
    nt_ops.uninstall()
    nt_ops.install(profile="fake", process_scope="exclusive_qwen3")

    from nt_ops.capabilities import record_hit

    record_hit("rms_norm")
    report = nt_ops.get_capability_report()
    assert report["status"] == "installed"
    assert report["fallback"] == ["embedding"]
    assert report["disabled"] == ["attention"]
    assert report["hits"]["rms_norm"] == 1
