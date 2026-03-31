from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import nt_ops


def _install_vllm_mlu_platform_stub(monkeypatch) -> None:
    class StubMLUPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            parallel_config = vllm_config.parallel_config
            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = "vllm_mlu.v1.worker.gpu_worker.MLUWorker"
                parallel_config.sd_worker_cls = "vllm_mlu.worker.worker.MLUWorker"

    module = ModuleType("vllm_mlu.platforms.mlu")
    module.__dict__["MLUPlatform"] = StubMLUPlatform

    monkeypatch.setitem(sys.modules, "vllm_mlu", ModuleType("vllm_mlu"))
    monkeypatch.setitem(
        sys.modules, "vllm_mlu.platforms", ModuleType("vllm_mlu.platforms")
    )
    monkeypatch.setitem(sys.modules, "vllm_mlu.platforms.mlu", module)


def test_register_nt_ops_mlu_platform_returns_platform_path():
    assert (
        nt_ops.register_nt_ops_mlu_platform() == "nt_ops.platforms.mlu.NTOpsMLUPlatform"
    )


def test_platform_rewrites_vllm_mlu_worker_classes(monkeypatch):
    _install_vllm_mlu_platform_stub(monkeypatch)
    module = importlib.import_module("nt_ops.platforms.mlu")
    module = importlib.reload(module)

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls="auto", sd_worker_cls=None),
    )

    module.NTOpsMLUPlatform.check_and_update_config(config)

    assert config.parallel_config.worker_cls == "nt_ops.workers.mlu.NTOpsMLUV1Worker"
    assert config.parallel_config.sd_worker_cls == "nt_ops.workers.mlu.NTOpsMLUV0Worker"
