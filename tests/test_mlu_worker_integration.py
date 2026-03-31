from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_worker_stub(monkeypatch, module_name: str, class_name: str) -> None:
    module = ModuleType(module_name)

    class StubWorker:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    module.__dict__[class_name] = StubWorker
    monkeypatch.setitem(sys.modules, module_name, module)


def test_nt_ops_mlu_workers_install_before_super(monkeypatch):
    _install_worker_stub(monkeypatch, "vllm_mlu.v1.worker.gpu_worker", "MLUWorker")

    calls: list[str] = []

    import nt_ops.workers.mlu as mlu_workers

    monkeypatch.setattr(
        mlu_workers,
        "install",
        lambda *, process_scope: calls.append(process_scope),
    )

    module = importlib.reload(mlu_workers)
    monkeypatch.setattr(
        module,
        "install",
        lambda *, process_scope: calls.append(process_scope),
    )
    worker = module.NTOpsMLUV1Worker("arg", flag=True)

    assert calls == [module.REQUIRED_PROCESS_SCOPE]
    assert worker.args == ("arg",)
    assert worker.kwargs == {"flag": True}


def test_compat_worker_warns_and_uses_mlu_worker(monkeypatch):
    _install_worker_stub(monkeypatch, "vllm_mlu.v1.worker.gpu_worker", "MLUWorker")

    monkeypatch.setenv("VLLM_USE_V1", "1")

    platforms_module = ModuleType("vllm.platforms")
    platforms_module.__dict__["current_platform"] = type(
        "Platform",
        (),
        {"device_type": "mlu"},
    )()
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    envs_module = ModuleType("vllm.envs")
    envs_module.__dict__["VLLM_USE_V1"] = True
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)

    import nt_ops.workers.mlu as mlu_workers

    monkeypatch.setattr(mlu_workers, "install", lambda *, process_scope: None)
    import nt_ops.worker as compat_worker

    compat_worker = importlib.reload(compat_worker)
    monkeypatch.setattr(compat_worker, "install", lambda *, process_scope: None)

    with __import__("pytest").warns(DeprecationWarning):
        worker = compat_worker.NTVLLMWorker()

    assert worker.__class__.__name__ == "_NTOpsMLUV1WorkerImpl"


def test_worker_module_imports_when_only_one_variant_exists(monkeypatch):
    _install_worker_stub(monkeypatch, "vllm_mlu.v1.worker.gpu_worker", "MLUWorker")

    import nt_ops.workers.mlu as mlu_workers

    module = importlib.reload(mlu_workers)
    monkeypatch.setattr(module, "install", lambda *, process_scope: None)
    worker = module.NTOpsMLUV1Worker()

    assert worker.__class__.__name__ == "_NTOpsMLUV1WorkerImpl"
