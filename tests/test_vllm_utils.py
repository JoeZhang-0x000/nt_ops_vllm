from __future__ import annotations

from types import SimpleNamespace

import pytest

import nt_ops


def test_get_vllm_capability_report_uses_collective_rpc():
    expected = {"status": "installed", "hits": {"rms_norm": 1}}

    class Engine:
        def collective_rpc(self, method: str):
            assert method == "get_nt_ops_report"
            return [expected]

    llm = SimpleNamespace(llm_engine=Engine())

    assert nt_ops.get_vllm_capability_report(llm) == expected


def test_get_vllm_capability_report_rejects_inconsistent_report_lists():
    class Engine:
        def collective_rpc(self, method: str):
            assert method == "get_nt_ops_report"
            return [{"status": "installed"}, {"status": "failed"}]

    llm = SimpleNamespace(llm_engine=Engine())

    with pytest.raises(RuntimeError, match="inconsistent nt_ops reports"):
        nt_ops.get_vllm_capability_report(llm)


def test_get_vllm_capability_report_falls_back_to_model_executor():
    expected = {"status": "installed", "hits": {"rms_norm": 1}}

    class ModelExecutor:
        def execute_method(self, method: str):
            assert method == "get_nt_ops_report"
            return [expected]

    llm = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=ModelExecutor()))

    assert nt_ops.get_vllm_capability_report(llm) == expected


def test_get_vllm_capability_report_checks_inner_engine():
    expected = {"status": "installed", "hits": {"rms_norm": 1}}

    class Engine:
        def collective_rpc(self, method: str):
            assert method == "get_nt_ops_report"
            return [expected]

    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine=Engine()))

    assert nt_ops.get_vllm_capability_report(llm) == expected


def test_get_vllm_capability_report_rejects_empty_reports():
    class Engine:
        def collective_rpc(self, method: str):
            assert method == "get_nt_ops_report"
            return []

    with pytest.raises(RuntimeError):
        nt_ops.get_vllm_capability_report(SimpleNamespace(llm_engine=Engine()))


def test_get_vllm_capability_report_falls_back_when_collective_rpc_rejects_method():
    expected = {"status": "installed", "hits": {"rms_norm": 1}}

    class DriverWorker:
        def get_nt_ops_report(self):
            return expected

    class ModelExecutor:
        driver_worker = DriverWorker()

    class Engine:
        model_executor = ModelExecutor()

        def collective_rpc(self, method: str):
            assert method == "get_nt_ops_report"
            raise RuntimeError(
                "Call to collective_rpc method failed: Method 'get_nt_ops_report' is not implemented."
            )

    llm = SimpleNamespace(llm_engine=Engine())

    assert nt_ops.get_vllm_capability_report(llm) == expected


def test_get_vllm_capability_report_checks_engine_core_model_executor():
    expected = {"status": "installed", "hits": {"rms_norm": 1}}

    class DriverWorker:
        def get_nt_ops_report(self):
            return expected

    class ModelExecutor:
        driver_worker = DriverWorker()

    class EngineCore:
        model_executor = ModelExecutor()

    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=EngineCore()))

    assert nt_ops.get_vllm_capability_report(llm) == expected
