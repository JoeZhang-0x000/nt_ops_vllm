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
