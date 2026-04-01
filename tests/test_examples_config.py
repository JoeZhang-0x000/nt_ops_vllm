from __future__ import annotations

from pathlib import Path

import nt_ops


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_examples_do_not_depend_on_manual_nt_ops_worker_override():
    basic = (REPO_ROOT / "examples" / "basic.py").read_text()
    benchmark = (REPO_ROOT / "examples" / "benchmark_throughput.py").read_text()

    assert "worker_cls=" not in basic
    assert "worker_cls=" not in benchmark
    assert "NTVLLMWorker" not in basic
    assert "NTVLLMWorker" not in benchmark


def test_readme_documents_phase1_plugin_and_spawn_contract():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "nt_ops_mlu,nt_ops_mlu_hijack" in readme
    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in readme
    assert "required part of the runtime stack" in readme


def test_setup_exports_both_phase1_plugin_entrypoints():
    setup_py = (REPO_ROOT / "setup.py").read_text()

    for plugin_name in nt_ops.get_phase1_mlu_plugins():
        assert plugin_name in setup_py
