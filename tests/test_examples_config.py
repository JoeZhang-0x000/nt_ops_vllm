from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_examples_do_not_depend_on_manual_nt_ops_worker_override():
    basic = (REPO_ROOT / "examples" / "basic.py").read_text()
    benchmark = (REPO_ROOT / "examples" / "benchmark_throughput.py").read_text()

    assert "worker_cls=" not in basic
    assert "worker_cls=" not in benchmark
    assert "NTVLLMWorker" not in basic
    assert "NTVLLMWorker" not in benchmark
