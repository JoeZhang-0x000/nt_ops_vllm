from __future__ import annotations

from types import SimpleNamespace

import nt_ops.mlu_dispatch as mlu_dispatch


def test_mlu_active_wrapper_intercepts_silu_gated(monkeypatch):
    calls: list[tuple[object, str, bool]] = []
    fake_input = SimpleNamespace(ndim=2, shape=(4, 8), dtype="bf16")

    def fake_original(input, act_mode: str, is_gated: bool):
        calls.append((input, act_mode, is_gated))
        return "original"

    monkeypatch.setattr(mlu_dispatch, "_run_silu_and_mul", lambda x: ("nt", x))

    wrapped = mlu_dispatch.build_mlu_active_wrapper(fake_original)

    assert wrapped(fake_input, "silu", True) == ("nt", fake_input)
    assert calls == []


def test_mlu_active_wrapper_falls_back_for_non_silu():
    calls: list[tuple[object, str, bool]] = []
    fake_input = SimpleNamespace(ndim=2, shape=(4, 8), dtype="bf16")

    def fake_original(input, act_mode: str, is_gated: bool):
        calls.append((input, act_mode, is_gated))
        return "original"

    wrapped = mlu_dispatch.build_mlu_active_wrapper(fake_original)

    assert wrapped(fake_input, "gelu", True) == "original"
    assert calls == [(fake_input, "gelu", True)]


def test_mlu_active_wrapper_falls_back_for_unsupported_shape():
    calls: list[tuple[object, str, bool]] = []
    fake_input = SimpleNamespace(ndim=4, shape=(1, 2, 3, 8), dtype="bf16")

    def fake_original(input, act_mode: str, is_gated: bool):
        calls.append((input, act_mode, is_gated))
        return "original"

    wrapped = mlu_dispatch.build_mlu_active_wrapper(fake_original)

    assert wrapped(fake_input, "silu", True) == "original"
    assert calls == [(fake_input, "silu", True)]
