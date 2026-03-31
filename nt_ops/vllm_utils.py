from __future__ import annotations

from collections.abc import Iterator


def _iter_engine_candidates(llm_or_engine: object) -> Iterator[object]:
    seen: set[int] = set()
    candidates = [getattr(llm_or_engine, "llm_engine", None), llm_or_engine]

    for candidate in candidates:
        if candidate is None:
            continue

        for engine in (candidate, getattr(candidate, "engine", None)):
            if engine is None or id(engine) in seen:
                continue
            seen.add(id(engine))
            yield engine


def _coerce_report(report_or_reports: object) -> dict[str, object]:
    if isinstance(report_or_reports, list):
        if not report_or_reports:
            raise RuntimeError("vLLM returned an empty nt_ops report list")
        report_or_reports = report_or_reports[0]

    if not isinstance(report_or_reports, dict):
        raise TypeError(
            "vLLM returned an nt_ops report with unexpected type "
            f"{type(report_or_reports).__name__}"
        )

    return report_or_reports


def get_vllm_capability_report(llm_or_engine: object) -> dict[str, object]:
    for engine in _iter_engine_candidates(llm_or_engine):
        collective_rpc = getattr(engine, "collective_rpc", None)
        if callable(collective_rpc):
            return _coerce_report(collective_rpc("get_nt_ops_report"))

        model_executor = getattr(engine, "model_executor", None)
        execute_method = getattr(model_executor, "execute_method", None)
        if callable(execute_method):
            return _coerce_report(execute_method("get_nt_ops_report"))

    raise AttributeError(
        "Unable to retrieve nt_ops report from the provided vLLM object. "
        "Expected llm_engine.collective_rpc(...) or model_executor.execute_method(...)."
    )
