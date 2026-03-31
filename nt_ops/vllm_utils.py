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


def _is_missing_method_error(exc: Exception, method: str) -> bool:
    message = str(exc)
    return method in message and "not implemented" in message.lower()


def get_vllm_capability_report(llm_or_engine: object) -> dict[str, object]:
    for engine in _iter_engine_candidates(llm_or_engine):
        collective_rpc = getattr(engine, "collective_rpc", None)
        if callable(collective_rpc):
            try:
                return _coerce_report(collective_rpc("get_nt_ops_report"))
            except Exception as exc:
                if not _is_missing_method_error(exc, "get_nt_ops_report"):
                    raise

        model_executor = getattr(engine, "model_executor", None)
        execute_method = getattr(model_executor, "execute_method", None)
        if callable(execute_method):
            try:
                return _coerce_report(execute_method("get_nt_ops_report"))
            except Exception as exc:
                if not _is_missing_method_error(exc, "get_nt_ops_report"):
                    raise

        driver_worker = getattr(model_executor, "driver_worker", None)
        report_method = getattr(driver_worker, "get_nt_ops_report", None)
        if callable(report_method):
            return _coerce_report(report_method())

        worker = getattr(driver_worker, "worker", None)
        report_method = getattr(worker, "get_nt_ops_report", None)
        if callable(report_method):
            return _coerce_report(report_method())

    raise AttributeError(
        "Unable to retrieve nt_ops report from the provided vLLM object. "
        "Expected llm_engine.collective_rpc(...) or model_executor.execute_method(...)."
    )
