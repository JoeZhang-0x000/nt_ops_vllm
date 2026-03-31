from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path


def _append_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_append_path(WORKSPACE_ROOT / "vllm")
_append_path(WORKSPACE_ROOT / "ninetoothed")


def _ensure_vllm_logger_stub() -> None:
    if "vllm.logger" in sys.modules:
        return

    try:
        import torch  # noqa: F401
        import vllm.logger  # noqa: F401
        return
    except Exception:
        pass

    vllm_module = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")

    class _StubLogger:
        def info(self, *args, **kwargs):
            del args, kwargs

        warning = info
        error = info
        debug = info
        info_once = info
        warning_once = info

    def init_logger(name: str):
        del name
        return _StubLogger()

    logger_module.init_logger = init_logger
    vllm_module.logger = logger_module
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules["vllm.logger"] = logger_module


_ensure_vllm_logger_stub()
