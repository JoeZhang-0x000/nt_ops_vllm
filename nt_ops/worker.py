from __future__ import annotations

import importlib
import os

from vllm.logger import init_logger
from vllm.platforms import current_platform

from nt_ops.runtime import REQUIRED_PROCESS_SCOPE, get_capability_report, install

logger = init_logger(__name__)

_BASE_WORKER_ENV_VAR = "NT_OPS_VLLM_BASE_WORKER_CLS"
_DEFAULT_BASE_WORKER_CANDIDATES = (
    "vllm_mlu.v1.worker.gpu_worker.MLUWorker",
    "vllm_mlu.worker.worker.MLUWorker",
    "vllm_mlu.v1.worker.worker.MLUWorker",
    "vllm_mlu.worker.mlu_worker.MLUWorker",
    "vllm_mlu.v1.worker.mlu_worker.MLUWorker",
)


def _resolve_obj_by_qualname(qualname: str) -> object:
    module_name, _, attr_path = qualname.partition(":")
    if not attr_path:
        module_name, _, attr_path = qualname.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid qualified name: {qualname!r}")

    obj = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _resolve_base_worker_cls() -> type:
    candidates: list[str] = []
    override = os.environ.get(_BASE_WORKER_ENV_VAR)
    platform_device_type = getattr(current_platform, "device_type", "unknown")
    if override:
        candidates.append(override)
    candidates.extend(_DEFAULT_BASE_WORKER_CANDIDATES)
    if getattr(current_platform, "is_cuda_alike", lambda: False)():
        candidates.append("vllm.v1.worker.gpu_worker.Worker")

    errors: list[str] = []
    for qualname in candidates:
        try:
            obj = _resolve_obj_by_qualname(qualname)
        except Exception as exc:
            errors.append(f"{qualname}: {exc}")
            continue

        if isinstance(obj, type):
            logger.info("Resolved nt_ops base worker class: %s", qualname)
            return obj

        errors.append(f"{qualname}: resolved object is not a class")

    raise ImportError(
        "Unable to resolve a base vLLM worker class for nt_ops. "
        f"Detected platform device_type={platform_device_type!r}. "
        f"Set {_BASE_WORKER_ENV_VAR} to the fully qualified worker class path. "
        f"Attempts: {'; '.join(errors)}"
    )


class NTVLLMWorker(_resolve_base_worker_cls()):
    def __init__(self, *args, **kwargs):
        install(process_scope=REQUIRED_PROCESS_SCOPE)
        super().__init__(*args, **kwargs)

    def get_nt_ops_report(self) -> dict[str, object]:
        return get_capability_report()
