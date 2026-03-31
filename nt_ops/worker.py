from __future__ import annotations

import importlib
import logging
import os
import warnings

from nt_ops.runtime import REQUIRED_PROCESS_SCOPE, install

logger = logging.getLogger(__name__)

_BASE_WORKER_ENV_VAR = "NT_OPS_VLLM_BASE_WORKER_CLS"


def _resolve_base_worker_cls() -> type:
    envs = importlib.import_module("vllm.envs")
    current_platform = importlib.import_module("vllm.platforms").current_platform
    mlu_workers = importlib.import_module("nt_ops.workers.mlu")

    override = os.environ.get(_BASE_WORKER_ENV_VAR)
    if override:
        logger.info(
            "Resolved nt_ops compatibility worker class from override: %s", override
        )
        return mlu_workers._resolve_worker_cls(override)

    device_type = getattr(current_platform, "device_type", "unknown")
    if device_type != "mlu":
        raise ImportError(
            "nt_ops.worker.NTVLLMWorker is only supported for MLU compatibility mode. "
            f"Detected platform device_type={device_type!r}."
        )

    return (
        mlu_workers.NTOpsMLUV1Worker
        if envs.VLLM_USE_V1
        else mlu_workers.NTOpsMLUV0Worker
    )


class NTVLLMWorker:
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "nt_ops.worker.NTVLLMWorker is deprecated; prefer MLU platform/device "
            "selection instead of manual worker_cls injection.",
            DeprecationWarning,
            stacklevel=2,
        )
        install(process_scope=REQUIRED_PROCESS_SCOPE)
        return _resolve_base_worker_cls()(*args, **kwargs)
