import os
from typing import Callable

from nt_ops.capabilities import reset_hits
from nt_ops.runtime import (
    InstallationError,
    RuntimeState,
    get_capability_report,
    get_runtime_state,
    install,
    uninstall,
)
from nt_ops.vllm_utils import get_vllm_capability_report

PHASE1_MLU_PLUGIN_NAMES = ("mlu", "nt_ops_mlu")
PHASE1_MLU_MULTIPROC_METHOD = "spawn"
_MLU_PLATFORM_PATCH_FLAG = "_nt_ops_worker_patch_applied"


def register_nt_ops_mlu_platform() -> str:
    return "nt_ops.platforms.mlu.NTOpsMLUPlatform"


def get_phase1_mlu_plugins() -> tuple[str, ...]:
    return PHASE1_MLU_PLUGIN_NAMES


def _rewrite_worker_qualname(worker_cls: str) -> str:
    if worker_cls == "vllm_mlu.v1.worker.gpu_worker.MLUWorker":
        return "nt_ops.workers.mlu.NTOpsMLUV1Worker"
    if worker_cls == "vllm_mlu.worker.worker.MLUWorker":
        return "nt_ops.workers.mlu.NTOpsMLUV0Worker"
    return worker_cls


def _patch_vllm_mlu_platform() -> None:
    from vllm_mlu.platforms.mlu import MLUPlatform

    if getattr(MLUPlatform, _MLU_PLATFORM_PATCH_FLAG, False):
        return

    original: Callable = getattr(MLUPlatform, "check_and_update_config")

    def check_and_update_config(cls, vllm_config) -> None:
        original(vllm_config)

        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = _rewrite_worker_qualname(
            parallel_config.worker_cls,
        )

        sd_worker_cls = getattr(parallel_config, "sd_worker_cls", None)
        if sd_worker_cls is not None:
            parallel_config.sd_worker_cls = _rewrite_worker_qualname(sd_worker_cls)

    setattr(MLUPlatform, "check_and_update_config", classmethod(check_and_update_config))
    setattr(MLUPlatform, _MLU_PLATFORM_PATCH_FLAG, True)


def register_nt_ops_mlu_plugin() -> None:
    os.environ.setdefault(
        "VLLM_WORKER_MULTIPROC_METHOD",
        PHASE1_MLU_MULTIPROC_METHOD,
    )
    from vllm_mlu import register_mlu_hijack

    register_mlu_hijack()
    _patch_vllm_mlu_platform()


def register_nt_ops_mlu_hijack() -> None:
    register_nt_ops_mlu_plugin()


__all__ = [
    "InstallationError",
    "PHASE1_MLU_MULTIPROC_METHOD",
    "RuntimeState",
    "get_capability_report",
    "get_phase1_mlu_plugins",
    "get_runtime_state",
    "get_vllm_capability_report",
    "install",
    "register_nt_ops_mlu_plugin",
    "register_nt_ops_mlu_hijack",
    "register_nt_ops_mlu_platform",
    "reset_hits",
    "uninstall",
]
