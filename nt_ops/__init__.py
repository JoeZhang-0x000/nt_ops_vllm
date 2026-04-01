import os

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

PHASE1_MLU_PLUGIN_NAMES = ("nt_ops_mlu", "nt_ops_mlu_hijack")
PHASE1_MLU_MULTIPROC_METHOD = "spawn"


def register_nt_ops_mlu_platform() -> str:
    return "nt_ops.platforms.mlu.NTOpsMLUPlatform"


def get_phase1_mlu_plugins() -> tuple[str, ...]:
    return PHASE1_MLU_PLUGIN_NAMES


def register_nt_ops_mlu_hijack() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", PHASE1_MLU_MULTIPROC_METHOD)
    from vllm_mlu import register_mlu_hijack

    register_mlu_hijack()


__all__ = [
    "InstallationError",
    "PHASE1_MLU_MULTIPROC_METHOD",
    "RuntimeState",
    "get_capability_report",
    "get_phase1_mlu_plugins",
    "get_runtime_state",
    "get_vllm_capability_report",
    "install",
    "register_nt_ops_mlu_hijack",
    "register_nt_ops_mlu_platform",
    "reset_hits",
    "uninstall",
]
