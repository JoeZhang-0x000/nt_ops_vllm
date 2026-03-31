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


def register_nt_ops_mlu_platform() -> str:
    return "nt_ops.platforms.mlu.NTOpsMLUPlatform"


__all__ = [
    "InstallationError",
    "RuntimeState",
    "get_capability_report",
    "get_runtime_state",
    "get_vllm_capability_report",
    "install",
    "register_nt_ops_mlu_platform",
    "reset_hits",
    "uninstall",
]
