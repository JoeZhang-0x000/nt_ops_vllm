import logging

from nt_ops.runtime import REQUIRED_PROCESS_SCOPE, RuntimeState, install

logger = logging.getLogger(__name__)


def apply_monkey_patches() -> RuntimeState:
    logger.warning(
        "nt_ops.monkey_patch.apply_monkey_patches() is deprecated. "
        "Prefer the MLU platform/device path; use nt_ops.install(profile='qwen3_minimal_dense', process_scope=%r) "
        "only for compatibility or direct runtime testing.",
        REQUIRED_PROCESS_SCOPE,
    )
    return install(
        profile="qwen3_minimal_dense",
        process_scope=REQUIRED_PROCESS_SCOPE,
    )
