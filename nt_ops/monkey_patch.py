from vllm.logger import init_logger

from nt_ops.runtime import REQUIRED_PROCESS_SCOPE, RuntimeState, install

logger = init_logger(__name__)


def apply_monkey_patches() -> RuntimeState:
    logger.warning(
        "nt_ops.monkey_patch.apply_monkey_patches() is deprecated. "
        "Use nt_ops.install(profile='qwen3_minimal_dense', process_scope=%r) instead.",
        REQUIRED_PROCESS_SCOPE,
    )
    return install(
        profile="qwen3_minimal_dense",
        process_scope=REQUIRED_PROCESS_SCOPE,
    )
