from __future__ import annotations

from vllm.logger import init_logger

from nt_ops.capabilities import record_hit
from nt_ops.debug import trace

logger = init_logger(__name__)


def _run_silu_and_mul(x):
    from nt_ops.activation import siluAndMul

    return siluAndMul(x)


def build_mlu_active_wrapper(original):
    def active(input, act_mode: str, is_gated: bool):
        if act_mode != "silu" or not is_gated:
            return original(input, act_mode, is_gated)

        if input.ndim not in (2, 3) or input.shape[-1] % 2 != 0:
            return original(input, act_mode, is_gated)

        logger.info_once("\033[32mNT SILU AND MUL is enabled via vllm_mlu.active.\033[0m")
        trace(
            "silu_and_mul",
            dispatched_via="vllm_mlu.active",
            shape=tuple(input.shape),
            dtype=input.dtype,
        )
        record_hit("silu_and_mul")
        return _run_silu_and_mul(input)

    return active
