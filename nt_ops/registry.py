from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nt_ops.capabilities import CapabilityProfile


PatchBuilder = Callable[[object], object]


@dataclass(frozen=True)
class PatchSpec:
    patch_id: str
    module_path: str
    attr_name: str
    object_name: str | None = None
    required: bool = True
    builder: PatchBuilder | None = None


def _activation_silu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.silu_and_mul_forward


def _rms_builder(original: object) -> object:
    from nt_ops import rms

    return rms.rms_norm_helper


def _fused_rms_builder(original: object) -> object:
    from nt_ops import rms

    return rms.fused_add_rms_norm_helper


def _rope_builder(original: object) -> object:
    from nt_ops import rope

    return rope.build_rotary_forward_cuda(original)


QWEN3_MINIMAL_DENSE_PROFILE = CapabilityProfile(
    name="qwen3_minimal_dense",
    enabled=("rms_norm", "fused_add_rms_norm", "silu_and_mul", "rope"),
    fallback=("embedding", "logits_processor", "sampler"),
    disabled=("linear", "attention", "lm_head", "flash_attn"),
)


_PROFILES: dict[str, tuple[CapabilityProfile, tuple[PatchSpec, ...]]] = {
    QWEN3_MINIMAL_DENSE_PROFILE.name: (
        QWEN3_MINIMAL_DENSE_PROFILE,
        (
            PatchSpec(
                patch_id="rms_norm",
                module_path="vllm.model_executor.layers.layernorm",
                attr_name="rms_norm",
                builder=_rms_builder,
            ),
            PatchSpec(
                patch_id="fused_add_rms_norm",
                module_path="vllm.model_executor.layers.layernorm",
                attr_name="fused_add_rms_norm",
                builder=_fused_rms_builder,
            ),
            PatchSpec(
                patch_id="silu_and_mul",
                module_path="vllm.model_executor.layers.activation",
                object_name="SiluAndMul",
                attr_name="forward",
                builder=_activation_silu_builder,
            ),
            PatchSpec(
                patch_id="rope",
                module_path="vllm.model_executor.layers.rotary_embedding.base",
                object_name="RotaryEmbedding",
                attr_name="forward_cuda",
                builder=_rope_builder,
            ),
        ),
    )
}


def get_profile(profile_name: str) -> tuple[CapabilityProfile, tuple[PatchSpec, ...]]:
    try:
        return _PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown nt_ops profile: {profile_name}") from exc
