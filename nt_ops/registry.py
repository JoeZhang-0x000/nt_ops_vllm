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


def _rms_forward_oot_builder(original: object) -> object:
    from nt_ops import rms

    return rms.build_rms_forward_oot(original)


def _activation_silu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.silu_and_mul_forward


def _activation_fatrelu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.fatrelu_and_mul_forward


def _activation_mul_and_silu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.mul_and_silu_forward


def _activation_gelu_and_mul_builder(original: object) -> object:
    from nt_ops import activation

    return activation.gelu_and_mul_forward


def _activation_swigluoai_and_mul_builder(original: object) -> object:
    from nt_ops import activation

    return activation.swigluoai_and_mul_forward


def _activation_gelu_new_builder(original: object) -> object:
    from nt_ops import activation

    return activation.gelu_new_forward


def _activation_gelu_fast_builder(original: object) -> object:
    from nt_ops import activation

    return activation.gelu_fast_forward


def _activation_quick_gelu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.quick_gelu_forward


def _activation_relu2_builder(original: object) -> object:
    from nt_ops import activation

    return activation.relu2_forward


def _activation_xielu_builder(original: object) -> object:
    from nt_ops import activation

    return activation.xielu_forward


def _rope_forward_oot_builder(original: object) -> object:
    from nt_ops import rope

    return rope.build_rotary_forward_oot(original)


QWEN3_MINIMAL_DENSE_PROFILE = CapabilityProfile(
    name="qwen3_minimal_dense",
    enabled=(
        "rms_norm",
        "fused_add_rms_norm",
        "silu_and_mul",
    ),
    # rope: RotaryEmbedding.forward_oot is never called on MLU — the MLU
    # FlashAttentionBackend applies RoPE internally as a fused op.  Patching
    # forward_oot has no effect; leave rope in fallback until a viable
    # hook point is identified inside the MLU attention backend.
    # Other activation kernels exist in nt_ops/activation.py, but Qwen3-0.6B
    # advertises hidden_act="silu", so only SiluAndMul is part of the active
    # Qwen3 minimal-dense path for this profile.
    fallback=(
        "rope",
        "fatrelu_and_mul",
        "mul_and_silu",
        "gelu_and_mul",
        "swigluoai_and_mul",
        "gelu_new",
        "gelu_fast",
        "quick_gelu",
        "relu2",
        "xielu",
        "embedding",
        "logits_processor",
        "sampler",
    ),
    disabled=("linear", "attention", "lm_head", "flash_attn"),
)


_PROFILES: dict[str, tuple[CapabilityProfile, tuple[PatchSpec, ...]]] = {
    QWEN3_MINIMAL_DENSE_PROFILE.name: (
        QWEN3_MINIMAL_DENSE_PROFILE,
        (
            # Single patch on RMSNorm.forward_oot covers both rms_norm and
            # fused_add_rms_norm: dispatch_forward on out-of-tree platforms
            # binds _forward_method to self.forward_oot, which previously
            # resolved to forward_native (or an MLU override).  Patching
            # forward_oot intercepts both residual and non-residual paths.
            PatchSpec(
                patch_id="rms_norm",
                module_path="vllm.model_executor.layers.layernorm",
                object_name="RMSNorm",
                attr_name="forward_oot",
                builder=_rms_forward_oot_builder,
            ),
            # Patch forward_oot (not forward): CustomOp.forward dispatches
            # via self._forward_method which is bound to self.forward_oot on
            # out-of-tree platforms.  Replacing forward_oot at the class
            # level before model instantiation ensures _forward_method points
            # to our implementation.
            PatchSpec(
                patch_id="silu_and_mul",
                module_path="vllm.model_executor.layers.activation",
                object_name="SiluAndMul",
                attr_name="forward_oot",
                builder=_activation_silu_builder,
            ),
            # NOTE: rope patch omitted — RotaryEmbedding.forward_oot is never
            # called on MLU because FlashAttentionBackend fuses RoPE into the
            # attention kernel.  _rope_forward_oot_builder is kept for when a
            # viable hook point inside the MLU attention backend is found.
        ),
    )
}


def get_profile(profile_name: str) -> tuple[CapabilityProfile, tuple[PatchSpec, ...]]:
    try:
        return _PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown nt_ops profile: {profile_name}") from exc
