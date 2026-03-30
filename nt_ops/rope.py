from __future__ import annotations

import torch

import ntops.torch
from vllm.logger import init_logger

from nt_ops.capabilities import record_hit

logger = init_logger(__name__)


def _apply_rotary(
    x: torch.Tensor,
    *,
    num_tokens: int,
    head_size: int,
    rotary_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    x_view = x.view(num_tokens, -1, head_size)
    x_rot = x_view[..., :rotary_dim].unsqueeze(0)
    rotated = ntops.torch.rotary_position_embedding(
        x_rot,
        sin,
        cos,
        interleaved=not is_neox_style,
        inplace=False,
    ).squeeze(0)

    if rotary_dim == head_size:
        return rotated.reshape_as(x)

    x_pass = x_view[..., rotary_dim:]
    return torch.cat((rotated, x_pass), dim=-1).reshape_as(x)


def build_rotary_forward_oot(original_forward_oot):
    """Replace RotaryEmbedding.forward_oot to intercept the MLU dispatch path.

    ``dispatch_forward`` on out-of-tree platforms binds ``_forward_method``
    to ``self.forward_oot``.  The ``is_cuda`` guard from the old
    ``forward_cuda`` wrapper is intentionally removed: on MLU tensors are
    not CUDA tensors, so that check would always fall back.

    Falls back to ``original_forward_oot`` (MLU kernel or PyTorch-native)
    for unsupported configurations (non-NTK style, odd rotary_dim).
    """

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.is_neox_style or self.rotary_dim % 2:
            return original_forward_oot(self, positions, query, key)

        logger.info_once("\033[32mNT RoPE is enabled.\033[0m")
        record_hit("rope")

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_out = _apply_rotary(
            query,
            num_tokens=num_tokens,
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            cos=cos,
            sin=sin,
            is_neox_style=self.is_neox_style,
        )
        query.copy_(query_out)

        if key is not None:
            key_out = _apply_rotary(
                key,
                num_tokens=num_tokens,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                cos=cos,
                sin=sin,
                is_neox_style=self.is_neox_style,
            )
            key.copy_(key_out)

        return query, key

    return forward_oot
