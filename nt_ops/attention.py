import ninetoothed
import ninetoothed.language as ntl
import torch
from nt_ops.config import NT_MAX_NUM_CONFIG, NT_STATIC_MODE
from nt_ops.kv_cache import get_kv_from_cache
from vllm.logger import init_logger

logger = init_logger(__name__)

class _VarLen:
    if NT_STATIC_MODE:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    else:
        BLOCK_SIZE_M = ninetoothed.block_size()
        BLOCK_SIZE_N = ninetoothed.block_size()

    def _arrangement(
        q, k, v, o, sm_scale, is_causal, BLOCK_SIZE_M=None, BLOCK_SIZE_N=None
    ):
        """
        qo: b s hk num_per_group d
        kv: b s hk 1             d
        ->
        qo: b hk num_per_group s_bm         1         | bm d
        kv: b hk num_queries_per_group s_bm 1 | s_bn  | bn d
        """
        # BLOCK_SIZE_M = ninetoothed.Symbol("BLOCK_SIZE_M", constexpr=True)
        # BLOCK_SIZE_N = ninetoothed.Symbol("BLOCK_SIZE_N", constexpr=True)
        if BLOCK_SIZE_M is None:
            BLOCK_SIZE_M = _VarLen.BLOCK_SIZE_M
        if BLOCK_SIZE_N is None:
            BLOCK_SIZE_N = _VarLen.BLOCK_SIZE_N

        def _arrange_qo(x):
            x_arranged = (
                x.permute((0, 2, 3, 1, 4)).tile(  # b s h g d  # b h g s d
                    (1, 1, 1, BLOCK_SIZE_M, -1)
                )  # b h g s_bm 1 | 1 1 1 bm d
            )
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2))
            return x_arranged

        def _arrange_kv(x):
            x_arranged = (
                x.permute((0, 2, 3, 1, 4))  # b s h 1 d  # b h 1 s d
                .tile((1, 1, 1, BLOCK_SIZE_N, -1))  # b h 1 s_bn 1 | 1 1 1 bn d
                .expand((-1, -1, q_arranged.shape[2], q_arranged.shape[3], -1))
                .tile((1, 1, 1, -1, -1))  # b h g 1 1 | 1 1 1 s_bn 1 | 1 1 1 bn d
            )
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2, 4))
            x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze((0, 1, 2))
            return x_arranged

        q_arranged = _arrange_qo(q)
        o_arranged = _arrange_qo(o)
        k_arranged = _arrange_kv(k)
        v_arranged = _arrange_kv(v)

        return q_arranged, k_arranged, v_arranged, o_arranged, sm_scale, is_causal

    def _application(q, k, v, o, sm_scale, is_causal):
        """
        qo: bm d
        kv: s_bn | bn d
        """
        q_i = ntl.cast(q, dtype=ntl.float32) * sm_scale
        m_i = ntl.full((q.shape[0],), float("-inf"), dtype=ntl.float32)
        l_i = ntl.zeros((q.shape[0],), dtype=ntl.float32)
        o_i = ntl.zeros(o.shape, dtype=ntl.float32)

        for j in range(k.shape[0]):
            k_j = ntl.cast(k[j], dtype=ntl.float32)
            v_j = ntl.cast(v[j], dtype=ntl.float32)
            s_ij = ntl.dot(q_i, ntl.trans(k_j))
            s_ij = ntl.where(
                k[j].offsets(1)[None, :] < k.source.shape[1], s_ij, float("-inf")
            )
            if is_causal:
                causal_mask = q.offsets(1)[:, None] >= k[j].offsets(1)[None, :]
                s_ij = ntl.where(causal_mask, s_ij, float("-inf"))
            m_ij = ntl.max(s_ij, axis=1)
            m_i_new = ntl.maximum(m_ij, m_i)
            p_ij = ntl.exp(s_ij - m_i_new[:, None])
            l_ij = ntl.sum(p_ij, axis=1)

            exp_diff = ntl.exp(m_i - m_i_new)
            l_i_new = l_i * exp_diff + l_ij

            o_i = (
                o_i * (l_i / l_i_new * exp_diff)[:, None]
                + ntl.dot(p_ij, v_j) / l_i_new[:, None]
            )

            m_i = m_i_new
            l_i = l_i_new
        o = ntl.cast(o_i, dtype=o.dtype)

    def _premake():
        shape_options = (
            None,
            None,
            None,
            None,
            {"constexpr": True, "upper_bound": 128},
        )
        tensors = (
            ninetoothed.Tensor(5, jagged_dim=1, shape_options=shape_options),
            ninetoothed.Tensor(5, jagged_dim=1, shape_options=shape_options),
            ninetoothed.Tensor(5, jagged_dim=1, shape_options=shape_options),
            ninetoothed.Tensor(5, jagged_dim=1, shape_options=shape_options),
            ninetoothed.Tensor(0),
            ninetoothed.Tensor(0),
        )
        kernel = ninetoothed.make(
            _VarLen._arrangement,
            _VarLen._application,
            tensors,
            max_num_configs=NT_MAX_NUM_CONFIG,
        )
        return kernel


kernel = {}
kernel["varlen"] = _VarLen._premake()


def apply_varlen(q, k, v, sm_scale=None, is_causal=True):
    o = torch.empty_like(q)
    hq = q.shape[2]
    hk = k.shape[2]
    assert hq % hk == 0, (
        "Number of heads in `query` must be divisible by number of heads in `key` and `value` when GQA is enabled."
    )
    num_queries_per_group = hq // hk
    q = q.view(
        q.shape[0],
        q.shape[1],
        q.shape[2] // num_queries_per_group,
        num_queries_per_group,
        q.shape[3],
    )
    o = o.view(q.shape)
    k = k.view(k.shape[0], k.shape[1], k.shape[2], 1, k.shape[3])
    v = v.view(k.shape)
    sm_scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    kernel["varlen"](q, k, v, o, sm_scale, is_causal)
    out_shape = o._values.shape
    return o._values.view(out_shape[0], -1, out_shape[-1])


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    softmax_scale=None,
    causal=True,
    **kwargs,
):
    logger.info_once("\033[32mNT FA Varlen is enabled.\033[0m")  
    q_jag = torch.nested.nested_tensor_from_jagged(
        q.clone(), cu_seqlens_q, jagged_dim=1
    )
    # Use the correct cu_seqlens for k and v
    k_jag = torch.nested.nested_tensor_from_jagged(
        k.clone(), cu_seqlens_k, jagged_dim=1
    )
    v_jag = torch.nested.nested_tensor_from_jagged(
        v.clone(), cu_seqlens_k, jagged_dim=1
    )
    return apply_varlen(q_jag, k_jag, v_jag, softmax_scale, causal)


class KvCache:
    def _arrangement(
        q, k_cache, v_cache, o, cache_seqlens, block_table, sm_scale, is_causal
    ):
        """
        qo: b 1 h g d
        kv_cache: block_num block_size h 1 d
        cache_seqlens: b 1 1 1 1
        block_table: b block_num 1 1 1
        """

        def _arrange_qo(x):
            # b 1 h g d
            x_arranged = x.permute((0, 2, 3, 1, 4))  # b h g 1 d
            x_arranged = x_arranged.tile((1, 1, 1, -1, -1))  # b h g 1 1| 1 1 1 1 d
            x_arranged = x_arranged.squeeze((3, 4))  # b h g | 1 1 1 1 d
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2))  # b h g | 1 d
            return x_arranged

        q_arranged = _arrange_qo(q)
        o_arranged = _arrange_qo(o)

        def _arrange_cache(x):
            x_arranged = x.permute((2, 3, 0, 1, 4))  # h 1 block_num block_size d
            x_arranged = x_arranged.tile((1, 1, 1, -1, -1))  # h 1 block_num 1 1 | 1 1 ...
            x_arranged = x_arranged.squeeze((3, 4)) # h 1 block_num | 1 1 1 ..
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2)) # h 1 block_num | block_size head_size
            x_arranged = x_arranged.tile((1, 1, -1)) # h 1 1 | 1 1 block_num | ...
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1)) # h 1 1 | block_num | ...
            x_arranged = x_arranged.permute((2, 0, 1)) # 1 h 1 | block_num | ...
            x_arranged = x_arranged.expand((q_arranged.shape[0], -1, q_arranged.shape[2])) # b h g
            return x_arranged

        k_cache_arranged = _arrange_cache(k_cache)
        v_cache_arranged = _arrange_cache(v_cache)

        def _arrange_seq(x):
            # b 1x4
            x_arranged = x.tile((1,) * 5)  # b 1x5 | 1x5
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2, 3))  # b 1x5 | 1
            x_arranged = x_arranged.squeeze((3, 4))  # b 1 1 | 1
            x_arranged = x_arranged.expand(
                (-1, q_arranged.shape[1], q_arranged.shape[2])
            )  # b h g | 1
            return x_arranged

        def _arrange_block(x):
            # b j 1x3
            x_arranged = x.tile((1, -1) + (1,) * 3)  # b 1x4 | 1 j 1x3
            x_arranged.dtype = x_arranged.dtype.squeeze((0, 2, 3, 4))  # b 1x4 | j
            x_arranged = x_arranged.squeeze((3, 4))  # b 1 1 | j
            x_arranged = x_arranged.expand(
                (-1, q_arranged.shape[1], q_arranged.shape[2])
            )  # b h g | j
            return x_arranged

        cache_seqlens_arranged = _arrange_seq(cache_seqlens)
        block_table_arranged = _arrange_block(block_table)

        return (
            q_arranged,
            k_cache_arranged,
            v_cache_arranged,
            o_arranged,
            cache_seqlens_arranged,
            block_table_arranged,
            sm_scale,
            is_causal,
        )

    def _application(
        q, k_cache, v_cache, o, cache_seqlens, block_table, sm_scale, is_causal
    ):
        # q/o: 1 d
        # k/v cache: num_blocks | block_size, head_size

        q_i = ntl.cast(q, dtype=ntl.float32) * sm_scale
        m_i = ntl.full((1,), float("-inf"), dtype=ntl.float32)
        l_i = ntl.full((1,), float(0), dtype=ntl.float32)
        o_i = ntl.zeros(q.shape, dtype=ntl.float32)

        block_nums = block_table.shape[0]
        block_size = k_cache[0].shape[0]
        seq_start = 0
        for blk in range(block_nums):
            blk_id = block_table[blk]
            k_j = ntl.cast(k_cache[blk_id], dtype=ntl.float32)
            v_j = ntl.cast(v_cache[blk_id], dtype=ntl.float32)

            k_j_t = ntl.trans(k_j)
            s_ij = ntl.dot(q_i, k_j_t)

            mask = (k_cache[blk].offsets(1) % block_size + seq_start) < cache_seqlens[0]
            s_ij = ntl.where(mask[None, :], s_ij, float("-inf"))

            if is_causal:
                pass

            m_ij = ntl.max(s_ij, axis=1)
            m_i_new = ntl.maximum(m_ij, m_i)
            p_ij = ntl.exp(s_ij - m_i_new[:, None])
            l_ij = ntl.sum(p_ij, axis=1)

            exp_diff = ntl.exp(m_i - m_i_new)
            l_i_new = l_i * exp_diff + l_ij

            o_i = (
                o_i * (l_i / l_i_new * exp_diff)[:, None]
                + ntl.dot(p_ij, v_j) / l_i_new[:, None]
            )
            m_i = m_i_new
            l_i = l_i_new
            seq_start = seq_start + block_size
        o = ntl.cast(o_i, o.dtype)

    def _premake():
        tensors = (
            # q
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # k cache
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # v cache
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # o
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # cache_seqlens
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # block_table
            ninetoothed.Tensor(5, shape_options={"constexpr": True}),
            # softmax_scale
            ninetoothed.Tensor(0),
            # is_causal
            ninetoothed.Tensor(0),
        )
        kernel = ninetoothed.make(
            KvCache._arrangement, KvCache._application, tensors, num_stages=4
        )
        return kernel


kernel["cache"] = KvCache._premake()


def apply_cache(
    q, k_cache, v_cache, cache_seqlens, block_table, sm_scale=None, is_causal=True
):
    sm_scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    o = torch.empty_like(q)
    kernel["cache"](
        q.view(
            q.shape[0], 1, k_cache.shape[-2], -1, q.shape[-1]
        ),  # t h d -> b 1 h d -> b 1 num_head_kv num_head_per_group head_size
        k_cache.unsqueeze(3),  # num_blocks, block_szie, num_head_kv, 1, head_size
        v_cache.unsqueeze(3),
        o.view(o.shape[0], 1, k_cache.shape[-2], -1, o.shape[-1]),
        cache_seqlens.view(cache_seqlens.shape + (1, 1, 1, 1)),  # b 1 1 1 1
        block_table.view(block_table.shape + (1, 1, 1)),  # b j 1 1 1
        sm_scale,
        is_causal,
    )
    return o


def flash_attn_with_kvcache(
    q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal
):
    logger.info_once("\033[32mNT FA with KVCache is enabled.\033[0m")  
    return apply_cache(
        q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal
    )


def unified_attention_2d(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
):
    assert causal
    # qo: shape=(num_tokens, num_head_q, head_size)
    # kv_cache: shape=(num_blocks, block_size, num_head_kv, head_size)

    is_prefill = q.shape[0] > (cu_seqlens_q.shape[0] - 1)

    if is_prefill:
        k, v, cu_seqlens_k = get_kv_from_cache(k, v, seqused_k, block_table)

        o = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, causal
        )
    else:
        o = flash_attn_with_kvcache(
            q, k, v, seqused_k, block_table, softmax_scale, causal
        )
    out.copy_(o)
