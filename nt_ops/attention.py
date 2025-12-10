import ninetoothed
import ninetoothed.language as ntl
import torch
from nt_ops.config import NT_MAX_NUM_CONFIG
BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


# TODO: varlen tile
def arrangement(q, k, v, out, seqused_k, block_table, softmax_scale):
    # q, o: shape=(batch, seq_len_q, num_head_kv, num_qeries_per_group, head_size)
    # k, v: shape=(num_blocks, block_size, num_head_kv, 1, head_size)
    # seqused_k: shape=(batch, )
    # block_table: shape=(batch num_blocks_per_batch)
    # -->
    # q, o: shape=(batch, num_head_kv, num_queries_per_group) (seq_len_q, head_size)
    # k, v: shape=(batch, num_head_kv, num_queries_per_group) (num_block_size) (block_size, head_size)
    # seqused_k: shape=(batch, -, -, -) (1, )
    # block_table: shape=(batch, -, -, -) (num_blocks_per_batch, )
    num_queries_per_group = q.shape[2] // k.shape[2]

    def arrange_q(x):
        x_arranged = (
            x.permute((0, 2, 3, 1, 4))  # b s n g h  # b n g s h
            .tile((1, 1, 1, -1, -1))  # b n g 1 1 | 1 1 1 s h
            .squeeze((3, 4))  # b n g | 1 1 1 s b
        )
        x_arranged.dtype = x_arranged.dtype.squeeze((0, 1, 2))  # b n g | s h
        return x_arranged

    def arrange_k(x):
        x_arranged = (x
            .permute((2, 3, 0, 1, 4)) # num_head_kv 1 num_blocks block_size head_size
            .tile((1, 1, -1, -1, -1)) # num_head_kv 1 1 1 1 | 1 1 ...
            .squeeze((2, 3, 4)) # num_head_kv 1 | 1 1 ...
            .unsqueeze(0) # 1 num_head_kv 1 | 1 1 ...
            .expand((q_arranged.shape[0], -1, q_arranged.shape[2])) # b num_head_kv g | 1 1 ...
        )
        x_arranged.dtype = x_arranged.dtype.squeeze((0, 1)) # b n g | num_blocks block_size head_size
        x_arranged.dtype = x_arranged.dtype.tile((1, -1, -1)).squeeze((1, 2)) # num_blocks | 1 block_size head_size
        x_arranged.dtype.dtype = x_arranged.dtype.dtype.squeeze((0, )) # num_blocks | block_size head_size
        return x_arranged

    def arrange_seq(x):
        x_arranged = (x # b 1 1 1 1
            .tile((1, 1, 1, 1, 1))  # b 1 1 1 1 | 1 1 1 1 1
            .squeeze((3, 4)) # b 1 1 | 1 1 1 1 1
            .expand((-1, q_arranged.shape[1], q_arranged.shape[2])) # b n g | 1 1 1 1 1
                      )
        x_arranged.dtype = x_arranged.dtype.squeeze((1, 2, 3, 4))
        return x_arranged

    def arrange_block(x):
        x_arranged = (
            x # b k 1 1 1
            .tile((1, -1, 1, 1, 1)) # b 1 1 1 1 | 1 k 1 1 1
            .squeeze((3, 4)) # b 1 1 | 1 k 1 1 1
            .expand((-1, q_arranged.shape[1], q_arranged.shape[2])) # b n g | 1 k 1 1 1
        )
        x_arranged.dtype = x_arranged.dtype.squeeze((0, 2, 3, 4)) # b n g | k
        return x_arranged


    q_arranged = arrange_q(q)
    out_arranged = arrange_q(out)
    k_arranged = arrange_k(k)
    v_arranged = arrange_k(v)
    # b, -> b | 1 -> b n g | 1
    seqused_k_arranged = arrange_seq(seqused_k)
    # b k -> b 1 | 1 k -> b 1 1 | 1 k -> b n g | k
    block_table_arranged = arrange_block(block_table)

    subs = {
        q: ninetoothed.Tensor(shape=(10, 5, 4, 2, 4)),
        k: ninetoothed.Tensor(shape=(40, 8, 4, 1, 4)),
        block_table: ninetoothed.Tensor(shape=(10, 3, 1, 1, 1)),
        seqused_k: ninetoothed.Tensor(shape=(10, 1, 1, 1, 1))
    }

    print(q_arranged.eval(subs).shape)
    print(k_arranged.eval(subs).shape)
    print(seqused_k.eval(subs).shape)
    print(block_table.eval(subs).shape)

    return (
        q_arranged,
        k_arranged,
        v_arranged,
        out_arranged,
        seqused_k_arranged,
        block_table_arranged,
        softmax_scale,
    )

def cdiv(x, y):
    return (x + y - 1) // y

def application(q, k, v, out, seqused_k, block_table, softmax_scale):
    # q/o: seq_len_q, head_size
    # k/v cache: num_blocks | block_size, head_size
    # seqused_k: 1
    # block_table: cdiv(seq_len_kv, block_size)
    q_i = ntl.cast(q, ntl.float32) * softmax_scale
    # o_i = ntl.zeros(q.shape, dtype=ntl.float32)
    # l_i = ntl.zeros(q.shape[0], dtype=ntl.float32)
    # m_i = ntl.full(q.shape[0], float("-inf"), dtype=ntl.float32)
    # len_seq_k = seqused_k
    # block_size = k.shape[1]
    # num_tiles = cdiv(len_seq_k, block_size)   
    # for blk_j in range(0, num_tiles):
    #     phy_id = block_table[blk_j]
    #     k_j = k[phy_id]
    #     v_j = v[phy_id]
    #     s_ij = ntl.dot(q_i, ntl.trans(k_j))
    #     # causal_mask = q.offsets(1) >= blk_j  * block_size + k.offsets(-1) % block_size
    #     # s_ij = ntl.where(causal_mask, s_ij, float("-inf"))
    #     m_ij = ntl.max(s_ij, axis=1)
    #     m_i_new = ntl.maximum(m_i, m_ij)
    #     p_ij = ntl.softmax(s_ij - m_i_new[:, None])
    #     l_ij = ntl.sum(p_ij, axis=1)
    #     exp_diff = ntl.exp(m_i - m_i_new)
    #     l_i_new = l_i * exp_diff + l_ij

    #     o_i = o_i * (l_i / l_i_new * exp_diff)[:, None] + ntl.dot(p_ij, v_j) / l_i_new[:, None]

    #     m_i = m_i_new
    #     l_i = l_i_new
    # out = ntl.cast(o_i, dtype=out.dtype)


def premake():
    Tensor = ninetoothed.Tensor
    shape_options = (None, None, None, None, {"constexpr": True, "upper_bound": 128})
    tensors = (
        # q
        # Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(5, shape_options=shape_options),
        # k
        Tensor(5, shape_options=shape_options),
        # v
        Tensor(5, shape_options=shape_options),
        # out
        # Tensor(5, jagged_dim=1, shape_options=shape_options),
        Tensor(5, shape_options=shape_options),
        # seqused_k
        Tensor(5, ),
        # block_table
        Tensor(5, ),
        # softmax_scale
        Tensor(0),
    )
    kernel = ninetoothed.make(
        arrangement, application, tensors, max_num_configs=NT_MAX_NUM_CONFIG
    )
    return kernel

kernel = premake()

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
    # kv: shape=(num_blocks, block_size, num_head_kv, head_size)
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_group = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    # jag tensor can not be view or reshape, thus, we make the layout before make jagged
    # qo: shape=(num_tokens,                num_head_kv, num_queries_per_group, head_size)
    # kv: shape=(num_blocks, block_size,    num_head_kv, 1,                     head_size)
    q = q.view(q.shape[0], num_kv_heads, num_queries_per_group, q.shape[-1])
    out = out.view(q.shape)
    k = k.unsqueeze(3)
    v = v.unsqueeze(3)

    # q_jag: shape=(batch, j, num_head_kv, num_queries_per_group, head_size)
    q_jag = torch.nested.nested_tensor_from_jagged(q, cu_seqlens_q, jagged_dim=1)
    out_jag = torch.nested.nested_tensor_from_jagged(out, cu_seqlens_q, jagged_dim=1)
    print(q_jag.shape)
    print(out_jag.shape)
    kernel(q_jag, k, v, out_jag, seqused_k, block_table, softmax_scale)
