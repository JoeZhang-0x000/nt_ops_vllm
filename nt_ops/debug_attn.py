import torch
import triton

def trion_attn(
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
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
):
    from vllm.attention.ops.triton_unified_attention import (
        kernel_unified_attention_3d,
        reduce_segments,
    )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Assigning default tile sizes for prefill and decode.
    # Note: each tile size must be at least 32 for "fp8" (q.element_size() == 1)
    # and at least 16 for all other data types.
    TILE_SIZE_PREFILL = 32
    TILE_SIZE_DECODE = 16 if q.element_size() >= 2 else 32
    # for initial version, NUM_SEGMENTS = 16 is chosen as a default
    # value that showed good performance in tests
    NUM_SEGMENTS = 16

    segm_output = torch.empty(
        q.shape[0],
        num_query_heads,
        NUM_SEGMENTS,
        triton.next_power_of_2(head_size),
        dtype=torch.float32,
        device=q.device,
    )
    segm_max = torch.empty(
        q.shape[0],
        num_query_heads,
        NUM_SEGMENTS,
        dtype=torch.float32,
        device=q.device,
    )
    segm_expsum = torch.empty(
        q.shape[0],
        num_query_heads,
        NUM_SEGMENTS,
        dtype=torch.float32,
        device=q.device,
    )

    kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE_DECODE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
    )
    reduce_segments[(q.shape[0], num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=TILE_SIZE_DECODE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        USE_FP8=output_scale is not None,
    )
