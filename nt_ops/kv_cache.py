import torch
import triton
import triton.language as tl
import typing


# grid: (num_seqs, max_num_blocks_per_seq)
@triton.jit
def get_kv_from_cache_kernel(
    K, V, K_CACHE, V_CACHE,
    seq_lens,          # shape: (num_seqs,)
    cu_seqlens,        # shape: (num_seqs + 1,)
    block_table,       # shape: (num_seqs, max_num_blocks_per_seq)
    
    stride_k_t, stride_k_h,
    stride_v_t, stride_v_h,
    
    stride_k_cache_n, stride_k_cache_b, stride_k_cache_h,
    stride_v_cache_n, stride_v_cache_b, stride_v_cache_h,
    
    stride_block_table_s,
    
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    # Each program handles one logical block from one sequence.
    seq_idx = tl.program_id(0)
    logical_block_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens + seq_idx)

    # Stop processing if this logical block is beyond the sequence length
    if logical_block_idx * BLOCK_SIZE >= seq_len:
        return

    # --- Load Phase ---
    # 1. Get the physical block ID from the block table
    physical_block_id = tl.load(block_table + seq_idx * stride_block_table_s + logical_block_idx)

    # 2. Define offsets for the tokens within the block and for the head dimension
    offs_token_in_block = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_h = tl.arange(0, NUM_HEADS)

    # 3. Calculate pointers to the source K/V cache
    k_cache_ptr = K_CACHE + physical_block_id * stride_k_cache_n
    k_cache_ptr += offs_token_in_block[:, None, None] * stride_k_cache_b
    k_cache_ptr += offs_h[None, :, None] * stride_k_cache_h
    k_cache_ptr += offs_d[None, None, :]
    
    v_cache_ptr = V_CACHE + physical_block_id * stride_v_cache_n
    v_cache_ptr += offs_token_in_block[:, None, None] * stride_v_cache_b
    v_cache_ptr += offs_h[None, :, None] * stride_v_cache_h
    v_cache_ptr += offs_d[None, None, :]

    # 4. Create a mask to avoid loading out-of-bound tokens
    token_idx_in_seq = logical_block_idx * BLOCK_SIZE + offs_token_in_block
    mask = token_idx_in_seq[:, None, None] < seq_len

    # 5. Load K and V values
    k_values = tl.load(k_cache_ptr, mask=mask, other=0.0)
    v_values = tl.load(v_cache_ptr, mask=mask, other=0.0)

    # --- Store Phase ---
    # 1. Get the starting global token index for the current sequence
    cu_seq_start = tl.load(cu_seqlens + seq_idx)
    global_token_idx = cu_seq_start + token_idx_in_seq

    # 2. Calculate pointers to the destination K/V tensors
    k_ptr = K + global_token_idx[:, None, None] * stride_k_t
    k_ptr += offs_h[None, :, None] * stride_k_h
    k_ptr += offs_d[None, None, :]

    v_ptr = V + global_token_idx[:, None, None] * stride_v_t
    v_ptr += offs_h[None, :, None] * stride_v_h
    v_ptr += offs_d[None, None, :]

    # 3. Store the loaded values
    tl.store(k_ptr, k_values, mask=mask)
    tl.store(v_ptr, v_values, mask=mask)


def get_kv_from_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor, # This is the individual sequence lengths
    block_table: torch.Tensor,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # k/v cache: shape=(num_blocks, block_size, num_head, head_size)
    # k/v: shape=(total_tokens, num_head, head_size)
    # seq_lens: shape=(num_seqs,)
    # block_table: shape=(num_seqs, max_num_blocks_per_seq)
    
    num_seqs = block_table.shape[0]
    
    # Correctly compute cumulative sequence lengths
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0))
    total_tokens = int(cu_seqlens[-1])
    
    if total_tokens == 0:
        return (
            torch.empty(0, k_cache.shape[2], k_cache.shape[3], dtype=k_cache.dtype, device=k_cache.device),
            torch.empty(0, v_cache.shape[2], v_cache.shape[3], dtype=v_cache.dtype, device=v_cache.device),
        )

    head_size = k_cache.shape[-1]
    num_heads = k_cache.shape[-2]
    block_size = k_cache.shape[1]
    max_num_blocks_per_seq = block_table.shape[1]

    # Prepare output tensors with the correct size
    k = torch.empty(
        (total_tokens, num_heads, head_size), dtype=k_cache.dtype, device=k_cache.device
    )
    v = torch.empty_like(k)

    # Launch grid
    grid = (num_seqs, max_num_blocks_per_seq)
    
    get_kv_from_cache_kernel[grid](
        k, v, k_cache, v_cache,
        seq_lens,
        cu_seqlens,
        block_table,
        
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        
        block_table.stride(0),
        
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        NUM_HEADS=num_heads,
    )
    return k, v, cu_seqlens