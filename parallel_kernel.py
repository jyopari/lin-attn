# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

# from fla.ops.utils import prepare_chunk_indices
# from fla.ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum
# from fla.ops.utils.op import safe_exp
# from fla.utils import (
#     autocast_custom_bwd,
#     autocast_custom_fwd,
#     check_shared_mem,
#     input_guard,
#     is_intel_alchemist,
#     is_nvidia_hopper
# )

# # https://github.com/intel/intel-xpu-backend-for-triton/issues/3449
# triton_config = {'grf_mode': 'large'} if is_intel_alchemist else {}
# NUM_WARPS = [2, 4, 8] if is_nvidia_hopper else [2, 4, 8, 16]


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [1]
    ],
    key=["BT", "BS", "BK", "BV"],
)
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    o,
    attn,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    o += i_k * B * T * H * V

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    if OUTPUT_ATTENTIONS:
        attn += (bos * H + i_h * T) * T + i_k * B * H * T * T
    stride_qk = H * K
    stride_vo = H * V

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        o_k += BS

    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    output_attentions: bool = False,
    chunk_size: int = 32,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    # if check_shared_mem('hopper', k.device.index):
    #     BK = min(256, triton.next_power_of_2(K))
    #     BV = min(256, triton.next_power_of_2(V))
    # elif check_shared_mem('ampere', k.device.index):
        #print('ampere')
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))
    # else:
    #     print('other')
    #     BK = min(64, triton.next_power_of_2(K))
    #     BV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    # chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    chunk_indices = None
    NT = triton.cdiv(T, BT)

    grid = (NK * NV, NT, B * H)
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None

    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        attn=attn,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    o = o.sum(0)

    if output_attentions:
        attn = attn.sum(0)
    return o, attn

class ParallelSimpleGLAFunction(torch.autograd.Function):
    @staticmethod
    #@input_guard
    #@autocast_custom_fwd
    def forward(ctx, q, k, v, scale, output_attentions, cu_seqlens):
        chunk_size = 32
        ctx.dtype = q.dtype

        o, attn = parallel_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            output_attentions=output_attentions,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, cu_seqlens)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), attn

def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    output_attentions: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
    if output_attentions:
        assert cu_seqlens is None, "output_attentions=True is not supported with variable-length sequences"

    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, attn = ParallelSimpleGLAFunction.apply(
        q,
        k,
        v,
        scale,
        output_attentions,
        cu_seqlens
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, attn


def fused_parallel_dataflow(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    count: bool,
    BT: int = 32,
    BS: int = 32,
    BK: int = 128,
    BV: int = 128,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, v.shape[-1]
    # if check_shared_mem('hopper', k.device.index):
    #     BK = min(256, triton.next_power_of_2(K))
    #     BV = min(256, triton.next_power_of_2(V))
    # elif check_shared_mem('ampere', k.device.index):
    # else:
    #     BK = min(64, triton.next_power_of_2(K))
    #     BV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = None
    NT = triton.cdiv(T, BT)

    o = torch.zeros((NK, B, T, H, V), device=q.device, dtype=q.dtype)
    statistics = {
        "DRAM->SRAM": 0,
        "SRAM->DRAM": 0,
        "MAC": 0,
    }
    for i_kv in range(NK * NV): # parallel for
        for i_t in range(NT): # parallel for
            for i_bh in range(B * H): # parallel for
                i_k, i_v = i_kv // NV, i_kv % NV
                i_b, i_h = i_bh // H, i_bh % H

                # on chip buffer
                b_o = torch.zeros((BT, BV), device=q.device, dtype=q.dtype)
                b_q = torch.empty((BT, BK), device=q.device, dtype=q.dtype)

                # on chip buffer
                m_s = torch.empty((BT, BS), device=q.device, dtype=torch.bool)
                b_s = torch.zeros((BT, BS), device=q.device, dtype=q.dtype)
                b_k = torch.empty((BK, BS), device=q.device, dtype=q.dtype)
                b_v = torch.empty((BS, BV), device=q.device, dtype=q.dtype)

                # DRAM -> SRAM copy
                for i in range(BT):
                    for j in range(BK):
                        if count:
                            statistics["DRAM->SRAM"] += q.element_size()
                        else:
                            # print(q.shape, i_b, i_t * BT + i, i_h, i_k * BK + j, NT, BT)
                            b_q[i, j] = q[i_b, i_t * BT + i, i_h, i_k * BK + j]

                for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):

                    # DRAM -> SRAM copy
                    for i in range(BK):
                        for j in range(BS):
                            if count:
                                statistics["DRAM->SRAM"] += k.element_size()
                            else:
                                b_k[i, j] = k[i_b, i_s + j, i_h, i_k * BK + i]

                    # DRAM -> SRAM copy
                    for i in range(BS):
                        for j in range(BV):
                            if count:
                                statistics["DRAM->SRAM"] += v.element_size()
                            else:
                                b_v[i, j] = v[i_b, i_s + i, i_h, i_v * BV + j]

                    # on chip compute 
                    for i in range(BT):
                        for j in range(BS):
                            m_s[i, j] = (i_t * BT + i) >= (i_s + j)

                    # on chip compute 

                    for i in range(BT):
                        for j in range(BS):
                            b_s[i,j] = 0
                            for k_i in range(BK):
                                if count:
                                    statistics["MAC"] += 1
                                else:
                                    b_s[i,j] += b_q[i,k_i] * b_k[k_i,j] 

                    # on chip compute 
                    for i in range(BT):
                        for j in range(BS):
                            if count:
                                statistics["MAC"] += 1
                            else:
                                b_s[i,j] *= m_s[i,j]

                    # on chip compute 
                    for i in range(BT):
                        for j in range(BV):
                            for s in range(BS):
                                if count:
                                    statistics["MAC"] += 1
                                else:
                                    b_o[i,j] += b_s[i,s] * b_v[s,j]

                for i_s in range(i_t * BT - BS, -BS, -BS):

                    # DRAM -> SRAM copy
                    for i in range(BK):
                        for j in range(BS):
                            if count:
                                statistics["DRAM->SRAM"] += k.element_size()
                            else:
                                b_k[i, j] = k[i_b, i_s + j, i_h, i_k * BK + i]

                    # DRAM -> SRAM copy
                    for i in range(BS):
                        for j in range(BV):
                            if count:
                                statistics["DRAM->SRAM"] += v.element_size()
                            else:
                                b_v[i, j] = v[i_b, i_s + i, i_h, i_v * BV + j]

                    # on chip compute 
                    for i in range(BT):
                        for j in range(BS):
                            b_s[i,j] = 0
                            for k_i in range(BK):
                                if count:
                                    statistics["MAC"] += 1
                                else:
                                    b_s[i,j] += b_q[i,k_i] * b_k[k_i,j]

                    # on chip compute 
                    if i_s >= 0:
                        for i in range(BT):
                            for j in range(BV):
                                for s in range(BS):
                                    if count:
                                        statistics["MAC"] += 1
                                    else:
                                        b_o[i,j] += b_s[i,s] * b_v[s,j]

                # SRAM -> DRAM copy
                for i in range(BT):
                    for j in range(BV):
                        if count:
                            statistics["SRAM->DRAM"] += o.element_size()
                        else:
                            o[i_k, i_b, i_t * BT + i, i_h, i_v * BV + j] += b_o[i,j]

    if count:
        statistics["SRAM-size"] = (
            b_q.numel() * b_q.element_size() +
            b_k.numel() * b_k.element_size() +
            b_v.numel() * b_v.element_size() +
            b_o.numel() * b_o.element_size() +
            b_s.numel() * b_s.element_size() +
            m_s.numel() * m_s.element_size()
        )
        statistics["arithmetic-intensity"] = (
            (statistics["MAC"] * 2) /
            (statistics["DRAM->SRAM"] + statistics["SRAM->DRAM"])
        )
        return statistics
    else:
        o_final = torch.zeros((B, T, H, V), device=q.device, dtype=q.dtype)
        for i_nk in range(NK):
            for b in range(B):
                for t in range(T):
                    for h in range(H):
                        for v in range(V):
                            o_final[b, t, h, v] += o[i_nk, b, t, h, v]
        return o_final



if __name__ == "__main__":
    import torch
    
    def linear_attention(
        Q: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
        K: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
        V: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
    ) -> torch.Tensor:
        # Compute QK^T: result has shape [batch, length, num_heads, length]
        scores = torch.einsum("b i h d, b j h d -> b i h j", Q, K)
        
        # Create a causal mask: lower-triangular matrix of shape [length, length]
        L = scores.shape[-1]  # e.g., 4 in our test case
        mask = torch.tril(torch.ones(L, L, device=scores.device, dtype=scores.dtype))
        
        # Reshape mask to [1, length, 1, length] so it broadcasts over batch and heads
        mask = mask.unsqueeze(0).unsqueeze(2)  # Now shape is [1, L, 1, L]
        
        # Apply the mask
        masked_scores = scores * mask
        
        # Compute the output by combining masked scores with V
        output = torch.einsum("b i h j, b j h d -> b i h d", masked_scores, V)
        return output

    # Test parameters
    batch_size = 1
    seq_len = 16
    num_heads = 8
    head_dim = 16
    value_dim = 16
    
    # Create same random input tensors for both implementations
    torch.manual_seed(42)  # For reproducibility
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, num_heads, value_dim, device='cuda')
    
    # Scale factor
    scale = 1.0 / (head_dim ** 0.5)
    
    # Apply scaling to match the parallel implementation
    q_scaled = q * scale
    
    # Run both implementations
    output_parallel, _ = parallel_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        output_attentions=False,
        cu_seqlens=None,
        head_first=False
    )
    
    output_reference = linear_attention(q_scaled, k, v)

    output_dataflow = fused_parallel_dataflow(q_scaled, k, v, count=False, BT=16, BS=16, BK=16, BV=16)
    
    # Compare outputs
    max_diff_parallel = torch.max(torch.abs(output_parallel - output_reference))
    mean_diff_parallel = torch.mean(torch.abs(output_parallel - output_reference))
    is_close_parallel = torch.allclose(output_parallel, output_reference, rtol=1e-5, atol=1e-5)
    
    max_diff_dataflow = torch.max(torch.abs(output_dataflow - output_reference))
    mean_diff_dataflow = torch.mean(torch.abs(output_dataflow - output_reference))
    is_close_dataflow = torch.allclose(output_dataflow, output_reference, rtol=1e-5, atol=1e-5)
    
    print("\nParallel implementation comparison:")
    print(f"Maximum absolute difference: {max_diff_parallel:.6e}")
    print(f"Mean absolute difference: {mean_diff_parallel:.6e}")
    print(f"Outputs match within tolerance: {is_close_parallel}")
    
    print("\nDataflow implementation comparison:")
    print(f"Maximum absolute difference: {max_diff_dataflow:.6e}")
    print(f"Mean absolute difference: {mean_diff_dataflow:.6e}")
    print(f"Outputs match within tolerance: {is_close_dataflow}")
    
    if not is_close_parallel or not is_close_dataflow:
        print("\nDetailed error analysis:")
        if not is_close_parallel:
            print("\nParallel implementation differences:")
            diff_parallel = torch.abs(output_parallel - output_reference)
            max_diff_idx_parallel = torch.where(diff_parallel == max_diff_parallel)
            print(f"Location of maximum difference: {tuple(idx.item() for idx in max_diff_idx_parallel)}")
            print(f"Parallel value: {output_parallel[max_diff_idx_parallel]}")
            print(f"Reference value: {output_reference[max_diff_idx_parallel]}")
            
        if not is_close_dataflow:
            print("\nDataflow implementation differences:")
            diff_dataflow = torch.abs(output_dataflow - output_reference)
            max_diff_idx_dataflow = torch.where(diff_dataflow == max_diff_dataflow)
            print(f"Location of maximum difference: {tuple(idx.item() for idx in max_diff_idx_dataflow)}")
            print(f"Dataflow value: {output_dataflow[max_diff_idx_dataflow]}")
            print(f"Reference value: {output_reference[max_diff_idx_dataflow]}")
