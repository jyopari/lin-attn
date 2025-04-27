# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla_ops.utils import prepare_chunk_indices
from fla_ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum
from fla_ops.utils.op import safe_exp
from fla_utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    check_shared_mem,
    input_guard,
    is_intel_alchemist,
    is_nvidia_hopper
)

# https://github.com/intel/intel-xpu-backend-for-triton/issues/3449
triton_config = {'grf_mode': 'large'} if is_intel_alchemist else {}
NUM_WARPS = [2, 4, 8] if is_nvidia_hopper else [2, 4, 8, 16]


@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BS", "BK", "BV", "USE_G"],
)
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
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
    USE_G: tl.constexpr
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
    if USE_G:
        g += bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += (bos * H + i_h * T) * T + i_k * B * H * T * T
    stride_qk = H * K
    stride_vo = H * V
    stride_g = H

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap.
    # masks required
    if USE_G:
        p_gq = tl.make_block_ptr(g, (T,), (stride_g,), (i_t * BT,), (BT,), (0,))
        # [BT,]
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
        # rescale interchunk output
    else:
        b_gq = None

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
        if USE_G:
            p_gk = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,))
            b_s *= safe_exp(b_gq[:, None] - b_gk[None, :])
            b_s = tl.where(m_s, b_s, 0)
        else:
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
        if USE_G:
            p_g = tl.make_block_ptr(g, (T,), (stride_g,), (i_s,), (BS,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * stride_g)
            b_gp = tl.load(g + (i_s-1) * stride_g) if i_s % BT > 0 else 0.
            # No concrete meaning. Just to avoid some layout bugs.
            b_s *= safe_exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += (b_gn - b_gp)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config(triton_config, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BS', 'BK', 'BV', 'USE_G'],
)

def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    output_attentions: bool = False,
    chunk_size: int = 128,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    if check_shared_mem('hopper', k.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    elif check_shared_mem('ampere', k.device.index):
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    else:
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # local cumulative decay in log space
    if g is not None:
        g = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)
    grid = (NK * NV, NT, B * H)
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None

    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
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
    return o, g, attn

def parallel_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
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
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
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
        q, k, v, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, g))
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
        g,
        scale,
        output_attentions,
        cu_seqlens
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, attn