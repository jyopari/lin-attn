import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum
from fla.ops.utils.op import exp
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=["BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    o,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    all = B * T

    p_q = q + (bos            ) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos            ) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos            ) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_k * all + bos) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V
