import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=["BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_kernel(
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


def fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    B, T, H, K = k.shape
    V = v.shape[-1]
    BK = min(K, 64)
    BV = min(V, 64)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    o = q.new_empty(NK, *v.shape, dtype=torch.float32)

    grid = (NV, NK, B * H)
    fused_recurrent_kernel[grid](
        q,
        k,
        v,
        o,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    o = o.sum(0)
    return o


def fused_recurrent_dataflow(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    count: bool,
    BK: int = 64,
    BV: int = 64,
) -> torch.Tensor:
    B, T, H, K = k.shape
    V = v.shape[-1]

    NK = int(K / BK)
    NV = int(V / BV)
    assert BK * NK == K
    assert BV * NV == V
    o_tmp = torch.empty([NK, B, T, H, V], dtype=v.dtype, device=v.device)

    statistics = {
        "DRAM->SRAM": 0,
        "SRAM->DRAM": 0,
        "MAC": 0,
    }
    for i_v in range(0, NV):  # parallel for
        for i_k in range(0, NK):  # parallel for
            for i_n in range(0, B):  # parallel for
                for i_h in range(0, H):  # parallel for

                    # on-chip buffer
                    b_q = torch.empty([BK], dtype=q.dtype, device=q.device)
                    b_k = torch.empty([BK], dtype=k.dtype, device=k.device)
                    b_v = torch.empty([BV], dtype=v.dtype, device=v.device)
                    b_h = torch.zeros([BV, BK], dtype=v.dtype, device=v.device)

                    for i_t in range(0, T):

                        # DRAM -> SRAM copy
                        for j_k in range(0, BK):
                            if count:
                                statistics["DRAM->SRAM"] += q.element_size()
                            else:
                                b_q[j_k] = q[i_n, i_t, i_h, i_k * BK + j_k]

                        for j_k in range(0, BK):
                            if count:
                                statistics["DRAM->SRAM"] += k.element_size()
                            else:
                                b_k[j_k] = k[i_n, i_t, i_h, i_k * BK + j_k]

                        for j_v in range(0, BV):
                            if count:
                                statistics["DRAM->SRAM"] += v.element_size()
                            else:
                                b_v[j_v] = v[i_n, i_t, i_h, i_v * BV + j_v]

                        # on-chip compute
                        for j_v in range(0, BV):
                            for j_k in range(0, BK):
                                if count:
                                    statistics["MAC"] += 1
                                else:
                                    b_h[j_v, j_k] += b_k[j_k] * b_v[j_v]

                        b_o = torch.zeros([BV], dtype=v.dtype, device=v.device)
                        for j_v in range(0, BV):
                            for j_k in range(0, BK):
                                if count:
                                    statistics["MAC"] += 1
                                else:
                                    b_o[j_v] += b_h[j_v, j_k] * b_q[j_k]

                        # SRAM -> DRAM store
                        for j_v in range(0, BV):
                            if count:
                                statistics["SRAM->DRAM"] += 1
                            else:
                                o_tmp[i_k, i_n, i_t, i_h, i_v * BV + j_v] = b_o[j_v]

    # ignore the statistics of separate reductions
    if not count:
        # reduce
        o = torch.zeros([B, T, H, V], dtype=v.dtype, device=v.device)
        for i_k in range(0, NK):
            for i_n in range(0, B):
                for i_t in range(0, T):
                    for i_h in range(0, H):
                        for i_v in range(0, V):
                            o[i_n, i_t, i_h, i_v] += o_tmp[i_k, i_n, i_t, i_h, i_v]

        # check against fused_recurrent
        o_ = fused_recurrent(q, k, v)

    statistics["SRAM-size"] = (
        b_q.numel() * b_q.element_size() +
        b_k.numel() * b_k.element_size() +
        b_v.numel() * b_v.element_size() +
        b_h.numel() * b_h.element_size()
    )
    statistics["arithmetic-intensity"] = (
        (statistics["MAC"] * 2) /
        (statistics["DRAM->SRAM"] + statistics["SRAM->DRAM"])
    )

    if count:
        return statistics
    else:
        return o, o_
