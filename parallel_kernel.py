import torch
import triton
import triton.language as tl

def linear_attention_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("b i h d, b j h d -> b i h j", Q, K)
    L = scores.shape[-1]
    mask = torch.tril(torch.ones(L, L, device=scores.device, dtype=scores.dtype))
    mask = mask.unsqueeze(0).unsqueeze(2)
    masked_scores = scores * mask
    output = torch.einsum("b i h j, b j h d -> b i h d", masked_scores, V)
    return output

@triton.jit
def _linear_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_ob, stride_ol, stride_oh, stride_od,
    seq_len: tl.int32,
    num_heads: tl.int32,
    head_dim: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base_ptr = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base_ptr = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base_ptr = V + pid_batch * stride_vb + pid_head * stride_vh
    o_base_ptr = O + pid_batch * stride_ob + pid_head * stride_oh

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        q_ptrs = q_base_ptr + (offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd)
        q_mask = offs_m[:, None] < seq_len
        q_block = tl.load(q_ptrs, mask=q_mask & (offs_d[None, :] < head_dim), other=0.0)

        k_ptrs = k_base_ptr + (offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd)
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores_block = tl.dot(q_block, tl.trans(k_block))

        causal_mask = offs_m[:, None] >= offs_n[None, :]

        scores_mask = causal_mask & (offs_n[None, :] < seq_len)
        scores_block = tl.where(scores_mask, scores_block, 0.0)

        v_ptrs = v_base_ptr + (offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd)
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)

        delta_acc = tl.dot(scores_block.to(v_block.dtype), v_block)

        acc += delta_acc

    o_ptrs = o_base_ptr + (offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

def linear_attention_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape, "Q, K, V must have the same shape"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Input tensors must be on CUDA device"

    O = torch.empty_like(Q)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = triton.next_power_of_2(head_dim) if head_dim > 16 else 16
    if head_dim <= BLOCK_D:
       BLOCK_D = max(16, head_dim)

    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

    _linear_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        seq_len, num_heads, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return O

if __name__ == "__main__":
    BATCH, N_HEADS, SEQLEN, D_HEAD = 2, 3, 8, 64

    dtype = torch.float32

    print(f"Config: Batch={BATCH}, Heads={N_HEADS}, SeqLen={SEQLEN}, HeadDim={D_HEAD}")

    device = torch.device("cuda")
    Q = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device, dtype=dtype)
    K = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device, dtype=dtype)
    V = torch.randn(BATCH, SEQLEN, N_HEADS, D_HEAD, device=device, dtype=dtype)

    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()

    print("\nRunning PyTorch version...")
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    output_pytorch = linear_attention_pytorch(Q, K, V)
    end_time.record()
    torch.cuda.synchronize()
    pytorch_time = start_time.elapsed_time(end_time)
    print(f"PyTorch Time: {pytorch_time:.3f} ms")

    print("\nRunning Triton version...")
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    output_triton = linear_attention_triton(Q, K, V)
    end_time.record()
    torch.cuda.synchronize()
    triton_time = start_time.elapsed_time(end_time)
    print(f"Triton Time: {triton_time:.3f} ms")

    print("\nComparing outputs...")
    output_pytorch_fp32 = output_pytorch.float()
    output_triton_fp32 = output_triton.float()

    rtol =  1e-2
    atol =  1e-2

    print(output_triton_fp32[0, 0, 0, :])
    print(output_pytorch_fp32[0, 0, 0, :])

    ms_pytorch = triton.testing.do_bench(lambda: linear_attention_pytorch(Q, K, V))
    ms_triton = triton.testing.do_bench(lambda: linear_attention_triton(Q, K, V))
    print(f"  PyTorch: {ms_pytorch:.3f} ms")
    print(f"  Triton:  {ms_triton:.3f} ms")
    speedup = ms_pytorch / ms_triton
    print(f"  Speedup: {speedup:.2f}x")
