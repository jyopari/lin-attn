import torch
import argparse
from parallel import linear_attention as parallel_linear_attention
from recurrent import linear_attention as recurrent_linear_attention
from recurrent_kernel import fused_recurrent
from parallel_kernel import linear_attention_triton
from typing import Callable


def profile_ncu(
    fn: Callable,
    n: int = 1,
    warmup: int = 3,
) -> None:
    # https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59

    for i in range(warmup + n):

        # start profiling after warmup iterations
        if i == warmup:
            torch.cuda.cudart().cudaProfilerStart()

        # push range for current iteration
        if i >= warmup:
            torch.cuda.nvtx.range_push(f"iteration-{i}")

        # push range for forward
        if i >= warmup:
            torch.cuda.nvtx.range_push("forward")
        output = fn()
        if i >= warmup:
            torch.cuda.nvtx.range_pop()

        # pop iteration range
        if i >= warmup:
            torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()


def main(name: str) -> None:

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a single test input
    batch, length, heads, dim = 4, 2048, 48, 128
    Q = torch.randn(batch, length, heads, dim, device="cuda") / 50.
    K = torch.randn(batch, length, heads, dim, device="cuda") / 50.
    V = torch.randn(batch, length, heads, dim, device="cuda") / 50.
    implementations = {
        "parallel_pytorch": parallel_linear_attention,
        "recurrent_pytorch": recurrent_linear_attention,
        "recurrent_triton": fused_recurrent,
        "parallel_triton": linear_attention_triton,
    }
    fn = implementations[name]
    print(f"Profiling {name} implementation...")
    profile_ncu(lambda: fn(Q, K, V))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=[
            "parallel_pytorch",
            "recurrent_pytorch",
            "recurrent_triton",
            "parallel_triton",
        ],
    )
    args = parser.parse_args()
    main(args.name)
