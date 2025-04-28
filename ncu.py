import torch
import argparse
import pandas as pd
from parallel import linear_attention as parallel_linear_attention
from recurrent import linear_attention as recurrent_linear_attention
from recurrent_kernel import fused_recurrent, fused_recurrent_dataflow
from parallel_kernel import parallel_simple_gla
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
        "parallel_triton": parallel_simple_gla,
    }
    fn = implementations[name]
    print(f"Profiling {name} implementation...")
    profile_ncu(lambda: fn(Q, K, V))


def statistics(name: str) -> pd.DataFrame:

    # Set random seed for reproducibility
    torch.manual_seed(42)

    implementations = {
        "recurrent": fused_recurrent_dataflow,
        # "parallel": parallel_simple_gla,
    }
    fn = implementations[name]
    results = []

    for batch in [1, 2, 4]:
        for length in [64]:
            for heads in [16]:
                for dim in [128]:
                    Q = torch.randn(batch, length, heads, dim, device="cuda") / 50.
                    K = torch.randn(batch, length, heads, dim, device="cuda") / 50.
                    V = torch.randn(batch, length, heads, dim, device="cuda") / 50.

                    if name == "recurrent":
                        for BK in [16, 64, 128]:
                            for BV in [16, 64, 128]:
                                stats = fn(Q, K, V, count=True, BK=BK, BV=BV)
                                results.append({
                                    "batch": batch,
                                    "length": length,
                                    "heads": heads,
                                    "dim": dim,
                                    "BK": BK,
                                    "BV": BV,
                                    **stats,
                                })

                    elif name == "parallel":
                        for BT in [16, 32]:
                            BS = BT
                            for BK in [16, 64, 128]:
                                for BV in [16, 64, 128]:
                                    stats = fn(Q, K, V, count=True, BT=BT, BS=BS, BK=BK, BV=BV)
                                    results.append({
                                        "batch": batch,
                                        "length": length,
                                        "heads": heads,
                                        "dim": dim,
                                        "BT": BT,
                                        "BS": BS,
                                        "BK": BK,
                                        "BV": BV,
                                        **stats,
                                    })

                    else:
                        raise ValueError(f"Unknown implementation: {name}")

    df = pd.DataFrame(results)
    return df


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
