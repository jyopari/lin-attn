import torch
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
