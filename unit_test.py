import torch
from parallel import linear_attention as parallel_linear_attention
from recurrent import linear_attention as recurrent_linear_attention
from recurrent_kernel import fused_recurrent

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a single test input
batch, length, heads, dim = 4, 2048, 48, 128
Q = torch.randn(batch, length, heads, dim, device="cuda") / 50.
K = torch.randn(batch, length, heads, dim, device="cuda") / 50.
V = torch.randn(batch, length, heads, dim, device="cuda") / 50.

# Run both implementations
parallel_output = parallel_linear_attention(Q, K, V)
recurrent_output = recurrent_linear_attention(Q, K, V)
recurrent_output2 = fused_recurrent(Q, K, V)

# Check shapes
assert parallel_output.shape == recurrent_output.shape, "Shape mismatch"
assert recurrent_output.shape == recurrent_output2.shape, "Shape mismatch"

# Check values with tolerance for floating point precision
assert torch.allclose(parallel_output, recurrent_output, rtol=1e-5, atol=1e-5), "Output mismatch"
assert torch.allclose(recurrent_output, recurrent_output2, rtol=1e-5, atol=1e-5), "Output mismatch"
