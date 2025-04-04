import torch
from parallel import linear_attention as parallel_linear_attention
from recurrent import linear_attention as recurrent_linear_attention

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a single test input
batch, length, heads, dim = 2, 4, 3, 8
Q = torch.randn(batch, length, heads, dim)
K = torch.randn(batch, length, heads, dim)
V = torch.randn(batch, length, heads, dim)

# Run both implementations
parallel_output = parallel_linear_attention(Q, K, V)
recurrent_output = recurrent_linear_attention(Q, K, V)

# Check shapes
assert parallel_output.shape == recurrent_output.shape, "Shape mismatch"

# Check values with tolerance for floating point precision
assert torch.allclose(parallel_output, recurrent_output, rtol=1e-5, atol=1e-5), "Output mismatch"

