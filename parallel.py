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

if __name__ == "__main__":
    Q = torch.randn(2, 4, 3, 8)
    K = torch.randn(2, 4, 3, 8)
    V = torch.randn(2, 4, 3, 8)
    print(linear_attention(Q, K, V))
