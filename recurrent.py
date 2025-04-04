import torch
from einops import einsum


def linear_attention(
    Q: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
    K: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
    V: torch.Tensor,  # shape: [batch, length, num_heads, num_units]
) -> torch.Tensor:    # shape: [batch, length, num_heads, num_units]
    # [batch, num_heads, num_units, num_units]
    Y = torch.zeros_like(V)
    S = torch.zeros([Q.shape[0], Q.shape[2], K.shape[3], V.shape[3]])
    for t in range(Q.shape[1]):
        S = S + einsum(K[:, t, ...], V[:, t, ...], "b h dk, b h dv -> b h dk dv")
        Y[:, t, ...] = einsum(Q[:, t, ...], S, "b h dk, b h dk dv -> b h dv")
    return Y
